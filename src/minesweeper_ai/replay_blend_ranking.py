"""Fine-tune Replay risk using the exact deployed neural/solver blend."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np

from .replay import read_replay_examples
from .replay_learning import _collate, _prepare_example, _torch_model
from .solver import ConstraintSolver


@dataclass(frozen=True)
class BlendRankingMetrics:
    states: int
    mean_regret: float
    optimal_rate: float
    blend_brier: float
    neural_brier: float


@dataclass(frozen=True)
class BlendRankingEpoch:
    epoch: int
    train_loss: float
    validation_regret: float
    validation_optimal_rate: float
    validation_blend_brier: float
    validation_neural_brier: float
    best: bool


def evaluate_blend_checkpoint(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    *,
    alpha: float = 0.75,
    solver_cells: int = 24,
    batch_size: int = 16,
    device_name: str = "cpu",
) -> BlendRankingMetrics:
    _validate_alpha(alpha)
    torch, ReplayAgentNet = _torch_model()
    checkpoint = torch.load(Path(checkpoint_path), map_location=device_name, weights_only=True)
    model = ReplayAgentNet(int(checkpoint["model_width"])).to(device_name)
    model.load_state_dict(checkpoint["state_dict"])
    examples = _load_blend_examples([dataset_path], solver_cells)
    if not examples:
        raise ValueError("blend evaluation dataset has no deployment gap states")
    return _blend_metrics(
        model, examples, alpha, batch_size, torch, torch.device(device_name)
    )


def fine_tune_blend_ranking(
    checkpoint_path: str | Path,
    dataset_paths: Sequence[str | Path],
    validation_path: str | Path,
    output_path: str | Path,
    *,
    alpha: float = 0.75,
    solver_cells: int = 24,
    epochs: int = 120,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    temperature: float = 0.05,
    brier_weight: float = 0.25,
    seed: int = 0,
    device_name: str = "auto",
    use_amp: bool = True,
    gpu_memory_fraction: float | None = None,
    batch_delay_ms: int = 0,
    patience: int = 15,
    on_epoch=None,
) -> tuple[BlendRankingMetrics, list[BlendRankingEpoch]]:
    _validate_alpha(alpha)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if brier_weight < 0:
        raise ValueError("brier_weight must be non-negative")
    torch, ReplayAgentNet = _torch_model()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_name must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    selected = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    device = torch.device("cpu" if selected == "auto" else selected)
    if gpu_memory_fraction is not None:
        if device.type != "cuda" or not 0 < gpu_memory_fraction <= 1:
            raise ValueError("GPU memory fraction requires CUDA and must be in (0, 1]")
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=index)

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    model = ReplayAgentNet(int(checkpoint["model_width"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.risk_head.parameters():
        parameter.requires_grad_(True)

    train_examples = _load_blend_examples(dataset_paths, solver_cells)
    validation_examples = _load_blend_examples([validation_path], solver_cells)
    if not train_examples or not validation_examples:
        raise ValueError("blend train and validation datasets must contain deployment gaps")

    optimizer = torch.optim.AdamW(model.risk_head.parameters(), lr=learning_rate, weight_decay=1e-4)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    initial = _blend_metrics(model, validation_examples, alpha, batch_size, torch, device)
    best_metrics = initial
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[BlendRankingEpoch] = []

    for epoch in range(1, epochs + 1):
        batches = [
            train_examples[start : start + batch_size]
            for start in range(0, len(train_examples), batch_size)
        ]
        random.shuffle(batches)
        loss_sum = 0.0
        state_count = 0
        model.train()
        for batch in batches:
            tensors = _collate_blend(batch, torch)
            tensors = {key: value.to(device) for key, value in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(tensors["inputs"], tensors["valid"])["risk"]
                neural = torch.sigmoid(logits)
                blended = alpha * neural + (1.0 - alpha) * tensors["solver_risk"]
                flat_blended = blended.flatten(1)
                flat_targets = tensors["risk_targets"].flatten(1)
                flat_mask = tensors["risk_mask"].bool().flatten(1)
                minimum = flat_targets.masked_fill(~flat_mask, float("inf")).min(dim=1).values
                optimal = flat_mask & (flat_targets <= minimum[:, None] + 1e-7)
                target_distribution = optimal.float() / optimal.sum(dim=1, keepdim=True).clamp_min(1)
                safe_scores = (-flat_blended / temperature).masked_fill(~flat_mask, -1e4)
                ranking_loss = -(
                    target_distribution * safe_scores.log_softmax(dim=1)
                ).sum(dim=1).mean()
                squared = (blended - tensors["risk_targets"]).square() * tensors["risk_mask"]
                brier_loss = squared.sum() / tensors["risk_mask"].sum().clamp_min(1)
                loss = ranking_loss + brier_weight * brier_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.risk_head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(batch)
            state_count += len(batch)
            if batch_delay_ms:
                time.sleep(batch_delay_ms / 1_000)

        validation = _blend_metrics(
            model, validation_examples, alpha, batch_size, torch, device
        )
        improved = _is_better(validation, best_metrics)
        if improved:
            best_metrics = validation
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        item = BlendRankingEpoch(
            epoch=epoch,
            train_loss=loss_sum / max(state_count, 1),
            validation_regret=validation.mean_regret,
            validation_optimal_rate=validation.optimal_rate,
            validation_blend_brier=validation.blend_brier,
            validation_neural_brier=validation.neural_brier,
            best=improved,
        )
        history.append(item)
        if on_epoch:
            on_epoch(item)
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        key: value for key, value in checkpoint.items() if key not in {"state_dict", "history", "training"}
    }
    metadata.update(
        {
            "kind": "replay-agent-blend-ranking-v10",
            "parent_checkpoint": str(Path(checkpoint_path)),
            "blend_ranking": {
                "datasets": [str(Path(path)) for path in dataset_paths],
                "validation": str(Path(validation_path)),
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "initial_validation": asdict(initial),
                "best_validation": asdict(best_metrics),
                "best_epoch": best_epoch,
                "epochs_run": len(history),
                "alpha": alpha,
                "solver_cells": solver_cells,
                "temperature": temperature,
                "brier_weight": brier_weight,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": str(device),
                "amp": amp_enabled,
                "gpu_memory_fraction": gpu_memory_fraction,
            },
        }
    )
    torch.save({"state_dict": model.state_dict(), **metadata}, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps({**metadata, "history": [asdict(item) for item in history]}, indent=2),
        encoding="utf-8",
    )
    return initial, history


def _validate_alpha(alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")


def _load_blend_examples(paths: Sequence[str | Path], solver_cells: int):
    solver = ConstraintSolver(solver_cells)
    result = []
    for path in paths:
        for example in read_replay_examples(path):
            prepared = _prepare_example(example)
            deployment_mask = prepared[3]
            if not deployment_mask.any():
                continue
            analysis = solver.analyse(example.observation, example.total_mines)
            if analysis.exact or analysis.safe or analysis.contradiction:
                continue
            height = len(example.observation)
            width = len(example.observation[0])
            solver_risk = np.zeros((height, width), dtype=np.float32)
            for row in range(height):
                for col in range(width):
                    if deployment_mask[row, col]:
                        probability = analysis.probabilities.get((row, col))
                        if probability is None:
                            deployment_mask[row, col] = 0.0
                        else:
                            solver_risk[row, col] = probability
            if not deployment_mask.any():
                continue
            result.append((prepared, solver_risk))
    return result


def _collate_blend(examples, torch):
    tensors = _collate([item[0] for item in examples], torch)
    height, width = tensors["valid"].shape[-2:]
    solver_risk = np.zeros((len(examples), height, width), dtype=np.float32)
    for index, (_, board_risk) in enumerate(examples):
        board_height, board_width = board_risk.shape
        solver_risk[index, :board_height, :board_width] = board_risk
    tensors["solver_risk"] = torch.from_numpy(solver_risk)
    return tensors


def _blend_metrics(model, examples, alpha, batch_size, torch, device) -> BlendRankingMetrics:
    regret_sum = blend_squared_sum = neural_squared_sum = cell_count = 0.0
    optimal_count = state_count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            tensors = {
                key: value.to(device) for key, value in _collate_blend(batch, torch).items()
            }
            neural = torch.sigmoid(model(tensors["inputs"], tensors["valid"])["risk"])
            blended = alpha * neural + (1.0 - alpha) * tensors["solver_risk"]
            flat_blended = blended.flatten(1)
            flat_targets = tensors["risk_targets"].flatten(1)
            flat_mask = tensors["risk_mask"].bool().flatten(1)
            selected = flat_blended.masked_fill(~flat_mask, float("inf")).argmin(dim=1)
            chosen_target = flat_targets.gather(1, selected[:, None]).squeeze(1)
            minimum = flat_targets.masked_fill(~flat_mask, float("inf")).min(dim=1).values
            regret_sum += float((chosen_target - minimum).sum())
            optimal_count += int((chosen_target <= minimum + 1e-7).sum())
            state_count += len(batch)
            mask = tensors["risk_mask"]
            targets = tensors["risk_targets"]
            blend_squared_sum += float(((blended - targets).square() * mask).sum())
            neural_squared_sum += float(((neural - targets).square() * mask).sum())
            cell_count += float(mask.sum())
    return BlendRankingMetrics(
        states=state_count,
        mean_regret=regret_sum / max(state_count, 1),
        optimal_rate=optimal_count / max(state_count, 1),
        blend_brier=blend_squared_sum / max(cell_count, 1.0),
        neural_brier=neural_squared_sum / max(cell_count, 1.0),
    )


def _is_better(candidate: BlendRankingMetrics, current: BlendRankingMetrics) -> bool:
    if candidate.mean_regret < current.mean_regret - 1e-8:
        return True
    if abs(candidate.mean_regret - current.mean_regret) > 1e-8:
        return False
    if candidate.optimal_rate > current.optimal_rate + 1e-8:
        return True
    return (
        abs(candidate.optimal_rate - current.optimal_rate) <= 1e-8
        and candidate.blend_brier < current.blend_brier - 1e-8
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--alpha", type=float, default=0.75)
    evaluate.add_argument("--solver-cells", type=int, default=24)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train = subparsers.add_parser("train")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--dataset", type=Path, nargs="+", required=True)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--alpha", type=float, default=0.75)
    train.add_argument("--solver-cells", type=int, default=24)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--temperature", type=float, default=0.05)
    train.add_argument("--brier-weight", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=20260923)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--gpu-memory-fraction", type=float)
    train.add_argument("--batch-delay-ms", type=int, default=0)
    train.add_argument("--patience", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        metrics = evaluate_blend_checkpoint(
            args.checkpoint,
            args.dataset,
            alpha=args.alpha,
            solver_cells=args.solver_cells,
            batch_size=args.batch_size,
            device_name=args.device,
        )
        print(json.dumps(asdict(metrics), separators=(",", ":")))
        return
    initial, history = fine_tune_blend_ranking(
        args.checkpoint,
        args.dataset,
        args.validation,
        args.output,
        alpha=args.alpha,
        solver_cells=args.solver_cells,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        brier_weight=args.brier_weight,
        seed=args.seed,
        device_name=args.device,
        use_amp=not args.no_amp,
        gpu_memory_fraction=args.gpu_memory_fraction,
        batch_delay_ms=args.batch_delay_ms,
        patience=args.patience,
        on_epoch=lambda item: print(json.dumps({"event": "epoch", **asdict(item)}), flush=True),
    )
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "initial": asdict(initial), "epochs": len(history)}
        )
    )


if __name__ == "__main__":
    main()
