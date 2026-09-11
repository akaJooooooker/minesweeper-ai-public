"""Decision-aligned Risk-head fine-tuning for non-exact Replay guess states."""

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

from .replay_risk_finetune import _load_risk_examples
from .replay_learning import _collate, _torch_model


@dataclass(frozen=True)
class RankingMetrics:
    states: int
    mean_regret: float
    optimal_rate: float
    brier: float


@dataclass(frozen=True)
class RankingEpoch:
    epoch: int
    train_loss: float
    validation_regret: float
    validation_optimal_rate: float
    validation_brier: float
    best: bool


def evaluate_ranking_checkpoint(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    *,
    batch_size: int = 8,
    device_name: str = "cpu",
) -> RankingMetrics:
    torch, ReplayAgentNet = _torch_model()
    checkpoint = torch.load(Path(checkpoint_path), map_location=device_name, weights_only=True)
    model = ReplayAgentNet(int(checkpoint["model_width"])).to(device_name)
    model.load_state_dict(checkpoint["state_dict"])
    examples = _load_risk_examples([dataset_path])
    return _ranking_metrics(model, examples, batch_size, torch, torch.device(device_name))


def fine_tune_risk_ranking(
    checkpoint_path: str | Path,
    dataset_paths: Sequence[str | Path],
    validation_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 200,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    brier_weight: float = 0.1,
    seed: int = 0,
    device_name: str = "auto",
    use_amp: bool = True,
    gpu_memory_fraction: float | None = None,
    batch_delay_ms: int = 0,
    patience: int = 20,
    on_epoch=None,
) -> tuple[RankingMetrics, list[RankingEpoch]]:
    torch, ReplayAgentNet = _torch_model()
    from torch.nn import functional

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
    train_examples = _load_risk_examples(dataset_paths)
    validation_examples = _load_risk_examples([validation_path])
    if not train_examples or not validation_examples:
        raise ValueError("ranking train and validation datasets must be non-empty")

    optimizer = torch.optim.AdamW(model.risk_head.parameters(), lr=learning_rate, weight_decay=1e-4)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    initial = _ranking_metrics(model, validation_examples, batch_size, torch, device)
    best_metrics = initial
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[RankingEpoch] = []

    for epoch in range(1, epochs + 1):
        batches = [train_examples[i : i + batch_size] for i in range(0, len(train_examples), batch_size)]
        random.shuffle(batches)
        loss_sum = 0.0
        state_count = 0
        model.train()
        for batch in batches:
            tensors = {key: value.to(device) for key, value in _collate(batch, torch).items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(tensors["inputs"], tensors["valid"])["risk"]
                mask = tensors["risk_mask"].bool()
                flat_logits = logits.flatten(1)
                flat_targets = tensors["risk_targets"].flatten(1)
                flat_mask = mask.flatten(1)
                minimum = flat_targets.masked_fill(~flat_mask, float("inf")).min(dim=1).values
                optimal = flat_mask & (flat_targets <= minimum[:, None] + 1e-7)
                target_distribution = optimal.float() / optimal.sum(dim=1, keepdim=True).clamp_min(1)
                safe_scores = (-flat_logits).masked_fill(~flat_mask, -1e4)
                ranking_loss = -(target_distribution * safe_scores.log_softmax(dim=1)).sum(dim=1).mean()
                bce_elements = functional.binary_cross_entropy_with_logits(
                    logits, tensors["risk_targets"], reduction="none"
                )
                bce_loss = (bce_elements * tensors["risk_mask"]).sum() / tensors[
                    "risk_mask"
                ].sum().clamp_min(1)
                loss = ranking_loss + brier_weight * bce_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.risk_head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(batch)
            state_count += len(batch)
            if batch_delay_ms:
                time.sleep(batch_delay_ms / 1_000)

        validation = _ranking_metrics(model, validation_examples, batch_size, torch, device)
        improved = _is_better(validation, best_metrics)
        if improved:
            best_metrics = validation
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        result = RankingEpoch(
            epoch=epoch,
            train_loss=loss_sum / max(state_count, 1),
            validation_regret=validation.mean_regret,
            validation_optimal_rate=validation.optimal_rate,
            validation_brier=validation.brier,
            best=improved,
        )
        history.append(result)
        if on_epoch:
            on_epoch(result)
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
            "kind": "replay-agent-risk-ranking-v5",
            "parent_checkpoint": str(Path(checkpoint_path)),
            "risk_ranking": {
                "datasets": [str(Path(path)) for path in dataset_paths],
                "validation": str(Path(validation_path)),
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "initial_validation": asdict(initial),
                "best_validation": asdict(best_metrics),
                "best_epoch": best_epoch,
                "epochs_run": len(history),
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "brier_weight": brier_weight,
                "seed": seed,
            },
        }
    )
    torch.save({"state_dict": model.state_dict(), **metadata}, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps({**metadata, "history": [asdict(item) for item in history]}, indent=2),
        encoding="utf-8",
    )
    return initial, history


def _is_better(candidate: RankingMetrics, current: RankingMetrics) -> bool:
    if candidate.mean_regret < current.mean_regret - 1e-8:
        return True
    return (
        abs(candidate.mean_regret - current.mean_regret) <= 1e-8
        and candidate.brier < current.brier - 1e-8
    )


def _ranking_metrics(model, examples, batch_size, torch, device) -> RankingMetrics:
    regret_sum = brier_sum = cell_count = 0.0
    optimal_count = state_count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            tensors = {
                key: value.to(device)
                for key, value in _collate(examples[start : start + batch_size], torch).items()
            }
            logits = model(tensors["inputs"], tensors["valid"])["risk"]
            probabilities = torch.sigmoid(logits)
            flat_logits = logits.flatten(1)
            flat_targets = tensors["risk_targets"].flatten(1)
            flat_mask = tensors["risk_mask"].bool().flatten(1)
            selected = flat_logits.masked_fill(~flat_mask, float("inf")).argmin(dim=1)
            chosen_target = flat_targets.gather(1, selected[:, None]).squeeze(1)
            minimum = flat_targets.masked_fill(~flat_mask, float("inf")).min(dim=1).values
            regret_sum += float((chosen_target - minimum).sum())
            optimal_count += int((chosen_target <= minimum + 1e-7).sum())
            state_count += len(examples[start : start + batch_size])
            squared = (probabilities - tensors["risk_targets"]).square() * tensors["risk_mask"]
            brier_sum += float(squared.sum())
            cell_count += float(tensors["risk_mask"].sum())
    return RankingMetrics(
        states=state_count,
        mean_regret=regret_sum / max(state_count, 1),
        optimal_rate=optimal_count / max(state_count, 1),
        brier=brier_sum / max(cell_count, 1.0),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train = subparsers.add_parser("train")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--dataset", type=Path, nargs="+", required=True)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--brier-weight", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=20260911)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--gpu-memory-fraction", type=float)
    train.add_argument("--batch-delay-ms", type=int, default=0)
    train.add_argument("--patience", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        metrics = evaluate_ranking_checkpoint(
            args.checkpoint, args.dataset, batch_size=args.batch_size, device_name=args.device
        )
        print(json.dumps(asdict(metrics), separators=(",", ":")))
        return
    initial, history = fine_tune_risk_ranking(
        args.checkpoint,
        args.dataset,
        args.validation,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        brier_weight=args.brier_weight,
        seed=args.seed,
        device_name=args.device,
        use_amp=not args.no_amp,
        gpu_memory_fraction=args.gpu_memory_fraction,
        batch_delay_ms=args.batch_delay_ms,
        patience=args.patience,
        on_epoch=lambda item: print(json.dumps({"event": "epoch", **asdict(item)})),
    )
    print(json.dumps({"output": str(args.output.resolve()), "initial": asdict(initial), "epochs": len(history)}))


if __name__ == "__main__":
    main()
