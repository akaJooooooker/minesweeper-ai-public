"""Fine-tune Replay risk with a smooth expected-regret deployment objective."""

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

from .replay_blend_anchor import _reservoir_anchor_examples, _within_brier_guard
from .replay_blend_ranking import (
    BlendRankingMetrics,
    _blend_metrics,
    _collate_blend,
    _load_blend_examples,
)
from .replay_learning import _collate, _torch_model


@dataclass(frozen=True)
class RegretEpoch:
    epoch: int
    train_loss: float
    expected_regret_loss: float
    blend_brier_loss: float
    anchor_loss: float
    validation_regret: float
    validation_optimal_rate: float
    validation_blend_brier: float
    within_brier_guard: bool
    best: bool


def fine_tune_expected_regret(
    checkpoint_path: str | Path,
    dataset_paths: Sequence[str | Path],
    validation_path: str | Path,
    anchor_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    alpha: float = 0.75,
    solver_cells: int = 24,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 3e-5,
    temperature: float = 0.05,
    regret_weight: float = 20.0,
    brier_weight: float = 5.0,
    anchor_weight: float = 10.0,
    anchor_examples: int = 4096,
    max_brier_regression: float = 0.02,
    seed: int = 0,
    device_name: str = "auto",
    use_amp: bool = True,
    gpu_memory_fraction: float | None = None,
    batch_delay_ms: int = 0,
    patience: int = 20,
    on_epoch=None,
) -> tuple[BlendRankingMetrics, list[RegretEpoch]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if min(regret_weight, brier_weight, anchor_weight) < 0:
        raise ValueError("loss weights must be non-negative")
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
    width = int(checkpoint["model_width"])
    model = ReplayAgentNet(width).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    parent = ReplayAgentNet(width).to(device)
    parent.load_state_dict(checkpoint["state_dict"])
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.risk_head.parameters():
        parameter.requires_grad_(True)

    train_examples = _load_blend_examples(dataset_paths, solver_cells)
    validation_examples = _load_blend_examples([validation_path], solver_cells)
    anchors = _reservoir_anchor_examples(anchor_paths, anchor_examples, seed + 1)
    if not train_examples or not validation_examples or not anchors:
        raise ValueError("deployment, validation, and anchor datasets must be non-empty")

    optimizer = torch.optim.AdamW(model.risk_head.parameters(), lr=learning_rate, weight_decay=1e-4)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    initial = _blend_metrics(model, validation_examples, alpha, batch_size, torch, device)
    best_metrics = initial
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[RegretEpoch] = []

    for epoch in range(1, epochs + 1):
        deployment_batches = [
            train_examples[start : start + batch_size]
            for start in range(0, len(train_examples), batch_size)
        ]
        random.shuffle(deployment_batches)
        random.shuffle(anchors)
        sums = {"total": 0.0, "regret": 0.0, "brier": 0.0, "anchor": 0.0}
        state_count = 0
        model.train()
        for batch_index, batch in enumerate(deployment_batches):
            tensors = {
                key: value.to(device) for key, value in _collate_blend(batch, torch).items()
            }
            anchor_start = (batch_index * batch_size) % len(anchors)
            anchor_batch = anchors[anchor_start : anchor_start + batch_size]
            if len(anchor_batch) < batch_size:
                anchor_batch += anchors[: batch_size - len(anchor_batch)]
            anchor_tensors = {
                key: value.to(device) for key, value in _collate(anchor_batch, torch).items()
            }
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
                scores = (-flat_blended / temperature).masked_fill(~flat_mask, -1e4)
                choice_distribution = scores.softmax(dim=1)
                minimum = flat_targets.masked_fill(~flat_mask, float("inf")).min(dim=1).values
                expected = (choice_distribution * flat_targets).sum(dim=1)
                expected_regret = (expected - minimum).mean()
                mask = tensors["risk_mask"]
                blend_brier = ((blended - tensors["risk_targets"]).square() * mask).sum()
                blend_brier = blend_brier / mask.sum().clamp_min(1)

                anchor_logits = model(anchor_tensors["inputs"], anchor_tensors["valid"])["risk"]
                with torch.no_grad():
                    parent_logits = parent(
                        anchor_tensors["inputs"], anchor_tensors["valid"]
                    )["risk"]
                anchor_mask = anchor_tensors["risk_mask"]
                anchor_loss = (
                    (anchor_logits - parent_logits).square() * anchor_mask
                ).sum() / anchor_mask.sum().clamp_min(1)
                loss = (
                    regret_weight * expected_regret
                    + brier_weight * blend_brier
                    + anchor_weight * anchor_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.risk_head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch)
            sums["total"] += float(loss.detach()) * count
            sums["regret"] += float(expected_regret.detach()) * count
            sums["brier"] += float(blend_brier.detach()) * count
            sums["anchor"] += float(anchor_loss.detach()) * count
            state_count += count
            if batch_delay_ms:
                time.sleep(batch_delay_ms / 1_000)

        validation = _blend_metrics(
            model, validation_examples, alpha, batch_size, torch, device
        )
        within_guard = _within_brier_guard(validation, initial, max_brier_regression)
        improved = within_guard and _is_better(validation, best_metrics)
        if improved:
            best_metrics = validation
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        item = RegretEpoch(
            epoch=epoch,
            train_loss=sums["total"] / max(state_count, 1),
            expected_regret_loss=sums["regret"] / max(state_count, 1),
            blend_brier_loss=sums["brier"] / max(state_count, 1),
            anchor_loss=sums["anchor"] / max(state_count, 1),
            validation_regret=validation.mean_regret,
            validation_optimal_rate=validation.optimal_rate,
            validation_blend_brier=validation.blend_brier,
            within_brier_guard=within_guard,
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
            "kind": "replay-agent-expected-regret-v18",
            "parent_checkpoint": str(Path(checkpoint_path)),
            "expected_regret": {
                "datasets": [str(Path(path)) for path in dataset_paths],
                "validation": str(Path(validation_path)),
                "anchors": [str(Path(path)) for path in anchor_paths],
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "anchor_examples": len(anchors),
                "initial_validation": asdict(initial),
                "best_validation": asdict(best_metrics),
                "best_epoch": best_epoch,
                "epochs_run": len(history),
                "alpha": alpha,
                "solver_cells": solver_cells,
                "temperature": temperature,
                "regret_weight": regret_weight,
                "brier_weight": brier_weight,
                "anchor_weight": anchor_weight,
                "max_brier_regression": max_brier_regression,
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


def _is_better(candidate: BlendRankingMetrics, current: BlendRankingMetrics) -> bool:
    if candidate.mean_regret < current.mean_regret - 1e-8:
        return True
    return (
        abs(candidate.mean_regret - current.mean_regret) <= 1e-8
        and candidate.optimal_rate > current.optimal_rate + 1e-8
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--solver-cells", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--regret-weight", type=float, default=20.0)
    parser.add_argument("--brier-weight", type=float, default=5.0)
    parser.add_argument("--anchor-weight", type=float, default=10.0)
    parser.add_argument("--anchor-examples", type=int, default=4096)
    parser.add_argument("--max-brier-regression", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260942)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--batch-delay-ms", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    initial, history = fine_tune_expected_regret(
        args.checkpoint,
        args.dataset,
        args.validation,
        args.anchor,
        args.output,
        alpha=args.alpha,
        solver_cells=args.solver_cells,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        regret_weight=args.regret_weight,
        brier_weight=args.brier_weight,
        anchor_weight=args.anchor_weight,
        anchor_examples=args.anchor_examples,
        max_brier_regression=args.max_brier_regression,
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
