"""Blend-aware Replay fine-tuning with broad-distribution logit anchoring."""

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
from .replay_blend_ranking import (
    BlendRankingMetrics,
    _blend_metrics,
    _collate_blend,
    _load_blend_examples,
)
from .replay_learning import _collate, _prepare_example, _torch_model


@dataclass(frozen=True)
class AnchoredEpoch:
    epoch: int
    train_loss: float
    deployment_loss: float
    anchor_loss: float
    validation_regret: float
    validation_optimal_rate: float
    validation_blend_brier: float
    validation_neural_brier: float
    within_brier_guard: bool
    best: bool


def fine_tune_blend_anchored(
    checkpoint_path: str | Path,
    dataset_paths: Sequence[str | Path],
    validation_path: str | Path,
    anchor_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    alpha: float = 0.75,
    solver_cells: int = 24,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 5e-5,
    temperature: float = 0.05,
    brier_weight: float = 0.25,
    anchor_weight: float = 1.0,
    anchor_examples: int = 2048,
    max_brier_regression: float = 0.01,
    seed: int = 0,
    device_name: str = "auto",
    use_amp: bool = True,
    gpu_memory_fraction: float | None = None,
    batch_delay_ms: int = 0,
    patience: int = 20,
    train_all: bool = False,
    on_epoch=None,
) -> tuple[BlendRankingMetrics, list[AnchoredEpoch]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if temperature <= 0 or brier_weight < 0 or anchor_weight < 0:
        raise ValueError("temperature must be positive and loss weights non-negative")
    if anchor_examples < 1 or max_brier_regression < 0:
        raise ValueError("anchor_examples must be positive and Brier guard non-negative")
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
    parent = ReplayAgentNet(int(checkpoint["model_width"])).to(device)
    parent.load_state_dict(checkpoint["state_dict"])
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    if not train_all:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.risk_head.parameters():
            parameter.requires_grad_(True)

    train_examples = _load_blend_examples(dataset_paths, solver_cells)
    validation_examples = _load_blend_examples([validation_path], solver_cells)
    anchors = _reservoir_anchor_examples(anchor_paths, anchor_examples, seed + 1)
    if not train_examples or not validation_examples or not anchors:
        raise ValueError("deployment, validation, and anchor datasets must be non-empty")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    initial = _blend_metrics(model, validation_examples, alpha, batch_size, torch, device)
    best_metrics = initial
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[AnchoredEpoch] = []

    for epoch in range(1, epochs + 1):
        deployment_batches = [
            train_examples[start : start + batch_size]
            for start in range(0, len(train_examples), batch_size)
        ]
        random.shuffle(deployment_batches)
        random.shuffle(anchors)
        deployment_sum = anchor_sum = total_sum = 0.0
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
                minimum = flat_targets.masked_fill(~flat_mask, float("inf")).min(dim=1).values
                optimal = flat_mask & (flat_targets <= minimum[:, None] + 1e-7)
                target_distribution = optimal.float() / optimal.sum(dim=1, keepdim=True).clamp_min(1)
                safe_scores = (-flat_blended / temperature).masked_fill(~flat_mask, -1e4)
                ranking_loss = -(
                    target_distribution * safe_scores.log_softmax(dim=1)
                ).sum(dim=1).mean()
                squared = (blended - tensors["risk_targets"]).square() * tensors["risk_mask"]
                blend_brier = squared.sum() / tensors["risk_mask"].sum().clamp_min(1)
                deployment_loss = ranking_loss + brier_weight * blend_brier

                anchor_logits = model(anchor_tensors["inputs"], anchor_tensors["valid"])["risk"]
                with torch.no_grad():
                    parent_logits = parent(
                        anchor_tensors["inputs"], anchor_tensors["valid"]
                    )["risk"]
                anchor_mask = anchor_tensors["risk_mask"]
                anchor_loss = (
                    (anchor_logits - parent_logits).square() * anchor_mask
                ).sum() / anchor_mask.sum().clamp_min(1)
                loss = deployment_loss + anchor_weight * anchor_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch)
            deployment_sum += float(deployment_loss.detach()) * count
            anchor_sum += float(anchor_loss.detach()) * count
            total_sum += float(loss.detach()) * count
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
        item = AnchoredEpoch(
            epoch=epoch,
            train_loss=total_sum / max(state_count, 1),
            deployment_loss=deployment_sum / max(state_count, 1),
            anchor_loss=anchor_sum / max(state_count, 1),
            validation_regret=validation.mean_regret,
            validation_optimal_rate=validation.optimal_rate,
            validation_blend_brier=validation.blend_brier,
            validation_neural_brier=validation.neural_brier,
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
            "kind": "replay-agent-blend-anchor-v12",
            "parent_checkpoint": str(Path(checkpoint_path)),
            "blend_anchor": {
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
                "brier_weight": brier_weight,
                "anchor_weight": anchor_weight,
                "max_brier_regression": max_brier_regression,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": str(device),
                "amp": amp_enabled,
                "gpu_memory_fraction": gpu_memory_fraction,
                "train_scope": "all" if train_all else "risk_head",
            },
        }
    )
    torch.save({"state_dict": model.state_dict(), **metadata}, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps({**metadata, "history": [asdict(item) for item in history]}, indent=2),
        encoding="utf-8",
    )
    return initial, history


def _reservoir_anchor_examples(paths: Sequence[str | Path], limit: int, seed: int):
    rng = random.Random(seed)
    selected = []
    seen = 0
    for path in paths:
        for example in read_replay_examples(path):
            if not any(any(row) for row in example.risk_mask):
                continue
            prepared = _prepare_example(example)
            seen += 1
            if len(selected) < limit:
                selected.append(prepared)
                continue
            replacement = rng.randrange(seen)
            if replacement < limit:
                selected[replacement] = prepared
    return selected


def _within_brier_guard(
    candidate: BlendRankingMetrics,
    initial: BlendRankingMetrics,
    maximum_regression: float,
) -> bool:
    return (
        candidate.blend_brier <= initial.blend_brier * (1.0 + maximum_regression)
        and candidate.neural_brier <= initial.neural_brier * (1.0 + maximum_regression)
    )


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--brier-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--anchor-examples", type=int, default=2048)
    parser.add_argument("--max-brier-regression", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--batch-delay-ms", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--train-all", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    initial, history = fine_tune_blend_anchored(
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
        train_all=args.train_all,
        on_epoch=lambda item: print(json.dumps({"event": "epoch", **asdict(item)}), flush=True),
    )
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "initial": asdict(initial), "epochs": len(history)}
        )
    )


if __name__ == "__main__":
    main()
