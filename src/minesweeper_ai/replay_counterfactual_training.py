"""Fine-tune the reveal policy from per-cell counterfactual win outcomes."""

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

from .replay_counterfactual import (
    CounterfactualExample,
    augment_counterfactual_example,
    read_counterfactual_examples,
)
from .replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model, encode_replay_board


@dataclass(frozen=True)
class CounterfactualMetrics:
    examples: int
    informative_examples: int
    candidates: int
    loss: float
    brier: float
    selected_win_rate: float
    oracle_win_rate: float
    decision_regret: float


@dataclass(frozen=True)
class CounterfactualEpoch:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_brier: float
    validation_selected_win_rate: float
    validation_oracle_win_rate: float
    validation_decision_regret: float
    best: bool


def fine_tune_counterfactual_policy(
    checkpoint_path: str | Path,
    dataset_paths: Sequence[str | Path],
    validation_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    ranking_weight: float = 1.0,
    bce_weight: float = 0.05,
    min_outcome_gap: float = 0.25,
    augment_train: bool = True,
    seed: int = 0,
    device_name: str = "auto",
    use_amp: bool = True,
    gpu_memory_fraction: float | None = None,
    batch_delay_ms: int = 0,
    patience: int = 8,
    on_epoch=None,
) -> tuple[CounterfactualMetrics, list[CounterfactualEpoch]]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if ranking_weight < 0:
        raise ValueError("ranking_weight must be non-negative")
    if bce_weight < 0:
        raise ValueError("bce_weight must be non-negative")
    if not 0.0 <= min_outcome_gap <= 1.0:
        raise ValueError("min_outcome_gap must be in [0, 1]")
    if patience < 1:
        raise ValueError("patience must be positive")

    torch, ReplayAgentNet = _torch_model()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _select_device(torch, device_name, gpu_memory_fraction)

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    if checkpoint.get("feature_channels") != REPLAY_FEATURE_CHANNELS:
        raise ValueError("checkpoint feature layout is incompatible")
    if checkpoint.get("action_order") != ["reveal", "flag", "chord"]:
        raise ValueError("checkpoint action layout is incompatible")
    model = ReplayAgentNet(int(checkpoint["model_width"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.standard_policy_head.parameters():
        parameter.requires_grad_(True)

    train_examples = _load_prepared(
        dataset_paths, min_outcome_gap, augment=augment_train
    )
    validation_examples = _load_prepared([validation_path], min_outcome_gap)
    if not train_examples or not validation_examples:
        raise ValueError("counterfactual train and validation datasets must be non-empty")
    if not _has_informative(train_examples) or not _has_informative(validation_examples):
        raise ValueError("counterfactual datasets must include an informative state")

    # No weight decay: the two unused action channels share this Conv2d
    # parameter and must remain byte-for-byte identical to the parent.
    optimizer = torch.optim.AdamW(
        model.standard_policy_head.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    initial = _metrics(
        model,
        validation_examples,
        batch_size,
        ranking_weight,
        bce_weight,
        torch,
        device,
    )
    best_metrics = initial
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[CounterfactualEpoch] = []

    for epoch in range(1, epochs + 1):
        batches = [
            train_examples[start : start + batch_size]
            for start in range(0, len(train_examples), batch_size)
        ]
        random.shuffle(batches)
        loss_sum = 0.0
        example_count = 0
        model.train()
        for batch in batches:
            tensors = {
                key: value.to(device)
                for key, value in _collate_counterfactual(batch, torch).items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(tensors["inputs"], tensors["valid"])[
                    "standard_policy"
                ][:, 0]
                loss = _counterfactual_loss(
                    logits,
                    tensors,
                    ranking_weight,
                    bce_weight,
                    torch,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.standard_policy_head.parameters(),
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(batch)
            example_count += len(batch)
            if batch_delay_ms:
                time.sleep(batch_delay_ms / 1_000)

        validation = _metrics(
            model,
            validation_examples,
            batch_size,
            ranking_weight,
            bce_weight,
            torch,
            device,
        )
        improved = _is_better(validation, best_metrics)
        if improved:
            best_metrics = validation
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        item = CounterfactualEpoch(
            epoch=epoch,
            train_loss=loss_sum / max(example_count, 1),
            validation_loss=validation.loss,
            validation_brier=validation.brier,
            validation_selected_win_rate=validation.selected_win_rate,
            validation_oracle_win_rate=validation.oracle_win_rate,
            validation_decision_regret=validation.decision_regret,
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
        key: value
        for key, value in checkpoint.items()
        if key not in {"state_dict", "history", "training"}
    }
    metadata.update(
        {
            "kind": "replay-agent-counterfactual-policy-v35",
            "parent_checkpoint": str(Path(checkpoint_path)),
            "counterfactual_policy": {
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
                "ranking_weight": ranking_weight,
                "bce_weight": bce_weight,
                "min_outcome_gap": min_outcome_gap,
                "augment_train": augment_train,
                "seed": seed,
                "device": str(device),
                "amp": amp_enabled,
                "gpu_memory_fraction": gpu_memory_fraction,
                "batch_delay_ms": batch_delay_ms,
                "patience": patience,
            },
        }
    )
    torch.save({"state_dict": model.state_dict(), **metadata}, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {**metadata, "history": [asdict(item) for item in history]},
            indent=2,
        ),
        encoding="utf-8",
    )
    return initial, history


def _select_device(torch, device_name: str, gpu_memory_fraction: float | None):
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
    return device


def _load_prepared(
    paths: Sequence[str | Path],
    min_outcome_gap: float = 0.25,
    *,
    augment: bool = False,
):
    prepared = []
    for path in paths:
        for example in read_counterfactual_examples(path):
            if len(example.candidates) < 2 or _outcome_gap(example) < min_outcome_gap:
                continue
            variants = augment_counterfactual_example(example) if augment else [example]
            prepared.extend(_prepare_example(item) for item in variants)
    return prepared


def _outcome_gap(example: CounterfactualExample) -> float:
    outcomes = [candidate.outcome for candidate in example.candidates]
    return max(outcomes) - min(outcomes)


def _prepare_example(example: CounterfactualExample):
    features, valid = encode_replay_board(
        example.observation,
        example.total_mines,
        mode="standard",
        source=example.source,
    )
    return features, valid, example


def _has_informative(examples) -> bool:
    return any(
        len({candidate.outcome for candidate in item[2].candidates}) > 1
        for item in examples
    )


def _collate_counterfactual(examples, torch):
    batch_size = len(examples)
    max_height = max(item[1].shape[0] for item in examples)
    max_width = max(item[1].shape[1] for item in examples)
    inputs = np.zeros(
        (batch_size, REPLAY_FEATURE_CHANNELS, max_height, max_width),
        dtype=np.float32,
    )
    valid = np.zeros((batch_size, max_height, max_width), dtype=np.float32)
    candidate_mask = np.zeros_like(valid, dtype=np.bool_)
    outcomes = np.zeros_like(valid, dtype=np.float32)
    informative = np.zeros(batch_size, dtype=np.bool_)

    for index, (features, board_valid, example) in enumerate(examples):
        height, width = board_valid.shape
        inputs[index, :, :height, :width] = features
        valid[index, :height, :width] = board_valid
        values = []
        for candidate in example.candidates:
            if not (0 <= candidate.row < height and 0 <= candidate.col < width):
                raise ValueError("counterfactual candidate is outside its board")
            if candidate_mask[index, candidate.row, candidate.col]:
                raise ValueError("counterfactual candidate coordinates must be unique")
            candidate_mask[index, candidate.row, candidate.col] = True
            outcomes[index, candidate.row, candidate.col] = candidate.outcome
            values.append(candidate.outcome)
        informative[index] = len(set(values)) > 1

    return {
        "inputs": torch.from_numpy(inputs),
        "valid": torch.from_numpy(valid),
        "candidate_mask": torch.from_numpy(candidate_mask),
        "outcomes": torch.from_numpy(outcomes),
        "informative": torch.from_numpy(informative),
    }


def _counterfactual_loss(
    logits, tensors, ranking_weight: float, bce_weight: float, torch
):
    from torch.nn import functional

    mask = tensors["candidate_mask"]
    outcomes = tensors["outcomes"]
    elements = functional.binary_cross_entropy_with_logits(
        logits,
        outcomes,
        reduction="none",
    )
    positives = (outcomes >= 0.5) & mask
    negatives = (outcomes < 0.5) & mask
    positive_count = positives.sum().float()
    negative_count = negatives.sum().float()
    total = (positive_count + negative_count).clamp_min(1.0)
    positive_weight = 0.5 * total / positive_count.clamp_min(1.0)
    negative_weight = 0.5 * total / negative_count.clamp_min(1.0)
    weights = torch.where(positives, positive_weight, negative_weight) * mask
    bce = (elements * weights).sum() / weights.sum().clamp_min(1.0)

    flat_mask = mask.flatten(1)
    flat_logits = logits.flatten(1).masked_fill(~flat_mask, -1e4)
    flat_outcomes = outcomes.flatten(1)
    best = flat_outcomes.masked_fill(~flat_mask, -1.0).amax(dim=1, keepdim=True)
    targets = (flat_outcomes == best) & flat_mask
    target_distribution = targets.float() / targets.sum(dim=1, keepdim=True).clamp_min(1.0)
    ranking_elements = -(target_distribution * flat_logits.log_softmax(dim=1)).sum(dim=1)
    informative = tensors["informative"].float()
    ranking = (ranking_elements * informative).sum() / informative.sum().clamp_min(1.0)
    return bce_weight * bce + ranking_weight * ranking


def _metrics(
    model,
    examples,
    batch_size: int,
    ranking_weight: float,
    bce_weight: float,
    torch,
    device,
) -> CounterfactualMetrics:
    loss_sum = 0.0
    brier_sum = 0.0
    candidate_count = 0
    selected_sum = 0.0
    oracle_sum = 0.0
    informative_count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            tensors = {
                key: value.to(device)
                for key, value in _collate_counterfactual(batch, torch).items()
            }
            logits = model(tensors["inputs"], tensors["valid"])[
                "standard_policy"
            ][:, 0]
            loss = _counterfactual_loss(
                logits,
                tensors,
                ranking_weight,
                bce_weight,
                torch,
            )
            loss_sum += float(loss) * len(batch)
            mask = tensors["candidate_mask"]
            outcomes = tensors["outcomes"]
            probabilities = logits.sigmoid()
            brier_sum += float(((probabilities - outcomes).square() * mask).sum())
            candidate_count += int(mask.sum())
            flat_mask = mask.flatten(1)
            flat_logits = logits.flatten(1).masked_fill(~flat_mask, -1e4)
            flat_outcomes = outcomes.flatten(1)
            selected = flat_outcomes.gather(
                1,
                flat_logits.argmax(dim=1, keepdim=True),
            ).squeeze(1)
            oracle = flat_outcomes.masked_fill(~flat_mask, -1.0).amax(dim=1)
            selected_sum += float(selected.sum())
            oracle_sum += float(oracle.sum())
            informative_count += int(tensors["informative"].sum())
    selected_rate = selected_sum / len(examples)
    oracle_rate = oracle_sum / len(examples)
    return CounterfactualMetrics(
        examples=len(examples),
        informative_examples=informative_count,
        candidates=candidate_count,
        loss=loss_sum / len(examples),
        brier=brier_sum / max(candidate_count, 1),
        selected_win_rate=selected_rate,
        oracle_win_rate=oracle_rate,
        decision_regret=oracle_rate - selected_rate,
    )


def _is_better(candidate: CounterfactualMetrics, best: CounterfactualMetrics) -> bool:
    if candidate.decision_regret < best.decision_regret - 1e-9:
        return True
    return (
        abs(candidate.decision_regret - best.decision_regret) <= 1e-9
        and candidate.brier < best.brier - 1e-9
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.05)
    parser.add_argument("--min-outcome-gap", type=float, default=0.25)
    parser.add_argument(
        "--augment-train",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=20260935)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--batch-delay-ms", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    initial, history = fine_tune_counterfactual_policy(
        args.checkpoint,
        args.dataset,
        args.validation,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ranking_weight=args.ranking_weight,
        bce_weight=args.bce_weight,
        min_outcome_gap=args.min_outcome_gap,
        augment_train=args.augment_train,
        seed=args.seed,
        device_name=args.device,
        use_amp=not args.no_amp,
        gpu_memory_fraction=args.gpu_memory_fraction,
        batch_delay_ms=args.batch_delay_ms,
        patience=args.patience,
        on_epoch=lambda item: print(
            json.dumps({"event": "epoch", **asdict(item)}),
            flush=True,
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "initial": asdict(initial),
                "epochs": len(history),
            }
        )
    )


if __name__ == "__main__":
    main()

