"""Fine-tune the standard guess policy from whole-game win/loss outcomes."""

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


@dataclass(frozen=True)
class OutcomePolicyMetrics:
    examples: int
    wins: int
    losses: int
    loss: float
    win_chosen_probability: float
    loss_chosen_probability: float
    separation: float


@dataclass(frozen=True)
class OutcomePolicyEpoch:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_win_probability: float
    validation_loss_probability: float
    validation_separation: float
    best: bool


def fine_tune_outcome_policy(
    checkpoint_path: str | Path,
    dataset_paths: Sequence[str | Path],
    validation_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    min_progress: float = 0.1,
    seed: int = 0,
    device_name: str = "auto",
    use_amp: bool = True,
    gpu_memory_fraction: float | None = None,
    batch_delay_ms: int = 0,
    patience: int = 10,
    on_epoch=None,
) -> tuple[OutcomePolicyMetrics, list[OutcomePolicyEpoch]]:
    if not 0.0 <= min_progress < 1.0:
        raise ValueError("min_progress must be in [0, 1)")
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
    for parameter in model.standard_policy_head.parameters():
        parameter.requires_grad_(True)

    train_examples = _load_outcome_examples(dataset_paths, min_progress)
    validation_examples = _load_outcome_examples([validation_path], min_progress)
    if not train_examples or not validation_examples:
        raise ValueError("outcome policy train and validation datasets must be non-empty")
    if not _has_both_outcomes(train_examples) or not _has_both_outcomes(validation_examples):
        raise ValueError("outcome policy datasets must contain wins and losses")

    optimizer = torch.optim.AdamW(
        model.standard_policy_head.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    initial = _outcome_metrics(model, validation_examples, batch_size, torch, device)
    best_metrics = initial
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[OutcomePolicyEpoch] = []

    for epoch in range(1, epochs + 1):
        batches = [
            train_examples[start : start + batch_size]
            for start in range(0, len(train_examples), batch_size)
        ]
        random.shuffle(batches)
        loss_sum = 0.0
        weight_sum = 0.0
        model.train()
        for batch in batches:
            tensors = {
                key: value.to(device)
                for key, value in _collate_outcome(batch, torch).items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(tensors["inputs"], tensors["valid"])["standard_policy"][:, 0]
                loss, effective_weight = _balanced_bandit_loss(logits, tensors, torch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.standard_policy_head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * effective_weight
            weight_sum += effective_weight
            if batch_delay_ms:
                time.sleep(batch_delay_ms / 1_000)

        validation = _outcome_metrics(
            model,
            validation_examples,
            batch_size,
            torch,
            device,
        )
        improved = validation.loss < best_metrics.loss - 1e-8
        if improved:
            best_metrics = validation
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        item = OutcomePolicyEpoch(
            epoch=epoch,
            train_loss=loss_sum / max(weight_sum, 1.0),
            validation_loss=validation.loss,
            validation_win_probability=validation.win_chosen_probability,
            validation_loss_probability=validation.loss_chosen_probability,
            validation_separation=validation.separation,
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
            "kind": "replay-agent-outcome-policy-v33",
            "parent_checkpoint": str(Path(checkpoint_path)),
            "outcome_policy": {
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
                "min_progress": min_progress,
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


def _load_outcome_examples(paths: Sequence[str | Path], min_progress: float):
    result = []
    for path in paths:
        for example in read_replay_examples(path):
            if (
                example.mode != "standard"
                or example.action != "reveal"
                or example.classification != "NECESSARY_GUESS"
                or not example.teacher_exact
            ):
                continue
            height = len(example.observation)
            width = len(example.observation[0])
            safe_cells = height * width - example.total_mines
            revealed = sum(cell >= 0 for row in example.observation for cell in row)
            progress = revealed / max(safe_cells, 1)
            if progress < min_progress:
                continue
            prepared = _prepare_example(example)
            risk_mask = prepared[3]
            if (
                not risk_mask.any()
                or not (0 <= example.row < height and 0 <= example.col < width)
                or not risk_mask[example.row, example.col]
            ):
                continue
            result.append((prepared, 0.5 + 0.5 * progress))
    return result


def _has_both_outcomes(examples) -> bool:
    outcomes = {int(item[0][4].value_target >= 0.5) for item in examples}
    return outcomes == {0, 1}


def _collate_outcome(examples, torch):
    prepared = [item[0] for item in examples]
    tensors = _collate(prepared, torch)
    _, height, width = tensors["valid"].shape
    chosen = np.asarray(
        [item[4].row * width + item[4].col for item in prepared],
        dtype=np.int64,
    )
    weights = np.asarray([item[1] for item in examples], dtype=np.float32)
    tensors["candidate_mask"] = tensors["risk_mask"].bool()
    tensors["chosen"] = torch.from_numpy(chosen)
    tensors["outcome"] = tensors["value_target"]
    tensors["outcome_weight"] = torch.from_numpy(weights)
    return tensors


def _balanced_bandit_loss(logits, tensors, torch):
    flat_mask = tensors["candidate_mask"].flatten(1)
    flat_logits = logits.flatten(1).masked_fill(~flat_mask, -1e4)
    chosen_probability = flat_logits.softmax(dim=1).gather(
        1,
        tensors["chosen"][:, None],
    ).squeeze(1)
    outcome = tensors["outcome"]
    positive = outcome >= 0.5
    negative = ~positive
    positive_weight = 0.5 / positive.float().mean().clamp_min(1e-6)
    negative_weight = 0.5 / negative.float().mean().clamp_min(1e-6)
    class_weight = torch.where(positive, positive_weight, negative_weight)
    weight = tensors["outcome_weight"] * class_weight
    elements = torch.where(
        positive,
        -chosen_probability.clamp_min(1e-7).log(),
        -(1.0 - chosen_probability).clamp_min(1e-7).log(),
    )
    effective_weight = float(weight.sum().detach())
    return (elements * weight).sum() / weight.sum().clamp_min(1e-6), effective_weight


def _outcome_metrics(model, examples, batch_size, torch, device) -> OutcomePolicyMetrics:
    loss_sum = weight_sum = 0.0
    win_probability_sum = loss_probability_sum = 0.0
    wins = losses = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            tensors = {
                key: value.to(device)
                for key, value in _collate_outcome(
                    examples[start : start + batch_size],
                    torch,
                ).items()
            }
            logits = model(tensors["inputs"], tensors["valid"])["standard_policy"][:, 0]
            loss, effective_weight = _balanced_bandit_loss(logits, tensors, torch)
            loss_sum += float(loss) * effective_weight
            weight_sum += effective_weight
            probabilities = logits.flatten(1).masked_fill(
                ~tensors["candidate_mask"].flatten(1),
                -1e4,
            ).softmax(dim=1)
            chosen = probabilities.gather(1, tensors["chosen"][:, None]).squeeze(1)
            positive = tensors["outcome"] >= 0.5
            win_probability_sum += float(chosen[positive].sum())
            loss_probability_sum += float(chosen[~positive].sum())
            wins += int(positive.sum())
            losses += int((~positive).sum())
    win_mean = win_probability_sum / max(wins, 1)
    loss_mean = loss_probability_sum / max(losses, 1)
    return OutcomePolicyMetrics(
        examples=len(examples),
        wins=wins,
        losses=losses,
        loss=loss_sum / max(weight_sum, 1.0),
        win_chosen_probability=win_mean,
        loss_chosen_probability=loss_mean,
        separation=win_mean - loss_mean,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--min-progress", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260933)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--batch-delay-ms", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    initial, history = fine_tune_outcome_policy(
        args.checkpoint,
        args.dataset,
        args.validation,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_progress=args.min_progress,
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

