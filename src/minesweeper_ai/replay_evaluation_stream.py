"""Memory-bounded streaming evaluation for Replay Agent checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from .replay import ReplayTrainingExample, read_replay_examples
from .replay_learning import _collate, _prepare_example, _torch_model


def evaluate_checkpoint_streaming(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    *,
    batch_size: int = 16,
    device_name: str = "cpu",
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    torch, ReplayAgentNet = _torch_model()
    checkpoint = torch.load(Path(checkpoint_path), map_location=device_name, weights_only=True)
    model = ReplayAgentNet(checkpoint["model_width"]).to(device_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    totals = {
        "examples": 0,
        "risk_squared": 0.0,
        "risk_count": 0.0,
        "policy_correct": 0,
        "policy_count": 0,
        "value_correct": 0,
        "value_squared": 0.0,
        "value_count": 0,
    }
    mode_counts = {"standard": 0, "no_guess": 0}
    stream = read_replay_examples(dataset_path)
    with torch.inference_mode():
        while batch := _take(stream, batch_size):
            for example in batch:
                mode_counts[example.mode] += 1
            prepared = [_prepare_example(example) for example in batch]
            tensors = {
                key: value.to(device_name)
                for key, value in _collate(prepared, torch).items()
            }
            output = model(tensors["inputs"], tensors["valid"])
            risk = torch.sigmoid(output["risk"])
            totals["risk_squared"] += float(
                (((risk - tensors["risk_targets"]) ** 2) * tensors["risk_mask"]).sum()
            )
            totals["risk_count"] += float(tensors["risk_mask"].sum())
            policy = torch.where(
                tensors["no_guess"][:, None, None, None],
                output["no_guess_policy"],
                output["standard_policy"],
            )
            policy = policy.masked_fill(~tensors["action_mask"], -1e4).flatten(1)
            eligible = tensors["policy_weight"] > 0
            totals["policy_correct"] += int(
                ((policy.argmax(dim=1) == tensors["policy_target"]) & eligible).sum()
            )
            totals["policy_count"] += int(eligible.sum())
            value = torch.sigmoid(output["value"])
            totals["value_correct"] += int(
                ((value >= 0.5) == (tensors["value_target"] >= 0.5)).sum()
            )
            totals["value_squared"] += float(((value - tensors["value_target"]) ** 2).sum())
            totals["value_count"] += len(batch)
            totals["examples"] += len(batch)
            del prepared, tensors, output, risk, policy, value
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "dataset": str(Path(dataset_path).resolve()),
        "examples": totals["examples"],
        "mode_examples": mode_counts,
        "risk_brier": totals["risk_squared"] / max(totals["risk_count"], 1.0),
        "risk_cells": int(totals["risk_count"]),
        "policy_top1_demonstration_accuracy": totals["policy_correct"]
        / max(totals["policy_count"], 1),
        "policy_examples": totals["policy_count"],
        "value_accuracy": totals["value_correct"] / max(totals["value_count"], 1),
        "value_brier": totals["value_squared"] / max(totals["value_count"], 1),
        "value_examples": totals["value_count"],
    }


def _take(
    stream: Iterable[ReplayTrainingExample], count: int
) -> list[ReplayTrainingExample]:
    iterator = iter(stream)
    items: list[ReplayTrainingExample] = []
    for _ in range(count):
        try:
            items.append(next(iterator))
        except StopIteration:
            break
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = evaluate_checkpoint_streaming(
        args.checkpoint,
        args.dataset,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
