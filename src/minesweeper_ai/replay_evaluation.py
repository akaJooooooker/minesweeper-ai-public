"""Holdout metrics for Replay Agent checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .replay import read_replay_examples
from .replay_learning import _collate, _prepare_example, _torch_model


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    *,
    batch_size: int = 32,
    device_name: str = "cpu",
) -> dict:
    torch, ReplayAgentNet = _torch_model()
    checkpoint = torch.load(Path(checkpoint_path), map_location=device_name, weights_only=True)
    model = ReplayAgentNet(checkpoint["model_width"]).to(device_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    examples = list(read_replay_examples(dataset_path))
    prepared = [_prepare_example(example) for example in examples]
    risk_squared = risk_count = 0.0
    policy_correct = policy_count = 0
    value_correct = value_count = 0
    value_squared = 0.0
    mode_counts = {"standard": 0, "no_guess": 0}

    with torch.inference_mode():
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            tensors = _collate(batch, torch)
            tensors = {key: value.to(device_name) for key, value in tensors.items()}
            output = model(tensors["inputs"], tensors["valid"])
            risk = torch.sigmoid(output["risk"])
            risk_squared += float(
                (((risk - tensors["risk_targets"]) ** 2) * tensors["risk_mask"]).sum()
            )
            risk_count += float(tensors["risk_mask"].sum())

            policy = torch.where(
                tensors["no_guess"][:, None, None, None],
                output["no_guess_policy"],
                output["standard_policy"],
            )
            policy = policy.masked_fill(~tensors["action_mask"], -1e4).flatten(1)
            eligible = tensors["policy_weight"] > 0
            policy_correct += int(
                ((policy.argmax(dim=1) == tensors["policy_target"]) & eligible).sum()
            )
            policy_count += int(eligible.sum())

            value = torch.sigmoid(output["value"])
            value_correct += int(((value >= 0.5) == (tensors["value_target"] >= 0.5)).sum())
            value_squared += float(((value - tensors["value_target"]) ** 2).sum())
            value_count += len(batch)

    for example in examples:
        mode_counts[example.mode] += 1
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "dataset": str(Path(dataset_path).resolve()),
        "examples": len(examples),
        "mode_examples": mode_counts,
        "risk_brier": risk_squared / max(risk_count, 1.0),
        "risk_cells": int(risk_count),
        "policy_top1_demonstration_accuracy": policy_correct / max(policy_count, 1),
        "policy_examples": policy_count,
        "value_accuracy": value_correct / max(value_count, 1),
        "value_brier": value_squared / max(value_count, 1),
        "value_examples": value_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m minesweeper_ai.replay_evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            evaluate_checkpoint(
                arguments.checkpoint,
                arguments.dataset,
                batch_size=arguments.batch_size,
                device_name=arguments.device,
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
