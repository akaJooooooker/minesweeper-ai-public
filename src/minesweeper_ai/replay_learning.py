"""Standalone Replay Policy/Risk/Value bootstrap and training entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Callable, Sequence

import numpy as np

from .data import BoardSpec
from .features import FEATURE_CHANNELS, encode_board
from .game import FLAGGED, UNKNOWN
from .replay import (
    ReplayAuditor,
    ReplayTrainingExample,
    generate_simulator_replays,
    read_replay_examples,
    read_replays,
    write_replay_examples,
    write_replays,
)
from .solver import ConstraintSolver

REPLAY_FEATURE_CHANNELS = FEATURE_CHANNELS + 5
ACTION_INDEX = {"reveal": 0, "flag": 1, "chord": 2}

STANDARD_SPECS = (
    BoardSpec(9, 9, 10, True),
    BoardSpec(16, 16, 40, True),
    BoardSpec(30, 16, 99, True),
)
NO_GUESS_SPECS = (
    BoardSpec(9, 9, 10, True),
    BoardSpec(16, 16, 40, True),
    BoardSpec(30, 16, 99, True),
    BoardSpec(30, 24, 130, True),
)


def encode_replay_board(
    observation,
    total_mines: int,
    *,
    mode: str,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    base, valid = encode_board(observation, total_mines)
    if mode not in {"standard", "no_guess"}:
        raise ValueError(f"unsupported replay mode: {mode}")
    height, width = valid.shape
    extra = np.zeros((5, height, width), dtype=np.float32)
    extra[0].fill(mode == "standard")
    extra[1].fill(mode == "no_guess")
    revealed = sum(cell >= 0 for row in observation for cell in row)
    extra[2].fill(min(1.0, revealed / max(height * width - total_mines, 1)))
    extra[3].fill(total_mines / (height * width))
    extra[4].fill(not source.startswith("simulator_"))
    return np.concatenate((base, extra), axis=0), valid


def _torch_model():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:
        raise RuntimeError("Replay learning requires PyTorch") from error

    class MaskedGroupNorm(nn.Module):
        """GroupNorm that excludes padded spatial cells from its moments."""

        def __init__(self, groups: int, channels: int, epsilon: float = 1e-5) -> None:
            super().__init__()
            if channels % groups:
                raise ValueError("channels must be divisible by groups")
            self.groups = groups
            self.epsilon = epsilon
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))

        def forward(self, inputs, mask):
            batch, channels, height, width = inputs.shape
            grouped = inputs.reshape(batch, self.groups, channels // self.groups, height, width)
            grouped_mask = mask.reshape(batch, 1, 1, height, width)
            denominator = (grouped_mask.sum(dim=(3, 4), keepdim=True) * (channels // self.groups)).clamp_min(1.0)
            mean = (grouped * grouped_mask).sum(dim=(2, 3, 4), keepdim=True) / denominator
            variance = (
                ((grouped - mean).square() * grouped_mask).sum(dim=(2, 3, 4), keepdim=True)
                / denominator
            )
            normalized = ((grouped - mean) / torch.sqrt(variance + self.epsilon)).reshape_as(inputs)
            return (
                normalized * self.weight.reshape(1, channels, 1, 1)
                + self.bias.reshape(1, channels, 1, 1)
            ) * mask

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                MaskedGroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                MaskedGroupNorm(8, channels),
            )
            self.activation = nn.SiLU()

        def forward(self, inputs, mask):
            hidden = self.body[0](inputs)
            hidden = self.body[1](hidden, mask)
            hidden = self.body[2](hidden)
            hidden = self.body[3](hidden)
            hidden = self.body[4](hidden, mask)
            return self.activation(inputs + hidden) * mask

    class ReplayAgentNet(nn.Module):
        def __init__(self, width: int = 64) -> None:
            super().__init__()
            if width % 8:
                raise ValueError("model width must be divisible by 8")
            self.width = width
            self.stem = nn.Sequential(
                nn.Conv2d(REPLAY_FEATURE_CHANNELS, width, 3, padding=1),
                MaskedGroupNorm(8, width),
                nn.SiLU(),
            )
            self.blocks = nn.Sequential(
                *(ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8, 1, 2))
            )
            self.global_projection = nn.Sequential(nn.Conv2d(width, width, 1), nn.SiLU())
            self.risk_head = nn.Sequential(
                nn.Conv2d(width, width // 2, 1),
                nn.SiLU(),
                nn.Conv2d(width // 2, 1, 1),
            )
            self.standard_policy_head = nn.Conv2d(width, 3, 1)
            self.no_guess_policy_head = nn.Conv2d(width, 3, 1)
            self.value_head = nn.Sequential(
                nn.Linear(width, width // 2),
                nn.SiLU(),
                nn.Linear(width // 2, 1),
            )

        def forward(self, inputs, valid_mask):
            mask = valid_mask.unsqueeze(1)
            hidden = self.stem[0](inputs)
            hidden = self.stem[1](hidden, mask)
            hidden = self.stem[2](hidden) * mask
            for block in self.blocks:
                hidden = block(hidden, mask)
            denominator = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            pooled = (hidden * mask).sum(dim=(2, 3), keepdim=True) / denominator
            hidden = (hidden + self.global_projection(pooled)) * mask
            return {
                "risk": self.risk_head(hidden).squeeze(1),
                "standard_policy": self.standard_policy_head(hidden),
                "no_guess_policy": self.no_guess_policy_head(hidden),
                "value": self.value_head(pooled.flatten(1)).squeeze(1),
            }

    return torch, ReplayAgentNet


def train_replay_agent(
    dataset_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    model_width: int = 64,
    seed: int = 0,
    device_name: str = "auto",
    use_amp: bool = True,
    gpu_memory_fraction: float | None = None,
    batch_delay_ms: int = 0,
    on_epoch: Callable[[dict], None] | None = None,
) -> list[dict[str, float]]:
    torch, ReplayAgentNet = _torch_model()
    from torch.nn import functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    paths = [Path(dataset_path)] if isinstance(dataset_path, (str, Path)) else [Path(p) for p in dataset_path]
    prepared = [
        _prepare_example(example)
        for path in paths
        for example in read_replay_examples(path)
    ]
    if not prepared:
        raise ValueError("Replay dataset is empty")
    prepared.sort(key=lambda item: item[1].size)
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

    model = ReplayAgentNet(model_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        batches = [prepared[i : i + batch_size] for i in range(0, len(prepared), batch_size)]
        random.shuffle(batches)
        sums = {"total": 0.0, "policy": 0.0, "risk": 0.0, "value": 0.0}
        sample_count = 0
        model.train()
        for batch in batches:
            tensors = _collate(batch, torch)
            tensors = {key: value.to(device) for key, value in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(tensors["inputs"], tensors["valid"])
                risk_elements = functional.binary_cross_entropy_with_logits(
                    output["risk"], tensors["risk_targets"], reduction="none"
                )
                risk_weight = tensors["risk_mask"].sum().clamp_min(1.0)
                risk_loss = (risk_elements * tensors["risk_mask"]).sum() / risk_weight

                policy_logits = torch.where(
                    tensors["no_guess"][:, None, None, None],
                    output["no_guess_policy"],
                    output["standard_policy"],
                )
                masked_policy = policy_logits.masked_fill(~tensors["action_mask"], -1e4)
                policy_elements = functional.cross_entropy(
                    masked_policy.flatten(1), tensors["policy_target"], reduction="none"
                )
                policy_denominator = tensors["policy_weight"].sum().clamp_min(1.0)
                policy_loss = (policy_elements * tensors["policy_weight"]).sum() / policy_denominator

                value_elements = functional.binary_cross_entropy_with_logits(
                    output["value"], tensors["value_target"], reduction="none"
                )
                value_loss = (
                    value_elements * tensors["value_weight"]
                ).sum() / tensors["value_weight"].sum().clamp_min(1.0)
                total_loss = policy_loss + 0.5 * risk_loss + 0.3 * value_loss
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch)
            sample_count += count
            sums["total"] += float(total_loss.detach()) * count
            sums["policy"] += float(policy_loss.detach()) * count
            sums["risk"] += float(risk_loss.detach()) * count
            sums["value"] += float(value_loss.detach()) * count
            if batch_delay_ms:
                time.sleep(batch_delay_ms / 1_000)
        metrics = {name: value / sample_count for name, value in sums.items()}
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        if on_epoch:
            on_epoch(metrics)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "replay-agent-v1",
        "feature_channels": REPLAY_FEATURE_CHANNELS,
        "action_order": ["reveal", "flag", "chord"],
        "model_width": model_width,
        "history": history,
        "training": {
            "datasets": [str(path) for path in paths],
            "examples": len(prepared),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "device": str(device),
            "amp": amp_enabled,
            "gpu_memory_fraction": gpu_memory_fraction,
            "batch_delay_ms": batch_delay_ms,
        },
    }
    torch.save({"state_dict": model.state_dict(), **metadata}, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return history


def _prepare_example(example: ReplayTrainingExample):
    features, valid = encode_replay_board(
        example.observation,
        example.total_mines,
        mode=example.mode,
        source=example.source,
    )
    risk_targets = np.asarray(
        [[0.0 if value is None else value for value in row] for row in example.probabilities],
        dtype=np.float32,
    )
    # Older audited datasets could contain labels for flagged cells because
    # flags were temporarily hidden from the teacher solver. Keep those files
    # readable, but never train or evaluate risk on an illegal reveal.
    unknown = np.asarray(
        [[cell == UNKNOWN for cell in row] for row in example.observation],
        dtype=np.float32,
    )
    risk_mask = np.asarray(example.risk_mask, dtype=np.float32) * unknown
    return features, valid, risk_targets, risk_mask, example


def _collate(examples, torch_module):
    batch_size = len(examples)
    max_height = max(item[1].shape[0] for item in examples)
    max_width = max(item[1].shape[1] for item in examples)
    inputs = np.zeros((batch_size, REPLAY_FEATURE_CHANNELS, max_height, max_width), np.float32)
    valid = np.zeros((batch_size, max_height, max_width), np.float32)
    risk_targets = np.zeros_like(valid)
    risk_mask = np.zeros_like(valid)
    action_mask = np.zeros((batch_size, 3, max_height, max_width), dtype=np.bool_)
    policy_target = np.zeros(batch_size, dtype=np.int64)
    policy_weight = np.zeros(batch_size, dtype=np.float32)
    value_target = np.zeros(batch_size, dtype=np.float32)
    value_weight = np.zeros(batch_size, dtype=np.float32)
    no_guess = np.zeros(batch_size, dtype=np.bool_)

    for index, (features, board_valid, board_risk, board_risk_mask, example) in enumerate(examples):
        height, width = board_valid.shape
        inputs[index, :, :height, :width] = features
        valid[index, :height, :width] = board_valid
        risk_targets[index, :height, :width] = board_risk
        risk_mask[index, :height, :width] = board_risk_mask
        observation = example.observation
        for row in range(height):
            for col in range(width):
                cell = observation[row][col]
                if cell == UNKNOWN:
                    action_mask[index, 0, row, col] = True
                    action_mask[index, 1, row, col] = True
                elif cell >= 0 and _can_chord(observation, row, col):
                    action_mask[index, 2, row, col] = True
        action = ACTION_INDEX[example.action]
        policy_target[index] = action * max_height * max_width + example.row * max_width + example.col
        target_is_legal = (
            0 <= example.row < height
            and 0 <= example.col < width
            and action_mask[index, action, example.row, example.col]
        )
        policy_weight[index] = example.policy_weight if target_is_legal else 0.0
        value_target[index] = example.value_target
        value_weight[index] = example.value_weight
        no_guess[index] = example.mode == "no_guess"
    return {
        "inputs": torch_module.from_numpy(inputs),
        "valid": torch_module.from_numpy(valid),
        "risk_targets": torch_module.from_numpy(risk_targets),
        "risk_mask": torch_module.from_numpy(risk_mask),
        "action_mask": torch_module.from_numpy(action_mask),
        "policy_target": torch_module.from_numpy(policy_target),
        "policy_weight": torch_module.from_numpy(policy_weight),
        "value_target": torch_module.from_numpy(value_target),
        "value_weight": torch_module.from_numpy(value_weight),
        "no_guess": torch_module.from_numpy(no_guess),
    }


def _can_chord(observation, row: int, col: int) -> bool:
    clue = observation[row][col]
    height = len(observation)
    width = len(observation[0])
    flags = covered = 0
    for next_row in range(max(0, row - 1), min(height, row + 2)):
        for next_col in range(max(0, col - 1), min(width, col + 2)):
            if (next_row, next_col) == (row, col):
                continue
            flags += observation[next_row][next_col] == FLAGGED
            covered += observation[next_row][next_col] == UNKNOWN
    return covered > 0 and flags == clue


def bootstrap_replay_dataset(
    raw_output: str | Path,
    dataset_output: str | Path,
    *,
    standard_games_per_spec: int,
    no_guess_games_per_spec: int,
    seed: int,
    teacher_cells: int,
    max_attempts: int,
) -> dict:
    teacher = ConstraintSolver(teacher_cells)
    replays = list(
        generate_simulator_replays(
            STANDARD_SPECS,
            standard_games_per_spec,
            mode="standard",
            seed=seed,
            solver=teacher,
        )
    )
    if no_guess_games_per_spec:
        replays.extend(
            generate_simulator_replays(
                NO_GUESS_SPECS,
                no_guess_games_per_spec,
                mode="no_guess",
                seed=seed + 1,
                solver=teacher,
                max_attempts=max_attempts,
            )
        )
    replay_count = write_replays(replays, raw_output)
    auditor = ReplayAuditor(teacher)
    examples = [example for replay in replays for example in auditor.audit(replay)]
    example_count = write_replay_examples(examples, dataset_output)
    classes: dict[str, int] = {}
    for example in examples:
        classes[example.classification] = classes.get(example.classification, 0) + 1
    return {
        "raw_output": str(Path(raw_output).resolve()),
        "dataset_output": str(Path(dataset_output).resolve()),
        "replays": replay_count,
        "examples": example_count,
        "won_replays": sum(replay.won for replay in replays),
        "modes": {
            "standard": sum(replay.mode == "standard" for replay in replays),
            "no_guess": sum(replay.mode == "no_guess" for replay in replays),
        },
        "classifications": classes,
    }


def _audit_existing(raw_input: Path, dataset_output: Path, teacher_cells: int) -> dict:
    auditor = ReplayAuditor(ConstraintSolver(teacher_cells))
    replays = list(read_replays(raw_input))
    examples = [example for replay in replays for example in auditor.audit(replay)]
    count = write_replay_examples(examples, dataset_output)
    return {"replays": len(replays), "examples": count, "output": str(dataset_output.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m minesweeper_ai.replay_learning")
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--raw-output", type=Path, required=True)
    bootstrap.add_argument("--dataset-output", type=Path, required=True)
    bootstrap.add_argument("--standard-games-per-spec", type=int, default=10)
    bootstrap.add_argument("--no-guess-games-per-spec", type=int, default=2)
    bootstrap.add_argument("--seed", type=int, default=20260902)
    bootstrap.add_argument("--teacher-cells", type=int, default=24)
    bootstrap.add_argument("--max-attempts", type=int, default=2000)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--teacher-cells", type=int, default=24)
    train = subparsers.add_parser("train")
    train.add_argument("--dataset", type=Path, nargs="+", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--model-width", type=int, default=64)
    train.add_argument("--seed", type=int, default=20260902)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--gpu-memory-fraction", type=float)
    train.add_argument("--batch-delay-ms", type=int, default=0)
    arguments = parser.parse_args(argv)

    if arguments.command == "bootstrap":
        result = bootstrap_replay_dataset(
            arguments.raw_output,
            arguments.dataset_output,
            standard_games_per_spec=arguments.standard_games_per_spec,
            no_guess_games_per_spec=arguments.no_guess_games_per_spec,
            seed=arguments.seed,
            teacher_cells=arguments.teacher_cells,
            max_attempts=arguments.max_attempts,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    if arguments.command == "audit":
        print(json.dumps(_audit_existing(arguments.input, arguments.output, arguments.teacher_cells)), flush=True)
        return 0

    def report(metrics: dict) -> None:
        print(json.dumps({"event": "epoch", **metrics}), flush=True)

    history = train_replay_agent(
        arguments.dataset,
        arguments.output,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        model_width=arguments.model_width,
        seed=arguments.seed,
        device_name=arguments.device,
        use_amp=not arguments.no_amp,
        gpu_memory_fraction=arguments.gpu_memory_fraction,
        batch_delay_ms=arguments.batch_delay_ms,
        on_epoch=report,
    )
    print(json.dumps({"output": str(arguments.output.resolve()), "history": history}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
