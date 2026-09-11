"""Masked soft-label training for the optional PyTorch model."""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Callable, Sequence

import numpy as np

from .data import read_jsonl
from .features import FEATURE_CHANNELS, encode_board


def train_model(
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
    on_epoch: Callable[[int, int, float], None] | None = None,
) -> list[float]:
    try:
        import torch
        from torch.nn import functional as functional
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Training requires PyTorch. Install with: python -m pip install -e '.[ml]'"
        ) from error

    from .model import MinesweeperNet

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if isinstance(dataset_path, (str, Path)):
        dataset_paths = [Path(dataset_path)]
    else:
        dataset_paths = [Path(path) for path in dataset_path]
    prepared_examples = [
        _prepare_example(example)
        for path in dataset_paths
        for example in read_jsonl(path)
    ]
    if not prepared_examples:
        raise ValueError("dataset is empty")
    prepared_examples.sort(key=lambda item: item[1].size)
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_name must be auto, cpu, or cuda")
    if batch_delay_ms < 0:
        raise ValueError("batch_delay_ms must not be negative")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    selected_device = (
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if selected_device == "auto":
        selected_device = "cpu"
    device = torch.device(selected_device)
    if gpu_memory_fraction is not None:
        if not 0 < gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must be in (0, 1]")
        if device.type != "cuda":
            raise ValueError("gpu_memory_fraction is only valid with CUDA")
        _set_gpu_memory_fraction(torch, device, gpu_memory_fraction)

    model = MinesweeperNet(model_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[float] = []

    for epoch_index in range(epochs):
        batches = [
            prepared_examples[start : start + batch_size]
            for start in range(0, len(prepared_examples), batch_size)
        ]
        random.shuffle(batches)
        epoch_loss = 0.0
        epoch_weight = 0.0
        model.train()
        for batch in batches:
            inputs, valid, targets, target_mask = _collate(batch, torch)
            inputs = inputs.to(device)
            valid = valid.to(device)
            targets = targets.to(device)
            target_mask = target_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(inputs, valid)
                element_loss = functional.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    reduction="none",
                )
                weight = target_mask.sum().clamp_min(1.0)
                loss = (element_loss * target_mask).sum() / weight
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach()) * float(weight)
            epoch_weight += float(weight)
            if batch_delay_ms:
                time.sleep(batch_delay_ms / 1_000)
        mean_loss = epoch_loss / max(epoch_weight, 1.0)
        history.append(mean_loss)
        if on_epoch is not None:
            on_epoch(epoch_index + 1, epochs, mean_loss)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_width": model_width,
            "feature_channels": FEATURE_CHANNELS,
            "history": history,
            "training": {
                "datasets": [str(path) for path in dataset_paths],
                "examples": len(prepared_examples),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": str(device),
                "amp": amp_enabled,
                "gpu_memory_fraction": gpu_memory_fraction,
                "batch_delay_ms": batch_delay_ms,
            },
        },
        output,
    )
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "loss": history,
                "device": str(device),
                "datasets": [str(path) for path in dataset_paths],
                "examples": len(prepared_examples),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "model_width": model_width,
                "seed": seed,
                "amp": amp_enabled,
                "gpu_memory_fraction": gpu_memory_fraction,
                "batch_delay_ms": batch_delay_ms,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return history


def _prepare_example(example):
    features, valid = encode_board(example.observation, example.total_mines)
    targets = np.asarray(
        [
            [0.0 if probability is None else probability for probability in row]
            for row in example.probabilities
        ],
        dtype=np.float32,
    )
    target_mask = np.asarray(example.target_mask, dtype=np.float32)
    return features, valid, targets, target_mask


def _collate(examples, torch_module):
    batch_size = len(examples)
    max_height = max(example[1].shape[0] for example in examples)
    max_width = max(example[1].shape[1] for example in examples)
    inputs = np.zeros(
        (batch_size, FEATURE_CHANNELS, max_height, max_width),
        dtype=np.float32,
    )
    valid = np.zeros((batch_size, max_height, max_width), dtype=np.float32)
    targets = np.zeros((batch_size, max_height, max_width), dtype=np.float32)
    target_mask = np.zeros((batch_size, max_height, max_width), dtype=np.float32)
    for index, (features, board_valid, board_targets, board_target_mask) in enumerate(examples):
        height, width = board_valid.shape
        inputs[index, :, :height, :width] = features
        valid[index, :height, :width] = board_valid
        targets[index, :height, :width] = board_targets
        target_mask[index, :height, :width] = board_target_mask
    return tuple(
        torch_module.from_numpy(array) for array in (inputs, valid, targets, target_mask)
    )


def _set_gpu_memory_fraction(torch_module, device, fraction: float) -> None:
    """Apply a CUDA allocator limit using an explicit integer device index."""
    device_index = device.index
    if device_index is None:
        device_index = torch_module.cuda.current_device()
    torch_module.cuda.set_per_process_memory_fraction(fraction, device=device_index)
