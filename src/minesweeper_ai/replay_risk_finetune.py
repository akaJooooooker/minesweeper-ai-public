"""Fine-tune only the Replay Risk head on exact teacher/student gap states."""

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
class RiskFinetuneEpoch:
    epoch: int
    train_bce: float
    validation_brier: float
    best: bool


def fine_tune_risk_head(
    checkpoint_path: str | Path,
    dataset_paths: Sequence[str | Path],
    validation_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 0,
    device_name: str = "auto",
    use_amp: bool = True,
    gpu_memory_fraction: float | None = None,
    batch_delay_ms: int = 0,
    patience: int = 8,
    on_epoch=None,
) -> list[RiskFinetuneEpoch]:
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
    if not train_examples:
        raise ValueError("risk fine-tune dataset is empty")
    if not validation_examples:
        raise ValueError("risk validation dataset is empty")
    optimizer = torch.optim.AdamW(model.risk_head.parameters(), lr=learning_rate, weight_decay=1e-4)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_state = deepcopy(model.state_dict())
    best_brier = _risk_brier(model, validation_examples, batch_size, torch, device)
    best_epoch = 0
    stale = 0
    history: list[RiskFinetuneEpoch] = []

    for epoch in range(1, epochs + 1):
        batches = [train_examples[i : i + batch_size] for i in range(0, len(train_examples), batch_size)]
        random.shuffle(batches)
        loss_sum = 0.0
        cell_count = 0.0
        model.train()
        for batch in batches:
            tensors = {key: value.to(device) for key, value in _collate(batch, torch).items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(tensors["inputs"], tensors["valid"])
                elements = functional.binary_cross_entropy_with_logits(
                    output["risk"], tensors["risk_targets"], reduction="none"
                )
                cells = tensors["risk_mask"].sum().clamp_min(1.0)
                loss = (elements * tensors["risk_mask"]).sum() / cells
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.risk_head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * float(cells)
            cell_count += float(cells)
            if batch_delay_ms:
                time.sleep(batch_delay_ms / 1_000)
        validation_brier = _risk_brier(model, validation_examples, batch_size, torch, device)
        improved = validation_brier < best_brier - 1e-8
        if improved:
            best_brier = validation_brier
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        metrics = RiskFinetuneEpoch(
            epoch=epoch,
            train_bce=loss_sum / max(cell_count, 1.0),
            validation_brier=validation_brier,
            best=improved,
        )
        history.append(metrics)
        if on_epoch:
            on_epoch(metrics)
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
            "kind": "replay-agent-risk-finetune-v3",
            "parent_checkpoint": str(Path(checkpoint_path)),
            "risk_finetune": {
                "datasets": [str(Path(path)) for path in dataset_paths],
                "validation": str(Path(validation_path)),
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "epochs_requested": epochs,
                "epochs_run": len(history),
                "best_epoch": best_epoch,
                "best_validation_brier": best_brier,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
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
        json.dumps({**metadata, "history": [asdict(item) for item in history]}, indent=2),
        encoding="utf-8",
    )
    return history


def _load_risk_examples(paths: Sequence[str | Path]):
    return [
        _prepare_example(example)
        for path in paths
        for example in read_replay_examples(path)
        if any(any(row) for row in example.risk_mask)
    ]


def _risk_brier(model, examples, batch_size, torch, device) -> float:
    squared_sum = 0.0
    cell_count = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            tensors = {
                key: value.to(device)
                for key, value in _collate(examples[start : start + batch_size], torch).items()
            }
            probabilities = torch.sigmoid(model(tensors["inputs"], tensors["valid"])["risk"])
            squared = (probabilities - tensors["risk_targets"]).square() * tensors["risk_mask"]
            squared_sum += float(squared.sum())
            cell_count += float(tensors["risk_mask"].sum())
    return squared_sum / max(cell_count, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--batch-delay-ms", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    history = fine_tune_risk_head(
        args.checkpoint,
        args.dataset,
        args.validation,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        use_amp=not args.no_amp,
        gpu_memory_fraction=args.gpu_memory_fraction,
        batch_delay_ms=args.batch_delay_ms,
        patience=args.patience,
        on_epoch=lambda item: print(json.dumps({"event": "epoch", **asdict(item)}), flush=True),
    )
    print(json.dumps({"output": str(args.output.resolve()), "epochs": len(history)}))


if __name__ == "__main__":
    main()
