"""Runtime adapter and paired gameplay benchmark for Replay checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

from .agent import HybridAgent
from .data import BoardSpec
from .game import UNKNOWN, Coord, MinesweeperGame
from .replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model, encode_replay_board
from .solver import ConstraintSolver


@dataclass(frozen=True)
class ReplayPrediction:
    risk: Mapping[Coord, float]
    policy_logits: Mapping[tuple[str, Coord], float]
    value: float


class ReplayAgentPredictor:
    """Load all Replay heads and expose board-size-independent predictions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        mode: str = "standard",
        source: str = "simulator_teacher",
        device: str | None = None,
        gpu_memory_fraction: float | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        if mode not in {"standard", "no_guess"}:
            raise ValueError("mode must be standard or no_guess")
        torch, ReplayAgentNet = _torch_model()
        if cpu_threads is not None:
            if cpu_threads < 1:
                raise ValueError("cpu_threads must be positive")
            torch.set_num_threads(cpu_threads)
            requested_interop = min(cpu_threads, 2)
            try:
                torch.set_num_interop_threads(requested_interop)
            except RuntimeError:
                if torch.get_num_interop_threads() != requested_interop:
                    raise
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if gpu_memory_fraction is not None:
            if self.device.type != "cuda" or not 0 < gpu_memory_fraction <= 1:
                raise ValueError("GPU memory fraction requires CUDA and must be in (0, 1]")
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=index)
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=True)
        if checkpoint.get("feature_channels") != REPLAY_FEATURE_CHANNELS:
            raise ValueError("checkpoint feature layout is incompatible")
        if checkpoint.get("action_order") != ["reveal", "flag", "chord"]:
            raise ValueError("checkpoint action layout is incompatible")
        self.model = ReplayAgentNet(int(checkpoint["model_width"])).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self._torch = torch
        self.mode = mode
        self.source = source

    def predict_all(
        self,
        observation: Sequence[Sequence[int]],
        total_mines: int,
    ) -> ReplayPrediction:
        features, valid = encode_replay_board(
            observation,
            total_mines,
            mode=self.mode,
            source=self.source,
        )
        torch = self._torch
        inputs = torch.from_numpy(features).unsqueeze(0).to(self.device)
        valid_mask = torch.from_numpy(valid).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(inputs, valid_mask)
            risk_tensor = torch.sigmoid(output["risk"])[0].cpu()
            policy_key = "no_guess_policy" if self.mode == "no_guess" else "standard_policy"
            policy_tensor = output[policy_key][0].cpu()
            value = float(torch.sigmoid(output["value"])[0].cpu())
        actions = ("reveal", "flag", "chord")
        height, width = valid.shape
        risk = {
            (row, col): float(risk_tensor[row, col])
            for row in range(height)
            for col in range(width)
            if observation[row][col] == UNKNOWN
        }
        policy = {
            (action, (row, col)): float(policy_tensor[action_index, row, col])
            for action_index, action in enumerate(actions)
            for row in range(height)
            for col in range(width)
        }
        return ReplayPrediction(risk=risk, policy_logits=policy, value=value)

    def predict(
        self,
        observation: Sequence[Sequence[int]],
        total_mines: int,
    ) -> Mapping[Coord, float]:
        """RiskPredictor protocol adapter used by HybridAgent."""
        return self.predict_all(observation, total_mines).risk


@dataclass(frozen=True)
class BenchmarkRow:
    width: int
    height: int
    mines: int
    games: int
    baseline_wins: int
    replay_wins: int
    replay_only_wins: int
    baseline_only_wins: int
    baseline_guesses: int
    replay_guesses: int
    baseline_stuck: int
    replay_stuck: int


def paired_benchmark(
    checkpoint: str | Path,
    *,
    games_per_spec: int = 20,
    seed: int = 0,
    device: str = "cpu",
    solver_cells: int = 24,
    gpu_memory_fraction: float | None = None,
    specs: Sequence[BoardSpec] = (
        BoardSpec(9, 9, 10, True),
        BoardSpec(16, 16, 40, True),
        BoardSpec(30, 16, 99, True),
    ),
) -> list[BenchmarkRow]:
    if games_per_spec < 1:
        raise ValueError("games_per_spec must be positive")
    predictor = ReplayAgentPredictor(
        checkpoint,
        mode="standard",
        device=device,
        gpu_memory_fraction=gpu_memory_fraction,
    )
    baseline = HybridAgent(ConstraintSolver(solver_cells))
    replay = HybridAgent(ConstraintSolver(solver_cells), risk_predictor=predictor)
    rng = random.Random(seed)
    rows: list[BenchmarkRow] = []
    for spec in specs:
        counters = {
            "baseline_wins": 0,
            "replay_wins": 0,
            "replay_only_wins": 0,
            "baseline_only_wins": 0,
            "baseline_guesses": 0,
            "replay_guesses": 0,
            "baseline_stuck": 0,
            "replay_stuck": 0,
        }
        for _ in range(games_per_spec):
            game_seed = rng.randrange(2**63)
            baseline_game = MinesweeperGame(
                spec.width,
                spec.height,
                spec.mines,
                seed=game_seed,
                first_click_zero=spec.first_click_zero,
            )
            replay_game = MinesweeperGame(
                spec.width,
                spec.height,
                spec.mines,
                seed=game_seed,
                first_click_zero=spec.first_click_zero,
            )
            baseline_result = baseline.play(baseline_game, allow_guess=True)
            replay_result = replay.play(replay_game, allow_guess=True)
            counters["baseline_wins"] += baseline_result.won
            counters["replay_wins"] += replay_result.won
            counters["replay_only_wins"] += replay_result.won and not baseline_result.won
            counters["baseline_only_wins"] += baseline_result.won and not replay_result.won
            counters["baseline_guesses"] += baseline_result.guesses
            counters["replay_guesses"] += replay_result.guesses
            counters["baseline_stuck"] += baseline_result.stuck
            counters["replay_stuck"] += replay_result.stuck
        rows.append(
            BenchmarkRow(
                width=spec.width,
                height=spec.height,
                mines=spec.mines,
                games=games_per_spec,
                **counters,
            )
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games-per-spec", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--solver-cells", type=int, default=24)
    parser.add_argument("--gpu-memory-fraction", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = paired_benchmark(
        args.checkpoint,
        games_per_spec=args.games_per_spec,
        seed=args.seed,
        device=args.device,
        solver_cells=args.solver_cells,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )
    print(json.dumps([asdict(row) for row in rows], separators=(",", ":")))


if __name__ == "__main__":
    main()
