"""Compare two Replay checkpoints on exactly the same Minesweeper boards."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Sequence

from .agent import HybridAgent
from .data import BoardSpec
from .game import MinesweeperGame
from .replay_inference import ReplayAgentPredictor
from .solver import ConstraintSolver


@dataclass(frozen=True)
class CheckpointComparison:
    games: int
    candidate_wins: int
    reference_wins: int
    candidate_only_wins: int
    reference_only_wins: int
    candidate_guesses: int
    reference_guesses: int


def compare_checkpoints(
    candidate_checkpoint: str | Path,
    reference_checkpoint: str | Path,
    *,
    games_per_spec: int,
    seed: int,
    alpha: float = 0.75,
    reference_alpha: float | None = None,
    policy_blend: float = 0.0,
    reference_policy_blend: float = 0.0,
    policy_top_k: int = 0,
    reference_policy_top_k: int = 0,
    policy_min_advantage: float = 0.0,
    reference_policy_min_advantage: float = 0.0,
    policy_min_board_cells: int = 0,
    reference_policy_min_board_cells: int = 0,
    solver_cells: int = 24,
    device: str = "cpu",
    specs: Sequence[BoardSpec] = (
        BoardSpec(16, 16, 40, True),
        BoardSpec(30, 16, 99, True),
    ),
) -> CheckpointComparison:
    if games_per_spec < 1:
        raise ValueError("games_per_spec must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if reference_alpha is None:
        reference_alpha = alpha
    if not 0.0 <= reference_alpha <= 1.0:
        raise ValueError("reference_alpha must be in [0, 1]")
    candidate = HybridAgent(
        ConstraintSolver(solver_cells),
        risk_predictor=ReplayAgentPredictor(
            candidate_checkpoint,
            mode="standard",
            device=device,
            cpu_threads=2,
        ),
        neural_blend=alpha,
        policy_blend=policy_blend,
        policy_top_k=policy_top_k,
        policy_min_advantage=policy_min_advantage,
        policy_min_board_cells=policy_min_board_cells,
    )
    reference = HybridAgent(
        ConstraintSolver(solver_cells),
        risk_predictor=ReplayAgentPredictor(
            reference_checkpoint,
            mode="standard",
            device=device,
            cpu_threads=2,
        ),
        neural_blend=reference_alpha,
        policy_blend=reference_policy_blend,
        policy_top_k=reference_policy_top_k,
        policy_min_advantage=reference_policy_min_advantage,
        policy_min_board_cells=reference_policy_min_board_cells,
    )
    counters = {
        "candidate_wins": 0,
        "reference_wins": 0,
        "candidate_only_wins": 0,
        "reference_only_wins": 0,
        "candidate_guesses": 0,
        "reference_guesses": 0,
    }
    rng = random.Random(seed)
    for spec in specs:
        for _ in range(games_per_spec):
            game_seed = rng.randrange(2**63)
            candidate_result = candidate.play(_game(spec, game_seed), allow_guess=True)
            reference_result = reference.play(_game(spec, game_seed), allow_guess=True)
            counters["candidate_wins"] += candidate_result.won
            counters["reference_wins"] += reference_result.won
            counters["candidate_only_wins"] += candidate_result.won and not reference_result.won
            counters["reference_only_wins"] += reference_result.won and not candidate_result.won
            counters["candidate_guesses"] += candidate_result.guesses
            counters["reference_guesses"] += reference_result.guesses
    return CheckpointComparison(games=games_per_spec * len(specs), **counters)


def _game(spec: BoardSpec, seed: int) -> MinesweeperGame:
    return MinesweeperGame(
        spec.width,
        spec.height,
        spec.mines,
        seed=seed,
        first_click_zero=spec.first_click_zero,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--games-per-spec", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--solver-cells", type=int, default=24)
    parser.add_argument("--reference-alpha", type=float)
    parser.add_argument("--policy-blend", type=float, default=0.0)
    parser.add_argument("--reference-policy-blend", type=float, default=0.0)
    parser.add_argument("--policy-top-k", type=int, default=0)
    parser.add_argument("--reference-policy-top-k", type=int, default=0)
    parser.add_argument("--policy-min-advantage", type=float, default=0.0)
    parser.add_argument("--reference-policy-min-advantage", type=float, default=0.0)
    parser.add_argument("--policy-min-board-cells", type=int, default=0)
    parser.add_argument("--reference-policy-min-board-cells", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = compare_checkpoints(
        args.candidate,
        args.reference,
        games_per_spec=args.games_per_spec,
        seed=args.seed,
        alpha=args.alpha,
        solver_cells=args.solver_cells,
        reference_alpha=args.reference_alpha,
        policy_blend=args.policy_blend,
        reference_policy_blend=args.reference_policy_blend,
        policy_top_k=args.policy_top_k,
        reference_policy_top_k=args.reference_policy_top_k,
        policy_min_advantage=args.policy_min_advantage,
        reference_policy_min_advantage=args.reference_policy_min_advantage,
        policy_min_board_cells=args.policy_min_board_cells,
        reference_policy_min_board_cells=args.reference_policy_min_board_cells,
        device=args.device,
    )
    print(json.dumps(asdict(result), separators=(",", ":")))


if __name__ == "__main__":
    main()
