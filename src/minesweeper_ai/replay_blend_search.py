"""Search solver/neural risk blending on paired unseen Minesweeper boards."""

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
class BlendResult:
    alpha: float
    games: int
    wins: int
    baseline_wins: int
    model_only_wins: int
    baseline_only_wins: int
    guesses: int


def search_blends(
    checkpoint: str | Path,
    *,
    alphas: Sequence[float],
    games_per_spec: int,
    seed: int,
    device: str = "cpu",
    solver_cells: int = 24,
    blend_mode: str = "probability",
    specs: Sequence[BoardSpec] = (
        BoardSpec(16, 16, 40, True),
        BoardSpec(30, 16, 99, True),
    ),
) -> list[BlendResult]:
    if games_per_spec < 1:
        raise ValueError("games_per_spec must be positive")
    if not alphas or any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("alphas must be non-empty values in [0, 1]")
    predictor = ReplayAgentPredictor(
        checkpoint, mode="standard", device=device, cpu_threads=2
    )
    baseline = HybridAgent(ConstraintSolver(solver_cells))
    agents = {
        alpha: HybridAgent(
            ConstraintSolver(solver_cells),
            risk_predictor=predictor,
            neural_blend=alpha,
            neural_blend_mode=blend_mode,
        )
        for alpha in alphas
    }
    counters = {
        alpha: {
            "wins": 0,
            "baseline_wins": 0,
            "model_only_wins": 0,
            "baseline_only_wins": 0,
            "guesses": 0,
        }
        for alpha in alphas
    }
    rng = random.Random(seed)
    for spec in specs:
        for _ in range(games_per_spec):
            game_seed = rng.randrange(2**63)
            baseline_result = baseline.play(_game(spec, game_seed), allow_guess=True)
            for alpha, agent in agents.items():
                result = agent.play(_game(spec, game_seed), allow_guess=True)
                counters[alpha]["wins"] += result.won
                counters[alpha]["baseline_wins"] += baseline_result.won
                counters[alpha]["model_only_wins"] += result.won and not baseline_result.won
                counters[alpha]["baseline_only_wins"] += baseline_result.won and not result.won
                counters[alpha]["guesses"] += result.guesses
    total_games = games_per_spec * len(specs)
    return [BlendResult(alpha=alpha, games=total_games, **counters[alpha]) for alpha in alphas]


def _game(spec: BoardSpec, seed: int) -> MinesweeperGame:
    return MinesweeperGame(
        spec.width,
        spec.height,
        spec.mines,
        seed=seed,
        first_click_zero=spec.first_click_zero,
    )


def _alpha_list(text: str) -> list[float]:
    try:
        values = [float(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from error
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("alphas must be in [0, 1]")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alphas", type=_alpha_list, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--games-per-spec", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--solver-cells", type=int, default=24)
    parser.add_argument("--blend-mode", choices=("probability", "rank"), default="probability")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results = search_blends(
        args.checkpoint,
        alphas=args.alphas,
        games_per_spec=args.games_per_spec,
        seed=args.seed,
        device=args.device,
        solver_cells=args.solver_cells,
        blend_mode=args.blend_mode,
    )
    print(json.dumps([asdict(result) for result in results], separators=(",", ":")))


if __name__ == "__main__":
    main()
