"""Seven-class Minesweeper scorecard with 3BV speed and click efficiency."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from statistics import median
import time
from typing import Sequence

from .agent import HybridAgent
from .data import BoardSpec, generate_no_guess_game
from .game import GameStatus, MinesweeperGame
from .metrics import calculate_3bv
from .replay_deployment import load_replay_deployment
from .solver import ConstraintSolver


STANDARD_PRESETS = {
    "beginner": BoardSpec(9, 9, 10, False),
    "intermediate": BoardSpec(16, 16, 40, False),
    "expert": BoardSpec(30, 16, 99, False),
}

NO_GUESS_PRESETS = {
    "beginner": BoardSpec(9, 9, 10, True),
    "intermediate": BoardSpec(16, 16, 40, True),
    "expert": BoardSpec(30, 16, 99, True),
    "evil": BoardSpec(20, 30, 130, True),
}


def benchmark_standard(
    agent: HybridAgent,
    spec: BoardSpec,
    *,
    games: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    records: list[dict[str, float | int | bool]] = []
    for _ in range(games):
        game = MinesweeperGame(
            spec.width,
            spec.height,
            spec.mines,
            seed=rng.randrange(2**63),
            first_click_safe=True,
            first_click_zero=False,
        )
        started = time.perf_counter()
        first = agent.choose_move(game.observation, game.mine_count, allow_guess=True)
        if first.coord is None:
            raise RuntimeError("agent produced no standard first click")
        opened = game.reveal(first.coord)
        result = agent.play(game, allow_guess=True)
        elapsed = time.perf_counter() - started
        complexity = calculate_3bv(game)
        records.append(
            {
                "won": result.won,
                "seconds": elapsed,
                "three_bv": complexity.value,
                "clicks": 1 + result.clicks,
                "reveals": 1 + result.reveals,
                "flags": result.flags,
                "chords": result.chords,
                "guesses": result.guesses,
                "opening_size": len(opened),
            }
        )
    return _summarize("standard", spec, records, "free_safe_only")


def benchmark_no_guess(
    agent: HybridAgent,
    spec: BoardSpec,
    *,
    games: int,
    seed: int,
    generator_solver: ConstraintSolver,
    max_attempts: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    records: list[dict[str, float | int | bool]] = []
    generation_seconds = 0.0
    prescribed_start = (spec.height // 2, spec.width // 2)
    for _ in range(games):
        generation_started = time.perf_counter()
        game = generate_no_guess_game(
            spec,
            seed=rng.randrange(2**63),
            solver=generator_solver,
            max_attempts=max_attempts,
        )
        generation_seconds += time.perf_counter() - generation_started

        started = time.perf_counter()
        opened = game.reveal(prescribed_start)
        result = agent.play(game, allow_guess=False)
        elapsed = time.perf_counter() - started
        complexity = calculate_3bv(game)
        records.append(
            {
                "won": result.won,
                "seconds": elapsed,
                "three_bv": complexity.value,
                "clicks": 1 + result.clicks,
                "reveals": 1 + result.reveals,
                "flags": result.flags,
                "chords": result.chords,
                "guesses": result.guesses,
                "opening_size": len(opened),
            }
        )
    summary = _summarize("no_guess", spec, records, "prescribed_zero")
    summary["generation_seconds"] = round(generation_seconds, 6)
    summary["prescribed_start"] = list(prescribed_start)
    return summary


def _summarize(
    track: str,
    spec: BoardSpec,
    records: Sequence[dict[str, float | int | bool]],
    first_click_rule: str,
) -> dict[str, object]:
    wins = [record for record in records if record["won"]]
    win_seconds = [float(record["seconds"]) for record in wins]
    win_3bv = sum(int(record["three_bv"]) for record in wins)
    win_clicks = sum(int(record["clicks"]) for record in wins)
    return {
        "track": track,
        "board": f"{spec.width}x{spec.height}/{spec.mines}",
        "first_click_rule": first_click_rule,
        "games": len(records),
        "wins": len(wins),
        "win_rate": len(wins) / len(records),
        "guesses": sum(int(record["guesses"]) for record in records),
        "three_bv_won": win_3bv,
        "solve_seconds": round(sum(win_seconds), 6),
        "three_bv_per_second": (
            win_3bv / sum(win_seconds) if win_seconds and sum(win_seconds) > 0.0 else 0.0
        ),
        "clicks_won": win_clicks,
        "ioe": win_3bv / win_clicks if win_clicks else 0.0,
        "reveals": sum(int(record["reveals"]) for record in records),
        "flags": sum(int(record["flags"]) for record in records),
        "chords": sum(int(record["chords"]) for record in records),
        "opening_size_mean": sum(int(record["opening_size"]) for record in records)
        / len(records),
        "opening_size_min": min(int(record["opening_size"]) for record in records),
        "solve_ms_median": _percentile(win_seconds, 0.5) * 1000.0,
        "solve_ms_p95": _percentile(win_seconds, 0.95) * 1000.0,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    if quantile == 0.5:
        return median(values)
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=("standard", "no_guess"),
        default=("standard", "no_guess"),
    )
    parser.add_argument("--seed", type=int, default=20260942)
    parser.add_argument("--solver-cells", type=int, default=24)
    parser.add_argument("--generator-solver-cells", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.games < 1:
        raise SystemExit("--games must be positive")
    if args.deployment:
        agent, _ = load_replay_deployment(args.deployment, device=args.device)
    else:
        agent = HybridAgent(ConstraintSolver(args.solver_cells))
    generator_solver = ConstraintSolver(args.generator_solver_cells)
    seed_rng = random.Random(args.seed)

    if "standard" in args.tracks:
        for name, spec in STANDARD_PRESETS.items():
            summary = benchmark_standard(
                agent,
                spec,
                games=args.games,
                seed=seed_rng.randrange(2**63),
            )
            summary["mode"] = name
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    if "no_guess" in args.tracks:
        for name, spec in NO_GUESS_PRESETS.items():
            summary = benchmark_no_guess(
                agent,
                spec,
                games=args.games,
                seed=seed_rng.randrange(2**63),
                generator_solver=generator_solver,
                max_attempts=args.max_attempts,
            )
            summary["mode"] = name
            print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
