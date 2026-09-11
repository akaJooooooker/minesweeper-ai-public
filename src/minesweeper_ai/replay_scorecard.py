"""Evaluate the agent on independent real no-guess replay boards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

from .agent import HybridAgent
from .data import BoardSpec
from .game import MinesweeperGame
from .replay import ReplayGame, read_replays
from .replay_deployment import load_replay_deployment
from .scorecard import _summarize
from .solver import ConstraintSolver
from .metrics import calculate_3bv


NO_GUESS_REPLAY_PRESETS = {
    "beginner": BoardSpec(9, 9, 10, True),
    "intermediate": BoardSpec(16, 16, 40, True),
    "expert": BoardSpec(30, 16, 99, True),
    "evil": BoardSpec(20, 30, 130, True),
}


def benchmark_no_guess_replays(
    agent: HybridAgent,
    replays: Iterable[ReplayGame],
    spec: BoardSpec,
    *,
    limit: int = 0,
) -> dict[str, object]:
    records: list[dict[str, float | int | bool]] = []
    human_three_bv = 0
    human_clicks = 0
    human_seconds = 0.0
    invalid_openings = 0

    for replay in replays:
        if replay.mode != "no_guess" or not _matches_spec(replay, spec):
            continue
        if not replay.moves or replay.moves[0].action != "reveal":
            invalid_openings += 1
            continue
        mines = {tuple(coord) for coord in replay.mine_positions}
        game = MinesweeperGame(
            replay.width,
            replay.height,
            replay.mines,
            mine_positions=mines,
        )
        first = replay.moves[0].coord
        started = time.perf_counter()
        opened = game.reveal(first)
        result = agent.play(game, allow_guess=False)
        elapsed = time.perf_counter() - started
        complexity = calculate_3bv(game)
        if len(opened) <= 1:
            invalid_openings += 1
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
        human_three_bv += complexity.value
        human_clicks += len(replay.moves)
        if replay.moves:
            human_seconds += max(0.0, replay.moves[-1].time_ms / 1000.0)
        if limit and len(records) >= limit:
            break

    if not records:
        raise ValueError("no matching no-guess replays")
    summary = _summarize("no_guess", spec, records, "replay_prescribed_zero")
    summary.update(
        {
            "source": "independent_real_replays",
            "invalid_openings": invalid_openings,
            "human_clicks": human_clicks,
            "human_ioe": human_three_bv / human_clicks if human_clicks else 0.0,
            "human_seconds": human_seconds,
            "human_three_bv_per_second": (
                human_three_bv / human_seconds if human_seconds > 0.0 else 0.0
            ),
        }
    )
    return summary


def _matches_spec(replay: ReplayGame, spec: BoardSpec) -> bool:
    return (
        replay.mines == spec.mines
        and sorted((replay.width, replay.height)) == sorted((spec.width, spec.height))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--limit-per-mode", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(NO_GUESS_REPLAY_PRESETS),
        default=tuple(NO_GUESS_REPLAY_PRESETS),
    )
    parser.add_argument("--solver-cells", type=int, default=24)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.limit_per_mode < 0:
        raise SystemExit("--limit-per-mode must be non-negative")
    if args.deployment:
        agent, _ = load_replay_deployment(args.deployment, device=args.device)
    else:
        agent = HybridAgent(ConstraintSolver(args.solver_cells))
    replays = list(read_replays(args.replays))
    for mode in args.modes:
        spec = NO_GUESS_REPLAY_PRESETS[mode]
        summary = benchmark_no_guess_replays(
            agent,
            replays,
            spec,
            limit=args.limit_per_mode,
        )
        summary["mode"] = mode
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
