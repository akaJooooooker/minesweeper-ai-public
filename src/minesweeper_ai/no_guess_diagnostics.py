"""Inspect states where the proof solver gets stuck on real no-guess boards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .agent import HybridAgent
from .game import UNKNOWN, MinesweeperGame
from .replay import read_replays
from .replay_scorecard import NO_GUESS_REPLAY_PRESETS, _matches_spec
from .solver import ConstraintSolver


def diagnose(
    replay_path: str | Path,
    *,
    mode: str,
    solver_cells: int,
    probe_cells: Sequence[int],
) -> list[dict[str, object]]:
    spec = NO_GUESS_REPLAY_PRESETS[mode]
    agent = HybridAgent(ConstraintSolver(solver_cells))
    probes = [(cells, ConstraintSolver(cells)) for cells in probe_cells]
    failures: list[dict[str, object]] = []
    for replay in read_replays(replay_path):
        if replay.mode != "no_guess" or not _matches_spec(replay, spec):
            continue
        game = MinesweeperGame(
            replay.width,
            replay.height,
            replay.mines,
            mine_positions={tuple(coord) for coord in replay.mine_positions},
        )
        game.reveal(replay.moves[0].coord)
        result = agent.play(game, allow_guess=False)
        if result.won:
            continue
        observation = game.observation
        probe_results = {}
        for cells, solver in probes:
            analysis = solver.analyse(observation, game.mine_count)
            probe_results[str(cells)] = {
                "exact": analysis.exact,
                "safe": len(analysis.safe),
                "mines": len(analysis.mines),
                "frontier_size": analysis.frontier_size,
                "model_count": analysis.model_count,
                "contradiction": analysis.contradiction,
            }
        failures.append(
            {
                "game_id": replay.game_id,
                "unknown": sum(cell == UNKNOWN for row in observation for cell in row),
                "reveals": result.reveals,
                "flags": result.flags,
                "chords": result.chords,
                "probes": probe_results,
            }
        )
    return failures


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(NO_GUESS_REPLAY_PRESETS), required=True)
    parser.add_argument("--solver-cells", type=int, default=34)
    parser.add_argument("--probe-cells", type=int, nargs="+", default=(36, 40))
    args = parser.parse_args(argv)
    failures = diagnose(
        args.replays,
        mode=args.mode,
        solver_cells=args.solver_cells,
        probe_cells=args.probe_cells,
    )
    print(json.dumps({"failures": len(failures), "cases": failures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
