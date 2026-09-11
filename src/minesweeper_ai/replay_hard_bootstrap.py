"""Generate standard expert Replay traces with a larger teacher budget."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from .data import BoardSpec
from .replay import generate_simulator_replays, write_replays
from .solver import ConstraintSolver


@dataclass(frozen=True)
class HardBootstrapStats:
    replays: int
    wins: int
    losses: int
    moves: int
    teacher_cells: int
    width: int
    height: int
    mines: int


def generate_hard_replays(
    output: str | Path,
    *,
    games: int,
    seed: int,
    teacher_cells: int,
    spec: BoardSpec = BoardSpec(30, 16, 99, True),
) -> HardBootstrapStats:
    replays = list(
        generate_simulator_replays(
            [spec],
            games,
            mode="standard",
            seed=seed,
            solver=ConstraintSolver(teacher_cells),
        )
    )
    write_replays(replays, output)
    wins = sum(replay.won for replay in replays)
    return HardBootstrapStats(
        replays=len(replays),
        wins=wins,
        losses=len(replays) - wins,
        moves=sum(len(replay.moves) for replay in replays),
        teacher_cells=teacher_cells,
        width=spec.width,
        height=spec.height,
        mines=spec.mines,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--teacher-cells", type=int, default=28)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stats = generate_hard_replays(
        args.output,
        games=args.games,
        seed=args.seed,
        teacher_cells=args.teacher_cells,
    )
    print(json.dumps(asdict(stats), separators=(",", ":")))


if __name__ == "__main__":
    main()
