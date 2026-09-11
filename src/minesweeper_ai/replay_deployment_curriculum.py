"""Build exact supervision for the states where the deployed Risk head runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from .game import GameStatus, MinesweeperGame
from .replay import (
    ReplayTrainingExample,
    _apply_move,
    _risk_targets,
    read_replays,
    write_replay_examples,
)
from .solver import ConstraintSolver


@dataclass(frozen=True)
class DeploymentCurriculumStats:
    replays: int
    states: int
    deployment_gaps: int
    teacher_exact_gaps: int
    selected: int
    risk_cells: int


def build_deployment_curriculum(
    replay_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    student_cells: int = 24,
    teacher_cells: int = 30,
) -> DeploymentCurriculumStats:
    if teacher_cells <= student_cells:
        raise ValueError("teacher_cells must be larger than student_cells")
    student = ConstraintSolver(student_cells)
    teacher = ConstraintSolver(teacher_cells)
    examples: list[ReplayTrainingExample] = []
    counters = {
        "replays": 0,
        "states": 0,
        "deployment_gaps": 0,
        "teacher_exact_gaps": 0,
        "risk_cells": 0,
    }
    for path in replay_paths:
        for replay in read_replays(path):
            counters["replays"] += 1
            game = MinesweeperGame(
                replay.width,
                replay.height,
                replay.mines,
                mine_positions={tuple(coord) for coord in replay.mine_positions},
            )
            safe_cells = replay.width * replay.height - replay.mines
            for move in replay.moves:
                if game.status in (GameStatus.WON, GameStatus.LOST):
                    break
                counters["states"] += 1
                observation = game.observation
                student_analysis = student.analyse(observation, replay.mines)
                deployment_gap = (
                    not student_analysis.exact
                    and not student_analysis.safe
                    and not student_analysis.contradiction
                )
                if deployment_gap:
                    counters["deployment_gaps"] += 1
                    teacher_analysis = teacher.analyse(observation, replay.mines)
                    if teacher_analysis.exact and not teacher_analysis.contradiction:
                        probabilities, risk_mask = _risk_targets(observation, teacher_analysis)
                        risk_cells = sum(sum(row) for row in risk_mask)
                        if risk_cells:
                            counters["teacher_exact_gaps"] += 1
                            counters["risk_cells"] += risk_cells
                            revealed = sum(cell >= 0 for row in observation for cell in row)
                            examples.append(
                                ReplayTrainingExample(
                                    game_id=replay.game_id,
                                    observation=[list(row) for row in observation],
                                    total_mines=replay.mines,
                                    mode=replay.mode,
                                    source=replay.source,
                                    action=move.action,
                                    row=move.row,
                                    col=move.col,
                                    time_ms=move.time_ms,
                                    classification="NECESSARY_GUESS",
                                    teacher_exact=True,
                                    mine_probability=teacher_analysis.probabilities.get(move.coord),
                                    probabilities=probabilities,
                                    risk_mask=risk_mask,
                                    policy_weight=0.0,
                                    value_target=1.0 if replay.won else 0.0,
                                    value_weight=0.0,
                                )
                            )
                _apply_move(game, move)
    write_replay_examples(examples, output_path)
    return DeploymentCurriculumStats(selected=len(examples), **counters)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-cells", type=int, default=24)
    parser.add_argument("--teacher-cells", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stats = build_deployment_curriculum(
        args.input,
        args.output,
        student_cells=args.student_cells,
        teacher_cells=args.teacher_cells,
    )
    print(json.dumps(asdict(stats), separators=(",", ":")))


if __name__ == "__main__":
    main()
