"""Mine exact-teacher / bounded-student gaps from canonical Replay traces."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Sequence

from .game import FLAGGED, UNKNOWN
from .replay import (
    ReplayAuditor,
    ReplayTrainingExample,
    read_replays,
    write_replay_examples,
)
from .solver import ConstraintSolver


@dataclass(frozen=True)
class CurriculumStats:
    replays: int
    states: int
    student_nonexact_guess_states: int
    teacher_exact_states: int
    selected: int
    risk_cells: int
    standard: int
    no_guess: int


def build_risk_curriculum(
    replay_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    student_cells: int = 24,
    teacher_cells: int = 28,
) -> CurriculumStats:
    """Keep states where a larger exact teacher can supervise runtime gaps.

    Selected examples have Policy and Value weights set to zero so repeated use
    of this curriculum changes only the Risk objective.
    """
    if teacher_cells <= student_cells:
        raise ValueError("teacher_cells must be larger than student_cells")
    student = ConstraintSolver(student_cells)
    teacher_auditor = ReplayAuditor(ConstraintSolver(teacher_cells))
    selected: list[ReplayTrainingExample] = []
    counters = {
        "replays": 0,
        "states": 0,
        "student_nonexact_guess_states": 0,
        "teacher_exact_states": 0,
        "risk_cells": 0,
        "standard": 0,
        "no_guess": 0,
    }
    for path in replay_paths:
        for replay in read_replays(path):
            counters["replays"] += 1
            for example in teacher_auditor.audit(replay):
                counters["states"] += 1
                if example.teacher_exact and any(any(row) for row in example.risk_mask):
                    counters["teacher_exact_states"] += 1
                audit_observation = tuple(
                    tuple(UNKNOWN if cell == FLAGGED else cell for cell in row)
                    for row in example.observation
                )
                student_analysis = student.analyse(audit_observation, example.total_mines)
                student_gap = (
                    not student_analysis.exact
                    and not student_analysis.safe
                    and not student_analysis.contradiction
                )
                if not student_gap:
                    continue
                counters["student_nonexact_guess_states"] += 1
                risk_cells = sum(sum(row) for row in example.risk_mask)
                if not example.teacher_exact or risk_cells == 0:
                    continue
                selected.append(replace(example, policy_weight=0.0, value_weight=0.0))
                counters["risk_cells"] += risk_cells
                counters[example.mode] += 1
    write_replay_examples(selected, output_path)
    return CurriculumStats(selected=len(selected), **counters)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-cells", type=int, default=24)
    parser.add_argument("--teacher-cells", type=int, default=28)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stats = build_risk_curriculum(
        args.input,
        args.output,
        student_cells=args.student_cells,
        teacher_cells=args.teacher_cells,
    )
    print(json.dumps(asdict(stats), separators=(",", ":")))


if __name__ == "__main__":
    main()
