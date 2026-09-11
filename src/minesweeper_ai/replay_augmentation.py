"""Geometry-preserving augmentation for audited Replay training examples."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence, TypeVar

from .replay import ReplayTrainingExample, read_replay_examples, write_replay_examples

T = TypeVar("T")


def transform_coord(
    row: int,
    col: int,
    height: int,
    width: int,
    *,
    rotations: int,
    mirror: bool,
) -> tuple[int, int, int, int]:
    """Apply a horizontal mirror followed by clockwise quarter-turns."""
    rotations %= 4
    if mirror:
        col = width - 1 - col
    for _ in range(rotations):
        row, col = col, height - 1 - row
        height, width = width, height
    return row, col, height, width


def transform_matrix(
    matrix: Sequence[Sequence[T]], *, rotations: int, mirror: bool
) -> list[list[T]]:
    height = len(matrix)
    width = len(matrix[0])
    _, _, output_height, output_width = transform_coord(
        0, 0, height, width, rotations=rotations, mirror=mirror
    )
    output: list[list[T | None]] = [
        [None for _ in range(output_width)] for _ in range(output_height)
    ]
    for row in range(height):
        for col in range(width):
            next_row, next_col, _, _ = transform_coord(
                row, col, height, width, rotations=rotations, mirror=mirror
            )
            output[next_row][next_col] = matrix[row][col]
    return [[value for value in row] for row in output]  # type: ignore[misc]


def augment_example(example: ReplayTrainingExample) -> list[ReplayTrainingExample]:
    height = len(example.observation)
    width = len(example.observation[0])
    augmented: list[ReplayTrainingExample] = []
    for mirror in (False, True):
        for rotations in range(4):
            row, col, _, _ = transform_coord(
                example.row,
                example.col,
                height,
                width,
                rotations=rotations,
                mirror=mirror,
            )
            augmented.append(
                replace(
                    example,
                    game_id=f"{example.game_id}-aug-m{int(mirror)}r{rotations}",
                    observation=transform_matrix(
                        example.observation, rotations=rotations, mirror=mirror
                    ),
                    row=row,
                    col=col,
                    probabilities=transform_matrix(
                        example.probabilities, rotations=rotations, mirror=mirror
                    ),
                    risk_mask=transform_matrix(
                        example.risk_mask, rotations=rotations, mirror=mirror
                    ),
                )
            )
    return augmented


def augment_dataset(input_path: str | Path, output_path: str | Path) -> tuple[int, int]:
    originals = list(read_replay_examples(input_path))
    augmented = [item for example in originals for item in augment_example(example)]
    write_replay_examples(augmented, output_path)
    return len(originals), len(augmented)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    originals, augmented = augment_dataset(args.input, args.output)
    print({"originals": originals, "augmented": augmented, "output": str(args.output.resolve())})


if __name__ == "__main__":
    main()
