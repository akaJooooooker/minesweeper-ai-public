"""Convert audited Replay examples into a Risk-only training dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Iterator, Sequence

from .replay import (
    ReplayTrainingExample,
    read_replay_examples,
    write_replay_examples,
)


@dataclass(frozen=True)
class RiskOnlyStats:
    inputs: int
    selected: int
    output: str


def build_risk_only_dataset(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
) -> RiskOnlyStats:
    input_count = 0
    selected_count = 0

    def examples() -> Iterator[ReplayTrainingExample]:
        nonlocal input_count, selected_count
        for path in input_paths:
            for example in read_replay_examples(path):
                input_count += 1
                if not any(any(row) for row in example.risk_mask):
                    continue
                selected_count += 1
                yield replace(
                    example,
                    policy_weight=0.0,
                    value_weight=0.0,
                )

    output = Path(output_path)
    write_replay_examples(examples(), output)
    return RiskOnlyStats(
        inputs=input_count,
        selected=selected_count,
        output=str(output.resolve()),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stats = build_risk_only_dataset(args.input, args.output)
    print(json.dumps(asdict(stats), separators=(",", ":")))


if __name__ == "__main__":
    main()

