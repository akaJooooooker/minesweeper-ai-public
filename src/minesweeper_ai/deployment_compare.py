"""Paired standard-board comparison for two deployment configurations."""

from __future__ import annotations

import argparse
from math import comb
import json
from pathlib import Path
import random
import time
from typing import Sequence

from .agent import HybridAgent
from .data import BoardSpec
from .game import MinesweeperGame
from .metrics import calculate_3bv
from .replay_deployment import load_replay_deployment
from .scorecard import STANDARD_PRESETS


def _mcnemar_exact_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = 2 * sum(comb(discordant, index) for index in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, probability)


def _play_standard(agent: HybridAgent, spec: BoardSpec, board_seed: int) -> dict[str, object]:
    game = MinesweeperGame(
        spec.width,
        spec.height,
        spec.mines,
        seed=board_seed,
        first_click_safe=True,
        first_click_zero=False,
    )
    started = time.perf_counter()
    first = agent.choose_move(game.observation, game.mine_count, allow_guess=True)
    if first.coord is None:
        raise RuntimeError("agent produced no standard first click")
    game.reveal(first.coord)
    result = agent.play(game, allow_guess=True)
    elapsed = time.perf_counter() - started
    return {
        "won": result.won,
        "seconds": elapsed,
        "three_bv": calculate_3bv(game).value,
        "clicks": 1 + result.clicks,
        "guesses": result.guesses,
    }


def _agent_summary(records: Sequence[dict[str, object]]) -> dict[str, object]:
    wins = [record for record in records if bool(record["won"])]
    seconds = sum(float(record["seconds"]) for record in wins)
    three_bv = sum(int(record["three_bv"]) for record in wins)
    clicks = sum(int(record["clicks"]) for record in wins)
    return {
        "wins": len(wins),
        "win_rate": len(wins) / len(records),
        "guesses": sum(int(record["guesses"]) for record in records),
        "three_bv_per_second": three_bv / seconds if seconds else 0.0,
        "ioe": three_bv / clicks if clicks else 0.0,
    }


def compare_standard(
    reference: HybridAgent,
    candidate: HybridAgent,
    spec: BoardSpec,
    *,
    games: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    reference_records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    paired = {"both_win": 0, "reference_only": 0, "candidate_only": 0, "both_lose": 0}
    for _ in range(games):
        board_seed = rng.randrange(2**63)
        reference_result = _play_standard(reference, spec, board_seed)
        candidate_result = _play_standard(candidate, spec, board_seed)
        reference_records.append(reference_result)
        candidate_records.append(candidate_result)
        reference_won = bool(reference_result["won"])
        candidate_won = bool(candidate_result["won"])
        if reference_won and candidate_won:
            paired["both_win"] += 1
        elif reference_won:
            paired["reference_only"] += 1
        elif candidate_won:
            paired["candidate_only"] += 1
        else:
            paired["both_lose"] += 1
    reference_summary = _agent_summary(reference_records)
    candidate_summary = _agent_summary(candidate_records)
    return {
        "track": "standard_paired",
        "board": f"{spec.width}x{spec.height}/{spec.mines}",
        "first_click_rule": "free_safe_only",
        "games": games,
        "reference": reference_summary,
        "candidate": candidate_summary,
        "win_rate_delta": float(candidate_summary["win_rate"])
        - float(reference_summary["win_rate"]),
        **paired,
        "mcnemar_exact_p": _mcnemar_exact_p(
            paired["reference_only"], paired["candidate_only"]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.games < 1:
        raise SystemExit("--games must be positive")
    reference, _ = load_replay_deployment(args.reference, device=args.device)
    candidate, _ = load_replay_deployment(args.candidate, device=args.device)
    seed_rng = random.Random(args.seed)
    for mode, spec in STANDARD_PRESETS.items():
        result = compare_standard(
            reference,
            candidate,
            spec,
            games=args.games,
            seed=seed_rng.randrange(2**63),
        )
        result["mode"] = mode
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
