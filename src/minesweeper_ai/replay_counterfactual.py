"""Generate per-cell counterfactual win labels from solver-shielded games.

At a difficult non-exact guess state, the generator clones the hidden board,
reveals several plausible candidate cells one at a time, and lets the same
continuation agent finish every branch. Hidden mines are used only to simulate
training outcomes; checkpoint inference still receives public observations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Iterable, Iterator, Literal, Sequence

from .agent import HybridAgent
from .data import BoardSpec
from .game import UNKNOWN, GameStatus, MinesweeperGame
from .replay_augmentation import transform_coord, transform_matrix
from .replay_inference import ReplayAgentPredictor
from .solver import Analysis, ConstraintSolver


@dataclass(frozen=True)
class CounterfactualCandidate:
    row: int
    col: int
    outcome: float
    mine_probability: float
    was_mine: bool
    continuation_guesses: float


@dataclass(frozen=True)
class CounterfactualExample:
    game_id: str
    state_index: int
    observation: list[list[int]]
    total_mines: int
    source: str
    candidates: list[CounterfactualCandidate]


@dataclass(frozen=True)
class CounterfactualStats:
    games: int
    baseline_wins: int
    states: int
    informative_states: int
    candidates: int
    candidate_wins: float
    immediate_mines: int
    solver_cells: int
    checkpoint: str
    neural_blend: float
    neural_blend_mode: str
    top_candidates: int
    max_states_per_game: int
    device: str
    cpu_threads: int
    continuation_mode: str
    rollouts_per_candidate: int
    stochastic_top_choices: int
    stochastic_temperature: float


def write_counterfactual_examples(
    examples: Iterable[CounterfactualExample],
    output: str | Path,
) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def read_counterfactual_examples(
    path: str | Path,
) -> Iterator[CounterfactualExample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            yield CounterfactualExample(
                game_id=raw["game_id"],
                state_index=int(raw["state_index"]),
                observation=raw["observation"],
                total_mines=int(raw["total_mines"]),
                source=raw.get("source", "simulator_counterfactual"),
                candidates=[
                    CounterfactualCandidate(**candidate)
                    for candidate in raw["candidates"]
                ],
            )


def augment_counterfactual_example(
    example: CounterfactualExample,
) -> list[CounterfactualExample]:
    height = len(example.observation)
    width = len(example.observation[0])
    augmented: list[CounterfactualExample] = []
    for mirror in (False, True):
        for rotations in range(4):
            candidates = []
            for candidate in example.candidates:
                row, col, _, _ = transform_coord(
                    candidate.row,
                    candidate.col,
                    height,
                    width,
                    rotations=rotations,
                    mirror=mirror,
                )
                candidates.append(replace(candidate, row=row, col=col))
            augmented.append(
                replace(
                    example,
                    game_id=f"{example.game_id}-aug-m{int(mirror)}r{rotations}",
                    observation=transform_matrix(
                        example.observation,
                        rotations=rotations,
                        mirror=mirror,
                    ),
                    candidates=candidates,
                )
            )
    return augmented


def generate_counterfactual_examples(
    checkpoint: str | Path,
    output: str | Path,
    *,
    games: int,
    seed: int,
    solver_cells: int = 24,
    neural_blend: float = 0.65,
    neural_blend_mode: Literal["probability", "rank"] = "probability",
    top_candidates: int = 4,
    max_states_per_game: int = 1,
    device: str = "cpu",
    cpu_threads: int = 2,
    continuation_mode: Literal["stochastic", "agent"] = "stochastic",
    rollouts_per_candidate: int = 4,
    stochastic_top_choices: int = 3,
    stochastic_temperature: float = 0.03,
    spec: BoardSpec = BoardSpec(30, 16, 99, True),
) -> CounterfactualStats:
    if games < 1:
        raise ValueError("games must be positive")
    if top_candidates < 2:
        raise ValueError("top_candidates must be at least 2")
    if max_states_per_game < 1:
        raise ValueError("max_states_per_game must be positive")
    if continuation_mode not in {"stochastic", "agent"}:
        raise ValueError("continuation_mode must be stochastic or agent")
    if continuation_mode == "agent" and rollouts_per_candidate != 1:
        raise ValueError("agent continuation requires exactly one rollout")
    if rollouts_per_candidate < 1:
        raise ValueError("rollouts_per_candidate must be positive")
    if stochastic_top_choices < 1:
        raise ValueError("stochastic_top_choices must be positive")
    if stochastic_temperature <= 0:
        raise ValueError("stochastic_temperature must be positive")
    if not 0.0 <= neural_blend <= 1.0:
        raise ValueError("neural_blend must be in [0, 1]")
    if neural_blend_mode not in {"probability", "rank"}:
        raise ValueError("neural_blend_mode must be probability or rank")

    predictor = ReplayAgentPredictor(
        checkpoint,
        mode="standard",
        source="simulator_counterfactual",
        device=device,
        cpu_threads=cpu_threads,
    )
    solver = ConstraintSolver(solver_cells)
    agent = HybridAgent(
        solver,
        risk_predictor=predictor,
        neural_blend=neural_blend,
        neural_blend_mode=neural_blend_mode,
    )
    game_rng = random.Random(seed)
    rollout_rng = random.Random(seed ^ 0x5A17C0DE)
    counters = {
        "baseline_wins": 0,
        "states": 0,
        "informative_states": 0,
        "candidates": 0,
        "candidate_wins": 0,
        "immediate_mines": 0,
    }

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for game_index in range(games):
            game_seed = game_rng.randrange(2**63)
            game = MinesweeperGame(
                spec.width,
                spec.height,
                spec.mines,
                seed=game_seed,
                first_click_zero=spec.first_click_zero,
            )
            examples, won = _collect_game(
                game,
                agent,
                predictor,
                game_id=f"counterfactual-{seed}-{game_index}",
                neural_blend=neural_blend,
                neural_blend_mode=neural_blend_mode,
                top_candidates=top_candidates,
                max_states=max_states_per_game,
                rollout_rng=rollout_rng,
                continuation_mode=continuation_mode,
                rollouts_per_candidate=rollouts_per_candidate,
                stochastic_top_choices=stochastic_top_choices,
                stochastic_temperature=stochastic_temperature,
            )
            counters["baseline_wins"] += int(won)
            for example in examples:
                handle.write(json.dumps(asdict(example), separators=(",", ":")))
                handle.write("\n")
                outcomes = {candidate.outcome for candidate in example.candidates}
                counters["states"] += 1
                counters["informative_states"] += len(outcomes) > 1
                counters["candidates"] += len(example.candidates)
                counters["candidate_wins"] += sum(
                    candidate.outcome for candidate in example.candidates
                )
                counters["immediate_mines"] += sum(
                    candidate.was_mine for candidate in example.candidates
                )

    return CounterfactualStats(
        games=games,
        solver_cells=solver_cells,
        checkpoint=str(Path(checkpoint)),
        neural_blend=neural_blend,
        neural_blend_mode=neural_blend_mode,
        top_candidates=top_candidates,
        max_states_per_game=max_states_per_game,
        device=device,
        cpu_threads=cpu_threads,
        continuation_mode=continuation_mode,
        rollouts_per_candidate=rollouts_per_candidate,
        stochastic_top_choices=stochastic_top_choices,
        stochastic_temperature=stochastic_temperature,
        **counters,
    )


def _collect_game(
    game: MinesweeperGame,
    agent: HybridAgent,
    predictor: ReplayAgentPredictor,
    *,
    game_id: str,
    neural_blend: float,
    neural_blend_mode: Literal["probability", "rank"],
    top_candidates: int,
    max_states: int,
    rollout_rng: random.Random,
    continuation_mode: Literal["stochastic", "agent"],
    rollouts_per_candidate: int,
    stochastic_top_choices: int,
    stochastic_temperature: float,
) -> tuple[list[CounterfactualExample], bool]:
    if game.status == GameStatus.READY:
        game.reveal((game.height // 2, game.width // 2))

    examples: list[CounterfactualExample] = []
    step_limit = game.width * game.height * 2
    steps = 0
    while game.status == GameStatus.ACTIVE and steps < step_limit:
        analysis = agent.solver.analyse(game.observation, game.mine_count)
        if analysis.contradiction:
            break
        for coord in sorted(analysis.mines):
            if game.observation[coord[0]][coord[1]] == UNKNOWN:
                game.toggle_flag(coord)

        decision = agent.choose_move(game.observation, game.mine_count, allow_guess=True)
        if decision.coord is None:
            break
        if (
            len(examples) < max_states
            and decision.analysis is not None
            and not decision.analysis.exact
            and decision.mine_probability not in (None, 0.0)
        ):
            scores = _candidate_scores(
                game.observation,
                game.mine_count,
                decision.analysis,
                predictor,
                neural_blend=neural_blend,
                neural_blend_mode=neural_blend_mode,
            )
            selected = sorted(scores, key=lambda coord: (scores[coord], coord))[
                :top_candidates
            ]
            if len(selected) >= 2:
                if continuation_mode == "stochastic":
                    rollout_seeds = tuple(
                        rollout_rng.randrange(2**63) for _ in range(rollouts_per_candidate)
                    )
                    rollout_predictor = predictor
                else:
                    rollout_seeds = None
                    rollout_predictor = None
                candidates = [
                    _rollout_candidate(
                        game,
                        coord,
                        agent,
                        mine_probability=scores[coord],
                        predictor=rollout_predictor,
                        rollouts=rollouts_per_candidate,
                        rollout_seeds=rollout_seeds,
                        neural_blend=neural_blend,
                        neural_blend_mode=neural_blend_mode,
                        stochastic_top_choices=stochastic_top_choices,
                        stochastic_temperature=stochastic_temperature,
                    )
                    for coord in selected
                ]
                examples.append(
                    CounterfactualExample(
                        game_id=game_id,
                        state_index=len(examples),
                        observation=[list(row) for row in game.observation],
                        total_mines=game.mine_count,
                        source="simulator_counterfactual",
                        candidates=candidates,
                    )
                )

        game.reveal(decision.coord)
        steps += 1
    return examples, game.status == GameStatus.WON


def _candidate_scores(
    observation,
    total_mines: int,
    analysis: Analysis,
    predictor: ReplayAgentPredictor,
    *,
    neural_blend: float,
    neural_blend_mode: Literal["probability", "rank"],
) -> dict[tuple[int, int], float]:
    learned = predictor.predict(observation, total_mines)
    eligible = {
        coord: probability
        for coord, probability in learned.items()
        if observation[coord[0]][coord[1]] == UNKNOWN
        and coord not in analysis.safe
        and coord not in analysis.mines
    }
    if neural_blend_mode == "rank":
        learned_values = _normalized_ranks(eligible)
        base_values = _normalized_ranks(
            {
                coord: analysis.probabilities.get(coord, probability)
                for coord, probability in eligible.items()
            }
        )
    else:
        learned_values = eligible
        base_values = {
            coord: analysis.probabilities.get(coord, probability)
            for coord, probability in eligible.items()
        }
    return {
        coord: neural_blend * learned_values[coord]
        + (1.0 - neural_blend) * base_values[coord]
        for coord in eligible
    }


def _rollout_candidate(
    game: MinesweeperGame,
    coord: tuple[int, int],
    continuation_agent: HybridAgent,
    *,
    mine_probability: float,
    predictor: ReplayAgentPredictor | None = None,
    rollouts: int = 1,
    rollout_seeds: Sequence[int] | None = None,
    neural_blend: float = 0.65,
    neural_blend_mode: Literal["probability", "rank"] = "probability",
    stochastic_top_choices: int = 3,
    stochastic_temperature: float = 0.03,
) -> CounterfactualCandidate:
    if rollouts < 1:
        raise ValueError("rollouts must be positive")
    if rollout_seeds is not None and len(rollout_seeds) != rollouts:
        raise ValueError("rollout_seeds length must match rollouts")
    was_mine = game.is_mine(coord)
    wins = 0
    guess_sum = 0
    for rollout_index in range(rollouts):
        branch = game.clone()
        branch.reveal(coord)
        if branch.status == GameStatus.ACTIVE:
            if predictor is not None and rollout_seeds is not None:
                branch_rng = random.Random(rollout_seeds[rollout_index])
                won, guesses = _play_stochastic(
                    branch,
                    continuation_agent,
                    predictor,
                    branch_rng,
                    neural_blend=neural_blend,
                    neural_blend_mode=neural_blend_mode,
                    top_choices=stochastic_top_choices,
                    temperature=stochastic_temperature,
                )
                wins += won
                guess_sum += guesses
            else:
                result = continuation_agent.play(branch, allow_guess=True)
                wins += result.won
                guess_sum += result.guesses
        else:
            wins += branch.status == GameStatus.WON
    return CounterfactualCandidate(
        row=coord[0],
        col=coord[1],
        outcome=wins / rollouts,
        mine_probability=float(mine_probability),
        was_mine=was_mine,
        continuation_guesses=guess_sum / rollouts,
    )


def _play_stochastic(
    game: MinesweeperGame,
    agent: HybridAgent,
    predictor: ReplayAgentPredictor,
    rng: random.Random,
    *,
    neural_blend: float,
    neural_blend_mode: Literal["probability", "rank"],
    top_choices: int,
    temperature: float,
) -> tuple[bool, int]:
    guesses = 0
    reveals = 0
    move_limit = game.width * game.height * 2
    while game.status == GameStatus.ACTIVE and reveals < move_limit:
        analysis = agent.solver.analyse(game.observation, game.mine_count)
        if analysis.contradiction:
            break
        for candidate in sorted(analysis.mines):
            if game.observation[candidate[0]][candidate[1]] == UNKNOWN:
                game.toggle_flag(candidate)

        decision = agent.choose_move(game.observation, game.mine_count, allow_guess=True)
        if decision.coord is None or decision.analysis is None:
            break
        coord = decision.coord
        if not decision.analysis.exact and decision.mine_probability not in (None, 0.0):
            scores = _candidate_scores(
                game.observation,
                game.mine_count,
                decision.analysis,
                predictor,
                neural_blend=neural_blend,
                neural_blend_mode=neural_blend_mode,
            )
            ranked = sorted(scores, key=lambda item: (scores[item], item))[:top_choices]
            if ranked:
                weights = _stable_choice_weights(scores, ranked, temperature)
                coord = (
                    rng.choices(ranked, weights=weights, k=1)[0]
                    if weights is not None
                    else ranked[0]
                )
            guesses += 1
        elif decision.mine_probability not in (None, 0.0):
            guesses += 1
        game.reveal(coord)
        reveals += 1
    return game.status == GameStatus.WON, guesses


def _stable_choice_weights(scores, ranked, temperature: float):
    finite = [scores[item] for item in ranked if math.isfinite(scores[item])]
    if not finite:
        return None
    minimum = min(finite)
    weights = [
        math.exp(-(scores[item] - minimum) / temperature)
        if math.isfinite(scores[item])
        else 0.0
        for item in ranked
    ]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        return None
    return weights


def _normalized_ranks(values) -> dict[tuple[int, int], float]:
    if not values:
        return {}
    ordered = sorted(set(values.values()))
    if len(ordered) == 1:
        return {coord: 0.0 for coord in values}
    ranks = {value: index / (len(ordered) - 1) for index, value in enumerate(ordered)}
    return {coord: ranks[value] for coord, value in values.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260935)
    parser.add_argument("--solver-cells", type=int, default=24)
    parser.add_argument("--neural-blend", type=float, default=0.65)
    parser.add_argument(
        "--neural-blend-mode",
        choices=("probability", "rank"),
        default="probability",
    )
    parser.add_argument("--top-candidates", type=int, default=4)
    parser.add_argument("--max-states-per-game", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--continuation-mode",
        choices=("stochastic", "agent"),
        default="stochastic",
    )
    parser.add_argument("--rollouts-per-candidate", type=int, default=4)
    parser.add_argument("--stochastic-top-choices", type=int, default=3)
    parser.add_argument("--stochastic-temperature", type=float, default=0.03)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--mines", type=int, default=99)
    parser.add_argument(
        "--first-click-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stats = generate_counterfactual_examples(
        args.checkpoint,
        args.output,
        games=args.games,
        seed=args.seed,
        solver_cells=args.solver_cells,
        neural_blend=args.neural_blend,
        neural_blend_mode=args.neural_blend_mode,
        top_candidates=args.top_candidates,
        max_states_per_game=args.max_states_per_game,
        device=args.device,
        cpu_threads=args.cpu_threads,
        continuation_mode=args.continuation_mode,
        rollouts_per_candidate=args.rollouts_per_candidate,
        stochastic_top_choices=args.stochastic_top_choices,
        stochastic_temperature=args.stochastic_temperature,
        spec=BoardSpec(
            args.width,
            args.height,
            args.mines,
            args.first_click_zero,
        ),
    )
    print(json.dumps(asdict(stats), separators=(",", ":")))


if __name__ == "__main__":
    main()

