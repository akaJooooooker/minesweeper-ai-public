"""Command-line entry points for benchmarks, dataset generation, and training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import comb
from pathlib import Path
import random
import time

from .agent import HybridAgent
from .data import (
    BoardSpec,
    DaggerGenerator,
    DatasetGenerator,
    generate_no_guess_game,
    write_jsonl,
)
from .game import MinesweeperGame
from .solver import ConstraintSolver

PRESETS = {
    "beginner": BoardSpec(9, 9, 10, True),
    "intermediate": BoardSpec(16, 16, 40, True),
    "expert": BoardSpec(30, 16, 99, True),
}

NO_GUESS_PRESETS = {
    "easy": BoardSpec(9, 9, 10, True),
    "medium": BoardSpec(16, 16, 40, True),
    "hard": BoardSpec(30, 16, 99, True),
    "evil": BoardSpec(30, 24, 130, True),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minesweeper-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="benchmark the hybrid solver")
    benchmark.add_argument("--games", type=int, default=20)
    benchmark.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(PRESETS),
        default=list(PRESETS),
    )
    benchmark.add_argument("--max-component-cells", type=int, default=24)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--first-click-safe", action="store_true")
    benchmark.add_argument("--model", type=Path)

    benchmark_ng = subparsers.add_parser(
        "benchmark-no-guess",
        help="generate teacher-solvable boards and test a no-guess solver budget",
    )
    benchmark_ng.add_argument("--games", type=int, default=10)
    benchmark_ng.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(NO_GUESS_PRESETS),
        default=["easy"],
    )
    benchmark_ng.add_argument("--generator-max-component-cells", type=int, default=24)
    benchmark_ng.add_argument("--solver-max-component-cells", type=int, default=24)
    benchmark_ng.add_argument("--fallback-max-component-cells", type=int)
    benchmark_ng.add_argument("--max-attempts", type=int, default=1_000)
    benchmark_ng.add_argument("--seed", type=int, default=0)

    compare = subparsers.add_parser(
        "compare",
        help="paired heuristic versus neural benchmark on identical boards",
    )
    compare.add_argument("--model", type=Path, required=True)
    compare.add_argument("--games", type=int, default=500)
    compare.add_argument(
        "--modes",
        nargs="*",
        choices=tuple(PRESETS),
        default=list(PRESETS),
    )
    compare.add_argument("--max-component-cells", type=int, default=8)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--first-click-safe", action="store_true")
    compare.add_argument("--model-device", choices=("cpu", "cuda"), default="cuda")
    compare.add_argument("--random-boards", type=int, default=0)
    compare.add_argument("--min-width", type=int, default=6)
    compare.add_argument("--max-width", type=int, default=40)
    compare.add_argument("--min-height", type=int, default=6)
    compare.add_argument("--max-height", type=int, default=30)
    compare.add_argument("--min-density", type=float, default=0.10)
    compare.add_argument("--max-density", type=float, default=0.24)

    generate = subparsers.add_parser("generate-data", help="write teacher states to JSONL")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--games-per-spec", type=int, default=100)
    generate.add_argument(
        "--modes",
        nargs="*",
        choices=tuple(PRESETS),
        default=None,
    )
    generate.add_argument(
        "--custom",
        nargs=3,
        type=int,
        action="append",
        metavar=("WIDTH", "HEIGHT", "MINES"),
        help="add an arbitrary board specification; may be repeated",
    )
    generate.add_argument("--random-specs", type=int, default=0)
    generate.add_argument("--min-width", type=int, default=6)
    generate.add_argument("--max-width", type=int, default=40)
    generate.add_argument("--min-height", type=int, default=6)
    generate.add_argument("--max-height", type=int, default=30)
    generate.add_argument("--min-density", type=float, default=0.10)
    generate.add_argument("--max-density", type=float, default=0.25)
    generate.add_argument("--max-component-cells", type=int, default=24)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--include-nonexact", action="store_true")
    generate.add_argument("--no-guess", action="store_true")
    generate.add_argument("--max-attempts", type=int, default=1_000)

    dagger = subparsers.add_parser(
        "generate-dagger",
        help="collect limited-student states and label them with the exact teacher",
    )
    dagger.add_argument("--output", type=Path, required=True)
    dagger.add_argument("--games-per-spec", type=int, default=100)
    dagger.add_argument("--modes", nargs="*", choices=tuple(PRESETS), default=None)
    dagger.add_argument(
        "--custom",
        nargs=3,
        type=int,
        action="append",
        metavar=("WIDTH", "HEIGHT", "MINES"),
    )
    dagger.add_argument("--random-specs", type=int, default=0)
    dagger.add_argument("--min-width", type=int, default=6)
    dagger.add_argument("--max-width", type=int, default=40)
    dagger.add_argument("--min-height", type=int, default=6)
    dagger.add_argument("--max-height", type=int, default=30)
    dagger.add_argument("--min-density", type=float, default=0.10)
    dagger.add_argument("--max-density", type=float, default=0.25)
    dagger.add_argument("--student-max-component-cells", type=int, default=8)
    dagger.add_argument("--teacher-max-component-cells", type=int, default=24)
    dagger.add_argument("--model", type=Path)
    dagger.add_argument("--model-device", choices=("cpu", "cuda"), default="cpu")
    dagger.add_argument("--seed", type=int, default=0)

    train = subparsers.add_parser("train", help="train the optional PyTorch model")
    train.add_argument("--dataset", type=Path, nargs="+", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--model-width", type=int, default=64)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--gpu-memory-fraction", type=float)
    train.add_argument("--batch-delay-ms", type=int, default=0)

    arguments = parser.parse_args(argv)
    if arguments.command == "benchmark":
        return _benchmark(arguments)
    if arguments.command == "benchmark-no-guess":
        return _benchmark_no_guess(arguments)
    if arguments.command == "compare":
        return _compare(arguments)
    if arguments.command == "generate-data":
        return _generate_data(arguments)
    if arguments.command == "generate-dagger":
        return _generate_dagger(arguments)
    if arguments.command == "train":
        return _train(arguments)
    parser.error("unknown command")
    return 2


def _benchmark(arguments: argparse.Namespace) -> int:
    if arguments.games < 1:
        raise SystemExit("--games must be positive")
    solver = ConstraintSolver(arguments.max_component_cells)
    predictor = None
    if arguments.model:
        from .inference import NeuralRiskPredictor

        predictor = NeuralRiskPredictor(arguments.model)
    agent = HybridAgent(solver, predictor)
    rng = random.Random(arguments.seed)
    for mode in arguments.modes:
        preset = PRESETS[mode]
        wins = guesses = stuck = reveals = 0
        started = time.perf_counter()
        for _ in range(arguments.games):
            game = MinesweeperGame(
                preset.width,
                preset.height,
                preset.mines,
                seed=rng.randrange(2**63),
                first_click_safe=True,
                first_click_zero=not arguments.first_click_safe,
            )
            result = agent.play(game, allow_guess=True)
            wins += int(result.won)
            guesses += result.guesses
            stuck += int(result.stuck)
            reveals += result.reveals
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "mode": mode,
                    "board": f"{preset.width}x{preset.height}/{preset.mines}",
                    "games": arguments.games,
                    "wins": wins,
                    "win_rate": wins / arguments.games,
                    "guesses": guesses,
                    "stuck": stuck,
                    "reveals": reveals,
                    "seconds": round(elapsed, 3),
                },
                ensure_ascii=False,
            )
        )
    return 0


def _benchmark_no_guess(arguments: argparse.Namespace) -> int:
    if arguments.games < 1:
        raise SystemExit("--games must be positive")
    generator_solver = ConstraintSolver(arguments.generator_max_component_cells)
    primary_solver = ConstraintSolver(arguments.solver_max_component_cells)
    fallback_solver = None
    if arguments.fallback_max_component_cells is not None:
        fallback_solver = ConstraintSolver(arguments.fallback_max_component_cells)
    target_agent = HybridAgent(primary_solver, fallback_solver=fallback_solver)
    rng = random.Random(arguments.seed)
    for mode in arguments.modes:
        spec = NO_GUESS_PRESETS[mode]
        solved = stuck = reveals = flags = escalations = 0
        started = time.perf_counter()
        for _ in range(arguments.games):
            game = generate_no_guess_game(
                spec,
                seed=rng.randrange(2**63),
                solver=generator_solver,
                max_attempts=arguments.max_attempts,
            )
            game.reveal((spec.height // 2, spec.width // 2))
            result = target_agent.play(game, allow_guess=False)
            solved += int(result.won)
            stuck += int(result.stuck)
            reveals += result.reveals
            flags += result.flags
            escalations += result.escalations
        print(
            json.dumps(
                {
                    "mode": mode,
                    "board": f"{spec.width}x{spec.height}/{spec.mines}",
                    "games": arguments.games,
                    "solved_without_guess": solved,
                    "solve_rate": solved / arguments.games,
                    "stuck": stuck,
                    "reveals": reveals,
                    "flags": flags,
                    "escalations": escalations,
                    "generator_max_component_cells": (
                        arguments.generator_max_component_cells
                    ),
                    "solver_max_component_cells": arguments.solver_max_component_cells,
                    "fallback_max_component_cells": (
                        arguments.fallback_max_component_cells
                    ),
                    "seconds": round(time.perf_counter() - started, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


def _compare(arguments: argparse.Namespace) -> int:
    if arguments.games < 1:
        raise SystemExit("--games must be positive")
    from .inference import NeuralRiskPredictor

    solver = ConstraintSolver(arguments.max_component_cells)
    heuristic_agent = HybridAgent(solver)
    neural_agent = HybridAgent(
        solver,
        NeuralRiskPredictor(arguments.model, device=arguments.model_device),
    )
    rng = random.Random(arguments.seed)
    for mode in arguments.modes:
        preset = PRESETS[mode]
        cases = [(preset, rng.randrange(2**63)) for _ in range(arguments.games)]
        _print_paired_summary(
            mode,
            f"{preset.width}x{preset.height}/{preset.mines}",
            cases,
            heuristic_agent,
            neural_agent,
            first_click_zero=not arguments.first_click_safe,
        )

    if arguments.random_boards:
        if arguments.random_boards < 1:
            raise SystemExit("--random-boards must be positive")
        if not (
            4 <= arguments.min_width <= arguments.max_width
            and 4 <= arguments.min_height <= arguments.max_height
            and 0 < arguments.min_density <= arguments.max_density < 1
        ):
            raise SystemExit("invalid random size or density range")
        cases = []
        for _ in range(arguments.random_boards):
            width = rng.randint(arguments.min_width, arguments.max_width)
            height = rng.randint(arguments.min_height, arguments.max_height)
            density = rng.uniform(arguments.min_density, arguments.max_density)
            mines = max(1, min(round(width * height * density), width * height - 9))
            cases.append((BoardSpec(width, height, mines, True), rng.randrange(2**63)))
        _print_paired_summary(
            "random",
            (
                f"width={arguments.min_width}..{arguments.max_width},"
                f"height={arguments.min_height}..{arguments.max_height},"
                f"density={arguments.min_density:.2f}..{arguments.max_density:.2f}"
            ),
            cases,
            heuristic_agent,
            neural_agent,
            first_click_zero=not arguments.first_click_safe,
        )
    return 0


def _print_paired_summary(
    mode: str,
    board_description: str,
    cases: list[tuple[BoardSpec, int]],
    heuristic_agent: HybridAgent,
    neural_agent: HybridAgent,
    *,
    first_click_zero: bool,
) -> None:
    paired = {
        "both_win": 0,
        "heuristic_only": 0,
        "neural_only": 0,
        "both_lose": 0,
    }
    heuristic_guesses = neural_guesses = 0
    started = time.perf_counter()
    for preset, board_seed in cases:
        results = []
        for agent in (heuristic_agent, neural_agent):
            game = MinesweeperGame(
                preset.width,
                preset.height,
                preset.mines,
                seed=board_seed,
                first_click_safe=True,
                first_click_zero=first_click_zero,
            )
            results.append(agent.play(game, allow_guess=True))
        heuristic_result, neural_result = results
        heuristic_guesses += heuristic_result.guesses
        neural_guesses += neural_result.guesses
        if heuristic_result.won and neural_result.won:
            paired["both_win"] += 1
        elif heuristic_result.won:
            paired["heuristic_only"] += 1
        elif neural_result.won:
            paired["neural_only"] += 1
        else:
            paired["both_lose"] += 1

    game_count = len(cases)
    heuristic_wins = paired["both_win"] + paired["heuristic_only"]
    neural_wins = paired["both_win"] + paired["neural_only"]
    print(
        json.dumps(
            {
                "mode": mode,
                "board": board_description,
                "games": game_count,
                "heuristic_wins": heuristic_wins,
                "heuristic_win_rate": heuristic_wins / game_count,
                "neural_wins": neural_wins,
                "neural_win_rate": neural_wins / game_count,
                "delta": (neural_wins - heuristic_wins) / game_count,
                **paired,
                "mcnemar_exact_p": _mcnemar_exact_p(
                    paired["heuristic_only"],
                    paired["neural_only"],
                ),
                "heuristic_guesses": heuristic_guesses,
                "neural_guesses": neural_guesses,
                "seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _mcnemar_exact_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = 2 * sum(comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return min(1.0, probability)


def _generate_data(arguments: argparse.Namespace) -> int:
    solver = ConstraintSolver(arguments.max_component_cells)
    generator = DatasetGenerator(solver, arguments.seed)
    specs = _build_specs(arguments)
    examples = generator.generate(
        specs,
        arguments.games_per_spec,
        include_nonexact=arguments.include_nonexact,
        no_guess=arguments.no_guess,
        max_attempts=arguments.max_attempts,
    )
    count = write_jsonl(examples, arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "examples": count,
                "specs": [asdict(spec) for spec in specs],
                "no_guess": arguments.no_guess,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _build_specs(arguments: argparse.Namespace) -> list[BoardSpec]:
    modes = arguments.modes
    if modes is None and not arguments.custom and not arguments.random_specs:
        modes = list(PRESETS)
    specs = [PRESETS[mode] for mode in (modes or [])]
    for width, height, mines in arguments.custom or []:
        specs.append(BoardSpec(width, height, mines, True))
    if arguments.random_specs < 0:
        raise SystemExit("--random-specs must not be negative")
    if not (
        4 <= arguments.min_width <= arguments.max_width
        and 4 <= arguments.min_height <= arguments.max_height
        and 0 < arguments.min_density <= arguments.max_density < 1
    ):
        raise SystemExit("invalid random size or density range")
    rng = random.Random(arguments.seed)
    for _ in range(arguments.random_specs):
        width = rng.randint(arguments.min_width, arguments.max_width)
        height = rng.randint(arguments.min_height, arguments.max_height)
        density = rng.uniform(arguments.min_density, arguments.max_density)
        mines = max(1, min(round(width * height * density), width * height - 9))
        specs.append(BoardSpec(width, height, mines, True))
    if not specs:
        raise SystemExit("select at least one preset, custom board, or random spec")
    return specs


def _generate_dagger(arguments: argparse.Namespace) -> int:
    if arguments.teacher_max_component_cells <= arguments.student_max_component_cells:
        raise SystemExit("teacher component budget must be larger than student budget")
    teacher = ConstraintSolver(arguments.teacher_max_component_cells)
    student = ConstraintSolver(arguments.student_max_component_cells)
    predictor = None
    if arguments.model:
        from .inference import NeuralRiskPredictor

        predictor = NeuralRiskPredictor(arguments.model, device=arguments.model_device)
    generator = DaggerGenerator(
        teacher,
        student,
        risk_predictor=predictor,
        seed=arguments.seed,
    )
    specs = _build_specs(arguments)
    count = write_jsonl(
        generator.generate(specs, arguments.games_per_spec),
        arguments.output,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "examples": count,
                "specs": [asdict(spec) for spec in specs],
                "student_max_component_cells": arguments.student_max_component_cells,
                "teacher_max_component_cells": arguments.teacher_max_component_cells,
                "model": str(arguments.model) if arguments.model else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _train(arguments: argparse.Namespace) -> int:
    from .training import train_model

    def report_epoch(epoch: int, total: int, loss: float) -> None:
        print(
            json.dumps(
                {"event": "epoch", "epoch": epoch, "total_epochs": total, "loss": loss}
            ),
            flush=True,
        )

    history = train_model(
        arguments.dataset,
        arguments.output,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        model_width=arguments.model_width,
        seed=arguments.seed,
        device_name=arguments.device,
        use_amp=not arguments.no_amp,
        gpu_memory_fraction=arguments.gpu_memory_fraction,
        batch_delay_ms=arguments.batch_delay_ms,
        on_epoch=report_epoch,
    )
    print(json.dumps({"output": str(arguments.output.resolve()), "loss": history}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
