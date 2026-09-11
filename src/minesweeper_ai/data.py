"""Generate supervised examples from the exact constraint teacher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Iterator, Sequence

from .agent import HybridAgent
from .game import FLAGGED, UNKNOWN, Coord, GameStatus, MinesweeperGame
from .solver import Analysis, ConstraintSolver


@dataclass(frozen=True)
class BoardSpec:
    width: int
    height: int
    mines: int
    first_click_zero: bool = True


@dataclass(frozen=True)
class TrainingExample:
    observation: list[list[int]]
    total_mines: int
    probabilities: list[list[float | None]]
    target_mask: list[list[int]]
    exact: bool
    source: str = "teacher_rollout"
    frontier_size: int = 0
    student_exact: bool | None = None

    @classmethod
    def from_analysis(
        cls,
        observation: Sequence[Sequence[int]],
        total_mines: int,
        analysis: Analysis,
        *,
        source: str = "teacher_rollout",
        student_exact: bool | None = None,
    ) -> "TrainingExample":
        height = len(observation)
        width = len(observation[0])
        probabilities: list[list[float | None]] = [
            [None for _ in range(width)] for _ in range(height)
        ]
        mask = [[0 for _ in range(width)] for _ in range(height)]
        for (row, col), probability in analysis.probabilities.items():
            if observation[row][col] == UNKNOWN:
                probabilities[row][col] = float(probability)
                mask[row][col] = 1
        return cls(
            observation=[list(row) for row in observation],
            total_mines=total_mines,
            probabilities=probabilities,
            target_mask=mask,
            exact=analysis.exact,
            source=source,
            frontier_size=analysis.frontier_size,
            student_exact=student_exact,
        )


class DatasetGenerator:
    def __init__(self, solver: ConstraintSolver | None = None, seed: int = 0) -> None:
        self.solver = solver or ConstraintSolver()
        self.agent = HybridAgent(self.solver)
        self._rng = random.Random(seed)

    def generate(
        self,
        specs: Sequence[BoardSpec],
        games_per_spec: int,
        *,
        include_nonexact: bool = False,
        no_guess: bool = False,
        max_attempts: int = 1_000,
    ) -> Iterator[TrainingExample]:
        if games_per_spec < 1:
            raise ValueError("games_per_spec must be positive")
        for spec in specs:
            for _ in range(games_per_spec):
                game_seed = self._rng.randrange(2**63)
                if no_guess:
                    game = generate_no_guess_game(
                        spec,
                        seed=game_seed,
                        solver=self.solver,
                        max_attempts=max_attempts,
                    )
                else:
                    game = MinesweeperGame(
                        spec.width,
                        spec.height,
                        spec.mines,
                        seed=game_seed,
                        first_click_zero=spec.first_click_zero,
                    )
                yield from self.collect_game(game, include_nonexact=include_nonexact)

    def collect_game(
        self,
        game: MinesweeperGame,
        *,
        include_nonexact: bool = False,
    ) -> Iterator[TrainingExample]:
        if game.status == GameStatus.READY:
            game.reveal((game.height // 2, game.width // 2))

        step_limit = game.width * game.height * 2
        steps = 0
        while game.status == GameStatus.ACTIVE and steps < step_limit:
            analysis = self.solver.analyse(game.observation, game.mine_count)
            if analysis.contradiction:
                break
            if analysis.exact or include_nonexact:
                example = TrainingExample.from_analysis(
                    game.observation,
                    game.mine_count,
                    analysis,
                )
                if any(any(row) for row in example.target_mask):
                    yield example

            for coord in sorted(analysis.mines):
                if game.observation[coord[0]][coord[1]] == UNKNOWN:
                    game.toggle_flag(coord)

            decision = self.agent.choose_move(
                game.observation,
                game.mine_count,
                allow_guess=True,
            )
            if decision.coord is None:
                break
            game.reveal(decision.coord)
            steps += 1


class DaggerGenerator:
    """Collect states visited by a limited student and label them with a teacher.

    The student deliberately uses a smaller exact-enumeration budget. A state is
    recorded only when the student falls back to approximation while the larger
    teacher can still provide an exact probability distribution.
    """

    def __init__(
        self,
        teacher_solver: ConstraintSolver,
        student_solver: ConstraintSolver,
        *,
        risk_predictor=None,
        seed: int = 0,
    ) -> None:
        if teacher_solver.max_component_cells <= student_solver.max_component_cells:
            raise ValueError("teacher must have a larger component budget than student")
        self.teacher_solver = teacher_solver
        self.student_solver = student_solver
        self.student_agent = HybridAgent(student_solver, risk_predictor)
        self._rng = random.Random(seed)

    def generate(
        self,
        specs: Sequence[BoardSpec],
        games_per_spec: int,
    ) -> Iterator[TrainingExample]:
        if games_per_spec < 1:
            raise ValueError("games_per_spec must be positive")
        for spec in specs:
            for _ in range(games_per_spec):
                game = MinesweeperGame(
                    spec.width,
                    spec.height,
                    spec.mines,
                    seed=self._rng.randrange(2**63),
                    first_click_zero=spec.first_click_zero,
                )
                yield from self.collect_game(game)

    def collect_game(self, game: MinesweeperGame) -> Iterator[TrainingExample]:
        if game.status == GameStatus.READY:
            game.reveal((game.height // 2, game.width // 2))

        step_limit = game.width * game.height * 2
        steps = 0
        while game.status == GameStatus.ACTIVE and steps < step_limit:
            student = self.student_solver.analyse(game.observation, game.mine_count)
            if student.contradiction:
                break
            if not student.exact:
                teacher = self.teacher_solver.analyse(game.observation, game.mine_count)
                if teacher.exact and not teacher.contradiction:
                    example = TrainingExample.from_analysis(
                        game.observation,
                        game.mine_count,
                        teacher,
                        source="dagger",
                        student_exact=False,
                    )
                    if any(any(row) for row in example.target_mask):
                        yield example

            for coord in sorted(student.mines):
                if game.observation[coord[0]][coord[1]] == UNKNOWN:
                    game.toggle_flag(coord)

            decision = self.student_agent.choose_move(
                game.observation,
                game.mine_count,
                allow_guess=True,
            )
            if decision.coord is None:
                break
            game.reveal(decision.coord)
            steps += 1


def write_jsonl(examples: Iterable[TrainingExample], output: str | Path) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[TrainingExample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield TrainingExample(**json.loads(line))


def generate_no_guess_game(
    spec: BoardSpec,
    *,
    seed: int,
    solver: ConstraintSolver | None = None,
    max_attempts: int = 1_000,
) -> MinesweeperGame:
    """Rejection-generate a board the current exact solver can finish unaided.

    This creates research boards with a no-guess guarantee relative to our
    solver. It does not attempt to clone minesweeper.online's private generator.
    """
    solver = solver or ConstraintSolver()
    agent = HybridAgent(solver)
    rng = random.Random(seed)
    start = (spec.height // 2, spec.width // 2)
    all_cells = [(r, c) for r in range(spec.height) for c in range(spec.width)]
    excluded: set[Coord] = {start}
    if spec.first_click_zero:
        start_row, start_col = start
        excluded.update(
            (row, col)
            for row in range(max(0, start_row - 1), min(spec.height, start_row + 2))
            for col in range(max(0, start_col - 1), min(spec.width, start_col + 2))
        )
    candidates = [coord for coord in all_cells if coord not in excluded]
    if spec.mines > len(candidates):
        raise ValueError("too many mines for the requested no-guess start")

    for _ in range(max_attempts):
        mines = set(rng.sample(candidates, spec.mines))
        trial = MinesweeperGame(
            spec.width,
            spec.height,
            spec.mines,
            mine_positions=mines,
        )
        trial.reveal(start)
        result = agent.play(trial, allow_guess=False)
        if result.won:
            return MinesweeperGame(
                spec.width,
                spec.height,
                spec.mines,
                mine_positions=mines,
            )
    raise RuntimeError(f"no no-guess board found after {max_attempts} attempts")
