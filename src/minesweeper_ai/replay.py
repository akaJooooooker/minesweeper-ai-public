"""Replay reconstruction, solver auditing, and imitation-learning examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Iterator, Literal, Sequence

from .agent import HybridAgent
from .data import BoardSpec, generate_no_guess_game
from .game import FLAGGED, UNKNOWN, Coord, GameStatus, MinesweeperGame
from .solver import Analysis, ConstraintSolver

ReplayMode = Literal["standard", "no_guess"]
ReplayAction = Literal["reveal", "flag", "chord"]
AuditClass = Literal[
    "PROVEN_GOOD",
    "LEGAL_UNPROVEN",
    "NECESSARY_GUESS",
    "UNNECESSARY_GUESS",
    "PROVEN_BAD",
    "AMBIGUOUS",
]


@dataclass(frozen=True)
class ReplayMove:
    action: ReplayAction
    row: int
    col: int
    time_ms: int = 0

    @property
    def coord(self) -> Coord:
        return self.row, self.col


@dataclass(frozen=True)
class ReplayGame:
    game_id: str
    width: int
    height: int
    mines: int
    mode: ReplayMode
    source: str
    won: bool
    mine_positions: list[list[int]]
    moves: list[ReplayMove]


@dataclass(frozen=True)
class ReplayTrainingExample:
    game_id: str
    observation: list[list[int]]
    total_mines: int
    mode: ReplayMode
    source: str
    action: ReplayAction
    row: int
    col: int
    time_ms: int
    classification: AuditClass
    teacher_exact: bool
    mine_probability: float | None
    probabilities: list[list[float | None]]
    risk_mask: list[list[int]]
    policy_weight: float
    value_target: float
    value_weight: float


class ReplayAuditor:
    """Rebuild every move and judge it using only the pre-move public board."""

    def __init__(self, solver: ConstraintSolver | None = None) -> None:
        self.solver = solver or ConstraintSolver()

    def audit(self, replay: ReplayGame) -> Iterator[ReplayTrainingExample]:
        _validate_replay(replay)
        mines = {(row, col) for row, col in replay.mine_positions}
        game = MinesweeperGame(
            replay.width,
            replay.height,
            replay.mines,
            mine_positions=mines,
        )
        safe_cell_count = replay.width * replay.height - replay.mines

        for move in replay.moves:
            if game.status in (GameStatus.WON, GameStatus.LOST):
                break
            observation = game.observation
            audit_observation = _without_unverified_flags(observation)
            analysis = self.solver.analyse(audit_observation, replay.mines)
            classification, probability = self._classify(
                game,
                move,
                audit_observation,
                analysis,
            )
            # Flags are treated as unknown while asking the solver for an
            # unbiased probability, but they are not legal reveal targets in
            # the observation that the model receives.
            probabilities, risk_mask = _risk_targets(observation, analysis)
            revealed = sum(cell >= 0 for row in observation for cell in row)
            progress = revealed / max(safe_cell_count, 1)
            yield ReplayTrainingExample(
                game_id=replay.game_id,
                observation=[list(row) for row in observation],
                total_mines=replay.mines,
                mode=replay.mode,
                source=replay.source,
                action=move.action,
                row=move.row,
                col=move.col,
                time_ms=move.time_ms,
                classification=classification,
                teacher_exact=analysis.exact,
                mine_probability=probability,
                probabilities=probabilities,
                risk_mask=risk_mask,
                policy_weight=1.0 if classification == "PROVEN_GOOD" else 0.0,
                value_target=1.0 if replay.won else 0.0,
                value_weight=0.5 + 0.5 * progress,
            )
            _apply_move(game, move)

    def _classify(
        self,
        game: MinesweeperGame,
        move: ReplayMove,
        audit_observation: Sequence[Sequence[int]],
        analysis: Analysis,
    ) -> tuple[AuditClass, float | None]:
        row, col = move.coord
        if not (0 <= row < game.height and 0 <= col < game.width):
            return "PROVEN_BAD", None
        if analysis.contradiction:
            return "AMBIGUOUS", None
        probability = analysis.probabilities.get(move.coord)

        if move.action == "reveal":
            if game.observation[row][col] != UNKNOWN:
                return "PROVEN_BAD", probability
            if probability == 0.0:
                return "PROVEN_GOOD", probability
            if probability == 1.0:
                return "PROVEN_BAD", probability
            if analysis.safe:
                return "UNNECESSARY_GUESS", probability
            if analysis.exact and probability is not None:
                return "NECESSARY_GUESS", probability
            return ("LEGAL_UNPROVEN" if probability is not None else "AMBIGUOUS"), probability

        if move.action == "flag":
            if game.observation[row][col] != UNKNOWN:
                return "PROVEN_BAD", probability
            if probability == 1.0:
                return "PROVEN_GOOD", probability
            if probability == 0.0:
                return "PROVEN_BAD", probability
            return ("LEGAL_UNPROVEN" if probability is not None else "AMBIGUOUS"), probability

        clue = game.observation[row][col]
        if clue < 0:
            return "PROVEN_BAD", probability
        neighbours = tuple(game.neighbours(move.coord))
        flags = [coord for coord in neighbours if game.observation[coord[0]][coord[1]] == FLAGGED]
        covered = [coord for coord in neighbours if game.observation[coord[0]][coord[1]] == UNKNOWN]
        if len(flags) != clue or not covered:
            return "PROVEN_BAD", probability
        flags_proven = all(analysis.probabilities.get(coord) == 1.0 for coord in flags)
        covered_safe = all(analysis.probabilities.get(coord) == 0.0 for coord in covered)
        if flags_proven and covered_safe:
            return "PROVEN_GOOD", probability
        return ("PROVEN_BAD" if analysis.exact else "AMBIGUOUS"), probability


def generate_simulator_replays(
    specs: Sequence[BoardSpec],
    games_per_spec: int,
    *,
    mode: ReplayMode,
    seed: int = 0,
    solver: ConstraintSolver | None = None,
    max_attempts: int = 1_000,
) -> Iterator[ReplayGame]:
    """Create traceable bootstrap replays without pretending they are human data."""
    if games_per_spec < 1:
        raise ValueError("games_per_spec must be positive")
    teacher = solver or ConstraintSolver()
    rng = random.Random(seed)
    for spec_index, spec in enumerate(specs):
        for game_index in range(games_per_spec):
            game_seed = rng.randrange(2**63)
            if mode == "no_guess":
                game = generate_no_guess_game(
                    spec,
                    seed=game_seed,
                    solver=teacher,
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
            yield _record_teacher_game(
                game,
                teacher,
                mode=mode,
                game_id=f"sim-{mode}-{seed}-{spec_index}-{game_index}",
            )


def _record_teacher_game(
    game: MinesweeperGame,
    solver: ConstraintSolver,
    *,
    mode: ReplayMode,
    game_id: str,
) -> ReplayGame:
    agent = HybridAgent(solver)
    moves: list[ReplayMove] = []
    elapsed = 0
    limit = game.width * game.height * 3
    while game.status not in (GameStatus.WON, GameStatus.LOST) and len(moves) < limit:
        if game.status == GameStatus.READY:
            coord = (game.height // 2, game.width // 2)
            moves.append(ReplayMove("reveal", coord[0], coord[1], elapsed))
            game.reveal(coord)
            elapsed += 100
            continue

        analysis = solver.analyse(game.observation, game.mine_count)
        if analysis.contradiction:
            break
        for coord in sorted(analysis.mines):
            if game.observation[coord[0]][coord[1]] == UNKNOWN:
                moves.append(ReplayMove("flag", coord[0], coord[1], elapsed))
                game.toggle_flag(coord)
                elapsed += 100
        decision = agent.choose_move(
            game.observation,
            game.mine_count,
            allow_guess=mode == "standard",
        )
        if decision.coord is None:
            break
        moves.append(ReplayMove("reveal", decision.coord[0], decision.coord[1], elapsed))
        game.reveal(decision.coord)
        elapsed += 100

    mine_positions = [
        [row, col]
        for row in range(game.height)
        for col in range(game.width)
        if game.is_mine((row, col))
    ]
    return ReplayGame(
        game_id=game_id,
        width=game.width,
        height=game.height,
        mines=game.mine_count,
        mode=mode,
        source="simulator_teacher",
        won=game.status == GameStatus.WON,
        mine_positions=mine_positions,
        moves=moves,
    )


def write_replays(replays: Iterable[ReplayGame], output: str | Path) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for replay in replays:
            handle.write(json.dumps(asdict(replay), separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def read_replays(path: str | Path) -> Iterator[ReplayGame]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item["moves"] = [ReplayMove(**move) for move in item["moves"]]
            yield ReplayGame(**item)


def write_replay_examples(
    examples: Iterable[ReplayTrainingExample],
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


def read_replay_examples(path: str | Path) -> Iterator[ReplayTrainingExample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield ReplayTrainingExample(**json.loads(line))


def _without_unverified_flags(
    observation: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(UNKNOWN if cell == FLAGGED else cell for cell in row)
        for row in observation
    )


def _risk_targets(
    observation: Sequence[Sequence[int]],
    analysis: Analysis,
) -> tuple[list[list[float | None]], list[list[int]]]:
    height = len(observation)
    width = len(observation[0])
    probabilities = [[None for _ in range(width)] for _ in range(height)]
    mask = [[0 for _ in range(width)] for _ in range(height)]
    if not analysis.exact or analysis.contradiction:
        return probabilities, mask
    for (row, col), probability in analysis.probabilities.items():
        if observation[row][col] == UNKNOWN:
            probabilities[row][col] = float(probability)
            mask[row][col] = 1
    return probabilities, mask


def _apply_move(game: MinesweeperGame, move: ReplayMove) -> None:
    row, col = move.coord
    if not (0 <= row < game.height and 0 <= col < game.width):
        return
    if move.action == "reveal":
        game.reveal(move.coord)
    elif move.action == "flag":
        game.toggle_flag(move.coord)
    else:
        game.chord(move.coord)


def _validate_replay(replay: ReplayGame) -> None:
    if replay.mode not in ("standard", "no_guess"):
        raise ValueError(f"unsupported replay mode: {replay.mode}")
    if len(replay.mine_positions) != replay.mines:
        raise ValueError("replay mine count does not match mine_positions")
    unique = {tuple(coord) for coord in replay.mine_positions}
    if len(unique) != replay.mines:
        raise ValueError("replay mine_positions contain duplicates")
    for row, col in unique:
        if not (0 <= row < replay.height and 0 <= col < replay.width):
            raise ValueError("replay mine position is out of bounds")
