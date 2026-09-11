"""Decision and mouse-control primitives for desktop Minesweeper play."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import sys
from typing import Literal, Mapping, Sequence

from .agent import HybridAgent
from .game import FLAGGED, UNKNOWN, Coord
from .solver import Analysis
from .windows_coordinates import physical_cursor_position, set_physical_cursor_position


@dataclass(frozen=True)
class LiveAction:
    action: Literal["reveal", "flag", "chord", "stuck"]
    coord: Coord | None
    reason: str
    mine_probability: float | None
    exact: bool


@dataclass(frozen=True)
class _ChordPlan:
    coord: Coord
    missing_mines: tuple[Coord, ...]
    safe_cells: int
    click_saving: int


class LiveDecisionEngine:
    """Turn a public board observation into one safe, visible UI action."""

    def __init__(self, agent: HybridAgent, *, use_flags: bool = False) -> None:
        self.agent = agent
        self.use_flags = use_flags
        self.trusted_flags: set[Coord] = set()

    def reset(self) -> None:
        self.trusted_flags.clear()

    def next_action(
        self,
        observation: Sequence[Sequence[int]],
        total_mines: int,
        *,
        allow_guess: bool,
        allowed: Sequence[Sequence[bool]] | None = None,
    ) -> LiveAction:
        height = len(observation)
        width = len(observation[0])
        allowed_grid = allowed or tuple(tuple(True for _ in row) for row in observation)
        self.trusted_flags.intersection_update(
            (row, col)
            for row in range(height)
            for col in range(width)
            if observation[row][col] == FLAGGED
        )

        decision = self.agent.choose_move(
            observation,
            total_mines,
            allow_guess=allow_guess,
        )
        analysis = decision.analysis
        if analysis is not None:
            if analysis.contradiction:
                return LiveAction("stuck", None, decision.reason, None, analysis.exact)
            plan = (
                _best_profitable_chord_plan(
                    observation, allowed_grid, self.trusted_flags, analysis
                )
                if self.use_flags else None
            )
            if plan is not None and plan.missing_mines:
                coord = plan.missing_mines[0]
                self.trusted_flags.add(coord)
                return LiveAction(
                    "flag",
                    coord,
                    f"profitable chord setup: opens {plan.safe_cells}, saves {plan.click_saving} clicks",
                    1.0,
                    analysis.exact,
                )
            if plan is not None:
                return LiveAction(
                    "chord",
                    plan.coord,
                    f"profitable chord: opens {plan.safe_cells}, saves {plan.click_saving} clicks",
                    0.0,
                    True,
                )
            candidates = [
                (row, col)
                for row in range(height)
                for col in range(width)
                if observation[row][col] == UNKNOWN and allowed_grid[row][col]
            ]
            safe = [coord for coord in candidates if coord in analysis.safe]
            if safe:
                coord = max(
                    safe,
                    key=lambda item: _expected_reveal_score(item, observation, analysis),
                )
                return LiveAction(
                    "reveal", coord, "solver-safe with maximum expected opening", 0.0,
                    analysis.exact,
                )
            if allow_guess and candidates:
                # The agent has already blended learned risk and outcome policy.
                # Preserve that decision when the target is actually clickable;
                # analysis.probabilities contains only the solver's raw estimates.
                if decision.action == "reveal" and decision.coord in candidates:
                    return LiveAction("reveal", decision.coord, decision.reason,
                                      decision.mine_probability, decision.exact)
                coord = min(
                    candidates,
                    key=lambda item: (
                        analysis.probabilities.get(item, 1.0),
                        -_information_score(item, observation),
                        item,
                    ),
                )
                return LiveAction(
                    "reveal",
                    coord,
                    "minimum-risk cell inside clickable region",
                    analysis.probabilities.get(coord),
                    analysis.exact,
                )

        if (decision.coord is not None and allowed_grid[decision.coord[0]][decision.coord[1]]
                and (allow_guess or decision.mine_probability == 0.0)):
            return LiveAction(
                "reveal",
                decision.coord,
                decision.reason,
                decision.mine_probability,
                decision.exact,
            )
        return LiveAction("stuck", None, decision.reason, None, decision.exact)


def detect_region_reset(
    previous: Sequence[Sequence[int]] | None,
    current: Sequence[Sequence[int]],
) -> bool:
    if previous is None:
        return False
    return any(
        before >= 0 and after == UNKNOWN
        for before_row, after_row in zip(previous, current)
        for before, after in zip(before_row, after_row)
    )


def _best_profitable_chord_plan(
    observation: Sequence[Sequence[int]],
    allowed: Sequence[Sequence[bool]],
    trusted_flags: set[Coord],
    analysis: Analysis,
) -> _ChordPlan | None:
    height = len(observation)
    width = len(observation[0])
    candidates: list[_ChordPlan] = []
    for row in range(height):
        for col in range(width):
            clue = observation[row][col]
            if clue <= 0:
                continue
            neighbours = set(_neighbours(row, col, height, width))
            flags = {
                coord for coord in neighbours
                if observation[coord[0]][coord[1]] == FLAGGED
            }
            if not flags <= trusted_flags:
                continue
            covered = {
                coord for coord in neighbours
                if observation[coord[0]][coord[1]] == UNKNOWN
            }
            proven_mines = covered & analysis.mines
            proven_safe = covered & analysis.safe
            if not covered or covered != proven_mines | proven_safe:
                continue
            if len(flags) + len(proven_mines) != clue:
                continue
            if not all(allowed[r][c] for r, c in covered):
                continue
            cost = len(proven_mines) + 1
            saving = len(proven_safe) - cost
            if saving > 0:
                candidates.append(_ChordPlan(
                    (row, col), tuple(sorted(proven_mines)), len(proven_safe), saving
                ))
    return max(
        candidates,
        key=lambda plan: (
            plan.click_saving,
            plan.safe_cells,
            -len(plan.missing_mines),
            plan.coord,
        ),
        default=None,
    )


def _expected_reveal_score(
    coord: Coord,
    observation: Sequence[Sequence[int]],
    analysis: Analysis,
) -> tuple[float, int, int, Coord]:
    """Rank safe cells by estimated chance and size of a zero expansion."""
    height = len(observation)
    width = len(observation[0])
    covered = [
        neighbour for neighbour in _neighbours(coord[0], coord[1], height, width)
        if observation[neighbour[0]][neighbour[1]] == UNKNOWN
    ]
    zero_probability = 1.0
    for neighbour in covered:
        probability = analysis.probabilities.get(neighbour, 0.5)
        if neighbour in analysis.safe:
            probability = 0.0
        elif neighbour in analysis.mines:
            probability = 1.0
        zero_probability *= 1.0 - probability
    expected_opening = zero_probability * (1 + len(covered))
    return expected_opening, len(covered), -_information_score(coord, observation), coord


def _information_score(coord: Coord, observation: Sequence[Sequence[int]]) -> int:
    height = len(observation)
    width = len(observation[0])
    return sum(
        observation[row][col] >= 0
        for row, col in _neighbours(coord[0], coord[1], height, width)
    )


def _neighbours(row: int, col: int, height: int, width: int):
    for row_delta in (-1, 0, 1):
        for col_delta in (-1, 0, 1):
            if row_delta == 0 and col_delta == 0:
                continue
            next_row, next_col = row + row_delta, col + col_delta
            if 0 <= next_row < height and 0 <= next_col < width:
                yield next_row, next_col


class WindowsMouse:
    LEFT_DOWN = 0x0002
    LEFT_UP = 0x0004
    RIGHT_DOWN = 0x0008
    RIGHT_UP = 0x0010

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("desktop mouse control is currently Windows-only")
        self.user32 = ctypes.windll.user32

    def position(self) -> tuple[int, int]:
        return physical_cursor_position()

    def move(self, coord: tuple[int, int]) -> None:
        set_physical_cursor_position(coord)

    def click(self, coord: tuple[int, int], *, button: Literal["left", "right"]) -> None:
        set_physical_cursor_position(coord)
        if button == "left":
            self.user32.mouse_event(self.LEFT_DOWN, 0, 0, 0, 0)
            self.user32.mouse_event(self.LEFT_UP, 0, 0, 0, 0)
        else:
            self.user32.mouse_event(self.RIGHT_DOWN, 0, 0, 0, 0)
            self.user32.mouse_event(self.RIGHT_UP, 0, 0, 0, 0)
