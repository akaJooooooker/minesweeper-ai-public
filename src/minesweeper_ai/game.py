"""A deterministic, testable Minesweeper environment with arbitrary dimensions."""

from __future__ import annotations

from collections import deque
from enum import Enum
import random
from typing import Iterable, Iterator, Sequence

Coord = tuple[int, int]

# Public observation values. Revealed clues always use 0..8.
UNKNOWN = -1
FLAGGED = -2


class GameStatus(str, Enum):
    READY = "ready"
    ACTIVE = "active"
    WON = "won"
    LOST = "lost"


class MinesweeperGame:
    """Minesweeper rules separated from any UI or learning framework.

    Mines are placed lazily on the first reveal unless ``mine_positions`` is
    supplied. ``first_click_zero`` excludes the first cell and its neighbours;
    otherwise ``first_click_safe`` excludes only the clicked cell.
    """

    def __init__(
        self,
        width: int,
        height: int,
        mines: int,
        *,
        seed: int | None = None,
        first_click_safe: bool = True,
        first_click_zero: bool = False,
        mine_positions: Iterable[Coord] | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if mines < 0 or mines >= width * height:
            raise ValueError("mines must be between 0 and cell_count - 1")

        self.width = width
        self.height = height
        self.mine_count = mines
        self.first_click_safe = first_click_safe or first_click_zero
        self.first_click_zero = first_click_zero
        self._rng = random.Random(seed)
        self._visible = [[UNKNOWN for _ in range(width)] for _ in range(height)]
        self._mines: set[Coord] | None = None
        self._clues: list[list[int]] | None = None
        self.status = GameStatus.READY
        self.exploded: Coord | None = None

        if mine_positions is not None:
            supplied = set(mine_positions)
            if len(supplied) != mines:
                raise ValueError("mine_positions count does not match mines")
            for coord in supplied:
                self._validate_coord(coord)
            self._set_mines(supplied)

    @property
    def observation(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self._visible)

    @property
    def mines_placed(self) -> bool:
        return self._mines is not None

    @property
    def remaining_mines(self) -> int:
        flags = sum(cell == FLAGGED for row in self._visible for cell in row)
        return self.mine_count - flags

    @property
    def covered_cells(self) -> tuple[Coord, ...]:
        return tuple(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if self._visible[row][col] == UNKNOWN
        )

    @property
    def flagged_cells(self) -> tuple[Coord, ...]:
        return tuple(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if self._visible[row][col] == FLAGGED
        )

    def clone(self) -> "MinesweeperGame":
        """Return an independent copy of the complete game state.

        This is intended for tests and dataset-generation rollouts. Public
        agents still receive only :attr:`observation`; cloning does not expose
        hidden mines through the inference API.
        """
        clone = MinesweeperGame(
            self.width,
            self.height,
            self.mine_count,
            first_click_safe=self.first_click_safe,
            first_click_zero=self.first_click_zero,
            mine_positions=self._mines,
        )
        clone._visible = [row.copy() for row in self._visible]
        clone.status = self.status
        clone.exploded = self.exploded
        clone._rng.setstate(self._rng.getstate())
        return clone

    def neighbours(self, coord: Coord) -> Iterator[Coord]:
        row, col = coord
        for next_row in range(max(0, row - 1), min(self.height, row + 2)):
            for next_col in range(max(0, col - 1), min(self.width, col + 2)):
                if (next_row, next_col) != coord:
                    yield next_row, next_col

    def reveal(self, coord: Coord) -> frozenset[Coord]:
        """Reveal a cell and return every cell opened by zero expansion."""
        self._validate_coord(coord)
        if self.status in (GameStatus.WON, GameStatus.LOST):
            return frozenset()

        row, col = coord
        if self._visible[row][col] == FLAGGED:
            return frozenset()
        if self._visible[row][col] != UNKNOWN:
            return frozenset()

        if self._mines is None:
            self._place_mines(coord)
        if self.status == GameStatus.READY:
            self.status = GameStatus.ACTIVE

        assert self._mines is not None
        assert self._clues is not None
        if coord in self._mines:
            self.status = GameStatus.LOST
            self.exploded = coord
            return frozenset({coord})

        opened: set[Coord] = set()
        queue: deque[Coord] = deque([coord])
        while queue:
            current = queue.popleft()
            current_row, current_col = current
            if self._visible[current_row][current_col] != UNKNOWN:
                continue
            if current in self._mines:
                continue
            clue = self._clues[current_row][current_col]
            self._visible[current_row][current_col] = clue
            opened.add(current)
            if clue == 0:
                for neighbour in self.neighbours(current):
                    n_row, n_col = neighbour
                    if self._visible[n_row][n_col] == UNKNOWN:
                        queue.append(neighbour)

        self._update_win_status()
        return frozenset(opened)

    def toggle_flag(self, coord: Coord) -> bool:
        """Toggle an unknown cell's flag and return whether it is now flagged."""
        self._validate_coord(coord)
        if self.status in (GameStatus.WON, GameStatus.LOST):
            return False
        row, col = coord
        if self._visible[row][col] == UNKNOWN:
            self._visible[row][col] = FLAGGED
            return True
        if self._visible[row][col] == FLAGGED:
            self._visible[row][col] = UNKNOWN
            return False
        return False

    def chord(self, coord: Coord) -> frozenset[Coord]:
        """Reveal unknown neighbours when the adjacent flag count matches."""
        self._validate_coord(coord)
        row, col = coord
        clue = self._visible[row][col]
        if clue < 0 or self.status in (GameStatus.WON, GameStatus.LOST):
            return frozenset()
        neighbours = tuple(self.neighbours(coord))
        flag_count = sum(self._visible[r][c] == FLAGGED for r, c in neighbours)
        if flag_count != clue:
            return frozenset()
        opened: set[Coord] = set()
        for neighbour in neighbours:
            n_row, n_col = neighbour
            if self._visible[n_row][n_col] == UNKNOWN:
                opened.update(self.reveal(neighbour))
                if self.status == GameStatus.LOST:
                    break
        return frozenset(opened)

    def is_mine(self, coord: Coord) -> bool:
        """Ground-truth helper intended for tests and dataset generation only."""
        if self._mines is None:
            raise RuntimeError("mines are not placed before the first reveal")
        self._validate_coord(coord)
        return coord in self._mines

    def _place_mines(self, first_click: Coord) -> None:
        excluded = {first_click} if self.first_click_safe else set()
        if self.first_click_zero:
            excluded.update(self.neighbours(first_click))
        candidates = [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if (row, col) not in excluded
        ]
        if self.mine_count > len(candidates):
            raise ValueError("too many mines for the requested first-click rule")
        self._set_mines(set(self._rng.sample(candidates, self.mine_count)))

    def _set_mines(self, mines: set[Coord]) -> None:
        self._mines = mines
        self._clues = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) in mines:
                    self._clues[row][col] = -1
                else:
                    self._clues[row][col] = sum(
                        neighbour in mines for neighbour in self.neighbours((row, col))
                    )

    def _update_win_status(self) -> None:
        if self._mines is None or self.status == GameStatus.LOST:
            return
        opened = sum(cell >= 0 for row in self._visible for cell in row)
        if opened == self.width * self.height - self.mine_count:
            self.status = GameStatus.WON

    def _validate_coord(self, coord: Coord) -> None:
        row, col = coord
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(f"coordinate out of bounds: {coord}")


def validate_observation(observation: Sequence[Sequence[int]]) -> tuple[int, int]:
    """Validate a rectangular public board and return ``(height, width)``."""
    if not observation or not observation[0]:
        raise ValueError("observation must not be empty")
    width = len(observation[0])
    if any(len(row) != width for row in observation):
        raise ValueError("observation must be rectangular")
    if any(cell < FLAGGED or cell > 8 for row in observation for cell in row):
        raise ValueError("observation contains an unsupported cell value")
    return len(observation), width

