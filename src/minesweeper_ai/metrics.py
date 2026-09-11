"""Competitive Minesweeper board and play metrics."""

from __future__ import annotations

from dataclasses import dataclass

from .game import Coord, MinesweeperGame


@dataclass(frozen=True)
class ThreeBV:
    value: int
    openings: int
    isolated_numbers: int


def calculate_3bv(game: MinesweeperGame) -> ThreeBV:
    """Return the board's 3BV after mines have been placed.

    Each connected zero opening counts once. A numbered safe cell counts once
    only when it does not border any opening.
    """
    if not game.mines_placed:
        raise ValueError("3BV requires a board with placed mines")

    safe = {
        (row, col)
        for row in range(game.height)
        for col in range(game.width)
        if not game.is_mine((row, col))
    }
    zeroes = {
        coord
        for coord in safe
        if all(not game.is_mine(neighbour) for neighbour in game.neighbours(coord))
    }

    remaining = set(zeroes)
    covered_by_openings: set[Coord] = set()
    openings = 0
    while remaining:
        openings += 1
        stack = [remaining.pop()]
        while stack:
            coord = stack.pop()
            covered_by_openings.add(coord)
            for neighbour in game.neighbours(coord):
                if neighbour in safe:
                    covered_by_openings.add(neighbour)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)

    isolated_numbers = len(safe - covered_by_openings)
    return ThreeBV(
        value=openings + isolated_numbers,
        openings=openings,
        isolated_numbers=isolated_numbers,
    )


def three_bv_per_second(three_bv: int, seconds: float) -> float:
    if three_bv < 0:
        raise ValueError("three_bv must be non-negative")
    if seconds <= 0.0:
        raise ValueError("seconds must be positive")
    return three_bv / seconds


def index_of_efficiency(three_bv: int, clicks: int) -> float:
    """Return conventional IOE: 3BV divided by total clicks/actions."""
    if three_bv < 0:
        raise ValueError("three_bv must be non-negative")
    if clicks <= 0:
        raise ValueError("clicks must be positive")
    return three_bv / clicks
