"""Local GUI session and honest, action-based play statistics.

The agent receives only observation and mine count. Ground truth below is used
solely by the game/statistics layer, never as an inference input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

from .game import Coord, GameStatus, MinesweeperGame
from .metrics import calculate_3bv

TERMINAL = (GameStatus.WON, GameStatus.LOST)


@dataclass
class ClickCounts:
    effective: dict[str, int] = field(default_factory=lambda: dict(left=0, right=0, chord=0))
    redundant: dict[str, int] = field(default_factory=lambda: dict(left=0, right=0, chord=0))

    @property
    def total(self) -> int:
        return sum(self.effective.values()) + sum(self.redundant.values())


class LocalGameSession:
    def __init__(self, game: MinesweeperGame, *, clock: Callable[[], float] = time.perf_counter):
        self.game = game
        self.clock = clock
        self.started: float | None = None
        self.finished: float | None = None
        self.paused_at: float | None = None
        self.paused_duration = 0.0
        self.clicks = ClickCounts()
        self.three_bv: int | None = None
        self.completed_bv = 0
        self.revision = 0
        self.history: list[dict] = []
        self.last_pre_observation = game.observation
        self._units: list[frozenset[Coord]] | None = None

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        end = (self.finished if self.finished is not None else
               self.paused_at if self.paused_at is not None else self.clock())
        return max(0.0, end - self.started - self.paused_duration)

    def pause_clock(self):
        if self.started is not None and self.finished is None and self.paused_at is None:
            self.paused_at = self.clock()

    def resume_clock(self):
        if self.paused_at is not None:
            self.paused_duration += self.clock() - self.paused_at
            self.paused_at = None

    def apply(self, action: str, coord: Coord, *, source: str = "human",
              decision: dict | None = None) -> bool:
        if action not in {"reveal", "flag", "chord"}:
            raise ValueError(f"unsupported action: {action}")
        if self.game.status in TERMINAL:
            return False
        before = self.game.observation
        self.last_pre_observation = before
        now = self.clock()
        if action == "flag":
            self.game.toggle_flag(coord)
        elif action == "chord":
            self.game.chord(coord)
        else:
            self.game.reveal(coord)
        changed = before != self.game.observation or self.game.status == GameStatus.LOST
        if self.started is None and self.game.status != GameStatus.READY:
            self.started = now
        if self.game.status in TERMINAL:
            self.finished = self.clock()
        category = {"reveal": "left", "flag": "right", "chord": "chord"}[action]
        bucket = self.clicks.effective if changed else self.clicks.redundant
        bucket[category] += 1
        self.revision += 1
        if self.game.mines_placed:
            if self._units is None:
                self.three_bv = calculate_3bv(self.game).value
                self._units = self._build_bv_units()
            observation = self.game.observation
            self.completed_bv = sum(
                all(observation[r][c] >= 0 for r, c in unit) for unit in self._units
            )
        self.history.append(dict(action=action, row=coord[0], col=coord[1],
                                 seconds=self.elapsed, effective=changed, source=source,
                                 decision=dict(decision) if decision is not None else None))
        return changed

    def _build_bv_units(self) -> list[frozenset[Coord]]:
        safe = {(r, c) for r in range(self.game.height) for c in range(self.game.width)
                if not self.game.is_mine((r, c))}
        zeroes = {cell for cell in safe if all(not self.game.is_mine(n)
                  for n in self.game.neighbours(cell))}
        remaining = set(zeroes)
        covered: set[Coord] = set()
        units = []
        while remaining:
            stack = [remaining.pop()]
            group: set[Coord] = set()
            while stack:
                cell = stack.pop()
                group.add(cell)
                covered.add(cell)
                for neighbour in self.game.neighbours(cell):
                    if neighbour in safe:
                        covered.add(neighbour)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
            units.append(frozenset(group))
        units.extend(frozenset({cell}) for cell in safe - covered)
        return units

    def summary(self) -> dict:
        seconds = self.elapsed
        return dict(status=self.game.status.value, rows=self.game.height, cols=self.game.width,
                    mines=self.game.mine_count, elapsed_seconds=seconds, three_bv=self.three_bv,
                    completed_bv=self.completed_bv,
                    three_bv_per_second=self.completed_bv / seconds if seconds > 0 else None,
                    efficiency=self.completed_bv / self.clicks.total if self.clicks.total else None,
                    effective_clicks=dict(self.clicks.effective),
                    redundant_clicks=dict(self.clicks.redundant), total_clicks=self.clicks.total)
