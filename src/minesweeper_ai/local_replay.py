"""Seekable public-board replay, with sparse deltas and bounded checkpoints."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from .game import GameStatus, MinesweeperGame
from .local_game import LocalGameSession, TERMINAL


@dataclass(frozen=True)
class ReplayBoard:
    width: int
    height: int
    mine_count: int
    observation: tuple[tuple[int, ...], ...]
    status: GameStatus
    exploded: tuple[int, int] | None
    mines: frozenset[tuple[int, int]]

    @property
    def remaining_mines(self):
        return self.mine_count - sum(cell == -2 for row in self.observation for cell in row)

    def is_mine(self, coord):
        return coord in self.mines


@dataclass(frozen=True)
class ReplayStep:
    changes: tuple[tuple[int, int, int, int], ...]
    status: GameStatus
    exploded: tuple[int, int] | None
    summary: dict
    action: dict | None


class ReplayTimeline:
    CHECKPOINT_INTERVAL = 32

    def __init__(self, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("重播紀錄必須是 JSON 物件")
        self.payload = deepcopy(payload)
        raw = payload.get("replay", {})
        if not isinstance(raw, dict):
            raise ValueError("重播資料格式錯誤")
        self.width = int(payload.get("cols", raw.get("width", 0)))
        self.height = int(payload.get("rows", raw.get("height", 0)))
        self.mine_count = int(payload.get("mines", raw.get("mines", -1)))
        if not (2 <= self.width <= 80 and 2 <= self.height <= 60):
            raise ValueError("重播盤面尺寸不在支援範圍內")
        positions = raw.get("mine_positions", payload.get("mine_positions"))
        if positions is not None:
            game = MinesweeperGame(self.width, self.height, self.mine_count,
                                  mine_positions=[tuple(cell) for cell in positions])
        else:
            if "seed" not in payload:
                raise ValueError("紀錄缺少種子或雷位，無法重建棋盤")
            game = MinesweeperGame(self.width, self.height, self.mine_count,
                                  seed=int(payload["seed"]),
                                  first_click_zero=bool(payload.get("first_click_zero", True)))
        actions = payload.get("actions")
        if actions is None:
            actions = [{**move, "seconds": move.get("time_ms", 0)/1000}
                       for move in raw.get("moves", [])]
        if not isinstance(actions, list):
            raise ValueError("重播操作資料格式錯誤")
        if raw.get("moves") is not None and len(raw["moves"]) != len(actions):
            raise ValueError("重播操作數與紀錄不一致")
        session = LocalGameSession(game, clock=lambda: 0.0)
        before = game.observation
        self.steps = [ReplayStep((), game.status, None, session.summary(), None)]
        self.checkpoints = {0: before}
        previous_seconds = 0.0
        for index, move in enumerate(actions, 1):
            if not isinstance(move, dict):
                raise ValueError("重播操作必須是物件")
            decision = move.get("decision")
            if decision is not None:
                if not isinstance(decision, dict):
                    raise ValueError("模型判斷紀錄格式錯誤")
                probability = decision.get("mine_probability")
                if probability is not None and (not isinstance(probability, (float, int))
                        or not math.isfinite(probability) or not 0 <= probability <= 1):
                    raise ValueError("紀錄中的雷率無效")
            if game.status in TERMINAL:
                raise ValueError("紀錄在遊戲結束後仍有動作")
            action, row, col = move["action"], int(move["row"]), int(move["col"])
            seconds = float(move.get("seconds", move.get("time_ms", 0)/1000))
            if not math.isfinite(seconds) or seconds < previous_seconds:
                raise ValueError("重播時間必須有效且不可倒退")
            if not (0 <= row < self.height and 0 <= col < self.width):
                raise ValueError("重播動作超出棋盤")
            if raw.get("moves") is not None:
                recorded = raw["moves"][index-1]
                if (action, row, col) != (recorded["action"], recorded["row"], recorded["col"]):
                    raise ValueError("重播操作與紀錄不一致")
            session.apply(action, (row, col), source=move.get("source", "replay"),
                          decision=move.get("decision"))
            observation = game.observation
            summary = session.summary()
            summary["elapsed_seconds"] = seconds
            summary["three_bv_per_second"] = session.completed_bv/seconds if seconds else None
            if not any(cell >= 0 for line in observation for cell in line):
                summary["three_bv"] = None
            changes = tuple((r,c,before[r][c],observation[r][c])
                            for r in range(self.height) for c in range(self.width)
                            if before[r][c] != observation[r][c])
            detail = {**move, "action":action, "row":row, "col":col, "seconds":seconds,
                      "effective":session.history[-1]["effective"]}
            self.steps.append(ReplayStep(changes,game.status,game.exploded,summary,detail))
            if index % self.CHECKPOINT_INTERVAL == 0:
                self.checkpoints[index] = observation
            before, previous_seconds = observation, seconds
        if payload.get("status") is not None and game.status.value != payload["status"]:
            raise ValueError("重建結果與紀錄中的遊戲狀態不一致")
        if "won" in raw and raw["won"] != (game.status == GameStatus.WON):
            raise ValueError("重建結果與紀錄中的勝敗不一致")
        self.mines = frozenset((r,c) for r in range(self.height) for c in range(self.width)
                               if game.mines_placed and game.is_mine((r,c)))
        self._cached_index = 0
        self._cached_observation = self.checkpoints[0]

    def __len__(self):
        return len(self.steps)-1

    @classmethod
    def from_session(cls, session, *, seed):
        game = session.game
        payload = dict(schema_version=1, seed=seed, first_click_zero=game.first_click_zero,
                       **session.summary(), actions=deepcopy(session.history))
        if game.mines_placed:
            payload["mine_positions"] = [[r,c] for r in range(game.height) for c in range(game.width)
                                         if game.is_mine((r,c))]
        return cls(payload)

    def board_at(self, index):
        if not 0 <= index <= len(self):
            raise IndexError("Replay position is outside the timeline")
        checkpoint_index = index // self.CHECKPOINT_INTERVAL * self.CHECKPOINT_INTERVAL
        if abs(index-self._cached_index) < index-checkpoint_index:
            start, initial = self._cached_index, self._cached_observation
        else:
            start, initial = checkpoint_index, self.checkpoints[checkpoint_index]
        rows = [list(row) for row in initial]
        if start <= index:
            for position in range(start+1,index+1):
                for r,c,old,new in self.steps[position].changes:
                    rows[r][c] = new
        else:
            for position in range(start,index,-1):
                for r,c,old,new in self.steps[position].changes:
                    rows[r][c] = old
        self._cached_index = index
        self._cached_observation = tuple(tuple(row) for row in rows)
        step = self.steps[index]
        return ReplayBoard(self.width,self.height,self.mine_count,self._cached_observation,
                           step.status,step.exploded,self.mines)

    def description(self,index):
        step = self.steps[index]
        if step.action is None:
            return f"開局前 · 0 / {len(self)} 步"
        move = step.action
        names = {"reveal":"左鍵開格","flag":"插旗／取消旗","chord":"連開"}
        text = f"第 {index} / {len(self)} 步 · {names[move['action']]} ({move['row']+1}, {move['col']+1})"
        decision = move.get("decision")
        if decision:
            probability = decision.get("mine_probability")
            if probability is not None:
                text += f" · 雷率 {probability:.2%}"
            text += " · 精確推理" if decision.get("exact") else " · 機率估計"
        if not move["effective"]:
            text += " · 無效動作"
        return text

    def reason(self,index):
        move = self.steps[index].action
        if not move:
            return "拖曳進度條，或逐步查看盤面。"
        decision = move.get("decision") or {}
        reason = decision.get("reason", "")
        translated = {
            "configured safe first click":"首點保證安全。",
            "solver-safe with maximum expected opening":"已證明安全；優先選擇可能展開更多格子的落點。",
            "minimum exact mine probability":"沒有確定安全格，選擇最低精確雷率的格子。",
            "minimum-risk cell inside clickable region":"在可點擊範圍選擇最低雷率的格子。",
            "minimum neural risk after exact deductions":"精確求解未完成，依學習到的風險估計選格。",
            "minimum neural risk with outcome policy":"結合風險估計與整局結果策略選格。",
        }
        return translated.get(reason, reason or "手動操作；紀錄沒有模型判斷理由。")
