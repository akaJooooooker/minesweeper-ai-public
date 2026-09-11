"""On-demand coaching from pre-action public observations only."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from .game import FLAGGED, UNKNOWN
from .solver import ConstraintSolver


@dataclass(frozen=True)
class MoveAssessment:
    text: str
    mine_probability: float | None = None
    exact: bool = False
    best_coord: tuple[int, int] | None = None
    best_probability: float | None = None
    safer_count: int = 0


def assess_move(observation, total_mines, action, *, predict_risks: Callable | None = None,
                solver=None, fallback=None, neural_blend=0.65):
    """No mine layout, seed, future observation or terminal outcome is accepted."""
    row, col = action["row"], action["col"]
    height, width = len(observation), len(observation[0])
    kind = action["action"]
    if kind not in ("reveal", "flag", "chord") or not (0 <= row < height and 0 <= col < width):
        raise ValueError("Invalid replay action")
    value = observation[row][col]
    if not any(v >= 0 for line in observation for v in line):
        return MoveAssessment("首點前：安全性由開局保護規則決定，此處不作一般雷率比較。")
    if kind == "reveal" and value != UNKNOWN:
        return MoveAssessment("沒有開啟新格：此格已有旗子或已揭露。這次開格不會觸雷。")
    if kind == "flag" and value >= 0:
        return MoveAssessment("此格已揭露，插旗動作不會改變盤面。")
    targets = [(row, col)]
    if kind == "chord":
        neighbours = [(r,c) for r in range(max(0,row-1),min(height,row+2))
                      for c in range(max(0,col-1),min(width,col+2)) if (r,c)!=(row,col)]
        targets = [(r,c) for r,c in neighbours if observation[r][c] == UNKNOWN]
        if value < 0 or sum(observation[r][c] == FLAGGED for r,c in neighbours) != value or not targets:
            return MoveAssessment("連開未觸發：沒有目標格，或旗數與數字不符；不會開啟新格。")
    # A player's flag is a hypothesis, not ground-truth evidence of a mine.
    public = tuple(tuple(UNKNOWN if v == FLAGGED else v for v in line) for line in observation)
    analysis = (solver or ConstraintSolver(24)).analyse(public, total_mines)
    if not analysis.exact and not analysis.safe and not analysis.contradiction:
        stronger = (fallback or ConstraintSolver(40)).analyse(public, total_mines)
        if stronger.contradiction or stronger.exact or stronger.safe:
            analysis = stronger
    if analysis.contradiction:
        return MoveAssessment("公開數字與總雷數不一致，無法可靠估算；請檢查紀錄。")
    risks = dict(analysis.probabilities)
    source = "精確枚舉" if analysis.exact else "求解器近似"
    if not analysis.exact and predict_risks is not None:
        try:
            learned = predict_risks(public, total_mines)
            for coord, probability in learned.items():
                if coord not in analysis.safe and coord not in analysis.mines:
                    risks[coord] = neural_blend * probability + (1-neural_blend) * risks.get(coord, probability)
            source = "V44 混合估計"
        except Exception:
            source = "求解器近似（模型無法載入）"
    def risk(coord):
        if coord in analysis.safe:
            return 0.0
        if coord in analysis.mines:
            return 1.0
        return risks.get(coord)
    if kind == "chord":
        values = [risk(coord) for coord in targets]
        if any(p is None for p in values):
            return MoveAssessment("連開目標的風險資料不足。")
        if all(coord in analysis.safe for coord in targets):
            return MoveAssessment(f"連開 {len(targets)} 格：全部已證明安全，踩雷率 0%。", 0., True)
        return MoveAssessment(f"連開 {len(targets)} 格；最高單格雷率 {max(values):.2%}（{source}）。\n"
                              "這不是整次連開的踩雷率；格子風險可能相關。")
    coord = (row,col)
    probability = risk(coord)
    if probability is None:
        return MoveAssessment("此落點的雷率資料不足。")
    exact = analysis.exact or coord in analysis.safe or coord in analysis.mines
    label = "邏輯證明" if coord in analysis.safe or coord in analysis.mines else source
    prefix = "取消旗子的格子" if kind == "flag" and value == FLAGGED else "插旗的格子" if kind == "flag" else "落點"
    text = f"{prefix}雷率 {probability:.2%}（{label}）。"
    if kind == "flag":
        text += ("\n此格已證明是雷。" if probability == 1 and exact else
                 "\n此格已證明安全，不應保留旗子。" if probability == 0 and exact else
                 "\n尚未證明是雷；插旗本身不觸雷，但可能影響之後連開。")
        return MoveAssessment(text, probability, exact)
    clickable = [(r,c) for r in range(height) for c in range(width)
                 if observation[r][c] == UNKNOWN and risk((r,c)) is not None]
    best = min(clickable, key=lambda cell: (risk(cell), cell))
    minimum = risk(best)
    safer = sum(risk(cell) + 1e-9 < probability for cell in clickable)
    proven_safe = sum(cell in analysis.safe for cell in clickable)
    if probability == 0 and exact:
        text += "\n安全開格；是否最有效仍取決於後續資訊。"
    elif safer:
        text += f"\n有 {safer} 格{'更安全' if analysis.exact else '估計風險更低'}；最低為 ({best[0]+1}, {best[1]+1}) {minimum:.2%}。"
        if proven_safe:
            text += f"其中 {proven_safe} 格已證明安全。"
    else:
        text += "\n屬於目前最低雷率的選擇。" if analysis.exact else "\n屬於目前最低估計雷率的選擇。"
    if not proven_safe and probability > 0:
        text += "\n這步承擔猜測風險；踩雷結果本身不代表選錯。"
    text += "\n依當時數字與總雷數計算；玩家旗子未當作已知雷。"
    return MoveAssessment(text, probability, exact, best, minimum, safer)


def observed_effect(step):
    """Descriptive after-action effect, kept separate from pre-action inference."""
    opened = sum(old < 0 <= new for _,_,old,new in step.changes)
    if not step.action["effective"]:
        effect = "無效動作，盤面沒有變化。"
    elif step.action["action"] == "flag":
        effect = "更改旗子標記，未取得新的數字資訊。"
    else:
        effect = f"實際展開 {opened} 個安全格。"
    return effect + "\n這是事後效果，並非長期勝率或資訊價值評分。"
