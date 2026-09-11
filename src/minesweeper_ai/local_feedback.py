"""Save local episodes and audit them into the existing replay training format.

This module belongs to the recorder/training side. Its hidden-board data is
never passed to the live decision worker.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .game import GameStatus, MinesweeperGame
from .local_game import LocalGameSession, TERMINAL
from .replay import ReplayAuditor, ReplayGame, ReplayMove, write_replay_examples
from .solver import ConstraintSolver

SCHEMA = "local-minesweeper-feedback-v1"


def build_feedback(session: LocalGameSession, *, seed: int, model: dict | None = None,
                   origin: str = "local_gui") -> dict:
    game = session.game
    if game.status not in TERMINAL or not game.mines_placed:
        raise ValueError("Only completed games can become training feedback")
    mines = [[r, c] for r in range(game.height) for c in range(game.width)
             if game.is_mine((r, c))]
    identity = json.dumps([game.width, game.height, mines], separators=(",", ":"))
    board_id = hashlib.sha256(identity.encode()).hexdigest()
    actors = {move["source"] for move in session.history}
    actor = "v44" if actors == {"V44"} else ("mixed" if "V44" in actors else "manual")
    replay = ReplayGame(
        game_id="local-" + board_id, width=game.width, height=game.height,
        mines=game.mine_count, mode="standard", source="simulator_local_" + actor,
        won=game.status == GameStatus.WON, mine_positions=mines,
        moves=[ReplayMove(move["action"], move["row"], move["col"],
                          round(move["seconds"] * 1000)) for move in session.history],
    )
    return dict(
        schema=SCHEMA, schema_version=2, origin=origin, seed=seed,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        first_click_zero=game.first_click_zero, **session.summary(),
        board_fingerprint=board_id, model=model,
        last_pre_observation=session.last_pre_observation,
        actions=session.history, replay=asdict(replay),
        training_status="recorded_not_trained",
    )


def save_feedback(payload: dict, directory: Path) -> Path:
    if payload.get("schema") != SCHEMA:
        raise ValueError("Unsupported feedback schema")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = directory / f"{stamp}-{payload['status']}-{uuid4().hex}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")
    temporary.replace(path)
    return path


def load_feedback_replay(payload: dict) -> ReplayGame:
    if payload.get("schema") != SCHEMA:
        raise ValueError("Unsupported feedback schema")
    raw = dict(payload["replay"])
    raw["moves"] = [ReplayMove(**move) for move in raw["moves"]]
    replay = ReplayGame(**raw)
    # Reject truncated/corrupt episodes before producing outcome labels.
    game = MinesweeperGame(replay.width, replay.height, replay.mines,
                          mine_positions=[tuple(cell) for cell in replay.mine_positions])
    for move in replay.moves:
        if game.status in TERMINAL:
            raise ValueError("Feedback contains actions after the game ended")
        if move.action == "reveal":
            game.reveal(move.coord)
        elif move.action == "flag":
            game.toggle_flag(move.coord)
        elif move.action == "chord":
            game.chord(move.coord)
        else:
            raise ValueError("Unsupported feedback move")
    if (game.status not in TERMINAL or game.status.value != payload["status"]
            or replay.won != (game.status == GameStatus.WON)):
        raise ValueError("Feedback outcome does not match reconstructed game")
    return replay


def audit_feedback(input_path: Path, output: Path, *, teacher_cells: int = 40) -> dict:
    input_path, output = Path(input_path), Path(output)
    paths = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.json"))
    auditor = ReplayAuditor(ConstraintSolver(teacher_cells))
    classes = Counter()
    outcomes = Counter()
    ids = set()
    replay_count = 0
    skipped_openings = 0

    def examples():
        nonlocal replay_count, skipped_openings
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != SCHEMA:
                continue
            replay = load_feedback_replay(payload)
            replay_count += 1
            outcomes[payload["status"]] += 1
            ids.add(replay.game_id)
            for example in auditor.audit(replay):
                # Lazy safe/zero placement is a game rule, not a learned guess.
                if not any(cell >= 0 for row in example.observation for cell in row):
                    skipped_openings += 1
                    continue
                classes[example.classification] += 1
                yield example

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        count = write_replay_examples(examples(), temporary)
        if replay_count == 0:
            raise ValueError("No completed local feedback episodes found")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return dict(replays=replay_count, unique_boards=len(ids), outcomes=dict(outcomes),
                examples=count, classifications=dict(classes), skipped_openings=skipped_openings,
                output=str(output.resolve()), training_performed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit local feedback for later replay training")
    parser.add_argument("--input", type=Path, default=Path("data/local-feedback"))
    parser.add_argument("--output", type=Path, default=Path("data/local-feedback-examples.jsonl"))
    parser.add_argument("--teacher-cells", type=int, default=40)
    args = parser.parse_args(argv)
    print(json.dumps(audit_feedback(args.input, args.output, teacher_cells=args.teacher_cells),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
