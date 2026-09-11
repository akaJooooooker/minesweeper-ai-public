"""Strict, privacy-preserving importer for minesweeper.online replay exports.

The importer intentionally consumes offline JSON exports.  It does not crawl the
website or automate account access.  The accepted payload is either a bare game
object, a JSON response wrapper, JSONL, or a captured Socket.IO text frame.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

from .replay import ReplayGame, ReplayMode, ReplayMove, write_replays


class ReplayImportError(ValueError):
    """Raised when an export cannot be converted without guessing."""


@dataclass(frozen=True)
class ImportStats:
    inputs: int
    imported: int
    moves: int
    standard: int
    no_guess: int
    wins: int


def read_export_values(path: str | Path) -> Iterator[Any]:
    """Read JSON, JSONL, or one-Socket.IO-frame-per-line text."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        yield json.loads(text)
        return
    except json.JSONDecodeError:
        pass

    parsed = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        value = _parse_json_or_socketio(line)
        if value is None:
            continue
        parsed += 1
        yield value
    if parsed == 0:
        raise ReplayImportError(f"{path}: no JSON payloads found")


def find_game_payloads(value: Any) -> Iterator[dict[str, Any]]:
    """Find game-shaped objects inside response envelopes without fixed nesting."""
    seen: set[int] = set()

    def visit(item: Any) -> Iterator[dict[str, Any]]:
        if isinstance(item, (dict, list)):
            identity = id(item)
            if identity in seen:
                return
            seen.add(identity)
        init_game = _init_game_payload(item)
        if init_game is not None:
            yield init_game
            return
        if isinstance(item, dict):
            if _looks_like_game(item):
                yield item
                return
            for child in item.values():
                yield from visit(child)
        elif isinstance(item, list):
            for child in item:
                yield from visit(child)

    yield from visit(value)


def convert_game_payload(
    payload: dict[str, Any],
    *,
    salt: str,
    mode: Literal["auto", "standard", "no_guess"] = "auto",
    mine_type_values: Sequence[int] | None = None,
    won: bool | None = None,
    source: str = "minesweeper_online_public",
) -> ReplayGame:
    """Convert one public export while omitting all player-identifying fields."""
    if not salt:
        raise ReplayImportError("a non-empty anonymization salt is required")

    original_id = _required(payload, ("id", "gameId", "game_id"), "game id")
    rows = _positive_int(_required(payload, ("sizeX", "height"), "sizeX/height"), "height")
    cols = _positive_int(_required(payload, ("sizeY", "width"), "sizeY/width"), "width")
    mine_count = _positive_int(_required(payload, ("mines", "mineCount"), "mines"), "mines")
    if mine_count >= rows * cols:
        raise ReplayImportError("mine count must be smaller than the board")

    replay_mode = _resolve_mode(payload, mode)
    replay_won = _resolve_won(payload, won)
    mine_positions = _extract_mines(payload, rows, cols, mine_count, mine_type_values)
    moves = _extract_moves(payload, rows, cols)
    if not moves:
        raise ReplayImportError("replay contains no supported moves")

    digest = hashlib.sha256(f"{salt}\0{source}\0{original_id}".encode("utf-8")).hexdigest()[:20]
    return ReplayGame(
        game_id=f"mso-{digest}",
        width=cols,
        height=rows,
        mines=mine_count,
        mode=replay_mode,
        source=source,
        won=replay_won,
        mine_positions=mine_positions,
        moves=moves,
    )


def convert_export(
    input_path: str | Path,
    output_path: str | Path,
    *,
    salt: str,
    mode: Literal["auto", "standard", "no_guess"] = "auto",
    mine_type_values: Sequence[int] | None = None,
    won: bool | None = None,
) -> ImportStats:
    replays: list[ReplayGame] = []
    inputs = 0
    seen: set[str] = set()
    for value in read_export_values(input_path):
        for payload in find_game_payloads(value):
            inputs += 1
            replay = convert_game_payload(
                payload,
                salt=salt,
                mode=mode,
                mine_type_values=mine_type_values,
                won=won,
            )
            if replay.game_id in seen:
                continue
            seen.add(replay.game_id)
            replays.append(replay)
    if inputs == 0:
        raise ReplayImportError("no game-shaped payload was found")
    write_replays(replays, output_path)
    return ImportStats(
        inputs=inputs,
        imported=len(replays),
        moves=sum(len(item.moves) for item in replays),
        standard=sum(item.mode == "standard" for item in replays),
        no_guess=sum(item.mode == "no_guess" for item in replays),
        wins=sum(item.won for item in replays),
    )


def inspect_export(path: str | Path) -> list[dict[str, Any]]:
    """Return non-identifying structural diagnostics for candidate payloads."""
    reports: list[dict[str, Any]] = []
    for value in read_export_values(path):
        for payload in find_game_payloads(value):
            cells = payload.get("cells")
            types = cells.get("t") if isinstance(cells, dict) else None
            histogram: dict[str, int] = {}
            if isinstance(types, list):
                for cell_type in types:
                    key = str(cell_type)
                    histogram[key] = histogram.get(key, 0) + 1
            clicks = payload.get("clicks", [])
            reports.append(
                {
                    "sizeX": payload.get("sizeX", payload.get("height")),
                    "sizeY": payload.get("sizeY", payload.get("width")),
                    "mines": payload.get("mines", payload.get("mineCount")),
                    "level": payload.get("level"),
                    "state": payload.get("state"),
                    "clicks": len(clicks) if isinstance(clicks, list) else None,
                    "cell_type_histogram": histogram,
                    "has_explicit_mines": any(
                        key in payload for key in ("minePositions", "mine_positions")
                    ),
                }
            )
    return reports


def _parse_json_or_socketio(line: str) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass
    # Socket.IO text packets start with a numeric packet code.  Namespaced
    # packets may contain text before their first JSON array/object.
    starts = [index for token in ("[", "{") if (index := line.find(token)) >= 0]
    if not starts:
        return None
    try:
        return json.loads(line[min(starts) :])
    except json.JSONDecodeError:
        return None


def _looks_like_game(item: dict[str, Any]) -> bool:
    has_size = ("sizeX" in item and "sizeY" in item) or ("height" in item and "width" in item)
    return (
        has_size
        and any(key in item for key in ("mines", "mineCount"))
        and isinstance(item.get("clicks"), list)
    )


def _init_game_payload(item: Any) -> dict[str, Any] | None:
    """Normalize the current Minesweeper Online ``InitGameEvent`` shape.

    A live Socket.IO response keeps metadata, board truth, and clicks in three
    separate array entries. Only fields needed for replay reconstruction are
    retained, so player identifiers and profile data never reach our output.
    """
    if not isinstance(item, list) or len(item) < 3 or item[1] != "InitGameEvent":
        return None
    arguments = item[2]
    if not isinstance(arguments, list) or len(arguments) < 3:
        return None
    metadata, cells, clicks = arguments[:3]
    if not isinstance(metadata, dict) or not isinstance(cells, dict) or not isinstance(clicks, list):
        return None
    required = ("id", "sizeX", "sizeY", "mines", "level", "state")
    if any(key not in metadata for key in required):
        return None
    return {
        "id": metadata["id"],
        "sizeX": metadata["sizeX"],
        "sizeY": metadata["sizeY"],
        "mines": metadata["mines"],
        "level": metadata["level"],
        "state": metadata["state"],
        "cells": {"t": cells.get("t")},
        "clicks": clicks,
    }


def _required(payload: dict[str, Any], keys: Sequence[str], label: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise ReplayImportError(f"missing {label}")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ReplayImportError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ReplayImportError(f"{label} must be an integer") from error
    if result <= 0 or result != value:
        raise ReplayImportError(f"{label} must be a positive integer")
    return result


def _resolve_mode(
    payload: dict[str, Any], mode: Literal["auto", "standard", "no_guess"]
) -> ReplayMode:
    if mode != "auto":
        return mode
    explicit = payload.get("mode")
    if explicit in ("standard", "no_guess"):
        return explicit
    level = payload.get("level")
    if isinstance(level, int) and 1 <= level <= 4:
        return "standard"
    if isinstance(level, int) and 11 <= level <= 15:
        return "no_guess"
    raise ReplayImportError("cannot infer mode; pass --mode standard or --mode no_guess")


def _resolve_won(payload: dict[str, Any], override: bool | None) -> bool:
    if override is not None:
        return override
    explicit = payload.get("won")
    if isinstance(explicit, bool):
        return explicit
    state = payload.get("state")
    if isinstance(state, int):
        # Current public game exports use 3 for a win and 2 for a loss.
        if state in (2, 3):
            return state == 3
    raise ReplayImportError("cannot infer outcome; pass --won or --lost")


def _extract_mines(
    payload: dict[str, Any],
    rows: int,
    cols: int,
    mine_count: int,
    mine_type_values: Sequence[int] | None,
) -> list[list[int]]:
    explicit = payload.get("minePositions", payload.get("mine_positions"))
    if explicit is not None:
        mines = [_coordinate(item, rows, cols, "mine position") for item in explicit]
    else:
        if not mine_type_values:
            raise ReplayImportError(
                "mine truth is absent; pass --mine-type after checking inspect output"
            )
        cells = payload.get("cells")
        types = cells.get("t") if isinstance(cells, dict) else None
        if not isinstance(types, list) or len(types) != rows * cols:
            raise ReplayImportError("cells.t must contain exactly sizeX*sizeY entries")
        accepted = set(mine_type_values)
        mines = [[index // cols, index % cols] for index, value in enumerate(types) if value in accepted]
    if len(mines) != mine_count:
        raise ReplayImportError(
            f"decoded {len(mines)} mines, but payload declares {mine_count}"
        )
    if len({tuple(coord) for coord in mines}) != mine_count:
        raise ReplayImportError("mine positions contain duplicates")
    return sorted(mines)


def _coordinate(value: Any, rows: int, cols: int, label: str) -> list[int]:
    if isinstance(value, dict):
        row, col = value.get("x", value.get("row")), value.get("y", value.get("col"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        row, col = value
    else:
        raise ReplayImportError(f"invalid {label}")
    if not isinstance(row, int) or isinstance(row, bool) or not isinstance(col, int) or isinstance(col, bool):
        raise ReplayImportError(f"invalid {label}")
    if not (0 <= row < rows and 0 <= col < cols):
        raise ReplayImportError(f"{label} is out of bounds")
    return [row, col]


def _extract_moves(payload: dict[str, Any], rows: int, cols: int) -> list[ReplayMove]:
    clicks = payload.get("clicks")
    if not isinstance(clicks, list):
        raise ReplayImportError("clicks must be an array")
    # MSO records the press/check phase of a chord as type 2 and the actual
    # board-changing chord as type 3. Training must ignore the former.
    action_map = {0: "reveal", 1: "flag", 3: "chord"}
    moves: list[ReplayMove] = []
    previous_time = 0
    for index, click in enumerate(clicks):
        if not isinstance(click, dict):
            raise ReplayImportError(f"click {index} is not an object")
        click_type = click.get("type")
        if click_type == 2:
            continue
        action = action_map.get(click_type)
        if action is None:
            raise ReplayImportError(f"click {index} has unsupported type {click_type!r}")
        row, col = _coordinate(click, rows, cols, f"click {index}")
        time_ms = click.get("time", 0)
        if not isinstance(time_ms, int) or isinstance(time_ms, bool) or time_ms < 0:
            raise ReplayImportError(f"click {index} has invalid time")
        if time_ms < previous_time:
            raise ReplayImportError("click times must be non-decreasing")
        previous_time = time_ms
        moves.append(ReplayMove(action, row, col, time_ms))
    return moves


def _mine_types(text: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("mine types must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one mine type is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="show only structural diagnostics")
    inspect_parser.add_argument("--input", required=True)

    convert_parser = subparsers.add_parser("convert", help="convert and anonymize exports")
    convert_parser.add_argument("--input", required=True)
    convert_parser.add_argument("--output", required=True)
    convert_parser.add_argument("--salt", required=True)
    convert_parser.add_argument("--mode", choices=("auto", "standard", "no_guess"), default="auto")
    convert_parser.add_argument("--mine-type", type=_mine_types)
    outcome = convert_parser.add_mutually_exclusive_group()
    outcome.add_argument("--won", action="store_const", const=True, dest="won")
    outcome.add_argument("--lost", action="store_const", const=False, dest="won")
    convert_parser.set_defaults(won=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(inspect_export(args.input), ensure_ascii=False, indent=2))
        return
    stats = convert_export(
        args.input,
        args.output,
        salt=args.salt,
        mode=args.mode,
        mine_type_values=args.mine_type,
        won=args.won,
    )
    print(json.dumps(asdict(stats), separators=(",", ":")))


if __name__ == "__main__":
    main()
