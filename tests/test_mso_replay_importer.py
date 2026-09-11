import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from minesweeper_ai.mso_replay_importer import (
    ReplayImportError,
    convert_export,
    convert_game_payload,
    find_game_payloads,
    inspect_export,
)
from minesweeper_ai.replay import read_replays


def sample_payload() -> dict:
    return {
        "id": 5960972235,
        "userId": 7334132,
        "nickname": "must-not-leak",
        "sizeX": 2,
        "sizeY": 4,
        "mines": 2,
        "level": 1,
        "state": 3,
        "clicks": [
            {"x": 1, "y": 0, "type": 0, "time": 100, "touchCells": []},
            {"x": 0, "y": 0, "type": 1, "time": 200, "touchCells": []},
            {"x": 1, "y": 1, "type": 2, "time": 250, "touchCells": []},
            {"x": 1, "y": 1, "type": 3, "time": 300, "touchCells": []},
        ],
        "cells": {"t": [9, 0, 0, 9, 0, 0, 0, 0], "o": [], "f": []},
    }


class MsoReplayImporterTests(unittest.TestCase):
    def test_converts_coordinates_actions_and_anonymizes(self) -> None:
        replay = convert_game_payload(sample_payload(), salt="dataset-secret", mine_type_values=[9])
        self.assertEqual((replay.height, replay.width, replay.mines), (2, 4, 2))
        self.assertEqual(replay.mine_positions, [[0, 0], [0, 3]])
        self.assertEqual([move.action for move in replay.moves], ["reveal", "flag", "chord"])
        self.assertEqual((replay.moves[0].row, replay.moves[0].col), (1, 0))
        serialized = json.dumps(asdict(replay))
        self.assertNotIn("5960972235", serialized)
        self.assertNotIn("7334132", serialized)
        self.assertNotIn("must-not-leak", serialized)

    def test_hash_is_stable_per_salt_but_changes_between_datasets(self) -> None:
        first = convert_game_payload(sample_payload(), salt="one", mine_type_values=[9])
        again = convert_game_payload(sample_payload(), salt="one", mine_type_values=[9])
        other = convert_game_payload(sample_payload(), salt="two", mine_type_values=[9])
        self.assertEqual(first.game_id, again.game_id)
        self.assertNotEqual(first.game_id, other.game_id)

    def test_rejects_missing_or_wrong_mine_truth(self) -> None:
        with self.assertRaisesRegex(ReplayImportError, "mine truth is absent"):
            convert_game_payload(sample_payload(), salt="x")
        with self.assertRaisesRegex(ReplayImportError, "decoded 6 mines"):
            convert_game_payload(sample_payload(), salt="x", mine_type_values=[0])

    def test_finds_nested_socket_response_game(self) -> None:
        wrapped = ["response", 1001, {"result": {"game": sample_payload()}}]
        self.assertEqual(list(find_game_payloads(wrapped)), [sample_payload()])
    def test_normalizes_live_init_game_event_without_identity(self) -> None:
        payload = sample_payload()
        metadata = {
            key: payload[key]
            for key in ("id", "userId", "nickname", "sizeX", "sizeY", "mines", "level", "state")
        }
        frame = ["response", [None, "InitGameEvent", [metadata, payload["cells"], payload["clicks"]]]]
        found = list(find_game_payloads(frame))
        self.assertEqual(len(found), 1)
        self.assertNotIn("userId", found[0])
        self.assertNotIn("nickname", found[0])
        replay = convert_game_payload(found[0], salt="safe", mine_type_values=[9])
        self.assertTrue(replay.won)
        self.assertEqual([move.action for move in replay.moves], ["reveal", "flag", "chord"])

    def test_current_site_loss_state(self) -> None:
        payload = sample_payload()
        payload["state"] = 2
        replay = convert_game_payload(payload, salt="safe", mine_type_values=[9])
        self.assertFalse(replay.won)


    def test_file_conversion_deduplicates_and_inspection_omits_identity(self) -> None:
        frame = '42["response",1001,{"game":' + json.dumps(sample_payload()) + "}]"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capture.txt"
            output = Path(directory) / "replays.jsonl"
            source.write_text(frame + "\n" + frame, encoding="utf-8")
            reports = inspect_export(source)
            stats = convert_export(source, output, salt="safe", mine_type_values=[9])
            replays = list(read_replays(output))
        self.assertEqual(len(reports), 2)
        self.assertNotIn("id", reports[0])
        self.assertEqual(stats.inputs, 2)
        self.assertEqual(stats.imported, 1)
        self.assertEqual(len(replays), 1)


if __name__ == "__main__":
    unittest.main()
