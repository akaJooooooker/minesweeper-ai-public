"""Feedback must retain public-board probabilities, not hindsight labels."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from minesweeper_ai.game import UNKNOWN, MinesweeperGame
from minesweeper_ai.local_game import LocalGameSession
from minesweeper_ai.local_feedback import (
    audit_feedback, build_feedback, load_feedback_replay, save_feedback,
)
from minesweeper_ai.replay import read_replay_examples


class LocalFeedbackTests(unittest.TestCase):
    def episode(self, *, won=False):
        session = LocalGameSession(MinesweeperGame(2, 2, 1, mine_positions={(0, 0)}))
        session.apply("reveal", (1, 1), source="V44",
                      decision=dict(mine_probability=0.0, exact=True, reason="safe opening"))
        session.apply("reveal", (0, 1) if won else (0, 0), source="V44",
                      decision=dict(mine_probability=1 / 3, exact=True, reason="minimum risk"))
        if won:
            session.apply("reveal", (1, 0), source="V44",
                          decision=dict(mine_probability=0.5, exact=True, reason="minimum risk"))
        return build_feedback(session, seed=7, model={"name": "V44"})

    def test_round_trip_retains_decision_and_pre_loss_board(self):
        payload = self.episode()
        with tempfile.TemporaryDirectory() as directory:
            path = save_feedback(payload, Path(directory))
            stored = json.loads(path.read_text(encoding="utf-8"))
        replay = load_feedback_replay(stored)
        self.assertFalse(replay.won)
        self.assertEqual(replay.source, "simulator_local_v44")
        self.assertEqual(replay.moves[-1].coord, (0, 0))
        self.assertEqual(stored["last_pre_observation"][0][0], UNKNOWN)
        self.assertAlmostEqual(stored["actions"][-1]["decision"]["mine_probability"], 1 / 3)
        self.assertEqual(stored["training_status"], "recorded_not_trained")

    def test_audit_guess_uses_pre_move_probability_not_exploded_mine(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = save_feedback(self.episode(), directory)
            output = directory / "examples.jsonl"
            report = audit_feedback(path, output)
            examples = list(read_replay_examples(output))
        self.assertFalse(report["training_performed"])
        self.assertEqual(report["skipped_openings"], 1)
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.classification, "NECESSARY_GUESS")
        self.assertTrue(example.teacher_exact)
        self.assertAlmostEqual(example.mine_probability, 1 / 3)
        self.assertAlmostEqual(example.probabilities[0][0], 1 / 3)
        self.assertEqual(example.risk_mask[0][0], 1)
        self.assertEqual(example.policy_weight, 0)
        self.assertEqual(example.observation[0][0], UNKNOWN)

    def test_wins_and_losses_saved_without_overwrite_and_share_board_identity(self):
        loss, win = self.episode(), self.episode(won=True)
        self.assertEqual(loss["board_fingerprint"], win["board_fingerprint"])
        self.assertEqual(loss["replay"]["game_id"], win["replay"]["game_id"])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = [save_feedback(payload, directory) for payload in (loss, win)]
            self.assertNotEqual(*paths)
            self.assertTrue(all(path.is_file() for path in paths))
            report = audit_feedback(directory, directory / "examples.jsonl")
        self.assertEqual(report["outcomes"], {"lost": 1, "won": 1})
        self.assertEqual(report["unique_boards"], 1)
        self.assertEqual(report["replays"], 2)

    def test_incomplete_and_corrupt_outcomes_rejected(self):
        session = LocalGameSession(MinesweeperGame(2, 2, 1, mine_positions={(0, 0)}))
        session.apply("reveal", (1, 1))
        with self.assertRaises(ValueError):
            build_feedback(session, seed=7)
        valid = self.episode()
        for corruption in ("truncated", "wrong_outcome", "post_terminal"):
            payload = deepcopy(valid)
            if corruption == "truncated":
                payload["replay"]["moves"].pop()
            elif corruption == "wrong_outcome":
                payload["replay"]["won"] = True
            else:
                payload["replay"]["moves"].append(payload["replay"]["moves"][-1])
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                load_feedback_replay(payload)

    def test_empty_audit_does_not_replace_existing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "examples.jsonl"
            output.write_text("existing dataset", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_feedback(Path(directory), output)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing dataset")
            self.assertFalse(output.with_name(output.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
