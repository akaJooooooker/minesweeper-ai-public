import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minesweeper_ai.agent import HybridAgent
from minesweeper_ai.local_feedback import load_feedback_replay
from minesweeper_ai.local_selfplay import _play_one, board_seed, collect
from minesweeper_ai.replay_deployment import ReplayDeploymentConfig
from minesweeper_ai.solver import ConstraintSolver


class LocalSelfplayTests(unittest.TestCase):
    def test_seed_is_independent_of_worker_scheduling_and_mode(self):
        values = [board_seed(17, mode, index) for mode in ("expert", "evil") for index in range(100)]
        self.assertEqual(len(set(values)), 200)
        self.assertEqual(board_seed(17, "evil", 7), board_seed(17, "evil", 7))
        self.assertNotEqual(board_seed(18, "evil", 7), board_seed(17, "evil", 7))

    def test_worker_records_terminal_reconstructable_games_and_never_overwrites(self):
        config = ReplayDeploymentConfig("replay-agent-deployment-v20", "test.pt", "standard",
                                        12, 0.0, "probability")
        agent = HybridAgent(ConstraintSolver(12))
        with tempfile.TemporaryDirectory() as directory, \
             patch("minesweeper_ai.local_selfplay._WORKER", (agent, config)):
            task = ("beginner", 0, 17, True, directory, 0)
            result = _play_one(task)
            payload = json.loads((Path(directory)/"episodes"/"beginner-000000.json").read_text())
            replay = load_feedback_replay(payload)
            self.assertIn(result["status"], ("won", "lost"))
            self.assertEqual(replay.won, result["status"] == "won")
            self.assertEqual(payload["collection"]["split"], "validation")
            self.assertEqual(payload["origin"], "headless_selfplay")
            self.assertTrue(all(move["decision"] is not None for move in payload["actions"]))
            self.assertEqual(result["examples"], [])
            with self.assertRaises(FileExistsError):
                _play_one(task)

    def test_invalid_runs_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"invalid"
            for modes, workers in ((["evil", "evil"], 4), (["evil"], 0)):
                with self.assertRaises(ValueError):
                    collect(Path("missing.json"), output, modes=modes, games_per_mode=1, workers=workers)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
