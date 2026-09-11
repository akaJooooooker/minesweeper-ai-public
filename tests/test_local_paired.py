import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from minesweeper_ai.agent import HybridAgent
from minesweeper_ai.solver import ConstraintSolver
from scripts.compare_local_deployments import paired, read_completed


class LocalPairedTests(unittest.TestCase):
    def test_identical_agents_have_identical_outcomes_on_shared_board(self):
        agent = HybridAgent(ConstraintSolver(12))
        with patch("scripts.compare_local_deployments._AGENTS", (agent, agent)):
            for first_zero in (False, True):
                for index in (0, 1):
                    result = paired(("beginner", index, 123, first_zero))
                    self.assertEqual(result["reference"]["won"], result["candidate"]["won"])
                    self.assertEqual(result["reference"]["guesses"], result["candidate"]["guesses"])

    def test_resume_rejects_duplicate_and_missing_outcomes(self):
        row = dict(mode="evil", index=0, reference={"won": True}, candidate={"won": False})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"pairs.jsonl"
            path.write_text(json.dumps(row)+"\n", encoding="utf-8")
            self.assertEqual(read_completed(path), [row])
            path.write_text((json.dumps(row)+"\n")*2, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_completed(path)
            row["candidate"]["won"] = None
            path.write_text(json.dumps(row)+"\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_completed(path)

    def test_paired_seed_is_reproducible_and_separate_from_collection(self):
        from minesweeper_ai.local_selfplay import board_seed
        agent = HybridAgent(ConstraintSolver(12))
        with patch("scripts.compare_local_deployments._AGENTS", (agent, agent)):
            a = paired(("beginner", 0, 123, True))
            b = paired(("beginner", 0, 123, True))
        self.assertEqual(a["seed"], b["seed"])
        self.assertNotEqual(a["seed"], board_seed(123, "beginner", 0))


if __name__ == "__main__":
    unittest.main()
