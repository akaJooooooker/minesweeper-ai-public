import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.game import FLAGGED
from minesweeper_ai.replay import ReplayGame, ReplayMove, read_replay_examples, write_replays
from minesweeper_ai.replay_deployment_curriculum import build_deployment_curriculum


class ReplayDeploymentCurriculumTests(unittest.TestCase):
    def test_preserves_flags_and_labels_only_unknown_cells(self) -> None:
        replay = ReplayGame(
            game_id="deploy-1",
            width=4,
            height=2,
            mines=2,
            mode="standard",
            source="simulator_on_policy",
            won=False,
            mine_positions=[[0, 0], [0, 3]],
            moves=[
                ReplayMove("reveal", 1, 0),
                ReplayMove("flag", 0, 0),
                ReplayMove("reveal", 1, 1),
                ReplayMove("reveal", 1, 2),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            output = Path(directory) / "deploy.jsonl"
            write_replays([replay], raw)
            stats = build_deployment_curriculum(
                [raw], output, student_cells=1, teacher_cells=20
            )
            examples = list(read_replay_examples(output))
        self.assertGreater(stats.selected, 0)
        for example in examples:
            for row in range(len(example.observation)):
                for col in range(len(example.observation[0])):
                    if example.observation[row][col] == FLAGGED:
                        self.assertEqual(example.risk_mask[row][col], 0)
        self.assertTrue(all(example.policy_weight == 0 for example in examples))
        self.assertTrue(all(example.value_weight == 0 for example in examples))


if __name__ == "__main__":
    unittest.main()
