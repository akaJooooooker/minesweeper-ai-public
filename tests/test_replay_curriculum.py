import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.replay import (
    ReplayGame,
    ReplayMove,
    read_replay_examples,
    write_replays,
)
from minesweeper_ai.replay_curriculum import build_risk_curriculum


class ReplayCurriculumTests(unittest.TestCase):
    def test_only_risk_is_weighted_for_teacher_student_gaps(self) -> None:
        replay = ReplayGame(
            game_id="gap-1",
            width=4,
            height=2,
            mines=2,
            mode="standard",
            source="simulator_teacher",
            won=False,
            mine_positions=[[0, 0], [0, 3]],
            moves=[
                ReplayMove("reveal", 1, 0),
                ReplayMove("reveal", 1, 1),
                ReplayMove("reveal", 1, 2),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            output = Path(directory) / "curriculum.jsonl"
            write_replays([replay], raw)
            stats = build_risk_curriculum(
                [raw], output, student_cells=1, teacher_cells=20
            )
            examples = list(read_replay_examples(output))
        self.assertGreater(stats.student_nonexact_guess_states, 0)
        self.assertGreater(stats.selected, 0)
        self.assertTrue(all(example.policy_weight == 0 for example in examples))
        self.assertTrue(all(example.value_weight == 0 for example in examples))
        self.assertTrue(all(sum(map(sum, example.risk_mask)) > 0 for example in examples))

    def test_teacher_must_be_larger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "larger"):
                build_risk_curriculum([], Path(directory) / "x", student_cells=4, teacher_cells=4)


if __name__ == "__main__":
    unittest.main()
