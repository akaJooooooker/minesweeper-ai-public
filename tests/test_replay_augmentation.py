import unittest
from dataclasses import replace

from minesweeper_ai.replay import ReplayTrainingExample
from minesweeper_ai.replay_augmentation import augment_example, transform_matrix


class ReplayAugmentationTests(unittest.TestCase):
    def example(self) -> ReplayTrainingExample:
        return ReplayTrainingExample(
            game_id="human",
            observation=[[0, 1, -1], [2, 3, -1]],
            total_mines=1,
            mode="standard",
            source="human",
            action="reveal",
            row=1,
            col=2,
            time_ms=10,
            classification="PROVEN_GOOD",
            teacher_exact=True,
            mine_probability=0.0,
            probabilities=[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]],
            risk_mask=[[1, 1, 1], [1, 1, 1]],
            policy_weight=1.0,
            value_target=1.0,
            value_weight=1.0,
        )

    def test_clockwise_rotation_changes_shape_and_action_consistently(self) -> None:
        variants = augment_example(self.example())
        rotated = variants[1]
        self.assertEqual(rotated.observation, [[2, 0], [3, 1], [-1, -1]])
        self.assertEqual((rotated.row, rotated.col), (2, 0))
        self.assertEqual(rotated.probabilities, [[0.3, 0.0], [0.4, 0.1], [0.5, 0.2]])

    def test_generates_all_eight_aligned_variants(self) -> None:
        variants = augment_example(self.example())
        self.assertEqual(len(variants), 8)
        self.assertEqual(len({item.game_id for item in variants}), 8)
        for item in variants:
            self.assertEqual(item.observation[item.row][item.col], -1)
            self.assertEqual(item.risk_mask[item.row][item.col], 1)


if __name__ == "__main__":
    unittest.main()
