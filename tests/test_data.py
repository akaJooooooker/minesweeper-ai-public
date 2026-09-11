import tempfile
import unittest
from pathlib import Path

import numpy as np

from minesweeper_ai.data import DaggerGenerator, DatasetGenerator, read_jsonl, write_jsonl
from minesweeper_ai.features import FEATURE_CHANNELS, encode_board
from minesweeper_ai.game import UNKNOWN, MinesweeperGame
from minesweeper_ai.solver import ConstraintSolver


class FeatureTests(unittest.TestCase):
    def test_variable_size_encoding(self) -> None:
        observation = (
            (0, 1, UNKNOWN),
            (0, 1, UNKNOWN),
        )
        features, valid = encode_board(observation, total_mines=1)
        self.assertEqual(features.shape, (FEATURE_CHANNELS, 2, 3))
        self.assertEqual(valid.shape, (2, 3))
        self.assertEqual(features[9, 0, 2], 1.0)
        self.assertEqual(features[11, 0, 2], 1.0)
        self.assertTrue(np.all(valid == 1.0))


class DatasetTests(unittest.TestCase):
    def test_teacher_examples_round_trip(self) -> None:
        game = MinesweeperGame(4, 4, 2, mine_positions={(0, 1), (3, 3)})
        game.reveal((0, 0))
        examples = list(DatasetGenerator(seed=1).collect_game(game))
        self.assertGreaterEqual(len(examples), 1)
        self.assertTrue(all(example.exact for example in examples))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            count = write_jsonl(examples, path)
            loaded = list(read_jsonl(path))
        self.assertEqual(count, len(examples))
        self.assertEqual(loaded, examples)

    def test_dagger_records_states_student_cannot_enumerate(self) -> None:
        game = MinesweeperGame(4, 4, 2, mine_positions={(0, 1), (3, 3)})
        game.reveal((0, 0))
        generator = DaggerGenerator(
            ConstraintSolver(max_component_cells=20),
            ConstraintSolver(max_component_cells=1),
            seed=2,
        )
        examples = list(generator.collect_game(game))
        self.assertGreaterEqual(len(examples), 1)
        self.assertTrue(all(example.source == "dagger" for example in examples))
        self.assertTrue(all(example.student_exact is False for example in examples))
        self.assertTrue(all(example.exact for example in examples))


if __name__ == "__main__":
    unittest.main()
