import unittest

from minesweeper_ai.game import MinesweeperGame
from minesweeper_ai.metrics import (
    calculate_3bv,
    index_of_efficiency,
    three_bv_per_second,
)


class ThreeBVTests(unittest.TestCase):
    def test_single_opening_counts_once(self) -> None:
        game = MinesweeperGame(3, 3, 1, mine_positions={(2, 2)})
        result = calculate_3bv(game)
        self.assertEqual(result.value, 1)
        self.assertEqual(result.openings, 1)
        self.assertEqual(result.isolated_numbers, 0)

    def test_board_without_zeroes_counts_every_safe_number(self) -> None:
        game = MinesweeperGame(2, 2, 1, mine_positions={(0, 0)})
        result = calculate_3bv(game)
        self.assertEqual(result.value, 3)
        self.assertEqual(result.openings, 0)
        self.assertEqual(result.isolated_numbers, 3)

    def test_rate_and_ioe(self) -> None:
        self.assertEqual(three_bv_per_second(30, 2.0), 15.0)
        self.assertEqual(index_of_efficiency(30, 40), 0.75)

    def test_unplaced_board_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "placed mines"):
            calculate_3bv(MinesweeperGame(9, 9, 10, seed=1))


if __name__ == "__main__":
    unittest.main()
