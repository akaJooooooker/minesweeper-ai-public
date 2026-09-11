import unittest

from minesweeper_ai.game import FLAGGED, UNKNOWN, GameStatus, MinesweeperGame


class MinesweeperGameTests(unittest.TestCase):
    def test_arbitrary_board_and_clues(self) -> None:
        game = MinesweeperGame(
            4,
            3,
            2,
            mine_positions={(0, 0), (2, 3)},
        )
        opened = game.reveal((1, 1))
        self.assertIn((1, 1), opened)
        self.assertEqual(game.observation[1][1], 1)
        self.assertEqual(game.status, GameStatus.ACTIVE)

    def test_zero_expansion_wins(self) -> None:
        game = MinesweeperGame(3, 3, 1, mine_positions={(2, 2)})
        opened = game.reveal((0, 0))
        self.assertEqual(len(opened), 8)
        self.assertEqual(game.status, GameStatus.WON)

    def test_loss(self) -> None:
        game = MinesweeperGame(2, 2, 1, mine_positions={(0, 0)})
        game.reveal((0, 0))
        self.assertEqual(game.status, GameStatus.LOST)
        self.assertEqual(game.exploded, (0, 0))

    def test_first_click_zero(self) -> None:
        game = MinesweeperGame(9, 9, 10, seed=7, first_click_zero=True)
        game.reveal((4, 4))
        self.assertEqual(game.observation[4][4], 0)
        self.assertTrue(all(not game.is_mine(cell) for cell in game.neighbours((4, 4))))

    def test_flag_toggle_and_chord(self) -> None:
        game = MinesweeperGame(3, 2, 1, mine_positions={(0, 0)})
        game.reveal((1, 1))
        self.assertTrue(game.toggle_flag((0, 0)))
        self.assertEqual(game.observation[0][0], FLAGGED)
        opened = game.chord((1, 1))
        self.assertGreater(len(opened), 0)
        self.assertNotEqual(game.observation[0][1], UNKNOWN)

    def test_clone_preserves_hidden_board_and_is_independent(self) -> None:
        game = MinesweeperGame(4, 3, 2, seed=7, first_click_zero=True)
        game.reveal((1, 1))
        clone = game.clone()

        self.assertEqual(clone.observation, game.observation)
        self.assertEqual(clone.status, game.status)
        for row in range(game.height):
            for col in range(game.width):
                self.assertEqual(clone.is_mine((row, col)), game.is_mine((row, col)))

        covered = clone.covered_cells[0]
        clone.toggle_flag(covered)
        self.assertNotEqual(clone.observation, game.observation)

    def test_clone_before_first_reveal_preserves_rng_state(self) -> None:
        game = MinesweeperGame(6, 6, 5, seed=11, first_click_zero=True)
        clone = game.clone()
        game.reveal((3, 3))
        clone.reveal((3, 3))
        for row in range(game.height):
            for col in range(game.width):
                self.assertEqual(clone.is_mine((row, col)), game.is_mine((row, col)))


if __name__ == "__main__":
    unittest.main()
