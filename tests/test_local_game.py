import unittest
from PIL import Image

from minesweeper_ai.game import FLAGGED, UNKNOWN, GameStatus, MinesweeperGame
from minesweeper_ai.local_game import LocalGameSession
from minesweeper_ai.local_app import preview_cells
from minesweeper_ai.local_skin import tile_image
from minesweeper_ai.desktop_vision import HdSkinRecognizer


class LocalGameTests(unittest.TestCase):
    def test_clock_clicks_chord_and_finish_are_frozen(self):
        now = [10.0]
        session = LocalGameSession(MinesweeperGame(2, 2, 1, mine_positions={(0, 0)}),
                                   clock=lambda: now[0])
        session.apply("flag", (0, 0))
        self.assertEqual(session.elapsed, 0)
        now[0] = 20
        session.apply("reveal", (1, 1))
        now[0] = 21
        session.apply("reveal", (0, 0))  # Flag blocks opening.
        self.assertEqual(session.clicks.redundant["left"], 1)
        now[0] = 22
        session.apply("chord", (1, 1))
        self.assertEqual(session.game.status, GameStatus.WON)
        self.assertEqual(session.completed_bv, 3)
        self.assertEqual(session.three_bv, 3)
        self.assertEqual(session.elapsed, 2)
        self.assertEqual(session.summary()["efficiency"], 3 / 4)
        now[0] = 30
        self.assertFalse(session.apply("flag", (0, 0)))
        self.assertEqual(session.elapsed, 2)
        self.assertEqual(session.clicks.total, 4)

    def test_failed_chord_and_cancel_flag(self):
        session = LocalGameSession(MinesweeperGame(2, 2, 1, mine_positions={(0, 0)}))
        session.apply("reveal", (1, 1))
        session.apply("chord", (1, 1))
        self.assertEqual(session.clicks.redundant["chord"], 1)
        session.apply("flag", (0, 0))
        session.apply("flag", (0, 0))
        self.assertEqual(session.clicks.effective["right"], 2)
        self.assertEqual(session.game.observation[0][0], UNKNOWN)

    def test_wrong_flag_can_lose_and_freezes_clock(self):
        now = [10]
        session = LocalGameSession(MinesweeperGame(2, 2, 1, mine_positions={(0, 0)}),
                                   clock=lambda: now[0])
        session.apply("reveal", (1, 1))
        session.apply("flag", (0, 1))
        now[0] = 12
        session.apply("chord", (1, 1))
        self.assertEqual(session.game.status, GameStatus.LOST)
        self.assertLess(session.completed_bv, session.three_bv)
        now[0] = 50
        self.assertEqual(session.elapsed, 2)

    def test_opening_progress_requires_all_zeroes(self):
        session = LocalGameSession(MinesweeperGame(3, 3, 1, mine_positions={(2, 2)}))
        session.apply("flag", (0, 0))
        session.apply("reveal", (0, 1))
        self.assertEqual(session.completed_bv, 0)
        session.apply("flag", (0, 0))
        session.apply("reveal", (0, 0))
        self.assertEqual(session.completed_bv, 1)
        self.assertEqual(session.three_bv, 1)

    def test_many_boards_bv_matches_at_win_and_seed_reproduces(self):
        for seed in range(15):
            a = LocalGameSession(MinesweeperGame(9, 9, 10, seed=seed, first_click_zero=True))
            b = MinesweeperGame(9, 9, 10, seed=seed, first_click_zero=True)
            a.apply("reveal", (4, 4))
            b.reveal((4, 4))
            self.assertEqual(a.game.observation, b.observation)
            self.assertEqual(a.game.observation[4][4], 0)
            for row in range(9):
                for col in range(9):
                    if not a.game.is_mine((row, col)):
                        a.apply("reveal", (row, col))
            self.assertEqual(a.game.status, GameStatus.WON)
            self.assertEqual(a.completed_bv, a.three_bv)

    def test_preview_handles_edges_and_excludes_flags_and_numbers(self):
        board = ((FLAGGED, UNKNOWN, 1), (1, 2, UNKNOWN), (0, UNKNOWN, 3))
        self.assertEqual(preview_cells(board, (1, 1)), {(0, 1), (1, 2), (2, 1)})
        self.assertEqual(preview_cells(board, (1, 0)), {(0, 1), (2, 1)})
        self.assertEqual(preview_cells(board, (0, 0)), set())
        self.assertEqual(preview_cells(board, None), set())

    def test_revealed_mine_icons_remain_non_clues_for_external_plugin(self):
        for size in (24,28,32,36,40,42,56):
            reader=HdSkinRecognizer(theme='dark')
            self.assertEqual(reader.classify_cell(tile_image('mine',size)).value,UNKNOWN)
            self.assertIsNone(reader.classify_cell(tile_image('mine',size)).terminal)
            self.assertEqual(reader.classify_cell(tile_image('exploded',size)).terminal,'lost')

    def test_all_tiles_read_by_existing_external_plugin(self):
        for size in (24, 28, 32, 36, 40, 42, 56):
            recognizer = HdSkinRecognizer(theme="dark")
            values = [UNKNOWN, FLAGGED, *range(9)]
            image = Image.new("RGB", (size * len(values), size))
            for col, value in enumerate(values):
                image.paste(tile_image(value, size), (col * size, 0))
            result = recognizer.read_image(image, 1, len(values), total_mines=3)
            self.assertEqual(result.observation[0], tuple(values), (size, result.observation))
            self.assertTrue(all(result.allowed[0]), size)
            self.assertIsNone(result.terminal)
            exploded = recognizer.classify_cell(tile_image("exploded", size))
            self.assertEqual(exploded.terminal, "lost")


if __name__ == "__main__":
    unittest.main()
