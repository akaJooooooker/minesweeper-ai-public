import unittest

from minesweeper_ai.agent import HybridAgent
from minesweeper_ai.data import BoardSpec
from minesweeper_ai.replay import ReplayGame, ReplayMove
from minesweeper_ai.replay_scorecard import benchmark_no_guess_replays


class ReplayScorecardTests(unittest.TestCase):
    def test_uses_replay_first_move_as_prescribed_opening(self) -> None:
        replay = ReplayGame(
            game_id="test",
            width=3,
            height=3,
            mines=1,
            mode="no_guess",
            source="test",
            won=True,
            mine_positions=[[2, 2]],
            moves=[ReplayMove("reveal", 0, 0, 100)],
        )
        summary = benchmark_no_guess_replays(
            HybridAgent(),
            [replay],
            BoardSpec(3, 3, 1, True),
        )
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["opening_size_min"], 8)
        self.assertEqual(summary["invalid_openings"], 0)
        self.assertEqual(summary["guesses"], 0)


if __name__ == "__main__":
    unittest.main()
