import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.data import BoardSpec
from minesweeper_ai.replay import read_replays
from minesweeper_ai.replay_hard_bootstrap import generate_hard_replays


class ReplayHardBootstrapTests(unittest.TestCase):
    def test_generates_requested_traceable_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hard.jsonl"
            stats = generate_hard_replays(
                output,
                games=2,
                seed=9,
                teacher_cells=20,
                spec=BoardSpec(4, 4, 2, True),
            )
            replays = list(read_replays(output))
        self.assertEqual(stats.replays, 2)
        self.assertEqual(len(replays), 2)
        self.assertTrue(all(replay.width == 4 and replay.height == 4 for replay in replays))
        self.assertEqual(stats.moves, sum(len(replay.moves) for replay in replays))


if __name__ == "__main__":
    unittest.main()
