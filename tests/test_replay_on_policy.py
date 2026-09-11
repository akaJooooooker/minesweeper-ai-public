import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.data import BoardSpec
from minesweeper_ai.replay import read_replays
from minesweeper_ai.replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model, encode_replay_board
from minesweeper_ai.replay_on_policy import generate_on_policy_replays


class ReplayOnPolicyTests(unittest.TestCase):
    def test_generates_trace_and_uses_simulator_source_plane(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "agent.pt"
            output = root / "replays.jsonl"
            model = ReplayAgentNet(16)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_channels": REPLAY_FEATURE_CHANNELS,
                    "action_order": ["reveal", "flag", "chord"],
                    "model_width": 16,
                },
                checkpoint,
            )
            stats = generate_on_policy_replays(
                checkpoint,
                output,
                games=2,
                seed=7,
                solver_cells=20,
                device="cpu",
                spec=BoardSpec(4, 4, 2, True),
            )
            replays = list(read_replays(output))
        self.assertEqual(stats.replays, 2)
        self.assertTrue(all(replay.source == "simulator_on_policy" for replay in replays))
        features, _ = encode_replay_board(
            ((-1, -1), (-1, -1)), 1, mode="standard", source="simulator_on_policy"
        )
        self.assertTrue((features[17] == 0).all())


if __name__ == "__main__":
    unittest.main()
