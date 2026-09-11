import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.data import BoardSpec
from minesweeper_ai.replay_checkpoint_compare import compare_checkpoints
from minesweeper_ai.replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model


class ReplayCheckpointCompareTests(unittest.TestCase):
    def test_identical_checkpoints_have_identical_outcomes(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "agent.pt"
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
            result = compare_checkpoints(
                checkpoint,
                checkpoint,
                games_per_spec=2,
                seed=7,
                solver_cells=2,
                device="cpu",
                specs=(BoardSpec(4, 4, 2, True),),
            )
        self.assertEqual(result.candidate_wins, result.reference_wins)
        self.assertEqual(result.candidate_only_wins, 0)
        self.assertEqual(result.reference_only_wins, 0)


if __name__ == "__main__":
    unittest.main()
