import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.replay_inference import ReplayAgentPredictor
from minesweeper_ai.replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model


class ReplayInferenceTests(unittest.TestCase):
    def test_checkpoint_prediction_exposes_all_heads(self) -> None:
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
            predictor = ReplayAgentPredictor(checkpoint, device="cpu")
            prediction = predictor.predict_all(((-1, -1), (0, 0)), 1)
        self.assertEqual(set(prediction.risk), {(0, 0), (0, 1)})
        self.assertEqual(len(prediction.policy_logits), 12)
        self.assertGreaterEqual(prediction.value, 0.0)
        self.assertLessEqual(prediction.value, 1.0)

    def test_rejects_incompatible_feature_layout(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bad.pt"
            model = ReplayAgentNet(16)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_channels": 999,
                    "action_order": ["reveal", "flag", "chord"],
                    "model_width": 16,
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "feature layout"):
                ReplayAgentPredictor(checkpoint, device="cpu")


if __name__ == "__main__":
    unittest.main()
