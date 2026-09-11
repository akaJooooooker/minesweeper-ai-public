import unittest

from minesweeper_ai.replay_inference import ReplayAgentPredictor
from minesweeper_ai.replay_learning import _torch_model


class ReplayResourceLimitTests(unittest.TestCase):
    def test_predictor_rejects_nonpositive_cpu_thread_limit(self) -> None:
        try:
            _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        with self.assertRaises(ValueError):
            ReplayAgentPredictor("missing.pt", cpu_threads=0)


if __name__ == "__main__":
    unittest.main()
