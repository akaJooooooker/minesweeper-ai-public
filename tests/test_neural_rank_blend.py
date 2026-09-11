import unittest

from minesweeper_ai.agent import HybridAgent
from tests.test_neural_blend import StubRisk, StubSolver


class NeuralRankBlendTests(unittest.TestCase):
    def test_rank_blend_combines_relative_order(self) -> None:
        decision = HybridAgent(
            StubSolver(),
            risk_predictor=StubRisk(),
            neural_blend=0.75,
            neural_blend_mode="rank",
        ).choose_move(((1, -1, -1),), 1)
        self.assertEqual(decision.coord, (0, 2))

    def test_mode_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "probability or rank"):
            HybridAgent(StubSolver(), neural_blend_mode="invalid")


if __name__ == "__main__":
    unittest.main()
