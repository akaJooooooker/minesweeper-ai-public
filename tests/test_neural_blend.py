import unittest

from minesweeper_ai.agent import HybridAgent
from minesweeper_ai.solver import Analysis


class StubSolver:
    def analyse(self, observation, total_mines):
        return Analysis(
            safe=frozenset(),
            mines=frozenset(),
            probabilities={(0, 1): 0.2, (0, 2): 0.3},
            exact=False,
            model_count=None,
            frontier_size=2,
        )


class StubRisk:
    def predict(self, observation, total_mines):
        return {(0, 1): 0.9, (0, 2): 0.1}


class StubPrediction:
    risk = {(0, 1): 0.2, (0, 2): 0.3}
    policy_logits = {("reveal", (0, 1)): -2.0, ("reveal", (0, 2)): 2.0}


class StubRiskAndPolicy(StubRisk):
    def predict_all(self, observation, total_mines):
        return StubPrediction()


class StubSolverThree:
    def analyse(self, observation, total_mines):
        return Analysis(
            safe=frozenset(),
            mines=frozenset(),
            probabilities={(0, 1): 0.1, (0, 2): 0.2, (0, 3): 0.9},
            exact=False,
            model_count=None,
            frontier_size=3,
        )


class StubPredictionThree:
    risk = {(0, 1): 0.1, (0, 2): 0.2, (0, 3): 0.9}
    policy_logits = {
        ("reveal", (0, 1)): 0.0,
        ("reveal", (0, 2)): 1.0,
        ("reveal", (0, 3)): 100.0,
    }


class StubRiskAndPolicyThree:
    def predict_all(self, observation, total_mines):
        return StubPredictionThree()


class NeuralBlendTests(unittest.TestCase):
    def test_zero_and_full_blend_choose_expected_sources(self) -> None:
        observation = ((1, -1, -1),)
        solver_choice = HybridAgent(
            StubSolver(), risk_predictor=StubRisk(), neural_blend=0.0
        ).choose_move(observation, 1)
        neural_choice = HybridAgent(
            StubSolver(), risk_predictor=StubRisk(), neural_blend=1.0
        ).choose_move(observation, 1)
        self.assertEqual(solver_choice.coord, (0, 1))
        self.assertEqual(neural_choice.coord, (0, 2))

    def test_blend_range_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            HybridAgent(StubSolver(), neural_blend=1.1)

    def test_outcome_policy_can_choose_between_uncertain_cells(self) -> None:
        observation = ((1, -1, -1),)
        decision = HybridAgent(
            StubSolver(),
            risk_predictor=StubRiskAndPolicy(),
            neural_blend=0.0,
            policy_blend=1.0,
        ).choose_move(observation, 1)
        self.assertEqual(decision.coord, (0, 2))
        self.assertEqual(decision.mine_probability, 0.3)
        self.assertIn("outcome policy", decision.reason)

    def test_policy_blend_range_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy_blend"):
            HybridAgent(StubSolver(), policy_blend=-0.1)

    def test_policy_top_k_keeps_policy_inside_low_risk_candidates(self) -> None:
        observation = ((1, -1, -1, -1),)
        decision = HybridAgent(
            StubSolverThree(),
            risk_predictor=StubRiskAndPolicyThree(),
            neural_blend=0.0,
            policy_blend=1.0,
            policy_top_k=2,
        ).choose_move(observation, 1)
        self.assertEqual(decision.coord, (0, 2))

    def test_policy_top_k_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy_top_k"):
            HybridAgent(StubSolver(), policy_top_k=-1)

    def test_policy_min_advantage_abstains_when_margin_is_too_small(self) -> None:
        observation = ((1, -1, -1),)
        decision = HybridAgent(
            StubSolver(),
            risk_predictor=StubRiskAndPolicy(),
            neural_blend=0.0,
            policy_blend=1.0,
            policy_min_advantage=0.9,
        ).choose_move(observation, 1)
        self.assertEqual(decision.coord, (0, 1))
        self.assertNotIn("outcome policy", decision.reason)

    def test_policy_min_advantage_allows_a_large_margin(self) -> None:
        observation = ((1, -1, -1),)
        decision = HybridAgent(
            StubSolver(),
            risk_predictor=StubRiskAndPolicy(),
            neural_blend=0.0,
            policy_blend=1.0,
            policy_min_advantage=0.5,
        ).choose_move(observation, 1)
        self.assertEqual(decision.coord, (0, 2))
        self.assertIn("outcome policy", decision.reason)

    def test_policy_min_advantage_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy_min_advantage"):
            HybridAgent(StubSolver(), policy_min_advantage=1.1)

    def test_policy_min_board_cells_disables_policy_on_small_boards(self) -> None:
        observation = ((1, -1, -1),)
        decision = HybridAgent(
            StubSolver(),
            risk_predictor=StubRiskAndPolicy(),
            neural_blend=0.0,
            policy_blend=1.0,
            policy_min_board_cells=4,
        ).choose_move(observation, 1)
        self.assertEqual(decision.coord, (0, 1))

    def test_policy_min_board_cells_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy_min_board_cells"):
            HybridAgent(StubSolver(), policy_min_board_cells=-1)




if __name__ == "__main__":
    unittest.main()
