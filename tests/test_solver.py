import unittest

from minesweeper_ai.agent import HybridAgent
from minesweeper_ai.game import FLAGGED, UNKNOWN, GameStatus, MinesweeperGame
from minesweeper_ai.solver import Analysis, ConstraintSolver


class ConstraintSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = ConstraintSolver(max_component_cells=20)

    def test_forced_mine(self) -> None:
        analysis = self.solver.analyse(((1, UNKNOWN),), total_mines=1)
        self.assertEqual(analysis.mines, frozenset({(0, 1)}))
        self.assertEqual(analysis.probabilities[(0, 1)], 1.0)
        self.assertTrue(analysis.exact)

    def test_equal_probability_frontier(self) -> None:
        observation = (
            (1, UNKNOWN),
            (1, UNKNOWN),
        )
        analysis = self.solver.analyse(observation, total_mines=1)
        self.assertEqual(analysis.model_count, 2)
        self.assertAlmostEqual(analysis.probabilities[(0, 1)], 0.5)
        self.assertAlmostEqual(analysis.probabilities[(1, 1)], 0.5)

    def test_global_mine_count_marks_unconstrained_cells_safe(self) -> None:
        analysis = self.solver.analyse(
            ((1, UNKNOWN, UNKNOWN, UNKNOWN),),
            total_mines=1,
        )
        self.assertIn((0, 1), analysis.mines)
        self.assertIn((0, 2), analysis.safe)
        self.assertIn((0, 3), analysis.safe)

    def test_flags_are_respected(self) -> None:
        analysis = self.solver.analyse(((1, FLAGGED, UNKNOWN),), total_mines=1)
        self.assertIn((0, 2), analysis.safe)
        self.assertEqual(analysis.probabilities[(0, 2)], 0.0)

    def test_inconsistent_board_is_reported(self) -> None:
        analysis = self.solver.analyse(((0, FLAGGED),), total_mines=1)
        self.assertIsNotNone(analysis.contradiction)
        self.assertIn("clue=0, flags=1, covered=0", analysis.contradiction or "")


class HybridAgentTests(unittest.TestCase):
    def test_agent_solves_a_deterministic_board_without_guessing(self) -> None:
        game = MinesweeperGame(3, 3, 1, mine_positions={(2, 2)})
        game.reveal((0, 0))
        result = HybridAgent().play(game, allow_guess=False)
        self.assertTrue(result.won)
        self.assertEqual(result.status, GameStatus.WON)
        self.assertEqual(result.guesses, 0)

    def test_agent_uses_chord_with_only_proven_flags(self) -> None:
        game = MinesweeperGame(3, 2, 2, mine_positions={(0, 0), (0, 2)})
        game.reveal((1, 0))
        game.reveal((1, 2))
        result = HybridAgent().play(game, allow_guess=False)
        self.assertTrue(result.won)
        self.assertEqual(result.flags, 2)
        self.assertEqual(result.chords, 1)
        self.assertEqual(result.reveals, 0)
        self.assertEqual(result.clicks, 3)


    def test_neural_risk_is_used_only_for_nonexact_analysis(self) -> None:
        class FakePredictor:
            def predict(self, observation, total_mines):
                return {(0, 1): 0.9, (1, 1): 0.1}

        observation = (
            (1, UNKNOWN),
            (1, UNKNOWN),
        )
        agent = HybridAgent(
            ConstraintSolver(max_component_cells=1),
            FakePredictor(),
        )
        decision = agent.choose_move(observation, total_mines=1)
        self.assertEqual(decision.coord, (1, 1))
        self.assertEqual(decision.reason, "minimum neural risk after exact deductions")

    def test_larger_solver_is_used_only_after_primary_budget_is_exceeded(self) -> None:
        class FakeFallback:
            max_component_cells = 4

            def analyse(self, observation, total_mines):
                return Analysis(
                    safe=frozenset({(1, 1)}),
                    mines=frozenset(),
                    probabilities={(0, 1): 1.0, (1, 1): 0.0},
                    exact=True,
                    model_count=1,
                    frontier_size=2,
                )

        observation = (
            (1, UNKNOWN),
            (1, UNKNOWN),
        )
        agent = HybridAgent(
            ConstraintSolver(max_component_cells=1),
            fallback_solver=FakeFallback(),
        )
        decision = agent.choose_move(observation, total_mines=1)
        self.assertEqual(decision.coord, (1, 1))
        self.assertEqual(decision.mine_probability, 0.0)
        self.assertEqual(decision.reason, "logically forced safe after exact escalation")

    def test_no_guess_mode_uses_proof_solver_even_when_primary_has_safe_move(self) -> None:
        class FakePrimary:
            max_component_cells = 1

            def analyse(self, observation, total_mines):
                return Analysis(
                    safe=frozenset({(0, 1)}),
                    mines=frozenset(),
                    probabilities={(0, 1): 0.0},
                    exact=False,
                    model_count=None,
                    frontier_size=2,
                )

        class FakeFallback:
            max_component_cells = 4

            def analyse(self, observation, total_mines):
                return Analysis(
                    safe=frozenset({(1, 1)}),
                    mines=frozenset(),
                    probabilities={(1, 1): 0.0},
                    exact=True,
                    model_count=1,
                    frontier_size=2,
                )

        observation = (
            (1, UNKNOWN),
            (1, UNKNOWN),
        )
        agent = HybridAgent(FakePrimary(), fallback_solver=FakeFallback())
        decision = agent.choose_move(observation, total_mines=1, allow_guess=False)
        self.assertEqual(decision.coord, (1, 1))
        self.assertEqual(decision.reason, "logically forced safe after exact escalation")


if __name__ == "__main__":
    unittest.main()
