import unittest

from minesweeper_ai.agent import HybridAgent
from minesweeper_ai.game import UNKNOWN
from minesweeper_ai.solver import Analysis, ConstraintSolver


class ProofEscalationTests(unittest.TestCase):
    def test_standard_mode_does_not_use_expensive_proof_tier(self) -> None:
        class FakeProof:
            max_component_cells = 64

            def analyse(self, observation, total_mines):
                raise AssertionError("standard mode must not use the no-guess proof tier")

        agent = HybridAgent(
            ConstraintSolver(1),
            fallback_solver=ConstraintSolver(2),
            proof_solver=FakeProof(),
        )
        decision = agent.choose_move(
            ((1, UNKNOWN), (1, UNKNOWN)),
            total_mines=1,
            allow_guess=True,
        )
        self.assertEqual(decision.action, "reveal")

    def test_no_guess_escalates_from_fallback_to_proof_solver(self) -> None:
        class FakePrimary:
            max_component_cells = 24

            def analyse(self, observation, total_mines):
                raise AssertionError("no-guess mode should bypass the primary solver")

        class FakeFallback:
            max_component_cells = 40

            def analyse(self, observation, total_mines):
                return Analysis(
                    safe=frozenset(),
                    mines=frozenset(),
                    probabilities={(0, 1): 0.5, (1, 1): 0.5},
                    exact=False,
                    model_count=None,
                    frontier_size=46,
                )

        class FakeProof:
            max_component_cells = 64

            def analyse(self, observation, total_mines):
                return Analysis(
                    safe=frozenset({(1, 1)}),
                    mines=frozenset(),
                    probabilities={(0, 1): 1.0, (1, 1): 0.0},
                    exact=True,
                    model_count=1,
                    frontier_size=46,
                )

        agent = HybridAgent(
            FakePrimary(),
            fallback_solver=FakeFallback(),
            proof_solver=FakeProof(),
        )
        decision = agent.choose_move(
            ((1, UNKNOWN), (1, UNKNOWN)),
            total_mines=1,
            allow_guess=False,
        )
        self.assertEqual(decision.coord, (1, 1))
        self.assertTrue(decision.exact)

    def test_proof_solver_budget_must_exceed_fallback(self) -> None:
        with self.assertRaises(ValueError):
            HybridAgent(
                ConstraintSolver(24),
                fallback_solver=ConstraintSolver(40),
                proof_solver=ConstraintSolver(40),
            )


if __name__ == "__main__":
    unittest.main()
