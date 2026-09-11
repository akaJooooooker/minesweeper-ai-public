import unittest

from minesweeper_ai.agent import Decision, HybridAgent
from minesweeper_ai.desktop_live import LiveDecisionEngine, detect_region_reset
from minesweeper_ai.game import FLAGGED, UNKNOWN
from minesweeper_ai.solver import Analysis, ConstraintSolver


class DesktopLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = LiveDecisionEngine(HybridAgent(ConstraintSolver(12)))

    def test_proven_mine_is_not_flagged_without_profitable_chord(self) -> None:
        action = self.engine.next_action(((1, UNKNOWN),), 1, allow_guess=False)
        self.assertEqual(action.action, "stuck")

    def test_flags_only_as_setup_for_profitable_chord(self) -> None:
        observation = (
            (UNKNOWN, UNKNOWN, UNKNOWN),
            (UNKNOWN, 1, UNKNOWN),
            (UNKNOWN, UNKNOWN, UNKNOWN),
        )
        mine = (0, 0)
        safe = frozenset(
            (row, col)
            for row in range(3)
            for col in range(3)
            if (row, col) not in {mine, (1, 1)}
        )
        analysis = Analysis(
            safe=safe,
            mines=frozenset({mine}),
            probabilities={mine: 1.0, **{coord: 0.0 for coord in safe}},
            exact=True,
            model_count=1,
            frontier_size=8,
        )

        class StaticAgent:
            def choose_move(self, *_args, **_kwargs):
                return Decision("reveal", (0, 1), 0.0, "safe", True, analysis)

        engine = LiveDecisionEngine(StaticAgent(), use_flags=True)
        flag = engine.next_action(observation, 1, allow_guess=False)
        self.assertEqual((flag.action, flag.coord), ("flag", mine))
        self.assertIn("saves 5 clicks", flag.reason)

        flagged_observation = (
            (FLAGGED, UNKNOWN, UNKNOWN),
            (UNKNOWN, 1, UNKNOWN),
            (UNKNOWN, UNKNOWN, UNKNOWN),
        )
        after_flag = Analysis(
            safe=safe,
            mines=frozenset(),
            probabilities={coord: 0.0 for coord in safe},
            exact=True,
            model_count=1,
            frontier_size=7,
        )
        analysis = after_flag
        chord = engine.next_action(flagged_observation, 1, allow_guess=False)
        self.assertEqual((chord.action, chord.coord), ("chord", (1, 1)))
        self.assertIn("saves 6 clicks", chord.reason)

    def test_clickable_neural_guess_keeps_learned_choice_and_probability(self):
        analysis = Analysis(frozenset(), frozenset(), {(0, 0): 0.1, (0, 1): 0.6},
                            False, None, 2)
        class LearnedAgent:
            def choose_move(self, *_args, **_kwargs):
                return Decision("reveal", (0, 1), 0.2, "minimum neural risk with outcome policy",
                                False, analysis)
        engine = LiveDecisionEngine(LearnedAgent())
        obs = ((UNKNOWN, UNKNOWN), (1, 1))
        action = engine.next_action(obs, 1, allow_guess=True)
        self.assertEqual(action.coord, (0, 1))
        self.assertEqual(action.mine_probability, 0.2)
        self.assertIn("outcome policy", action.reason)
        blocked = engine.next_action(obs, 1, allow_guess=True,
                                     allowed=((True, False), (True, True)))
        self.assertEqual(blocked.coord, (0, 0))
        self.assertEqual(blocked.mine_probability, 0.1)
        self.assertEqual(engine.next_action(obs, 1, allow_guess=False).action, "stuck")

    def test_contradiction_is_never_replaced_by_a_guess(self):
        action = self.engine.next_action(((0, FLAGGED), (UNKNOWN, UNKNOWN)),
                                         1, allow_guess=True)
        self.assertEqual(action.action, "stuck")
        self.assertIsNone(action.coord)

    def test_region_reset_detects_revealed_to_covered(self) -> None:
        before = ((0, 1), (UNKNOWN, UNKNOWN))
        after = ((UNKNOWN, UNKNOWN), (UNKNOWN, UNKNOWN))
        self.assertTrue(detect_region_reset(before, after))
        self.assertFalse(detect_region_reset(None, after))


if __name__ == "__main__":
    unittest.main()
