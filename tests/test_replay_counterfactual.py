import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.agent import HybridAgent
from minesweeper_ai.game import GameStatus, MinesweeperGame
from minesweeper_ai.replay_counterfactual import (
    CounterfactualCandidate,
    CounterfactualExample,
    _rollout_candidate,
    _stable_choice_weights,
    augment_counterfactual_example,
    generate_counterfactual_examples,
    read_counterfactual_examples,
    write_counterfactual_examples,
)
from minesweeper_ai.solver import ConstraintSolver


class ReplayCounterfactualTests(unittest.TestCase):
    def test_jsonl_round_trip(self) -> None:
        example = CounterfactualExample(
            game_id="g-1",
            state_index=0,
            observation=[[1, -1], [-1, -2]],
            total_mines=1,
            source="simulator_counterfactual",
            candidates=[
                CounterfactualCandidate(0, 1, 1.0, 0.2, False, 1),
                CounterfactualCandidate(1, 0, 0.0, 0.3, True, 0),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counterfactual.jsonl"
            count = write_counterfactual_examples([example], path)
            loaded = list(read_counterfactual_examples(path))
        self.assertEqual(count, 1)
        self.assertEqual(loaded, [example])

    def test_augmentation_rotates_observation_and_all_candidates(self) -> None:
        example = CounterfactualExample(
            game_id="g-aug",
            state_index=0,
            observation=[[1, -1], [-1, -2]],
            total_mines=1,
            source="simulator_counterfactual",
            candidates=[
                CounterfactualCandidate(0, 1, 1.0, 0.2, False, 1),
                CounterfactualCandidate(1, 0, 0.0, 0.3, True, 0),
            ],
        )
        variants = augment_counterfactual_example(example)
        rotated = variants[1]
        self.assertEqual(len(variants), 8)
        self.assertEqual(rotated.observation, [[-1, 1], [-2, -1]])
        self.assertEqual(
            [(item.row, item.col, item.outcome) for item in rotated.candidates],
            [(1, 1, 1.0), (0, 0, 0.0)],
        )
        self.assertEqual(len({item.game_id for item in variants}), 8)

    def test_choice_weights_ignore_non_finite_scores(self) -> None:
        ranked = [(0, 0), (0, 1), (0, 2)]
        weights = _stable_choice_weights(
            {(0, 0): 0.25, (0, 1): float("inf"), (0, 2): float("nan")},
            ranked,
            0.03,
        )
        self.assertEqual(weights, [1.0, 0.0, 0.0])
        self.assertIsNone(
            _stable_choice_weights(
                {(0, 0): float("inf"), (0, 1): float("nan")},
                ranked[:2],
                0.03,
            )
        )

    def test_agent_continuation_requires_one_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused.jsonl"
            with self.assertRaisesRegex(ValueError, "exactly one"):
                generate_counterfactual_examples(
                    "missing.pt",
                    output,
                    games=1,
                    seed=0,
                    continuation_mode="agent",
                    rollouts_per_candidate=2,
                )


    def test_candidate_rollout_uses_independent_hidden_board(self) -> None:
        game = MinesweeperGame(2, 2, 1, mine_positions={(0, 0)})
        game.reveal((0, 1))
        game.reveal((1, 0))
        original_observation = game.observation
        agent = HybridAgent(ConstraintSolver(24))

        losing = _rollout_candidate(
            game,
            (0, 0),
            agent,
            mine_probability=0.5,
        )
        winning = _rollout_candidate(
            game,
            (1, 1),
            agent,
            mine_probability=0.5,
        )

        self.assertEqual(losing.outcome, 0.0)
        self.assertTrue(losing.was_mine)
        self.assertEqual(winning.outcome, 1.0)
        self.assertFalse(winning.was_mine)
        self.assertEqual(game.status, GameStatus.ACTIVE)
        self.assertEqual(game.observation, original_observation)


if __name__ == "__main__":
    unittest.main()

