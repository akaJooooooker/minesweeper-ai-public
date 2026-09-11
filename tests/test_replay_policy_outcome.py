import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.replay import ReplayTrainingExample, write_replay_examples
from minesweeper_ai.replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model
from minesweeper_ai.replay_policy_outcome import fine_tune_outcome_policy


def _example(game_id: str, won: bool, row: int, col: int) -> ReplayTrainingExample:
    return ReplayTrainingExample(
        game_id=game_id,
        observation=[[1, -1, -1], [1, -1, -1]],
        total_mines=1,
        mode="standard",
        source="simulator_on_policy",
        action="reveal",
        row=row,
        col=col,
        time_ms=100,
        classification="NECESSARY_GUESS",
        teacher_exact=True,
        mine_probability=0.25,
        probabilities=[[None, 0.25, 0.25], [None, 0.25, 0.25]],
        risk_mask=[[0, 1, 1], [0, 1, 1]],
        policy_weight=0.0,
        value_target=1.0 if won else 0.0,
        value_weight=0.0,
    )


class ReplayOutcomePolicyTests(unittest.TestCase):
    def test_cpu_smoke_training_changes_only_standard_policy_head(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        examples = [
            _example("win-a", True, 0, 1),
            _example("win-b", True, 1, 1),
            _example("loss-a", False, 0, 2),
            _example("loss-b", False, 1, 2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_path = root / "parent.pt"
            dataset_path = root / "outcomes.jsonl"
            output_path = root / "v33.pt"
            parent = ReplayAgentNet(16)
            torch.save(
                {
                    "state_dict": parent.state_dict(),
                    "feature_channels": REPLAY_FEATURE_CHANNELS,
                    "action_order": ["reveal", "flag", "chord"],
                    "model_width": 16,
                },
                parent_path,
            )
            write_replay_examples(examples, dataset_path)
            initial, history = fine_tune_outcome_policy(
                parent_path,
                [dataset_path],
                dataset_path,
                output_path,
                epochs=3,
                batch_size=4,
                learning_rate=1e-3,
                min_progress=0.0,
                device_name="cpu",
                patience=3,
            )
            trained = torch.load(output_path, map_location="cpu", weights_only=True)
            parent_state = torch.load(parent_path, map_location="cpu", weights_only=True)[
                "state_dict"
            ]

        self.assertEqual(initial.examples, 4)
        self.assertTrue(history)
        self.assertEqual(trained["kind"], "replay-agent-outcome-policy-v33")
        for name, tensor in parent_state.items():
            if not name.startswith("standard_policy_head."):
                self.assertTrue(torch.equal(tensor, trained["state_dict"][name]), name)


if __name__ == "__main__":
    unittest.main()

