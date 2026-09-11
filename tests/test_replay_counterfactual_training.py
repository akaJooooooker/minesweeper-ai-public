import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.replay_counterfactual import (
    CounterfactualCandidate,
    CounterfactualExample,
    write_counterfactual_examples,
)
from minesweeper_ai.replay_counterfactual_training import (
    _outcome_gap,
    fine_tune_counterfactual_policy,
)
from minesweeper_ai.replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model


def _example(game_id: str, flip: bool = False) -> CounterfactualExample:
    candidates = [
        CounterfactualCandidate(0, 1, 0.0 if flip else 1.0, 0.2, False, 1),
        CounterfactualCandidate(1, 2, 1.0 if flip else 0.0, 0.3, False, 1),
    ]
    return CounterfactualExample(
        game_id=game_id,
        state_index=0,
        observation=[[1, -1, -1], [1, -1, -1]],
        total_mines=1,
        source="simulator_counterfactual",
        candidates=candidates,
    )


class ReplayCounterfactualTrainingTests(unittest.TestCase):
    def test_outcome_gap_uses_full_candidate_spread(self) -> None:
        example = _example("tied-best")
        example.candidates.append(
            CounterfactualCandidate(1, 1, 1.0, 0.25, False, 1)
        )
        self.assertEqual(_outcome_gap(example), 1.0)

    def test_cpu_smoke_changes_only_reveal_policy_channel(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        examples = [_example(f"state-{index}") for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_path = root / "parent.pt"
            dataset_path = root / "counterfactual.jsonl"
            output_path = root / "v35.pt"
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
            write_counterfactual_examples(examples, dataset_path)
            initial, history = fine_tune_counterfactual_policy(
                parent_path,
                [dataset_path],
                dataset_path,
                output_path,
                epochs=4,
                batch_size=4,
                learning_rate=1e-2,
                device_name="cpu",
                use_amp=False,
                patience=4,
            )
            trained = torch.load(output_path, map_location="cpu", weights_only=True)
            parent_state = torch.load(
                parent_path,
                map_location="cpu",
                weights_only=True,
            )["state_dict"]

        self.assertEqual(initial.examples, 8)
        self.assertEqual(initial.informative_examples, 8)
        self.assertTrue(history)
        self.assertEqual(trained["kind"], "replay-agent-counterfactual-policy-v35")
        for name, tensor in parent_state.items():
            updated = trained["state_dict"][name]
            if name == "standard_policy_head.weight":
                self.assertTrue(torch.equal(tensor[1:], updated[1:]), name)
            elif name == "standard_policy_head.bias":
                self.assertTrue(torch.equal(tensor[1:], updated[1:]), name)
            elif not name.startswith("standard_policy_head."):
                self.assertTrue(torch.equal(tensor, updated), name)


if __name__ == "__main__":
    unittest.main()

