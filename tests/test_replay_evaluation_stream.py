import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.replay import ReplayAuditor, ReplayGame, ReplayMove, write_replay_examples
from minesweeper_ai.replay_evaluation import evaluate_checkpoint
from minesweeper_ai.replay_evaluation_stream import evaluate_checkpoint_streaming
from minesweeper_ai.replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model


class ReplayStreamingEvaluationTests(unittest.TestCase):
    def test_streaming_matches_materialized_evaluator(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        replay = ReplayGame(
            game_id="eval-1",
            width=3,
            height=2,
            mines=1,
            mode="standard",
            source="simulator_teacher",
            won=True,
            mine_positions=[[0, 0]],
            moves=[ReplayMove("reveal", 1, 1), ReplayMove("flag", 0, 0)],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "agent.pt"
            dataset = root / "examples.jsonl"
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
            write_replay_examples(ReplayAuditor().audit(replay), dataset)
            expected = evaluate_checkpoint(checkpoint, dataset, batch_size=2)
            actual = evaluate_checkpoint_streaming(checkpoint, dataset, batch_size=1)
        for key in (
            "examples",
            "risk_brier",
            "risk_cells",
            "policy_top1_demonstration_accuracy",
            "policy_examples",
            "value_accuracy",
            "value_brier",
        ):
            self.assertAlmostEqual(actual[key], expected[key], places=6)


if __name__ == "__main__":
    unittest.main()
