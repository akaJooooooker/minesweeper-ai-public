import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.replay import ReplayAuditor, ReplayGame, ReplayMove, write_replay_examples
from minesweeper_ai.replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model
from minesweeper_ai.replay_risk_finetune import fine_tune_risk_head


class ReplayRiskFinetuneTests(unittest.TestCase):
    def test_cpu_smoke_finetune_writes_checkpoint(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        replay = ReplayGame(
            game_id="risk-1",
            width=3,
            height=2,
            mines=1,
            mode="standard",
            source="simulator_teacher",
            won=True,
            mine_positions=[[0, 0]],
            moves=[ReplayMove("reveal", 1, 1), ReplayMove("flag", 0, 0)],
        )
        examples = list(ReplayAuditor().audit(replay))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.pt"
            dataset = root / "risk.jsonl"
            output = root / "v3.pt"
            model = ReplayAgentNet(16)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_channels": REPLAY_FEATURE_CHANNELS,
                    "action_order": ["reveal", "flag", "chord"],
                    "model_width": 16,
                },
                parent,
            )
            write_replay_examples(examples, dataset)
            history = fine_tune_risk_head(
                parent,
                [dataset],
                dataset,
                output,
                epochs=2,
                batch_size=2,
                device_name="cpu",
                patience=2,
            )
            saved = torch.load(output, map_location="cpu", weights_only=True)
        self.assertTrue(history)
        self.assertEqual(saved["kind"], "replay-agent-risk-finetune-v3")
        self.assertIn("risk_finetune", saved)


if __name__ == "__main__":
    unittest.main()
