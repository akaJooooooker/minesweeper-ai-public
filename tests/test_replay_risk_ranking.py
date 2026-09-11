import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.replay import ReplayAuditor, ReplayGame, ReplayMove, write_replay_examples
from minesweeper_ai.replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model
from minesweeper_ai.replay_risk_ranking import (
    evaluate_ranking_checkpoint,
    fine_tune_risk_ranking,
)


class ReplayRiskRankingTests(unittest.TestCase):
    def test_cpu_ranking_smoke_train_and_evaluate(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        replay = ReplayGame(
            game_id="rank-1",
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
            parent = root / "parent.pt"
            dataset = root / "rank.jsonl"
            output = root / "v5.pt"
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
            write_replay_examples(ReplayAuditor().audit(replay), dataset)
            initial, history = fine_tune_risk_ranking(
                parent,
                [dataset],
                dataset,
                output,
                epochs=2,
                batch_size=2,
                device_name="cpu",
                patience=2,
            )
            measured = evaluate_ranking_checkpoint(output, dataset, batch_size=1)
        self.assertTrue(history)
        self.assertEqual(initial.states, len(list(ReplayAuditor().audit(replay))))
        self.assertGreaterEqual(measured.optimal_rate, 0.0)
        self.assertLessEqual(measured.optimal_rate, 1.0)
        self.assertGreaterEqual(measured.mean_regret, 0.0)


if __name__ == "__main__":
    unittest.main()
