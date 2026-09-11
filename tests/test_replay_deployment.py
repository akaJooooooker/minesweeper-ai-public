import json
import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.replay_deployment import load_replay_deployment
from minesweeper_ai.replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model


class ReplayDeploymentTests(unittest.TestCase):
    def test_loads_relative_checkpoint_and_selected_blend(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "agent.pt"
            config_path = root / "deployment.json"
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
            config_path.write_text(
                json.dumps(
                    {
                        "kind": "replay-agent-deployment-v7",
                        "checkpoint": "agent.pt",
                        "mode": "standard",
                        "solver_cells": 20,
                        "fallback_solver_cells": 24,
                        "neural_blend": 0.75,
                        "neural_blend_mode": "probability",
                        "policy_blend": 0.2,
                        "policy_top_k": 4,
                        "policy_min_advantage": 0.075,
                        "policy_min_board_cells": 480,
                    }
                ),
                encoding="utf-8",
            )
            agent, config = load_replay_deployment(config_path)
        self.assertEqual(config.neural_blend, 0.75)
        self.assertEqual(agent.neural_blend, 0.75)
        self.assertEqual(agent.policy_blend, 0.2)
        self.assertEqual(agent.policy_top_k, 4)
        self.assertEqual(config.policy_min_advantage, 0.075)
        self.assertEqual(agent.policy_min_advantage, 0.075)
        self.assertEqual(config.policy_min_board_cells, 480)
        self.assertEqual(agent.policy_min_board_cells, 480)
        self.assertEqual(agent.solver.max_component_cells, 20)
        self.assertEqual(config.fallback_solver_cells, 24)
        self.assertEqual(agent.fallback_solver.max_component_cells, 24)


if __name__ == "__main__":
    unittest.main()
