import tempfile
import unittest
from pathlib import Path

import numpy as np

from minesweeper_ai.game import FLAGGED, UNKNOWN
from minesweeper_ai.data import BoardSpec
from minesweeper_ai.replay import (
    ReplayAuditor,
    ReplayGame,
    ReplayMove,
    ReplayTrainingExample,
    generate_simulator_replays,
    read_replay_examples,
    read_replays,
    write_replay_examples,
    write_replays,
)
from minesweeper_ai.replay_learning import (
    REPLAY_FEATURE_CHANNELS,
    _prepare_example,
    _torch_model,
    encode_replay_board,
    train_replay_agent,
)
from minesweeper_ai.solver import ConstraintSolver


class ReplayAuditTests(unittest.TestCase):
    def test_flag_and_chord_are_independently_proven(self) -> None:
        replay = ReplayGame(
            game_id="manual-1",
            width=4,
            height=2,
            mines=2,
            mode="standard",
            source="human",
            won=False,
            mine_positions=[[0, 0], [0, 3]],
            moves=[
                ReplayMove("reveal", 1, 0),
                ReplayMove("reveal", 1, 1),
                ReplayMove("reveal", 1, 2),
                ReplayMove("reveal", 1, 3),
                ReplayMove("flag", 0, 3),
                ReplayMove("chord", 1, 3),
            ],
        )
        examples = list(ReplayAuditor(ConstraintSolver(20)).audit(replay))
        self.assertEqual(len(examples), 6)
        self.assertEqual(examples[4].classification, "PROVEN_GOOD")
        self.assertEqual(examples[5].classification, "PROVEN_GOOD")
        self.assertEqual(examples[4].mine_probability, 1.0)
        self.assertEqual(examples[5].risk_mask[0][3], 0)
        self.assertIsNone(examples[5].probabilities[0][3])

    def test_simulator_replay_round_trip_and_audit(self) -> None:
        replays = list(
            generate_simulator_replays(
                [BoardSpec(4, 4, 2, True)],
                1,
                mode="standard",
                seed=3,
                solver=ConstraintSolver(20),
            )
        )
        self.assertEqual(len(replays), 1)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            dataset = Path(directory) / "dataset.jsonl"
            write_replays(replays, raw)
            loaded = list(read_replays(raw))
            examples = [item for replay in loaded for item in ReplayAuditor().audit(replay)]
            write_replay_examples(examples, dataset)
            reloaded = list(read_replay_examples(dataset))
        self.assertEqual(loaded, replays)
        self.assertEqual(reloaded, examples)
        self.assertGreater(len(examples), 0)

    def test_prepare_example_masks_legacy_flag_risk_labels(self) -> None:
        example = ReplayTrainingExample(
            game_id="legacy-flags",
            observation=[[FLAGGED, UNKNOWN]],
            total_mines=1,
            mode="standard",
            source="human",
            action="reveal",
            row=0,
            col=1,
            time_ms=0,
            classification="NECESSARY_GUESS",
            teacher_exact=True,
            mine_probability=0.0,
            probabilities=[[1.0, 0.0]],
            risk_mask=[[1, 1]],
            policy_weight=0.0,
            value_target=1.0,
            value_weight=0.0,
        )
        _, _, _, risk_mask, _ = _prepare_example(example)
        self.assertEqual(risk_mask.tolist(), [[0.0, 1.0]])


class ReplayModelTests(unittest.TestCase):
    def test_replay_encoding_has_mode_and_source_planes(self) -> None:
        observation = ((0, 1, -1), (0, 1, -1))
        features, valid = encode_replay_board(
            observation,
            1,
            mode="no_guess",
            source="simulator_teacher",
        )
        self.assertEqual(features.shape, (REPLAY_FEATURE_CHANNELS, 2, 3))
        self.assertTrue(np.all(features[14] == 1.0))
        self.assertTrue(np.all(features[17] == 0.0))
        self.assertTrue(np.all(valid == 1.0))

    def test_multi_head_model_shapes(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        model = ReplayAgentNet(32)
        output = model(
            torch.zeros((2, REPLAY_FEATURE_CHANNELS, 9, 11)),
            torch.ones((2, 9, 11)),
        )
        self.assertEqual(tuple(output["risk"].shape), (2, 9, 11))
        self.assertEqual(tuple(output["standard_policy"].shape), (2, 3, 9, 11))
        self.assertEqual(tuple(output["no_guess_policy"].shape), (2, 3, 9, 11))
        self.assertEqual(tuple(output["value"].shape), (2,))

    def test_cpu_smoke_training_writes_replay_checkpoint(self) -> None:
        try:
            _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        replay = ReplayGame(
            game_id="train-1",
            width=3,
            height=2,
            mines=1,
            mode="standard",
            source="human",
            won=True,
            mine_positions=[[0, 0]],
            moves=[ReplayMove("reveal", 1, 1), ReplayMove("flag", 0, 0)],
        )
        examples = list(ReplayAuditor(ConstraintSolver(20)).audit(replay))
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            checkpoint = Path(directory) / "replay.pt"
            write_replay_examples(examples, dataset)
            history = train_replay_agent(
                dataset,
                checkpoint,
                epochs=1,
                batch_size=2,
                model_width=16,
                device_name="cpu",
            )
            self.assertTrue(checkpoint.exists())
            self.assertEqual(len(history), 1)
            self.assertTrue(np.isfinite(history[0]["total"]))


if __name__ == "__main__":
    unittest.main()
