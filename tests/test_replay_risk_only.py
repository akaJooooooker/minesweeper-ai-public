import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.replay import (
    ReplayTrainingExample,
    read_replay_examples,
    write_replay_examples,
)
from minesweeper_ai.replay_risk_only import build_risk_only_dataset


def _example(game_id: str, risk: bool) -> ReplayTrainingExample:
    return ReplayTrainingExample(
        game_id=game_id,
        observation=[[1, -1]],
        total_mines=1,
        mode="standard",
        source="test",
        action="reveal",
        row=0,
        col=1,
        time_ms=0,
        classification="NECESSARY_GUESS",
        teacher_exact=True,
        mine_probability=0.5,
        probabilities=[[None, 0.5]],
        risk_mask=[[0, int(risk)]],
        policy_weight=1.0,
        value_target=1.0,
        value_weight=1.0,
    )


class ReplayRiskOnlyTests(unittest.TestCase):
    def test_keeps_risk_examples_and_disables_other_heads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "risk.jsonl"
            write_replay_examples(
                [_example("selected", True), _example("ignored", False)],
                source,
            )
            stats = build_risk_only_dataset([source], output)
            examples = list(read_replay_examples(output))

        self.assertEqual(stats.inputs, 2)
        self.assertEqual(stats.selected, 1)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].policy_weight, 0.0)
        self.assertEqual(examples[0].value_weight, 0.0)


if __name__ == "__main__":
    unittest.main()

