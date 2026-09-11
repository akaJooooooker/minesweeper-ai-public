"""Checkpoint-backed risk predictor used only when exact enumeration is too large."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .features import encode_board
from .game import UNKNOWN, Coord, validate_observation


class NeuralRiskPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("Neural inference requires the optional PyTorch dependency") from error

        from .model import MinesweeperNet

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.device = torch.device(selected_device)
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=self.device,
            weights_only=True,
        )
        self.model = MinesweeperNet(checkpoint["model_width"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(
        self,
        observation: Sequence[Sequence[int]],
        total_mines: int,
    ) -> Mapping[Coord, float]:
        height, width = validate_observation(observation)
        features, valid = encode_board(observation, total_mines)
        torch = self._torch
        inputs = torch.from_numpy(features).unsqueeze(0).to(self.device)
        valid_mask = torch.from_numpy(valid).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(inputs, valid_mask))[0].cpu()
        return {
            (row, col): float(probabilities[row, col])
            for row in range(height)
            for col in range(width)
            if observation[row][col] == UNKNOWN
        }

