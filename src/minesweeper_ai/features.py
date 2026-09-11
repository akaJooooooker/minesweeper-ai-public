"""Board encoding shared by training and future inference adapters."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .game import FLAGGED, UNKNOWN, validate_observation

FEATURE_CHANNELS = 13


def encode_board(
    observation: Sequence[Sequence[int]],
    total_mines: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a board as ``[channels, height, width]`` float32 features.

    Channels 0..8 are clue one-hot planes, 9 is unknown, 10 is flagged,
    11 is the active frontier, and 12 is the remaining-mine ratio.
    """
    height, width = validate_observation(observation)
    array = np.asarray(observation, dtype=np.int8)
    features = np.zeros((FEATURE_CHANNELS, height, width), dtype=np.float32)
    for clue in range(9):
        features[clue] = array == clue
    features[9] = array == UNKNOWN
    features[10] = array == FLAGGED

    frontier = np.zeros((height, width), dtype=np.float32)
    for row in range(height):
        for col in range(width):
            if array[row, col] != UNKNOWN:
                continue
            for next_row in range(max(0, row - 1), min(height, row + 2)):
                for next_col in range(max(0, col - 1), min(width, col + 2)):
                    if array[next_row, next_col] >= 0:
                        frontier[row, col] = 1.0
                        break
                if frontier[row, col]:
                    break
    features[11] = frontier

    flags = int(np.count_nonzero(array == FLAGGED))
    unknown = int(np.count_nonzero(array == UNKNOWN))
    remaining = max(0, total_mines - flags)
    features[12].fill(remaining / unknown if unknown else 0.0)
    valid_mask = np.ones((height, width), dtype=np.float32)
    return features, valid_mask

