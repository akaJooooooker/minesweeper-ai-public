"""Optional PyTorch fully-convolutional probability model."""

from __future__ import annotations

from .features import FEATURE_CHANNELS

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # Core solver deliberately works without PyTorch.
    torch = None
    nn = None


if nn is not None:

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
            )
            self.activation = nn.SiLU()

        def forward(self, inputs):
            return self.activation(inputs + self.body(inputs))


    class MinesweeperNet(nn.Module):
        """Variable-height/width baseline with local and global context."""

        def __init__(self, width: int = 64) -> None:
            super().__init__()
            if width % 8:
                raise ValueError("model width must be divisible by 8")
            self.width = width
            self.stem = nn.Sequential(
                nn.Conv2d(FEATURE_CHANNELS, width, 3, padding=1),
                nn.GroupNorm(8, width),
                nn.SiLU(),
            )
            self.blocks = nn.Sequential(
                *(ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8, 1, 2))
            )
            self.global_projection = nn.Sequential(
                nn.Conv2d(width, width, 1),
                nn.SiLU(),
            )
            self.head = nn.Sequential(
                nn.Conv2d(width, width // 2, 1),
                nn.SiLU(),
                nn.Conv2d(width // 2, 1, 1),
            )

        def forward(self, inputs, valid_mask):
            hidden = self.blocks(self.stem(inputs))
            mask = valid_mask.unsqueeze(1)
            denominator = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            global_context = (hidden * mask).sum(dim=(2, 3), keepdim=True) / denominator
            hidden = hidden + self.global_projection(global_context)
            return self.head(hidden).squeeze(1)

else:

    class MinesweeperNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is optional and not installed. Install the project with: "
                "python -m pip install -e '.[ml]'"
            )

