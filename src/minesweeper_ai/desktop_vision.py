"""Screen-grid geometry and Classic/HD Minesweeper Online recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .game import FLAGGED, UNKNOWN

try:
    from PIL import Image, ImageGrab
except ImportError as error:  # pragma: no cover - exercised only without desktop extra
    raise RuntimeError(
        "Desktop mode requires Pillow; install with `pip install -e .[desktop]`."
    ) from error


TerminalState = Literal["won", "lost"]
Theme = Literal["auto", "light", "dark"]


@dataclass(frozen=True)
class GridGeometry:
    left: int
    top: int
    right: int
    bottom: int
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be positive")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("grid rectangle must have positive size")

    @classmethod
    def from_corner_centres(
        cls,
        first: tuple[int, int],
        last: tuple[int, int],
        *,
        rows: int,
        cols: int,
    ) -> "GridGeometry":
        if rows < 2 or cols < 2:
            raise ValueError("desktop calibration requires at least 2 rows and 2 columns")
        step_x = (last[0] - first[0]) / (cols - 1)
        step_y = (last[1] - first[1]) / (rows - 1)
        if step_x <= 2 or step_y <= 2:
            raise ValueError("corner centres are reversed or too close")
        aspect = step_x / step_y
        if not 0.72 <= aspect <= 1.38:
            raise ValueError(
                "selected cells do not form a square grid; check rows/columns and click again"
            )
        # Browser tiles occupy an integer number of screen pixels. Treat the
        # first click as the anchor and estimate one square cell size. A small
        # last-click error can no longer stretch the final crop into the frame.
        cell_size = max(3, round((step_x + step_y) / 2))
        left = int(first[0]) - cell_size // 2
        top = int(first[1]) - cell_size // 2
        return cls(
            left,
            top,
            left + cols * cell_size,
            top + rows * cell_size,
            rows,
            cols,
        )

    def cell_box(self, row: int, col: int) -> tuple[int, int, int, int]:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError((row, col))
        width = self.right - self.left
        height = self.bottom - self.top
        x0 = self.left + round(width * col / self.cols)
        x1 = self.left + round(width * (col + 1) / self.cols)
        y0 = self.top + round(height * row / self.rows)
        y1 = self.top + round(height * (row + 1) / self.rows)
        return x0, y0, x1, y1

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        x0, y0, x1, y1 = self.cell_box(row, col)
        return (x0 + x1) // 2, (y0 + y1) // 2


@dataclass(frozen=True)
class CellRead:
    value: int
    allowed: bool
    confidence: float
    terminal: TerminalState | None = None
    start: bool = False


@dataclass(frozen=True)
class BoardRead:
    observation: tuple[tuple[int, ...], ...]
    allowed: tuple[tuple[bool, ...], ...]
    confidence: float
    terminal: TerminalState | None
    errors: tuple[str, ...] = ()
    start_cell: tuple[int, int] | None = None


LIGHT_NUMBER_COLOURS: dict[int, tuple[tuple[int, int, int], ...]] = {
    1: ((0, 0, 247),),
    2: ((0, 119, 0),),
    3: ((236, 0, 0),),
    4: ((0, 0, 128),),
    5: ((128, 0, 0),),
    6: ((0, 128, 128),),
    7: ((0, 0, 0),),
    8: ((112, 112, 112),),
}

# The site exposes neutral and slightly blue-tinted Classic/HD dark palettes.
DARK_NUMBER_COLOURS: dict[int, tuple[tuple[int, int, int], ...]] = {
    1: ((102, 164, 221), (124, 199, 255)),
    2: ((80, 160, 80), (102, 194, 102)),
    3: ((204, 102, 119), (255, 119, 136)),
    4: ((187, 119, 221), (238, 136, 255)),
    5: ((170, 153, 0), (221, 170, 34)),
    6: ((85, 170, 170), (102, 204, 204)),
    7: ((153, 153, 153), (136, 136, 136)),
    8: ((204, 204, 204), (208, 216, 224)),
}


class HdSkinRecognizer:
    """Recognize the site's light or dark Classic/HD skin.

    Recognition intentionally uses only visible pixels. Custom skins, browser
    colour filters, non-uniform scaling, and animations should be treated as
    unsupported until a calibration screenshot is supplied.
    """

    def __init__(self, *, colour_tolerance: int = 58, theme: Theme = "dark") -> None:
        if not 10 <= colour_tolerance <= 120:
            raise ValueError("colour_tolerance must be between 10 and 120")
        if theme not in {"auto", "light", "dark"}:
            raise ValueError("theme must be auto, light, or dark")
        self.colour_tolerance = colour_tolerance
        self.theme = theme
        self._cell_cache: dict[
            tuple[str, int, int, int, int, int, int], tuple[bytes, CellRead]
        ] = {}

    def read_screen(self, geometry: GridGeometry, *, total_mines: int) -> BoardRead:
        image = ImageGrab.grab(
            bbox=(geometry.left, geometry.top, geometry.right, geometry.bottom),
            all_screens=True,
        )
        return self.read_image(image, geometry.rows, geometry.cols, total_mines=total_mines)

    def read_image(
        self,
        image: "Image.Image",
        rows: int,
        cols: int,
        *,
        total_mines: int,
    ) -> BoardRead:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be positive")
        values: list[list[int]] = []
        allowed: list[list[bool]] = []
        confidences: list[float] = []
        terminals: list[TerminalState] = []
        starts: list[tuple[int, int]] = []
        for row in range(rows):
            value_row: list[int] = []
            allowed_row: list[bool] = []
            for col in range(cols):
                x0 = round(image.width * col / cols)
                x1 = round(image.width * (col + 1) / cols)
                y0 = round(image.height * row / rows)
                y1 = round(image.height * (row + 1) / rows)
                cell = image.crop((x0, y0, x1, y1))
                cache_key = (
                    image.mode, image.width, image.height, rows, cols, row, col
                )
                signature = cell.tobytes()
                cached = self._cell_cache.get(cache_key)
                if cached is not None and cached[0] == signature:
                    result = cached[1]
                else:
                    result = self.classify_cell(cell)
                    self._cell_cache[cache_key] = (signature, result)
                value_row.append(result.value)
                allowed_row.append(result.allowed)
                confidences.append(result.confidence)
                if result.terminal:
                    terminals.append(result.terminal)
                if result.start:
                    starts.append((row, col))
            values.append(value_row)
            allowed.append(allowed_row)

        observation = tuple(tuple(row) for row in values)
        allowed_grid = tuple(tuple(row) for row in allowed)
        covered = sum(cell in (UNKNOWN, FLAGGED) for row in observation for cell in row)
        revealed = rows * cols - covered
        terminal: TerminalState | None = "lost" if "lost" in terminals else None
        if terminal is None and revealed > 0 and covered == total_mines:
            terminal = "won"
        errors: list[str] = []
        confidence = min(confidences, default=0.0)
        if confidence < 0.35:
            errors.append("one or more cells have low visual confidence")
        if len(starts) > 1:
            errors.append("multiple no-guess start markers detected")
        return BoardRead(
            observation, allowed_grid, confidence, terminal, tuple(errors),
            starts[0] if len(starts) == 1 else None,
        )

    def classify_cell(self, image: "Image.Image") -> CellRead:
        sample = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        pixels = np.asarray(sample, dtype=np.int32)
        tolerance = self.colour_tolerance

        def colour_count(colour: Sequence[int], custom_tolerance: int | None = None) -> int:
            threshold = tolerance if custom_tolerance is None else custom_tolerance
            target = np.asarray(colour, dtype=np.int32)
            distance = np.sqrt(np.sum((pixels - target) ** 2, axis=2))
            return int(np.count_nonzero(distance <= threshold))

        mean_brightness = float(pixels.mean())
        theme = self.theme
        if theme == "auto":
            theme = "dark" if mean_brightness < 135 else "light"
        if theme == "dark":
            return self._classify_dark(pixels, colour_count)
        return self._classify_light(pixels, colour_count)

    def _classify_light(self, pixels, colour_count) -> CellRead:
        white = colour_count((255, 255, 255), 42)
        red = max(colour_count((255, 0, 0), 70), colour_count((231, 0, 48), 55))
        dark = int(np.count_nonzero(np.max(pixels, axis=2) < 68))
        mean_brightness = float(pixels.mean())
        raised = white >= 18
        green = colour_count((102, 221, 102), 48)
        if _has_broad_red_fill(
            pixels,
            (((255, 0, 0), 70), ((231, 0, 48), 55)),
        ):
            return CellRead(UNKNOWN, True, min(1.0, red / 600), "lost")
        if raised:
            if red >= 5 and dark >= 3:
                return CellRead(FLAGGED, True, min(1.0, (red + dark) / 35))
            if green >= 5:
                return CellRead(UNKNOWN, True, min(1.0, 0.65 + green / 100), start=True)
            return CellRead(UNKNOWN, True, min(1.0, white / 45))
        if mean_brightness < 92:
            return CellRead(UNKNOWN, False, 0.75)
        return self._read_number_or_zero(pixels, LIGHT_NUMBER_COLOURS, dark)

    def _classify_dark(self, pixels, colour_count) -> CellRead:
        luminance = pixels.mean(axis=2)
        top_left = float(luminance[2:8, 2:8].mean())
        bottom_right = float(luminance[24:30, 24:30].mean())
        centre_rgb = pixels[9:23, 9:23].mean(axis=(0, 1))
        centre = float(centre_rgb.mean())
        raised = top_left - bottom_right >= 14.0
        red = max(
            colour_count((221, 80, 80), 58),
            colour_count((247, 80, 80), 58),
            colour_count((204, 102, 102), 52),
            colour_count((238, 102, 102), 52),
        )
        black = int(np.count_nonzero(np.max(pixels, axis=2) < 24))
        if _has_broad_red_fill(
            pixels,
            (((221, 80, 80), 58), ((247, 80, 80), 58),
             ((204, 102, 102), 52), ((238, 102, 102), 52)),
        ):
            return CellRead(UNKNOWN, True, min(1.0, red / 600), "lost")
        green = max(
            colour_count((102, 221, 102), 44),
            colour_count((87, 151, 95), 36),
        )
        if raised:
            if red >= 5:
                return CellRead(FLAGGED, True, min(1.0, 0.62 + red / 80))
            if green >= 5:
                return CellRead(UNKNOWN, True, min(1.0, 0.65 + green / 100), start=True)
            blue_tinted = float(centre_rgb[2] - centre_rgb[0]) >= 5.0
            blackout_threshold = 77.5 if blue_tinted else 63.5
            allowed = centre >= blackout_threshold
            return CellRead(UNKNOWN, allowed, 0.82 if allowed else 0.72)
        # A revealed, non-exploded mine can also appear on a completed winning
        # board. Preserve it as a non-clue cell; infer win/loss at board level.
        mine_ink = np.max(pixels, axis=2) < 40
        if int(mine_ink[9:23, 9:23].sum()) >= 80 and 100 <= int(mine_ink.sum()) <= 320:
            return CellRead(UNKNOWN, True, 0.95)
        return self._read_number_or_zero(
            pixels, DARK_NUMBER_COLOURS, black, dark_theme=True
        )

    def _read_number_or_zero(
        self,
        pixels,
        palettes: dict[int, tuple[tuple[int, int, int], ...]],
        dark_pixels: int,
        *,
        dark_theme: bool = False,
    ) -> CellRead:
        interior = pixels[2:-2, 2:-2]

        def interior_count(colour, tolerance: int) -> int:
            target = np.asarray(colour, dtype=np.int32)
            distance = np.sqrt(np.sum((interior - target) ** 2, axis=2))
            return int(np.count_nonzero(distance <= tolerance))

        scores = {
            number: max(
                interior_count(colour, 45 if dark_theme else (48 if number != 7 else 34))
                for colour in colours
            )
            for number, colours in palettes.items()
        }
        number, score = max(scores.items(), key=lambda item: (item[1], -item[0]))
        if score >= 4:
            if not dark_theme and number == 7 and dark_pixels >= 130:
                return CellRead(UNKNOWN, True, min(1.0, dark_pixels / 220), "lost")
            return CellRead(number, True, min(1.0, 0.55 + score / 32))
        return CellRead(0, True, 0.88)


def _has_broad_red_fill(
    pixels: np.ndarray,
    candidates: Sequence[tuple[Sequence[int], int]],
) -> bool:
    """Distinguish an exploded red tile background from a red 3 or flag.

    Number glyphs can exceed a simple red-pixel threshold after antialiasing.
    An exploded cell differs because red covers both a large part of the tile
    and its outer band.
    """
    outer = np.ones(pixels.shape[:2], dtype=bool)
    outer[7:25, 7:25] = False
    for colour, tolerance in candidates:
        target = np.asarray(colour, dtype=np.int32)
        distance = np.sqrt(np.sum((pixels - target) ** 2, axis=2))
        mask = distance <= tolerance
        if np.count_nonzero(mask) >= 300 and np.count_nonzero(mask & outer) >= 100:
            return True
    return False


def format_observation(observation: Sequence[Sequence[int]]) -> str:
    symbols = {UNKNOWN: "■", FLAGGED: "⚑", 0: "·"}
    return "\n".join(" ".join(symbols.get(cell, str(cell)) for cell in row) for row in observation)
