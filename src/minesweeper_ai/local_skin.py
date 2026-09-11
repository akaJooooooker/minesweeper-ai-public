"""Stable local tile art compatible with the existing dark HD recognizer."""
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLOURS = {1: "#66a4dd", 2: "#50a050", 3: "#cc6677", 4: "#bb77dd",
           5: "#aa9900", 6: "#55aaaa", 7: "#999999", 8: "#cccccc"}


@lru_cache(maxsize=256)
def tile_image(value: int | str, size: int = 32) -> Image.Image:
    # Draw at fixed resolution to preserve the recognizer's colour/shape rules.
    image = Image.new("RGB", (32, 32), "#30353c")
    draw = ImageDraw.Draw(image)
    if value in (-1, -2):
        draw.rectangle((0, 0, 31, 31), fill="#606870")
        draw.polygon([(0, 0), (31, 0), (26, 5), (5, 5), (5, 26), (0, 31)], fill="#9098a0")
        draw.polygon([(31, 0), (31, 31), (0, 31), (5, 26), (26, 26), (26, 5)], fill="#333940")
        if value == -2:
            draw.line((17, 8, 17, 24), fill="#171b20", width=2)
            draw.polygon([(16, 7), (7, 12), (16, 16)], fill="#dd5050")
            draw.rectangle((10, 24, 23, 26), fill="#171b20")
    else:
        draw.rectangle((0, 0, 31, 31), outline="#242a31")
        if value in ("mine", "exploded"):
            if value == "exploded":
                draw.rectangle((0, 0, 31, 31), fill="#dd5050")
            draw.ellipse((10, 10, 22, 22), fill="#10151a")
            for line in ((16, 6, 16, 26), (6, 16, 26, 16), (9, 9, 23, 23), (9, 23, 23, 9)):
                draw.line(line, fill="#10151a", width=2)
            draw.rectangle((12, 11, 14, 13), fill="#d0d8e0")
        elif value in COLOURS:
            font = None
            for path in ("C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
                if Path(path).exists():
                    font = ImageFont.truetype(path, 23)
                    break
            font = font or ImageFont.load_default(size=23)
            draw.text((16, 16), str(value), font=font, fill=COLOURS[value], anchor="mm")
    return image.resize((size, size), Image.Resampling.NEAREST)
