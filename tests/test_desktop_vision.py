import unittest
from unittest import mock

from PIL import Image, ImageDraw

from minesweeper_ai.desktop_vision import GridGeometry, HdSkinRecognizer
from minesweeper_ai.game import FLAGGED, UNKNOWN


def dark_raised(*, blackout: bool = False, flag: bool = False) -> Image.Image:
    image = Image.new("RGB", (32, 32), "#202020")
    draw = ImageDraw.Draw(image)
    bright = "#535353" if blackout else "#606060"
    centre = "#3b3b3b" if blackout else "#444444"
    draw.polygon(((0, 0), (31, 0), (20, 20), (0, 31)), fill=bright)
    draw.rectangle((4, 4, 27, 27), fill=centre)
    if flag:
        draw.rectangle((15, 9, 17, 23), fill="#c7c7c7")
        draw.polygon(((8, 9), (17, 7), (17, 17)), fill="#dd5050")
    return image


def dark_revealed(number_colour: str | None = None) -> Image.Image:
    image = Image.new("RGB", (32, 32), "#202020")
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 31, 31), fill="#343434")
    if number_colour:
        draw.rectangle((13, 8, 18, 24), fill=number_colour)
    return image


class DesktopVisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recognizer = HdSkinRecognizer(theme="dark", colour_tolerance=32)

    def test_geometry_from_first_and_last_centres(self) -> None:
        geometry = GridGeometry.from_corner_centres((110, 210), (290, 370), rows=9, cols=10)
        self.assertEqual(geometry.cell_center(0, 0), (110, 210))
        self.assertEqual(geometry.cell_center(8, 9), (290, 370))

    def test_geometry_snaps_imprecise_last_click_to_square_integer_cells(self) -> None:
        geometry = GridGeometry.from_corner_centres(
            (-2306, 465), (-1256, 1013), rows=16, cols=30
        )
        self.assertEqual(geometry.right - geometry.left, 30 * 36)
        self.assertEqual(geometry.bottom - geometry.top, 16 * 36)
        self.assertEqual(geometry.cell_center(0, 0), (-2306, 465))
        self.assertEqual(geometry.cell_center(15, 29), (-1262, 1005))

    def test_geometry_rejects_wrong_row_or_column_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "square grid"):
            GridGeometry.from_corner_centres(
                (110, 210), (290, 900), rows=16, cols=30
            )

    def test_unchanged_cells_are_reused_from_cache(self) -> None:
        image = Image.new("RGB", (64, 64))
        for row in range(2):
            for col in range(2):
                image.paste(dark_raised(), (col * 32, row * 32))
        with mock.patch.object(
            self.recognizer, "classify_cell", wraps=self.recognizer.classify_cell
        ) as classify:
            self.recognizer.read_image(image, 2, 2, total_mines=1)
            self.recognizer.read_image(image, 2, 2, total_mines=1)
            self.assertEqual(classify.call_count, 4)
            image.putpixel((1, 1), (255, 255, 255))
            self.recognizer.read_image(image, 2, 2, total_mines=1)
            self.assertEqual(classify.call_count, 5)

    def test_dark_closed_and_blackout_are_distinguished(self) -> None:
        closed = self.recognizer.classify_cell(dark_raised())
        blackout = self.recognizer.classify_cell(dark_raised(blackout=True))
        self.assertEqual(closed.value, UNKNOWN)
        self.assertTrue(closed.allowed)
        self.assertEqual(blackout.value, UNKNOWN)
        self.assertFalse(blackout.allowed)

    def test_dark_flag_and_revealed_number(self) -> None:
        flag = self.recognizer.classify_cell(dark_raised(flag=True))
        one = self.recognizer.classify_cell(dark_revealed("#66a4dd"))
        self.assertEqual(flag.value, FLAGGED)
        self.assertEqual(one.value, 1)

    def test_dark_zero_ignores_bright_outer_board_edge(self) -> None:
        image = dark_revealed()
        ImageDraw.Draw(image).line((0, 31, 31, 31), fill="#727a82", width=1)
        result = self.recognizer.classify_cell(image)
        self.assertEqual(result.value, 0)
        self.assertIsNone(result.terminal)

    def test_dark_green_x_is_prescribed_start_cell(self) -> None:
        image = dark_raised()
        draw = ImageDraw.Draw(image)
        draw.line((9, 9, 22, 22), fill="#66dd66", width=4)
        draw.line((22, 9, 9, 22), fill="#66dd66", width=4)
        result = self.recognizer.classify_cell(image)
        self.assertTrue(result.start)

    def test_dark_red_three_is_not_mistaken_for_explosion(self) -> None:
        image = dark_revealed()
        draw = ImageDraw.Draw(image)
        red = "#ee6666"
        draw.line((8, 8, 24, 8), fill=red, width=4)
        draw.line((21, 8, 24, 24), fill=red, width=4)
        draw.line((10, 16, 24, 16), fill=red, width=4)
        draw.line((8, 24, 24, 24), fill=red, width=4)
        result = HdSkinRecognizer(theme="dark", colour_tolerance=58).classify_cell(image)
        self.assertEqual(result.value, 3)
        self.assertIsNone(result.terminal)

    def test_dark_revealed_zero_and_exploded_mine(self) -> None:
        zero = self.recognizer.classify_cell(dark_revealed())
        mine = dark_revealed()
        ImageDraw.Draw(mine).rectangle((3, 3, 29, 29), fill="#cc6666")
        exploded = self.recognizer.classify_cell(mine)
        self.assertEqual(zero.value, 0)
        self.assertEqual(exploded.terminal, "lost")


if __name__ == "__main__":
    unittest.main()
