import unittest


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class ModelTests(unittest.TestCase):
    def test_model_accepts_multiple_board_shapes(self) -> None:
        from minesweeper_ai.model import MinesweeperNet

        model = MinesweeperNet(width=32)
        for height, width in ((9, 9), (16, 30), (12, 17)):
            inputs = torch.zeros((1, 13, height, width))
            mask = torch.ones((1, height, width))
            output = model(inputs, mask)
            self.assertEqual(tuple(output.shape), (1, height, width))
            self.assertTrue(bool(torch.isfinite(output).all()))


if __name__ == "__main__":
    unittest.main()
