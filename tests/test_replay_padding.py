import unittest

from minesweeper_ai.replay_learning import REPLAY_FEATURE_CHANNELS, _torch_model


class ReplayPaddingTests(unittest.TestCase):
    def test_small_board_prediction_is_invariant_to_batch_padding(self) -> None:
        try:
            torch, ReplayAgentNet = _torch_model()
        except RuntimeError:
            self.skipTest("optional PyTorch dependency is not installed")
        torch.manual_seed(4)
        model = ReplayAgentNet(16).eval()
        small = torch.randn((1, REPLAY_FEATURE_CHANNELS, 3, 4))
        small_valid = torch.ones((1, 3, 4))
        padded = torch.zeros((2, REPLAY_FEATURE_CHANNELS, 7, 9))
        padded[0, :, :3, :4] = small[0]
        padded[1] = torch.randn((REPLAY_FEATURE_CHANNELS, 7, 9))
        padded_valid = torch.zeros((2, 7, 9))
        padded_valid[0, :3, :4] = 1
        padded_valid[1] = 1
        with torch.inference_mode():
            alone = model(small, small_valid)
            batched = model(padded, padded_valid)
        for key in ("risk", "standard_policy", "no_guess_policy"):
            expected = alone[key][0]
            actual = batched[key][0, ..., :3, :4]
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(batched["value"][0], alone["value"][0], rtol=1e-4, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
