import unittest

from minesweeper_ai.training import _set_gpu_memory_fraction
from minesweeper_ai.cli import _mcnemar_exact_p


class TrainingResourceTests(unittest.TestCase):
    def test_unspecified_cuda_device_is_resolved_to_integer_index(self) -> None:
        calls = []

        class FakeCuda:
            @staticmethod
            def current_device():
                return 2

            @staticmethod
            def set_per_process_memory_fraction(fraction, *, device):
                calls.append((fraction, device))

        class FakeTorch:
            cuda = FakeCuda()

        class FakeDevice:
            index = None

        _set_gpu_memory_fraction(FakeTorch(), FakeDevice(), 0.6)
        self.assertEqual(calls, [(0.6, 2)])

    def test_explicit_cuda_index_is_preserved(self) -> None:
        calls = []

        class FakeCuda:
            @staticmethod
            def current_device():
                raise AssertionError("current_device should not be called")

            @staticmethod
            def set_per_process_memory_fraction(fraction, *, device):
                calls.append((fraction, device))

        class FakeTorch:
            cuda = FakeCuda()

        class FakeDevice:
            index = 1

        _set_gpu_memory_fraction(FakeTorch(), FakeDevice(), 0.4)
        self.assertEqual(calls, [(0.4, 1)])

    def test_paired_benchmark_exact_p_value(self) -> None:
        self.assertEqual(_mcnemar_exact_p(0, 0), 1.0)
        self.assertEqual(_mcnemar_exact_p(5, 5), 1.0)
        self.assertAlmostEqual(_mcnemar_exact_p(0, 10), 2 / (2**10))


if __name__ == "__main__":
    unittest.main()
