import unittest

from minesweeper_ai.deployment_compare import _mcnemar_exact_p


class DeploymentCompareTests(unittest.TestCase):
    def test_exact_p_value(self) -> None:
        self.assertEqual(_mcnemar_exact_p(0, 0), 1.0)
        self.assertEqual(_mcnemar_exact_p(5, 5), 1.0)
        self.assertAlmostEqual(_mcnemar_exact_p(0, 10), 2 / (2**10))


if __name__ == "__main__":
    unittest.main()
