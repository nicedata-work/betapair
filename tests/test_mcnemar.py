"""Conformance against DFBA::dfba_mcnemar and the properties the extension relies on."""

from __future__ import annotations

import math
import unittest

from pydfba.mcnemar import McNemarResult, mcnemar


class DfbaConformanceTest(unittest.TestCase):
    """DFBA vignette example: n_01 = 17, n_10 = 2, uniform prior."""

    def setUp(self) -> None:
        self.result = mcnemar(17, 2)

    def test_posterior_shapes(self) -> None:
        self.assertEqual(self.result.a_post, 18.0)
        self.assertEqual(self.result.b_post, 3.0)

    def test_point_estimates(self) -> None:
        self.assertAlmostEqual(self.result.post_mean, 0.857143, places=6)
        self.assertAlmostEqual(self.result.post_median, 0.8685263, places=6)

    def test_equal_tail_interval(self) -> None:
        self.assertAlmostEqual(self.result.eti_low, 0.683017, places=6)
        self.assertAlmostEqual(self.result.eti_high, 0.9679291, places=6)

    def test_bayes_factors(self) -> None:
        self.assertAlmostEqual(self.result.bf10_point, 153.3006, places=4)
        self.assertAlmostEqual(self.result.bf10_interval, 4968.555, places=3)

    def test_posterior_probability(self) -> None:
        self.assertAlmostEqual(self.result.p_improved, 0.9997988, places=7)

    def test_jeffreys_prior_variant(self) -> None:
        result = mcnemar(17, 2, a0=0.5, b0=0.5, prob_interval=0.99)
        self.assertEqual(result.a_post, 17.5)
        self.assertEqual(result.b_post, 2.5)
        self.assertLess(result.eti_low, self.result.eti_low)
        self.assertGreater(result.eti_high, self.result.eti_high)


class McNemarPropertiesTest(unittest.TestCase):
    def test_no_discordant_pairs_is_uninformative(self) -> None:
        result = mcnemar(0, 0)
        self.assertEqual(result.post_mean, 0.5)
        self.assertAlmostEqual(result.p_improved, 0.5, places=12)
        self.assertAlmostEqual(result.bf10_point, 1.0, places=12)
        self.assertEqual(result.discordant, 0)
        self.assertTrue(math.isnan(result.switched_up))

    def test_swapped_counts_invert_the_evidence(self) -> None:
        forward = mcnemar(12, 3)
        reverse = mcnemar(3, 12)
        self.assertAlmostEqual(forward.p_improved, 1.0 - reverse.p_improved, places=10)
        self.assertAlmostEqual(forward.bf10_interval, 1.0 / reverse.bf10_interval, places=6)
        # The point-null density at .5 is symmetric in the shapes.
        self.assertAlmostEqual(forward.bf10_point, reverse.bf10_point, places=9)

    def test_more_improvements_never_lower_the_evidence(self) -> None:
        probabilities = [mcnemar(k, 5).p_improved for k in range(0, 16)]
        self.assertEqual(probabilities, sorted(probabilities))

    def test_unchanged_pairs_do_not_enter_the_test(self) -> None:
        """Only the discordant sub-population is analysed (McNemar's core)."""
        self.assertEqual(mcnemar(4, 2).post_mean, mcnemar(4, 2).post_mean)
        result = mcnemar(4, 2)
        self.assertEqual(result.discordant, 6)
        self.assertAlmostEqual(result.switched_up, 4 / 6, places=12)

    def test_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            mcnemar(-1, 2)
        with self.assertRaises(ValueError):
            mcnemar(1, 2, a0=0.0)
        with self.assertRaises(ValueError):
            mcnemar(1, 2, prob_interval=1.0)

    def test_result_is_frozen(self) -> None:
        result = mcnemar(5, 1)
        self.assertIsInstance(result, McNemarResult)
        with self.assertRaises(AttributeError):
            result.n_improved = 6  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
