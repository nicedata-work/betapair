"""Beta numerics: reference values, inverses, and the closed-form marginals."""

from __future__ import annotations

import math
import random
import unittest

from betapair.beta import (
    beta_cdf,
    beta_mean,
    beta_pdf,
    beta_ppf,
    beta_sample,
    beta_sf,
    log_beta_binomial,
    log_choose,
)


class BetaQuantileTest(unittest.TestCase):
    def test_matches_dfba_published_output(self) -> None:
        """DFBA::dfba_mcnemar(n_01 = 17, n_10 = 2) prints these quantiles."""
        self.assertAlmostEqual(beta_ppf(0.5, 18, 3), 0.8685263, places=6)
        self.assertAlmostEqual(beta_ppf(0.025, 18, 3), 0.683017, places=6)
        self.assertAlmostEqual(beta_ppf(0.975, 18, 3), 0.9679291, places=6)

    def test_ppf_inverts_cdf(self) -> None:
        for a, b in ((2.0, 5.0), (18.0, 3.0), (101.0, 101.0), (0.5, 0.5)):
            for p in (0.025, 0.5, 0.9, 0.975):
                self.assertAlmostEqual(beta_cdf(beta_ppf(p, a, b), a, b), p, places=9)

    def test_symmetry(self) -> None:
        for x in (0.1, 0.5, 0.8):
            self.assertAlmostEqual(
                beta_cdf(x, 2.0, 7.0), 1.0 - beta_cdf(1.0 - x, 7.0, 2.0), places=12
            )
        self.assertAlmostEqual(beta_pdf(0.3, 2.0, 7.0), beta_pdf(0.7, 7.0, 2.0), places=12)

    def test_known_density_and_moments(self) -> None:
        self.assertAlmostEqual(beta_pdf(0.5, 2.0, 2.0), 1.5, places=12)
        self.assertAlmostEqual(beta_cdf(0.5, 2.0, 2.0), 0.5, places=12)
        self.assertAlmostEqual(beta_mean(3.0, 5.0), 0.375, places=12)

    def test_boundaries(self) -> None:
        self.assertEqual(beta_cdf(0.0, 3.0, 4.0), 0.0)
        self.assertEqual(beta_cdf(1.0, 3.0, 4.0), 1.0)
        self.assertEqual(beta_pdf(0.0, 1.0, 4.0), 4.0)
        self.assertEqual(beta_pdf(1.0, 4.0, 1.0), 4.0)
        self.assertEqual(beta_pdf(0.0, 2.0, 4.0), 0.0)
        self.assertEqual(beta_pdf(-0.1, 2.0, 4.0), 0.0)

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            beta_ppf(0.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            beta_cdf(0.5, 0.0, 1.0)
        with self.assertRaises(ValueError):
            beta_pdf(0.5, 1.0, -1.0)


class BetaSamplingTest(unittest.TestCase):
    def test_sample_mean_tracks_the_analytic_mean(self) -> None:
        rng = random.Random(11)
        draws = [beta_sample(rng, 3.0, 5.0) for _ in range(4000)]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in draws))
        self.assertAlmostEqual(sum(draws) / len(draws), 3.0 / 8.0, places=2)

    def test_shape_below_one_is_supported(self) -> None:
        rng = random.Random(3)
        draws = [beta_sample(rng, 0.4, 0.6) for _ in range(200)]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in draws))

    def test_tiny_shapes_do_not_collapse_onto_the_mean(self) -> None:
        """Regression: both gamma draws underflowing must not return a/(a+b)."""
        rng = random.Random(99)
        draws = [beta_sample(rng, 1e-3, 1e-3) for _ in range(5000)]
        # The old fallback returned exactly 0.5 whenever both draws underflowed
        # (~24% of the time), i.e. a point mass at the least likely value.
        self.assertEqual(sum(1 for value in draws if value == 0.5), 0)
        # A U-shaped Beta(1e-3, 1e-3) puts only ~0.5% of its mass in the middle.
        middle = sum(1 for value in draws if 0.01 < value < 0.99)
        self.assertLess(middle, 0.05 * len(draws))
        self.assertGreater(sum(1 for value in draws if value < 0.01), 0)
        self.assertGreater(sum(1 for value in draws if value > 0.99), 0)
        self.assertTrue(all(0.0 < value < 1.0 for value in draws))

    def test_invalid_shapes_are_rejected(self) -> None:
        rng = random.Random(0)
        with self.assertRaises(ValueError):
            beta_sample(rng, 0.0, 1.0)
        with self.assertRaises(ValueError):
            beta_sample(rng, 1.0, -1.0)

    def test_near_zero_tail_keeps_its_spread(self) -> None:
        """A clamped tail sits at the double limit, not an arbitrary epsilon."""
        rng = random.Random(4)
        draws = [beta_sample(rng, 0.1, 1.0) for _ in range(5000)]
        self.assertTrue(all(0.0 < value < 1.0 for value in draws))
        tail = [value for value in draws if value < 1e-12]
        self.assertTrue(tail)
        # Sampling in log space keeps the whole tail distinct instead of
        # collapsing every draw below the old 1e-12 clamp onto one value.
        self.assertGreater(len(set(tail)), 1)
        self.assertLess(min(draws), 1e-20)


class BetaSurvivalTest(unittest.TestCase):
    """The memoised upper tail the per-cell verdicts read."""

    def test_matches_the_cdf_complement(self) -> None:
        for a, b in ((0.5, 0.5), (2.0, 5.0), (18.0, 3.0), (101.0, 101.0)):
            for x in (0.1, 0.5, 0.9):
                self.assertAlmostEqual(beta_sf(x, a, b), 1.0 - beta_cdf(x, a, b), places=12)

    def test_reflection(self) -> None:
        self.assertAlmostEqual(beta_sf(0.3, 2.0, 7.0), beta_cdf(0.7, 7.0, 2.0), places=12)

    def test_repeat_calls_are_served_from_cache(self) -> None:
        beta_sf.cache_clear()
        beta_sf(0.5, 12.0, 7.0)
        before = beta_sf.cache_info()
        self.assertEqual(beta_sf(0.5, 12.0, 7.0), beta_sf(0.5, 12.0, 7.0))
        after = beta_sf.cache_info()
        self.assertGreater(after.hits, before.hits)
        self.assertEqual(after.misses, before.misses)

    def test_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            beta_sf(0.5, 0.0, 1.0)


class BetaBinomialTest(unittest.TestCase):
    def test_closed_form_matches_numeric_integration(self) -> None:
        """The hierarchy's marginal is exact: compare it with Simpson quadrature."""
        k, n, a, b = 3, 8, 2.0, 3.0
        steps = 4000
        total = 0.0
        for i in range(steps):
            x = (i + 0.5) / steps
            binomial = math.comb(n, k) * x**k * (1.0 - x) ** (n - k)
            total += binomial * beta_pdf(x, a, b) / steps
        self.assertAlmostEqual(math.exp(log_beta_binomial(k, n, a, b)), total, places=4)

    def test_marginal_is_uniform_for_flat_prior(self) -> None:
        # Beta(1,1) prior: every count pattern has marginal likelihood 1/(n+1).
        for k in range(5):
            self.assertAlmostEqual(math.exp(log_beta_binomial(k, 4, 1.0, 1.0)), 0.2, places=12)

    def test_log_choose(self) -> None:
        self.assertAlmostEqual(math.exp(log_choose(10, 3)), math.comb(10, 3), places=9)

    def test_invalid_counts(self) -> None:
        with self.assertRaises(ValueError):
            log_beta_binomial(5, 4, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
