"""Multi-strata pooling and the homogeneity guardrail.

The guardrail is the safety-critical part: a pooled health score must not be
able to hide a regression in one view behind improvements in the others. These
tests pin that behaviour for homogeneous, globally-regressing and
heterogeneous (masked) layers, and anchor the predictive check's calibration
under the null.
"""

from __future__ import annotations

import math
import random
import unittest

from betapair.beta import beta_sf
from betapair.hierarchy import (
    Cell,
    cell_posteriors,
    fit,
    homogeneity_ppc,
    pooled_counts,
)


def cells(pattern: list[tuple[int, int]], stratum: str = "view") -> list[Cell]:
    """Cells where every pair moved: ``pairs == improved + regressed``."""
    return [
        Cell(
            stratum=f"{stratum}_{index}",
            outcome="answer",
            improved=k,
            regressed=r,
            passed=k,
            pairs=k + r,
        )
        for index, (k, r) in enumerate(pattern)
    ]


class CellTest(unittest.TestCase):
    def test_derived_counts(self) -> None:
        cell = Cell(stratum="s", outcome="o", improved=4, regressed=1, passed=12, pairs=20)
        self.assertEqual(cell.discordant, 5)
        self.assertEqual(cell.unchanged, 15)
        self.assertEqual(cell.baseline_passed, 9)
        self.assertAlmostEqual(cell.pass_rate, 0.6)
        self.assertAlmostEqual(cell.baseline_pass_rate, 0.45)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            Cell(stratum="s", outcome="o", improved=-1, regressed=0, pairs=3)
        with self.assertRaises(ValueError):
            Cell(stratum="s", outcome="o", improved=0, regressed=0, passed=5, pairs=3)
        with self.assertRaises(ValueError):
            Cell(stratum="s", outcome="o", improved=4, regressed=4, pairs=5)

    def test_incoherent_pass_counts_are_rejected(self) -> None:
        """``baseline_passed`` must stay a count, never a negative pass rate."""
        with self.assertRaises(ValueError):
            # passed=0 cannot contain the four 0 -> 1 switches.
            Cell(stratum="s", outcome="o", improved=4, regressed=1, pairs=20)
        with self.assertRaises(ValueError):
            # passed=20 leaves no room for the five 1 -> 0 switches.
            Cell(stratum="s", outcome="o", improved=0, regressed=5, passed=20, pairs=20)
        # The boundary cases are legal.
        self.assertEqual(
            Cell(
                stratum="s", outcome="o", improved=4, regressed=1, passed=4, pairs=20
            ).baseline_passed,
            1,
        )
        self.assertEqual(
            Cell(
                stratum="s", outcome="o", improved=0, regressed=5, passed=15, pairs=20
            ).baseline_passed,
            20,
        )

    def test_pooled_counts(self) -> None:
        self.assertEqual(pooled_counts(cells([(14, 6), (2, 18)])), (16, 24))


class FitTest(unittest.TestCase):
    def test_pooled_rate_recovers_a_single_switch_rate(self) -> None:
        result = fit(cells([(14, 6)] * 10))
        self.assertAlmostEqual(result.mu, 0.7, delta=0.04)
        self.assertLess(result.mu_eti[0], 0.7)
        self.assertGreater(result.mu_eti[1], 0.7)
        self.assertEqual(result.cells, 10)
        self.assertTrue(math.isfinite(result.log_marginal))

    def test_unchanged_pairs_do_not_dilute_the_fit(self) -> None:
        """The fit is over discordant pairs only — bank size must not move ``mu``."""
        short = [
            Cell(stratum=f"v{i}", outcome="answer", improved=14, regressed=6, passed=14, pairs=20)
            for i in range(10)
        ]
        long = [
            Cell(stratum=f"v{i}", outcome="answer", improved=14, regressed=6, passed=88, pairs=100)
            for i in range(10)
        ]
        self.assertAlmostEqual(fit(short).mu, 0.7, delta=0.04)
        # 80 unchanged questions per view change nothing about the switch rate.
        self.assertAlmostEqual(fit(long).mu, fit(short).mu, delta=0.02)
        self.assertAlmostEqual(fit(long).mu, 0.7, delta=0.04)

    def test_regression_moves_mu_below_half(self) -> None:
        result = fit(cells([(6, 14)] * 10))
        self.assertLess(result.mu, 0.5)
        self.assertLess(result.mu_eti[1], 0.5)

    def test_empty_cells_are_ignored(self) -> None:
        with_empty = fit([*cells([(10, 5)] * 3), Cell(stratum="pending", outcome="answer")])
        without = fit(cells([(10, 5)] * 3))
        self.assertEqual(with_empty.cells, 3)
        self.assertAlmostEqual(with_empty.mu, without.mu, places=12)

    def test_deterministic_for_a_given_state(self) -> None:
        payload = cells([(14, 6)] * 4 + [(3, 9)])
        first = fit(payload)
        second = fit(payload)
        self.assertEqual(first.points, second.points)
        self.assertEqual(first.mu, second.mu)

    def test_pruned_grid_still_carries_the_mass(self) -> None:
        """Trimming negligible grid points must not move the posterior."""
        result = fit(cells([(1, 0)] * 2))
        self.assertTrue(0.0 < len(result.points) <= 49 * 24)
        self.assertAlmostEqual(sum(weight for _, _, weight in result.points), 1.0, places=6)
        for _, _, weight in result.points:
            self.assertGreaterEqual(weight, 1e-9)

    def test_grid_validation(self) -> None:
        with self.assertRaises(ValueError):
            fit(cells([(1, 1)]), mu_points=1)
        with self.assertRaises(ValueError):
            fit(cells([(1, 1)]), kappa_min=10.0, kappa_max=1.0)


class CellPosteriorTest(unittest.TestCase):
    def test_small_sample_is_shrunk_towards_the_layer(self) -> None:
        payload = [
            *cells([(14, 6)] * 9),
            Cell(stratum="tiny", outcome="answer", improved=3, regressed=1, passed=3, pairs=4),
        ]
        result = fit(payload)
        posterior = cell_posteriors(result, payload, draws=128, seed=1)
        tiny = posterior.summaries[("tiny", "answer")]
        self.assertLess(tiny.mean, 0.75)  # plugin rate is 3/4
        self.assertGreater(tiny.mean, 0.5)
        self.assertLess(tiny.eti_low, tiny.mean)
        self.assertGreater(tiny.eti_high, tiny.mean)

    def test_p_improved_is_monotone_in_improvements(self) -> None:
        previous = -1.0
        for k in range(0, 11):
            payload = [
                Cell(
                    stratum="s", outcome="answer", improved=k, regressed=10 - k, passed=k, pairs=10
                )
            ]
            value = fit(payload).cell_p_improved(payload[0])
            self.assertGreater(value, previous)
            previous = value

    def test_p_improved_reuses_the_cached_tails(self) -> None:
        """A refresh whose cells have not changed pays no tail evaluations."""
        payload = cells([(10, 5), (5, 10), (8, 8)])
        fit(payload).cell_p_improved(payload[0])
        before = beta_sf.cache_info()
        fit(payload).cell_p_improved(payload[0])
        after = beta_sf.cache_info()
        self.assertGreater(after.hits, before.hits)
        self.assertEqual(after.misses, before.misses)

    def test_draws_are_correlated_across_cells(self) -> None:
        payload = cells([(10, 5), (5, 10), (8, 8)])
        posterior = cell_posteriors(fit(payload), payload, draws=64, seed=2)
        self.assertEqual(len(posterior.phi_draws), 64)
        self.assertTrue(all(len(row) == 3 for row in posterior.phi_draws))
        self.assertEqual(set(posterior.summaries), {cell.key for cell in payload})

    def test_draws_validation(self) -> None:
        with self.assertRaises(ValueError):
            cell_posteriors(fit(cells([(1, 1)])), cells([(1, 1)]), draws=1)


class HomogeneityGuardrailTest(unittest.TestCase):
    """The guardrail must fire on disagreement, and stay quiet when there is none."""

    def test_identical_cells_are_homogeneous(self) -> None:
        check = homogeneity_ppc(cells([(14, 6)] * 10), draws=192, seed=0)
        assert check is not None
        self.assertGreater(check.ppc, 0.05)
        self.assertFalse(check.heterogeneous)

    def test_global_regression_stays_homogeneous(self) -> None:
        check = homogeneity_ppc(cells([(6, 14)] * 10), draws=192, seed=0)
        assert check is not None
        self.assertGreater(check.ppc, 0.05)

    def test_masked_regression_is_flagged(self) -> None:
        """Nine views improve, one collapses: the pooled rate looks fine at ~.58."""
        check = homogeneity_ppc(cells([(14, 6)] * 9 + [(2, 18)]), draws=192, seed=0)
        assert check is not None
        self.assertLess(check.ppc, 0.05)
        self.assertTrue(check.heterogeneous)
        self.assertGreater(check.observed, check.expected)

    def test_masked_regression_with_unchanged_questions_is_flagged(self) -> None:
        """The realistic bank shape: most questions do not move.

        Regression guard against the bug where the guardrail counted *all*
        pairs instead of discordant ones: with 80 unchanged questions per view
        the masked regression used to slip through (p ~ 0.2).
        """
        healthy = [
            Cell(stratum=f"v{i}", outcome="answer", improved=14, regressed=6, passed=88, pairs=100)
            for i in range(9)
        ]
        broken = Cell(
            stratum="broken", outcome="answer", improved=2, regressed=18, passed=80, pairs=100
        )
        check = homogeneity_ppc([*healthy, broken], draws=192, seed=0)
        assert check is not None
        self.assertLess(check.ppc, 0.05)
        self.assertTrue(check.heterogeneous)

    def test_mild_heterogeneity_is_not_overflagged(self) -> None:
        check = homogeneity_ppc(cells([(12, 8)] * 9 + [(8, 12)]), draws=192, seed=0)
        assert check is not None
        self.assertGreater(check.ppc, 0.05)

    def test_single_cell_cannot_test_homogeneity(self) -> None:
        check = homogeneity_ppc(cells([(14, 6)]), draws=192, seed=0)
        assert check is not None
        self.assertGreater(check.ppc, 0.05)
        self.assertEqual(check.cells, 1)

    def test_no_pairs_returns_none(self) -> None:
        self.assertIsNone(homogeneity_ppc([], draws=64))
        self.assertIsNone(homogeneity_ppc([Cell(stratum="s", outcome="answer")], draws=64))

    def test_calibration_under_the_null(self) -> None:
        """Homogeneous data must not trip the guardrail: tails stay usable.

        A real calibration check, not a smoke test: 200 trials with realistic
        banks (most questions unchanged), fixed seeds so it is deterministic,
        and a false-positive bound tight enough to catch a liberal check.
        """
        rng = random.Random(2024)
        trials = 200
        tails: list[float] = []
        for trial in range(trials):
            switch_rate = rng.uniform(0.35, 0.65)
            payload = []
            for index in range(6):
                discordant = 16
                improved = sum(1 for _ in range(discordant) if rng.random() < switch_rate)
                payload.append(
                    Cell(
                        stratum=f"v{index}",
                        outcome="answer",
                        improved=improved,
                        regressed=discordant - improved,
                        passed=improved,
                        pairs=discordant + 40,
                    )
                )
            check = homogeneity_ppc(payload, draws=128, seed=trial)
            assert check is not None
            tails.append(check.ppc)
        flagged = sum(1 for value in tails if value < 0.05)
        # Binomial(200, .05) puts 10% far in the upper tail; nominal is 5%.
        self.assertLessEqual(flagged, 20, f"false positives: {flagged}/{trials}")
        self.assertGreater(sum(tails) / len(tails), 0.3)


class DiscordanceDrawTest(unittest.TestCase):
    """The delta model must carry how much moved, not only which way."""

    def test_discordance_draws_shrink_as_pairs_grow(self) -> None:
        small = [Cell(stratum="s", outcome="o", improved=0, regressed=0, pairs=10)]
        big = [Cell(stratum="s", outcome="o", improved=0, regressed=0, pairs=10000)]
        posterior_small = cell_posteriors(fit(small), small, draws=64, seed=0)
        posterior_big = cell_posteriors(fit(big), big, draws=64, seed=0)
        self.assertEqual(len(posterior_small.discordance_draws), 64)
        self.assertEqual(len(posterior_small.discordance_draws[0]), 1)
        mean_small = sum(posterior_small.discordance_draws[0]) / 64
        mean_big = sum(posterior_big.discordance_draws[0]) / 64
        self.assertLess(mean_big, mean_small)

    def test_discordance_prior_validation(self) -> None:
        with self.assertRaises(ValueError):
            cell_posteriors(fit(cells([(1, 1)])), cells([(1, 1)]), a_d=0.0)


if __name__ == "__main__":
    unittest.main()
