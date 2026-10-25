"""The score-over-time band: within-epoch noise vs between-epoch volatility."""

from __future__ import annotations

import math
import unittest

from betapair import HealthReport, RetrospectiveWindow, ScorePoint, retrospective, score_band


def point(score: float, pairs: float = 0.0) -> ScorePoint:
    return ScorePoint(score=score, pairs=pairs)


class ScorePointTest(unittest.TestCase):
    def test_interval_from_pairs_is_centred_on_the_score(self) -> None:
        low, high = point(50.0, pairs=100.0).interval()
        self.assertLess(low, 50.0)
        self.assertGreater(high, 50.0)
        self.assertAlmostEqual((low + high) / 2, 50.0, delta=0.5)

    def test_more_pairs_narrow_the_interval(self) -> None:
        wide = point(50.0, pairs=25.0).interval()
        narrow = point(50.0, pairs=2500.0).interval()
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_stored_bounds_win(self) -> None:
        stored = ScorePoint(score=50.0, pairs=100.0, low=40.0, high=60.0)
        self.assertEqual(stored.interval(), (40.0, 60.0))

    def test_point_without_pairs_has_no_width(self) -> None:
        self.assertEqual(point(63.0).interval(), (63.0, 63.0))

    def test_invalid_level(self) -> None:
        with self.assertRaises(ValueError):
            point(50.0, pairs=10.0).interval(level=1.0)

    def test_snapshot_round_trip(self) -> None:
        stored = ScorePoint(score=61.5, pairs=40.0, epoch="2026-07-01", low=55.0, high=67.0)
        self.assertEqual(ScorePoint.from_snapshot(stored.snapshot()), stored)
        plain = point(61.5, pairs=40.0)
        self.assertEqual(ScorePoint.from_snapshot(plain.snapshot()), plain)

    def test_snapshot_round_trips_the_regression_base_rate(self) -> None:
        stored = ScorePoint(score=61.5, pairs=40.0, pi_mean=0.2)
        self.assertEqual(ScorePoint.from_snapshot(stored.snapshot()), stored)
        self.assertEqual(stored.pi_mean, 0.2)

    def test_payload_without_a_base_rate_still_loads(self) -> None:
        """Additive encoding: a timeline written before pi_mean existed loads."""
        restored = ScorePoint.from_snapshot({"score": 61.5, "pairs": 40.0})
        self.assertIsNone(restored.pi_mean)

    def test_from_report_carries_the_base_rate(self) -> None:
        report = HealthReport(status="stable", score=60.0, pairs=50, pi_mean=0.15)
        extracted = ScorePoint.from_report(report)
        assert extracted is not None
        self.assertEqual(extracted.pi_mean, 0.15)

    def test_snapshot_round_trips_the_unresolved_share(self) -> None:
        stored = ScorePoint(score=61.5, pairs=40.0, unresolved_share=0.2)
        self.assertEqual(ScorePoint.from_snapshot(stored.snapshot()), stored)
        self.assertEqual(stored.unresolved_share, 0.2)

    def test_payload_without_an_unresolved_share_still_loads(self) -> None:
        restored = ScorePoint.from_snapshot({"score": 61.5, "pairs": 40.0})
        self.assertIsNone(restored.unresolved_share)

    def test_from_report_carries_the_unresolved_share(self) -> None:
        report = HealthReport(status="stable", score=60.0, pairs=50, unresolved_share=0.1)
        extracted = ScorePoint.from_report(report)
        assert extracted is not None
        self.assertEqual(extracted.unresolved_share, 0.1)

    def test_snapshot_rejects_half_an_interval(self) -> None:
        with self.assertRaises(ValueError):
            ScorePoint.from_snapshot({"score": 50.0, "low": 40.0})

    def test_snapshot_rejects_bad_types(self) -> None:
        for payload in ("nope", {"score": "high"}, {"score": 50.0, "epoch": 7}):
            with self.assertRaises(ValueError):
                ScorePoint.from_snapshot(payload)


class ScoreBandTest(unittest.TestCase):
    def test_needs_at_least_one_scored_point(self) -> None:
        with self.assertRaises(ValueError):
            score_band([])
        # A report with no paired items carries no score and is not a point.
        with self.assertRaises(ValueError):
            score_band([HealthReport(status="insufficient")])

    def test_consistency_is_narrower_than_volatility(self) -> None:
        steady = [point(score, pairs=500.0) for score in (70, 70.5, 69.5, 70, 70.2)]
        volatile = [point(score, pairs=500.0) for score in (40, 80, 55, 90, 45)]
        self.assertLess(score_band(steady).width, score_band(volatile).width)

    def test_more_items_narrow_the_band(self) -> None:
        scores = (60, 62, 58, 61, 59)
        few = [point(score, pairs=20.0) for score in scores]
        many = [point(score, pairs=2000.0) for score in scores]
        self.assertLess(score_band(many).width, score_band(few).width)

    def test_sampling_noise_can_explain_the_spread(self) -> None:
        series = [point(50.0, pairs=100.0), point(51.0, pairs=100.0)]
        band = score_band(series)
        self.assertEqual(band.between_sd, 0.0)
        self.assertGreater(band.within_sd, 4.0)  # ~5pp binomial noise at n=100

    def test_a_real_swing_shows_up_as_between_sd(self) -> None:
        band = score_band([point(20.0, pairs=100.0), point(80.0, pairs=100.0)])
        self.assertGreater(band.between_sd, 0.0)

    def test_window_keeps_the_recent_points(self) -> None:
        band = score_band([point(10.0), point(50.0), point(90.0)], window=2)
        self.assertEqual(band.points, 2)
        self.assertAlmostEqual(band.center, 70.0)

    def test_single_point_falls_back_to_its_own_interval(self) -> None:
        lone = point(50.0, pairs=100.0)
        band = score_band([lone])
        low, high = lone.interval()
        self.assertEqual(band.points, 1)
        self.assertAlmostEqual(band.low, low)
        self.assertAlmostEqual(band.high, high)
        self.assertEqual(band.between_sd, 0.0)

    def test_bare_numbers_and_reports_are_accepted(self) -> None:
        reports = [
            HealthReport(status="stable", score=60.0, pairs=50),
            HealthReport(status="stable", score=62.0, pairs=50),
        ]
        self.assertAlmostEqual(score_band([60.0, 62.0]).center, 61.0)
        self.assertAlmostEqual(score_band(reports).center, 61.0)

    def test_bounds_are_clamped_to_the_percentage_scale(self) -> None:
        self.assertGreaterEqual(score_band([point(1.0), point(99.0)]).low, 0.0)
        self.assertLessEqual(score_band([point(1.0), point(99.0)]).high, 100.0)

    def test_invalid_window_level_and_input(self) -> None:
        with self.assertRaises(ValueError):
            score_band([60.0], window=0)
        with self.assertRaises(ValueError):
            score_band([60.0], level=0.0)
        with self.assertRaises(TypeError):
            score_band(["60"])  # type: ignore[list-item]


class RetrospectiveTest(unittest.TestCase):
    """A window pools the level and reads the verdict off its two halves."""

    @staticmethod
    def _series(scores: tuple[float, ...], pairs: float = 2000.0) -> list[ScorePoint]:
        return [ScorePoint(score=score, pairs=pairs) for score in scores]

    def test_level_is_the_pair_mass_pooled_score(self) -> None:
        points = [ScorePoint(score=40.0, pairs=100.0), ScorePoint(score=60.0, pairs=300.0)]
        window = retrospective(points)
        self.assertAlmostEqual(window.score, 55.0, places=9)
        low, high = ScorePoint(score=55.0, pairs=400.0).interval()
        self.assertEqual((window.score_low, window.score_high), (low, high))
        self.assertEqual(window.points, 2)

    def test_more_points_narrow_the_windowed_level(self) -> None:
        two = retrospective(self._series((50.0, 50.0)))
        eight = retrospective(self._series((50.0,) * 8))
        self.assertLess(eight.score_high - eight.score_low, two.score_high - two.score_low)

    def test_a_sustained_rise_reads_improving(self) -> None:
        window = retrospective(self._series((40.0, 42.0, 44.0, 46.0, 48.0, 50.0)), rope_pp=1.0)
        self.assertEqual(window.status, "improving")
        assert window.delta_pp is not None and window.delta_low is not None
        self.assertGreater(window.delta_pp, 0.0)
        self.assertGreater(window.delta_low, 1.0)

    def test_a_sustained_fall_reads_regressing(self) -> None:
        window = retrospective(self._series((50.0, 48.0, 46.0, 44.0, 42.0, 40.0)), rope_pp=1.0)
        self.assertEqual(window.status, "regressing")
        assert window.delta_high is not None
        self.assertLess(window.delta_high, -1.0)

    def test_a_flat_series_is_stable(self) -> None:
        window = retrospective(self._series((50.0, 50.0, 50.0, 50.0)), rope_pp=1.0)
        self.assertEqual(window.status, "stable")
        self.assertAlmostEqual(window.delta_pp or 0.0, 0.0, places=9)

    def test_a_wide_window_can_still_be_undecided(self) -> None:
        # A 5pp move with 20 pairs per epoch: the interval crosses the ROPE and
        # the point estimate is outside it, so the window does not pretend.
        points = self._series((40.0, 45.0, 50.0, 55.0), pairs=20.0)
        window = retrospective(points, rope_pp=1.0)
        self.assertEqual(window.status, "insufficient")

    def test_window_keeps_only_the_recent_points(self) -> None:
        points = self._series((90.0, 50.0, 50.0, 50.0))
        window = retrospective(points, window=2)
        self.assertEqual(window.points, 2)
        self.assertAlmostEqual(window.score, 50.0, places=9)

    def test_flips_count_sign_changes(self) -> None:
        # +2, -3, +4, -5: three sign changes.
        window = retrospective(self._series((50.0, 52.0, 49.0, 53.0, 48.0)))
        self.assertEqual(window.flips, 3)
        self.assertEqual(retrospective(self._series((40.0, 42.0, 44.0, 46.0))).flips, 0)

    def test_trend_is_the_pair_weighted_slope(self) -> None:
        window = retrospective(self._series((50.0, 52.0, 54.0, 56.0)))
        self.assertAlmostEqual(window.trend_pp or 0.0, 2.0, places=9)
        # A single weighted point has no slope to measure.
        self.assertIsNone(retrospective(self._series((50.0,))).trend_pp)

    def test_a_single_point_has_no_delta(self) -> None:
        window = retrospective(self._series((61.0,)))
        self.assertIsNone(window.delta_pp)
        self.assertIsNone(window.delta_low)
        self.assertIsNone(window.delta_high)
        self.assertEqual(window.status, "insufficient")
        self.assertEqual(window.flips, 0)

    def test_a_series_without_pairs_cannot_be_pooled(self) -> None:
        with self.assertRaises(ValueError):
            retrospective([ScorePoint(score=50.0), ScorePoint(score=52.0)])
        with self.assertRaises(ValueError):
            retrospective([])
        with self.assertRaises(ValueError):
            retrospective(self._series((50.0,)), window=0)
        with self.assertRaises(ValueError):
            retrospective(self._series((50.0,)), rope_pp=-1.0)
        with self.assertRaises(ValueError):
            retrospective(self._series((50.0,)), level=1.0)

    def test_window_fields_validate_themselves(self) -> None:
        with self.assertRaises(ValueError):
            RetrospectiveWindow(50.0, 60.0, 55.0, None, None, None, None, 0, "stable", 1)
        with self.assertRaises(ValueError):
            RetrospectiveWindow(50.0, 55.0, 45.0, None, None, None, None, 0, "stable", 1)
        with self.assertRaises(ValueError):
            RetrospectiveWindow(101.0, 45.0, 155.0, None, None, None, None, 0, "stable", 1)
        with self.assertRaises(ValueError):
            RetrospectiveWindow(50.0, 45.0, 55.0, 2.0, 1.0, 1.5, None, 0, "stable", 1)
        with self.assertRaises(ValueError):
            RetrospectiveWindow(50.0, 45.0, 55.0, None, None, None, math.inf, 0, "stable", 1)
        with self.assertRaises(ValueError):
            RetrospectiveWindow(50.0, 45.0, 55.0, None, None, None, None, 0, "stable", 0)
        with self.assertRaises(ValueError):
            RetrospectiveWindow(50.0, 45.0, 55.0, None, None, None, None, 0, "stable", 1, 0)
        with self.assertRaises(ValueError):
            RetrospectiveWindow(50.0, 45.0, 55.0, 2.0, None, None, None, 0, "stable", 1)
        with self.assertRaises(ValueError):
            RetrospectiveWindow(50.0, 45.0, 55.0, 2.0, 1.0, 3.0, None, -1, "stable", 1)
        with self.assertRaises(ValueError):
            RetrospectiveWindow(50.0, 45.0, 55.0, 2.0, 1.0, 3.0, None, 0, "mixed", 1)

    def test_results_are_deterministic(self) -> None:
        points = self._series((48.0, 52.0, 51.0, 55.0, 54.0, 58.0))
        self.assertEqual(retrospective(points), retrospective(points))


class ScorePointCompositionTest(unittest.TestCase):
    """A point can carry the bank it was scored on."""

    def test_snapshot_round_trips_the_composition(self) -> None:
        stored = ScorePoint(
            score=75.0,
            pairs=80.0,
            epoch="e2",
            low=70.0,
            high=80.0,
            score_held=60.0,
            paired=80,
            retired=20,
            failed=3,
            bank="abc123",
        )
        self.assertEqual(ScorePoint.from_snapshot(stored.snapshot()), stored)

    def test_a_legacy_point_has_no_composition(self) -> None:
        restored = ScorePoint.from_snapshot({"score": 50.0})
        self.assertIsNone(restored.score_held)
        self.assertIsNone(restored.paired)
        self.assertIsNone(restored.retired)
        self.assertIsNone(restored.failed)
        self.assertIsNone(restored.bank)

    def test_malformed_composition_is_rejected(self) -> None:
        for payload in (
            {"score": 50.0, "score_held": 101.0},
            {"score": 50.0, "paired": -1},
            {"score": 50.0, "paired": True},
            {"score": 50.0, "retired": 1.5},
            {"score": 50.0, "bank": ""},
            {"score": 50.0, "bank": 7},
        ):
            with self.assertRaises(ValueError):
                ScorePoint.from_snapshot(payload)


class BankStableBandTest(unittest.TestCase):
    """A bank change is composition, not volatility."""

    def test_a_stable_bank_books_genuine_movement(self) -> None:
        band = score_band(
            [ScorePoint(60.0, pairs=100.0, bank="a"), ScorePoint(75.0, pairs=80.0, bank="a")]
        )
        self.assertTrue(band.bank_stable)
        self.assertGreater(band.between_sd, 9.0)

    def test_the_attrition_reproduction_does_not_book_volatility(self) -> None:
        band = score_band(
            [ScorePoint(60.0, pairs=100.0, bank="a"), ScorePoint(75.0, pairs=80.0, bank="b")]
        )
        self.assertFalse(band.bank_stable)
        self.assertEqual(band.between_sd, 0.0)
        self.assertGreater(band.width, 0.0)  # the sampling noise is still shown

    def test_movement_is_measured_within_each_bank(self) -> None:
        band = score_band(
            [
                ScorePoint(60.0, pairs=2000.0, bank="a"),
                ScorePoint(66.0, pairs=2000.0, bank="a"),
                ScorePoint(80.0, pairs=2000.0, bank="b"),
                ScorePoint(86.0, pairs=2000.0, bank="b"),
            ]
        )
        self.assertFalse(band.bank_stable)
        # The within-bank step is 6pp (~4pp dispersion); the cross-bank jump
        # (~11pp dispersion) is ignored.
        self.assertLess(band.between_sd, 6.0)
        self.assertGreater(band.between_sd, 3.0)

    def test_no_bank_information_keeps_the_old_estimate(self) -> None:
        band = score_band([ScorePoint(60.0, pairs=100.0), ScorePoint(75.0, pairs=80.0)])
        self.assertIsNone(band.bank_stable)
        self.assertGreater(band.between_sd, 9.0)

    def test_partial_bank_information_is_not_stable(self) -> None:
        band = score_band([ScorePoint(60.0, pairs=100.0, bank="a"), ScorePoint(75.0, pairs=80.0)])
        self.assertFalse(band.bank_stable)
        self.assertEqual(band.between_sd, 0.0)

    def test_a_single_point_reports_its_bank_state(self) -> None:
        self.assertTrue(score_band([ScorePoint(60.0, pairs=100.0, bank="a")]).bank_stable)
        self.assertIsNone(score_band([ScorePoint(60.0, pairs=100.0)]).bank_stable)


if __name__ == "__main__":
    unittest.main()
