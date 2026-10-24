"""The score-over-time band: within-epoch noise vs between-epoch volatility."""

from __future__ import annotations

import unittest

from betapair import HealthReport, ScorePoint, score_band


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


if __name__ == "__main__":
    unittest.main()
