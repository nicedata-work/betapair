"""Dispersion of a score over time.

One :func:`~betapair.report.build_report` answers "what is the score, and how
sure are we of the change?". A monitoring series answers a different question:
"how much does the score move?". That movement has two parts, and conflating
them is the usual mistake:

* **within-epoch noise** — the sampling uncertainty of one epoch's score,
  which shrinks as items are added; and
* **between-epoch dispersion** — genuine volatility of the metric, which does
  not shrink no matter how many items each epoch runs.

:func:`score_band` separates the two with a variance-components estimate
(observed score variance minus the mean within-epoch variance, floored at
zero) and returns a reference band for where an epoch's score falls. A
consistent series collapses onto its within-epoch noise; a volatile one stays
wide. Both standard deviations are returned, so a caller can see *why* a band
is wide.

The band is descriptive, not a hypothesis test: it says where scores have been
likely to land, so a dashboard can plot an average together with its natural
range. With only a handful of epochs the dispersion estimate is itself noisy;
:attr:`ScoreBand.points` records how much history the band rests on.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist

from betapair.beta import beta_ppf
from betapair.report import HealthReport

__all__ = ["ScoreBand", "ScorePoint", "score_band"]


@dataclass(frozen=True)
class ScorePoint:
    """One epoch's score, with what is needed to size its own noise.

    ``pairs`` is the **effective** paired count behind the score (aggregation
    weights included). When ``low``/``high`` are absent, the sampling interval
    is derived from ``score`` and ``pairs``. ``epoch`` is an optional label for
    the series (a date, a round number, …).
    """

    score: float
    epoch: str | None = None
    pairs: float = 0.0
    low: float | None = None
    high: float | None = None

    @classmethod
    def from_report(cls, report: HealthReport) -> ScorePoint | None:
        """Extract a point from a report; ``None`` when the report has no score.

        A report with no paired items is ``insufficient`` and carries no
        score; such a refresh is not a data point and is skipped rather than
        recorded as a zero.
        """
        if report.score is None:
            return None
        return cls(
            score=float(report.score),
            epoch=report.candidate_epoch,
            pairs=float(report.pairs),
        )

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """This epoch's own credible interval on its pass rate, 0-100.

        Uses the stored ``low``/``high`` when present, otherwise the
        ``Beta(1 + k, 1 + n - k)`` posterior, whose mode is exactly
        ``score`` — so the interval never moves the point estimate. A point
        with no ``pairs`` carries no sampling width and returns itself.

        @raise ValueError: If ``level`` is outside (0, 1).
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        if self.low is not None and self.high is not None:
            return self.low, self.high
        if self.pairs > 0.0:
            score = min(100.0, max(0.0, self.score))
            successes = self.pairs * score / 100.0
            tail = 0.5 * (1.0 - level)
            return (
                100.0 * beta_ppf(tail, 1.0 + successes, 1.0 + self.pairs - successes),
                100.0 * beta_ppf(1.0 - tail, 1.0 + successes, 1.0 + self.pairs - successes),
            )
        return self.score, self.score

    def snapshot(self) -> dict[str, float | str | None]:
        """JSON-ready mapping, the inverse of :meth:`from_snapshot`."""
        return {
            "score": self.score,
            "epoch": self.epoch,
            "pairs": self.pairs,
            "low": self.low,
            "high": self.high,
        }

    @classmethod
    def from_snapshot(cls, payload: object) -> ScorePoint:
        """Parse one :meth:`snapshot` mapping.

        @raise ValueError: If a field is missing or has the wrong type.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"score point must be a mapping, got {payload!r}")
        epoch = payload.get("epoch")
        if epoch is not None and not isinstance(epoch, str):
            raise ValueError(f"score point epoch must be text or null, got {epoch!r}")
        low = _optional_number(payload.get("low"), "low")
        high = _optional_number(payload.get("high"), "high")
        if (low is None) != (high is None):
            raise ValueError("score point needs both low and high, or neither")
        return cls(
            score=_number(payload.get("score"), "score"),
            epoch=epoch,
            pairs=_number(payload.get("pairs", 0.0), "pairs"),
            low=low,
            high=high,
        )


@dataclass(frozen=True)
class ScoreBand:
    """Where an epoch's score has been landing, split into its two variances.

    ``center`` is the mean of the window and ``low``/``high`` the reference
    band around it. ``within_sd`` is the mean sampling noise of one epoch and
    ``between_sd`` the genuine epoch-to-epoch dispersion: when the whole
    observed spread is explained by sampling noise, ``between_sd`` is zero.
    """

    center: float
    low: float
    high: float
    level: float
    points: int
    within_sd: float
    between_sd: float
    window: int | None = None

    @property
    def width(self) -> float:
        """Full width of the band in percentage points."""
        return self.high - self.low


def score_band(
    points: Iterable[ScorePoint | HealthReport | float | int],
    *,
    window: int | None = None,
    level: float = 0.95,
) -> ScoreBand:
    """Reference band for a score series, separating noise from volatility.

    @param points: The series — :class:`ScorePoint`, raw percentages, or
        :class:`~betapair.report.HealthReport` objects (unscored reports are
        skipped).
    @param window: Keep only the most recent this-many points; ``None`` uses
        the whole series.
    @param level: Mass of the reference band, e.g. ``.95`` for a 95% band.
    @return: Center, bounds, and the within/between standard deviations.
    @note: The band is ``center ± z * sqrt(between_sd^2 + within_sd^2)`` — the
        spread of an *individual* epoch's score, not the much narrower
        uncertainty of the mean. A single point has no dispersion to estimate,
        so it falls back to its own interval.
    @raise ValueError: On an empty (or fully unscored) series, a window below
        one, or a level outside (0, 1).
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    data = _as_points(points)
    if window is not None:
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        data = data[-window:]
    if not data:
        raise ValueError("score_band needs at least one scored point")

    scores = [point.score for point in data]
    center = statistics.fmean(scores)
    z = NormalDist().inv_cdf(0.5 + 0.5 * level)
    within_variance = statistics.fmean(
        (((high - low) / (2.0 * z)) ** 2 for low, high in (p.interval(level) for p in data))
    )
    within_sd = math.sqrt(within_variance)

    if len(data) == 1:
        low, high = data[0].interval(level)
        return ScoreBand(
            center=center,
            low=max(0.0, low),
            high=min(100.0, high),
            level=level,
            points=1,
            within_sd=within_sd,
            between_sd=0.0,
            window=window,
        )

    between_variance = max(0.0, statistics.variance(scores) - within_variance)
    half_width = z * math.sqrt(between_variance + within_variance)
    return ScoreBand(
        center=center,
        low=max(0.0, center - half_width),
        high=min(100.0, center + half_width),
        level=level,
        points=len(data),
        within_sd=within_sd,
        between_sd=math.sqrt(between_variance),
        window=window,
    )


def _as_points(points: Iterable[ScorePoint | HealthReport | float | int]) -> list[ScorePoint]:
    """Normalise the accepted series shapes into :class:`ScorePoint` objects."""
    out: list[ScorePoint] = []
    for point in points:
        if isinstance(point, ScorePoint):
            out.append(point)
        elif isinstance(point, HealthReport):
            extracted = ScorePoint.from_report(point)
            if extracted is not None:
                out.append(extracted)
        elif isinstance(point, (int, float)) and not isinstance(point, bool):
            out.append(ScorePoint(score=float(point)))
        else:
            raise TypeError(
                "score points must be ScorePoint, HealthReport or a number, "
                f"got {type(point).__name__}"
            )
    return out


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"score point {field} must be a number, got {value!r}")
    return float(value)


def _optional_number(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field)
