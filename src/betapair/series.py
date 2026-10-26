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
from itertools import pairwise
from statistics import NormalDist

from betapair.beta import beta_ppf
from betapair.models import HealthReport, RetrospectiveWindow, Status

__all__ = ["RetrospectiveWindow", "ScoreBand", "ScorePoint", "retrospective", "score_band"]


@dataclass(frozen=True)
class ScorePoint:
    """One epoch's score, with what is needed to size its own noise.

    ``pairs`` is the **effective** paired count behind the score (aggregation
    weights included). When ``low``/``high`` are absent, the sampling interval
    is derived from ``score`` and ``pairs``. ``epoch`` is an optional label for
    the series (a date, a round number, …).

    ``pi_mean`` is the epoch's estimated share of cells that truly regressed
    (``HealthReport.pi_mean`` when the point came from a report). It is
    ``None`` for a bare score and is the empirical-Bayes evidence the next
    epoch's calibration uses to set its regression prior.

    ``unresolved_share`` is the fraction of tracked rows still unresolved
    (pending or harness-errored) when the point was recorded. It is ``None``
    for a bare score and is the observed distribution the next epoch's
    coverage threshold is calibrated from.

    The remaining fields record the **bank** the score was computed on, so a
    later band can tell a real move from a composition change: ``score_held``
    is the level with retired items held at their last outcome,
    ``paired``/``retired``/``failed`` are item counts, and ``bank`` is a digest
    of the active (scored) item keys. All are optional — a bare score carries
    none — and additive in the snapshot.
    """

    score: float
    epoch: str | None = None
    pairs: float = 0.0
    low: float | None = None
    high: float | None = None
    pi_mean: float | None = None
    unresolved_share: float | None = None
    score_held: float | None = None
    paired: int | None = None
    retired: int | None = None
    failed: int | None = None
    bank: str | None = None

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
            low=report.score_low,
            high=report.score_high,
            pi_mean=report.pi_mean,
            unresolved_share=report.unresolved_share,
            score_held=report.score_held,
            paired=report.coverage.paired,
            retired=report.retired,
            failed=report.failed,
        )

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """This epoch's own credible interval on its pass rate, 0-100.

        Uses the stored ``low``/``high`` when present, otherwise the
        ``Beta(1 + k, 1 + n - k)`` posterior, whose mode is exactly
        ``score`` — so the interval never moves the point estimate. A point
        with no ``pairs`` carries no sampling width and returns itself.

        Raises:
            ValueError: If ``level`` is outside (0, 1).
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
            "pi_mean": self.pi_mean,
            "unresolved_share": self.unresolved_share,
            "score_held": self.score_held,
            "paired": self.paired,
            "retired": self.retired,
            "failed": self.failed,
            "bank": self.bank,
        }

    @classmethod
    def from_snapshot(cls, payload: object) -> ScorePoint:
        """Parse one :meth:`snapshot` mapping.

        Raises:
            ValueError: If a field is missing or has the wrong type.
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
        score_held = _optional_number(payload.get("score_held"), "score_held")
        if score_held is not None and not 0.0 <= score_held <= 100.0:
            raise ValueError(f"score point score_held must be in [0, 100], got {score_held}")
        return cls(
            score=_number(payload.get("score"), "score"),
            epoch=epoch,
            pairs=_number(payload.get("pairs", 0.0), "pairs"),
            low=low,
            high=high,
            pi_mean=_optional_number(payload.get("pi_mean"), "pi_mean"),
            unresolved_share=_optional_number(payload.get("unresolved_share"), "unresolved_share"),
            score_held=score_held,
            paired=_optional_count(payload.get("paired"), "paired"),
            retired=_optional_count(payload.get("retired"), "retired"),
            failed=_optional_count(payload.get("failed"), "failed"),
            bank=_optional_text(payload.get("bank"), "bank"),
        )


@dataclass(frozen=True)
class ScoreBand:
    """Where an epoch's score has been landing, split into its two variances.

    ``center`` is the mean of the window and ``low``/``high`` the reference
    band around it. ``within_sd`` is the mean sampling noise of one epoch and
    ``between_sd`` the genuine epoch-to-epoch dispersion: when the whole
    observed spread is explained by sampling noise, ``between_sd`` is zero.

    ``bank_stable`` says whether the points were all scored on the same bank:
    ``True`` when every point carries a bank digest and they agree, ``False``
    when the digests differ (or only some points carry one), and ``None`` when
    no bank information was recorded at all. When it is ``False``, the
    between-epoch dispersion is estimated **within** each bank only, so a
    composition change (retiring hard items, adding new ones) is not booked as
    genuine volatility; if no bank has two points, ``between_sd`` is zero.
    """

    center: float
    low: float
    high: float
    level: float
    points: int
    within_sd: float
    between_sd: float
    window: int | None = None
    bank_stable: bool | None = None

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

    Args:
        points: The series — :class:`ScorePoint`, raw percentages, or
            :class:`~betapair.report.HealthReport` objects (unscored reports are
            skipped).
        window: Keep only the most recent this-many points; ``None`` uses
            the whole series.
        level: Mass of the reference band, e.g. ``.95`` for a 95% band.

    Returns:
        Center, bounds, and the within/between standard deviations.

    Raises:
        ValueError: On an empty (or fully unscored) series, a window below
            one, or a level outside (0, 1).

    Notes:
        The band is ``center ± z * sqrt(between_sd^2 + within_sd^2)`` — the
            spread of an *individual* epoch's score, not the much narrower
            uncertainty of the mean. A single point has no dispersion to estimate,
            so it falls back to its own interval.

        When the points carry different bank digests, ``between_sd`` is
            estimated within each bank only (see :attr:`ScoreBand.bank_stable`), so
            attrition or a bank edit is not mistaken for volatility. Points with no
            bank information keep the old whole-series estimate.
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
    banks = [point.bank for point in data]
    if all(bank is None for bank in banks):
        bank_stable: bool | None = None
    else:
        bank_stable = all(bank is not None and bank == banks[0] for bank in banks)
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
            bank_stable=bank_stable,
        )

    if bank_stable is False:
        grouped: dict[str, list[ScorePoint]] = {}
        for point in data:
            if point.bank is not None:
                grouped.setdefault(point.bank, []).append(point)
        estimates = []
        for members in grouped.values():
            if len(members) < 2:
                continue
            member_scores = [point.score for point in members]
            member_within = statistics.fmean(
                (
                    ((high - low) / (2.0 * z)) ** 2
                    for low, high in (point.interval(level) for point in members)
                )
            )
            estimates.append(max(0.0, statistics.variance(member_scores) - member_within))
        between_variance = statistics.fmean(estimates) if estimates else 0.0
    else:
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
        bank_stable=bank_stable,
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


def _optional_count(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"score point {field} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"score point {field} must be non-negative, got {value}")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"score point {field} must be non-empty text, got {value!r}")
    return value


def _pooled_level(points: list[ScorePoint], level: float) -> tuple[float, float, float] | None:
    """Pair-mass pooled score and its Beta interval, or ``None`` without pairs."""
    weighted = [point for point in points if point.pairs > 0.0]
    if not weighted:
        return None
    pairs = math.fsum(point.pairs for point in weighted)
    successes = math.fsum(point.pairs * point.score / 100.0 for point in weighted)
    score = 100.0 * successes / pairs
    low, high = ScorePoint(score=score, pairs=pairs).interval(level)
    return score, low, high


def _flips(scores: list[float]) -> int:
    """Sign changes between consecutive movements, ignoring exact ties."""
    signs = [
        1 if later > earlier else -1 for earlier, later in pairwise(scores) if later != earlier
    ]
    return sum(1 for left, right in pairwise(signs) if left != right)


def _trend(points: list[ScorePoint]) -> float | None:
    """Pair-weighted least-squares slope of score on epoch index."""
    weighted = [(index, point) for index, point in enumerate(points) if point.pairs > 0.0]
    if len(weighted) < 2:
        return None
    total = math.fsum(point.pairs for _index, point in weighted)
    mean_x = math.fsum(index * point.pairs for index, point in weighted) / total
    mean_y = math.fsum(point.score * point.pairs for _index, point in weighted) / total
    variance = math.fsum(point.pairs * (index - mean_x) ** 2 for index, point in weighted)
    covariance = math.fsum(
        point.pairs * (index - mean_x) * (point.score - mean_y) for index, point in weighted
    )
    return covariance / variance


def retrospective(
    points: Iterable[ScorePoint | HealthReport | float | int],
    *,
    window: int | None = None,
    rope_pp: float = 1.0,
    level: float = 0.95,
) -> RetrospectiveWindow:
    """Stabilise a score series over its most recent ``window`` epochs.

    The level is the pair-mass pooled score of the window, with the same
    ``Beta(1 + k, 1 + n - k)`` interval rule as the headline; the verdict
    compares the pooled first and second halves, with the conservative
    interval ``[second.low - first.high, second.high - first.low]`` against
    ``rope_pp``. Requiring the whole interval to clear the ROPE is deliberately
    cautious: a window is for cutting flapping, not for manufacturing a change.

    Args:
        points: The series — :class:`ScorePoint`, raw percentages, or
            :class:`~betapair.report.HealthReport` objects (unscored reports are
            skipped), like :func:`score_band`.
        window: Keep only the most recent this-many points; ``None`` uses
            the whole series.
        rope_pp: Region of practical equivalence, in percentage points.
        level: Mass of the level and half-comparison intervals.

    Returns:
        The windowed level, half-to-half change, trend, flips and verdict.

    Raises:
        ValueError: On an empty series, a window below one, a negative ROPE,
            a level outside (0, 1), or a series with no effective pairs.

    Notes:
        This is a **level-only** reading. It cannot see the current epoch's
            per-cell evidence, so it never hides a credible regression; the report's
            own ``status`` remains authoritative.

        At least one point must carry an effective pair count (as every
            tracker timeline point does); a bare ``ScorePoint(score=...)`` series
            has nothing to pool.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if rope_pp < 0.0:
        raise ValueError(f"rope_pp must be non-negative, got {rope_pp}")
    if window is not None and window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    data = _as_points(points)
    if window is not None:
        data = data[-window:]
    if not data:
        raise ValueError("retrospective needs at least one scored point")
    pooled = _pooled_level(data, level)
    if pooled is None:
        raise ValueError("retrospective needs at least one point with pairs > 0")
    score, score_low, score_high = pooled

    delta = delta_low = delta_high = None
    status: Status = "insufficient"
    if len(data) >= 2:
        cut = len(data) // 2
        first = _pooled_level(data[:cut], level)
        second = _pooled_level(data[cut:], level)
        if first is not None and second is not None:
            delta = second[0] - first[0]
            delta_low = second[1] - first[2]
            delta_high = second[2] - first[1]
            if delta_high < -rope_pp:
                status = "regressing"
            elif delta_low > rope_pp:
                status = "improving"
            elif abs(delta) <= rope_pp:
                status = "stable"

    return RetrospectiveWindow(
        score=score,
        score_low=score_low,
        score_high=score_high,
        delta_pp=delta,
        delta_low=delta_low,
        delta_high=delta_high,
        trend_pp=_trend(data),
        flips=_flips([point.score for point in data]),
        status=status,
        points=len(data),
        window=window,
    )
