"""Bayesian McNemar test for paired binary outcomes.

Prior art: Chechile (2020) ch. 7, *Bayesian Statistics for Experimental
Scientists*. The change cases are a Bernoulli process, so the population
switch rate ``phi_rb`` has a Beta posterior:

    phi_rb | data ~ Beta(a0 + n_01, b0 + n_10)

``n_01`` counts pairs that went 0 → 1 (an item that failed before and passes
now) and ``n_10`` the 1 → 0 regressions. Pairs that did not change carry no
information about ``phi_rb`` — that focus on the discordant sub-population is
the whole point of McNemar, and it is why the test stays usable on small
samples instead of drowning in unchanged items.

Reported quantities: posterior mean/median, equal-tail interval, the
point-null Bayes factor at ``phi_rb = .5`` (Savage-Dickey: prior density over
posterior density at the null), the interval Bayes factor, and
``P(phi_rb > .5)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from betapair.beta import beta_mean, beta_pdf, beta_ppf, beta_sf

__all__ = ["NULL_SWITCH_RATE", "McNemarResult", "SwitchEvidence", "mcnemar", "switch_evidence"]

#: Null value of the switch rate: no net change between the two epochs.
NULL_SWITCH_RATE = 0.5
#: Cap on the memoised per-cell posteriors. A streaming report asks for the
#: same counts on every refresh, so the cache turns a repeat into a lookup;
#: the bound stops a long session from growing it without limit.
_CACHE_SIZE = 4096


@dataclass(frozen=True)
class McNemarResult:
    """Posterior summary for one paired binary comparison."""

    n_improved: int
    n_regressed: int
    a_prior: float
    b_prior: float
    prob_interval: float
    a_post: float
    b_post: float
    post_mean: float
    post_median: float
    eti_low: float
    eti_high: float
    bf10_point: float
    bf10_interval: float
    p_improved: float

    @property
    def discordant(self) -> int:
        """Pairs that changed at all — the effective sample size of the test."""
        return self.n_improved + self.n_regressed

    @property
    def switched_up(self) -> float:
        """Plugin estimate ``n_01 / (n_01 + n_10)``; ``nan`` when nothing moved."""
        if self.discordant == 0:
            return math.nan
        return self.n_improved / self.discordant


@dataclass(frozen=True)
class SwitchEvidence:
    """The two per-cell numbers the verdicts read, without the interval.

    ``discordant`` is McNemar's effective sample size and ``p_improved`` is
    ``P(phi_rb > .5)``. The report asks this of every cell on every refresh
    and never reads the posterior median or credible interval — the expensive
    part of :func:`mcnemar` — so the tail probability is all that is needed.
    """

    discordant: int
    p_improved: float


def _validate(n_improved: int, n_regressed: int, a0: float, b0: float) -> None:
    """Shared count/prior validation, so the two entry points cannot drift."""
    if n_improved < 0 or n_regressed < 0:
        raise ValueError(
            f"counts must be non-negative, got n_improved={n_improved}, n_regressed={n_regressed}"
        )
    if a0 <= 0.0 or b0 <= 0.0:
        raise ValueError(f"prior shapes must be positive, got a0={a0}, b0={b0}")


@lru_cache(maxsize=_CACHE_SIZE)
def _switch_evidence(n_improved: int, n_regressed: int, a0: float, b0: float) -> SwitchEvidence:
    """Cached body of :func:`switch_evidence` (all args already normalised)."""
    return SwitchEvidence(
        discordant=n_improved + n_regressed,
        p_improved=beta_sf(NULL_SWITCH_RATE, a0 + n_improved, b0 + n_regressed),
    )


def switch_evidence(
    n_improved: int,
    n_regressed: int,
    *,
    a0: float = 1.0,
    b0: float = 1.0,
) -> SwitchEvidence:
    """Discordant count and ``P(phi_rb > .5)``, without the credible interval.

    This is the subset of :func:`mcnemar` that the report's per-cell verdicts
    and the per-tier roll-up actually consume. It is memoised on the exact
    counts, because a streaming refresh asks the same question of every cell
    whose data has not changed since the last report.

    Raises:
        ValueError: If counts are negative or prior shapes are not positive.

    Notes:
        The result is bit-identical to reading the same fields off
            :func:`mcnemar`; only the interval and Bayes factors are skipped.
    """
    _validate(n_improved, n_regressed, a0, b0)
    return _switch_evidence(int(n_improved), int(n_regressed), float(a0), float(b0))


@lru_cache(maxsize=_CACHE_SIZE)
def _mcnemar(
    n_improved: int,
    n_regressed: int,
    a0: float,
    b0: float,
    prob_interval: float,
) -> McNemarResult:
    """Cached body of :func:`mcnemar` (inputs already validated)."""
    a_post = a0 + n_improved
    b_post = b0 + n_regressed
    tail = 0.5 * (1.0 - prob_interval)
    p_improved = beta_sf(NULL_SWITCH_RATE, a_post, b_post)

    prior_density = beta_pdf(NULL_SWITCH_RATE, a0, b0)
    post_density = beta_pdf(NULL_SWITCH_RATE, a_post, b_post)
    # The zero branch only fires for degenerate shapes.
    bf10_point = prior_density / post_density if post_density > 0.0 else math.inf

    prior_h1 = beta_sf(NULL_SWITCH_RATE, a0, b0)
    post_h0 = 1.0 - p_improved
    if prior_h1 <= 0.0 or post_h0 <= 0.0:
        bf10_interval = math.inf
    else:
        bf10_interval = (p_improved / post_h0) / (prior_h1 / (1.0 - prior_h1))

    return McNemarResult(
        n_improved=n_improved,
        n_regressed=n_regressed,
        a_prior=a0,
        b_prior=b0,
        prob_interval=prob_interval,
        a_post=a_post,
        b_post=b_post,
        post_mean=beta_mean(a_post, b_post),
        post_median=beta_ppf(0.5, a_post, b_post),
        eti_low=beta_ppf(tail, a_post, b_post),
        eti_high=beta_ppf(1.0 - tail, a_post, b_post),
        bf10_point=bf10_point,
        bf10_interval=bf10_interval,
        p_improved=p_improved,
    )


def mcnemar(
    n_improved: int,
    n_regressed: int,
    *,
    a0: float = 1.0,
    b0: float = 1.0,
    prob_interval: float = 0.95,
) -> McNemarResult:
    """Bayesian McNemar test from the two discordant counts.

    Args:
        n_improved: 0 → 1 switch count (``n_01`` in the McNemar table).
        n_regressed: 1 → 0 switch count (``n_10`` in the McNemar table).
        a0: Beta prior shape on the switch rate — uniform by default.
        b0: Beta prior shape on the switch rate; ``(0.5, 0.5)`` gives
            Jeffreys' prior.
        prob_interval: Mass of the equal-tail credible interval.

    Returns:
        Posterior summary of the switch rate.

    Raises:
        ValueError: If counts are negative, prior shapes are not positive,
            or ``prob_interval`` is outside (0, 1).

    Notes:
        The interval Bayes factor compares the interval null
            ``H0: phi <= .5`` against ``H1: phi > .5`` as the ratio of posterior
            odds to prior odds; the point null uses the Savage-Dickey density
            ratio.

        Memoised on the exact inputs. A report that is refreshed without
            new observations — a dashboard on a timer — reuses every cell's
            posterior instead of recomputing the beta quantiles.
    """
    _validate(n_improved, n_regressed, a0, b0)
    if not 0.0 < prob_interval < 1.0:
        raise ValueError(f"prob_interval must be in (0, 1), got {prob_interval}")
    return _mcnemar(int(n_improved), int(n_regressed), float(a0), float(b0), float(prob_interval))
