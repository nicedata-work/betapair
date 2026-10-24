"""Multi-strata, multi-outcome pooling of McNemar comparisons.

``mcnemar`` answers "did this pair of conditions get better?" for *one* binary
comparison. A real evaluation has many: several strata (models, datasets,
groups) times several outcomes (correctness, conformance, refusal, …). Pooling
them into one number needs a model that

* borrows strength across cells — a 4-item stratum is too small to stand on
  its own, but it is not noise either; and
* keeps a per-cell posterior, so the single number can always be expanded
  back into the cells that moved it.

This module is that model: a hierarchical (multi-strata) extension of the
Bayesian McNemar test, i.e. the Bayesian analogue of the stratified McNemar /
Cochran–Mantel–Haenszel pooling.

    phi_c | mu, kappa  ~ Beta(mu * kappa, (1 - mu) * kappa)      # cell switch rate
    n01_c | phi_c      ~ Binomial(n01_c + n10_c, phi_c)          # discordant pairs

``mu`` is the pooled switch rate of the whole population (the *direction*:
``mu > .5`` means more strata improved than regressed), ``kappa`` is the
concentration — large ``kappa`` means the strata move together, small
``kappa`` means they disagree. The marginal likelihood of a cell given
``(mu, kappa)`` is a beta-binomial, which is closed form, so the posterior over
``(mu, kappa)`` is a small deterministic grid: no MCMC, no compiled numerics,
and a fit costs milliseconds.

The guardrail is a **posterior predictive heterogeneity check**, not a Bayes
factor for ``kappa`` (see ``homogeneity_ppc``): under the pooled model, the
observed dispersion of discordant switch rates across cells is compared with
its posterior predictive distribution. Small tail probabilities say the strata
are *not* moving together — exactly the condition under which a pooled score can hide a
regression in one stratum behind improvements in the others.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from pydfba.beta import (
    beta_cdf,
    beta_mean,
    beta_pdf,
    beta_sample,
    log_beta_binomial,
)

#: Default grid resolution. 49 x 24 is ~1k beta-binomial evaluations per fit,
#: which keeps a streaming refresh in the low milliseconds per cell.
MU_POINTS = 49
KAPPA_POINTS = 24
#: Concentration grid span. Below .5 the prior is a near-point mass on
#: "every cell disagrees"; above 1e5 it is numerically indistinguishable from
#: the fully pooled model.
KAPPA_MIN = 0.5
KAPPA_MAX = 1.0e5
#: Inverse-Gamma(alpha, beta) prior on kappa (alpha=2, beta=4: mode 4/3,
#: slowly decaying tail — weakly informative, the data dominate once a cell
#: carries a handful of discordant pairs).
KAPPA_PRIOR = (2.0, 4.0)
#: Monte-Carlo draws used for intervals and joint draws.
DRAWS = 512
#: Posterior predictive draws for the homogeneity guardrail.
PPC_DRAWS = 192
#: Grid points carrying less posterior mass than this are dropped after the
#: marginals are summarised: a realistic fit keeps ~10% of the 49x24 grid, and
#: every per-cell summary then costs a tenth of the work.
GRID_WEIGHT_FLOOR = 1e-9


@dataclass(frozen=True)
class Cell:
    """One stratum x outcome paired comparison, as discordant counts.

    ``improved`` are pairs that failed in the baseline epoch and pass in the
    candidate epoch (McNemar's ``n_01``), ``regressed`` the reverse (``n_10``).
    ``passed`` is the candidate-epoch pass count over the paired items and
    ``pairs`` the number of items observed in both epochs. ``category`` is the
    criticality tier the stratum (semantic view) belongs to — a label, never
    used by the statistics themselves; the report turns it into a weight.

    The counts must describe one coherent set of paired items: ``passed``
    contains every ``improved`` pair and the candidate-epoch failures contain
    every ``regressed`` pair, so ``baseline_passed`` stays inside
    ``[0, pairs]``.
    """

    stratum: str
    outcome: str
    improved: int = 0
    regressed: int = 0
    passed: int = 0
    pairs: int = 0
    category: str | None = None

    def __post_init__(self) -> None:
        if self.improved < 0 or self.regressed < 0:
            raise ValueError(f"counts must be non-negative: {self}")
        if self.pairs < 0 or self.passed > self.pairs:
            raise ValueError(f"passed={self.passed} out of range for pairs={self.pairs}")
        if self.discordant > self.pairs:
            raise ValueError(f"discordant pairs exceed pairs: {self}")
        # ``passed`` counts candidate-epoch passes and must therefore contain
        # every 0 -> 1 switch; the candidate-epoch failures must contain every
        # 1 -> 0 switch. Otherwise ``baseline_passed`` can go negative.
        if self.passed < self.improved:
            raise ValueError(
                f"passed={self.passed} cannot be below improved={self.improved}: {self}"
            )
        if self.pairs - self.passed < self.regressed:
            raise ValueError(
                f"candidate failures={self.pairs - self.passed} cannot be below "
                f"regressed={self.regressed}: {self}"
            )

    @property
    def key(self) -> tuple[str, str]:
        return (self.stratum, self.outcome)

    @property
    def discordant(self) -> int:
        return self.improved + self.regressed

    @property
    def unchanged(self) -> int:
        return self.pairs - self.discordant

    @property
    def baseline_passed(self) -> int:
        return self.passed - self.improved + self.regressed

    @property
    def pass_rate(self) -> float:
        return 0.0 if self.pairs == 0 else self.passed / self.pairs

    @property
    def baseline_pass_rate(self) -> float:
        return 0.0 if self.pairs == 0 else self.baseline_passed / self.pairs


@dataclass(frozen=True)
class Fit:
    """Posterior over the pooled switch rate ``mu`` and concentration ``kappa``.

    @cvar points: ``(mu, kappa, posterior weight)`` triples, pruned to the grid
        points that carry meaningful posterior mass.
    """

    mu: float
    mu_eti: tuple[float, float]
    kappa: float
    kappa_eti: tuple[float, float]
    log_marginal: float
    points: tuple[tuple[float, float, float], ...]
    cells: int

    def cell_mean(self, cell: Cell) -> float:
        """Exact posterior mean of the cell switch rate, given the grid."""
        total = 0.0
        for mu, kappa, weight in self.points:
            a = mu * kappa + cell.improved
            b = (1.0 - mu) * kappa + cell.regressed
            total += weight * beta_mean(a, b)
        return total

    def cell_p_improved(self, cell: Cell) -> float:
        """``P(phi_cell > .5 | data)`` — exact mixture, no sampling noise."""
        total = 0.0
        for mu, kappa, weight in self.points:
            a = mu * kappa + cell.improved
            b = (1.0 - mu) * kappa + cell.regressed
            total += weight * (1.0 - beta_cdf(0.5, a, b))
        return total


@dataclass(frozen=True)
class CellPosterior:
    """Posterior summary of one cell's switch rate."""

    mean: float
    eti_low: float
    eti_high: float
    p_improved: float
    logit_shift: float


@dataclass(frozen=True)
class CellPosteriors:
    """Per-cell summaries plus the joint draws they came from.

    ``phi_draws[i][j]`` is the switch rate of ``cells[j]`` on draw ``i``, all
    from the *same* ``(mu, kappa)`` draw — so the draws are correctly
    correlated across cells and a pooled delta computed from them carries the
    hierarchical uncertainty.

    ``discordance_draws[i][j]`` is the fraction of ``cells[j]``'s pairs that
    changed at all, drawn from ``Beta(a_d + discordant, b_d + unchanged)``.
    The switch rate alone conditions on the observed discordant count, so a
    bank that did not move would look like certainty; multiplying by this draw
    is what makes a delta interval (and a ROPE on it) a statement about the
    population rather than a reconstruction of the sample. The two draw sets
    are independent by construction: ``d`` carries how much moved, ``phi``
    carries which way.
    """

    cells: tuple[Cell, ...]
    summaries: dict[tuple[str, str], CellPosterior]
    phi_draws: tuple[tuple[float, ...], ...]
    discordance_draws: tuple[tuple[float, ...], ...]
    params: tuple[tuple[float, float], ...]


def _logit(probability: float) -> float:
    """Log-odds of a rate already clamped inside (0, 1)."""
    return math.log(probability / (1.0 - probability))


def _invgamma_log_pdf(kappa: float, alpha: float, beta: float) -> float:
    return (
        alpha * math.log(beta) - math.lgamma(alpha) - (alpha + 1.0) * math.log(kappa) - beta / kappa
    )


def _weighted_quantiles(
    values: list[float], weights: list[float], probs: tuple[float, ...]
) -> tuple[float, ...]:
    """Quantiles of a discrete distribution (grid points/marginal weights)."""
    order = sorted(range(len(values)), key=values.__getitem__)
    total = sum(weights)
    cumulative = 0.0
    out: list[float] = []
    running: dict[int, float] = {}
    for position in order:
        cumulative += weights[position]
        running[position] = cumulative / total if total > 0 else 0.0
    for prob in probs:
        result = values[order[-1]]
        for position in order:
            if running[position] >= prob:
                result = values[position]
                break
        out.append(result)
    return tuple(out)


def fit(
    cells: list[Cell],
    *,
    a0: float = 1.0,
    b0: float = 1.0,
    kappa_prior: tuple[float, float] = KAPPA_PRIOR,
    mu_points: int = MU_POINTS,
    kappa_points: int = KAPPA_POINTS,
    kappa_min: float = KAPPA_MIN,
    kappa_max: float = KAPPA_MAX,
) -> Fit:
    """Posterior over ``(mu, kappa)`` by deterministic grid quadrature.

    The prior is ``mu ~ Beta(a0, b0)`` (uniform by default, matching the
    per-cell ``mcnemar`` prior) and ``kappa ~ InverseGamma(*kappa_prior)``.
    Cells with no paired items drop out — they carry no information and would
    only dilute the returned ``cells`` count.

    @note: Grid weights fold the prior density into the midpoint-rule spacing:
        ``mu`` is uniform on [0, 1] while ``kappa`` is gridded in log space, so
        its weight carries a ``dlog`` factor.
    @note: The likelihood uses the **discordant** pair count as the binomial
        ``n`` (McNemar). Using ``cell.pairs`` would count every unchanged item
        as a failed switch and dilute ``mu`` toward 0 on a real question bank.
    @note: Marginals are summarised on the full grid *before* points below
        :data:`GRID_WEIGHT_FLOOR` are pruned, so pruning cannot move the
        reported intervals.
    @return: The fitted posterior, with ``points`` already pruned.
    @raise ValueError: If the grid is degenerate (fewer than 2x2 points or an
        invalid ``kappa`` span).
    """
    if mu_points < 2 or kappa_points < 2:
        raise ValueError("grid resolution must be at least 2x2")
    if kappa_min <= 0.0 or kappa_max <= kappa_min:
        raise ValueError(f"invalid kappa grid: [{kappa_min}, {kappa_max}]")

    active = [cell for cell in cells if cell.pairs > 0]
    mus = [(i + 0.5) / mu_points for i in range(mu_points)]
    log_mu_prior = [math.log(beta_pdf(mu, a0, b0) / mu_points) for mu in mus]
    log_span = math.log(kappa_max) - math.log(kappa_min)
    dlog = log_span / kappa_points
    kappas = [math.exp(math.log(kappa_min) + (j + 0.5) * dlog) for j in range(kappa_points)]
    log_kappa_prior = [
        _invgamma_log_pdf(kappa, *kappa_prior) + math.log(kappa) + math.log(dlog)
        for kappa in kappas
    ]

    scores: list[tuple[float, float, float]] = []
    for i, mu in enumerate(mus):
        for j, kappa in enumerate(kappas):
            a = mu * kappa
            b = (1.0 - mu) * kappa
            log_likelihood = 0.0
            for cell in active:
                log_likelihood += log_beta_binomial(cell.improved, cell.discordant, a, b)
            scores.append((mu, kappa, log_mu_prior[i] + log_kappa_prior[j] + log_likelihood))

    peak = max(score for _, _, score in scores)
    unnormalised = [(mu, kappa, math.exp(score - peak)) for mu, kappa, score in scores]
    mass = sum(weight for _, _, weight in unnormalised)
    log_marginal = peak + math.log(mass)
    points = tuple((mu, kappa, weight / mass) for mu, kappa, weight in unnormalised)

    mu_marginal: dict[float, float] = {}
    kappa_marginal: dict[float, float] = {}
    for mu, kappa, weight in points:
        mu_marginal[mu] = mu_marginal.get(mu, 0.0) + weight
        kappa_marginal[kappa] = kappa_marginal.get(kappa, 0.0) + weight
    mu_values = list(mu_marginal)
    kappa_values = list(kappa_marginal)
    mu_eti = _weighted_quantiles(mu_values, [mu_marginal[v] for v in mu_values], (0.025, 0.975))
    kappa_eti = _weighted_quantiles(
        kappa_values, [kappa_marginal[v] for v in kappa_values], (0.025, 0.975)
    )
    points = tuple(point for point in points if point[2] >= GRID_WEIGHT_FLOOR)
    return Fit(
        mu=sum(mu * weight for mu, _, weight in points),
        mu_eti=mu_eti,
        kappa=sum(kappa * weight for _, kappa, weight in points),
        kappa_eti=kappa_eti,
        log_marginal=log_marginal,
        points=points,
        cells=len(active),
    )


def _sample_params(fit: Fit, draws: int, rng: random.Random) -> list[tuple[float, float]]:
    """Draw ``(mu, kappa)`` pairs from the grid posterior (inverse-CDF).

    @note: The grid is small, so the linear scan over the cumulative weights
        beats rebuilding a bisect structure on every call.
    """
    cumulative: list[float] = []
    running = 0.0
    for _, _, weight in fit.points:
        running += weight
        cumulative.append(running)
    total = cumulative[-1]
    sampled: list[tuple[float, float]] = []
    for _ in range(draws):
        target = rng.random() * total
        for index, bound in enumerate(cumulative):
            if target <= bound:
                mu, kappa, _ = fit.points[index]
                sampled.append((mu, kappa))
                break
        else:  # pragma: no cover - cumulative[-1] >= target by construction
            mu, kappa, _ = fit.points[-1]
            sampled.append((mu, kappa))
    return sampled


def cell_posteriors(
    fit: Fit,
    cells: list[Cell],
    *,
    draws: int = DRAWS,
    seed: int = 0,
    a_d: float = 1.0,
    b_d: float = 1.0,
) -> CellPosteriors:
    """Summarise every cell, keeping the joint draws for pooled quantities.

    Means and tail probabilities come from the exact grid mixture; the
    equal-tail interval and logit shift are Monte-Carlo summaries of the
    conditional beta draws. The seed is fixed by default so a given state
    always serialises to the same report (important for a monitoring series).

    ``a_d``/``b_d`` are the Beta prior on each cell's discordance rate. The
    uniform ``(1, 1)`` default is weakly informative: it never moves the
    sample mean (``(a_d + discordant) / (a_d + b_d + pairs)`` is the MLE when
    the prior is uniform) but it does keep the interval honest on a bank where
    almost nothing moved, because it shrinks the discordance rate towards the
    prior rather than pinning it at the observed count.

    @note: The discordance draws are taken after every switch-rate row so the
        phi draws stay byte-identical to the pre-discordance model for a given
        seed (reports remain comparable across a package upgrade).
    """
    if draws < 2:
        raise ValueError(f"draws must be at least 2, got {draws}")
    if a_d <= 0.0 or b_d <= 0.0:
        raise ValueError(f"discordance prior shapes must be positive, got a_d={a_d}, b_d={b_d}")
    active = [cell for cell in cells if cell.pairs > 0]
    rng = random.Random(seed)
    params = _sample_params(fit, draws, rng)
    phi_draws: list[list[float]] = []
    for mu, kappa in params:
        row: list[float] = []
        for cell in active:
            a = mu * kappa + cell.improved
            b = (1.0 - mu) * kappa + cell.regressed
            row.append(beta_sample(rng, a, b))
        phi_draws.append(row)
    discordance_draws: list[list[float]] = []
    for _ in params:
        row = []
        for cell in active:
            row.append(beta_sample(rng, a_d + cell.discordant, b_d + cell.unchanged))
        discordance_draws.append(row)

    summaries: dict[tuple[str, str], CellPosterior] = {}
    lower_index = int(0.025 * draws)
    upper_index = min(draws - 1, int(0.975 * draws))
    for position, cell in enumerate(active):
        column = sorted(row[position] for row in phi_draws)
        shifts = [_logit(row[position]) - _logit(mu) for (mu, _), row in zip(params, phi_draws)]
        summaries[cell.key] = CellPosterior(
            mean=fit.cell_mean(cell),
            eti_low=column[lower_index],
            eti_high=column[upper_index],
            p_improved=fit.cell_p_improved(cell),
            logit_shift=sum(shifts) / len(shifts),
        )
    return CellPosteriors(
        cells=tuple(active),
        summaries=summaries,
        phi_draws=tuple(tuple(row) for row in phi_draws),
        discordance_draws=tuple(tuple(row) for row in discordance_draws),
        params=tuple(params),
    )


def pooled_counts(cells: list[Cell]) -> tuple[int, int]:
    """Total discordant counts, for the additive multi-strata McNemar test.

    Summing the discordant pairs is the Bayes-analogue of pooling strata in
    the stratified McNemar / CMH test: it yields the pooled switch rate, and
    ``p_improved`` from it is the "did the layer move" headline.
    """
    return (sum(cell.improved for cell in cells), sum(cell.regressed for cell in cells))


@dataclass(frozen=True)
class Homogeneity:
    """Posterior predictive check of "the strata move together"."""

    ppc: float
    """Posterior predictive tail probability (Bayesian model check)."""
    observed: float
    expected: float
    draws: int
    cells: int

    @property
    def heterogeneous(self) -> bool:
        return self.ppc < 0.05


def homogeneity_ppc(
    cells: list[Cell],
    *,
    a0: float = 1.0,
    b0: float = 1.0,
    draws: int = PPC_DRAWS,
    seed: int = 0,
) -> Homogeneity | None:
    """Posterior predictive test of homogeneous stratum effects.

    The statistic is McNemar's dispersion on the discordant sub-population,

        T = sum_c (n01_c - n_c * phi)^2 / (n_c * phi * (1 - phi)),

    evaluated once against the observed counts and once against data
    simulated from the *pooled* model ``phi | data ~ Beta(a0 + K, b0 + N - K)``
    (``K``/``N`` the pooled discordant switches/trials). The reported ``ppc``
    is the fraction of replicated ``T`` reaching the observed one.

    @note: Both the trial count and the Pearson statistic use the **discordant**
        sub-population (``improved + regressed``); a cell with no discordant
        pairs carries no information and is excluded.
    @note: A large ``T`` means the cells do not share one switch rate — the
        guardrail condition. Why not a Bayes factor for ``kappa``? Because the
        pooled model is the ``kappa -> inf`` limit of the hierarchy, a boundary
        null whose Bayes factor is dominated by prior mass in a tail the data
        cannot inform (the Lindley/Bartlett effect). The predictive check asks
        the question that matters for monitoring — "would a layer whose views
        all moved together plausibly produce this spread?" — and is calibrated
        under the null, which ``hierarchy_test.py`` anchors.
    """
    data = [(cell.improved, cell.discordant) for cell in cells if cell.discordant > 0]
    trials = sum(n for _, n in data)
    if trials == 0 or draws < 2:
        return None
    successes = sum(k for k, _ in data)
    rng = random.Random(seed)
    a_post = a0 + successes
    b_post = b0 + trials - successes

    def dispersion(phi: float, counts: list[tuple[int, int]]) -> float:
        total = 0.0
        variance = phi * (1.0 - phi)
        if variance <= 0.0:
            return 0.0
        for k, n in counts:
            total += (k - n * phi) ** 2 / (n * variance)
        return total

    observed = 0.0
    expected = 0.0
    exceed = 0
    for _ in range(draws):
        phi = beta_sample(rng, a_post, b_post)
        observed_draw = dispersion(phi, data)
        simulated = [(_binomial(rng, n, phi), n) for _, n in data]
        replicated = dispersion(phi, simulated)
        observed += observed_draw
        expected += replicated
        exceed += replicated >= observed_draw
    return Homogeneity(
        ppc=exceed / draws,
        observed=observed / draws,
        expected=expected / draws,
        draws=draws,
        cells=len(data),
    )


def _binomial(rng: random.Random, n: int, p: float) -> int:
    """Binomial draw by Bernoulli trials — ``n`` is a question-bank size."""
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    return sum(1 for _ in range(n) if rng.random() < p)
