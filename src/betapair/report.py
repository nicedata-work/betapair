"""The report: one score, its movement, and the cells that moved it.

Everything here is derived from two layers:

* **stock** — the paired pass rate of the candidate condition (``score``,
  0-100), the number a dashboard plots over time; and
* **flow** — the multi-strata Bayesian McNemar posterior over the paired
  change (``delta_pp``), which is what tells you whether the movement is real
  and which cells caused it.

The two are algebraically linked: for paired items,

    delta_pass_rate = sum_c w_c (n01_c - n10_c) / sum_c w_c pairs_c
    score - baseline_score = 100 * delta_pass_rate

so the single number can always be expanded into exact per-cell contributions
(``contributions``) plus a guardrail that says whether the contributions are
homogeneous (``guardrail``). A regression in one stratum that is offset by
improvements elsewhere leaves ``score`` flat but cannot hide: it shows up as a
credible per-cell regression and as heterogeneity, both of which the guardrail
names.

Aggregation is a weighted micro-average over cells (stratum x outcome). The
effective weight of a cell is ``weight[outcome] * category_weight[category]``:

* ``weights`` — per-outcome importance (an outcome you trust more can count
  more);
* ``category_weights`` — the weight of the **category** a stratum belongs to
  (``tier-1=3`` means a tier-1 stratum's items weigh three times a default
  stratum's). Categories are external labels on the stratum; they change the
  aggregation and the breakdown, never the paired test itself.

Cells with no paired items are excluded from the score so a half-finished
stream never looks like a regression — the tracker reports those as
``pending``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from betapair.beta import beta_cdf, log_beta
from betapair.hierarchy import (
    DRAWS,
    PPC_DRAWS,
    Cell,
    Homogeneity,
    cell_posteriors,
    fit,
    homogeneity_ppc,
    pooled_counts,
)
from betapair.mcnemar import mcnemar, switch_evidence

#: Bumped when the JSON shape changes — reports are persisted per refresh.
#: v4 added ``retired``, ``score_held``, ``composition_effect_pp`` and
#: ``retired_held``; v5 added ``failed`` (view-caused failures) to ``Coverage``
#: and ``HealthReport``.
REPORT_VERSION = 5

#: The tier an untagged view falls back to **when tiers are in use at all**.
#: The vocabulary is fixed: ``tier-1`` critical, ``tier-2`` dependent,
#: ``tier-3`` auxiliary — so a project that tags only its critical views does
#: not accidentally score the rest as unimportant. When *no* view in the
#: comparison carries a tier, nothing is defaulted and every cell weighs the
#: same (the uniform case).
DEFAULT_CATEGORY = "tier-2"

#: Layer/tier verdict vocabulary. ``mixed`` is the honest middle when the
#: pooled movement is positive but the strata credibly disagree: neither a
#: clean improvement nor a regression, and not stable either.
Status = Literal["improving", "stable", "regressing", "mixed", "insufficient"]


@dataclass(frozen=True)
class CellCoverage:
    """Resolved/pending/errored/retired item counts for one cell.

    The tracker owns item-level state, so it is the only layer that can tell
    ``pending`` (not evaluated yet) from ``errored`` (the *harness* failed) from
    ``failed`` (the *view* failed) from ``retired`` (withdrawn from the bank).
    The report consumes this to gate on coverage: a crashing view must not make
    health look better by shrinking its own denominator.

    ``failed`` is the only one of the four that is **also** resolved: a
    view-caused failure is scored as a failure, so those items are counted in
    ``paired`` as well. See :data:`betapair.tracker.FAILED`.
    """

    stratum: str
    outcome: str
    category: str | None = None
    paired: int = 0
    pending: int = 0
    errored: int = 0
    retired: int = 0
    failed: int = 0
    """Paired items scored as a failure because the view caused the failure."""

    @property
    def unresolved(self) -> int:
        """Items that are still expected to produce a result.

        Retirement is deliberately excluded: withdrawing an item is a
        deliberate bank change, not data that failed to arrive, so it must not
        gate the verdict as if the pipeline were broken.
        """
        return self.pending + self.errored


class McNemarSummary(BaseModel):
    """Posterior summary of a paired comparison."""

    n_improved: int
    n_regressed: int
    discordant: int
    post_mean: float
    post_median: float
    eti_low: float
    eti_high: float
    bf10_point: float
    bf10_interval: float
    p_improved: float
    """``P(phi > .5)`` — the probability that switches ran net-improving."""


class HierarchySummary(BaseModel):
    """Pooled switch rate and between-strata concentration."""

    mu: float
    mu_eti_low: float
    mu_eti_high: float
    kappa: float
    kappa_eti_low: float
    kappa_eti_high: float
    log_marginal: float
    cells: int


class Guardrail(BaseModel):
    """Homogeneity check across strata, plus the cells it flags."""

    status: Literal["ok", "heterogeneous", "insufficient"]
    ppc: float | None = None
    """Posterior predictive tail probability; small means the strata disagree.
    A Bayesian model-check quantity, not a sampling p-value."""
    dispersion: float | None = None
    expected_dispersion: float | None = None
    draws: int = 0
    flagged: list[str] = Field(default_factory=list)
    """Cells with a credible regression, as ``stratum/outcome``.

    A cell is listed when the spike-and-slab regression model puts it in the
    posterior-FDR-controlled flag set (see :func:`regression_model`), so a wide
    layer does not manufacture a regression by chance.
    """
    reason: str = ""


class CategorySummary(BaseModel):
    """One criticality tier, expanded from the same cells as the layer score.

    Within a tier the category weight is constant, so it cancels out of
    ``score``/``delta_pp`` — the tier is summarised on its own terms, and
    ``weight`` only says how much it moved the layer's headline.
    """

    category: str | None = None
    weight: float = 1.0
    defaulted: int = 0
    """Cells in this tier whose view carried no tag and took ``DEFAULT_CATEGORY``."""
    views: int = 0
    cells: int = 0
    pairs: int = 0
    score: float | None = None
    baseline_score: float | None = None
    delta_pp: float | None = None
    discordant: int = 0
    p_improved: float | None = None
    """Pooled McNemar ``P(phi > .5)`` for this tier alone."""
    status: Status = "insufficient"
    delta_p_improved: float | None = None
    """Weighted ``P(delta > 0)`` for this tier, from the hierarchical draws."""


class Coverage(BaseModel):
    """How much of the comparison actually resolved, and whether that is enough."""

    paired: int = 0
    pending: int = 0
    errored: int = 0
    retired: int = 0
    """Items withdrawn from the bank — reported, but never gating."""
    failed: int = 0
    """Paired items scored as failures because the view, not the harness, failed.

    Already inside ``paired``: unlike ``pending`` and ``errored`` these did
    resolve, so they neither shrink the denominator nor gate the verdict. They
    are surfaced so a view that is *breaking* is distinguishable from one that
    is merely wrong — both cost the same score, but only one is an incident.
    """
    unresolved_share: float = 0.0
    critical_unresolved: int = 0
    """Pending/errored items in the highest-criticality tier present."""
    complete: bool = True
    reason: str = ""


class CellContribution(BaseModel):
    """One view x outcome cell, expanded from the pooled metric."""

    stratum: str
    outcome: str
    category: str | None = None
    """Effective tier used for the weight (may be the default, not a tag)."""
    defaulted: bool = False
    """True when the view carried no tier tag and fell back to the default."""
    weight: float = 1.0
    """Effective aggregation weight: outcome importance x category weight."""
    pairs: int
    improved: int
    regressed: int
    unchanged: int
    pass_rate: float
    baseline_pass_rate: float
    delta_pp: float
    """Observed pass-rate change in percentage points (improvement positive)."""
    delta_pp_low: float
    delta_pp_high: float
    p_improved: float
    p_regressed: float = 0.0
    """Posterior probability this cell is a true regression, from the
    spike-and-slab model (see :func:`regression_model`)."""
    logit_shift: float
    """Shrunken log-odds shift of this cell against the pooled layer."""
    verdict: Literal["improving", "stable", "regressing"]
    delta_pp_contribution: float = 0.0
    """This cell's additive share of ``delta_pp``: ``100 * w * (n01 - n10) /
    sum(w * pairs)``. Summing this field over every contribution reproduces
    ``delta_pp`` exactly, even when cells offset."""


class HealthReport(BaseModel):
    """The assembled paired-comparison report, ready to persist or serve as JSON."""

    version: int = REPORT_VERSION
    status: Status
    score: float | None = None
    """Paired candidate-epoch pass rate, 0-100."""
    baseline_score: float | None = None
    delta_pp: float | None = None
    delta_pp_low: float | None = None
    delta_pp_high: float | None = None
    delta_p_improved: float | None = None
    """Weighted ``P(delta_pp > 0)`` from the hierarchical draws — the
    probability behind the layer verdict. ``p_improved`` below is the
    unweighted pooled exact test, kept as a diagnostic only."""
    p_no_regression: float | None = None
    """``P(no cell truly regressed | data)`` from the spike-and-slab model."""
    bfdr: float | None = None
    """Posterior expected false discovery rate of the flagged set."""
    pi_mean: float | None = None
    """Posterior mean share of cells that truly regressed."""
    rope_pp: float = 0.0
    """ROPE half-width in percentage points used for the verdict."""
    p_improved: float | None = None
    pairs: int = 0
    pending: int = 0
    failed: int = 0
    """Paired items scored as failures because the view itself failed.

    A view-caused error (a timeout, a crash) is a result, so it is counted
    against the level instead of being dropped from it — the incentive TS2
    encodes as "quitting is a surrender".
    """
    retired: int = 0
    """Items withdrawn from the bank since the comparison began."""
    retired_held: int = 0
    """Retired items with a known outcome — the ones ``score_held`` folds in."""
    score_held: float | None = None
    """Level with retired items held at their last known outcome, 0-100.

    Equal to ``score`` when nothing was retired. This is what the bank would
    still read had its withdrawn items kept performing as they last did.
    """
    composition_effect_pp: float | None = None
    """``score - score_held`` in percentage points.

    How much of the headline level is explained by items *leaving* the bank
    rather than by views changing. Positive means a shrinking bank is
    flattering the score — the failure mode a paired test cannot see.
    """
    strata: int = 0
    outcomes: list[str] = Field(default_factory=list)
    baseline_epoch: str | None = None
    candidate_epoch: str | None = None
    pooled: McNemarSummary | None = None
    hierarchy: HierarchySummary | None = None
    category_weights: dict[str, float] = Field(default_factory=dict)
    default_category: str | None = DEFAULT_CATEGORY
    tiered: bool = False
    """True when at least one cell declared a tier, so weighting was applied."""
    coverage: Coverage = Field(default_factory=Coverage)
    """Resolved/pending/errored accounting and whether the status was gated."""
    categories: list[CategorySummary] = Field(default_factory=list)
    guardrail: Guardrail = Field(
        default_factory=lambda: Guardrail(status="insufficient", reason="not computed")
    )
    contributions: list[CellContribution] = Field(default_factory=list)


@dataclass(frozen=True)
class RegressionModel:
    """Bayesian two-group (spike-and-slab) posterior over which cells regressed.

    Each cell's switch rate is either exactly ``.5`` (the spike: no regression)
    or drawn from a Beta slab on ``[0, .5]``. The mixture weight ``pi`` gets a
    Beta hyperprior and is integrated out, so no empirical-Bayes plug-in is
    needed. ``flagged`` is the largest set whose posterior expected false
    discovery rate is within the target, which is the Bayesian counterpart of
    a multiplicity correction.
    """

    p_regressed: dict[tuple[str, str], float]
    """``P(cell is a true regression | data)`` for every cell."""
    flagged: frozenset[tuple[str, str]]
    bfdr: float
    """Posterior expected false discovery rate of ``flagged``."""
    p_no_regression: float
    """``P(no cell in the layer truly regressed | data)``."""
    pi_mean: float
    """Posterior mean of the mixture weight (share of true regressions)."""


@dataclass(frozen=True)
class CellVerdict:
    """A cell's exact McNemar evidence and the verdict drawn from it."""

    p_improved: float
    """``P(phi > .5)`` — the Bayesian evidence of improvement."""
    p_regressed: float
    """Posterior probability this cell is a true regression, from the
    spike-and-slab model (see :func:`regression_model`)."""
    verdict: Literal["improving", "stable", "regressing"]


def _log1p_exp(value: float) -> float:
    """``log(1 + exp(value))`` without overflow."""
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _sigmoid(value: float) -> float:
    """Logistic function, stable in both tails."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def regression_model(
    cells: list[Cell],
    *,
    a0: float = 1.0,
    b0: float = 1.0,
    fdr: float = 0.05,
    weights: dict[tuple[str, str], float] | None = None,
    pi_prior: tuple[float, float] = (1.0, 1.0),
    pi_points: int = 199,
) -> RegressionModel:
    """Fit the spike-and-slab model and pick the flag set by Bayesian FDR.

    For each cell the Bayes factor of "regressed" (slab on ``[0, .5]``) against
    "not regressed" (point spike at ``.5``) is computed in closed form from the
    beta-binomial marginal, so this needs no sampling and stays in the stdlib
    ``math`` core::

        r_c = [m_c * P(phi_c <= .5 | data) / Z1] / [C(n, k) * .5**n]

    ``pi ~ Beta(*pi_prior)`` is integrated out on a grid, and the per-cell
    posterior odds are multiplied by the cell's normalised effective weight, so
    a critical cell starts with a higher prior odds of having regressed. The
    flag set is the largest prefix of cells ordered by ``P(regressed | data)``
    whose posterior expected FDR stays within ``fdr``. A cell with weight 0 is
    muted and can never be flagged.

    @note: The decision is Bayesian end to end: ``fdr`` is a posterior expected
        FDR, not a frequentist family-wise error rate, and no p-value enters.
    @return: The fitted :class:`RegressionModel`.
    @raise ValueError: If ``fdr`` is outside (0, 1), a prior shape is not
        positive, or ``pi_points`` is below 2.
    """
    if not 0.0 < fdr < 1.0:
        raise ValueError(f"fdr must be in (0, 1), got {fdr}")
    if pi_prior[0] <= 0.0 or pi_prior[1] <= 0.0:
        raise ValueError(f"pi_prior shapes must be positive, got {pi_prior}")
    if pi_points < 2:
        raise ValueError(f"pi_points must be at least 2, got {pi_points}")

    keys = [cell.key for cell in cells]
    z1 = beta_cdf(0.5, a0, b0)
    if z1 <= 0.0:
        z1 = 1e-300
    log_ratio: dict[tuple[str, str], float] = {}
    for cell in cells:
        test = switch_evidence(cell.improved, cell.regressed, a0=a0, b0=b0)
        discordant = test.discordant
        if discordant <= 0:
            log_ratio[cell.key] = 0.0
            continue
        p_low = 1.0 - test.p_improved
        if p_low <= 0.0:
            log_ratio[cell.key] = -math.inf
            continue
        log_ratio[cell.key] = (
            log_beta(a0 + cell.improved, b0 + cell.regressed)
            - log_beta(a0, b0)
            + math.log(p_low)
            - math.log(z1)
            - discordant * math.log(0.5)
        )

    raw = {
        key: (max(0.0, float(weights.get(key, 1.0))) if weights is not None else 1.0)
        for key in keys
    }
    mean_weight = sum(raw.values()) / len(raw) if raw else 1.0
    log_multiplier = {
        key: (math.log(value / mean_weight) if value > 0.0 and mean_weight > 0.0 else -math.inf)
        for key, value in raw.items()
    }

    grid = [(index + 0.5) / pi_points for index in range(pi_points)]
    log_prior = [
        (pi_prior[0] - 1.0) * math.log(pi)
        + (pi_prior[1] - 1.0) * math.log1p(-pi)
        - log_beta(pi_prior[0], pi_prior[1])
        for pi in grid
    ]
    log_posterior: list[float] = []
    for pi, prior in zip(grid, log_prior, strict=True):
        base = math.log(pi / (1.0 - pi))
        value = prior
        for key in keys:
            value += _log1p_exp(base + log_multiplier[key] + log_ratio[key])
            value -= _log1p_exp(base + log_multiplier[key])
        log_posterior.append(value)
    peak = max(log_posterior)
    mass = [math.exp(value - peak) for value in log_posterior]
    total = sum(mass)
    posterior = [value / total for value in mass]

    p_regressed: dict[tuple[str, str], float] = {}
    for key in keys:
        accumulator = 0.0
        for weight, pi in zip(posterior, grid, strict=True):
            log_odds = math.log(pi / (1.0 - pi)) + log_multiplier[key] + log_ratio[key]
            accumulator += weight * _sigmoid(log_odds)
        p_regressed[key] = accumulator

    p_no_regression = 0.0
    for weight, pi in zip(posterior, grid, strict=True):
        base = math.log(pi / (1.0 - pi))
        product = 1.0
        for key in keys:
            product *= math.exp(-_log1p_exp(base + log_multiplier[key] + log_ratio[key]))
        p_no_regression += weight * product

    ordered = sorted(keys, key=lambda key: -p_regressed[key])
    cumulative = 0.0
    best = 0
    for index, key in enumerate(ordered, start=1):
        cumulative += 1.0 - p_regressed[key]
        if cumulative / index <= fdr:
            best = index
    flagged = frozenset(ordered[:best])
    bfdr = sum(1.0 - p_regressed[key] for key in flagged) / len(flagged) if flagged else 0.0
    return RegressionModel(
        p_regressed=p_regressed,
        flagged=flagged,
        bfdr=bfdr,
        p_no_regression=p_no_regression,
        pi_mean=sum(weight * pi for weight, pi in zip(posterior, grid, strict=True)),
    )


def verdicts_from_model(
    cells: list[Cell],
    model: RegressionModel,
    *,
    a0: float = 1.0,
    b0: float = 1.0,
    improving_prob: float = 0.95,
) -> dict[tuple[str, str], CellVerdict]:
    """Turn a fitted :class:`RegressionModel` into per-cell verdicts.

    ``improving`` and ``stable`` come from the cell's own McNemar posterior at
    ``improving_prob``; ``regressing`` is membership of the model's FDR-
    controlled flag set. Separated from :func:`regression_model` so the report
    fits the model once and still has the global quantities to hand.
    """
    verdicts: dict[tuple[str, str], CellVerdict] = {}
    for cell in cells:
        test = switch_evidence(cell.improved, cell.regressed, a0=a0, b0=b0)
        if test.discordant and test.p_improved >= improving_prob:
            verdict: Literal["improving", "stable", "regressing"] = "improving"
        elif cell.key in model.flagged:
            verdict = "regressing"
        else:
            verdict = "stable"
        verdicts[cell.key] = CellVerdict(
            p_improved=test.p_improved,
            p_regressed=model.p_regressed[cell.key],
            verdict=verdict,
        )
    return verdicts


def cell_verdicts(
    cells: list[Cell],
    *,
    a0: float = 1.0,
    b0: float = 1.0,
    improving_prob: float = 0.95,
    fdr: float = 0.05,
    weights: dict[tuple[str, str], float] | None = None,
    pi_prior: tuple[float, float] = (1.0, 1.0),
    pi_points: int = 199,
) -> dict[tuple[str, str], CellVerdict]:
    """Per-cell verdict from the Bayesian two-group regression model.

    ``improving`` and ``stable`` come from the cell's own McNemar posterior at
    ``improving_prob``. ``regressing`` is the FDR-controlled flag set of
    :func:`regression_model` — a Bayesian multiplicity correction, so a wide
    layer does not manufacture regressions by chance and a critical cell is
    not drowned by a field of auxiliary ones. ``weights`` is the effective
    importance of each cell; it multiplies the prior odds that the cell
    regressed (normalised so the overall sparsity is unchanged), and a weight
    of 0 mutes the cell.

    @note: Unlike the previous weighted-Holm gate, nothing here is a p-value
        or a family-wise error rate: the flag is a posterior decision under an
        explicit spike-and-slab model. See ``docs/validation.md`` for the
        frequentist operating characteristics of the procedure.
    """
    model = regression_model(
        cells,
        a0=a0,
        b0=b0,
        fdr=fdr,
        weights=weights,
        pi_prior=pi_prior,
        pi_points=pi_points,
    )
    return verdicts_from_model(
        cells,
        model,
        a0=a0,
        b0=b0,
        improving_prob=improving_prob,
    )


def delta_status(
    delta_draws: list[float] | tuple[float, ...],
    *,
    rope: float,
    alpha: float,
    flagged: bool = False,
    heterogeneous: bool = False,
) -> Status:
    """The layer/tier verdict from the weighted posterior of its delta.

    One function so the report, the tier roll-up and any downstream timeline
    cannot drift apart. The verdict is a Bayesian region-of-practical-equivalence
    decision on the *weighted* delta draws, which is the same quantity the
    headline ``delta_pp`` reports:

    * ``regressing`` when a credible per-cell regression exists anywhere in
      the layer (``flagged``), or ``P(delta < -rope) >= 1 - alpha``;
    * ``improving`` when ``P(delta > rope) >= 1 - alpha`` — downgraded to
      ``mixed`` when the homogeneity guardrail says the strata disagree;
    * ``stable`` when ``P(|delta| <= rope) >= 1 - alpha``;
    * ``insufficient`` when the draws do not support any of the above.

    @note: Because the draws include the discordance-rate posterior, an
        unchanged bank is not automatically ``insufficient``: as the pair
        count grows the delta concentrates inside the ROPE, which is how
        "20,000 items and nothing moved" becomes positive evidence of
        stability.
    @note: A credible regression always wins, even if the weighted net is
        positive — that is the invariant that keeps one bad view from being
        averaged away.
    """
    if not delta_draws:
        return "insufficient"
    total = len(delta_draws)
    p_up = sum(1 for value in delta_draws if value > rope) / total
    p_down = sum(1 for value in delta_draws if value < -rope) / total
    p_rope = sum(1 for value in delta_draws if -rope <= value <= rope) / total
    if flagged or p_down >= 1.0 - alpha:
        return "regressing"
    if p_up >= 1.0 - alpha:
        return "mixed" if heterogeneous else "improving"
    if p_rope >= 1.0 - alpha:
        return "stable"
    return "insufficient"


def _quantile(sorted_values: list[float], prob: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty series")
    index = min(len(sorted_values) - 1, max(0, int(prob * len(sorted_values))))
    return sorted_values[index]


@dataclass(frozen=True)
class Weighting:
    """What each cell weighs under the outcome x criticality-tier policy.

    One object decides three things so the report, the tracker and the
    service's timeline cannot drift apart: whether the layer is tiered at all,
    which tier an untagged cell fell back to, and the effective weight
    (``outcome importance x tier weight``) of every cell.
    """

    tiered: bool
    default_category: str | None
    weights: dict[tuple[str, str], float]
    """Effective importance: outcome importance x tier weight.

    One number governs both how much a cell matters to the score and the
    delta, and how much of a head start it gets in the spike-and-slab
    regression model. A cell with weight 0 is muted everywhere: excluded from
    the score and unable to be flagged.
    """
    categories: dict[tuple[str, str], str | None]

    def weight(self, cell: Cell) -> float:
        return self.weights[cell.key]

    def category(self, cell: Cell) -> str | None:
        return self.categories[cell.key]

    def defaulted(self, cell: Cell) -> bool:
        """True when the cell carried no tier tag and took the fallback tier."""
        return self.tiered and cell.category is None


def weighting(
    cells: list[Cell],
    *,
    weights: dict[str, float] | None = None,
    category_weights: dict[str, float] | None = None,
    default_category: str | None = DEFAULT_CATEGORY,
    tiered: bool | None = None,
) -> Weighting:
    """Resolve effective weights for ``cells`` (see :class:`Weighting`).

    Tiering applies only when at least one cell declares a category; an
    untagged cell then takes ``default_category``. A set with no tier anywhere
    stays uniform: every cell weighs 1.0, and nothing is marked defaulted.

    @param tiered: Override the auto-detection. The held-level overlay passes
        the layer's own decision, so adding cells that exist only because an
        item was retired cannot flip the policy and re-weight the report.
    """
    weight_of = dict(weights or {})
    tier_of = dict(category_weights or {})
    for name, value in (*weight_of.items(), *tier_of.items()):
        if value < 0.0:
            raise ValueError(f"weight for {name!r} must be >= 0, got {value}")
    if tiered is None:
        tiered = any(cell.category is not None for cell in cells)
    cell_weights: dict[tuple[str, str], float] = {}
    cell_categories: dict[tuple[str, str], str | None] = {}
    for cell in cells:
        category = cell.category
        if category is None and tiered:
            category = default_category
        tier = tier_of.get(category, 1.0) if category is not None else 1.0
        cell_weights[cell.key] = weight_of.get(cell.outcome, 1.0) * tier
        cell_categories[cell.key] = category
    return Weighting(
        tiered=tiered,
        default_category=default_category,
        weights=cell_weights,
        categories=cell_categories,
    )


def build_report(
    cells: list[Cell],
    *,
    weights: dict[str, float] | None = None,
    category_weights: dict[str, float] | None = None,
    default_category: str | None = DEFAULT_CATEGORY,
    a0: float = 1.0,
    b0: float = 1.0,
    a_d: float = 1.0,
    b_d: float = 1.0,
    prob_interval: float = 0.95,
    draws: int = DRAWS,
    ppc_draws: int = PPC_DRAWS,
    seed: int = 0,
    alpha: float = 0.05,
    rope_pp: float = 1.0,
    pi_prior: tuple[float, float] = (1.0, 1.0),
    pi_points: int = 199,
    pending: int = 0,
    failed: int = 0,
    retired: int = 0,
    coverage: list[CellCoverage] | None = None,
    held_counts: Mapping[tuple[str, str], tuple[int, int]] | None = None,
    max_unresolved_share: float = 0.25,
    baseline_epoch: str | None = None,
    candidate_epoch: str | None = None,
) -> HealthReport:
    """Assemble the health report from per-cell discordant counts.

    ``weights`` is per-outcome importance, ``category_weights`` the weight of
    the category a cell's stratum belongs to. Categories are applied only when
    at least one cell declares one: an uncategorised cell then falls back to
    ``default_category``, while a comparison with no category anywhere stays
    uniform (every cell weighs 1.0). The category weight also multiplies the
    prior odds that a cell truly regressed in the spike-and-slab gate (see
    :func:`regression_model`), so a critical view is not drowned by a wide
    field of auxiliary cells.

    ``alpha`` is the single posterior threshold behind every decision: the
    homogeneity check, the per-cell improving threshold (``>= 1 - alpha``),
    the weighted delta ROPE, and the posterior FDR target for the regression
    gate. ``pi_prior`` is the Beta hyperprior on the share of cells that truly
    regressed, gridded at ``pi_points``.

    ``rope_pp`` is the region of practical equivalence for the layer verdict,
    in percentage points: a net movement under this size counts as ``stable``
    rather than improving/regressing. The per-cell regression flag is
    unaffected — a credible regression in any view still forces ``regressing``.

    ``coverage`` carries per-cell pending/errored/retired/failed counts from the
    tracker; when the unresolved share exceeds ``max_unresolved_share``, or any
    item in the most critical tier present is unresolved, the status is gated to
    ``insufficient`` so a crashing view cannot improve health by shrinking its
    own denominator. ``failed`` items are *not* unresolved: they are scored as
    failures and counted in ``paired`` (see :data:`betapair.tracker.FAILED`),
    and are passed through only so the report can show them. Retirement is
    excluded from the gate — it is a deliberate bank change, not missing data —
    and is reported through ``score_held`` and ``composition_effect_pp``
    instead.

    ``held_counts`` maps a cell to ``(items, passes)`` for items retired with a
    known outcome. They are folded back into ``score_held`` only, so a level
    that rose because hard questions were deleted is separable from one that
    rose because the views changed.

    @note: The layer verdict is derived from the *weighted* hierarchical delta
        draws — the same quantity as the headline ``delta_pp`` — not from the
        unweighted pooled test. ``p_improved`` is kept as an unweighted pooled
        diagnostic; ``delta_p_improved`` is the probability behind the verdict.
    @note: The delta draws multiply the switch rate by a discordance-rate draw,
        so the interval and the ROPE are statements about the population, not
        reconstructions of the observed sample delta.
    @note: A cell's improving/stable evidence comes from its own exact McNemar
        posterior, never from the borrowed-strength hierarchy estimate; the
        regression flag comes from the layer-wide spike-and-slab model. The
        hierarchy still shapes ``logit_shift`` and the delta interval.
    @note: A cell with weight 0 is muted: excluded from the score, the pooled
        diagnostic and the homogeneity check, so a view that does not count
        cannot make the layer look inconsistent.
    @note: Coverage gates everything except a credible regression: incomplete
        data must not manufacture health, but it must never bury a flagged
        regression either.
    @return: The assembled :class:`HealthReport`.
    @raise ValueError: If ``alpha`` is outside (0, 0.5), ``rope_pp`` is
        negative, or ``max_unresolved_share`` is outside [0, 1].
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if rope_pp < 0.0:
        raise ValueError(f"rope_pp must be >= 0, got {rope_pp}")
    if not 0.0 <= max_unresolved_share <= 1.0:
        raise ValueError(f"max_unresolved_share must be in [0, 1], got {max_unresolved_share}")
    tier_of = dict(category_weights or {})
    active = [cell for cell in cells if cell.pairs > 0]
    policy = weighting(
        active,
        weights=weights,
        category_weights=category_weights,
        default_category=default_category,
    )
    tiered = policy.tiered
    outcomes = sorted({cell.outcome for cell in active})
    strata = len({cell.stratum for cell in active})
    pairs = sum(cell.pairs for cell in active)
    cov = _coverage(
        coverage,
        tier_of,
        pending=pending,
        failed=failed,
        pairs=pairs,
        max_unresolved_share=max_unresolved_share,
    )

    if pairs == 0 or not active:
        return HealthReport(
            status="insufficient",
            pending=pending,
            failed=failed,
            retired=retired,
            strata=strata,
            outcomes=outcomes,
            baseline_epoch=baseline_epoch,
            candidate_epoch=candidate_epoch,
            category_weights=dict(tier_of),
            default_category=default_category,
            coverage=cov,
            guardrail=Guardrail(status="insufficient", reason="no paired items yet"),
        )

    weighted_pairs = sum(policy.weight(cell) * cell.pairs for cell in active)
    if weighted_pairs <= 0.0:
        return HealthReport(
            status="insufficient",
            pending=pending,
            failed=failed,
            strata=strata,
            outcomes=outcomes,
            baseline_epoch=baseline_epoch,
            candidate_epoch=candidate_epoch,
            category_weights=dict(tier_of),
            default_category=default_category,
            tiered=tiered,
            retired=retired,
            coverage=cov,
            guardrail=Guardrail(status="insufficient", reason="all outcome weights are zero"),
        )

    level = sum(policy.weight(cell) * cell.passed for cell in active) / weighted_pairs
    baseline_level = (
        sum(policy.weight(cell) * cell.baseline_passed for cell in active) / weighted_pairs
    )
    observed_delta = level - baseline_level
    epsilon = rope_pp / 100.0
    held = _held_overlay(
        active,
        held_counts,
        coverage,
        policy,
        score=100.0 * level,
        weights=weights,
        category_weights=category_weights,
        default_category=default_category,
    )

    hierarchy_fit = fit(active, a0=a0, b0=b0)
    posteriors = cell_posteriors(hierarchy_fit, active, draws=draws, seed=seed, a_d=a_d, b_d=b_d)

    delta_draws: list[float] = []
    per_cell_delta: dict[tuple[str, str], list[float]] = {cell.key: [] for cell in posteriors.cells}
    for row, discordance in zip(posteriors.phi_draws, posteriors.discordance_draws, strict=True):
        total = 0.0
        for position, cell in enumerate(posteriors.cells):
            weight = policy.weight(cell)
            cell_delta = (2.0 * row[position] - 1.0) * discordance[position]
            total += weight * cell.pairs * cell_delta
            per_cell_delta[cell.key].append(cell_delta)
        delta_draws.append(total / weighted_pairs)
    delta_draws.sort()
    delta_p_improved = sum(1 for value in delta_draws if value > 0.0) / len(delta_draws)

    stat_cells = [cell for cell in active if policy.weight(cell) > 0.0]
    n_improved, n_regressed = pooled_counts(stat_cells)
    pooled = mcnemar(n_improved, n_regressed, a0=a0, b0=b0, prob_interval=prob_interval)
    pooled_p = pooled.p_improved if pooled.discordant > 0 else None

    model = regression_model(
        active,
        a0=a0,
        b0=b0,
        fdr=alpha,
        weights=policy.weights,
        pi_prior=pi_prior,
        pi_points=pi_points,
    )
    verdicts = verdicts_from_model(
        active,
        model,
        a0=a0,
        b0=b0,
        improving_prob=1.0 - alpha,
    )

    contributions: list[CellContribution] = []
    flagged: list[str] = []
    for cell in active:
        summary = posteriors.summaries[cell.key]
        weight = policy.weight(cell)
        observed_cell_delta = 100.0 * (cell.improved - cell.regressed) / cell.pairs
        interval = sorted(per_cell_delta[cell.key])
        cell_test = verdicts[cell.key]
        verdict = cell_test.verdict
        if verdict == "regressing":
            flagged.append(f"{cell.stratum}/{cell.outcome}")
        contribution = 100.0 * weight * (cell.improved - cell.regressed) / weighted_pairs
        contributions.append(
            CellContribution(
                stratum=cell.stratum,
                outcome=cell.outcome,
                category=policy.category(cell),
                defaulted=policy.defaulted(cell),
                weight=weight,
                pairs=cell.pairs,
                improved=cell.improved,
                regressed=cell.regressed,
                unchanged=cell.unchanged,
                pass_rate=cell.pass_rate,
                baseline_pass_rate=cell.baseline_pass_rate,
                delta_pp=observed_cell_delta,
                delta_pp_low=100.0 * _quantile(interval, 0.025),
                delta_pp_high=100.0 * _quantile(interval, 0.975),
                p_improved=cell_test.p_improved,
                p_regressed=cell_test.p_regressed,
                logit_shift=summary.logit_shift,
                verdict=verdict,
                delta_pp_contribution=contribution,
            )
        )
    contributions.sort(
        key=lambda item: (
            -abs(item.weight * (item.improved - item.regressed)),
            item.stratum,
            item.outcome,
        )
    )
    flagged.sort()

    category_delta_draws = _category_delta_draws(active, policy, per_cell_delta, len(delta_draws))
    categories = _categories(
        active,
        policy,
        tier_of,
        a0=a0,
        b0=b0,
        alpha=alpha,
        rope=epsilon,
        category_delta_draws=category_delta_draws,
        verdicts=verdicts,
    )

    heterogeneity = homogeneity_ppc(stat_cells, a0=a0, b0=b0, draws=ppc_draws, seed=seed)
    guardrail = _guardrail(
        heterogeneity,
        flagged,
        pooled_p,
        alpha,
        cells=len(stat_cells),
        discordant=pooled.discordant,
    )

    layer_verdict = delta_status(
        delta_draws,
        rope=epsilon,
        alpha=alpha,
        flagged=bool(flagged),
        heterogeneous=guardrail.status == "heterogeneous",
    )
    status: Status = (
        "insufficient" if not cov.complete and layer_verdict != "regressing" else layer_verdict
    )

    return HealthReport(
        status=status,
        score=100.0 * level,
        baseline_score=100.0 * baseline_level,
        delta_pp=100.0 * observed_delta,
        delta_pp_low=100.0 * _quantile(delta_draws, 0.025),
        delta_pp_high=100.0 * _quantile(delta_draws, 0.975),
        delta_p_improved=delta_p_improved,
        p_no_regression=model.p_no_regression,
        bfdr=model.bfdr,
        pi_mean=model.pi_mean,
        rope_pp=rope_pp,
        p_improved=pooled_p,
        pairs=pairs,
        pending=pending,
        failed=failed,
        retired=retired,
        retired_held=held.items,
        score_held=held.score,
        composition_effect_pp=held.effect_pp,
        coverage=cov,
        strata=strata,
        outcomes=outcomes,
        baseline_epoch=baseline_epoch,
        candidate_epoch=candidate_epoch,
        pooled=McNemarSummary(
            n_improved=pooled.n_improved,
            n_regressed=pooled.n_regressed,
            discordant=pooled.discordant,
            post_mean=pooled.post_mean,
            post_median=pooled.post_median,
            eti_low=pooled.eti_low,
            eti_high=pooled.eti_high,
            bf10_point=pooled.bf10_point,
            bf10_interval=pooled.bf10_interval,
            p_improved=pooled.p_improved,
        ),
        hierarchy=HierarchySummary(
            mu=hierarchy_fit.mu,
            mu_eti_low=hierarchy_fit.mu_eti[0],
            mu_eti_high=hierarchy_fit.mu_eti[1],
            kappa=hierarchy_fit.kappa,
            kappa_eti_low=hierarchy_fit.kappa_eti[0],
            kappa_eti_high=hierarchy_fit.kappa_eti[1],
            log_marginal=hierarchy_fit.log_marginal,
            cells=hierarchy_fit.cells,
        ),
        category_weights=dict(tier_of),
        default_category=default_category,
        tiered=tiered,
        categories=categories,
        guardrail=guardrail,
        contributions=contributions,
    )


def parse_category_weights(text: str | None) -> dict[str, float]:
    """Parse ``"tier-1=3,tier-2=1.5"`` into ``{"tier-1": 3.0, "tier-2": 1.5}``.

    A convenience for configuration surfaces that carry one string instead of
    a mapping. Blank means "no category counts for more".

    @raise ValueError: If an entry is not ``name=value``, is not numeric, or
        carries a negative weight.
    """
    if text is None:
        return {}
    weights: dict[str, float] = {}
    for chunk in text.replace("\n", ",").split(","):
        entry = chunk.strip()
        if not entry:
            continue
        name, separator, raw = entry.partition("=")
        name = name.strip()
        if not separator or not name:
            raise ValueError(f"category weight {entry!r} is not 'name=value'")
        try:
            value = float(raw.strip())
        except ValueError as exc:
            raise ValueError(f"category weight {entry!r} is not a number") from exc
        if value < 0.0:
            raise ValueError(f"category weight for {name!r} must be >= 0, got {value}")
        weights[name] = value
    return weights


@dataclass(frozen=True)
class _Held:
    """Level with retired items held at their last known outcome."""

    score: float | None
    effect_pp: float | None
    items: int


def _held_overlay(
    active: list[Cell],
    held_counts: Mapping[tuple[str, str], tuple[int, int]] | None,
    coverage: list[CellCoverage] | None,
    policy: Weighting,
    *,
    score: float | None,
    weights: dict[str, float] | None,
    category_weights: dict[str, float] | None,
    default_category: str | None,
) -> _Held:
    """Read the level again with retired items held at their last known outcome.

    ``held_counts`` maps a cell to ``(items, passes)``: the retired items whose
    last real outcome is known. Folding them back into the level turns the
    silent failure mode of a shrinking bank into a number — ``score -
    score_held`` — so a level that only rose because hard questions were deleted
    is visible instead of being read as an improvement.

    @note: The overlay is arithmetic only. Retired items are added to the
        numerator and denominator of the level and to nothing else, so the
        paired test, the hierarchy, the guardrail and every interval are
        bit-identical to a run without retirement.
    @note: Weights for the cells that exist only because of retirement are
        resolved with the layer's own ``tiered`` decision, so the overlay can
        never re-weight the cells that ``score`` is built from.
    @note: An item retired before it was ever measured has no outcome to hold
        and is left out; it is still visible through ``retired`` and
        ``coverage.retired``.
    @return: The held level, the composition effect, and how many retired items
        carried a known outcome.
    """
    extra = {key: counts for key, counts in (held_counts or {}).items() if counts[0] > 0}
    if not extra or score is None:
        return _Held(score=score, effect_pp=0.0 if score is not None else None, items=0)
    by_key = {cell.key: cell for cell in active}
    stratum_category: dict[str, str | None] = {
        row.stratum: row.category for row in (coverage or [])
    }
    stratum_category.update({cell.stratum: cell.category for cell in active})
    held_only = [
        Cell(
            stratum=stratum,
            outcome=outcome,
            pairs=counts[0],
            passed=counts[1],
            category=stratum_category.get(stratum),
        )
        for (stratum, outcome), counts in sorted(extra.items())
        if (stratum, outcome) not in by_key
    ]
    extended = weighting(
        [*active, *held_only],
        weights=weights,
        category_weights=category_weights,
        default_category=default_category,
        tiered=policy.tiered,
    )
    weighted_pairs = 0.0
    weighted_passes = 0.0
    for key in sorted(set(by_key) | set(extra)):
        weight = extended.weights[key]
        base = by_key.get(key)
        held_pairs, held_passed = extra.get(key, (0, 0))
        weighted_pairs += weight * ((base.pairs if base is not None else 0) + held_pairs)
        weighted_passes += weight * ((base.passed if base is not None else 0) + held_passed)
    if weighted_pairs <= 0.0:
        return _Held(score=None, effect_pp=None, items=0)
    held_score = 100.0 * weighted_passes / weighted_pairs
    return _Held(
        score=held_score,
        effect_pp=score - held_score,
        items=sum(counts[0] for counts in extra.values()),
    )


def _coverage(
    coverage: list[CellCoverage] | None,
    category_weights: dict[str, float],
    *,
    pending: int,
    pairs: int,
    max_unresolved_share: float,
    failed: int = 0,
) -> Coverage:
    """Decide whether the comparison resolved enough to be judged.

    With no per-cell coverage (a direct :func:`build_report` caller) this
    falls back to the scalar pending count over the paired count. The tracker
    supplies per-cell counts so the critical tier can be held to a stricter
    standard: any unresolved item in the most critical tier present gates the
    verdict regardless of the global share.

    @note: The critical tier is detected from the coverage itself, not from the
        active cells: a tier-1 view that is entirely pending or errored has no
        active cell, and that is exactly the case the gate must catch.
    @note: Retirement is counted but does not gate (see
        :attr:`CellCoverage.unresolved`). It is reported so a shrinking bank is
        visible rather than silently redefining the score's denominator.
    """
    if coverage is None:
        total = pairs + pending
        share = pending / total if total > 0 else 0.0
        complete = share <= max_unresolved_share
        reason = (
            ""
            if complete
            else f"{pending} of {total} items unresolved ({share:.0%} > {max_unresolved_share:.0%})"
        )
        return Coverage(
            paired=pairs,
            pending=pending,
            errored=0,
            failed=failed,
            unresolved_share=share,
            complete=complete,
            reason=reason,
        )

    paired = sum(item.paired for item in coverage)
    pend = sum(item.pending for item in coverage)
    errored = sum(item.errored for item in coverage)
    retired = sum(item.retired for item in coverage)
    view_failed = sum(item.failed for item in coverage)
    # Retired items are out of the comparison entirely, so they are excluded
    # from the gate's denominator as well as its numerator: a bank that shrank
    # to five questions with all five pending is a real coverage failure.
    total = paired + pend + errored
    share = (pend + errored) / total if total > 0 else 0.0

    critical_unresolved = 0
    tier_weights = {
        item.category: category_weights.get(item.category, 1.0)
        for item in coverage
        if item.category is not None
    }
    if tier_weights:
        top = max(tier_weights.values())
        if top > 0.0:
            critical_unresolved = sum(
                item.unresolved
                for item in coverage
                if item.category is not None and tier_weights[item.category] == top
            )

    reasons: list[str] = []
    if share > max_unresolved_share:
        reasons.append(
            f"{pend + errored} of {total} items unresolved ({share:.0%} > {max_unresolved_share:.0%})"
        )
    if critical_unresolved:
        reasons.append(f"{critical_unresolved} unresolved items in the most critical tier")
    return Coverage(
        paired=paired,
        pending=pend,
        errored=errored,
        retired=retired,
        failed=view_failed,
        unresolved_share=share,
        critical_unresolved=critical_unresolved,
        complete=not reasons,
        reason="; ".join(reasons),
    )


def _category_delta_draws(
    cells: list[Cell],
    policy: Weighting,
    per_cell_delta: dict[tuple[str, str], list[float]],
    draws: int,
) -> dict[str | None, list[float]]:
    """Aggregate the per-cell delta draws into a weighted draw per tier.

    The tier's own category weight is constant inside the group and cancels,
    but the outcome weights do not, so a tier with two differently-weighted
    outcomes is summarised with the same weighting as the layer.
    """
    grouped: dict[str | None, list[Cell]] = {}
    for cell in cells:
        grouped.setdefault(policy.category(cell), []).append(cell)
    out: dict[str | None, list[float]] = {}
    for category, group in grouped.items():
        denominator = sum(policy.weight(cell) * cell.pairs for cell in group)
        if denominator <= 0.0:
            continue
        out[category] = [
            sum(
                policy.weight(cell) * cell.pairs * per_cell_delta[cell.key][index] for cell in group
            )
            / denominator
            for index in range(draws)
        ]
    return out


def _categories(
    cells: list[Cell],
    policy: Weighting,
    category_weights: dict[str, float],
    *,
    a0: float,
    b0: float,
    alpha: float,
    rope: float,
    category_delta_draws: dict[str | None, list[float]],
    verdicts: dict[tuple[str, str], CellVerdict],
) -> list[CategorySummary]:
    """Per-tier roll-up of the same cells the layer score is built from.

    Numbers are comparable across tiers because the tier's own weight is a
    constant inside the group and cancels. ``status`` comes from the tier's own
    weighted delta draws via :func:`delta_status`, and includes the tier's own
    flagged cells — a tier with a credible regression can never read
    ``improving`` while that regression sits inside it.

    @note: Per-cell escalation stays inside the tier, so a tier's verdict is
        never driven by a cell outside it.
    """
    grouped: dict[str | None, list[Cell]] = {}
    for cell in cells:
        grouped.setdefault(policy.category(cell), []).append(cell)

    summaries: list[CategorySummary] = []
    for category, group in grouped.items():
        weight = category_weights.get(category, 1.0) if category is not None else 1.0
        defaulted = sum(1 for cell in group if policy.defaulted(cell))
        total = sum(policy.weight(cell) * cell.pairs for cell in group)
        if total > 0.0:
            level = sum(policy.weight(cell) * cell.passed for cell in group) / total
            baseline = sum(policy.weight(cell) * cell.baseline_passed for cell in group) / total
            score, baseline_score, delta = (
                100.0 * level,
                100.0 * baseline,
                100.0 * (level - baseline),
            )
        else:
            score, baseline_score, delta = None, None, None
        improved = sum(cell.improved for cell in group)
        regressed = sum(cell.regressed for cell in group)
        pooled = switch_evidence(improved, regressed, a0=a0, b0=b0)
        p_improved = pooled.p_improved if pooled.discordant > 0 else None
        group_flagged = [
            f"{cell.stratum}/{cell.outcome}"
            for cell in group
            if verdicts[cell.key].verdict == "regressing"
        ]
        draws = category_delta_draws.get(category, [])
        delta_p_improved = sum(1 for value in draws if value > 0.0) / len(draws) if draws else None
        status: Status = (
            delta_status(draws, rope=rope, alpha=alpha, flagged=bool(group_flagged))
            if draws
            else "insufficient"
        )
        summaries.append(
            CategorySummary(
                category=category,
                weight=weight,
                defaulted=defaulted,
                views=len({cell.stratum for cell in group}),
                cells=len(group),
                pairs=sum(cell.pairs for cell in group),
                score=score,
                baseline_score=baseline_score,
                delta_pp=delta,
                discordant=improved + regressed,
                p_improved=p_improved,
                status=status,
                delta_p_improved=delta_p_improved,
            )
        )
    summaries.sort(key=lambda summary: (-summary.weight, summary.category or ""))
    return summaries


def _guardrail(
    heterogeneity: Homogeneity | None,
    flagged: list[str],
    pooled_p: float | None,
    alpha: float,
    *,
    cells: int,
    discordant: int,
) -> Guardrail:
    """Turn the predictive check and the per-cell flags into one verdict."""
    if cells < 2 or discordant < 2:
        detail = f" (cells={cells}, discordant={discordant})"
        if flagged:
            detail = f" — flagged: {', '.join(flagged)}" + detail
        return Guardrail(
            status="insufficient",
            flagged=flagged,
            reason="not enough strata to test homogeneity" + detail,
        )
    if heterogeneity is None:
        return Guardrail(
            status="insufficient", flagged=flagged, reason="homogeneity check unavailable"
        )
    ppc = heterogeneity.ppc
    observed = heterogeneity.observed
    expected = heterogeneity.expected
    draws = heterogeneity.draws
    if ppc < alpha:
        return Guardrail(
            status="heterogeneous",
            ppc=ppc,
            dispersion=observed,
            expected_dispersion=expected,
            draws=draws,
            flagged=flagged,
            reason=(
                "stratum switch rates are not homogeneous"
                f" (posterior predictive tail={ppc:.3f}, alpha={alpha})"
            ),
        )
    if flagged and pooled_p is not None and pooled_p > alpha:
        return Guardrail(
            status="heterogeneous",
            ppc=ppc,
            dispersion=observed,
            expected_dispersion=expected,
            draws=draws,
            flagged=flagged,
            reason=(
                "a view regressed credibly while the pooled trend is not a regression"
                f" (flagged={', '.join(flagged)})"
            ),
        )
    return Guardrail(
        status="ok",
        ppc=ppc,
        dispersion=observed,
        expected_dispersion=expected,
        draws=draws,
        flagged=flagged,
        reason="stratum effects are consistent with one pooled switch rate",
    )
