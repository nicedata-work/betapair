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

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from pydfba.hierarchy import (
    DRAWS,
    PPC_DRAWS,
    Cell,
    Homogeneity,
    cell_posteriors,
    fit,
    homogeneity_ppc,
    pooled_counts,
)
from pydfba.mcnemar import McNemarResult, mcnemar

#: Bumped when the JSON shape changes — reports are persisted per refresh.
REPORT_VERSION = 1

#: The tier an untagged view falls back to **when tiers are in use at all**.
#: The vocabulary is fixed: ``tier-1`` critical, ``tier-2`` dependent,
#: ``tier-3`` auxiliary — so a project that tags only its critical views does
#: not accidentally score the rest as unimportant. When *no* view in the
#: comparison carries a tier, nothing is defaulted and every cell weighs the
#: same (the uniform case).
DEFAULT_CATEGORY = "tier-2"


class McNemarSummary(BaseModel):
    """DFBA-style posterior summary of a paired comparison."""

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
    p_value: float | None = None
    """Posterior predictive p-value; small means the strata disagree."""
    dispersion: float | None = None
    expected_dispersion: float | None = None
    draws: int = 0
    flagged: list[str] = Field(default_factory=list)
    """Cells with a credible regression, as ``stratum/outcome``.

    A cell is listed when its own exact McNemar test rejects at a Holm-corrected
    family-wise level (see :func:`cell_verdicts`), so a wide layer does not
    manufacture a regression by chance.
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
    status: Literal["improving", "stable", "regressing", "insufficient"] = "insufficient"


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
    logit_shift: float
    """Shrunken log-odds shift of this cell against the pooled layer."""
    verdict: Literal["improving", "stable", "regressing"]
    share: float | None = None
    """Fraction of the layer's net change attributable to this cell."""


class HealthReport(BaseModel):
    """The assembled paired-comparison report, ready to persist or serve as JSON."""

    version: int = REPORT_VERSION
    status: Literal["improving", "stable", "regressing", "insufficient"]
    score: float | None = None
    """Paired candidate-epoch pass rate, 0-100."""
    baseline_score: float | None = None
    delta_pp: float | None = None
    delta_pp_low: float | None = None
    delta_pp_high: float | None = None
    p_improved: float | None = None
    pairs: int = 0
    pending: int = 0
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
    categories: list[CategorySummary] = Field(default_factory=list)
    guardrail: Guardrail = Field(
        default_factory=lambda: Guardrail(status="insufficient", reason="not computed")
    )
    contributions: list[CellContribution] = Field(default_factory=list)


@dataclass(frozen=True)
class CellVerdict:
    """A cell's exact McNemar evidence and the verdict drawn from it."""

    p_improved: float
    verdict: Literal["improving", "stable", "regressing"]


def cell_verdicts(
    cells: list[Cell],
    *,
    a0: float = 1.0,
    b0: float = 1.0,
    prob_interval: float = 0.95,
    regression_prob: float = 0.95,
) -> dict[tuple[str, str], CellVerdict]:
    """Exact per-cell McNemar result and its verdict.

    ``improving`` and ``stable`` come from the cell's own exact test at
    ``regression_prob``. ``regressing`` is the safety-critical direction and is
    **Holm-corrected across the comparison** at family-wise level ``1 -
    regression_prob``: without that control a layer of ~60 drifting cells would
    produce a spurious "regressing" verdict in ~97% of reports. A single cell
    is its own family, so its threshold is unchanged. The reported
    ``p_improved`` is always the cell's own exact value — only the decision is
    corrected, so a cell's evidence is never borrowed from its neighbours.
    """
    alpha = 1.0 - regression_prob
    tests: dict[tuple[str, str], McNemarResult] = {
        cell.key: mcnemar(cell.improved, cell.regressed, a0=a0, b0=b0, prob_interval=prob_interval)
        for cell in cells
    }
    ordered = sorted(
        (key for key, test in tests.items() if test.discordant > 0),
        key=lambda key: tests[key].p_improved,
    )
    regressing: set[tuple[str, str]] = set()
    family = len(ordered)
    for rank, key in enumerate(ordered):
        if tests[key].p_improved <= alpha / (family - rank):
            regressing.add(key)
        else:
            break
    verdicts: dict[tuple[str, str], CellVerdict] = {}
    for key, test in tests.items():
        if test.discordant and test.p_improved >= regression_prob:
            verdict: Literal["improving", "stable", "regressing"] = "improving"
        elif key in regressing:
            verdict = "regressing"
        else:
            verdict = "stable"
        verdicts[key] = CellVerdict(p_improved=test.p_improved, verdict=verdict)
    return verdicts


def layer_status(
    pooled_p: float | None,
    flagged: list[str] | set[str] | tuple[str, ...],
    alpha: float,
) -> Literal["improving", "stable", "regressing", "insufficient"]:
    """The layer's verdict from its pooled test plus any credible regression.

    One function so the report, the tier roll-up and the web timeline cannot
    drift apart: a credible regression anywhere in ``flagged`` is a regression
    for the layer even when the pooled trend says otherwise.

    @note: This is the invariant that keeps one bad view from being averaged
        away by improvements elsewhere in the layer.
    """
    if pooled_p is None:
        return "insufficient"
    if pooled_p <= alpha or flagged:
        return "regressing"
    if pooled_p >= 1.0 - alpha:
        return "improving"
    return "stable"


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
) -> Weighting:
    """Resolve effective weights for ``cells`` (see :class:`Weighting`).

    Tiering applies only when at least one cell declares a category; an
    untagged cell then takes ``default_category``. A set with no tier anywhere
    stays uniform: every cell weighs 1.0, and nothing is marked defaulted.
    """
    weight_of = dict(weights or {})
    tier_of = dict(category_weights or {})
    for name, value in (*weight_of.items(), *tier_of.items()):
        if value < 0.0:
            raise ValueError(f"weight for {name!r} must be >= 0, got {value}")
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
    prob_interval: float = 0.95,
    draws: int = DRAWS,
    ppc_draws: int = PPC_DRAWS,
    seed: int = 0,
    alpha: float = 0.05,
    regression_prob: float = 0.95,
    pending: int = 0,
    baseline_epoch: str | None = None,
    candidate_epoch: str | None = None,
) -> HealthReport:
    """Assemble the health report from per-cell discordant counts.

    ``weights`` is per-outcome importance, ``category_weights`` the weight of
    the category a cell's stratum belongs to. Categories are applied only when
    at least one cell declares one: an uncategorised cell then falls back to
    ``default_category``, while a comparison with no category anywhere stays
    uniform (every cell weighs 1.0).
    ``alpha`` is the tail probability used for the pooled trend and the
    heterogeneity check. A cell's reported ``p_improved`` is its own exact
    McNemar test and the hierarchical posterior supplies the shrunken
    ``logit_shift`` and the delta interval. ``regression_prob`` is the
    posterior probability a cell must reach to be called improving
    (``>= regression_prob``); the regressing direction is the same exact test
    but Holm-corrected across cells (family-wise level ``1 - regression_prob``)
    so a wide layer does not manufacture a regression by chance. See
    :func:`cell_verdicts`.

    @note: The global delta interval is built from draws that share one
        ``(mu, kappa)`` replicate across all cells, so it carries the
        hierarchical correlation instead of pretending the cells are
        independent.
    @note: A cell's verdict comes from its own exact DFBA test, never from the
        borrowed-strength hierarchy estimate: pooling is for ranking and
        intervals, and letting the layer pull a small cell's verdict would be
        exactly the masking the guardrail exists to prevent. The hierarchy
        still shapes ``logit_shift`` and the delta interval.
    @return: The assembled :class:`HealthReport`.
    @raise ValueError: If ``alpha`` is outside (0, 0.5) or ``regression_prob``
        is outside (0.5, 1).
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if not 0.5 < regression_prob < 1.0:
        raise ValueError(f"regression_prob must be in (0.5, 1), got {regression_prob}")
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

    if pairs == 0 or not active:
        return HealthReport(
            status="insufficient",
            pending=pending,
            strata=strata,
            outcomes=outcomes,
            baseline_epoch=baseline_epoch,
            candidate_epoch=candidate_epoch,
            category_weights=dict(tier_of),
            default_category=default_category,
            guardrail=Guardrail(status="insufficient", reason="no paired items yet"),
        )

    weighted_pairs = sum(policy.weight(cell) * cell.pairs for cell in active)
    if weighted_pairs <= 0.0:
        return HealthReport(
            status="insufficient",
            pending=pending,
            strata=strata,
            outcomes=outcomes,
            baseline_epoch=baseline_epoch,
            candidate_epoch=candidate_epoch,
            category_weights=dict(tier_of),
            default_category=default_category,
            tiered=tiered,
            guardrail=Guardrail(status="insufficient", reason="all outcome weights are zero"),
        )

    level = sum(policy.weight(cell) * cell.passed for cell in active) / weighted_pairs
    baseline_level = (
        sum(policy.weight(cell) * cell.baseline_passed for cell in active) / weighted_pairs
    )
    observed_delta = level - baseline_level

    hierarchy_fit = fit(active, a0=a0, b0=b0)
    posteriors = cell_posteriors(hierarchy_fit, active, draws=draws, seed=seed)

    delta_draws: list[float] = []
    per_cell_delta: dict[tuple[str, str], list[float]] = {cell.key: [] for cell in posteriors.cells}
    for row in posteriors.phi_draws:
        total = 0.0
        for position, cell in enumerate(posteriors.cells):
            weight = policy.weight(cell)
            cell_delta = (2.0 * row[position] - 1.0) * cell.discordant / cell.pairs
            total += weight * cell.pairs * cell_delta
            per_cell_delta[cell.key].append(cell_delta)
        delta_draws.append(total / weighted_pairs)
    delta_draws.sort()

    n_improved, n_regressed = pooled_counts(active)
    pooled = mcnemar(n_improved, n_regressed, a0=a0, b0=b0, prob_interval=prob_interval)
    pooled_p = pooled.p_improved if pooled.discordant > 0 else None

    verdicts = cell_verdicts(
        active,
        a0=a0,
        b0=b0,
        prob_interval=prob_interval,
        regression_prob=regression_prob,
    )

    contributions: list[CellContribution] = []
    flagged: list[str] = []
    net_change = sum(policy.weight(cell) * (cell.improved - cell.regressed) for cell in active)
    for cell in active:
        summary = posteriors.summaries[cell.key]
        weight = policy.weight(cell)
        observed_cell_delta = 100.0 * (cell.improved - cell.regressed) / cell.pairs
        interval = sorted(per_cell_delta[cell.key])
        cell_test = verdicts[cell.key]
        verdict = cell_test.verdict
        if verdict == "regressing":
            flagged.append(f"{cell.stratum}/{cell.outcome}")
        share = None if net_change == 0 else weight * (cell.improved - cell.regressed) / net_change
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
                logit_shift=summary.logit_shift,
                verdict=verdict,
                share=share,
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

    categories = _categories(
        active,
        policy,
        tier_of,
        a0=a0,
        b0=b0,
        prob_interval=prob_interval,
        alpha=alpha,
    )

    heterogeneity = homogeneity_ppc(active, a0=a0, b0=b0, draws=ppc_draws, seed=seed)
    guardrail = _guardrail(
        heterogeneity, flagged, pooled_p, alpha, cells=len(active), discordant=pooled.discordant
    )

    status = layer_status(pooled_p, flagged, alpha)

    return HealthReport(
        status=status,
        score=100.0 * level,
        baseline_score=100.0 * baseline_level,
        delta_pp=100.0 * observed_delta,
        delta_pp_low=100.0 * _quantile(delta_draws, 0.025),
        delta_pp_high=100.0 * _quantile(delta_draws, 0.975),
        p_improved=pooled_p,
        pairs=pairs,
        pending=pending,
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


def _categories(
    cells: list[Cell],
    policy: Weighting,
    category_weights: dict[str, float],
    *,
    a0: float,
    b0: float,
    prob_interval: float,
    alpha: float,
) -> list[CategorySummary]:
    """Per-tier roll-up of the same cells the layer score is built from.

    Numbers are comparable across tiers because the tier's own weight is a
    constant inside the group and cancels. ``status`` comes from the tier's
    pooled McNemar test; per-cell escalation (``contributions``) names the
    views when a tier is heterogeneous.

    @note: Per-cell escalation stays at the layer level, so a tier's verdict is
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
        pooled = mcnemar(improved, regressed, a0=a0, b0=b0, prob_interval=prob_interval)
        p_improved = pooled.p_improved if pooled.discordant > 0 else None
        status = layer_status(p_improved, [], alpha)
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
    p_value = heterogeneity.p_value
    observed = heterogeneity.observed
    expected = heterogeneity.expected
    draws = heterogeneity.draws
    if p_value < alpha:
        return Guardrail(
            status="heterogeneous",
            p_value=p_value,
            dispersion=observed,
            expected_dispersion=expected,
            draws=draws,
            flagged=flagged,
            reason=(
                "stratum switch rates are not homogeneous"
                f" (posterior predictive p={p_value:.3f}, alpha={alpha})"
            ),
        )
    if flagged and pooled_p is not None and pooled_p > alpha:
        return Guardrail(
            status="heterogeneous",
            p_value=p_value,
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
        p_value=p_value,
        dispersion=observed,
        expected_dispersion=expected,
        draws=draws,
        flagged=flagged,
        reason="stratum effects are consistent with one pooled switch rate",
    )
