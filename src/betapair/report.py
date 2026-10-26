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

A stratum may also carry a ``group`` — the shared upstream dependency it
belongs to (source table, domain, metric family). Unlike a category it carries
no weight and gates nothing: it is a label the report passes through per cell
so views that fail *together* can be read as one incident rather than many
independent regressions. Nothing in this module's numbers depends on it.

Cells with no paired items are excluded from the score so a half-finished
stream never looks like a regression — the tracker reports those as
``pending``.

The headline ``score`` carries its own interval (``score_low``/``score_high``):
the equal-tail credible bounds of the same ``Beta(1 + k, 1 + n - k)`` posterior
whose mode *is* ``score``, taken over the **weighted effective** pairs so the
interval describes exactly the micro-average it belongs to (see
``series.ScorePoint.interval`` for the unweighted per-epoch form).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from betapair.beta import beta_ppf
from betapair.gate import (
    CellVerdict,
    RegressionModel,
    _admitted,
    _group_weight,
    cell_verdicts,
    delta_status,
    regression_model,
    verdicts_from_model,
)
from betapair.hierarchy import (
    DRAWS,
    PPC_DRAWS,
    Cell,
    CellPosteriors,
    HierarchyFit,
    Homogeneity,
    cell_posteriors,
    fit,
    homogeneity_ppc,
    pooled_counts,
)
from betapair.irt import (
    DEFAULT_SIGMA,
    ItemDrift,
    ItemKey,
    ItemResponse,
    ReferenceBank,
    bank_digest,
    equated_rates,
    fit_rasch,
    sigmoid,
)
from betapair.mcnemar import McNemarResult, mcnemar, switch_evidence
from betapair.models import (
    DEFAULT_CATEGORY,
    REPORT_VERSION,
    Calibration,
    CategorySummary,
    CellContribution,
    CellCoverage,
    Coverage,
    GroupSummary,
    Guardrail,
    HealthReport,
    HierarchySummary,
    IrTSummary,
    McNemarSummary,
    ReferenceBankSummary,
    RetrospectiveWindow,
    Status,
    UnpairedCell,
    Weighting,
    validate_decision_config,
)
from betapair.series import ScorePoint, retrospective, score_band

__all__ = [
    "COVERAGE_MARGIN",
    "COVERAGE_MAX",
    "COVERAGE_MIN",
    "COVERAGE_MIN_POINTS",
    "DEFAULT_CATEGORY",
    "PI_MIN_POINTS",
    "PI_PRIOR_MAX",
    "PI_PRIOR_MIN",
    "PI_PRIOR_STRENGTH",
    "REPORT_VERSION",
    "ROPE_BAND_FACTOR",
    "ROPE_MIN_POINTS",
    "Calibration",
    "CategorySummary",
    "CellContribution",
    "CellCoverage",
    "CellVerdict",
    "Coverage",
    "GroupSummary",
    "Guardrail",
    "HealthReport",
    "HierarchySummary",
    "IrTSummary",
    "McNemarSummary",
    "ReferenceBankSummary",
    "RegressionModel",
    "ReportOptions",
    "RetrospectiveWindow",
    "Status",
    "UnpairedCell",
    "Weighting",
    "build_report",
    "cell_verdicts",
    "delta_status",
    "parse_category_weights",
    "regression_model",
    "resolve_calibration",
    "verdicts_from_model",
    "weighting",
]

#: Multiplier on the measured between-epoch dispersion when the ROPE is
#: calibrated from the metric's own history. One measured sd means "a move
#: smaller than the metric's genuine movement is not a change".
ROPE_BAND_FACTOR = 1.0

#: Scored points the history needs before its band is trusted to size the
#: ROPE. With fewer, the hand-set default stands (the dispersion estimate is
#: itself too noisy to decide a dead zone).
ROPE_MIN_POINTS = 3

#: Total strength (``a + b``) the default ``pi_prior = (1, 1)`` carries. A
#: calibrated prior keeps this strength and moves only its mean onto the base
#: rate the history measured, so it stays as weakly informative as before.
PI_PRIOR_STRENGTH = 2.0

#: Bounds on the calibrated ``pi_prior`` mean, away from 0 and 1: a prior that
#: says "nothing ever regresses" (or "everything does") would be unrecoverable
#: from data.
PI_PRIOR_MIN = 0.01
PI_PRIOR_MAX = 0.99

#: History points carrying a ``pi_mean`` the prior needs before it is taken
#: from them rather than the default. One previous epoch is enough;
#: subsequent epochs then accumulate.
PI_MIN_POINTS = 1

#: Completed epochs the history needs before its unresolved shares are
#: trusted to size the coverage gate. This is a cold-start guard: on the first
#: runs nearly everything is unresolved, and a threshold derived from that
#: would gate every early report. Below this count the default stands.
COVERAGE_MIN_POINTS = 3

#: How far above the history's mean unresolved share the gate tolerates before
#: it fires — room for a normal, noisy epoch without letting a stalled
#: pipeline through.
COVERAGE_MARGIN = 0.05

#: The derived coverage threshold is kept inside these bounds: at least a
#: sliver of unresolved data is always tolerated, and more than half
#: unresolved can never be waved through.
COVERAGE_MIN = 0.02
COVERAGE_MAX = 0.50


def _quantile(sorted_values: list[float], prob: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty series")
    index = min(len(sorted_values) - 1, max(0, int(prob * len(sorted_values))))
    return sorted_values[index]


def weighting(
    cells: Sequence[Cell | UnpairedCell],
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

    Args:
        cells: The paired and/or cold-start cells to weight.
        weights: Per-outcome importance; an unnamed outcome weighs 1.0.
        category_weights: Weight of each category/tier; a missing tier weighs 1.0.
        default_category: Tier an untagged cell falls back to when tiers are in use.
        tiered: Override the auto-detection. The held-level overlay passes
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
    cell_groups: dict[tuple[str, str], str | None] = {}
    for cell in cells:
        category = cell.category
        if category is None and tiered:
            category = default_category
        tier = tier_of.get(category, 1.0) if category is not None else 1.0
        cell_weights[cell.key] = weight_of.get(cell.outcome, 1.0) * tier
        cell_categories[cell.key] = category
        cell_groups[cell.key] = cell.group
    return Weighting(
        tiered=tiered,
        default_category=default_category,
        weights=cell_weights,
        categories=cell_categories,
        groups=cell_groups,
    )


def _score_interval(
    successes: float, pairs: float, level: float, prob_interval: float
) -> tuple[float, float]:
    """Equal-tail credible bounds on a weighted pass rate, 0-100.

    The posterior is ``Beta(1 + k, 1 + n - k)`` over the weighted effective
    pairs ``n`` and successes ``k``, whose mode is ``level`` — the headline
    micro-average. A rate with no effective pairs has no width and collapses
    onto ``level``, mirroring ``series.ScorePoint.interval``.
    """
    if pairs <= 0.0:
        return 100.0 * level, 100.0 * level
    tail = 0.5 * (1.0 - prob_interval)
    return (
        100.0 * beta_ppf(tail, 1.0 + successes, 1.0 + pairs - successes),
        100.0 * beta_ppf(1.0 - tail, 1.0 + successes, 1.0 + pairs - successes),
    )


def resolve_calibration(
    history: Sequence[ScorePoint] | None,
    *,
    calibrate: bool,
    rope_pp: float | None,
    pi_prior: tuple[float, float],
    max_unresolved_share: float,
) -> Calibration:
    """Resolve the report's thresholds, deriving any the history supports.

    The thresholds are the caller's defaults until there is enough history to
    measure them from, so passing no history — or leaving ``calibrate``
    False — leaves every number exactly as it was. Each derivation lands
    behind this one function: the ROPE from the score band's between-epoch
    dispersion, the regression prior from the previous epoch's base rate, and
    the coverage gate from the observed pending share. Resolving in one place
    is what lets the report publish exactly which value it used.

    Returns:
        The resolved :class:`Calibration`.

    Notes:
        The ROPE derivation (B2) needs :data:`ROPE_MIN_POINTS` scored points
            before the band is trusted, and keeps ``rope_pp`` when the band
            measures no between-epoch movement at all — a series whose spread is
            pure sampling noise has no dead zone to size, so the hand-set value
            still stands. ``rope_pp`` is the fallback whenever calibration cannot
            size it. The import is local because :mod:`betapair.series` imports
            this module.

        The prior derivation (B3) averages the ``pi_mean`` of the history
            points that carry one — the previous epochs' measured base rate of
            true regressions — and keeps the default ``pi_prior`` strength while
            moving its mean. The mean is clipped to
            :data:`PI_PRIOR_MIN` and :data:`PI_PRIOR_MAX` so the prior cannot
            collapse onto 0 or 1.

        The coverage derivation (B4) averages the epochs' unresolved shares
            and adds :data:`COVERAGE_MARGIN`, clamped to
            :data:`COVERAGE_MIN`/:data:`COVERAGE_MAX`. It waits for
            :data:`COVERAGE_MIN_POINTS` epochs on purpose: the first runs are
            mostly unresolved, and sizing the gate from them would gate every
            early report (the cold-start guard).
    """
    points = 0 if history is None else len(history)
    derived: list[str] = []
    between_sd: float | None = None
    resolved_rope = 1.0 if rope_pp is None else rope_pp
    if calibrate and history is not None and points >= ROPE_MIN_POINTS:
        band = score_band(history)
        between_sd = band.between_sd
        if between_sd > 0.0:
            resolved_rope = ROPE_BAND_FACTOR * between_sd
            derived.append("rope_pp")
    observed_pi_mean: float | None = None
    observed = [point.pi_mean for point in history or [] if point.pi_mean is not None]
    if calibrate and len(observed) >= PI_MIN_POINTS:
        observed_pi_mean = sum(observed) / len(observed)
        clipped = min(PI_PRIOR_MAX, max(PI_PRIOR_MIN, observed_pi_mean))
        pi_prior = (clipped * PI_PRIOR_STRENGTH, (1.0 - clipped) * PI_PRIOR_STRENGTH)
        derived.append("pi_prior")
    observed_unresolved_share: float | None = None
    shares = [
        point.unresolved_share for point in history or [] if point.unresolved_share is not None
    ]
    if calibrate and len(shares) >= COVERAGE_MIN_POINTS:
        observed_unresolved_share = sum(shares) / len(shares)
        max_unresolved_share = min(
            COVERAGE_MAX, max(COVERAGE_MIN, observed_unresolved_share + COVERAGE_MARGIN)
        )
        derived.append("max_unresolved_share")
    return Calibration(
        rope_pp=resolved_rope,
        pi_prior=pi_prior,
        max_unresolved_share=max_unresolved_share,
        enabled=calibrate,
        history_points=points,
        between_sd=between_sd,
        observed_pi_mean=observed_pi_mean,
        observed_unresolved_share=observed_unresolved_share,
        derived=tuple(derived),
    )


@dataclass(frozen=True)
class _Difficulty:
    """The resolved difficulty adjustment: equated rates plus the level."""

    summary: IrTSummary
    rates: dict[tuple[str, str], float]
    score: float
    low: float
    high: float


def _difficulty_adjustment(
    cells: list[Cell],
    responses: Sequence[ItemResponse],
    policy: Weighting,
    weighted_pairs: float,
    *,
    reference_items: Sequence[ItemKey] | None,
    reference_bank: ReferenceBank | None,
    sigma_theta: float,
    sigma_d: float,
    draws: int,
    seed: int,
    prob_interval: float,
) -> _Difficulty:
    """Fit one shared-ability Rasch model, equate every cell, aggregate.

    A **single** fit spans all outcomes: items stay outcome-specific
    (``(outcome, item)``), but each view has one ability informed by every
    outcome it ran. Equating then replaces a cell's own-bank pass rate with the
    rate it would score on that outcome's reference bank,
    ``mean_i sigmoid(theta_v + d_{o,i})``, and the adjusted level is the
    **same** weighted micro-average as ``score`` — so the two differ only
    through the rates. The reference bank defaults to each outcome's observed
    union; it can be pinned with raw ``reference_items`` (exact: a missing item
    is an error) or with a versioned
    :class:`~betapair.irt.ReferenceBank`, whose items are intersected with the
    fitted ones so a temporarily absent anchor degrades the width of the
    reference set rather than the whole report. The declared and used sizes are
    published on ``IrTSummary.reference`` either way.

    The interval comes from seeded posterior draws of the shared ``theta``
    (:meth:`RaschFit.sample_abilities`), each mapped to an equated rate and
    aggregated with the same weights, so ability uncertainty sizes the level —
    and because one draw set serves every outcome, a view's cells move together
    instead of being treated as independent fits.

    Raises:
        ValueError: If ``draws`` is not positive, no responses were given,
            a pinned bank names no item for a fitted outcome (or none of its items
            was observed), or a weighted cell has no response behind it (the
            adjusted average would then be undefined).
    """
    if draws < 1:
        raise ValueError(f"draws must be at least 1, got {draws}")
    if not responses:
        raise ValueError("adjust_difficulty requires responses")
    pinned: dict[str, list[ItemKey]] = {}
    if reference_items is not None:
        seen: set[ItemKey] = set()
        for key in reference_items:
            if key in seen:
                raise ValueError(f"reference_items repeats item {key!r}")
            seen.add(key)
            pinned.setdefault(key[0], []).append(key)
    by_outcome: dict[str, list[ItemResponse]] = {}
    for response in responses:
        by_outcome.setdefault(response.outcome, []).append(response)

    fit = fit_rasch(responses, sigma_theta=sigma_theta, sigma_d=sigma_d)
    draw_totals = [0.0] * draws
    declared_items = 0
    resolved_keys: list[ItemKey] = []
    reference_difficulties: dict[str, list[float]] = {}
    for outcome in sorted(by_outcome):
        if reference_bank is not None:
            declared = [key for key in reference_bank.items if key[0] == outcome]
            if not declared:
                raise ValueError(
                    f"reference bank {reference_bank.label} has no items for outcome {outcome!r}"
                )
            keys = [key for key in declared if key in fit.difficulties]
            if not keys:
                raise ValueError(
                    f"no item of reference bank {reference_bank.label} was observed for "
                    f"outcome {outcome!r}"
                )
        elif reference_items is not None:
            declared = list(pinned.get(outcome, ()))
            if not declared:
                raise ValueError(f"reference_items pins no items for outcome {outcome!r}")
            keys = declared
        else:
            declared = [key for key in fit.difficulties if key[0] == outcome]
            keys = declared
        unknown = [key for key in keys if key not in fit.difficulties]
        if unknown:
            raise ValueError(f"reference item not in the fit: {unknown[0]}")
        reference_difficulties[outcome] = [fit.difficulties[key] for key in keys]
        declared_items += len(declared)
        resolved_keys.extend(keys)

    rates = equated_rates(fit, reference_items=resolved_keys)
    sampled = fit.sample_abilities(draws=draws, seed=seed)
    for cell in cells:
        if cell.key not in rates:
            continue
        mass = policy.weight(cell) * cell.pairs
        if mass == 0.0:
            continue
        difficulties = reference_difficulties[cell.outcome]
        for position, ability in enumerate(sampled[cell.stratum]):
            equated = sum(sigmoid(ability + value) for value in difficulties) / len(difficulties)
            draw_totals[position] += mass * equated

    missing = [cell.key for cell in cells if policy.weight(cell) > 0.0 and cell.key not in rates]
    if missing:
        raise ValueError(f"responses do not cover cell {missing[0]!r}")

    score = (
        100.0
        * sum(
            policy.weight(cell) * cell.pairs * rates[cell.key]
            for cell in cells
            if policy.weight(cell) > 0.0
        )
        / weighted_pairs
    )
    tail = 0.5 * (1.0 - prob_interval)
    levels = sorted(100.0 * total / weighted_pairs for total in draw_totals)
    summary = IrTSummary(
        outcomes=list(fit.outcomes),
        sigma_theta=sigma_theta,
        sigma_d=sigma_d,
        connected=fit.diagnostics.connected,
        components=fit.diagnostics.components,
        weakest_link=fit.diagnostics.weakest_link,
        shared_items=fit.diagnostics.shared_items,
        prior_only=fit.diagnostics.prior_only,
        prior_dominated=len(fit.diagnostics.prior_dominated),
        separated_items=fit.diagnostics.separated_items,
        observed_pairs=len(responses),
        items=len(fit.difficulties),
        reference=ReferenceBankSummary(
            name=None if reference_bank is None else reference_bank.name,
            version=None if reference_bank is None else reference_bank.version,
            digest=bank_digest(resolved_keys),
            items=declared_items,
            available=len(resolved_keys),
        ),
    )
    return _Difficulty(
        summary=summary,
        rates=rates,
        score=score,
        low=_quantile(levels, tail),
        high=_quantile(levels, 1.0 - tail),
    )


@dataclass(frozen=True)
class ReportOptions:
    """The configurable inputs to :func:`build_report`, validated on construction.

    A frozen bundle so the report assembler and :meth:`HealthTracker.report`
    share one configuration object instead of threading three dozen keyword
    arguments through every frame. The field semantics are exactly the keyword
    arguments the assembler has always accepted; validating in
    :meth:`__post_init__` means a bad configuration fails where it is set
    rather than somewhere deep in a computation.
    """

    weights: dict[str, float] | None = None
    category_weights: dict[str, float] | None = None
    default_category: str | None = DEFAULT_CATEGORY
    a0: float = 1.0
    b0: float = 1.0
    a_d: float = 1.0
    b_d: float = 1.0
    prob_interval: float = 0.95
    draws: int = DRAWS
    ppc_draws: int = PPC_DRAWS
    seed: int = 0
    alpha: float = 0.05
    rope_pp: float | None = None
    pi_prior: tuple[float, float] = (1.0, 1.0)
    pi_points: int = 199
    pending: int = 0
    failed: int = 0
    retired: int = 0
    coverage: list[CellCoverage] | None = None
    held_counts: Mapping[tuple[str, str], tuple[int, int]] | None = None
    max_unresolved_share: float = 0.25
    baseline_epoch: str | None = None
    candidate_epoch: str | None = None
    history: Sequence[ScorePoint] | None = None
    calibrate: bool = False
    retrospective_window: int | None = None
    unpaired_cells: Sequence[UnpairedCell] | None = None
    unresolved_share: float | None = None
    drift: Sequence[ItemDrift] | None = None
    responses: Sequence[ItemResponse] | None = None
    adjust_difficulty: bool = False
    reference_items: Sequence[ItemKey] | None = None
    reference_bank: ReferenceBank | None = None
    sigma_theta: float = DEFAULT_SIGMA
    sigma_d: float = DEFAULT_SIGMA

    def __post_init__(self) -> None:
        validate_decision_config(
            alpha=self.alpha,
            rope_pp=self.rope_pp,
            max_unresolved_share=self.max_unresolved_share,
            unresolved_share=self.unresolved_share,
            prob_interval=self.prob_interval,
            draws=self.draws,
            ppc_draws=self.ppc_draws,
            pi_points=self.pi_points,
            a0=self.a0,
            b0=self.b0,
            a_d=self.a_d,
            b_d=self.b_d,
            pi_prior=self.pi_prior,
            sigma_theta=self.sigma_theta,
            sigma_d=self.sigma_d,
        )
        if self.reference_bank is not None and self.reference_items is not None:
            raise ValueError("pass reference_bank or reference_items, not both")
        if self.retrospective_window is not None and self.retrospective_window < 1:
            raise ValueError(
                f"retrospective_window must be at least 1, got {self.retrospective_window}"
            )


@dataclass(frozen=True)
class _Inputs:
    """Validated inputs and derived label/coverage state shared by the steps."""

    options: ReportOptions
    calibration: Calibration
    tier_of: dict[str, float]
    active: list[Cell]
    unpaired_active: list[UnpairedCell]
    policy: Weighting
    tiered: bool
    outcomes: list[str]
    strata: int
    pairs: int
    weighted_pairs: float
    unpaired_total: int
    score_unpaired: float | None
    score_unpaired_low: float | None
    score_unpaired_high: float | None
    unresolved_share: float | None
    cov: Coverage


def _insufficient_report(inputs: _Inputs, reason: str, *, tiered: bool = False) -> HealthReport:
    """The single early-return shape for a report with no paired evidence.

    Both cold-start exits (no active cells, and every outcome weight zero)
    publish the same object. The no-cells case keeps ``tiered`` at its default
    (``False``) to match the historical behaviour; only the all-weights-zero
    exit passes the resolved policy.
    """
    return HealthReport(
        status="insufficient",
        tiered=tiered,
        pending=inputs.options.pending,
        failed=inputs.options.failed,
        retired=inputs.options.retired,
        strata=inputs.strata,
        outcomes=inputs.outcomes,
        baseline_epoch=inputs.options.baseline_epoch,
        candidate_epoch=inputs.options.candidate_epoch,
        category_weights=dict(inputs.tier_of),
        default_category=inputs.options.default_category,
        coverage=inputs.cov,
        guardrail=Guardrail(status="insufficient", reason=reason),
        calibration=inputs.calibration,
        score_unpaired=inputs.score_unpaired,
        score_unpaired_low=inputs.score_unpaired_low,
        score_unpaired_high=inputs.score_unpaired_high,
        unpaired=inputs.unpaired_total,
        unresolved_share=inputs.unresolved_share,
    )


def _resolve_inputs(cells: list[Cell], options: ReportOptions) -> _Inputs:
    """Resolve calibration, weighting policy, coverage and the cold-start level.

    One policy covers the paired cells and the cold-start ones, so a new view's
    tier and outcome importance are resolved the same way as the established
    ones and the unpaired level is a true micro-average.
    """
    calibration = resolve_calibration(
        options.history,
        calibrate=options.calibrate,
        rope_pp=options.rope_pp,
        pi_prior=options.pi_prior,
        max_unresolved_share=options.max_unresolved_share,
    )
    tier_of = dict(options.category_weights or {})
    active = [cell for cell in cells if cell.pairs > 0]
    unpaired_active = [cell for cell in (options.unpaired_cells or []) if cell.resolved > 0]
    policy = weighting(
        [*active, *unpaired_active],
        weights=options.weights,
        category_weights=options.category_weights,
        default_category=options.default_category,
    )
    outcomes = sorted({cell.outcome for cell in active})
    strata = len({cell.stratum for cell in active})
    pairs = sum(cell.pairs for cell in active)
    weighted_pairs = sum(policy.weight(cell) * cell.pairs for cell in active)
    unpaired_total = sum(cell.resolved for cell in unpaired_active)
    score_unpaired: float | None = None
    score_unpaired_low: float | None = None
    score_unpaired_high: float | None = None
    unpaired_pairs = sum(policy.weight(cell) * cell.resolved for cell in unpaired_active)
    if unpaired_active and unpaired_pairs > 0.0:
        unpaired_successes = sum(policy.weight(cell) * cell.passed for cell in unpaired_active)
        unpaired_level = unpaired_successes / unpaired_pairs
        score_unpaired = 100.0 * unpaired_level
        score_unpaired_low, score_unpaired_high = _score_interval(
            unpaired_successes, unpaired_pairs, unpaired_level, options.prob_interval
        )
    unresolved_share = options.unresolved_share
    if unresolved_share is None and options.coverage:
        unresolved_rows = sum(cell.pending + cell.errored for cell in options.coverage)
        total_rows = sum(
            cell.paired + cell.pending + cell.errored + cell.retired for cell in options.coverage
        )
        if total_rows:
            unresolved_share = unresolved_rows / total_rows
    cov = _coverage(
        options.coverage,
        tier_of,
        pending=options.pending,
        failed=options.failed,
        pairs=pairs,
        max_unresolved_share=calibration.max_unresolved_share,
    )
    return _Inputs(
        options=options,
        calibration=calibration,
        tier_of=tier_of,
        active=active,
        unpaired_active=unpaired_active,
        policy=policy,
        tiered=policy.tiered,
        outcomes=outcomes,
        strata=strata,
        pairs=pairs,
        weighted_pairs=weighted_pairs,
        unpaired_total=unpaired_total,
        score_unpaired=score_unpaired,
        score_unpaired_low=score_unpaired_low,
        score_unpaired_high=score_unpaired_high,
        unresolved_share=unresolved_share,
        cov=cov,
    )


@dataclass(frozen=True)
class _Levels:
    """The score/baseline level, its interval, and the difficulty adjustment."""

    weighted_pairs: float
    level: float
    baseline_level: float
    score_low: float
    score_high: float
    observed_delta: float
    epsilon: float
    held: _Held
    difficulty: _Difficulty | None


def _score_levels(inputs: _Inputs) -> _Levels:
    """Compute the weighted level, the baseline, and the opt-in equated level."""
    options = inputs.options
    active = inputs.active
    policy = inputs.policy
    weighted_pairs = inputs.weighted_pairs
    weighted_successes = sum(policy.weight(cell) * cell.passed for cell in active)
    level = weighted_successes / weighted_pairs
    difficulty: _Difficulty | None = None
    if options.adjust_difficulty:
        difficulty = _difficulty_adjustment(
            active,
            options.responses or (),
            policy,
            weighted_pairs,
            reference_items=options.reference_items,
            reference_bank=options.reference_bank,
            sigma_theta=options.sigma_theta,
            sigma_d=options.sigma_d,
            draws=options.draws,
            seed=options.seed,
            prob_interval=options.prob_interval,
        )
    baseline_level = (
        sum(policy.weight(cell) * cell.baseline_passed for cell in active) / weighted_pairs
    )
    score_low, score_high = _score_interval(
        weighted_successes, weighted_pairs, level, options.prob_interval
    )
    observed_delta = level - baseline_level
    held = _held_overlay(
        active,
        options.held_counts,
        options.coverage,
        policy,
        score=100.0 * level,
        weights=options.weights,
        category_weights=options.category_weights,
        default_category=options.default_category,
    )
    return _Levels(
        weighted_pairs=weighted_pairs,
        level=level,
        baseline_level=baseline_level,
        score_low=score_low,
        score_high=score_high,
        observed_delta=observed_delta,
        epsilon=inputs.calibration.rope_pp / 100.0,
        held=held,
        difficulty=difficulty,
    )


@dataclass(frozen=True)
class _Hierarchy:
    """The hierarchical pooling fit and the weighted delta draws it produced."""

    fit: HierarchyFit
    posteriors: CellPosteriors
    delta_draws: list[float]
    per_cell_delta: dict[tuple[str, str], list[float]]
    delta_p_improved: float
    stat_cells: list[Cell]
    pooled: McNemarResult
    pooled_p: float | None


def _fit_hierarchy(inputs: _Inputs, levels: _Levels) -> _Hierarchy:
    """Fit the hierarchy, draw the weighted delta, and pool the exact test.

    The delta draws multiply the switch rate by a discordance-rate draw, so the
    interval and the ROPE are statements about the population, not
    reconstructions of the observed sample delta. Cells with weight 0 are muted
    and excluded from the pooled diagnostic.
    """
    options = inputs.options
    active = inputs.active
    policy = inputs.policy
    hierarchy_fit = fit(active, a0=options.a0, b0=options.b0)
    posteriors = cell_posteriors(
        hierarchy_fit,
        active,
        draws=options.draws,
        seed=options.seed,
        a_d=options.a_d,
        b_d=options.b_d,
    )
    delta_draws: list[float] = []
    per_cell_delta: dict[tuple[str, str], list[float]] = {cell.key: [] for cell in posteriors.cells}
    for row, discordance in zip(posteriors.phi_draws, posteriors.discordance_draws, strict=True):
        total = 0.0
        for position, cell in enumerate(posteriors.cells):
            weight = policy.weight(cell)
            cell_delta = (2.0 * row[position] - 1.0) * discordance[position]
            total += weight * cell.pairs * cell_delta
            per_cell_delta[cell.key].append(cell_delta)
        delta_draws.append(total / levels.weighted_pairs)
    delta_draws.sort()
    delta_p_improved = sum(1 for value in delta_draws if value > 0.0) / len(delta_draws)

    stat_cells = [cell for cell in active if policy.weight(cell) > 0.0]
    n_improved, n_regressed = pooled_counts(stat_cells)
    pooled = mcnemar(
        n_improved, n_regressed, a0=options.a0, b0=options.b0, prob_interval=options.prob_interval
    )
    pooled_p = pooled.p_improved if pooled.discordant > 0 else None
    return _Hierarchy(
        fit=hierarchy_fit,
        posteriors=posteriors,
        delta_draws=delta_draws,
        per_cell_delta=per_cell_delta,
        delta_p_improved=delta_p_improved,
        stat_cells=stat_cells,
        pooled=pooled,
        pooled_p=pooled_p,
    )


def _assemble(cells: list[Cell], options: ReportOptions) -> HealthReport:
    """Run the four assembly steps and publish the report.

    The steps are kept separate so a caller (and a test) can substitute one:
    :func:`_resolve_inputs` validates and resolves the policy,
    :func:`_score_levels` computes the level, :func:`_fit_hierarchy` fits the
    pooling and draws the delta, and this function turns the fitted pieces and
    the regression gate into the serialisable :class:`HealthReport`.
    """
    inputs = _resolve_inputs(cells, options)
    if inputs.pairs == 0 or not inputs.active:
        return _insufficient_report(inputs, "no paired items yet")
    if inputs.weighted_pairs <= 0.0:
        return _insufficient_report(inputs, "all outcome weights are zero", tiered=inputs.tiered)

    levels = _score_levels(inputs)

    policy = inputs.policy
    active = inputs.active
    calibration = inputs.calibration
    hierarchy = _fit_hierarchy(inputs, levels)

    model = regression_model(
        active,
        a0=options.a0,
        b0=options.b0,
        fdr=options.alpha,
        weights=policy.weights,
        pi_prior=calibration.pi_prior,
        pi_points=options.pi_points,
    )
    verdicts = verdicts_from_model(
        active,
        model,
        a0=options.a0,
        b0=options.b0,
        improving_prob=1.0 - options.alpha,
    )

    difficulty = levels.difficulty
    contributions: list[CellContribution] = []
    flagged: list[str] = []
    for cell in active:
        summary = hierarchy.posteriors.summaries[cell.key]
        weight = policy.weight(cell)
        observed_cell_delta = 100.0 * (cell.improved - cell.regressed) / cell.pairs
        interval = sorted(hierarchy.per_cell_delta[cell.key])
        cell_test = verdicts[cell.key]
        verdict = cell_test.verdict
        if verdict == "regressing":
            flagged.append(f"{cell.stratum}/{cell.outcome}")
        contribution = 100.0 * weight * (cell.improved - cell.regressed) / levels.weighted_pairs
        equated = None if difficulty is None else difficulty.rates.get(cell.key)
        contributions.append(
            CellContribution(
                stratum=cell.stratum,
                outcome=cell.outcome,
                category=policy.category(cell),
                group=policy.group(cell),
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
                equated_pass_rate=equated,
                difficulty_shift_pp=(
                    None if equated is None else 100.0 * (equated - cell.pass_rate)
                ),
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

    category_delta_draws = _category_delta_draws(
        active, policy, hierarchy.per_cell_delta, len(hierarchy.delta_draws)
    )
    categories = _categories(
        active,
        policy,
        inputs.tier_of,
        a0=options.a0,
        b0=options.b0,
        alpha=options.alpha,
        rope=levels.epsilon,
        category_delta_draws=category_delta_draws,
        verdicts=verdicts,
    )
    groups = _groups(active, policy, model, levels.weighted_pairs)

    heterogeneity = homogeneity_ppc(
        hierarchy.stat_cells,
        a0=options.a0,
        b0=options.b0,
        draws=options.ppc_draws,
        seed=options.seed,
    )
    guardrail = _guardrail(
        heterogeneity,
        flagged,
        hierarchy.pooled_p,
        options.alpha,
        cells=len(hierarchy.stat_cells),
        discordant=hierarchy.pooled.discordant,
    )

    layer_verdict = delta_status(
        hierarchy.delta_draws,
        rope=levels.epsilon,
        alpha=options.alpha,
        flagged=bool(flagged),
        heterogeneous=guardrail.status == "heterogeneous",
    )
    status: Status = (
        "insufficient"
        if not inputs.cov.complete and layer_verdict != "regressing"
        else layer_verdict
    )

    windowed = None
    if options.retrospective_window is not None:
        scored = list(options.history or ())
        if scored:
            windowed = retrospective(
                scored,
                window=options.retrospective_window,
                rope_pp=calibration.rope_pp,
                level=options.prob_interval,
            )

    pooled = hierarchy.pooled
    hierarchy_fit = hierarchy.fit
    delta_draws = hierarchy.delta_draws
    difficulty = levels.difficulty
    return HealthReport(
        status=status,
        score=100.0 * levels.level,
        score_low=levels.score_low,
        score_high=levels.score_high,
        baseline_score=100.0 * levels.baseline_level,
        delta_pp=100.0 * levels.observed_delta,
        delta_pp_low=100.0 * _quantile(delta_draws, 0.025),
        delta_pp_high=100.0 * _quantile(delta_draws, 0.975),
        delta_p_improved=hierarchy.delta_p_improved,
        p_no_regression=model.p_no_regression,
        bfdr=model.bfdr,
        pi_mean=model.pi_mean,
        rope_pp=calibration.rope_pp,
        p_improved=hierarchy.pooled_p,
        pairs=inputs.pairs,
        pending=options.pending,
        failed=options.failed,
        retired=options.retired,
        retired_held=levels.held.items,
        score_held=levels.held.score,
        composition_effect_pp=levels.held.effect_pp,
        coverage=inputs.cov,
        strata=inputs.strata,
        outcomes=inputs.outcomes,
        baseline_epoch=options.baseline_epoch,
        candidate_epoch=options.candidate_epoch,
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
        category_weights=dict(inputs.tier_of),
        default_category=options.default_category,
        tiered=inputs.tiered,
        categories=categories,
        groups=groups,
        retrospective=windowed,
        drift=sorted(set(options.drift or ())),
        guardrail=guardrail,
        contributions=contributions,
        calibration=calibration,
        score_unpaired=inputs.score_unpaired,
        score_unpaired_low=inputs.score_unpaired_low,
        score_unpaired_high=inputs.score_unpaired_high,
        unpaired=inputs.unpaired_total,
        unresolved_share=inputs.unresolved_share,
        score_adjusted=None if difficulty is None else difficulty.score,
        score_adjusted_low=None if difficulty is None else difficulty.low,
        score_adjusted_high=None if difficulty is None else difficulty.high,
        difficulty_adjustment_pp=(
            None if difficulty is None else difficulty.score - 100.0 * levels.level
        ),
        irt=None if difficulty is None else difficulty.summary,
    )


def build_report(
    cells: list[Cell],
    *,
    options: ReportOptions | None = None,
    **overrides: Any,
) -> HealthReport:
    """Assemble the health report from per-cell discordant counts.

    ``options`` is the validated :class:`ReportOptions` bundle; the individual
    keyword arguments are accepted as a convenience and forwarded into it, so
    existing callers keep working unchanged. The validation, calibration and
    four assembly steps live in :func:`_assemble` and the private steps it
    calls.

    ``weights`` is per-outcome importance, ``category_weights`` the weight of
    the category a cell's stratum belongs to. Categories are applied only when
    at least one cell declares one: an uncategorised cell then falls back to
    ``default_category``, while a comparison with no category anywhere stays
    uniform (every cell weighs 1.0). The category weight also multiplies the
    prior odds that a cell truly regressed in the spike-and-slab gate (see
    :func:`regression_model`), so a critical view is not drowned by a wide
    field of auxiliary cells.

    ``group`` is the other label a cell may carry: the shared upstream
    dependency its stratum belongs to. It is passed straight through to
    ``CellContribution.group`` and enters no number — no weight, no gate, no
    interval. It exists so a caller can gather the views that fail together
    into one incident; making the *gate* dependence-aware is a separate step.

    ``drift`` carries content-fingerprint mismatches from the tracker: item ids
    whose question text changed while the id stayed the same. It is reported,
    not gated — the paired comparison and the score are unchanged — and is
    sorted/de-duplicated so input order cannot change the report.

    ``alpha`` is the single posterior threshold behind every decision: the
    homogeneity check, the per-cell improving threshold (``>= 1 - alpha``),
    the weighted delta ROPE, and the posterior FDR target for the regression
    gate. ``pi_prior`` is the Beta hyperprior on the share of cells that truly
    regressed, gridded at ``pi_points``; with ``calibrate=True`` it is
    re-centred on the history's measured base rate when the history carries
    one (see :class:`Calibration`).

    ``rope_pp`` is the region of practical equivalence for the layer verdict,
    in percentage points: a net movement under this size counts as ``stable``
    rather than improving/regressing. The per-cell regression flag is
    unaffected — a credible regression in any view still forces ``regressing``.
    ``None`` (the default) means the hand-set ``1.0``. With ``calibrate=True``
    and enough history, the measured band sizes it instead; ``rope_pp`` is the
    fallback whenever history cannot.

    ``prob_interval`` is the coverage of every credible interval the report
    carries, including the headline ``score_low``/``score_high``.

    ``retrospective_window`` opts into a stabilised read of the most recent
    this-many ``history`` points: ``HealthReport.retrospective`` carries the
    pooled level, the half-to-half change and the windowed verdict. It is a
    coarser, level-only summary — the headline ``status`` is unchanged, and a
    credible per-cell regression is never hidden. ``None`` (the default) leaves
    the report exactly as before; an empty history yields ``None``.

    ``history`` is the metric's own scored series (e.g. a tracker's
    ``timeline``). With ``calibrate=True`` the report may size its decision
    thresholds from that history instead of the hand-set defaults — see
    :class:`Calibration` — and publishes the resolved values on
    ``HealthReport.calibration``. Calibration is opt-in and additive: with no
    history, or with ``calibrate`` left False, every number is bit-identical
    to a report built without the argument.

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

    ``unpaired_cells`` carries candidate-epoch results whose baseline has not
    run yet (a cold start, or a view introduced mid-stream). They are summed
    into ``score_unpaired`` — an absolute level only, with the same weighting
    policy — and never enter the paired score, the delta, the verdict or the
    gate. ``unresolved_share`` is the share of tracked rows still unresolved;
    when omitted it is derived from ``coverage``. It is what the next epoch
    calibrates its coverage threshold from.

    ``adjust_difficulty`` opts into the latent-difficulty adjustment of the
    *level*: ``responses`` (from ``HealthTracker.item_responses()``) are fitted
    one outcome at a time with :func:`betapair.irt.fit_rasch`, each view is
    equated onto a common reference bank, and ``score_adjusted`` is the same
    weighted micro-average as ``score`` over the equated rates. The raw score
    and the shift are always both present, and ``irt`` carries the fit's
    linking diagnostics and the resolved reference bank
    (``irt.reference``). The bank defaults to each outcome's observed union;
    pin it with a versioned :class:`betapair.irt.ReferenceBank` (whose items are
    intersected with the fitted ones, so a temporarily absent anchor narrows the
    reference set and is reported) or with raw ``reference_items`` (exact: a
    missing item is an error). ``reference_bank`` and ``reference_items`` are
    mutually exclusive — one object should own the scale. ``sigma_theta``
    and ``sigma_d`` are the fit's prior scales. Off by default: with it off —
    or with the equivalent tracker config unset — every number is bit-identical
    to a report built without the argument. The paired delta is unaffected;
    difficulty already cancels in McNemar's discordant pairs, so only the
    headline level is adjusted.

    Returns:
        The assembled :class:`HealthReport`.

    Raises:
        ValueError: If ``alpha`` is outside (0, 0.5), ``rope_pp`` is
            negative, ``max_unresolved_share`` is outside [0, 1], or both a
            reference bank and raw reference items are pinned.

    Notes:
        The layer verdict is derived from the *weighted* hierarchical delta
            draws — the same quantity as the headline ``delta_pp`` — not from the
            unweighted pooled test. ``p_improved`` is kept as an unweighted pooled
            diagnostic; ``delta_p_improved`` is the probability behind the verdict.

        The delta draws multiply the switch rate by a discordance-rate draw,
            so the interval and the ROPE are statements about the population, not
            reconstructions of the observed sample delta.

        A cell's improving/stable evidence comes from its own exact McNemar
            posterior, never from the borrowed-strength hierarchy estimate; the
            regression flag comes from the layer-wide spike-and-slab model. The
            hierarchy still shapes ``logit_shift`` and the delta interval.

        A cell with weight 0 is muted: excluded from the score, the pooled
            diagnostic and the homogeneity check, so a view that does not count
            cannot make the layer look inconsistent.

        Coverage gates everything except a credible regression: incomplete
            data must not manufacture health, but it must never bury a flagged
            regression either.
    """
    if options is not None and overrides:
        raise TypeError("pass ReportOptions as options or as keyword arguments, not both")
    if options is None:
        options = ReportOptions(**overrides)
    return _assemble(cells, options)


def parse_category_weights(text: str | None) -> dict[str, float]:
    """Parse ``"tier-1=3,tier-2=1.5"`` into ``{"tier-1": 3.0, "tier-2": 1.5}``.

    A convenience for configuration surfaces that carry one string instead of
    a mapping. Blank means "no category counts for more".

    Raises:
        ValueError: If an entry is not ``name=value``, is not numeric, or
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

    Returns:
        The held level, the composition effect, and how many retired items
            carried a known outcome.

    Notes:
        The overlay is arithmetic only. Retired items are added to the
            numerator and denominator of the level and to nothing else, so the
            paired test, the hierarchy, the guardrail and every interval are
            bit-identical to a run without retirement.

        Weights for the cells that exist only because of retirement are
            resolved with the layer's own ``tiered`` decision, so the overlay can
            never re-weight the cells that ``score`` is built from.

        An item retired before it was ever measured has no outcome to hold
            and is left out; it is still visible through ``retired`` and
            ``coverage.retired``.
    """
    extra = {key: counts for key, counts in (held_counts or {}).items() if counts[0] > 0}
    if not extra or score is None:
        return _Held(score=score, effect_pp=0.0 if score is not None else None, items=0)
    by_key = {cell.key: cell for cell in active}
    stratum_category: dict[str, str | None] = {
        row.stratum: row.category for row in (coverage or [])
    }
    stratum_category.update({cell.stratum: cell.category for cell in active})
    stratum_group: dict[str, str | None] = {row.stratum: row.group for row in (coverage or [])}
    stratum_group.update({cell.stratum: cell.group for cell in active})
    held_only = [
        Cell(
            stratum=stratum,
            outcome=outcome,
            pairs=counts[0],
            passed=counts[1],
            category=stratum_category.get(stratum),
            group=stratum_group.get(stratum),
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

    Notes:
        The critical tier is detected from the coverage itself, not from the
            active cells: a tier-1 view that is entirely pending or errored has no
            active cell, and that is exactly the case the gate must catch.

        Retirement is counted but does not gate (see
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


def _groups(
    cells: list[Cell],
    policy: Weighting,
    model: RegressionModel,
    weighted_pairs: float,
) -> list[GroupSummary]:
    """Roll the grouped cells up into one row per shared dependency.

    The row is the *incident* view of the layer: the pooled evidence of the
    group and the group-as-unit gate's posterior. The members' per-cell flags
    remain in ``contributions``; this is the second, coarser reading. The
    weight is the pair-mass weighted mean of the admitted members' effective
    weights, so a large group's evidence does not inflate its importance in the
    same step. Muted members stay in the descriptive counts — they exist in the
    layer — but contribute no evidence and no mass to the incident, which is
    why their row can never be flagged.
    """
    grouped: dict[str, list[Cell]] = {}
    for cell in cells:
        if cell.group is not None:
            grouped.setdefault(cell.group, []).append(cell)
    weights = {cell.key: policy.weight(cell) for cell in cells}
    summaries: list[GroupSummary] = []
    for name, members in grouped.items():
        admitted = _admitted(members, weights)
        pairs = sum(cell.pairs for cell in members)
        passed = sum(cell.passed for cell in members)
        baseline = sum(cell.baseline_passed for cell in members)
        weight = _group_weight(admitted, weights)
        improved = sum(cell.improved for cell in members)
        regressed = sum(cell.regressed for cell in members)
        contribution = (
            100.0
            * sum(policy.weight(cell) * (cell.improved - cell.regressed) for cell in members)
            / weighted_pairs
            if weighted_pairs > 0.0
            else 0.0
        )
        summaries.append(
            GroupSummary(
                group=name,
                views=len({cell.stratum for cell in members}),
                cells=len(members),
                pairs=pairs,
                improved=improved,
                regressed=regressed,
                discordant=improved + regressed,
                pass_rate=None if pairs <= 0 else passed / pairs,
                baseline_pass_rate=None if pairs <= 0 else baseline / pairs,
                score=None if pairs <= 0 else 100.0 * passed / pairs,
                baseline_score=None if pairs <= 0 else 100.0 * baseline / pairs,
                delta_pp=None if pairs <= 0 else 100.0 * (improved - regressed) / pairs,
                delta_pp_contribution=contribution,
                weight=weight,
                p_incident=model.group_incident.get(name, 0.0),
                flagged=name in model.flagged_groups,
            )
        )
    summaries.sort(key=lambda summary: (-summary.p_incident, summary.group))
    return summaries


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

    Notes:
        Per-cell escalation stays inside the tier, so a tier's verdict is
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
