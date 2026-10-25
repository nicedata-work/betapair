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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from betapair.beta import beta_cdf, beta_ppf, log_beta
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
from betapair.irt import (
    DEFAULT_SIGMA,
    ItemKey,
    ItemResponse,
    RaschFit,
    ReferenceBank,
    bank_digest,
    equated_rates,
    fit_rasch,
    sigmoid,
)
from betapair.mcnemar import mcnemar, switch_evidence

if TYPE_CHECKING:  # pragma: no cover - annotation only, avoids a report<->series cycle
    from betapair.series import ScorePoint

#: Bumped when the JSON shape changes — reports are persisted per refresh.
#: v4 added ``retired``, ``score_held``, ``composition_effect_pp`` and
#: ``retired_held``; v5 added ``failed`` (view-caused failures) to ``Coverage``
#: and ``HealthReport``; v6 added ``score_low``/``score_high``; v7 added
#: ``score_unpaired``/``unpaired`` (the cold-start level) and ``unresolved_share``;
#: v8 added ``score_adjusted``/``difficulty_adjustment_pp``/``irt`` (the
#: opt-in difficulty adjustment) and the per-cell equated rate; v9 added
#: ``irt.reference`` (the resolved reference bank's name/version/digest/size);
#: v10 added ``CellContribution.group`` (the shared-dependency label); v11 added
#: ``HealthReport.groups`` (the per-incident roll-up of the group-as-unit gate).
REPORT_VERSION = 11

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
    group: str | None = None
    """Shared upstream dependency the view belongs to, if one is bound.

    A label only: unlike ``category`` it carries no weight and gates nothing.
    It travels with the cell so callers can recognise the views that fail
    together as one incident.
    """
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


class GroupSummary(BaseModel):
    """One shared-dependency group, read as **one** test rather than N cells.

    A group is the upstream thing several views depend on (a source table, a
    domain). Its cells are not independent, so the regression gate fits them as
    a single unit and this object carries the result: ``p_incident`` is the
    posterior that the group is one incident, and ``flagged`` says whether it
    is in the FDR-controlled incident set. ``weight`` is the pair-mass weighted
    mean of the members' effective weights — the evidence already scales with
    the group's size, so the importance multiplier must not. The per-cell flags
    still exist beside this; a flagged group implicates its cells, and a cell's
    own flag survives whether or not its group is flagged.
    """

    group: str
    views: int = 0
    cells: int = 0
    pairs: int = 0
    improved: int = 0
    regressed: int = 0
    discordant: int = 0
    pass_rate: float | None = None
    baseline_pass_rate: float | None = None
    score: float | None = None
    """Candidate-epoch pass rate of the group's cells, 0-100."""
    baseline_score: float | None = None
    """Baseline-epoch pass rate of the group's cells, 0-100."""
    delta_pp: float | None = None
    """Observed pass-rate change of the group, in percentage points."""
    delta_pp_contribution: float = 0.0
    """This group's additive share of the layer ``delta_pp``.

    The sum of the members' :attr:`CellContribution.delta_pp_contribution`, so
    the layer's movement decomposes exactly into group rows plus the ungrouped
    cells — the attribution a downstream fluctuation is investigated with. The
    members themselves stay in :attr:`HealthReport.contributions` (each tagged
    with ``group``), so a contributing group expands to its cells.
    """
    weight: float = 1.0
    p_incident: float = 0.0
    """``P(this group is one incident | data)`` from the group-as-unit fit."""
    flagged: bool = False
    """Whether the group is in the FDR-controlled incident set."""


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
    group: str | None = None
    """Shared upstream dependency the view belongs to, if one is bound.

    A label only — it never changes ``weight``, the score or the gate. It is
    here so a report can group the views that share a dependency and read a
    joint failure as one incident (see the roadmap's Direction E).
    """
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
    equated_pass_rate: float | None = None
    """Candidate pass rate this cell would score on the common reference bank.

    Present only when ``build_report(adjust_difficulty=True, responses=...)``
    was used. In 0-1, the same scale as ``pass_rate``; on its own bank the
    cell's raw rate is ``pass_rate``, and the gap between the two is the bank
    difficulty the equating removed.
    """
    difficulty_shift_pp: float | None = None
    """``100 * (equated_pass_rate - pass_rate)`` — how far equating moved this
    cell, in percentage points. The layer ``difficulty_adjustment_pp`` is the
    pair-mass-weighted mean of this field (weights ``w_c * pairs_c``), so the
    adjustment is auditable back to the cells."""


@dataclass(frozen=True)
class Calibration:
    """The decision thresholds the report used, and where each came from.

    Every threshold has a hand-set default. When the metric's own history is
    supplied (``history=`` with ``calibrate=True``), the report may size some
    of them from the measured series instead — the ROPE from the band's
    genuine epoch-to-epoch movement, the regression prior from the previous
    epoch's base rate, the coverage gate from the observed pending share — so
    a volatile metric is not judged against a fixed dead zone. This object is
    the *resolved* set: the numbers actually used, plus the provenance needed
    to read them.

    ``enabled`` is whether calibration was requested, ``history_points`` how
    many scored points were consulted, and the ``observed_*`` fields the
    measurements each derivation was based on (``between_sd`` for the ROPE,
    ``observed_pi_mean`` for the regression prior, ``observed_unresolved_share``
    for the coverage gate); each is ``None`` when not measured. ``derived``
    names the thresholds taken from history rather than the caller's default.
    ``derived`` is empty until enough history exists, so ``calibrated`` is
    False exactly when the defaults are in force.
    """

    rope_pp: float = 1.0
    pi_prior: tuple[float, float] = (1.0, 1.0)
    max_unresolved_share: float = 0.25
    enabled: bool = False
    history_points: int = 0
    between_sd: float | None = None
    observed_pi_mean: float | None = None
    observed_unresolved_share: float | None = None
    derived: tuple[str, ...] = ()

    @property
    def calibrated(self) -> bool:
        """True when at least one threshold came from the metric's history."""
        return bool(self.derived)


class ReferenceBankSummary(BaseModel):
    """Which reference bank defined the adjusted score, and how complete it was.

    ``name``/``version`` are set when a versioned
    :class:`~betapair.irt.ReferenceBank` was pinned and ``None`` for an
    unversioned pin or the observed-union default. ``digest`` is a content
    hash of the items *actually used* (:func:`betapair.irt.bank_digest`), so
    two epochs can be compared even when they were pinned under different
    labels. ``items`` is how many bank items the fitted outcomes declared and
    ``available`` how many of them were observed in this epoch and therefore
    equated on; a shortfall means the adjusted score rests on fewer anchors
    than the bank names.
    """

    name: str | None = None
    version: str | None = None
    digest: str
    items: int
    available: int


class IrTSummary(BaseModel):
    """How the difficulty fit linked the views, without the item table.

    Carried on :attr:`HealthReport.irt` only when ``adjust_difficulty`` was on.
    ``score_adjusted`` is the level the views would score on a common
    reference bank; this object is what makes that number readable — whether
    the banks are actually linked (``connected``, ``weakest_link``) or the
    comparison rests on the prior, how many pairs the fit used, and how many
    views/items were perfectly separated (so their estimates are the prior's
    rather than the data's).

    The model is fitted **one outcome at a time**; the summary is taken
    conservatively across those fits, so ``connected`` requires *every*
    outcome's bank graph to be connected and ``weakest_link`` is the global
    minimum over view pairs and outcomes. ``prior_dominated`` and
    ``separated_items`` count (outcome, view) and (outcome, item) entries.
    """

    outcomes: list[str] = Field(default_factory=list)
    sigma_theta: float
    sigma_d: float
    connected: bool = True
    components: int = 0
    """Largest connected-component count over the per-outcome bank graphs."""
    weakest_link: int = 0
    """Fewest shared items over all view pairs and outcomes (0 = prior only)."""
    prior_dominated: int = 0
    """Views whose responses were perfectly separated, across all fits."""
    separated_items: int = 0
    """Items in a perfectly separated all-pass/all-fail set, across all fits."""
    observed_pairs: int = 0
    """Responses the fit consumed — one per paired (view, outcome, item) row."""
    items: int = 0
    """Distinct global items over all fitted outcomes."""
    reference: ReferenceBankSummary | None = None
    """The bank the adjusted score was equated on, and its completeness.

    Always present when ``irt`` is, because ``score_adjusted`` is only
    meaningful relative to a bank; ``name``/``version`` are ``None`` for the
    observed-union default or an unversioned pin, and the ``digest`` still
    identifies the exact item set used.
    """


@dataclass(frozen=True)
class UnpairedCell:
    """One cell's candidate-epoch results that have no baseline to pair with.

    A cold start — the first run, or a view introduced mid-stream — produces
    candidate results before the matching baseline exists. Those rows cannot
    enter the paired ``score`` (there is nothing to compare against), but they
    do carry the candidate epoch's absolute pass rate, which is the initial
    level a dashboard needs. This object is that evidence per cell: how many
    rows resolved and how many of them passed.

    ``resolved`` is the number of rows whose **candidate** side is a real
    result (a :data:`betapair.tracker.FAILED` counts as a failure) while the
    baseline side is still absent. ``passed`` is the subset that passed.
    """

    stratum: str
    outcome: str
    resolved: int = 0
    passed: int = 0
    category: str | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        if not self.stratum or not self.outcome:
            raise ValueError("stratum and outcome must be non-empty")
        if self.group is not None and not self.group:
            raise ValueError("group must be non-empty when set")
        if self.resolved < 0 or self.passed < 0 or self.passed > self.resolved:
            raise ValueError(f"resolved={self.resolved}, passed={self.passed} out of range")

    @property
    def key(self) -> tuple[str, str]:
        return (self.stratum, self.outcome)


class HealthReport(BaseModel):
    """The assembled paired-comparison report, ready to persist or serve as JSON."""

    version: int = REPORT_VERSION
    status: Status
    score: float | None = None
    """Paired candidate-epoch pass rate, 0-100."""
    score_low: float | None = None
    """Lower credible bound on ``score``, 0-100.

    The equal-tail bound of the one-parameter ``Beta(1 + k, 1 + n - k)``
    posterior over the **weighted effective** pairs, whose mode is exactly
    ``score`` — so the interval never moves the point estimate and describes
    the same micro-average, aggregation weights included. ``None`` when the
    report has no paired items at all.
    """
    score_high: float | None = None
    """Upper credible bound on ``score``, 0-100 (see ``score_low``)."""
    score_unpaired: float | None = None
    """Cold-start level, 0-100: candidate-epoch pass rate with no baseline.

    Computed over every row whose candidate side resolved but whose baseline
    side has not run yet — the first run, or a view introduced mid-stream.
    It is **not** the paired ``score`` and never enters ``delta_pp``, the
    verdict, the guardrail or the regression gate; it exists so a brand-new
    view has a number to show while the comparison is still incomplete.
    ``None`` when no such row exists.
    """
    score_unpaired_low: float | None = None
    """Lower credible bound on ``score_unpaired``, 0-100 (see ``score_low``)."""
    score_unpaired_high: float | None = None
    """Upper credible bound on ``score_unpaired``, 0-100."""
    unpaired: int = 0
    """Rows behind ``score_unpaired``: candidate resolved, baseline absent."""
    score_adjusted: float | None = None
    """Difficulty-adjusted level, 0-100: the pass rate the views would score
    on a common reference bank, aggregated with the same weights as ``score``.

    Opt-in: ``None`` unless ``build_report(adjust_difficulty=True,
    responses=...)`` was used. The raw ``score`` is never overwritten — the two
    differ only through the per-cell rates, and ``difficulty_adjustment_pp``
    is the difference, so a bank-difficulty effect is separable from a change.
    """
    score_adjusted_low: float | None = None
    """Lower credible bound on ``score_adjusted``, 0-100.

    The equal-tail bounds of the seeded posterior draws of ``theta`` propagated
    through the equated rates (``RaschFit.sample_abilities``), so the ability
    uncertainty is what sizes the interval. ``None`` alongside
    ``score_adjusted``.
    """
    score_adjusted_high: float | None = None
    """Upper credible bound on ``score_adjusted``, 0-100 (see
    ``score_adjusted_low``)."""
    difficulty_adjustment_pp: float | None = None
    """``score_adjusted - score`` in percentage points.

    Positive means the view was measured on a bank harder than the reference,
    so equating raises its level; negative means the opposite. It is always
    reported beside the raw score, never instead of it.
    """
    irt: IrTSummary | None = None
    """The difficulty fit behind ``score_adjusted``, and its linking caveats."""
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
    groups: list[GroupSummary] = Field(default_factory=list)
    """One row per declared shared-dependency group (empty when none).

    Read as **incidents**, not cells: ``p_incident``/``flagged`` come from the
    group-as-unit regression gate, so a dependency that breaks is one row and
    one flag however many views it takes down. A group's cells still carry
    their own per-cell flags in ``contributions``.
    """
    guardrail: Guardrail = Field(
        default_factory=lambda: Guardrail(status="insufficient", reason="not computed")
    )
    contributions: list[CellContribution] = Field(default_factory=list)
    calibration: Calibration = Field(default_factory=Calibration)
    """The decision thresholds used, and which the metric's history set."""
    unresolved_share: float | None = None
    """Share of tracked rows still unresolved when this report was built.

    ``(pending + errored) / all tracked rows``, over every state — the
    current epoch's coverage pressure, and the evidence the next epoch's
    coverage threshold is calibrated from. ``None`` when no per-cell coverage
    was supplied and the row count is unknown.
    """


@dataclass(frozen=True)
class RegressionModel:
    """Bayesian two-group (spike-and-slab) posterior over which cells regressed.

    Each cell's switch rate is either exactly ``.5`` (the spike: no regression)
    or drawn from a Beta slab on ``[0, .5]``. The mixture weight ``pi`` gets a
    Beta hyperprior and is integrated out, so no empirical-Bayes plug-in is
    needed. ``flagged`` is the largest set whose posterior expected false
    discovery rate is within the target, which is the Bayesian counterpart of
    a multiplicity correction.

    When cells declare a ``group``, the same model is *also* fitted on the
    **groups** as units — a group is one test, its members' evidence pooled —
    and that fit supplies ``group_incident``/``flagged_groups``. Counting a
    group of twenty views as one test is the dependence correction: the FDR
    target is computed against the number of independent things that can break,
    so a shared incident cannot become a flag storm. The per-cell fit is kept
    and its flags are unioned with the flagged groups, so a group can never
    hide an individual regression.
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
    group_incident: dict[str, float] = field(default_factory=dict)
    """``P(the declared group is one incident | data)`` per group.

    Empty when no cell declares a ``group``. The probability comes from the
    **group-as-unit** fit: the group's members are pooled into a single test, so
    twenty views on one source table count as one hypothesis. That is what
    keeps a shared break from being read as twenty independent regressions.
    """
    flagged_groups: frozenset[str] = frozenset()
    """Groups in the FDR-controlled incident set (empty when none declare one)."""
    bfdr_groups: float = 0.0
    """Posterior expected FDR of ``flagged_groups`` (0 with no groups)."""
    muted: frozenset[tuple[str, str]] = frozenset()
    """Cells whose effective weight is 0 — excluded from every flag.

    Muting is a statement that the cell does not matter to the layer, so it is
    muted *everywhere*: it is left out of the score, it cannot be flagged on
    its own evidence, and its group's incident cannot implicate it or be
    created from its evidence. Empty when no weight is 0, so an unweighted
    report is unchanged.
    """


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


@dataclass(frozen=True)
class _Unit:
    """One gate unit: a cell, or a group of cells pooled into one test."""

    keys: tuple[tuple[str, str], ...]
    improved: int
    regressed: int
    weight: float


@dataclass(frozen=True)
class _UnitFit:
    """The spike-and-slab posterior over a list of units (see :func:`_fit_units`)."""

    p_regressed: tuple[float, ...]
    flagged: tuple[int, ...]
    """Indices into the unit list, the FDR-controlled flag set."""
    bfdr: float
    p_no_regression: float
    pi_mean: float


def _fit_units(
    units: list[_Unit],
    *,
    a0: float,
    b0: float,
    fdr: float,
    pi_prior: tuple[float, float],
    pi_points: int,
) -> _UnitFit:
    """Spike-and-slab fit over arbitrary evidence units (cells or groups).

    One arithmetic core for both flags: the per-cell gate passes one unit per
    cell, the group gate passes one unit per group (its members' discordant
    counts summed and its weight their pair-mass-weighted mean). Unit order is
    the caller's and is preserved, so the result maps straight back.
    """
    count = len(units)
    z1 = beta_cdf(0.5, a0, b0)
    if z1 <= 0.0:
        z1 = 1e-300
    log_ratio: list[float] = []
    for unit in units:
        test = switch_evidence(unit.improved, unit.regressed, a0=a0, b0=b0)
        discordant = test.discordant
        if discordant <= 0:
            log_ratio.append(0.0)
            continue
        p_low = 1.0 - test.p_improved
        if p_low <= 0.0:
            log_ratio.append(-math.inf)
            continue
        log_ratio.append(
            log_beta(a0 + unit.improved, b0 + unit.regressed)
            - log_beta(a0, b0)
            + math.log(p_low)
            - math.log(z1)
            - discordant * math.log(0.5)
        )

    raw = [max(0.0, unit.weight) for unit in units]
    mean_weight = sum(raw) / count if count else 1.0
    log_multiplier = [
        math.log(value / mean_weight) if value > 0.0 and mean_weight > 0.0 else -math.inf
        for value in raw
    ]

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
        for index in range(count):
            value += _log1p_exp(base + log_multiplier[index] + log_ratio[index])
            value -= _log1p_exp(base + log_multiplier[index])
        log_posterior.append(value)
    peak = max(log_posterior)
    mass = [math.exp(value - peak) for value in log_posterior]
    total = sum(mass)
    posterior = [value / total for value in mass]

    p_regressed: list[float] = []
    for index in range(count):
        accumulator = 0.0
        for weight, pi in zip(posterior, grid, strict=True):
            log_odds = math.log(pi / (1.0 - pi)) + log_multiplier[index] + log_ratio[index]
            accumulator += weight * _sigmoid(log_odds)
        p_regressed.append(accumulator)

    p_no_regression = 0.0
    for weight, pi in zip(posterior, grid, strict=True):
        base = math.log(pi / (1.0 - pi))
        product = 1.0
        for index in range(count):
            product *= math.exp(-_log1p_exp(base + log_multiplier[index] + log_ratio[index]))
        p_no_regression += weight * product

    ordered = sorted(range(count), key=lambda index: -p_regressed[index])
    cumulative = 0.0
    best = 0
    for position, index in enumerate(ordered, start=1):
        cumulative += 1.0 - p_regressed[index]
        if cumulative / position <= fdr:
            best = position
    flagged = tuple(sorted(ordered[:best]))
    bfdr = sum(1.0 - p_regressed[index] for index in flagged) / len(flagged) if flagged else 0.0
    return _UnitFit(
        p_regressed=tuple(p_regressed),
        flagged=flagged,
        bfdr=bfdr,
        p_no_regression=p_no_regression,
        pi_mean=sum(weight * pi for weight, pi in zip(posterior, grid, strict=True)),
    )


def _admitted(members: list[Cell], raw_weight: dict[tuple[str, str], float]) -> list[Cell]:
    """The members whose evidence a group incident may read.

    A cell muted by its weight is excluded from the score and from every flag,
    so its discordance must not be able to manufacture a group incident and a
    group incident must not be able to implicate it. A group whose members are
    all muted has no admissible evidence and is not tested.
    """
    return [cell for cell in members if raw_weight[cell.key] > 0.0]


def _group_weight(members: list[Cell], raw_weight: dict[tuple[str, str], float]) -> float:
    """Pair-mass weighted mean of the admitted members' effective weights.

    The evidence already scales with the group's size, so the importance
    multiplier must not (`\u03a3 w_c\u00b7pairs_c / \u03a3 pairs_c`); members with no
    pairs fall back to a plain mean of their weights.
    """
    observed = [cell for cell in members if cell.pairs > 0]
    if observed:
        mass = sum(cell.pairs for cell in observed)
        return sum(raw_weight[cell.key] * cell.pairs for cell in observed) / mass
    if not members:
        return 0.0
    return sum(raw_weight[cell.key] for cell in members) / len(members)


def _group_units(
    cells: list[Cell], raw_weight: dict[tuple[str, str], float]
) -> tuple[list[_Unit], dict[str, int]]:
    """Pool each declared group's cells into one unit; keep the rest as cells.

    A group unit carries the summed discordant counts and the **pair-mass
    weighted mean** of its members' effective weights — the evidence scales
    with the group's size, so the importance multiplier must not scale with it
    too (`\u03a3 w_c\u00b7pairs_c / \u03a3 pairs_c`). Muted members contribute no
    evidence and no mass, so muting a view cannot create or shrink an incident
    through a group. Group order is sorted, ungrouped cells keep their input
    order, so a report is deterministic.
    """
    grouped: dict[str, list[Cell]] = {}
    ungrouped: list[Cell] = []
    for cell in cells:
        if cell.group is None:
            ungrouped.append(cell)
        else:
            grouped.setdefault(cell.group, []).append(cell)
    units: list[_Unit] = []
    index: dict[str, int] = {}
    for name in sorted(grouped):
        members = grouped[name]
        admitted = _admitted(members, raw_weight)
        if not admitted:
            continue
        index[name] = len(units)
        units.append(
            _Unit(
                keys=tuple(cell.key for cell in members),
                improved=sum(cell.improved for cell in admitted),
                regressed=sum(cell.regressed for cell in admitted),
                weight=_group_weight(admitted, raw_weight),
            )
        )
    for cell in ungrouped:
        units.append(
            _Unit(
                keys=(cell.key,),
                improved=cell.improved,
                regressed=cell.regressed,
                weight=raw_weight[cell.key],
            )
        )
    return units, index


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

    When cells declare a ``group``, the model is fitted a second time with the
    **group as the unit** (its members' evidence pooled, one test per group),
    and that fit's flag set is published as ``flagged_groups`` — the groups
    that are one incident. The per-cell fit is untouched, so an individual
    regression is never hidden by its group; the two flag sets are unioned by
    :func:`verdicts_from_model`. Grouping changes only *which units the FDR is
    computed over*, never the score or the delta.

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

    raw_weight = {
        cell.key: (max(0.0, float(weights.get(cell.key, 1.0))) if weights is not None else 1.0)
        for cell in cells
    }
    units = [
        _Unit(
            keys=(cell.key,),
            improved=cell.improved,
            regressed=cell.regressed,
            weight=raw_weight[cell.key],
        )
        for cell in cells
    ]
    fit = _fit_units(units, a0=a0, b0=b0, fdr=fdr, pi_prior=pi_prior, pi_points=pi_points)
    p_regressed = {units[index].keys[0]: fit.p_regressed[index] for index in range(len(units))}
    flagged = frozenset(units[index].keys[0] for index in fit.flagged)

    group_incident: dict[str, float] = {}
    flagged_groups: frozenset[str] = frozenset()
    bfdr_groups = 0.0
    if any(cell.group is not None for cell in cells):
        group_units, group_index = _group_units(cells, raw_weight)
        if group_index:  # a group whose members are all muted is not a test
            group_fit = _fit_units(
                group_units, a0=a0, b0=b0, fdr=fdr, pi_prior=pi_prior, pi_points=pi_points
            )
            group_incident = {
                name: group_fit.p_regressed[position] for name, position in group_index.items()
            }
            flagged_set = set(group_fit.flagged)
            flagged_groups = frozenset(
                name for name, position in group_index.items() if position in flagged_set
            )
            bfdr_groups = group_fit.bfdr

    return RegressionModel(
        p_regressed=p_regressed,
        flagged=flagged,
        bfdr=fit.bfdr,
        p_no_regression=fit.p_no_regression,
        pi_mean=fit.pi_mean,
        group_incident=group_incident,
        flagged_groups=flagged_groups,
        bfdr_groups=bfdr_groups,
        muted=frozenset(key for key, weight in raw_weight.items() if weight <= 0.0),
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
    controlled flag set, **or** membership of a group in
    ``model.flagged_groups`` — a group incident implicates its cells, and a
    group can never hide an individual cell's own flag. Separated from
    :func:`regression_model` so the report fits the model once and still has the
    global quantities to hand.
    """
    verdicts: dict[tuple[str, str], CellVerdict] = {}
    for cell in cells:
        test = switch_evidence(cell.improved, cell.regressed, a0=a0, b0=b0)
        in_incident = (
            cell.group is not None
            and cell.group in model.flagged_groups
            and cell.key not in model.muted
        )
        if test.discordant and test.p_improved >= improving_prob:
            verdict: Literal["improving", "stable", "regressing"] = "improving"
        elif cell.key in model.flagged or in_incident:
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
    groups: dict[tuple[str, str], str | None] = field(default_factory=dict)
    """The shared-dependency label of every cell, resolved by the same policy.

    A label only: it does not enter ``weights``. Owning it here keeps the
    report, the tracker and the timeline reading one cell's group from the
    same place, and gives the dependence-aware gate a single source for it.
    """

    def weight(self, cell: Cell | UnpairedCell) -> float:
        return self.weights[cell.key]

    def category(self, cell: Cell | UnpairedCell) -> str | None:
        return self.categories[cell.key]

    def group(self, cell: Cell | UnpairedCell) -> str | None:
        return self.groups[cell.key]

    def defaulted(self, cell: Cell | UnpairedCell) -> bool:
        """True when the cell carried no tier tag and took the fallback tier."""
        return self.tiered and cell.category is None


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

    @note: The ROPE derivation (B2) needs :data:`ROPE_MIN_POINTS` scored points
        before the band is trusted, and keeps ``rope_pp`` when the band
        measures no between-epoch movement at all — a series whose spread is
        pure sampling noise has no dead zone to size, so the hand-set value
        still stands. ``rope_pp`` is the fallback whenever calibration cannot
        size it. The import is local because :mod:`betapair.series` imports
        this module.
    @note: The prior derivation (B3) averages the ``pi_mean`` of the history
        points that carry one — the previous epochs' measured base rate of
        true regressions — and keeps the default ``pi_prior`` strength while
        moving its mean. The mean is clipped to
        :data:`PI_PRIOR_MIN` and :data:`PI_PRIOR_MAX` so the prior cannot
        collapse onto 0 or 1.
    @note: The coverage derivation (B4) averages the epochs' unresolved shares
        and adds :data:`COVERAGE_MARGIN`, clamped to
        :data:`COVERAGE_MIN`/:data:`COVERAGE_MAX`. It waits for
        :data:`COVERAGE_MIN_POINTS` epochs on purpose: the first runs are
        mostly unresolved, and sizing the gate from them would gate every
        early report (the cold-start guard).
    @return: The resolved :class:`Calibration`.
    """
    points = 0 if history is None else len(history)
    derived: list[str] = []
    between_sd: float | None = None
    resolved_rope = 1.0 if rope_pp is None else rope_pp
    if calibrate and history is not None and points >= ROPE_MIN_POINTS:
        from betapair.series import score_band

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
    """Fit one Rasch model per outcome, equate every active cell, aggregate.

    Equating replaces a cell's own-bank pass rate with the rate it would score
    on a common reference bank, ``mean_i sigmoid(theta_v + d_i)``, and the
    adjusted level is the **same** weighted micro-average as ``score`` — so the
    two differ only through the rates. The reference bank defaults to each
    outcome's observed union; it can be pinned with raw ``reference_items``
    (exact: a missing item is an error) or with a versioned
    :class:`~betapair.irt.ReferenceBank`, whose items are intersected with the
    fitted ones so a temporarily absent anchor degrades the width of the
    reference set rather than the whole report. The declared and used sizes are
    published on ``IrTSummary.reference`` either way.

    The interval comes from seeded posterior draws of ``theta``
    (:meth:`RaschFit.sample_abilities`), each mapped to an equated rate and
    aggregated with the same weights, so ability uncertainty sizes the level.
    Each outcome's draws use ``seed + outcome index`` so one report is
    reproducible.

    @raise ValueError: If ``draws`` is not positive, no responses were given,
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

    rates: dict[tuple[str, str], float] = {}
    fitted: list[RaschFit] = []
    draw_totals = [0.0] * draws
    observed_pairs = 0
    declared_items = 0
    resolved_keys: list[ItemKey] = []
    for index, outcome in enumerate(sorted(by_outcome)):
        outcome_responses = by_outcome[outcome]
        fit = fit_rasch(outcome_responses, sigma_theta=sigma_theta, sigma_d=sigma_d)
        fitted.append(fit)
        observed_pairs += len(outcome_responses)
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
            declared = list(fit.difficulties)
            keys = declared
        outcome_rates = equated_rates(fit, reference_items=keys)
        rates.update(outcome_rates)
        declared_items += len(declared)
        resolved_keys.extend(keys)
        reference_difficulties = [fit.difficulties[key] for key in keys]
        sampled = fit.sample_abilities(draws=draws, seed=seed + index)
        for cell in cells:
            if cell.outcome != outcome or cell.key not in outcome_rates:
                continue
            mass = policy.weight(cell) * cell.pairs
            if mass == 0.0:
                continue
            for position, ability in enumerate(sampled[cell.stratum]):
                equated = sum(sigmoid(ability + value) for value in reference_difficulties) / len(
                    reference_difficulties
                )
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
        outcomes=sorted(by_outcome),
        sigma_theta=sigma_theta,
        sigma_d=sigma_d,
        connected=all(fit.diagnostics.connected for fit in fitted),
        components=max((fit.diagnostics.components for fit in fitted), default=0),
        weakest_link=min((fit.diagnostics.weakest_link for fit in fitted), default=0),
        prior_dominated=sum(len(fit.diagnostics.prior_dominated) for fit in fitted),
        separated_items=sum(fit.diagnostics.separated_items for fit in fitted),
        observed_pairs=observed_pairs,
        items=sum(len(fit.difficulties) for fit in fitted),
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
    rope_pp: float | None = None,
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
    history: Sequence[ScorePoint] | None = None,
    calibrate: bool = False,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    unresolved_share: float | None = None,
    responses: Sequence[ItemResponse] | None = None,
    adjust_difficulty: bool = False,
    reference_items: Sequence[ItemKey] | None = None,
    reference_bank: ReferenceBank | None = None,
    sigma_theta: float = DEFAULT_SIGMA,
    sigma_d: float = DEFAULT_SIGMA,
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

    ``group`` is the other label a cell may carry: the shared upstream
    dependency its stratum belongs to. It is passed straight through to
    ``CellContribution.group`` and enters no number — no weight, no gate, no
    interval. It exists so a caller can gather the views that fail together
    into one incident; making the *gate* dependence-aware is a separate step.

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
    if rope_pp is not None and rope_pp < 0.0:
        raise ValueError(f"rope_pp must be >= 0, got {rope_pp}")
    if not 0.0 <= max_unresolved_share <= 1.0:
        raise ValueError(f"max_unresolved_share must be in [0, 1], got {max_unresolved_share}")
    if unresolved_share is not None and not 0.0 <= unresolved_share <= 1.0:
        raise ValueError(f"unresolved_share must be in [0, 1], got {unresolved_share}")
    if reference_bank is not None and reference_items is not None:
        raise ValueError("pass reference_bank or reference_items, not both")
    calibration = resolve_calibration(
        history,
        calibrate=calibrate,
        rope_pp=rope_pp,
        pi_prior=pi_prior,
        max_unresolved_share=max_unresolved_share,
    )
    tier_of = dict(category_weights or {})
    active = [cell for cell in cells if cell.pairs > 0]
    unpaired_active = [cell for cell in (unpaired_cells or []) if cell.resolved > 0]
    # One policy covers the paired cells and the cold-start ones, so a new
    # view's tier and outcome importance are resolved the same way as the
    # established ones and the unpaired level is a true micro-average.
    policy = weighting(
        [*active, *unpaired_active],
        weights=weights,
        category_weights=category_weights,
        default_category=default_category,
    )
    tiered = policy.tiered
    outcomes = sorted({cell.outcome for cell in active})
    strata = len({cell.stratum for cell in active})
    pairs = sum(cell.pairs for cell in active)
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
            unpaired_successes, unpaired_pairs, unpaired_level, prob_interval
        )
    if unresolved_share is None and coverage:
        unresolved_rows = sum(cell.pending + cell.errored for cell in coverage)
        total_rows = sum(
            cell.paired + cell.pending + cell.errored + cell.retired for cell in coverage
        )
        if total_rows:
            unresolved_share = unresolved_rows / total_rows
    cov = _coverage(
        coverage,
        tier_of,
        pending=pending,
        failed=failed,
        pairs=pairs,
        max_unresolved_share=calibration.max_unresolved_share,
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
            calibration=calibration,
            score_unpaired=score_unpaired,
            score_unpaired_low=score_unpaired_low,
            score_unpaired_high=score_unpaired_high,
            unpaired=unpaired_total,
            unresolved_share=unresolved_share,
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
            calibration=calibration,
            score_unpaired=score_unpaired,
            score_unpaired_low=score_unpaired_low,
            score_unpaired_high=score_unpaired_high,
            unpaired=unpaired_total,
            unresolved_share=unresolved_share,
        )

    weighted_successes = sum(policy.weight(cell) * cell.passed for cell in active)
    level = weighted_successes / weighted_pairs
    difficulty: _Difficulty | None = None
    if adjust_difficulty:
        difficulty = _difficulty_adjustment(
            active,
            responses or (),
            policy,
            weighted_pairs,
            reference_items=reference_items,
            reference_bank=reference_bank,
            sigma_theta=sigma_theta,
            sigma_d=sigma_d,
            draws=draws,
            seed=seed,
            prob_interval=prob_interval,
        )
    baseline_level = (
        sum(policy.weight(cell) * cell.baseline_passed for cell in active) / weighted_pairs
    )
    score_low, score_high = _score_interval(
        weighted_successes, weighted_pairs, level, prob_interval
    )
    observed_delta = level - baseline_level
    epsilon = calibration.rope_pp / 100.0
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
        pi_prior=calibration.pi_prior,
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
    groups = _groups(active, policy, model, weighted_pairs)

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
        score_low=score_low,
        score_high=score_high,
        baseline_score=100.0 * baseline_level,
        delta_pp=100.0 * observed_delta,
        delta_pp_low=100.0 * _quantile(delta_draws, 0.025),
        delta_pp_high=100.0 * _quantile(delta_draws, 0.975),
        delta_p_improved=delta_p_improved,
        p_no_regression=model.p_no_regression,
        bfdr=model.bfdr,
        pi_mean=model.pi_mean,
        rope_pp=calibration.rope_pp,
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
        groups=groups,
        guardrail=guardrail,
        contributions=contributions,
        calibration=calibration,
        score_unpaired=score_unpaired,
        score_unpaired_low=score_unpaired_low,
        score_unpaired_high=score_unpaired_high,
        unpaired=unpaired_total,
        unresolved_share=unresolved_share,
        score_adjusted=None if difficulty is None else difficulty.score,
        score_adjusted_low=None if difficulty is None else difficulty.low,
        score_adjusted_high=None if difficulty is None else difficulty.high,
        difficulty_adjustment_pp=(None if difficulty is None else difficulty.score - 100.0 * level),
        irt=None if difficulty is None else difficulty.summary,
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
