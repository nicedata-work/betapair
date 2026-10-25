"""The report's serialisable schema and pure-data policy objects.

Split out of :mod:`betapair.report` so the persisted JSON shape and the
label-carrying data structures live where they are used instead of beside the
assembler. This is also what breaks the old ``report <-> series`` import
cycle: :mod:`betapair.series` imports this module, and
:mod:`betapair.report` imports :mod:`betapair.series` at module level.

Nothing here computes a statistic. The pydantic models are the wire format
(see :data:`REPORT_VERSION`) and the dataclasses are the pure resolution
objects the policy functions produce; ``import betapair`` therefore pulls
``pydantic`` only for the persistence surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from betapair.hierarchy import Cell
from betapair.irt import ItemDrift

#: Bumped when the JSON shape changes — reports are persisted per refresh.
#: v4 added ``retired``, ``score_held``, ``composition_effect_pp`` and
#: ``retired_held``; v5 added ``failed`` (view-caused failures) to ``Coverage``
#: and ``HealthReport``; v6 added ``score_low``/``score_high``; v7 added
#: ``score_unpaired``/``unpaired`` (the cold-start level) and ``unresolved_share``;
#: v8 added ``score_adjusted``/``difficulty_adjustment_pp``/``irt`` (the
#: opt-in difficulty adjustment) and the per-cell equated rate; v9 added
#: ``irt.reference`` (the resolved reference bank's name/version/digest/size);
#: v10 added ``CellContribution.group`` (the shared-dependency label); v11 added
#: ``HealthReport.groups`` (the per-incident roll-up of the group-as-unit gate);
#: v12 added ``HealthReport.drift`` (content-fingerprint mismatches); v13 added
#: ``irt.shared_items``/``irt.prior_only`` (the cross-view linking diagnostic);
#: v14 added ``HealthReport.retrospective`` (the windowed level/verdict).
REPORT_VERSION = 14

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

    The model is fitted **once across outcomes**, with one shared ability per
    view; item difficulties stay outcome-specific (``(outcome, item)``). The
    linking diagnostics therefore describe that single view-item graph:
    ``connected``/``components``/``weakest_link``/``shared_items`` are not
    combined over separate fits, and ``prior_only`` means no item links the
    views at all. ``prior_dominated`` and ``separated_items`` count views and
    ``(outcome, item)`` entries respectively.
    """

    outcomes: list[str] = Field(default_factory=list)
    sigma_theta: float
    sigma_d: float
    connected: bool = True
    components: int = 0
    """Largest connected-component count over the per-outcome bank graphs."""
    weakest_link: int = 0
    """Fewest shared items over all view pairs and outcomes (0 = prior only)."""
    shared_items: int = 0
    """Item keys observed by more than one view, summed over fitted outcomes.

    Unlike ``weakest_link`` this is global: a single disjoint view pair no
    longer hides the overlap that does exist. Zero means no cross-view evidence
    entered the fit, so the adjusted comparison is prior-bridged — the report
    says so explicitly with ``prior_only``.
    """
    prior_only: bool = False
    """True when a multi-view outcome shares no item across views.

    The adjusted numbers are then prior-bridged rather than measured: views'
    abilities are exchangeable draws from the prior, not evidence about their
    relative health. ``False`` for a single-view fit (there is no cross-view
    comparison) and ``False`` as soon as any item links the banks.
    """
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


@dataclass(frozen=True)
class RetrospectiveWindow:
    """A stabilised read of the last few epochs in a score series.

    The headline report answers "what changed in *this* epoch pair?". A windowed
    read answers the slower question: "where is the series now, and is it
    moving?" — pooling the window's effective pairs for the level (so it is
    narrower than any single epoch) and comparing the pooled first and second
    halves for the verdict.

    ``score``/``score_low``/``score_high`` are the pooled level and its
    credible interval; ``delta_pp`` and its bounds are the change between the
    window's two halves (``None`` with fewer than two scorable epochs).
    ``trend_pp`` is the pair-weighted least-squares slope per epoch, and
    ``flips`` counts how often the epoch-to-epoch movement changed sign — the
    flapping diagnostic the window is meant to cut. ``status`` is the window's
    verdict under the same ROPE the report uses, restricted to the level
    evidence: a credible movement wins, otherwise ``stable`` means the window's
    central movement is inside the ROPE (a *no practical change observed*
    reading, not proof of invariance), and ``insufficient`` means the window
    saw movement it cannot place. It never overrides a credible per-cell
    regression on the current report (see :func:`delta_status`).
    """

    score: float
    score_low: float
    score_high: float
    delta_pp: float | None
    delta_low: float | None
    delta_high: float | None
    trend_pp: float | None
    flips: int
    status: Status
    points: int
    window: int | None = None

    def __post_init__(self) -> None:
        for name in ("score", "score_low", "score_high"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be in [0, 100], got {value}")
        if not self.score_low <= self.score <= self.score_high:
            raise ValueError("score must lie between score_low and score_high")
        has_delta = self.delta_pp is not None
        if has_delta != (self.delta_low is not None) or has_delta != (self.delta_high is not None):
            raise ValueError("delta_pp, delta_low and delta_high are all-or-none")
        if has_delta:
            assert self.delta_pp is not None
            assert self.delta_low is not None
            assert self.delta_high is not None
            if not self.delta_low <= self.delta_pp <= self.delta_high:
                raise ValueError("delta_pp must lie between delta_low and delta_high")
        if self.trend_pp is not None and not math.isfinite(self.trend_pp):
            raise ValueError("trend_pp must be finite when set")
        if self.flips < 0:
            raise ValueError(f"flips must be non-negative, got {self.flips}")
        if self.points < 1:
            raise ValueError(f"points must be positive, got {self.points}")
        if self.window is not None and self.window < 1:
            raise ValueError(f"window must be at least 1, got {self.window}")
        if self.status not in ("improving", "stable", "regressing", "insufficient"):
            raise ValueError(f"unknown retrospective status {self.status!r}")


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
    retrospective: RetrospectiveWindow | None = None
    """Stabilised read of the last few epochs (``retrospective_window``).

    ``None`` unless the caller asked for a window. The level is the pair-mass
    pooled score of the window and the verdict compares its two halves, so a
    noisy series reads as one stable summary instead of alternating epoch
    verdicts. It is a coarser, level-only reading: the headline ``status``
    remains authoritative and a credible per-cell regression is never hidden.
    """
    drift: list[ItemDrift] = Field(default_factory=list)
    """Item ids observed with changed content (empty when none).

    A row means the same ``(outcome, item)`` id carried two different content
    fingerprints, so its pooled difficulty and paired history mix two
    questions. Reported, not gated: the score and the gate are unchanged,
    because identity is the caller's contract and silently re-keying the row
    would hide the mixed evidence. Sorted and de-duplicated so input order
    cannot change the report.
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
