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
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

#: Constrained field aliases for the wire schema. A persisted failure is a
#: failure to read, so the model enforces the ranges the producers already
#: guarantee instead of trusting an edited or handwritten payload.
Score = Annotated[float, Field(ge=0.0, le=100.0)]
Rate = Annotated[float, Field(ge=0.0, le=1.0)]
Count = Annotated[int, Field(ge=0)]
NonNegative = Annotated[float, Field(ge=0.0)]
Positive = Annotated[float, Field(gt=0.0)]


def _as_number(name: str, value: object) -> float:
    """Coerce a decision knob to ``float``, rejecting bools and non-numbers."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def _as_int(name: str, value: object) -> int:
    """Require a genuine integer decision knob (no ``True``, no ``2.0``)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def validate_decision_config(
    *,
    alpha: float,
    rope_pp: float | None,
    max_unresolved_share: float,
    unresolved_share: float | None = None,
    prob_interval: float = 0.95,
    draws: int = 2,
    ppc_draws: int = 1,
    pi_points: int = 199,
    a0: float = 1.0,
    b0: float = 1.0,
    a_d: float = 1.0,
    b_d: float = 1.0,
    pi_prior: tuple[float, float] = (1.0, 1.0),
    sigma_theta: float = 1.5,
    sigma_d: float = 1.5,
) -> None:
    """Validate every numeric decision knob in one place.

    Shared by :class:`~betapair.report.ReportOptions` (direct ``build_report``
    callers) and :class:`~betapair.tracker.HealthTracker` (its configuration),
    so a bad value fails where it is set — at construction — rather than deep
    inside a draw or a quantile. The bounds mirror what the producers require:
    ``draws >= 2`` because :func:`betapair.hierarchy.cell_posteriors` takes two
    draws, ``pi_points >= 2`` because the spike-and-slab grid needs two points,
    and positive prior scales because every beta/gamma sample rejects zero.

    Raises:
        ValueError: If any knob is out of range.
    """
    alpha_value = _as_number("alpha", alpha)
    if not 0.0 < alpha_value < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if rope_pp is not None and _as_number("rope_pp", rope_pp) < 0.0:
        raise ValueError(f"rope_pp must be >= 0, got {rope_pp}")
    if not 0.0 <= _as_number("max_unresolved_share", max_unresolved_share) <= 1.0:
        raise ValueError(f"max_unresolved_share must be in [0, 1], got {max_unresolved_share}")
    if (
        unresolved_share is not None
        and not 0.0 <= _as_number("unresolved_share", unresolved_share) <= 1.0
    ):
        raise ValueError(f"unresolved_share must be in [0, 1], got {unresolved_share}")
    if not 0.0 < _as_number("prob_interval", prob_interval) < 1.0:
        raise ValueError(f"prob_interval must be in (0, 1), got {prob_interval}")
    if _as_int("draws", draws) < 2:
        raise ValueError(f"draws must be an integer >= 2, got {draws!r}")
    if _as_int("ppc_draws", ppc_draws) < 1:
        raise ValueError(f"ppc_draws must be an integer >= 1, got {ppc_draws!r}")
    if _as_int("pi_points", pi_points) < 2:
        raise ValueError(f"pi_points must be an integer >= 2, got {pi_points!r}")
    for name, value in (
        ("a0", a0),
        ("b0", b0),
        ("a_d", a_d),
        ("b_d", b_d),
        ("sigma_theta", sigma_theta),
        ("sigma_d", sigma_d),
    ):
        number = _as_number(name, value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be a positive finite number, got {value}")
    if len(pi_prior) != 2 or min(_as_number("pi_prior", part) for part in pi_prior) <= 0.0:
        raise ValueError(f"pi_prior shapes must be positive, got {pi_prior}")


__all__ = [
    "DEFAULT_CATEGORY",
    "REPORT_VERSION",
    "Calibration",
    "CategorySummary",
    "CellContribution",
    "CellCoverage",
    "Coverage",
    "GroupSummary",
    "Guardrail",
    "HealthReport",
    "HierarchySummary",
    "IrTSummary",
    "McNemarSummary",
    "ReferenceBankSummary",
    "RetrospectiveWindow",
    "Status",
    "UnpairedCell",
    "Weighting",
    "validate_decision_config",
]


class _WireModel(BaseModel):
    """Base for every persisted model: unknown keys are rejected, and frozen.

    ``extra="forbid"`` means a payload with a key the schema does not define
    is a read error rather than silently dropped data; ``frozen=True`` means a
    parsed report cannot be mutated after validation. Together they make
    :data:`REPORT_VERSION` an actual contract instead of a comment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


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


class McNemarSummary(_WireModel):
    """Posterior summary of a paired comparison."""

    n_improved: Count
    n_regressed: Count
    discordant: Count
    post_mean: Rate
    post_median: Rate
    eti_low: Rate
    eti_high: Rate
    bf10_point: NonNegative
    bf10_interval: NonNegative
    p_improved: Rate
    """``P(phi > .5)`` — the probability that switches ran net-improving."""


class HierarchySummary(_WireModel):
    """Pooled switch rate and between-strata concentration."""

    mu: Rate
    mu_eti_low: Rate
    mu_eti_high: Rate
    kappa: NonNegative
    kappa_eti_low: NonNegative
    kappa_eti_high: NonNegative
    log_marginal: float
    cells: Count


class Guardrail(_WireModel):
    """Homogeneity check across strata, plus the cells it flags."""

    status: Literal["ok", "heterogeneous", "insufficient"]
    ppc: Rate | None = None
    """Posterior predictive tail probability; small means the strata disagree.
    A Bayesian model-check quantity, not a sampling p-value."""
    dispersion: NonNegative | None = None
    expected_dispersion: NonNegative | None = None
    draws: Count = 0
    flagged: list[str] = Field(default_factory=list)
    """Cells with a credible regression, as ``stratum/outcome``.

    A cell is listed when the spike-and-slab regression model puts it in the
    posterior-FDR-controlled flag set (see :func:`regression_model`), so a wide
    layer does not manufacture a regression by chance.
    """
    reason: str = ""


class CategorySummary(_WireModel):
    """One criticality tier, expanded from the same cells as the layer score.

    Within a tier the category weight is constant, so it cancels out of
    ``score``/``delta_pp`` — the tier is summarised on its own terms, and
    ``weight`` only says how much it moved the layer's headline.
    """

    category: str | None = None
    weight: NonNegative = 1.0
    defaulted: Count = 0
    """Cells in this tier whose view carried no tag and took ``DEFAULT_CATEGORY``."""
    views: Count = 0
    cells: Count = 0
    pairs: Count = 0
    score: Score | None = None
    baseline_score: Score | None = None
    delta_pp: float | None = None
    discordant: Count = 0
    p_improved: Rate | None = None
    """Pooled McNemar ``P(phi > .5)`` for this tier alone."""
    status: Status = "insufficient"
    delta_p_improved: Rate | None = None
    """Weighted ``P(delta > 0)`` for this tier, from the hierarchical draws."""


class GroupSummary(_WireModel):
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
    views: Count = 0
    cells: Count = 0
    pairs: Count = 0
    improved: Count = 0
    regressed: Count = 0
    discordant: Count = 0
    pass_rate: Rate | None = None
    baseline_pass_rate: Rate | None = None
    score: Score | None = None
    """Candidate-epoch pass rate of the group's cells, 0-100."""
    baseline_score: Score | None = None
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
    weight: NonNegative = 1.0
    p_incident: Rate = 0.0
    """``P(this group is one incident | data)`` from the group-as-unit fit."""
    flagged: bool = False
    """Whether the group is in the FDR-controlled incident set."""


class Coverage(_WireModel):
    """How much of the comparison actually resolved, and whether that is enough."""

    paired: Count = 0
    pending: Count = 0
    errored: Count = 0
    retired: Count = 0
    """Items withdrawn from the bank — reported, but never gating."""
    failed: Count = 0
    """Paired items scored as failures because the view, not the harness, failed.

    Already inside ``paired``: unlike ``pending`` and ``errored`` these did
    resolve, so they neither shrink the denominator nor gate the verdict. They
    are surfaced so a view that is *breaking* is distinguishable from one that
    is merely wrong — both cost the same score, but only one is an incident.
    """
    unresolved_share: Rate = 0.0
    critical_unresolved: Count = 0
    """Pending/errored items in the highest-criticality tier present."""
    complete: bool = True
    reason: str = ""


class CellContribution(_WireModel):
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
    weight: NonNegative = 1.0
    """Effective aggregation weight: outcome importance x category weight."""
    pairs: Count
    improved: Count
    regressed: Count
    unchanged: Count
    pass_rate: Rate
    baseline_pass_rate: Rate
    delta_pp: float
    """Observed pass-rate change in percentage points (improvement positive)."""
    delta_pp_low: float
    delta_pp_high: float
    p_improved: Rate
    p_regressed: Rate = 0.0
    """Posterior probability this cell is a true regression, from the
    spike-and-slab model (see :func:`regression_model`)."""
    logit_shift: float
    """Shrunken log-odds shift of this cell against the pooled layer."""
    verdict: Literal["improving", "stable", "regressing"]
    delta_pp_contribution: float = 0.0
    """This cell's additive share of ``delta_pp``: ``100 * w * (n01 - n10) /
    sum(w * pairs)``. Summing this field over every contribution reproduces
    ``delta_pp`` exactly, even when cells offset."""
    equated_pass_rate: Rate | None = None
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


class ReferenceBankSummary(_WireModel):
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
    items: Count
    available: Count


class IrTSummary(_WireModel):
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
    sigma_theta: Positive
    sigma_d: Positive
    connected: bool = True
    components: Count = 0
    """Largest connected-component count over the per-outcome bank graphs."""
    weakest_link: Count = 0
    """Fewest shared items over all view pairs and outcomes (0 = prior only)."""
    shared_items: Count = 0
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
    prior_dominated: Count = 0
    """Views whose responses were perfectly separated, across all fits."""
    separated_items: Count = 0
    """Items in a perfectly separated all-pass/all-fail set, across all fits."""
    observed_pairs: Count = 0
    """Responses the fit consumed — one per paired (view, outcome, item) row."""
    items: Count = 0
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
        """``(stratum, outcome)`` — the cell's identity in the report."""
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


class HealthReport(_WireModel):
    """The assembled paired-comparison report, ready to persist or serve as JSON."""

    version: Count = REPORT_VERSION
    status: Status
    score: Score | None = None
    """Paired candidate-epoch pass rate, 0-100."""
    score_low: Score | None = None
    """Lower credible bound on ``score``, 0-100.

    The equal-tail bound of the one-parameter ``Beta(1 + k, 1 + n - k)``
    posterior over the **weighted effective** pairs, whose mode is exactly
    ``score`` — so the interval never moves the point estimate and describes
    the same micro-average, aggregation weights included. ``None`` when the
    report has no paired items at all.
    """
    score_high: Score | None = None
    """Upper credible bound on ``score``, 0-100 (see ``score_low``)."""
    score_unpaired: Score | None = None
    """Cold-start level, 0-100: candidate-epoch pass rate with no baseline.

    Computed over every row whose candidate side resolved but whose baseline
    side has not run yet — the first run, or a view introduced mid-stream.
    It is **not** the paired ``score`` and never enters ``delta_pp``, the
    verdict, the guardrail or the regression gate; it exists so a brand-new
    view has a number to show while the comparison is still incomplete.
    ``None`` when no such row exists.
    """
    score_unpaired_low: Score | None = None
    """Lower credible bound on ``score_unpaired``, 0-100 (see ``score_low``)."""
    score_unpaired_high: Score | None = None
    """Upper credible bound on ``score_unpaired``, 0-100."""
    unpaired: Count = 0
    """Rows behind ``score_unpaired``: candidate resolved, baseline absent."""
    score_adjusted: Score | None = None
    """Difficulty-adjusted level, 0-100: the pass rate the views would score
    on a common reference bank, aggregated with the same weights as ``score``.

    Opt-in: ``None`` unless ``build_report(adjust_difficulty=True,
    responses=...)`` was used. The raw ``score`` is never overwritten — the two
    differ only through the per-cell rates, and ``difficulty_adjustment_pp``
    is the difference, so a bank-difficulty effect is separable from a change.
    """
    score_adjusted_low: Score | None = None
    """Lower credible bound on ``score_adjusted``, 0-100.

    The equal-tail bounds of the seeded posterior draws of ``theta`` propagated
    through the equated rates (``RaschFit.sample_abilities``), so the ability
    uncertainty is what sizes the interval. ``None`` alongside
    ``score_adjusted``.
    """
    score_adjusted_high: Score | None = None
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
    baseline_score: Score | None = None
    delta_pp: float | None = None
    delta_pp_low: float | None = None
    delta_pp_high: float | None = None
    delta_p_improved: Rate | None = None
    """Weighted ``P(delta_pp > 0)`` from the hierarchical draws — the
    probability behind the layer verdict. ``p_improved`` below is the
    unweighted pooled exact test, kept as a diagnostic only."""
    p_no_regression: Rate | None = None
    """``P(no cell truly regressed | data)`` from the spike-and-slab model."""
    bfdr: Rate | None = None
    """Posterior expected false discovery rate of the flagged set."""
    pi_mean: Rate | None = None
    """Posterior mean share of cells that truly regressed."""
    rope_pp: NonNegative = 0.0
    """ROPE half-width in percentage points used for the verdict."""
    p_improved: Rate | None = None
    pairs: Count = 0
    pending: Count = 0
    failed: Count = 0
    """Paired items scored as failures because the view itself failed.

    A view-caused error (a timeout, a crash) is a result, so it is counted
    against the level instead of being dropped from it — the incentive TS2
    encodes as "quitting is a surrender".
    """
    retired: Count = 0
    """Items withdrawn from the bank since the comparison began."""
    retired_held: Count = 0
    """Retired items with a known outcome — the ones ``score_held`` folds in."""
    score_held: Score | None = None
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
    strata: Count = 0
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
    unresolved_share: Rate | None = None
    """Share of tracked rows still unresolved when this report was built.

    ``(pending + errored) / all tracked rows``, over every state — the
    current epoch's coverage pressure, and the evidence the next epoch's
    coverage threshold is calibrated from. ``None`` when no per-cell coverage
    was supplied and the row count is unknown.
    """

    @model_validator(mode="after")
    def _check_credible_bounds(self) -> HealthReport:
        """Credible bounds are all-or-none, and ordered when present.

        A lone ``score_low`` (or ``delta_pp_low``, or an adjusted-bound pair)
        describes nothing, so a payload carrying exactly one side is rejected
        rather than half-read.
        """
        for low, high in (
            (self.score_low, self.score_high),
            (self.score_unpaired_low, self.score_unpaired_high),
            (self.score_adjusted_low, self.score_adjusted_high),
        ):
            if (low is None) != (high is None):
                raise ValueError("credible bounds must be both present or both absent")
            if low is not None and high is not None and low > high:
                raise ValueError(f"credible bound low {low} exceeds high {high}")
        delta = (self.delta_pp, self.delta_pp_low, self.delta_pp_high)
        missing = sum(value is None for value in delta)
        if 0 < missing < len(delta):
            raise ValueError("delta_pp, delta_pp_low and delta_pp_high are all-or-none")
        return self

    @classmethod
    def load_json(cls, text: str) -> HealthReport:
        """Parse persisted report JSON, refusing another schema version.

        :meth:`pydantic.BaseModel.model_validate_json` answers "is this
        well-formed JSON for the current fields?"; this additionally answers
        "was it written by this schema version?", which is what
        :data:`REPORT_VERSION` exists for. Extra keys already fail through
        ``extra="forbid"``; a same-shaped payload from an older version fails
        here.

        Raises:
            ValueError: If the payload is malformed or its ``version`` is
                not :data:`REPORT_VERSION`.
        """
        report = cls.model_validate_json(text)
        if report.version != REPORT_VERSION:
            raise ValueError(
                f"unsupported report version {report.version!r}; expected {REPORT_VERSION}"
            )
        return report


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
        """Effective importance of ``cell`` (outcome importance x tier)."""
        return self.weights[cell.key]

    def category(self, cell: Cell | UnpairedCell) -> str | None:
        """Resolved tier of ``cell`` (the fallback when it carried no tag)."""
        return self.categories[cell.key]

    def group(self, cell: Cell | UnpairedCell) -> str | None:
        """Shared-dependency label of ``cell``, or ``None`` when ungrouped."""
        return self.groups[cell.key]

    def defaulted(self, cell: Cell | UnpairedCell) -> bool:
        """True when the cell carried no tier tag and took the fallback tier."""
        return self.tiered and cell.category is None
