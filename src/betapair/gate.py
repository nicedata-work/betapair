"""The dependence-aware regression gate and the layer verdict.

Extracted from :mod:`betapair.report` as a self-contained Bayesian model: the
spike-and-slab posterior over which cells (or shared-dependency groups) truly
regressed, the per-cell verdict derived from it, and the ROPE decision that
turns weighted delta draws into a layer status. It has its own tests
(``tests/test_report.py``, ``tests/test_calibration.py``) and no dependency on
the assembler.

The logistic helpers are shared with :mod:`betapair.irt` from
:mod:`betapair.beta` (imported here as ``sigmoid``/``log1p_exp``), so the two
modules cannot drift apart.

Notes:
    The decision is Bayesian end to end: ``fdr`` below is a posterior
        expected FDR, not a frequentist family-wise error rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from betapair.beta import beta_cdf, log1p_exp, log_beta, sigmoid
from betapair.hierarchy import Cell
from betapair.mcnemar import switch_evidence
from betapair.models import Status

__all__ = [
    "CellVerdict",
    "RegressionModel",
    "cell_verdicts",
    "delta_status",
    "regression_model",
    "verdicts_from_model",
]


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
            value += log1p_exp(base + log_multiplier[index] + log_ratio[index])
            value -= log1p_exp(base + log_multiplier[index])
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
            accumulator += weight * sigmoid(log_odds)
        p_regressed.append(accumulator)

    p_no_regression = 0.0
    for weight, pi in zip(posterior, grid, strict=True):
        base = math.log(pi / (1.0 - pi))
        product = 1.0
        for index in range(count):
            product *= math.exp(-log1p_exp(base + log_multiplier[index] + log_ratio[index]))
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

    Returns:
        The fitted :class:`RegressionModel`.

    Raises:
        ValueError: If ``fdr`` is outside (0, 1), a prior shape is not
            positive, or ``pi_points`` is below 2.

    Notes:
        The decision is Bayesian end to end: ``fdr`` is a posterior expected
            FDR, not a frequentist family-wise error rate, and no p-value enters.
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

    Notes:
        Unlike the previous weighted-Holm gate, nothing here is a p-value
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

    Notes:
        Because the draws include the discordance-rate posterior, an
            unchanged bank is not automatically ``insufficient``: as the pair
            count grows the delta concentrates inside the ROPE, which is how
            "20,000 items and nothing moved" becomes positive evidence of
            stability.

        A credible regression always wins, even if the weighted net is
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
