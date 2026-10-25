"""Operating-characteristic simulations for the report's decisions.

Run with::

    uv run python tools/calibration.py            # print markdown to stdout
    uv run python tools/calibration.py --write    # (re)write docs/validation.md
    uv run python tools/calibration.py --check    # fail if that document is stale

The ``--check`` mode regenerates the tables and compares them byte-for-byte
with the committed document, so a statistical change cannot land without its
evidence being regenerated; it is what ``ci.yml`` runs.

The regression gate is now Bayesian (a spike-and-slab model with a posterior
FDR rule), so the natural questions are:

* under a realistic global null, how often does it flag anything, and how many
  cells per layer?
* does criticality (a prior-odds multiplier) raise the power to flag a true
  tier-1 regression?
* can a heterogeneous layer still read as ``improving``?
* does a self-calibrated ROPE stop a volatile metric from reading ``improving``?
* does a self-calibrated regression prior track the measured base rate?
* does a self-calibrated coverage gate tighten when the pipeline is healthy?
* does the difficulty-adjusted interval cover, and how much does weak bank
  linking cost the equated score?
* does equating actually remove a bank's difficulty from the level?

Everything is seeded, so the tables are reproducible. These are the frequentist
operating characteristics of a Bayesian procedure: the FDR target is a
posterior quantity, but it is still useful to see how the rule behaves when the
null is true.
"""

from __future__ import annotations

import argparse
import difflib
import math
import random
import sys
from pathlib import Path

from betapair import (
    Cell,
    CellCoverage,
    ItemResponse,
    ReferenceBank,
    ScorePoint,
    build_report,
    cell_verdicts,
    equated_rates,
    fit_rasch,
    regression_model,
    resolve_calibration,
)
from betapair.irt import sigmoid

TRIALS = 400
DISCORDANT = 20
#: Auxiliary cells the prior table's layers carry.
DISCORDANT_AUX = 40
CRITICAL_WEIGHT = 10.0
#: Discordant pairs and total size of the layers in the self-calibration tables.
CALIBRATION_DISCORDANT = 340
CALIBRATION_PAIRS = 2000
#: True net improvements (pp) the verdict-stability table sweeps.
NET_IMPROVEMENTS = (0.0, 1.0, 3.0, 6.0, 9.0)
#: Volatility levels (pp) the calibrated ROPE is sized from.
VOLATILITY_LEVELS = (0.0, 3.0, 6.0, 9.0)
#: Base rates (share of cells that truly regressed) the prior table sweeps, and
#: the (bounded) trials it measures each with.
PRIOR_RATES = (0.02, 0.2, 0.5, 0.8)
PRIOR_TRIALS = 200
#: Unresolved shares the coverage table's histories carry. The probe epoch is
#: 23% unresolved — the share that passes the hand-set 25% gate.
COVERAGE_RATES = (0.02, 0.10, 0.23, 0.45)
COVERAGE_PROBE = (77, 23)
#: Newton tolerance for the calibration fits. The library default (1e-8) is a
#: precision knob, not a modelling one; these tables are about bias and
#: coverage, and 1e-6 converges in a handful of sweeps instead of hitting the
#: sweep cap.
IRT_TOL = 1e-6
#: Design of the adjusted-interval coverage table.
IRT_COVERAGE_VIEWS = 4
IRT_COVERAGE_ITEMS = 16
IRT_COVERAGE_TRIALS = 100
#: Linking-regime table: views per group, items per own bank, shared anchors.
IRT_LINK_VIEWS = 2
IRT_LINK_ITEMS = 12
IRT_LINK_ANCHORS = 8
IRT_LINK_TRIALS = 150
#: Bank-imbalance table: anchors every view runs, and own-bank items.
IRT_IMBALANCE_VIEWS = 4
IRT_IMBALANCE_ANCHORS = 12
IRT_IMBALANCE_ITEMS = 10
IRT_IMBALANCE_TRIALS = 150
IMBALANCE_OFFSETS = (0.0, 1.0, 2.0, 3.0)
#: Shared-dependency table: groups per layer, cells per group, discordant pairs
#: per cell, and the switch rates a group-wide break is swept over.
GROUP_TRIALS = 100
#: Null trials are cheaper than the power sweep (one fit, not two), so the
#: false-incident rate is measured more precisely.
GROUP_NULL_TRIALS = 400
GROUP_GROUPS = 6
GROUP_CELLS = 10
GROUP_DISCORDANT = 20
GROUP_BREAK_RATES = (0.45, 0.40, 0.35, 0.30, 0.25, 0.15)


def _draw(rng: random.Random, n: int, q: float) -> tuple[int, int]:
    improved = sum(1 for _ in range(n) if rng.random() < q)
    return improved, n - improved


def _cell(
    stratum: str,
    improved: int,
    regressed: int,
    category: str | None = None,
    group: str | None = None,
) -> Cell:
    pairs = improved + regressed
    return Cell(
        stratum=stratum,
        outcome="answer",
        improved=improved,
        regressed=regressed,
        passed=improved,
        pairs=pairs,
        category=category,
        group=group,
    )


def _weights(cells: list[Cell], weighted: bool) -> dict[tuple[str, str], float] | None:
    if not weighted:
        return None
    return {cell.key: (CRITICAL_WEIGHT if cell.category == "tier-1" else 1.0) for cell in cells}


def null_flags(
    m: int, *, weighted: bool, trials: int = TRIALS, seed: int = 0
) -> tuple[float, float]:
    """Under the global null, (fraction of layers with a flag, flags per layer)."""
    rng = random.Random(seed)
    flagged_trials = 0
    total_flags = 0
    for _ in range(trials):
        cells = []
        for index in range(m):
            improved, regressed = _draw(rng, DISCORDANT, 0.5)
            category = "tier-1" if (weighted and index == 0) else ("tier-3" if weighted else None)
            cells.append(_cell(f"v{index}", improved, regressed, category))
        verdicts = cell_verdicts(cells, weights=_weights(cells, weighted))
        flags = [result for result in verdicts.values() if result.verdict == "regressing"]
        total_flags += len(flags)
        flagged_trials += bool(flags)
    return flagged_trials / trials, total_flags / trials


def power(
    n_critical: int,
    *,
    weighted: bool,
    n_aux: int = 40,
    q_critical: float = 0.2,
    trials: int = TRIALS,
    seed: int = 0,
) -> float:
    """Probability of flagging a tier-1 cell whose true switch rate is q_critical."""
    rng = random.Random(seed)
    detected = 0
    for _ in range(trials):
        improved, regressed = _draw(rng, n_critical, q_critical)
        cells = [_cell("critical", improved, regressed, "tier-1" if weighted else None)]
        for index in range(n_aux):
            aux_improved, aux_regressed = _draw(rng, DISCORDANT, 0.5)
            cells.append(
                _cell(f"aux{index}", aux_improved, aux_regressed, "tier-3" if weighted else None)
            )
        verdicts = cell_verdicts(cells, weights=_weights(cells, weighted))
        detected += verdicts[("critical", "answer")].verdict == "regressing"
    return detected / trials


def _history_for_sd(sd: float) -> list[ScorePoint]:
    """Three scored epochs whose sample sd of scores is exactly ``sd``.

    Each point's stored bounds equal its score, so the band's within-epoch
    variance is zero and its ``between_sd`` is the score sd itself — the
    controlled knob this table turns.
    """
    return [
        ScorePoint(score=50.0 - sd, low=50.0 - sd, high=50.0 - sd),
        ScorePoint(score=50.0, low=50.0, high=50.0),
        ScorePoint(score=50.0 + sd, low=50.0 + sd, high=50.0 + sd),
    ]


def _layer(net_pp: float) -> list[Cell]:
    """One layer whose true net movement is exactly ``net_pp`` percentage points.

    The discordant total is fixed, so the improvement is the only knob and the
    ROPE's verdict is the only thing that can change between rows.
    """
    improved = round((CALIBRATION_DISCORDANT + net_pp * CALIBRATION_PAIRS / 100.0) / 2.0)
    regressed = CALIBRATION_DISCORDANT - improved
    return [
        Cell(
            stratum="v",
            outcome="answer",
            improved=improved,
            regressed=regressed,
            passed=(CALIBRATION_PAIRS - CALIBRATION_DISCORDANT) // 2 + improved,
            pairs=CALIBRATION_PAIRS,
        )
    ]


def verdict_stability() -> list[tuple[float, str, tuple[str, ...]]]:
    """Verdicts for a real improvement under the default and calibrated ROPEs.

    Returns ``(true improvement pp, default verdict, calibrated verdicts)``
    rows, with one calibrated verdict per :data:`VOLATILITY_LEVELS` entry.
    Only the history's volatility changes across a row's calibrated columns,
    so every difference is the ROPE's.
    """
    rows: list[tuple[float, str, tuple[str, ...]]] = []
    for net_pp in NET_IMPROVEMENTS:
        cells = _layer(net_pp)
        default = build_report(cells, draws=96, ppc_draws=96, seed=1)
        calibrated = tuple(
            build_report(
                cells,
                history=_history_for_sd(sd),
                calibrate=True,
                draws=96,
                ppc_draws=96,
                seed=1,
            ).status
            for sd in VOLATILITY_LEVELS
        )
        rows.append((net_pp, default.status, calibrated))
    return rows


def _resolved_prior(rate: float) -> tuple[float, float]:
    """The ``pi_prior`` the report resolves from a history measuring ``rate``."""
    resolved = resolve_calibration(
        [ScorePoint(score=60.0, pairs=50.0, pi_mean=rate)],
        calibrate=True,
        rope_pp=None,
        pi_prior=(1.0, 1.0),
        max_unresolved_share=0.25,
    )
    return resolved.pi_prior


def _power_with_prior(prior: tuple[float, float], discordant: int) -> float:
    """Probability of flagging one true regressor (20% switch rate) under ``prior``."""
    rng = random.Random(0)
    detected = 0
    for _ in range(PRIOR_TRIALS):
        improved, regressed = _draw(rng, discordant, 0.2)
        cells = [_cell("critical", improved, regressed)]
        for index in range(DISCORDANT_AUX):
            aux_improved, aux_regressed = _draw(rng, DISCORDANT, 0.5)
            cells.append(_cell(f"aux{index}", aux_improved, aux_regressed))
        verdicts = cell_verdicts(cells, pi_prior=prior)
        detected += verdicts[("critical", "answer")].verdict == "regressing"
    return detected / PRIOR_TRIALS


def prior_calibration() -> list[tuple[float, float, float, float, float]]:
    """Power as the measured base rate re-centres the prior.

    Each row resolves ``pi_prior`` from a history whose measured base rate is
    the row label, then measures the power to flag one true regressor at two
    evidence levels: 12 and 20 discordant pairs.
    """
    rows: list[tuple[float, float, float, float, float]] = []
    for rate in PRIOR_RATES:
        prior = _resolved_prior(rate)
        rows.append(
            (rate, prior[0], prior[1], _power_with_prior(prior, 12), _power_with_prior(prior, 20))
        )
    return rows


def coverage_calibration() -> list[tuple[float, float, bool]]:
    """The resolved gate and whether a 23%-unresolved epoch clears it.

    Each row resolves the gate from three completed epochs whose mean
    unresolved share is the row label, then judges an epoch that is 23%
    unresolved — the share that passes the hand-set 25% gate.
    """
    rows: list[tuple[float, float, bool]] = []
    for rate in COVERAGE_RATES:
        history = [ScorePoint(score=60.0, pairs=50.0, unresolved_share=rate) for _ in range(3)]
        report = build_report(
            [Cell(stratum="v", outcome="answer", improved=10, regressed=5, passed=70, pairs=100)],
            coverage=[
                CellCoverage("v", "answer", paired=COVERAGE_PROBE[0], pending=COVERAGE_PROBE[1])
            ],
            pending=COVERAGE_PROBE[1],
            history=history,
            calibrate=True,
            draws=96,
            ppc_draws=96,
            seed=1,
        )
        rows.append((rate, report.calibration.max_unresolved_share, report.coverage.complete))
    return rows


def _irt_truth(
    rng: random.Random,
    views: int,
    items: list[str],
    *,
    sigma_theta: float = 1.2,
    sigma_d: float = 1.0,
    shift: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Known abilities and item easinesses for one simulated epoch."""
    abilities = {f"v{index}": rng.gauss(0.0, sigma_theta) for index in range(views)}
    difficulties = {item: rng.gauss(shift, sigma_d) for item in items}
    return abilities, difficulties


def _irt_responses(
    rng: random.Random,
    abilities: dict[str, float],
    difficulties: dict[str, float],
    observations: dict[str, list[str]],
) -> list[ItemResponse]:
    """Candidate-epoch responses of each view on the items it ran."""
    return [
        ItemResponse(stratum, "answer", item, rng.random() < sigmoid(ability + difficulties[item]))
        for stratum, ability in abilities.items()
        for item in observations[stratum]
    ]


def _irt_cells(responses: list[ItemResponse]) -> list[Cell]:
    """One discordant-free cell per view, matching its response count."""
    counts: dict[str, list[int]] = {}
    for response in responses:
        row = counts.setdefault(response.stratum, [0, 0])
        row[0] += int(response.passed)
        row[1] += 1
    return [
        Cell(stratum=stratum, outcome="answer", improved=0, regressed=0, passed=passed, pairs=pairs)
        for stratum, (passed, pairs) in sorted(counts.items())
    ]


def _true_equated(ability: float, items: list[str], difficulties: dict[str, float]) -> float:
    """True equated pass rate of one view on ``items``, 0-1."""
    return sum(sigmoid(ability + difficulties[item]) for item in items) / len(items)


def adjusted_interval_coverage(
    trials: int = IRT_COVERAGE_TRIALS, seed: int = 0
) -> tuple[float, float, float, float]:
    """Coverage of the raw and adjusted intervals at the 95% nominal level.

    Every view runs the same bank, so the true raw pass rate and the true
    equated rate are the same quantity — the population mean of
    ``sigmoid(theta_v + d_i)`` over the views and items. Returns ``(raw
    coverage, adjusted coverage, raw width, adjusted width)``, the widths in
    percentage points.
    """
    rng = random.Random(seed)
    raw_covered = adjusted_covered = 0
    raw_widths: list[float] = []
    adjusted_widths: list[float] = []
    for trial in range(trials):
        items = [f"q{index}" for index in range(IRT_COVERAGE_ITEMS)]
        abilities, difficulties = _irt_truth(rng, IRT_COVERAGE_VIEWS, items)
        observations = {stratum: items for stratum in abilities}
        responses = _irt_responses(rng, abilities, difficulties, observations)
        bank = ReferenceBank("standard", "coverage", tuple(("answer", item) for item in items))
        report = build_report(
            _irt_cells(responses),
            responses=responses,
            adjust_difficulty=True,
            reference_bank=bank,
            draws=96,
            ppc_draws=64,
            seed=trial,
        )
        assert report.score_low is not None and report.score_high is not None
        assert report.score_adjusted_low is not None and report.score_adjusted_high is not None
        truth = (
            100.0
            * sum(_true_equated(ability, items, difficulties) for ability in abilities.values())
            / len(abilities)
        )
        raw_covered += report.score_low <= truth <= report.score_high
        adjusted_covered += report.score_adjusted_low <= truth <= report.score_adjusted_high
        raw_widths.append(report.score_high - report.score_low)
        adjusted_widths.append(report.score_adjusted_high - report.score_adjusted_low)
    return (
        raw_covered / trials,
        adjusted_covered / trials,
        sum(raw_widths) / trials,
        sum(adjusted_widths) / trials,
    )


def _linking_trial(rng: random.Random, regime: str) -> list[float]:
    """Signed equated-rate errors (pp) for one simulated layer."""
    hard = [f"h{index}" for index in range(IRT_LINK_ITEMS)]
    easy = [f"e{index}" for index in range(IRT_LINK_ITEMS)]
    anchors = [f"a{index}" for index in range(IRT_LINK_ANCHORS)]
    reference = hard + easy
    abilities: dict[str, float] = {}
    for group, centre in (("hard", -0.8), ("easy", 0.8)):
        for index in range(IRT_LINK_VIEWS):
            abilities[f"{group}{index}"] = rng.gauss(centre, 0.4)
    difficulties = {item: rng.gauss(-1.5, 0.5) for item in hard}
    difficulties |= {item: rng.gauss(1.5, 0.5) for item in easy}
    difficulties |= {item: rng.gauss(0.0, 0.5) for item in anchors}
    if regime == "shared":
        observations = {stratum: reference + anchors for stratum in abilities}
    elif regime == "partial":
        observations = {
            stratum: (hard if stratum.startswith("hard") else easy) + anchors
            for stratum in abilities
        }
    else:
        observations = {
            stratum: hard if stratum.startswith("hard") else easy for stratum in abilities
        }
    responses = _irt_responses(rng, abilities, difficulties, observations)
    fit = fit_rasch(responses, tol=IRT_TOL)
    rates = equated_rates(fit, reference_items=[("answer", item) for item in reference])
    return [
        100.0 * (rates[(stratum, "answer")] - _true_equated(ability, reference, difficulties))
        for stratum, ability in abilities.items()
    ]


def linking_regimes(
    trials: int = IRT_LINK_TRIALS, seed: int = 0
) -> list[tuple[str, int, float, float]]:
    """Comparability error under shared, partially linked and disjoint banks.

    Two groups of views run a hard and an easy bank; the reference bank is
    their union. ``shared`` lets every view run everything (fully identified),
    ``partial`` links the groups with shared anchors, and ``disjoint`` leaves
    the Gaussian priors to bridge. Each row is ``(regime, shared anchors, bias
    pp, RMSE pp)``: near-zero bias in every regime is expected — weaker linking
    costs variance, not a systematic shift, and ``disjoint`` shows exactly how
    much of the comparability rests on the prior.
    """
    rows = []
    for regime, anchors in (
        ("shared", IRT_LINK_ITEMS),
        ("partial", IRT_LINK_ANCHORS),
        ("disjoint", 0),
    ):
        rng = random.Random(seed)
        errors: list[float] = []
        for _ in range(trials):
            errors.extend(_linking_trial(rng, regime))
        bias = sum(errors) / len(errors)
        rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
        rows.append((regime, anchors, bias, rmse))
    return rows


def _imbalance_trial(rng: random.Random, offset: float) -> list[float]:
    """Raw then equated errors (pp) for every view under one bank offset."""
    anchors = [f"a{index}" for index in range(IRT_IMBALANCE_ANCHORS)]
    own = [f"o{index}" for index in range(IRT_IMBALANCE_ITEMS)]
    abilities = {f"v{index}": rng.gauss(0.0, 0.7) for index in range(IRT_IMBALANCE_VIEWS)}
    anchor_easiness = {item: rng.gauss(0.0, 0.5) for item in anchors}
    own_easiness = {item: rng.gauss(offset, 0.5) for item in own}
    difficulties = anchor_easiness | own_easiness
    observations = {stratum: anchors + own for stratum in abilities}
    responses = _irt_responses(rng, abilities, difficulties, observations)
    fit = fit_rasch(responses, tol=IRT_TOL)
    rates = equated_rates(fit, reference_items=[("answer", item) for item in anchors])
    errors: list[float] = []
    for stratum, ability in abilities.items():
        truth = _true_equated(ability, anchors, anchor_easiness)
        rows = [response for response in responses if response.stratum == stratum]
        raw = sum(response.passed for response in rows) / len(rows)
        errors.extend([100.0 * (raw - truth), 100.0 * (rates[(stratum, "answer")] - truth)])
    return errors


def bank_imbalance(
    trials: int = IRT_IMBALANCE_TRIALS, seed: int = 0
) -> list[tuple[float, float, float]]:
    """Bias of the raw and the equated score as the view's own bank shifts.

    Every view runs the same reference anchors plus an own bank whose items are
    ``offset`` logits easier on average. The true score is the pass rate on the
    anchors, so the raw score inherits the bank's difficulty while the equated
    rate does not. Each row is ``(offset, raw bias pp, adjusted bias pp)``.
    """
    rows = []
    for offset in IMBALANCE_OFFSETS:
        rng = random.Random(seed)
        raw_errors: list[float] = []
        adjusted_errors: list[float] = []
        for _ in range(trials):
            errors = _imbalance_trial(rng, offset)
            raw_errors.extend(errors[0::2])
            adjusted_errors.extend(errors[1::2])
        rows.append(
            (
                offset,
                sum(raw_errors) / len(raw_errors),
                sum(adjusted_errors) / len(adjusted_errors),
            )
        )
    return rows


def _group_layer(rng: random.Random, q: float | None = None, broken: int = 0) -> list[Cell]:
    """A layer of groups; the first ``broken`` switch at ``q`` instead of .5.

    The cells are the same as every other null table's, so a group break is the
    only difference: a shared regression that no single cell is strong enough
    to prove on its own.
    """
    cells: list[Cell] = []
    for g in range(GROUP_GROUPS):
        rate = 0.5 if (g >= broken or q is None) else q
        for c in range(GROUP_CELLS):
            improved, regressed = _draw(rng, GROUP_DISCORDANT, rate)
            cells.append(_cell(f"g{g}c{c}", improved, regressed, group=f"grp{g}"))
    return cells


def _without_groups(cells: list[Cell]) -> list[Cell]:
    return [
        Cell(
            stratum=cell.stratum,
            outcome=cell.outcome,
            improved=cell.improved,
            regressed=cell.regressed,
            passed=cell.passed,
            pairs=cell.pairs,
        )
        for cell in cells
    ]


def group_incidents() -> tuple[float, float, float, list[tuple[float, float, float, float]]]:
    """The group-as-unit gate: false incidents, and power on a shared break.

    Returns ``(null_any_incident, null_incidents_per_layer, null_cell_flags,
    rows)`` where each row is ``(break_rate, independent flags in the broken
    group, incident flags per layer, P(broken group flagged))``. The per-cell
    gate is run on the same draws without the group labels, so the two columns
    differ only in the unit the FDR is computed over.
    """
    null_any = null_incidents = null_cells = 0
    for trial in range(GROUP_NULL_TRIALS):
        model = regression_model(_group_layer(random.Random(9000 + trial)))
        null_any += bool(model.flagged_groups)
        null_incidents += len(model.flagged_groups)
        null_cells += len(model.flagged)
    rows: list[tuple[float, float, float, float]] = []
    for q in GROUP_BREAK_RATES:
        independent = incident_flags = detected = 0
        for trial in range(GROUP_TRIALS):
            cells = _group_layer(random.Random(1000 + trial), q=q, broken=1)
            grouped = regression_model(cells)
            plain = regression_model(_without_groups(cells))
            independent += sum(1 for key in plain.flagged if key[0].startswith("g0c"))
            incident_flags += len(grouped.flagged_groups)
            detected += "grp0" in grouped.flagged_groups
        rows.append(
            (q, independent / GROUP_TRIALS, incident_flags / GROUP_TRIALS, detected / GROUP_TRIALS)
        )
    return (
        null_any / GROUP_NULL_TRIALS,
        null_incidents / GROUP_NULL_TRIALS,
        null_cells / GROUP_NULL_TRIALS,
        rows,
    )


def heterogeneous_never_reads_improving(trials: int = 200, seed: int = 0) -> int:
    """Count layers that are heterogeneous but still report ``improving``."""
    rng = random.Random(seed)
    violations = 0
    for _ in range(trials):
        cells = []
        for index in range(9):
            improved, regressed = _draw(rng, DISCORDANT, 0.62)
            cells.append(_cell(f"v{index}", improved, regressed))
        improved, regressed = _draw(rng, DISCORDANT, 0.15)
        cells.append(_cell("broken", improved, regressed))
        report = build_report(cells, draws=96, ppc_draws=96, seed=seed)
        if report.guardrail.status == "heterogeneous" and report.status == "improving":
            violations += 1
    return violations


def build_markdown() -> str:
    lines = [
        "# Decision calibration",
        "",
        "Generated by `uv run python tools/calibration.py --write`. Seeded, so the",
        f"numbers are reproducible. Null simulations use {TRIALS} trials, "
        f"{DISCORDANT} discordant pairs per cell, and a tier-1 weight of "
        f"{CRITICAL_WEIGHT:g}; the prior table uses {PRIOR_TRIALS} trials.",
        "",
        "The regression gate is a Bayesian spike-and-slab model with a posterior",
        "FDR target of 0.05. The tables below are its **frequentist** operating",
        "characteristics: what happens when the null is actually true, and how",
        "often a real tier-1 regression is caught.",
        "",
        "## False flags under a realistic global null",
        "",
        "Every cell truly switches at 0.5. A quiet gate flags few layers and few",
        "cells per layer.",
        "",
        "| cells | uniform any-flag | uniform flags/cell | weighted any-flag | weighted flags/cell |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in (5, 20, 60):
        u_any, u_per = null_flags(m, weighted=False)
        w_any, w_per = null_flags(m, weighted=True)
        lines.append(f"| {m} | {u_any:.3f} | {u_per:.3f} | {w_any:.3f} | {w_per:.3f} |")
    lines += [
        "",
        "## Power to flag a true tier-1 regression",
        "",
        "One tier-1 cell truly switches at 0.2 among 40 auxiliary cells at 0.5.",
        "Criticality multiplies the cell's prior odds of having regressed, so it",
        "should detect it more often.",
        "",
        "| tier-1 discordant pairs | uniform | weighted |",
        "| ---: | ---: | ---: |",
    ]
    for n_critical in (8, 12, 20):
        uniform = power(n_critical, weighted=False)
        weighted = power(n_critical, weighted=True)
        lines.append(f"| {n_critical} | {uniform:.3f} | {weighted:.3f} |")
    violations = heterogeneous_never_reads_improving()
    null_any, null_incidents, null_cell_flags, group_rows = group_incidents()
    stability_rows = verdict_stability()
    prior_rows = prior_calibration()
    coverage_rows = coverage_calibration()
    irt_coverage = adjusted_interval_coverage()
    linking_rows = linking_regimes()
    imbalance_rows = bank_imbalance()
    lines += [
        "",
        "## Verdict stability as volatility changes",
        "",
        "`rope_pp` is hand-set to 1pp, but the report can size it from the metric's",
        "own history (`calibrate=True`, `rope = k * between_sd` with `k = 1`). Each",
        "row is one layer with a genuine net improvement; the calibrated columns",
        "judge the same layer against a history whose only difference is its",
        "volatility. A volatile metric needs a bigger movement before the verdict",
        "reads `improving`.",
        "",
        "| true improvement (pp) | default | sd=0 | sd=3 | sd=6 | sd=9 |",
        "| ---: | :--- | :--- | :--- | :--- | :--- |",
    ]
    for net_pp, default_status, calibrated in stability_rows:
        cells = " | ".join(calibrated)
        lines.append(f"| {net_pp:.1f} | {default_status} | {cells} |")
    lines += [
        "",
        "A zero `between_sd` means the observed spread is pure sampling noise:",
        "there is no measured dead zone to size, so the 1pp default stands",
        "(`derived` stays empty).",
        "",
        "## Self-calibration of the regression prior",
        "",
        "`pi_prior` is the Beta prior on the share of cells that truly regressed.",
        "The report can re-centre it on the base rate the previous epoch measured",
        "(`pi_mean`), keeping the prior's strength. Each row then measures the",
        "power to flag one true regressor (20% switch rate, 40 auxiliary cells at",
        "0.5) at two evidence levels. A low measured rate tightens the gate; a",
        "high one flags sooner. FDR control is the posterior rule's job, so",
        "re-centring moves the power, not the global null.",
        "",
        "| measured base rate | prior (a, b) | power, 12 pairs | power, 20 pairs |",
        "| ---: | :--- | ---: | ---: |",
    ]
    for rate, a, b, low_evidence, high_evidence in prior_rows:
        lines.append(
            f"| {rate:.2f} | ({a:.2f}, {b:.2f}) | {low_evidence:.3f} | {high_evidence:.3f} |"
        )
    lines += [
        "",
        "## Self-calibration of the coverage gate",
        "",
        "The gate is the history's mean unresolved share plus 5pp, clamped to",
        "[2%, 50%], and it waits for three completed epochs before sizing itself",
        "(early runs are mostly unresolved). Each row resolves it from a history",
        "at the row's mean, then judges an epoch that is 23% unresolved — the",
        "share that passes the hand-set 25% gate.",
        "",
        "| history mean unresolved | derived gate | 23% epoch complete |",
        "| ---: | ---: | :--- |",
    ]
    for rate, gate, complete in coverage_rows:
        lines.append(f"| {rate:.2f} | {gate:.2f} | {str(complete).lower()} |")
    raw_cover, adjusted_cover, raw_width, adjusted_width = irt_coverage
    lines += [
        "",
        "## Difficulty-adjusted scores",
        "",
        "The adjustment fits a Rasch model to the candidate responses and equates",
        "each view onto a common reference bank. Three questions matter for trust:",
        "does its interval cover, how much does weak linking cost, and does it",
        "actually remove the bank's pull?",
        "",
        "### Interval coverage",
        "",
        f"{IRT_COVERAGE_VIEWS} views run the same {IRT_COVERAGE_ITEMS}-item bank, so",
        "the true raw pass rate and the true equated rate are the same population",
        f"quantity. Over {IRT_COVERAGE_TRIALS} seeded epochs:",
        "",
        "| interval | 95% nominal coverage | mean width (pp) |",
        "| :--- | ---: | ---: |",
        f"| raw `score_low`/`score_high` | {raw_cover:.3f} | {raw_width:.1f} |",
        f"| adjusted `score_adjusted_low`/`_high` | {adjusted_cover:.3f} | {adjusted_width:.1f} |",
        "",
        "The adjusted interval tracks the raw one at or above the nominal level,",
        "and is slightly wider: it carries the shared-item component of the",
        "ability uncertainty (views that ran the same items move together), which",
        "holding item easinesses at their mode otherwise leaves out.",
        "",
        "### Cross-view comparability under linking regimes",
        "",
        "Two groups of views run a hard and an easy bank (true easinesses centred",
        "at -1.5 and +1.5); the reference bank is their union. `shared` lets every",
        "view run everything, `partial` links the groups with shared anchor items,",
        "and `disjoint` leaves the priors to bridge. Errors are against the true",
        "equated rate on the union bank, in percentage points.",
        "",
        "| regime | shared anchors | bias (pp) | RMSE (pp) |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for regime, anchors, bias, rmse in linking_rows:
        lines.append(f"| {regime} | {anchors} | {bias:+.2f} | {rmse:.2f} |")
    lines += [
        "",
        "Linking does not bias the equated score in any regime — it buys",
        "precision. Disjoint banks are still fitted (the priors bridge them), but",
        "their comparability error is several times the shared-bank case: this is",
        "the assumption `RaschDiagnostics.connected`/`weakest_link` surfaces, and",
        "the number to quote when someone asks how much a disjoint fit rests on",
        "the prior.",
        "",
        "### Equating removes the bank's pull",
        "",
        f"Every view runs the same {IRT_IMBALANCE_ANCHORS} reference anchors plus an",
        f"own bank of {IRT_IMBALANCE_ITEMS} items that are, on average, `offset`",
        "logits easier. The true score is the pass rate on the anchors; the raw",
        "score inherits the own bank's difficulty, the equated rate does not.",
        "",
        "| own-bank easiness offset | raw bias (pp) | adjusted bias (pp) |",
        "| ---: | ---: | ---: |",
    ]
    for offset, raw_bias, adjusted_bias in imbalance_rows:
        lines.append(f"| {offset:+.1f} | {raw_bias:+.2f} | {adjusted_bias:+.2f} |")
    lines += [
        "",
        "Equating removes most of the raw score's bank bias, though not all of it",
        "at large offsets: the residual is the small-sample (JML) bias of the",
        "ability estimates and the flattening from the Gaussian priors, both of",
        "which shrink toward zero. It is why the raw score and",
        "`difficulty_adjustment_pp` are always reported beside the adjusted one.",
        "",
        "## Shared-dependency incidents",
        "",
        "Cells tagged with the same `group` share an upstream dependency, so the",
        "gate fits the group as **one** test instead of counting its cells as",
        "independent. Under a global null that changes little:",
        f"{null_any:.3f} of {GROUP_NULL_TRIALS} null layers flag any incident",
        f"and {null_incidents:.3f} incidents per layer are flagged",
        f"({null_cell_flags:.3f} per-cell flags). One of",
        f"{GROUP_GROUPS} groups then breaks — every cell in it switching at the",
        "row's rate — and the independent gate's per-cell answer depends on how",
        "hard it broke: a mild break is invisible cell by cell but is one strong",
        "pooled test, while a severe one is a storm of individual flags the",
        "incident unit collapses back to one:",
        "",
        "| broken group's switch rate | per-cell flags (of its cells) | incident flags / layer | P(one incident) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for q, independent, incident_flags, detected in group_rows:
        lines.append(f"| {q:.2f} | {independent:.2f} | {incident_flags:.2f} | {detected:.3f} |")
    lines += [
        "",
        "The `per-cell flags` column runs the gate on the same draws without the",
        f"group labels, over the {GROUP_CELLS} cells of the broken group. Reading",
        "the group as one test both recovers the mild breaks the cells cannot",
        "prove alone and turns a broken dependency into one row and one flag,",
        "however many views it takes down. The per-cell gate still runs on every",
        "cell, so a single regressing view is never hidden by a healthy group.",
        "",
        "## Heterogeneity cannot read as improving",
        "",
        "Across 200 heterogeneous layers (nine views improving, one collapsing),",
        f"the number reporting `improving` instead of `mixed`/`regressing` was **{violations}**.",
        "",
    ]
    return "\n".join(lines)


def _target_path() -> Path:
    """The committed document this script renders."""
    return Path(__file__).resolve().parent.parent / "docs" / "validation.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write docs/validation.md")
    mode.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if docs/validation.md differs from what this would write",
    )
    args = parser.parse_args()
    markdown = build_markdown()
    if args.write:
        target = _target_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown + "\n", encoding="utf-8")
        print(f"wrote {target}")
    elif args.check:
        target = _target_path()
        expected = markdown + "\n"
        actual = target.read_text(encoding="utf-8") if target.exists() else ""
        if actual == expected:
            print(f"{target} is up to date")
            return
        sys.stderr.writelines(
            difflib.unified_diff(
                actual.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile=str(target),
                tofile=f"{target.name} (regenerated)",
            )
        )
        raise SystemExit(
            f"{target} is out of date; run `uv run python tools/calibration.py --write`"
        )
    else:
        print(markdown)


if __name__ == "__main__":
    main()
