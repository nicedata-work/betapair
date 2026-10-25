"""The assembled health report: identity, decomposition, guardrail, JSON."""

from __future__ import annotations

import random
import unittest

from betapair import DEFAULT_CATEGORY, ItemDrift, ItemResponse, ReferenceBank, bank_digest
from betapair.hierarchy import Cell
from betapair.report import (
    COVERAGE_MARGIN,
    COVERAGE_MAX,
    COVERAGE_MIN,
    COVERAGE_MIN_POINTS,
    PI_PRIOR_MAX,
    PI_PRIOR_MIN,
    PI_PRIOR_STRENGTH,
    REPORT_VERSION,
    ROPE_BAND_FACTOR,
    ROPE_MIN_POINTS,
    Calibration,
    CellCoverage,
    HealthReport,
    ReportOptions,
    UnpairedCell,
    _score_interval,
    build_report,
    cell_verdicts,
    delta_status,
    parse_category_weights,
    regression_model,
    resolve_calibration,
    weighting,
)
from betapair.series import ScorePoint

FAST = {"draws": 128, "ppc_draws": 128}


def history_with_sd(sd: float) -> list[ScorePoint]:
    """Three scored epochs whose band has ``between_sd`` exactly ``sd``.

    The stored bounds equal the scores, so the within-epoch variance is zero
    and the band's dispersion is the score sd — a controlled history with no
    sampling-noise component.
    """
    return [
        ScorePoint(score=50.0 - sd, low=50.0 - sd, high=50.0 - sd),
        ScorePoint(score=50.0, low=50.0, high=50.0),
        ScorePoint(score=50.0 + sd, low=50.0 + sd, high=50.0 + sd),
    ]


def coverage_history(shares: list[float]) -> list[ScorePoint]:
    """Scored epochs each carrying an observed unresolved share."""
    return [ScorePoint(score=60.0, pairs=50.0, unresolved_share=share) for share in shares]


def cell(
    name: str,
    improved: int,
    regressed: int,
    pairs: int,
    *,
    outcome: str = "answer",
    category: str | None = None,
    group: str | None = None,
    unchanged_pass: int | None = None,
) -> Cell:
    unchanged = pairs - improved - regressed
    if unchanged_pass is None:
        unchanged_pass = unchanged // 2
    return Cell(
        stratum=name,
        outcome=outcome,
        improved=improved,
        regressed=regressed,
        passed=unchanged_pass + improved,
        pairs=pairs,
        category=category,
        group=group,
    )


class ReportIdentityTest(unittest.TestCase):
    def test_score_gap_equals_the_delta(self) -> None:
        """The paired identity: score - baseline_score == delta_pp, exactly."""
        report = build_report(
            [cell("a", 4, 1, 20), cell("b", 2, 5, 20), cell("c", 0, 3, 10)], **FAST
        )
        assert report.score is not None
        assert report.baseline_score is not None
        assert report.delta_pp is not None
        self.assertAlmostEqual(report.score - report.baseline_score, report.delta_pp, places=9)

    def test_contributions_carry_the_whole_delta(self) -> None:
        cells = [cell("a", 4, 1, 20), cell("b", 2, 5, 20), cell("c", 3, 0, 10)]
        report = build_report(cells, **FAST)
        weighted = sum(
            item.delta_pp * cell_item.pairs
            for item, cell_item in zip(report.contributions, cells, strict=True)
        )
        total_pairs = sum(cell_item.pairs for cell_item in cells)
        assert report.delta_pp is not None
        self.assertAlmostEqual(weighted / total_pairs, report.delta_pp, places=9)

    def test_contributions_are_ranked_by_impact(self) -> None:
        report = build_report(
            [cell("a", 1, 0, 20), cell("b", 6, 1, 20), cell("c", 0, 4, 20)], **FAST
        )
        magnitudes = [abs(item.improved - item.regressed) for item in report.contributions]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_weights_change_the_score(self) -> None:
        cells = [
            cell("a", 5, 0, 20, outcome="answer", unchanged_pass=15),
            cell("b", 0, 5, 20, outcome="spec", unchanged_pass=0),
        ]
        equal = build_report(cells, **FAST)
        answer_only = build_report(cells, weights={"spec": 0.0}, **FAST)
        double_spec = build_report(cells, weights={"spec": 3.0}, **FAST)
        assert equal.score is not None and answer_only.score is not None
        self.assertLess(equal.score, answer_only.score)
        self.assertLess(double_spec.score or 0.0, answer_only.score)

    def test_pending_is_carried_not_counted(self) -> None:
        report = build_report([cell("a", 3, 1, 10)], pending=9, **FAST)
        self.assertEqual(report.pending, 9)
        self.assertEqual(report.pairs, 10)

    def test_different_outcomes_are_separate_cells(self) -> None:
        report = build_report([cell("a", 3, 1, 10), cell("a", 1, 0, 10, outcome="spec")], **FAST)
        self.assertEqual(report.outcomes, ["answer", "spec"])
        self.assertEqual(len(report.contributions), 2)
        self.assertEqual(report.strata, 1)


class CategoryWeightTest(unittest.TestCase):
    """Criticality tiers: weights move the headline, tiers keep their own view."""

    def test_category_weight_scales_the_score(self) -> None:
        cells = [
            cell("critical", 10, 0, 20, category="tier-1", unchanged_pass=10),
            cell("routine", 0, 10, 20, category="tier-3", unchanged_pass=0),
        ]
        plain = build_report(cells, **FAST)
        weighted = build_report(cells, category_weights={"tier-1": 3.0, "tier-3": 1.0}, **FAST)
        self.assertAlmostEqual(plain.score or 0.0, 50.0, places=9)
        self.assertAlmostEqual(weighted.score or 0.0, 75.0, places=9)
        self.assertEqual(weighted.category_weights, {"tier-1": 3.0, "tier-3": 1.0})

    def test_tier_numbers_are_comparable_regardless_of_weight(self) -> None:
        cells = [
            cell("critical", 10, 0, 20, category="tier-1", unchanged_pass=10),
            cell("routine", 0, 10, 20, category="tier-3", unchanged_pass=0),
        ]
        report = build_report(cells, category_weights={"tier-1": 5.0}, **FAST)
        first, second = report.categories
        self.assertEqual(first.category, "tier-1")
        self.assertEqual(first.weight, 5.0)
        self.assertEqual(first.views, 1)
        self.assertEqual(first.pairs, 20)
        self.assertAlmostEqual(first.score or 0.0, 100.0, places=9)
        self.assertAlmostEqual(first.delta_pp or 0.0, 50.0, places=9)
        self.assertEqual(first.status, "improving")
        self.assertGreater(first.p_improved or 0.0, 0.95)
        self.assertEqual(second.category, "tier-3")
        self.assertEqual(second.weight, 1.0)
        self.assertAlmostEqual(second.score or 0.0, 0.0, places=9)
        self.assertEqual(second.status, "regressing")

    def test_tiers_are_ordered_by_weight(self) -> None:
        cells = [
            cell("a", 2, 0, 10, category="tier-3"),
            cell("b", 2, 0, 10, category="tier-1"),
            cell("c", 2, 0, 10, category="tier-2"),
        ]
        report = build_report(
            cells, category_weights={"tier-1": 3.0, "tier-2": 2.0, "tier-3": 1.0}, **FAST
        )
        self.assertEqual(
            [summary.category for summary in report.categories], ["tier-1", "tier-2", "tier-3"]
        )

    def test_contributions_carry_their_tier_and_effective_weight(self) -> None:
        cells = [
            cell("a", 4, 0, 10, outcome="spec", category="tier-1"),
            cell("b", 1, 0, 10, category="uncategorised-tier"),
        ]
        report = build_report(
            cells, weights={"spec": 2.0}, category_weights={"tier-1": 3.0}, **FAST
        )
        by_stratum = {item.stratum: item for item in report.contributions}
        self.assertEqual(by_stratum["a"].category, "tier-1")
        self.assertEqual(by_stratum["a"].weight, 6.0)  # outcome 2 x tier 3
        self.assertEqual(by_stratum["b"].category, "uncategorised-tier")
        self.assertEqual(by_stratum["b"].weight, 1.0)

    def test_zero_weight_tier_is_excluded_but_summarised(self) -> None:
        cells = [
            cell("a", 4, 1, 10, category="tier-1"),
            cell("muted", 1, 4, 10, category="tier-4"),
        ]
        report = build_report(cells, category_weights={"tier-4": 0.0}, **FAST)
        # 4 improved of 10, 2 unchanged passes: 60% — the muted tier contributes 0 weight
        self.assertAlmostEqual(report.score or 0.0, 60.0, places=9)
        muted = next(summary for summary in report.categories if summary.category == "tier-4")
        self.assertIsNone(muted.score)
        self.assertEqual(muted.weight, 0.0)
        self.assertEqual(muted.discordant, 5)

    def test_partial_tiers_default_untagged_views_to_tier_2(self) -> None:
        """Some views tagged, some not: the untagged ones take the dependent tier."""
        cells = [
            cell("critical", 10, 0, 20, category="tier-1", unchanged_pass=10),
            cell("untagged", 0, 10, 20, unchanged_pass=0),
        ]
        report = build_report(
            cells, category_weights={"tier-1": 3.0, "tier-2": 2.0, "tier-3": 1.0}, **FAST
        )
        self.assertTrue(report.tiered)
        self.assertEqual(report.default_category, DEFAULT_CATEGORY)
        # (3 * 20 + 2 * 0) / (3 * 20 + 2 * 20) = 60%
        self.assertAlmostEqual(report.score or 0.0, 60.0, places=9)
        by_stratum = {item.stratum: item for item in report.contributions}
        self.assertEqual(by_stratum["critical"].category, "tier-1")
        self.assertFalse(by_stratum["critical"].defaulted)
        self.assertEqual(by_stratum["critical"].weight, 3.0)
        self.assertEqual(by_stratum["untagged"].category, DEFAULT_CATEGORY)
        self.assertTrue(by_stratum["untagged"].defaulted)
        self.assertEqual(by_stratum["untagged"].weight, 2.0)
        tiers = {summary.category: summary for summary in report.categories}
        self.assertEqual(tiers["tier-2"].defaulted, 1)
        self.assertEqual(tiers["tier-1"].defaulted, 0)

    def test_no_tier_anywhere_scores_uniformly(self) -> None:
        """The documented default: no tags in the comparison means no weighting."""
        cells = [
            cell("a", 10, 0, 20, unchanged_pass=10),
            cell("b", 0, 10, 20, unchanged_pass=0),
        ]
        report = build_report(
            cells, category_weights={"tier-1": 3.0, "tier-2": 2.0, "tier-3": 1.0}, **FAST
        )
        self.assertFalse(report.tiered)
        self.assertAlmostEqual(report.score or 0.0, 50.0, places=9)
        self.assertIsNone(report.categories[0].category)
        self.assertEqual(report.categories[0].defaulted, 0)
        self.assertTrue(all(item.weight == 1.0 for item in report.contributions))
        self.assertTrue(all(not item.defaulted for item in report.contributions))

    def test_defaulting_can_be_disabled(self) -> None:
        cells = [
            cell("critical", 10, 0, 20, category="tier-1", unchanged_pass=10),
            cell("untagged", 0, 10, 20, unchanged_pass=0),
        ]
        report = build_report(
            cells,
            category_weights={"tier-1": 3.0, "tier-2": 2.0},
            default_category=None,
            **FAST,
        )
        untagged = next(item for item in report.contributions if item.stratum == "untagged")
        self.assertIsNone(untagged.category)
        self.assertEqual(untagged.weight, 1.0)
        # (3 * 20 + 1 * 0) / (3 * 20 + 1 * 20) = 75%
        self.assertAlmostEqual(report.score or 0.0, 75.0, places=9)

    def test_uncategorised_cells_form_their_own_group(self) -> None:
        report = build_report([cell("a", 2, 0, 10)], **FAST)
        self.assertEqual(len(report.categories), 1)
        self.assertIsNone(report.categories[0].category)
        self.assertEqual(report.categories[0].weight, 1.0)

    def test_weight_validation(self) -> None:
        with self.assertRaises(ValueError):
            build_report([cell("a", 1, 1, 4)], category_weights={"tier-1": -1.0})


class GroupLabelTest(unittest.TestCase):
    """A group is reported per cell and changes no number on its own."""

    def test_group_is_passed_through_and_not_weighted(self) -> None:
        labelled = [
            cell("a", 4, 1, 10, group="orders_table"),
            cell("b", 1, 4, 10),
        ]
        plain = [cell("a", 4, 1, 10), cell("b", 1, 4, 10)]
        report = build_report(labelled, **FAST)
        baseline = build_report(plain, **FAST)
        by_stratum = {item.stratum: item for item in report.contributions}
        self.assertEqual(by_stratum["a"].group, "orders_table")
        self.assertIsNone(by_stratum["b"].group)
        self.assertEqual(by_stratum["a"].weight, 1.0)
        self.assertAlmostEqual(report.score or 0.0, baseline.score or 0.0, places=9)
        self.assertAlmostEqual(report.delta_pp or 0.0, baseline.delta_pp or 0.0, places=9)
        self.assertEqual(
            [item.verdict for item in report.contributions],
            [item.verdict for item in baseline.contributions],
        )

    def test_weighting_resolves_the_group_per_cell(self) -> None:
        cells = [cell("a", 4, 1, 10, group="orders_table"), cell("b", 1, 4, 10)]
        policy = weighting(cells)
        self.assertEqual(policy.group(cells[0]), "orders_table")
        self.assertIsNone(policy.group(cells[1]))
        # The label never enters the effective weight.
        self.assertEqual(policy.weight(cells[0]), policy.weight(cells[1]))

    def test_unpaired_cell_group_validation(self) -> None:
        self.assertIsNone(UnpairedCell("a", "answer", resolved=1).group)
        self.assertEqual(
            UnpairedCell("a", "answer", resolved=1, group="orders_table").group,
            "orders_table",
        )
        with self.assertRaises(ValueError):
            UnpairedCell("a", "answer", resolved=1, group="")


class GroupIncidentTest(unittest.TestCase):
    """The gate counts a group as one test; a group never hides a cell."""

    def test_a_group_wide_regression_is_one_incident_not_many_flags(self) -> None:
        # Each cell is only mildly regressing (8/12), so no single cell is
        # flagged; pooled, the twelve cells are a clear regression.
        cells = [cell(f"v{index}", 8, 12, 20, group="table_x") for index in range(12)]
        model = regression_model(cells)
        self.assertEqual(model.flagged, frozenset())
        self.assertEqual(model.flagged_groups, frozenset({"table_x"}))
        self.assertGreater(model.group_incident["table_x"], 0.95)
        report = build_report(cells, **FAST)
        self.assertEqual([row.group for row in report.groups], ["table_x"])
        self.assertTrue(report.groups[0].flagged)
        self.assertEqual(report.status, "regressing")

    def test_a_clean_group_is_not_an_incident(self) -> None:
        cells = [cell(f"v{index}", 10, 10, 20, group="clean") for index in range(12)]
        model = regression_model(cells)
        self.assertEqual(model.flagged_groups, frozenset())
        self.assertLess(model.group_incident["clean"], 0.5)

    def test_a_group_cannot_hide_an_individual_regression(self) -> None:
        # Nine cells improving hard, one collapsing: the per-cell flag survives.
        cells = [cell("bad", 0, 20, 20, group="g")]
        cells += [cell(f"ok{index}", 18, 2, 20, group="g") for index in range(9)]
        model = regression_model(cells)
        self.assertIn(("bad", "answer"), model.flagged)
        report = build_report(cells, **FAST)
        bad = next(row for row in report.contributions if row.stratum == "bad")
        self.assertEqual(bad.verdict, "regressing")

    def test_a_flagged_group_implicates_its_cells(self) -> None:
        cells = [cell(f"v{index}", 8, 12, 20, group="table_x") for index in range(12)]
        report = build_report(cells, **FAST)
        self.assertTrue(all(row.verdict == "regressing" for row in report.contributions))

    def test_group_contribution_is_additive(self) -> None:
        cells = [cell(f"a{index}", 8, 12, 20, group="g1") for index in range(3)]
        cells += [cell(f"b{index}", 2, 8, 10, group="g2") for index in range(3)]
        cells.append(cell("c", 6, 4, 10))
        report = build_report(cells, **FAST)
        grouped = sum(row.delta_pp_contribution for row in report.groups)
        ungrouped = sum(
            row.delta_pp_contribution for row in report.contributions if row.group is None
        )
        self.assertAlmostEqual(grouped + ungrouped, report.delta_pp or 0.0, places=9)
        # The incident row expands to its members, each tagged with the group.
        members = [row for row in report.contributions if row.group == "g1"]
        self.assertEqual(len(members), 3)
        g1 = next(row for row in report.groups if row.group == "g1")
        self.assertAlmostEqual(
            sum(row.delta_pp_contribution for row in members),
            g1.delta_pp_contribution,
            places=9,
        )

    def test_no_declared_group_leaves_the_gate_independent(self) -> None:
        cells = [cell("a", 8, 12, 20), cell("b", 2, 8, 10)]
        model = regression_model(cells)
        self.assertEqual(model.group_incident, {})
        self.assertEqual(model.flagged_groups, frozenset())
        self.assertEqual(model.bfdr_groups, 0.0)
        self.assertEqual(build_report(cells, **FAST).groups, [])


class GroupInvariantTest(unittest.TestCase):
    """Cross-checks: tiers and muting never break a group invariant."""

    def test_a_split_tier_group_is_still_one_incident(self) -> None:
        # Half the members are critical (x3) and half auxiliary: the gate still
        # counts the twelve cells as one test, not as twelve or as six.
        cells = [
            cell(
                f"v{index}",
                8,
                12,
                20,
                group="table_x",
                category="critical" if index % 2 else "normal",
            )
            for index in range(12)
        ]
        model = regression_model(
            cells,
            weights={c.key: (3.0 if c.category == "critical" else 1.0) for c in cells},
        )
        self.assertEqual(model.flagged, frozenset())
        self.assertEqual(model.flagged_groups, frozenset({"table_x"}))
        report = build_report(
            cells, category_weights={"critical": 3.0}, default_category=None, **FAST
        )
        self.assertEqual(len(report.groups), 1)
        self.assertTrue(report.groups[0].flagged)
        # The incident weight is the pair-mass mean of the tiers (3 and 1).
        self.assertAlmostEqual(report.groups[0].weight, 2.0, places=9)
        self.assertEqual(report.status, "regressing")
        self.assertTrue(all(row.verdict == "regressing" for row in report.contributions))

    def test_a_low_tier_regression_survives_a_high_tier_group(self) -> None:
        # The layer nets positive, so only invariant 1 can keep this honest.
        cells = [cell("bad", 0, 20, 20, group="g", category="normal")]
        cells += [
            cell(f"ok{index}", 18, 2, 20, group="g", category="critical") for index in range(9)
        ]
        report = build_report(
            cells, category_weights={"critical": 3.0}, default_category=None, **FAST
        )
        assert report.delta_pp is not None
        self.assertGreater(report.delta_pp, 0.0)
        bad = next(row for row in report.contributions if row.stratum == "bad")
        self.assertEqual(bad.verdict, "regressing")
        self.assertEqual(report.status, "regressing")

    def test_a_muted_member_cannot_create_or_receive_an_incident(self) -> None:
        # Three muted views collapse and one unmuted view is flat. If the gate
        # read the muted evidence the group would be flagged and the muted
        # cells implicated; both would contradict "muted everywhere".
        cells = [cell(f"m{index}", 0, 20, 20, group="g", category="muted") for index in range(3)]
        cells.append(cell("ok", 0, 0, 20, group="g", category="normal"))
        weights = {c.key: (0.0 if c.category == "muted" else 1.0) for c in cells}
        model = regression_model(cells, weights=weights)
        self.assertEqual(model.flagged, frozenset())
        self.assertEqual(model.flagged_groups, frozenset())
        self.assertEqual(model.muted, frozenset(c.key for c in cells if c.category == "muted"))
        report = build_report(cells, category_weights={"muted": 0.0}, default_category=None, **FAST)
        self.assertNotEqual(report.status, "regressing")
        muted = [row for row in report.contributions if row.category == "muted"]
        self.assertEqual(len(muted), 3)
        self.assertTrue(all(row.verdict == "stable" for row in muted))
        self.assertTrue(all(row.weight == 0.0 for row in muted))

    def test_a_fully_muted_group_is_not_tested(self) -> None:
        cells = [cell(f"m{index}", 0, 20, 20, group="g", category="muted") for index in range(3)]
        cells.append(cell("ok", 10, 0, 20, category="normal"))
        model = regression_model(
            cells, weights={c.key: 0.0 if c.category == "muted" else 1.0 for c in cells}
        )
        self.assertEqual(model.group_incident, {})
        self.assertEqual(model.flagged_groups, frozenset())
        self.assertEqual(model.bfdr_groups, 0.0)
        report = build_report(cells, category_weights={"muted": 0.0}, default_category=None, **FAST)
        self.assertEqual(len(report.groups), 1)
        self.assertFalse(report.groups[0].flagged)
        self.assertEqual(report.groups[0].p_incident, 0.0)
        self.assertEqual(report.status, "improving")

    def test_group_contributions_are_additive_across_tiers(self) -> None:
        cells = [
            cell(f"a{index}", 8, 12, 20, group="g1", category="critical") for index in range(3)
        ]
        cells += [cell(f"b{index}", 2, 8, 10, group="g2", category="normal") for index in range(3)]
        cells.append(cell("c", 6, 4, 10, category="critical"))
        report = build_report(
            cells,
            category_weights={"critical": 3.0, "normal": 0.5},
            default_category=None,
            **FAST,
        )
        grouped = sum(row.delta_pp_contribution for row in report.groups)
        ungrouped = sum(
            row.delta_pp_contribution for row in report.contributions if row.group is None
        )
        self.assertAlmostEqual(grouped + ungrouped, report.delta_pp or 0.0, places=9)


class CategoryWeightsParsingTest(unittest.TestCase):
    def test_parses_the_config_string(self) -> None:
        self.assertEqual(
            parse_category_weights("tier-1=3, tier-2=1.5"),
            {"tier-1": 3.0, "tier-2": 1.5},
        )
        self.assertEqual(parse_category_weights(" tier-1=2 "), {"tier-1": 2.0})
        self.assertEqual(parse_category_weights(""), {})
        self.assertEqual(parse_category_weights(None), {})

    def test_rejects_malformed_weights(self) -> None:
        for text in ("tier-1", "=3", "tier-1=three", "tier-1=-2"):
            with self.assertRaises(ValueError):
                parse_category_weights(text)


class RegressionMultiplicityTest(unittest.TestCase):
    """The regression flag is a Bayesian FDR decision, not a p-value family."""

    def test_a_realistic_global_null_does_not_mass_flag(self) -> None:
        """Under a genuine null the Bayesian gate stays quiet.

        A pathological all-at-0/4 layer is overwhelming evidence against a
        shared .5 switch rate, so a coherent model flags it; a realistic null
        (each cell's improvements drawn from Bin(20, .5)) is a different story.
        """
        rng = random.Random(0)
        cells = []
        for index in range(40):
            improved = sum(1 for _ in range(20) if rng.random() < 0.5)
            cells.append(cell(f"v{index}", improved, 20 - improved, 20))
        model = regression_model(cells)
        self.assertEqual(len(model.flagged), 0)
        self.assertLessEqual(model.bfdr, 0.05)

    def test_single_cell_flags_only_on_strong_evidence(self) -> None:
        """With one cell the prior is the only multiplicity control."""
        strong = cell_verdicts([cell("only", 0, 10, 20)])
        self.assertEqual(strong[("only", "answer")].verdict, "regressing")
        self.assertGreater(strong[("only", "answer")].p_regressed, 0.95)
        weak = cell_verdicts([cell("only", 0, 4, 20)])
        self.assertEqual(weak[("only", "answer")].verdict, "stable")

    def test_family_wise_false_regressions_stay_rare_under_the_null(self) -> None:
        rng = random.Random(7)
        trials = 200
        false_flags = 0
        for _ in range(trials):
            cells = []
            for index in range(40):
                improved = sum(1 for _ in range(20) if rng.random() < 0.5)
                cells.append(cell(f"v{index}", improved, 20 - improved, 20))
            flagged = [
                result for result in cell_verdicts(cells).values() if result.verdict == "regressing"
            ]
            false_flags += bool(flagged)
        # Uncorrected this would be ~1 - 0.95^40 ~= 87% of trials.
        self.assertLessEqual(false_flags / trials, 0.10, f"false flags: {false_flags}/{trials}")


class ReportStatusTest(unittest.TestCase):
    def test_empty_report_is_insufficient(self) -> None:
        report = build_report([])
        self.assertEqual(report.status, "insufficient")
        self.assertIsNone(report.score)
        self.assertEqual(report.guardrail.status, "insufficient")

    def test_no_discordant_pairs_is_insufficient_not_stable(self) -> None:
        report = build_report([cell("a", 0, 0, 10), cell("b", 0, 0, 8)], **FAST)
        self.assertEqual(report.status, "insufficient")
        self.assertIsNone(report.p_improved)
        self.assertEqual(report.guardrail.status, "insufficient")
        self.assertIn("not enough strata", report.guardrail.reason)
        self.assertIsNotNone(report.delta_pp)
        self.assertAlmostEqual(report.delta_pp, 0.0, places=12)

    def test_homogeneous_improvement(self) -> None:
        report = build_report([cell(f"v{i}", 14, 6, 20) for i in range(10)], **FAST)
        self.assertEqual(report.status, "improving")
        self.assertEqual(report.guardrail.status, "ok")
        assert report.p_improved is not None
        self.assertGreater(report.p_improved, 0.95)
        self.assertTrue(all(item.verdict == "improving" for item in report.contributions))
        assert report.delta_pp_low is not None and report.delta_pp_high is not None
        self.assertGreater(report.delta_pp_high, report.delta_pp_low)

    def test_global_regression(self) -> None:
        report = build_report([cell(f"v{i}", 6, 14, 20) for i in range(10)], **FAST)
        self.assertEqual(report.status, "regressing")
        assert report.p_improved is not None
        self.assertLess(report.p_improved, 0.05)
        self.assertLess(report.delta_pp or 0.0, 0.0)

    def test_one_cell_still_names_its_regression(self) -> None:
        """No homogeneity test with one cell — but the flag and reason survive."""
        report = build_report([cell("only", 0, 10, 12)], **FAST)
        self.assertEqual(report.status, "regressing")
        self.assertEqual(report.guardrail.status, "insufficient")
        self.assertIn("only/answer", report.guardrail.flagged)
        self.assertIn("flagged: only/answer", report.guardrail.reason)

    def test_masked_regression_is_caught_by_the_guardrail(self) -> None:
        """One view collapses while nine improve: the pooled rate is nearly flat."""
        cells = [cell(f"v{i}", 14, 6, 20) for i in range(9)]
        cells.append(cell("broken", 2, 18, 20))
        report = build_report(cells, **FAST)
        assert report.p_improved is not None
        self.assertGreater(report.p_improved, 0.05)  # the pooled test is fooled
        self.assertEqual(report.guardrail.status, "heterogeneous")
        self.assertIn("broken/answer", report.guardrail.flagged)
        self.assertEqual(report.status, "regressing")
        broken = next(item for item in report.contributions if item.stratum == "broken")
        self.assertEqual(broken.verdict, "regressing")
        self.assertLess(broken.delta_pp, 0.0)

    def test_reports_are_deterministic(self) -> None:
        cells = [cell(f"v{i}", 9 + i, 4, 20) for i in range(6)]
        first = build_report(cells, **FAST)
        second = build_report(cells, **FAST)
        self.assertEqual(first.model_dump(), second.model_dump())

    def test_json_round_trip(self) -> None:
        report = build_report([cell("a", 5, 2, 20), cell("b", 1, 3, 12)], **FAST)
        payload = report.model_dump_json()
        restored = HealthReport.model_validate_json(payload)
        self.assertEqual(restored.model_dump(), report.model_dump())

    def test_threshold_validation(self) -> None:
        with self.assertRaises(ValueError):
            build_report([cell("a", 1, 1, 4)], alpha=0.6)
        with self.assertRaises(ValueError):
            build_report([cell("a", 1, 1, 4)], weights={"answer": -1.0})


class WeightedStatusTest(unittest.TestCase):
    """The verdict must come from the same weighted delta the headline reports."""

    def test_status_follows_the_weighted_delta_not_the_unweighted_pool(self) -> None:
        cells = [
            cell("big_aux", 60, 20, 400, outcome="spec", category="tier-3"),
            cell("crit1", 0, 3, 60, outcome="answer", category="tier-1"),
            cell("crit2", 0, 3, 60, outcome="answer", category="tier-1"),
            cell("crit3", 0, 3, 60, outcome="answer", category="tier-1"),
        ]
        report = build_report(cells, category_weights={"tier-1": 5.0, "tier-3": 1.0}, **FAST)
        # The unweighted pooled test is fooled by the huge auxiliary cell...
        assert report.p_improved is not None
        self.assertGreater(report.p_improved, 0.95)
        # ...but the weighted delta is negative, so the status must not say improving.
        assert report.delta_pp is not None
        self.assertLess(report.delta_pp, 0.0)
        self.assertNotEqual(report.status, "improving")

    def test_muted_outcome_cannot_drive_the_status(self) -> None:
        cells = [
            cell("muted", 0, 40, 100, outcome="spec"),
            cell("real", 12, 0, 40, outcome="answer"),
        ]
        report = build_report(cells, weights={"spec": 0.0}, **FAST)
        self.assertEqual(report.status, "improving")
        self.assertEqual(report.guardrail.flagged, [])


class TierRollupTest(unittest.TestCase):
    """A tier's verdict must include the flags inside that tier, and only those."""

    def test_tier_status_includes_its_own_flagged_cell(self) -> None:
        cells = [cell(f"t1_{i}", 9, 1, 20, category="tier-1") for i in range(4)]
        cells.append(cell("t1_bad", 0, 10, 20, category="tier-1"))
        report = build_report(cells, **FAST)
        tier = next(summary for summary in report.categories if summary.category == "tier-1")
        self.assertIn("t1_bad/answer", report.guardrail.flagged)
        self.assertEqual(tier.status, "regressing")

    def test_tier_rollup_ignores_flags_from_other_tiers(self) -> None:
        cells = [
            cell("good1", 9, 1, 20, category="tier-1"),
            cell("good2", 9, 1, 20, category="tier-1"),
            cell("bad", 0, 8, 20, category="tier-3"),
        ]
        report = build_report(cells, **FAST)
        tier = next(summary for summary in report.categories if summary.category == "tier-1")
        self.assertIn("bad/answer", report.guardrail.flagged)
        self.assertEqual(tier.status, "improving")


class BayesianMultiplicityTest(unittest.TestCase):
    """Criticality is a prior-odds multiplier in the spike-and-slab model."""

    def test_criticality_weight_lifts_a_borderline_regression(self) -> None:
        rng = random.Random(0)
        aux = []
        for index in range(40):
            improved = sum(1 for _ in range(20) if rng.random() < 0.5)
            aux.append(cell(f"v{index}", improved, 20 - improved, 20, category="tier-3"))
        critical = cell("critical", 0, 8, 20, category="tier-1")
        plain = build_report([*aux, critical], **FAST)
        weighted = build_report(
            [*aux, critical],
            category_weights={"tier-1": 10.0, "tier-3": 1.0},
            **FAST,
        )
        self.assertNotIn("critical/answer", plain.guardrail.flagged)
        self.assertIn("critical/answer", weighted.guardrail.flagged)

    def test_bfdr_is_within_the_target(self) -> None:
        cells = [cell("a", 0, 8, 20), cell("b", 2, 18, 20)]
        model = regression_model(cells, fdr=0.05)
        self.assertTrue(model.flagged)
        self.assertLessEqual(model.bfdr, 0.05)
        self.assertTrue(0.0 <= model.p_no_regression <= 1.0)


class RopeStatusTest(unittest.TestCase):
    """The ROPE makes stability a positive claim, and keeps it N-sensitive."""

    def test_large_unchanged_bank_is_stable(self) -> None:
        report = build_report([cell("a", 0, 0, 20000), cell("b", 0, 0, 20000)], **FAST)
        self.assertEqual(report.status, "stable")
        self.assertAlmostEqual(report.delta_pp or 0.0, 0.0, places=9)

    def test_small_unchanged_bank_is_not_yet_stable(self) -> None:
        report = build_report([cell("a", 0, 0, 10), cell("b", 0, 0, 8)], **FAST)
        self.assertNotEqual(report.status, "stable")

    def test_delta_status_is_a_rope_decision(self) -> None:
        self.assertEqual(delta_status([-0.5] * 100, rope=0.01, alpha=0.05), "regressing")
        self.assertEqual(delta_status([0.5] * 100, rope=0.01, alpha=0.05), "improving")
        self.assertEqual(
            delta_status([0.5] * 100, rope=0.01, alpha=0.05, heterogeneous=True), "mixed"
        )
        self.assertEqual(delta_status([0.0] * 100, rope=0.01, alpha=0.05), "stable")
        self.assertEqual(
            delta_status([0.5] * 50 + [-0.5] * 50, rope=0.01, alpha=0.05), "insufficient"
        )
        self.assertEqual(delta_status([], rope=0.01, alpha=0.05), "insufficient")
        self.assertEqual(
            delta_status([0.5] * 100, rope=0.01, alpha=0.05, flagged=True), "regressing"
        )


class HeterogeneityStatusTest(unittest.TestCase):
    def test_heterogeneous_improvement_is_mixed(self) -> None:
        cells = [cell(f"t1_{i}", 9, 1, 20, category="tier-1") for i in range(8)]
        cells.append(cell("t1_bad", 0, 6, 20, category="tier-1"))
        report = build_report(cells, **FAST)
        self.assertEqual(report.guardrail.status, "heterogeneous")
        self.assertEqual(report.guardrail.flagged, [])
        self.assertEqual(report.status, "mixed")


class AdditiveContributionTest(unittest.TestCase):
    def test_delta_pp_contributions_sum_to_the_headline(self) -> None:
        cells = [cell("a", 4, 1, 20), cell("b", 2, 5, 20), cell("c", 3, 0, 10)]
        report = build_report(cells, **FAST)
        total = sum(item.delta_pp_contribution for item in report.contributions)
        assert report.delta_pp is not None
        self.assertAlmostEqual(total, report.delta_pp, places=9)

    def test_contributions_are_additive_under_weights(self) -> None:
        cells = [
            cell("critical", 10, 0, 100, category="tier-1"),
            cell("routine", 0, 9, 100, category="tier-3"),
        ]
        report = build_report(cells, category_weights={"tier-1": 3.0, "tier-3": 1.0}, **FAST)
        total = sum(item.delta_pp_contribution for item in report.contributions)
        assert report.delta_pp is not None
        self.assertAlmostEqual(total, report.delta_pp, places=9)

    def test_offsetting_contributions_still_sum_to_the_headline(self) -> None:
        report = build_report([cell("up", 10, 0, 100), cell("down", 0, 10, 100)], **FAST)
        self.assertAlmostEqual(report.delta_pp or 0.0, 0.0, places=9)
        total = sum(item.delta_pp_contribution for item in report.contributions)
        self.assertAlmostEqual(total, 0.0, places=9)


class ThresholdConsistencyTest(unittest.TestCase):
    def test_rope_must_be_non_negative(self) -> None:
        with self.assertRaises(ValueError):
            build_report([cell("a", 8, 2, 20)], rope_pp=-1.0, **FAST)


class ReportOptionsTest(unittest.TestCase):
    """The shared config bundle validates where it is set, not where it is used."""

    def test_options_bundle_matches_keyword_arguments(self) -> None:
        cells = [cell("a", 3, 1, 10), cell("b", 1, 2, 10)]
        self.assertEqual(
            build_report(cells, options=ReportOptions(**FAST)).model_dump(),
            build_report(cells, **FAST).model_dump(),
        )

    def test_options_and_overrides_are_mutually_exclusive(self) -> None:
        with self.assertRaises(TypeError):
            build_report([cell("a", 1, 1, 4)], options=ReportOptions(), alpha=0.1)

    def test_alpha_must_be_in_range(self) -> None:
        with self.assertRaises(ValueError):
            ReportOptions(alpha=0.9)

    def test_rope_must_be_non_negative(self) -> None:
        with self.assertRaises(ValueError):
            ReportOptions(rope_pp=-1.0)

    def test_coverage_threshold_must_be_a_share(self) -> None:
        with self.assertRaises(ValueError):
            ReportOptions(max_unresolved_share=1.5)

    def test_unresolved_share_must_be_a_share(self) -> None:
        with self.assertRaises(ValueError):
            ReportOptions(unresolved_share=2.0)

    def test_reference_bank_and_items_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            ReportOptions(
                reference_items=[("answer", "q1")],
                reference_bank=ReferenceBank("ref", "v1", (("answer", "q1"),)),
            )

    def test_retrospective_window_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            ReportOptions(retrospective_window=0)

    def test_all_muted_weights_take_the_insufficient_path(self) -> None:
        report = build_report([cell("a", 3, 1, 10)], weights={"answer": 0.0}, **FAST)
        self.assertEqual(report.status, "insufficient")
        self.assertEqual(report.guardrail.reason, "all outcome weights are zero")


class CoverageGateTest(unittest.TestCase):
    def test_pending_share_gates_the_status(self) -> None:
        report = build_report(
            [cell("a", 5, 0, 10)],
            coverage=[CellCoverage("a", "answer", paired=10, pending=5)],
            **FAST,
        )
        self.assertFalse(report.coverage.complete)
        self.assertEqual(report.status, "insufficient")

    def test_critical_tier_pending_gates_even_a_tiny_global_share(self) -> None:
        cells = [
            cell("crit", 5, 0, 100, category="tier-1"),
            cell("aux", 5, 0, 100, category="tier-3"),
        ]
        coverage = [
            CellCoverage("crit", "answer", category="tier-1", paired=100, pending=1),
            CellCoverage("aux", "answer", category="tier-3", paired=100, pending=0),
        ]
        report = build_report(
            cells,
            category_weights={"tier-1": 3.0, "tier-3": 1.0},
            coverage=coverage,
            **FAST,
        )
        self.assertEqual(report.coverage.critical_unresolved, 1)
        self.assertFalse(report.coverage.complete)
        self.assertEqual(report.status, "insufficient")

    def test_coverage_never_buries_a_flagged_regression(self) -> None:
        report = build_report(
            [cell("bad", 0, 10, 10)],
            coverage=[CellCoverage("bad", "answer", paired=10, pending=100)],
            **FAST,
        )
        self.assertIn("bad/answer", report.guardrail.flagged)
        self.assertEqual(report.status, "regressing")


class ScoreIntervalTest(unittest.TestCase):
    def test_bounds_bracket_the_point_estimate(self) -> None:
        report = build_report([cell("a", 4, 1, 20)], **FAST)
        assert report.score is not None
        assert report.score_low is not None
        assert report.score_high is not None
        self.assertLess(report.score_low, report.score)
        self.assertLess(report.score, report.score_high)

    def test_matches_the_unweighted_beta_posterior(self) -> None:
        report = build_report([cell("a", 0, 0, 100)], **FAST)
        assert report.score is not None and report.score_low is not None
        point = ScorePoint(score=report.score, pairs=float(report.pairs))
        low, high = point.interval()
        self.assertAlmostEqual(report.score_low, low, places=9)
        assert report.score_high is not None
        self.assertAlmostEqual(report.score_high, high, places=9)

    def test_more_pairs_narrow_the_interval(self) -> None:
        wide = build_report([cell("a", 0, 0, 25)], **FAST)
        narrow = build_report([cell("a", 0, 0, 2500)], **FAST)
        assert wide.score_low is not None and wide.score_high is not None
        assert narrow.score_low is not None and narrow.score_high is not None
        self.assertLess(narrow.score_high - narrow.score_low, wide.score_high - wide.score_low)

    def test_interval_uses_the_weighted_effective_pairs(self) -> None:
        """Under tiers the width follows ``sum(w * pairs)``, not the raw count."""
        cells = [
            cell("crit", 0, 0, 40, category="tier-1"),
            cell("aux", 0, 0, 10, category="tier-3"),
        ]
        report = build_report(cells, category_weights={"tier-1": 3.0, "tier-3": 1.0}, **FAST)
        assert report.score is not None and report.score_low is not None
        weighted = ScorePoint(score=report.score, pairs=130.0).interval()
        raw = ScorePoint(score=report.score, pairs=float(report.pairs)).interval()
        self.assertAlmostEqual(report.score_low, weighted[0], places=9)
        assert report.score_high is not None
        self.assertAlmostEqual(report.score_high, weighted[1], places=9)
        self.assertLess(report.score_high - report.score_low, raw[1] - raw[0])

    def test_interval_collapses_without_pairs(self) -> None:
        self.assertEqual(_score_interval(0.0, 0.0, 0.5, 0.95), (50.0, 50.0))

    def test_unscored_report_carries_no_bounds(self) -> None:
        report = build_report([], **FAST)
        self.assertIsNone(report.score_low)
        self.assertIsNone(report.score_high)

    def test_json_round_trip(self) -> None:
        report = build_report([cell("a", 4, 1, 20)], **FAST)
        restored = HealthReport.model_validate_json(report.model_dump_json())
        self.assertEqual(restored.score_low, report.score_low)
        self.assertEqual(restored.score_high, report.score_high)
        self.assertEqual(restored.version, REPORT_VERSION)


class CalibrationPlumbingTest(unittest.TestCase):
    """B1: history reaches the report, and nothing moves until it is asked to."""

    def test_default_calibration_mirrors_the_hand_set_thresholds(self) -> None:
        report = build_report([cell("a", 4, 1, 20)], **FAST)
        self.assertEqual(
            report.calibration,
            Calibration(rope_pp=1.0, pi_prior=(1.0, 1.0), max_unresolved_share=0.25),
        )
        self.assertFalse(report.calibration.calibrated)
        self.assertEqual(report.rope_pp, report.calibration.rope_pp)

    def test_history_is_counted_and_published(self) -> None:
        history = [ScorePoint(score=60.0, pairs=50.0), ScorePoint(score=62.0, pairs=50.0)]
        report = build_report([cell("a", 4, 1, 20)], history=history, calibrate=True, **FAST)
        self.assertEqual(report.calibration.history_points, 2)
        self.assertTrue(report.calibration.enabled)
        # No threshold is derived yet (B2-B4); the values are still the defaults.
        self.assertFalse(report.calibration.calibrated)
        self.assertEqual(report.calibration.derived, ())

    def test_no_history_is_bit_identical_to_the_old_call(self) -> None:
        """Opt-in: the new argument cannot move an existing number by itself."""
        cells = [cell("a", 4, 1, 20), cell("b", 2, 5, 20)]
        baseline = build_report(cells, **FAST)
        with_history = build_report(cells, history=[], calibrate=False, **FAST)
        self.assertEqual(with_history.model_dump(), baseline.model_dump())

    def test_calibration_is_part_of_the_json(self) -> None:
        history = [ScorePoint(score=60.0, pairs=50.0)]
        report = build_report([cell("a", 4, 1, 20)], history=history, calibrate=True, **FAST)
        restored = HealthReport.model_validate_json(report.model_dump_json())
        self.assertEqual(restored.calibration, report.calibration)
        self.assertEqual(restored.calibration.pi_prior, (1.0, 1.0))

    def test_directly_built_report_still_carries_a_default(self) -> None:
        report = HealthReport(status="insufficient")
        self.assertEqual(report.calibration.rope_pp, 1.0)
        self.assertFalse(report.calibration.enabled)

    def test_resolve_calibration_passes_the_defaults_through(self) -> None:
        resolved = resolve_calibration(
            None,
            calibrate=True,
            rope_pp=2.5,
            pi_prior=(2.0, 3.0),
            max_unresolved_share=0.1,
        )
        self.assertEqual(resolved.rope_pp, 2.5)
        self.assertEqual(resolved.pi_prior, (2.0, 3.0))
        self.assertEqual(resolved.max_unresolved_share, 0.1)
        self.assertEqual(resolved.history_points, 0)
        self.assertTrue(resolved.enabled)
        self.assertFalse(resolved.calibrated)


class RopeCalibrationTest(unittest.TestCase):
    """B2: the ROPE is sized from the metric's measured epoch-to-epoch movement."""

    #: A layer with a genuine +3.0pp improvement over 2000 paired items.
    LAYER = cell("a", 200, 140, 2000)

    def test_volatile_history_widens_the_rope_and_stops_reading_improving(self) -> None:
        baseline = build_report([self.LAYER], **FAST)
        self.assertEqual(baseline.status, "improving")
        self.assertEqual(baseline.rope_pp, 1.0)

        calibrated = build_report(
            [self.LAYER], history=history_with_sd(9.0), calibrate=True, **FAST
        )
        self.assertAlmostEqual(calibrated.rope_pp, ROPE_BAND_FACTOR * 9.0)
        self.assertEqual(calibrated.status, "stable")
        self.assertEqual(calibrated.calibration.derived, ("rope_pp",))
        self.assertEqual(calibrated.calibration.between_sd, 9.0)
        self.assertTrue(calibrated.calibration.calibrated)

    def test_quiet_history_keeps_todays_rope(self) -> None:
        report = build_report([self.LAYER], history=history_with_sd(0.0), calibrate=True, **FAST)
        self.assertEqual(report.rope_pp, 1.0)
        self.assertEqual(report.status, "improving")
        self.assertEqual(report.calibration.derived, ())
        self.assertEqual(report.calibration.between_sd, 0.0)
        self.assertFalse(report.calibration.calibrated)

    def test_calibration_off_leaves_the_default_rope(self) -> None:
        report = build_report([self.LAYER], history=history_with_sd(9.0), **FAST)
        self.assertEqual(report.rope_pp, 1.0)
        self.assertEqual(report.status, "improving")
        self.assertFalse(report.calibration.enabled)
        self.assertFalse(report.calibration.calibrated)

    def test_too_little_history_falls_back(self) -> None:
        report = build_report(
            [self.LAYER],
            history=history_with_sd(9.0)[: ROPE_MIN_POINTS - 1],
            calibrate=True,
            **FAST,
        )
        self.assertEqual(report.rope_pp, 1.0)
        self.assertEqual(report.calibration.derived, ())
        self.assertEqual(report.calibration.history_points, ROPE_MIN_POINTS - 1)

    def test_explicit_rope_is_the_fallback(self) -> None:
        insufficient = build_report([self.LAYER], history=[], calibrate=True, rope_pp=2.0, **FAST)
        self.assertEqual(insufficient.rope_pp, 2.0)
        # Calibration, when it can measure, still wins over the fallback.
        calibrated = build_report(
            [self.LAYER], history=history_with_sd(9.0), calibrate=True, rope_pp=2.0, **FAST
        )
        self.assertAlmostEqual(calibrated.rope_pp, 9.0)

    def test_resolve_calibration_derives_the_rope_from_the_band(self) -> None:
        resolved = resolve_calibration(
            history_with_sd(4.0),
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertAlmostEqual(resolved.rope_pp, ROPE_BAND_FACTOR * 4.0)
        self.assertEqual(resolved.between_sd, 4.0)
        self.assertEqual(resolved.derived, ("rope_pp",))
        self.assertTrue(resolved.calibrated)

    def test_resolve_calibration_without_calibrate_never_reads_the_band(self) -> None:
        resolved = resolve_calibration(
            history_with_sd(9.0),
            calibrate=False,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertEqual(resolved.rope_pp, 1.0)
        self.assertIsNone(resolved.between_sd)
        self.assertEqual(resolved.derived, ())

    def test_negative_rope_is_rejected_even_with_history(self) -> None:
        with self.assertRaises(ValueError):
            build_report(
                [self.LAYER], history=history_with_sd(9.0), calibrate=True, rope_pp=-1.0, **FAST
            )


class ColdStartTest(unittest.TestCase):
    """The first run has no history and no baseline — but still a level."""

    def test_empty_tracker_is_insufficient_with_no_level(self) -> None:
        report = build_report([], calibrate=True, **FAST)
        self.assertEqual(report.status, "insufficient")
        self.assertIsNone(report.score)
        self.assertIsNone(report.score_unpaired)
        self.assertEqual(report.unpaired, 0)
        self.assertEqual(report.calibration.derived, ())
        self.assertTrue(report.calibration.enabled)

    def test_first_paired_run_scores_with_no_calibration_applied(self) -> None:
        """A first paired epoch produces a score, judged with the defaults."""
        report = build_report([cell("a", 3, 1, 10)], calibrate=True, **FAST)
        self.assertEqual(report.score, 60.0)
        self.assertEqual(report.pairs, 10)
        self.assertEqual(report.calibration.derived, ())
        self.assertEqual(report.calibration.rope_pp, 1.0)

    def test_unpaired_level_is_reported_but_never_the_paired_score(self) -> None:
        unpaired = [UnpairedCell("v", "answer", resolved=10, passed=7)]
        report = build_report([], unpaired_cells=unpaired, calibrate=True, **FAST)
        self.assertEqual(report.status, "insufficient")
        self.assertIsNone(report.score)
        self.assertEqual(report.unpaired, 10)
        self.assertEqual(report.score_unpaired, 70.0)
        assert report.score_unpaired_low is not None and report.score_unpaired_high is not None
        self.assertLess(report.score_unpaired_low, 70.0)
        self.assertGreater(report.score_unpaired_high, 70.0)
        # It does not leak into the paired comparison or the verdict.
        self.assertIsNone(report.delta_pp)
        self.assertEqual(report.guardrail.status, "insufficient")

    def test_unpaired_does_not_move_the_paired_report(self) -> None:
        paired = [cell("a", 4, 1, 20)]
        baseline = build_report(paired, **FAST)
        with_cold = build_report(
            paired, unpaired_cells=[UnpairedCell("new", "answer", resolved=5, passed=5)], **FAST
        )
        self.assertEqual(with_cold.score, baseline.score)
        self.assertEqual(with_cold.delta_pp, baseline.delta_pp)
        self.assertEqual(with_cold.status, baseline.status)
        self.assertEqual(with_cold.score_unpaired, 100.0)

    def test_unpaired_level_uses_the_weighting_policy(self) -> None:
        cells = [
            UnpairedCell("crit", "answer", resolved=10, passed=10, category="tier-1"),
            UnpairedCell("aux", "answer", resolved=10, passed=0, category="tier-3"),
        ]
        report = build_report(
            [],
            unpaired_cells=cells,
            category_weights={"tier-1": 3.0, "tier-3": 1.0},
            **FAST,
        )
        self.assertAlmostEqual(report.score_unpaired, 75.0)

    def test_all_muted_unpaired_cells_yield_no_level(self) -> None:
        report = build_report(
            [],
            unpaired_cells=[UnpairedCell("a", "answer", resolved=10, passed=10)],
            weights={"answer": 0.0},
            **FAST,
        )
        self.assertEqual(report.unpaired, 10)
        self.assertIsNone(report.score_unpaired)

    def test_unpaired_cell_validates_its_counts(self) -> None:
        with self.assertRaises(ValueError):
            UnpairedCell("", "answer", resolved=1, passed=1)
        with self.assertRaises(ValueError):
            UnpairedCell("a", "answer", resolved=1, passed=2)
        with self.assertRaises(ValueError):
            UnpairedCell("a", "answer", resolved=-1, passed=0)


class CoverageCalibrationTest(unittest.TestCase):
    """B4: the coverage gate is sized from the epochs' unresolved shares."""

    def test_threshold_is_the_observed_mean_plus_margin(self) -> None:
        resolved = resolve_calibration(
            coverage_history([0.1, 0.1, 0.1]),
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertAlmostEqual(resolved.observed_unresolved_share, 0.1)
        self.assertAlmostEqual(resolved.max_unresolved_share, 0.15)
        self.assertIn("max_unresolved_share", resolved.derived)

    def test_cold_start_guard_waits_for_enough_epochs(self) -> None:
        """The first runs are mostly unresolved; sizing the gate from them
        would gate every early report, so the default stands until there are
        enough completed epochs."""
        resolved = resolve_calibration(
            coverage_history([1.0] * (COVERAGE_MIN_POINTS - 1)),
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertEqual(resolved.max_unresolved_share, 0.25)
        self.assertNotIn("max_unresolved_share", resolved.derived)
        self.assertIsNone(resolved.observed_unresolved_share)

    def test_threshold_is_clamped_to_the_sane_range(self) -> None:
        quiet = resolve_calibration(
            coverage_history([0.0, 0.0, 0.0]),
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertAlmostEqual(quiet.max_unresolved_share, COVERAGE_MARGIN)
        self.assertGreaterEqual(quiet.max_unresolved_share, COVERAGE_MIN)
        stalled = resolve_calibration(
            coverage_history([0.9, 0.9, 0.9]),
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertEqual(stalled.max_unresolved_share, COVERAGE_MAX)

    def test_points_without_a_share_do_not_calibrate_the_gate(self) -> None:
        resolved = resolve_calibration(
            history_with_sd(3.0),
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertEqual(resolved.max_unresolved_share, 0.25)
        self.assertIsNone(resolved.observed_unresolved_share)

    def test_calibration_off_ignores_the_shares(self) -> None:
        resolved = resolve_calibration(
            coverage_history([0.1, 0.1, 0.1]),
            calibrate=False,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertEqual(resolved.max_unresolved_share, 0.25)
        self.assertIsNone(resolved.observed_unresolved_share)

    def test_a_tighter_gate_derived_from_history_actually_fires(self) -> None:
        """The resolved threshold is the one the coverage gate uses."""
        coverage = [CellCoverage("a", "answer", paired=10, pending=3)]
        default = build_report([cell("a", 3, 1, 10)], coverage=coverage, pending=3, **FAST)
        self.assertTrue(default.coverage.complete)
        tightened = build_report(
            [cell("a", 3, 1, 10)],
            coverage=coverage,
            pending=3,
            history=coverage_history([0.0, 0.0, 0.0]),
            calibrate=True,
            **FAST,
        )
        self.assertLess(tightened.calibration.max_unresolved_share, 0.25)
        self.assertFalse(tightened.coverage.complete)
        self.assertEqual(tightened.status, "insufficient")


class PriorCalibrationTest(unittest.TestCase):
    """B3: the regression prior is re-centred on the measured base rate."""

    @staticmethod
    def point(pi_mean: float) -> ScorePoint:
        return ScorePoint(score=60.0, pairs=50.0, pi_mean=pi_mean)

    def test_prior_is_recentred_on_the_measured_base_rate(self) -> None:
        resolved = resolve_calibration(
            [self.point(0.2)],
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertEqual(resolved.observed_pi_mean, 0.2)
        self.assertEqual(resolved.pi_prior, (0.4, 1.6))
        self.assertIn("pi_prior", resolved.derived)
        self.assertAlmostEqual(sum(resolved.pi_prior), PI_PRIOR_STRENGTH)

    def test_the_prior_keeps_its_strength(self) -> None:
        resolved = resolve_calibration(
            [self.point(0.3)],
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertAlmostEqual(resolved.pi_prior[0] + resolved.pi_prior[1], PI_PRIOR_STRENGTH)
        self.assertLess(resolved.pi_prior[0], resolved.pi_prior[1])

    def test_base_rates_are_averaged_over_history(self) -> None:
        resolved = resolve_calibration(
            [self.point(0.1), self.point(0.3)],
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertAlmostEqual(resolved.observed_pi_mean, 0.2)
        self.assertEqual(resolved.pi_prior, (0.4, 1.6))

    def test_prior_mean_cannot_collapse_to_zero_or_one(self) -> None:
        low = resolve_calibration(
            [self.point(0.0)],
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertEqual(low.pi_prior, (PI_PRIOR_MIN * 2.0, (1.0 - PI_PRIOR_MIN) * 2.0))
        self.assertEqual(low.observed_pi_mean, 0.0)
        high = resolve_calibration(
            [self.point(1.0)],
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertEqual(high.pi_prior, (PI_PRIOR_MAX * 2.0, (1.0 - PI_PRIOR_MAX) * 2.0))
        self.assertGreater(high.pi_prior[0], 0.0)
        self.assertGreater(high.pi_prior[1], 0.0)

    def test_history_without_a_base_rate_keeps_the_default_prior(self) -> None:
        resolved = resolve_calibration(
            history_with_sd(9.0),
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertEqual(resolved.pi_prior, (1.0, 1.0))
        self.assertNotIn("pi_prior", resolved.derived)
        self.assertIsNone(resolved.observed_pi_mean)

    def test_calibration_off_ignores_the_base_rate(self) -> None:
        resolved = resolve_calibration(
            [self.point(0.2)],
            calibrate=False,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        self.assertEqual(resolved.pi_prior, (1.0, 1.0))
        self.assertIsNone(resolved.observed_pi_mean)

    def test_report_publishes_the_calibrated_prior(self) -> None:
        report = build_report(
            [cell("a", 4, 1, 20)], history=[self.point(0.2)], calibrate=True, **FAST
        )
        self.assertEqual(report.calibration.pi_prior, (0.4, 1.6))
        self.assertIn("pi_prior", report.calibration.derived)
        # An explicit prior is the fallback when nothing was measured.
        default = build_report([cell("a", 4, 1, 20)], history=[], calibrate=True, **FAST)
        self.assertEqual(default.calibration.pi_prior, (1.0, 1.0))


def _difficulty_banks() -> tuple[list[Cell], list[ItemResponse]]:
    """Two views on a hard and an easy bank, linked by ten shared anchors.

    ``hard`` fails its own ten hard items and passes the anchors; ``easy``
    passes both its own ten easy items and the anchors. Their raw levels are
    0.5 and 1.0, so equality of ability is not what separates them — the bank
    is. The anchors are what let equating tell the difference.
    """
    cells = [
        cell("hard", improved=10, regressed=0, pairs=20, unchanged_pass=0),
        cell("easy", improved=10, regressed=0, pairs=20, unchanged_pass=10),
    ]
    responses = (
        [ItemResponse("hard", "answer", f"anchor{index}", True) for index in range(10)]
        + [ItemResponse("hard", "answer", f"hard{index}", False) for index in range(10)]
        + [ItemResponse("easy", "answer", f"anchor{index}", True) for index in range(10)]
        + [ItemResponse("easy", "answer", f"easy{index}", True) for index in range(10)]
    )
    return cells, responses


class DifficultyAdjustmentTest(unittest.TestCase):
    """The opt-in latent-difficulty adjustment of the headline level."""

    def test_off_by_default_and_inert_when_responses_are_passed(self) -> None:
        cells, responses = _difficulty_banks()
        plain = build_report(cells, **FAST)
        with_responses = build_report(cells, responses=responses, **FAST)
        self.assertIsNone(plain.score_adjusted)
        self.assertIsNone(plain.irt)
        self.assertEqual(plain.model_dump(), with_responses.model_dump())

    def test_responses_are_required_when_adjusting(self) -> None:
        cells, _ = _difficulty_banks()
        with self.assertRaises(ValueError):
            build_report(cells, adjust_difficulty=True, **FAST)
        with self.assertRaises(ValueError):
            build_report(cells, adjust_difficulty=True, responses=[], **FAST)

    def test_draws_must_be_positive(self) -> None:
        cells, responses = _difficulty_banks()
        with self.assertRaises(ValueError):
            build_report(
                cells,
                adjust_difficulty=True,
                responses=responses,
                **{**FAST, "draws": 0},
            )

    def test_a_weighted_cell_must_have_responses(self) -> None:
        _, responses = _difficulty_banks()
        cells = [
            cell("hard", improved=10, regressed=0, pairs=20, unchanged_pass=0),
            cell("spec_view", improved=4, regressed=1, pairs=10, outcome="spec"),
        ]
        with self.assertRaises(ValueError):
            build_report(cells, adjust_difficulty=True, responses=responses, **FAST)

    def test_equating_pulls_two_banks_together(self) -> None:
        cells, responses = _difficulty_banks()
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        rates = {item.stratum: item for item in report.contributions}
        hard, easy = rates["hard"], rates["easy"]
        raw_gap = abs(hard.pass_rate - easy.pass_rate)
        equated_gap = abs((hard.equated_pass_rate or 0.0) - (easy.equated_pass_rate or 0.0))
        self.assertLess(equated_gap, raw_gap)
        # The hard view was measured on a hard bank, so equating raises it;
        # the easy view was flattered by its bank, so equating lowers it.
        self.assertGreater(hard.equated_pass_rate or 0.0, hard.pass_rate)
        self.assertLess(easy.equated_pass_rate or 1.0, easy.pass_rate)

    def test_adjusted_level_and_shift_are_consistent(self) -> None:
        cells, responses = _difficulty_banks()
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        assert report.score is not None
        assert report.score_adjusted is not None
        assert report.difficulty_adjustment_pp is not None
        self.assertAlmostEqual(
            report.difficulty_adjustment_pp, report.score_adjusted - report.score, places=9
        )
        self.assertLessEqual(report.score_adjusted_low or 0.0, report.score_adjusted)
        self.assertGreaterEqual(report.score_adjusted_high or 0.0, report.score_adjusted)
        total = sum(item.weight * item.pairs for item in report.contributions)
        weighted = sum(
            item.weight * item.pairs * (item.difficulty_shift_pp or 0.0)
            for item in report.contributions
        )
        self.assertAlmostEqual(weighted / total, report.difficulty_adjustment_pp, places=9)

    def test_summary_reports_the_linking(self) -> None:
        cells, responses = _difficulty_banks()
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        assert report.irt is not None
        self.assertEqual(report.irt.outcomes, ["answer"])
        self.assertTrue(report.irt.connected)
        self.assertEqual(report.irt.weakest_link, 10)
        self.assertEqual(report.irt.observed_pairs, len(responses))
        self.assertEqual(report.irt.items, 30)

    def test_pinned_reference_bank_changes_what_is_equated(self) -> None:
        cells, responses = _difficulty_banks()
        union = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        pinned = build_report(
            cells,
            adjust_difficulty=True,
            responses=responses,
            reference_items=[("answer", "anchor0")],
            **FAST,
        )
        self.assertIsNotNone(pinned.score_adjusted)
        self.assertNotEqual(union.score_adjusted, pinned.score_adjusted)

    def test_a_pinned_bank_must_cover_every_fitted_outcome(self) -> None:
        _, responses = _difficulty_banks()
        cells = [
            cell("hard", improved=10, regressed=0, pairs=20, unchanged_pass=0),
            cell("spec_view", improved=4, regressed=1, pairs=10, outcome="spec"),
        ]
        responses = [
            *responses,
            ItemResponse("spec_view", "spec", "s0", True),
            ItemResponse("spec_view", "spec", "s1", False),
        ]
        with self.assertRaises(ValueError):
            build_report(
                cells,
                adjust_difficulty=True,
                responses=responses,
                reference_items=[("answer", "anchor0")],
                **FAST,
            )

    def test_duplicate_reference_items_are_rejected(self) -> None:
        cells, responses = _difficulty_banks()
        with self.assertRaises(ValueError):
            build_report(
                cells,
                adjust_difficulty=True,
                responses=responses,
                reference_items=[("answer", "anchor0"), ("answer", "anchor0")],
                **FAST,
            )

    def test_multiple_outcomes_are_fitted_separately(self) -> None:
        cells = [
            cell("hard", improved=10, regressed=0, pairs=20, unchanged_pass=0),
            cell("easy", improved=10, regressed=0, pairs=20, unchanged_pass=10),
            cell("spec_view", improved=4, regressed=1, pairs=10, outcome="spec"),
        ]
        responses = (
            [ItemResponse("hard", "answer", f"anchor{index}", True) for index in range(10)]
            + [ItemResponse("hard", "answer", f"hard{index}", False) for index in range(10)]
            + [ItemResponse("easy", "answer", f"anchor{index}", True) for index in range(10)]
            + [ItemResponse("easy", "answer", f"easy{index}", True) for index in range(10)]
            + [
                ItemResponse("spec_view", "spec", f"s{index}", index % 2 == 0)
                for index in range(10)
            ]
        )
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        assert report.irt is not None
        self.assertEqual(report.irt.outcomes, ["answer", "spec"])
        self.assertEqual(report.irt.observed_pairs, len(responses))
        by_key = {(item.stratum, item.outcome): item for item in report.contributions}
        self.assertIsNotNone(by_key[("spec_view", "spec")].equated_pass_rate)

    def test_a_muted_cell_need_not_have_responses(self) -> None:
        _, responses = _difficulty_banks()
        cells = [
            cell("hard", improved=10, regressed=0, pairs=20, unchanged_pass=0),
            cell("spec_view", improved=4, regressed=1, pairs=10, outcome="spec"),
        ]
        report = build_report(
            cells, weights={"spec": 0.0}, adjust_difficulty=True, responses=responses, **FAST
        )
        by_key = {(item.stratum, item.outcome): item for item in report.contributions}
        self.assertIsNone(by_key[("spec_view", "spec")].equated_pass_rate)
        self.assertIsNotNone(report.score_adjusted)

    def test_a_muted_cell_with_responses_is_skipped_in_the_draws(self) -> None:
        _, responses = _difficulty_banks()
        responses = [
            *responses,
            ItemResponse("spec_view", "spec", "s0", True),
            ItemResponse("spec_view", "spec", "s1", False),
        ]
        cells = [
            cell("hard", improved=10, regressed=0, pairs=20, unchanged_pass=0),
            cell("easy", improved=10, regressed=0, pairs=20, unchanged_pass=10),
            cell("spec_view", improved=4, regressed=1, pairs=10, outcome="spec"),
        ]
        report = build_report(
            cells, weights={"spec": 0.0}, adjust_difficulty=True, responses=responses, **FAST
        )
        without = build_report(cells[:2], adjust_difficulty=True, responses=responses, **FAST)
        self.assertIsNotNone(report.score_adjusted)
        # The muted cell does not touch the draw aggregation or the level.
        self.assertEqual(report.score_adjusted, without.score_adjusted)

    def test_report_json_round_trips_the_adjustment(self) -> None:
        cells, responses = _difficulty_banks()
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        restored = HealthReport.model_validate_json(report.model_dump_json())
        self.assertEqual(restored.score_adjusted, report.score_adjusted)
        self.assertEqual(restored.irt, report.irt)
        self.assertEqual(restored.difficulty_adjustment_pp, report.difficulty_adjustment_pp)
        self.assertEqual(restored.version, REPORT_VERSION)

    def test_the_adjusted_report_is_deterministic(self) -> None:
        cells, responses = _difficulty_banks()
        first = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        second = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        self.assertEqual(first.model_dump_json(), second.model_dump_json())


class ReferenceBankTest(unittest.TestCase):
    """A versioned reference bank fixes what the adjusted score means."""

    def test_union_default_still_publishes_a_digest(self) -> None:
        cells, responses = _difficulty_banks()
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        assert report.irt is not None and report.irt.reference is not None
        reference = report.irt.reference
        self.assertIsNone(reference.name)
        self.assertIsNone(reference.version)
        self.assertEqual(reference.items, reference.available)
        self.assertEqual(reference.digest, bank_digest([row.key for row in responses]))

    def test_a_versioned_bank_is_published(self) -> None:
        cells, responses = _difficulty_banks()
        bank = ReferenceBank.from_responses(responses, name="standard", version="2026-07")
        report = build_report(
            cells, adjust_difficulty=True, responses=responses, reference_bank=bank, **FAST
        )
        assert report.irt is not None and report.irt.reference is not None
        reference = report.irt.reference
        self.assertEqual((reference.name, reference.version), ("standard", "2026-07"))
        self.assertEqual(reference.digest, bank.digest)
        self.assertEqual(reference.items, reference.available)

    def test_a_partially_observed_bank_narrows_and_reports_the_shortfall(self) -> None:
        cells, responses = _difficulty_banks()
        observed = {row.key for row in responses}
        bank = ReferenceBank(name="standard", version="1", items=(*observed, ("answer", "retired")))
        report = build_report(
            cells, adjust_difficulty=True, responses=responses, reference_bank=bank, **FAST
        )
        assert report.irt is not None and report.irt.reference is not None
        reference = report.irt.reference
        self.assertEqual(reference.items, len(bank.items))
        self.assertEqual(reference.available, len(observed))
        self.assertEqual(reference.digest, bank_digest(observed))

    def test_a_bank_with_no_items_for_a_fitted_outcome_is_rejected(self) -> None:
        _, responses = _difficulty_banks()
        cells = [
            cell("hard", improved=10, regressed=0, pairs=20, unchanged_pass=0),
            cell("spec_view", improved=4, regressed=1, pairs=10, outcome="spec"),
        ]
        responses = [
            *responses,
            ItemResponse("spec_view", "spec", "s0", True),
            ItemResponse("spec_view", "spec", "s1", False),
        ]
        answer_only = [row for row in responses if row.outcome == "answer"]
        bank = ReferenceBank.from_responses(answer_only, name="standard", version="1")
        with self.assertRaises(ValueError):
            build_report(
                cells, adjust_difficulty=True, responses=responses, reference_bank=bank, **FAST
            )

    def test_a_bank_whose_items_were_never_observed_is_rejected(self) -> None:
        cells, responses = _difficulty_banks()
        bank = ReferenceBank(name="standard", version="1", items=(("answer", "ghost"),))
        with self.assertRaises(ValueError):
            build_report(
                cells, adjust_difficulty=True, responses=responses, reference_bank=bank, **FAST
            )

    def test_a_bank_and_raw_items_are_mutually_exclusive(self) -> None:
        cells, responses = _difficulty_banks()
        bank = ReferenceBank.from_responses(responses, name="standard", version="1")
        with self.assertRaises(ValueError):
            build_report(
                cells,
                adjust_difficulty=True,
                responses=responses,
                reference_bank=bank,
                reference_items=list(bank.items),
                **FAST,
            )

    def test_the_bank_identity_round_trips_through_json(self) -> None:
        cells, responses = _difficulty_banks()
        bank = ReferenceBank.from_responses(responses, name="standard", version="1")
        report = build_report(
            cells, adjust_difficulty=True, responses=responses, reference_bank=bank, **FAST
        )
        restored = HealthReport.model_validate_json(report.model_dump_json())
        self.assertEqual(restored.irt, report.irt)
        assert restored.irt is not None and restored.irt.reference is not None
        self.assertEqual(restored.irt.reference.name, "standard")


class DriftReportTest(unittest.TestCase):
    """Content-fingerprint drift is reported, never gated or re-keyed."""

    def test_no_drift_is_empty(self) -> None:
        self.assertEqual(build_report([cell("a", 4, 1, 20)], **FAST).drift, [])

    def test_drift_is_sorted_and_deduplicated(self) -> None:
        later = ItemDrift("answer", "q2", "fp1_a", "fp1_b")
        first = ItemDrift("answer", "q1", "fp1_c", "fp1_d")
        report = build_report([cell("a", 4, 1, 20)], drift=[later, first, later], **FAST)
        self.assertEqual(report.drift, [first, later])
        # Reported, not gated: the level, delta and verdict are unchanged.
        baseline = build_report([cell("a", 4, 1, 20)], **FAST)
        self.assertEqual(report.score, baseline.score)
        self.assertEqual(report.delta_pp, baseline.delta_pp)
        self.assertEqual(report.status, baseline.status)

    def test_json_round_trip_keeps_drift(self) -> None:
        row = ItemDrift("answer", "q1", "fp1_a", "fp1_b")
        report = build_report([cell("a", 4, 1, 20)], drift=[row], **FAST)
        restored = HealthReport.model_validate_json(report.model_dump_json())
        self.assertEqual(restored.drift, [row])
        self.assertEqual(restored.version, REPORT_VERSION)


def _disjoint_banks() -> tuple[list[Cell], list[ItemResponse]]:
    """Two views on banks that share no item, so linking is prior-only."""
    cells = [
        cell("a", improved=10, regressed=0, pairs=20, unchanged_pass=0),
        cell("b", improved=10, regressed=0, pairs=20, unchanged_pass=10),
    ]
    responses = [
        ItemResponse("a", "answer", f"a{index}", index % 2 == 0) for index in range(20)
    ] + [ItemResponse("b", "answer", f"b{index}", index % 2 == 0) for index in range(20)]
    return cells, responses


class SharedItemsReportTest(unittest.TestCase):
    """C3: zero shared items must be stated, not left implicit."""

    def test_linked_banks_report_the_overlap(self) -> None:
        cells, responses = _difficulty_banks()
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        assert report.irt is not None
        self.assertEqual(report.irt.shared_items, 10)
        self.assertFalse(report.irt.prior_only)
        self.assertTrue(report.irt.connected)

    def test_disjoint_banks_say_the_comparison_is_prior_only(self) -> None:
        cells, responses = _disjoint_banks()
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        assert report.irt is not None
        self.assertEqual(report.irt.shared_items, 0)
        self.assertTrue(report.irt.prior_only)
        self.assertFalse(report.irt.connected)
        self.assertEqual(report.irt.weakest_link, 0)

    def test_a_single_view_is_not_prior_only(self) -> None:
        only = cell("only", improved=10, regressed=0, pairs=20, unchanged_pass=5)
        responses = [
            ItemResponse("only", "answer", f"q{index}", index % 2 == 0) for index in range(20)
        ]
        report = build_report([only], adjust_difficulty=True, responses=responses, **FAST)
        assert report.irt is not None
        self.assertEqual(report.irt.shared_items, 0)
        self.assertFalse(report.irt.prior_only)

    def test_the_linking_diagnostic_round_trips_json(self) -> None:
        cells, responses = _disjoint_banks()
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        restored = HealthReport.model_validate_json(report.model_dump_json())
        self.assertEqual(restored.irt, report.irt)
        self.assertEqual(restored.version, REPORT_VERSION)
        assert restored.irt is not None
        self.assertTrue(restored.irt.prior_only)


class RetrospectiveReportTest(unittest.TestCase):
    """The windowed read is opt-in, additive, and never overrides the gate."""

    @staticmethod
    def _history() -> list[ScorePoint]:
        return [
            ScorePoint(score=40.0 + 2 * index, pairs=5000.0, epoch=f"e{index}")
            for index in range(6)
        ]

    def test_off_by_default(self) -> None:
        report = build_report([cell("a", 4, 1, 20)], history=self._history(), **FAST)
        self.assertIsNone(report.retrospective)

    def test_a_window_is_attached_when_asked(self) -> None:
        report = build_report(
            [cell("a", 4, 1, 20)], history=self._history(), retrospective_window=6, **FAST
        )
        assert report.retrospective is not None
        self.assertEqual(report.retrospective.points, 6)
        self.assertEqual(report.retrospective.status, "improving")
        self.assertEqual(report.retrospective.window, 6)

    def test_an_empty_history_yields_none(self) -> None:
        report = build_report([cell("a", 4, 1, 20)], history=[], retrospective_window=3, **FAST)
        self.assertIsNone(report.retrospective)

    def test_a_bad_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_report(
                [cell("a", 4, 1, 20)], history=self._history(), retrospective_window=0, **FAST
            )

    def test_a_windowed_stable_does_not_hide_a_flagged_regression(self) -> None:
        history = [ScorePoint(score=50.0, pairs=5000.0) for _ in range(4)]
        report = build_report(
            [cell("bad", 0, 20, 20)], history=history, retrospective_window=4, **FAST
        )
        assert report.retrospective is not None
        self.assertEqual(report.retrospective.status, "stable")
        self.assertEqual(report.status, "regressing")

    def test_json_round_trip_keeps_the_window(self) -> None:
        report = build_report(
            [cell("a", 4, 1, 20)], history=self._history(), retrospective_window=4, **FAST
        )
        restored = HealthReport.model_validate_json(report.model_dump_json())
        self.assertEqual(restored.retrospective, report.retrospective)
        self.assertEqual(restored.version, REPORT_VERSION)


class TimelineCompositionTest(unittest.TestCase):
    """A timeline point carries the bank composition behind its score."""

    def test_from_report_carries_counts_and_the_held_level(self) -> None:
        cells = [cell("a", 4, 1, 20)]
        coverage = [CellCoverage(stratum="a", outcome="answer", paired=20)]
        report = build_report(cells, coverage=coverage, retired=3, failed=2, **FAST)
        point = ScorePoint.from_report(report)
        assert point is not None
        self.assertEqual(point.paired, 20)
        self.assertEqual(point.retired, 3)
        self.assertEqual(point.failed, 2)
        self.assertEqual(point.score_held, report.score_held)
        self.assertIsNone(point.bank)

    def test_an_unscored_report_is_still_not_a_point(self) -> None:
        self.assertIsNone(ScorePoint.from_report(build_report([], **FAST)))


class SharedAbilityReportTest(unittest.TestCase):
    """F1: one ability per view across outcomes, items stay outcome-specific."""

    @staticmethod
    def _banks() -> tuple[list[Cell], list[ItemResponse]]:
        rng = random.Random(9)
        responses = []
        for view in ("a", "b"):
            responses += [
                ItemResponse(view, "answer", f"a{index}", rng.random() < 0.6) for index in range(10)
            ]
            responses += [
                ItemResponse(view, "spec", f"s{index}", rng.random() < 0.4) for index in range(3)
            ]
        cells = [
            cell("a", 6, 4, 10, outcome="answer"),
            cell("b", 6, 4, 10, outcome="answer"),
            cell("a", 1, 2, 3, outcome="spec"),
            cell("b", 1, 2, 3, outcome="spec"),
        ]
        return cells, responses

    def test_one_fit_covers_every_outcome(self) -> None:
        cells, responses = self._banks()
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        assert report.irt is not None
        self.assertEqual(report.irt.outcomes, ["answer", "spec"])
        self.assertEqual(report.irt.observed_pairs, len(responses))
        self.assertEqual(report.irt.items, 13)
        self.assertIsNotNone(report.score_adjusted)

    def test_every_cell_gets_an_equated_rate(self) -> None:
        cells, responses = self._banks()
        report = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        equated = {
            (row.stratum, row.outcome)
            for row in report.contributions
            if row.equated_pass_rate is not None
        }
        self.assertEqual(equated, {cell.key for cell in cells})

    def test_a_pinned_bank_still_resolves_per_outcome(self) -> None:
        cells, responses = self._banks()
        bank = ReferenceBank.from_responses(responses, name="std", version="1")
        report = build_report(
            cells,
            adjust_difficulty=True,
            responses=responses,
            reference_bank=bank,
            **FAST,
        )
        assert report.irt is not None and report.irt.reference is not None
        self.assertEqual(report.irt.reference.available, 13)
        self.assertEqual(report.irt.reference.items, 13)

    def test_a_pin_naming_an_unfitted_item_is_rejected(self) -> None:
        cells, responses = self._banks()
        with self.assertRaises(ValueError):
            build_report(
                cells,
                adjust_difficulty=True,
                responses=responses,
                reference_items=[("answer", "missing")],
                **FAST,
            )

    def test_the_shared_fit_is_deterministic(self) -> None:
        cells, responses = self._banks()
        first = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        second = build_report(cells, adjust_difficulty=True, responses=responses, **FAST)
        self.assertEqual(first.model_dump_json(), second.model_dump_json())


if __name__ == "__main__":
    unittest.main()
