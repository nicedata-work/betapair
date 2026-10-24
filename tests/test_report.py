"""The assembled health report: identity, decomposition, guardrail, JSON."""

from __future__ import annotations

import random
import unittest

from betapair import DEFAULT_CATEGORY
from betapair.hierarchy import Cell
from betapair.report import (
    CellCoverage,
    HealthReport,
    build_report,
    cell_verdicts,
    delta_status,
    parse_category_weights,
    regression_model,
)

FAST = {"draws": 128, "ppc_draws": 128}


def cell(
    name: str,
    improved: int,
    regressed: int,
    pairs: int,
    *,
    outcome: str = "answer",
    category: str | None = None,
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


if __name__ == "__main__":
    unittest.main()
