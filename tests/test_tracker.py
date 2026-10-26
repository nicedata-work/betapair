"""Streaming accumulation: partial refreshes, rollover, merges, snapshots."""

from __future__ import annotations

import json
import random
import unittest

from betapair import ERROR, FAILED, RETIRED, HealthTracker, PairRecord, ReferenceBank, ScorePoint

FAST = {"draws": 96, "ppc_draws": 96}


def records() -> list[PairRecord]:
    """12 items across two views and two outcomes, with a real mix of verdicts."""
    rng = random.Random(5)
    out: list[PairRecord] = []
    for view in ("sv_orders", "sv_customers"):
        for outcome in ("answer", "spec"):
            for index in range(3):
                item = f"q{index}"
                out.append(
                    PairRecord(
                        stratum=view,
                        outcome=outcome,
                        item=item,
                        baseline=rng.random() < 0.5,
                        candidate=rng.random() < 0.6,
                    )
                )
    return out


class StreamingTest(unittest.TestCase):
    def test_batch_and_streaming_agree(self) -> None:
        payload = records()
        batch = HealthTracker(seed=0, **FAST)
        batch.observe_many(payload)
        streamed = HealthTracker(seed=0, **FAST)
        for record in reversed(payload):
            streamed.observe(
                record.stratum, record.outcome, record.item, record.baseline, record.candidate
            )
        self.assertEqual(batch.report().model_dump(), streamed.report().model_dump())

    def test_repeated_observation_is_idempotent(self) -> None:
        """At-least-once delivery must not double-count a question."""
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        first = tracker.report()
        tracker.observe("v", "answer", "q1", False, True)
        second = tracker.report()
        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertEqual(len(tracker), 1)
        self.assertEqual(tracker.paired, 1)

    def test_re_evaluation_overwrites_the_item(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, False)
        tracker.observe("v", "answer", "q1", False, True)
        report = tracker.report()
        self.assertEqual(report.contributions[0].improved, 1)

    def test_partial_refresh_excludes_pending_items(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", False, None)  # not evaluated yet
        tracker.observe("v", "answer", "q3", None, True)  # new question, no baseline
        report = tracker.report()
        self.assertEqual(report.pairs, 1)
        self.assertEqual(report.pending, 2)
        self.assertEqual(tracker.pending, 2)
        assert report.score is not None
        self.assertAlmostEqual(report.score, 100.0, places=9)

    def test_report_on_empty_tracker(self) -> None:
        tracker = HealthTracker(**FAST)
        self.assertEqual(tracker.report().status, "insufficient")
        self.assertEqual(len(tracker), 0)


class EpochRolloverTest(unittest.TestCase):
    def test_advance_promotes_candidate_to_baseline(self) -> None:
        tracker = HealthTracker(**FAST, candidate_epoch="2026-07-01")
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", True, True)
        tracker.observe("v", "answer", "q3", False, None)
        tracker.advance(candidate_epoch="2026-07-02")
        self.assertEqual(tracker.baseline_epoch, "2026-07-01")
        self.assertEqual(tracker.candidate_epoch, "2026-07-02")
        self.assertEqual(tracker.paired, 0)
        self.assertEqual(tracker.pending, 2)  # q1, q2 keep their new baseline
        self.assertEqual(len(tracker), 2)  # q3 had nothing to roll
        self.assertEqual(tracker.report().status, "insufficient")

        tracker.observe("v", "answer", "q1", True, False)
        tracker.observe("v", "answer", "q2", True, True)
        report = tracker.report()
        self.assertEqual(report.pairs, 2)
        self.assertEqual(report.baseline_epoch, "2026-07-01")
        self.assertEqual(report.candidate_epoch, "2026-07-02")
        self.assertEqual(report.contributions[0].regressed, 1)

    def test_advance_without_candidate_reports_empty(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, None)
        tracker.advance()
        self.assertEqual(len(tracker), 0)


class MergeTest(unittest.TestCase):
    def test_merge_fills_gaps_and_prefers_the_fresher_side(self) -> None:
        primary = HealthTracker(**FAST)
        primary.observe("v", "answer", "q1", False, None)
        primary.observe("v", "answer", "q2", False, False)
        secondary = HealthTracker(**FAST)
        secondary.observe("v", "answer", "q1", False, True)
        secondary.observe("v", "answer", "q2", False, True)
        secondary.observe("v", "answer", "q3", True, True)

        primary.merge(secondary)
        self.assertEqual(primary.paired, 3)
        self.assertEqual(primary.pending, 0)
        report = primary.report()
        self.assertEqual(report.contributions[0].improved, 2)
        self.assertEqual(len(primary), 3)

    def test_merge_prefer_self_keeps_local_verdicts(self) -> None:
        primary = HealthTracker(**FAST)
        primary.observe("v", "answer", "q1", False, False)
        secondary = HealthTracker(**FAST)
        secondary.observe("v", "answer", "q1", False, True)
        primary.merge(secondary, prefer="self")
        self.assertEqual(primary.report().contributions[0].improved, 0)


class CategoryTrackerTest(unittest.TestCase):
    def test_categories_are_bound_per_view_and_weighted(self) -> None:
        tracker = HealthTracker(
            **FAST,
            categories={"critical": "tier-1"},
            category_weights={"tier-1": 3.0},
        )
        tracker.observe("critical", "answer", "q1", False, True)
        tracker.observe("critical", "answer", "q2", False, True)
        tracker.observe("routine", "answer", "q1", False, True)
        tracker.observe("routine", "answer", "q2", False, False)

        cells = {cell.stratum: cell for cell in tracker.cells()}
        self.assertEqual(cells["critical"].category, "tier-1")
        self.assertIsNone(cells["routine"].category)
        report = tracker.report()
        self.assertEqual(report.category_weights, {"tier-1": 3.0})
        self.assertAlmostEqual(report.score or 0.0, 87.5, places=9)  # (3*2 + 1*1) / (3*2 + 1*2)
        # the untagged view falls back to tier-2 once any view is tagged
        self.assertEqual([summary.category for summary in report.categories], ["tier-1", "tier-2"])

    def test_untagged_views_default_to_tier_2_once_tiers_exist(self) -> None:
        tracker = HealthTracker(
            weights={"answer": 1.0},
            categories={"critical": "tier-1"},
            category_weights={"tier-1": 3.0, "tier-2": 2.0, "tier-3": 1.0},
            **FAST,
        )
        tracker.observe("critical", "answer", "q1", False, True)
        tracker.observe("critical", "answer", "q2", False, True)
        tracker.observe("routine", "answer", "q1", False, True)
        tracker.observe("routine", "answer", "q2", False, False)
        report = tracker.report()
        self.assertTrue(report.tiered)
        # (3 * 2 + 2 * 1) / (3 * 2 + 2 * 2) = 8/10
        self.assertAlmostEqual(report.score or 0.0, 80.0, places=9)
        routine = next(item for item in report.contributions if item.stratum == "routine")
        self.assertEqual(routine.category, "tier-2")
        self.assertTrue(routine.defaulted)

    def test_default_category_is_configurable_and_persisted(self) -> None:
        tracker = HealthTracker(
            categories={"critical": "tier-1"},
            category_weights={"tier-1": 3.0, "tier-3": 1.0},
            default_category="tier-3",
            **FAST,
        )
        tracker.observe("critical", "answer", "q1", False, True)
        tracker.observe("routine", "answer", "q1", False, True)
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertEqual(restored.default_category, "tier-3")
        routine = next(
            item for item in restored.report().contributions if item.stratum == "routine"
        )
        self.assertEqual(routine.category, "tier-3")
        self.assertTrue(routine.defaulted)  # fell back to the configured tier

    def test_a_category_can_be_bound_after_the_fact(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        self.assertIsNone(tracker.category_of("v"))
        tracker.set_category("v", "tier-2")
        self.assertEqual(tracker.category_of("v"), "tier-2")
        tracker.set_category("v", None)
        self.assertIsNone(tracker.category_of("v"))

    def test_categories_survive_a_snapshot(self) -> None:
        tracker = HealthTracker(
            categories={"v": "tier-1"}, category_weights={"tier-1": 2.0}, **FAST
        )
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", True, False)
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertEqual(restored.categories, {"v": "tier-1"})
        self.assertEqual(restored.category_weights, {"tier-1": 2.0})
        self.assertEqual(restored.report().model_dump(), tracker.report().model_dump())

    def test_merge_keeps_the_first_bound_category(self) -> None:
        primary = HealthTracker(**FAST)
        primary.observe("v", "answer", "q1", False, True)
        secondary = HealthTracker(categories={"v": "tier-1"}, category_weights={"tier-1": 2.0})
        primary.merge(secondary)
        self.assertEqual(primary.category_of("v"), "tier-1")
        self.assertEqual(primary.category_weights, {"tier-1": 2.0})

    def test_invalid_category_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HealthTracker(category_weights={"tier-1": -1.0})


class GroupTrackerTest(unittest.TestCase):
    """A group is an incident label: it never moves the score or the delta."""

    def test_a_group_is_carried_and_only_adds_incidents(self) -> None:
        labelled = HealthTracker(seed=0, **FAST, groups={"orders_a": "orders_table"})
        plain = HealthTracker(seed=0, **FAST)
        for tracker in (labelled, plain):
            tracker.observe("orders_a", "answer", "q1", False, True)
            tracker.observe("orders_a", "answer", "q2", True, False)
            tracker.observe("orders_b", "answer", "q1", False, True)
        cells = {cell.stratum: cell for cell in labelled.cells()}
        self.assertEqual(cells["orders_a"].group, "orders_table")
        self.assertIsNone(cells["orders_b"].group)
        report = labelled.report()
        baseline = plain.report()
        contribution = next(row for row in report.contributions if row.stratum == "orders_a")
        self.assertEqual(contribution.group, "orders_table")
        # Aggregation and per-cell gate are untouched: a group is not a weight.
        self.assertEqual(report.score, baseline.score)
        self.assertEqual(report.delta_pp, baseline.delta_pp)
        self.assertEqual(report.pairs, baseline.pairs)
        self.assertEqual(
            [row.p_regressed for row in report.contributions],
            [row.p_regressed for row in baseline.contributions],
        )
        # The group appears only as the incident roll-up.
        self.assertEqual([row.group for row in report.groups], ["orders_table"])
        self.assertEqual(baseline.groups, [])

    def test_a_group_can_be_bound_after_the_fact(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        self.assertIsNone(tracker.group_of("v"))
        tracker.set_group("v", "orders_table")
        self.assertEqual(tracker.group_of("v"), "orders_table")
        self.assertEqual(tracker.cells()[0].group, "orders_table")
        tracker.set_group("v", None)
        self.assertIsNone(tracker.group_of("v"))

    def test_groups_survive_a_snapshot(self) -> None:
        tracker = HealthTracker(groups={"v": "orders_table"}, **FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", True, False)
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertEqual(restored.groups, {"v": "orders_table"})
        self.assertEqual(restored.report().model_dump(), tracker.report().model_dump())

    def test_merge_keeps_the_first_bound_group(self) -> None:
        primary = HealthTracker(**FAST)
        primary.observe("v", "answer", "q1", False, True)
        secondary = HealthTracker(groups={"v": "orders_table"})
        primary.merge(secondary)
        self.assertEqual(primary.group_of("v"), "orders_table")

    def test_merge_prefer_other_takes_the_other_group(self) -> None:
        primary = HealthTracker(groups={"v": "orders_table"})
        secondary = HealthTracker(groups={"v": "customers_table"})
        primary.merge(secondary)
        self.assertEqual(primary.group_of("v"), "customers_table")

    def test_coverage_and_unpaired_cells_carry_the_group(self) -> None:
        tracker = HealthTracker(groups={"v": "orders_table"}, **FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", None, True)  # candidate only
        self.assertEqual(tracker.coverage()[0].group, "orders_table")
        self.assertEqual(tracker.unpaired_cells()[0].group, "orders_table")

    def test_an_empty_group_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HealthTracker(groups={"v": ""})
        tracker = HealthTracker(**FAST)
        with self.assertRaises(ValueError):
            tracker.set_group("v", "")

    def test_malformed_snapshot_group_is_rejected(self) -> None:
        for payload in (
            {"version": 3, "records": [], "groups": "orders_table"},
            {"version": 3, "records": [], "groups": {"v": 3}},
            {"version": 3, "records": [], "groups": {"v": ""}},
        ):
            with self.assertRaises(ValueError):
                HealthTracker.from_snapshot(payload)


class SnapshotTest(unittest.TestCase):
    def test_json_round_trip_preserves_state_and_report(self) -> None:
        tracker = HealthTracker(
            **FAST,
            weights={"answer": 1.0, "spec": 0.5},
            baseline_epoch="2026-06-30",
            candidate_epoch="2026-07-01",
        )
        tracker.observe_many(records())
        text = tracker.to_json()
        restored = HealthTracker.from_json(text, **FAST)
        self.assertEqual(restored.snapshot(), tracker.snapshot())
        self.assertEqual(restored.report().model_dump(), tracker.report().model_dump())
        self.assertEqual(restored.weights, tracker.weights)
        self.assertEqual(restored.baseline_epoch, "2026-06-30")

    def test_snapshot_is_deterministic_and_sorted(self) -> None:
        first = HealthTracker(seed=0)
        first.observe_many(records())
        second = HealthTracker(seed=0)
        second.observe_many(list(reversed(records())))
        self.assertEqual(first.to_json(), second.to_json())
        payload = json.loads(first.to_json())
        self.assertEqual(payload["version"], 3)
        self.assertEqual(len(payload["records"]), 12)

    def test_snapshot_survives_a_json_round_trip_through_text(self) -> None:
        tracker = HealthTracker()
        tracker.observe("v", "answer", "q1", None, False)
        restored = HealthTracker.from_snapshot(json.loads(tracker.to_json()))
        self.assertEqual(restored.pending, 1)
        self.assertIsNone(restored._records[("v", "answer")]["q1"][0])

    def test_a_payload_without_a_timeline_still_loads(self) -> None:
        """The score history is an additive key; a payload that predates it is fine."""
        restored = HealthTracker.from_snapshot({"version": 3, "records": []})
        self.assertEqual(restored.timeline, [])

    def test_unknown_snapshot_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HealthTracker.from_snapshot({"version": 99, "records": []})

    def test_unversioned_snapshot_is_rejected(self) -> None:
        """Without a version there is nothing to check the shape against."""
        with self.assertRaises(ValueError):
            HealthTracker.from_snapshot({"records": []})

    def test_malformed_snapshot_is_rejected(self) -> None:
        for payload in (
            {"version": 3, "records": "nope"},
            {"version": 3, "records": [["v", "answer", "q1", "nonsense", True]]},
            {"version": 3, "records": [["v", "answer", "q1", True]]},
            {"version": 3, "records": [], "weights": {"answer": "heavy"}},
            {"version": 3, "records": [], "weights": "heavy"},
            {"version": 3, "records": [], "categories": {"v": 3}},
            {"version": 3, "records": [], "categories": "heavy"},
            {"version": 3, "records": [], "baseline_epoch": 7},
            {"version": 3, "records": [], "seed": "nope"},
            {"version": 3, "records": [], "timeline": "nope"},
            {"version": 3, "records": [], "timeline": [{"score": "high"}]},
        ):
            with self.assertRaises(ValueError):
                HealthTracker.from_snapshot(payload)

    def test_invalid_record_is_rejected(self) -> None:
        tracker = HealthTracker()
        with self.assertRaises(ValueError):
            tracker.observe("", "answer", "q1", True, False)


class SideValidationTest(unittest.TestCase):
    """A side is a real result, ``None``, :data:`ERROR`, :data:`FAILED` or
    :data:`RETIRED`."""

    def test_non_binary_sides_are_rejected(self) -> None:
        tracker = HealthTracker(**FAST)
        # The *strings* "error"/"failed"/"retired" are snapshot encodings, not
        # API values.
        for junk in ("yes", "", 1, 0, "pending", "retired", "error", "failed"):
            with self.assertRaises(ValueError):
                tracker.observe("v", "answer", "q1", junk, True)
            with self.assertRaises(ValueError):
                tracker.observe("v", "answer", "q1", True, junk)
        self.assertEqual(len(tracker), 0)

    def test_coverage_agrees_with_every_other_counter(self) -> None:
        """One record is paired, pending, errored or retired in exactly one place."""
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", False, None)
        tracker.observe("v", "answer", "q3", False, ERROR)
        tracker.observe("v", "answer", "q4", False, RETIRED)
        tracker.observe("v", "answer", "q5", False, FAILED)
        coverage = tracker.coverage()[0]
        self.assertEqual(coverage.paired, tracker.paired)
        self.assertEqual(coverage.pending, tracker.pending)
        self.assertEqual(coverage.errored, tracker.errored)
        self.assertEqual(coverage.retired, tracker.retired)
        self.assertEqual(coverage.failed, tracker.failed)
        self.assertEqual(coverage.paired, sum(cell.pairs for cell in tracker.cells()))
        self.assertEqual(
            coverage.paired + coverage.pending + coverage.errored + coverage.retired,
            len(tracker),
        )
        # ``failed`` is a subset of ``paired``: those items are resolved, so
        # they are counted in neither ``pending`` nor ``errored``.
        self.assertLessEqual(coverage.failed, coverage.paired)
        self.assertEqual(coverage.unresolved, tracker.pending + tracker.errored)


class ErroredStateTest(unittest.TestCase):
    def test_error_is_distinct_from_pending(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, ERROR)
        self.assertEqual(tracker.errored, 1)
        self.assertEqual(tracker.pending, 0)
        self.assertEqual(tracker.paired, 0)
        report = tracker.report()
        self.assertEqual(report.coverage.errored, 1)
        self.assertEqual(report.status, "insufficient")

    def test_error_does_not_count_as_a_pair(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, ERROR)
        tracker.observe("v", "answer", "q2", False, True)
        self.assertEqual(tracker.paired, 1)
        report = tracker.report()
        self.assertEqual(report.pairs, 1)
        self.assertEqual(report.contributions[0].improved, 1)

    def test_error_survives_snapshot(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, ERROR)
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertEqual(restored.errored, 1)
        self.assertEqual(restored.coverage()[0].errored, 1)

    def test_advance_drops_an_errored_candidate(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, ERROR)
        tracker.advance()
        self.assertEqual(len(tracker), 0)


class FailedStateTest(unittest.TestCase):
    """A view-caused failure is a result: it scores as a failure, not an absence.

    ``ERROR`` says the harness could not evaluate the item; ``FAILED`` says the
    view could not answer it. The second is a health signal about the view, so
    dropping it from the denominator — exactly what ``ERROR`` does — would let a
    view raise its own score by breaking on the questions it was failing.
    """

    def test_view_caused_failure_is_paired_not_errored(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, FAILED)
        self.assertEqual(tracker.failed, 1)
        self.assertEqual(tracker.paired, 1)
        self.assertEqual(tracker.errored, 0)
        self.assertEqual(tracker.pending, 0)
        report = tracker.report()
        self.assertEqual(report.pairs, 1)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.coverage.failed, 1)
        self.assertEqual(report.score, 0.0)
        self.assertTrue(report.coverage.complete)

    def test_view_caused_failure_scores_as_a_failure(self) -> None:
        """A FAILED candidate is a FAIL: no improvement, a regression if it passed before."""
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "was_pass", True, FAILED)
        tracker.observe("v", "answer", "was_fail", False, FAILED)
        cell = tracker.cells()[0]
        self.assertEqual(cell.pairs, 2)
        self.assertEqual(cell.passed, 0)
        self.assertEqual(cell.regressed, 1)
        self.assertEqual(cell.improved, 0)

    def test_erroring_out_of_failing_items_does_not_raise_the_score(self) -> None:
        """The roadmap reproduction: 100 items at 60%.

        Losing the 20 always-failing items to ``ERROR`` lifts the level to 75.0
        with a silent ``+0.0`` composition effect. Scoring the same 20 as
        view-caused failures leaves the level at 60.0, because a failure is a
        result — no ledger and no composition term are needed.
        """

        def build(sentinel: object) -> HealthTracker:
            tracker = HealthTracker(**FAST)
            for index in range(100):
                if index < 60:
                    candidate: object = True
                elif index < 80:
                    candidate = sentinel
                else:
                    candidate = False
                tracker.observe("v", "answer", f"q{index}", index < 60, candidate)  # type: ignore[arg-type]
            return tracker

        errored = build(ERROR).report()
        self.assertEqual(errored.score, 75.0)
        self.assertEqual(errored.composition_effect_pp, 0.0)

        failed = build(FAILED).report()
        self.assertEqual(failed.score, 60.0)
        self.assertEqual(failed.baseline_score, 60.0)
        self.assertEqual(failed.composition_effect_pp, 0.0)
        self.assertEqual(failed.pairs, 100)
        self.assertEqual(failed.failed, 20)

    def test_harness_error_outranks_a_view_caused_failure(self) -> None:
        """One side unresolved is still unresolved: the harness must retry it."""
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", FAILED, ERROR)
        self.assertEqual(tracker.errored, 1)
        self.assertEqual(tracker.failed, 0)
        self.assertEqual(tracker.paired, 0)

    def test_retirement_outranks_a_view_caused_failure(self) -> None:
        """A retired record is not scored, so it is not counted as failed either."""
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", FAILED, RETIRED)
        self.assertEqual(tracker.retired, 1)
        self.assertEqual(tracker.failed, 0)
        self.assertEqual(tracker.coverage()[0].failed, 0)

    def test_advance_promotes_a_view_caused_failure_as_fail(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", True, FAILED)
        tracker.advance()
        self.assertEqual(len(tracker), 1)
        self.assertEqual(tracker.paired, 0)
        self.assertEqual(tracker._records[("v", "answer")]["q1"], (False, None))

    def test_retiring_a_failed_item_holds_it_as_a_failure(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", False, FAILED)
        tracker.retire("v", "answer", "q2")
        self.assertEqual(tracker.retired_ledger[("v", "answer")]["q2"], False)
        report = tracker.report()
        self.assertEqual(report.score, 100.0)
        self.assertAlmostEqual(report.score_held, 50.0)
        self.assertAlmostEqual(report.composition_effect_pp, 50.0)

    def test_failed_survives_snapshot(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, FAILED)
        payload = tracker.to_json()
        self.assertIn('"failed"', payload)
        restored = HealthTracker.from_json(payload, **FAST)
        self.assertEqual(restored.snapshot(), tracker.snapshot())
        self.assertEqual(restored.failed, 1)
        self.assertEqual(restored.coverage()[0].failed, 1)

    def test_payload_without_a_failed_side_still_loads(self) -> None:
        """Additive encoding: a snapshot written before FAILED existed loads."""
        restored = HealthTracker.from_snapshot(
            {
                "version": 3,
                "records": [
                    ["v", "answer", "q1", "error", False],
                    ["v", "answer", "q2", False, True],
                ],
            },
            **FAST,
        )
        self.assertEqual(restored.errored, 1)
        self.assertEqual(restored.failed, 0)
        self.assertEqual(restored.paired, 1)


class RetiredStateTest(unittest.TestCase):
    """A withdrawn item is neither outstanding work nor a failure."""

    def test_retired_is_distinct_from_pending_and_errored(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, RETIRED)
        self.assertEqual(tracker.retired, 1)
        self.assertEqual(tracker.pending, 0)
        self.assertEqual(tracker.errored, 0)
        self.assertEqual(tracker.paired, 0)

    def test_retirement_outranks_error_and_pending(self) -> None:
        """Retirement is a statement about the item, so it holds either side."""
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", ERROR, RETIRED)
        tracker.observe("v", "answer", "q2", None, RETIRED)
        self.assertEqual(tracker.retired, 2)
        self.assertEqual(tracker.errored, 0)
        self.assertEqual(tracker.pending, 0)

    def test_retire_keeps_the_last_known_outcome(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.retire("v", "answer", "q1")
        self.assertEqual(tracker.retired, 1)
        self.assertEqual(tracker.retired_ledger[("v", "answer")]["q1"], True)

    def test_retire_prefers_the_most_recent_outcome(self) -> None:
        """The candidate epoch is newer than the baseline one."""
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", True, False)  # regressed
        tracker.retire("v", "answer", "q1")
        self.assertEqual(tracker.retired_ledger[("v", "answer")]["q1"], False)

    def test_retiring_an_unseen_item_is_allowed(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.retire("v", "answer", "never-run")
        self.assertEqual(tracker.retired, 1)
        self.assertIsNone(tracker.retired_ledger[("v", "answer")]["never-run"])

    def test_retire_rejects_empty_keys(self) -> None:
        tracker = HealthTracker(**FAST)
        with self.assertRaises(ValueError):
            tracker.retire("", "answer", "q1")
        with self.assertRaises(ValueError):
            tracker.retire("v", "answer", "")

    def test_retired_item_leaves_the_paired_test(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", False, False)
        tracker.retire("v", "answer", "q2")
        report = tracker.report()
        self.assertEqual(report.pairs, 1)
        self.assertEqual([item.stratum for item in report.contributions], ["v"])
        self.assertEqual(report.contributions[0].pairs, 1)

    def test_retirement_does_not_gate_coverage(self) -> None:
        """The point of the sentinel: a shrinking bank is not a broken pipeline.

        Ten retired items and four pending-free pairs would be a 71% unresolved
        share if retirement were folded into pending, which would gate the
        layer to ``insufficient``.
        """
        tracker = HealthTracker(seed=0, **FAST)
        for index in range(4):
            tracker.observe("v", "answer", f"keep{index}", False, True)
        for index in range(10):
            tracker.retire("v", "answer", f"drop{index}")
        report = tracker.report()
        self.assertTrue(report.coverage.complete)
        self.assertEqual(report.coverage.retired, 10)
        self.assertEqual(report.coverage.unresolved_share, 0.0)
        self.assertNotEqual(report.status, "insufficient")

    def test_a_shrunken_bank_is_visible_on_the_report(self) -> None:
        """The level follows the survivors, and ``retired`` says so out loud."""
        tracker = HealthTracker(**FAST)
        for index in range(4):
            tracker.observe("v", "answer", f"keep{index}", False, True)
        for index in range(6):
            tracker.observe("v", "answer", f"drop{index}", False, False)
        before = tracker.report()
        self.assertEqual(before.score, 40.0)
        self.assertEqual(before.retired, 0)
        for index in range(6):
            tracker.retire("v", "answer", f"drop{index}")
        after = tracker.report()
        self.assertEqual(after.pairs, 4)
        self.assertEqual(after.score, 100.0)
        self.assertEqual(after.retired, 6)

    def test_advance_drops_the_row_but_keeps_the_ledger(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.retire("v", "answer", "q1")
        tracker.advance()
        self.assertEqual(len(tracker), 0)
        self.assertEqual(tracker.retired, 0)
        self.assertEqual(tracker.retired_ledger[("v", "answer")]["q1"], True)

    def test_retirement_survives_snapshot(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.retire("v", "answer", "q1")
        tracker.retire("v", "answer", "never-run")
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertEqual(restored.retired, 2)
        self.assertEqual(restored.coverage()[0].retired, 2)
        self.assertEqual(restored.retired_ledger[("v", "answer")]["q1"], True)
        self.assertIsNone(restored.retired_ledger[("v", "answer")]["never-run"])

    def test_ledger_rejects_a_non_boolean_outcome(self) -> None:
        tracker = HealthTracker(**FAST)
        payload = tracker.snapshot()
        payload["retired_ledger"] = [["v", "answer", "q1", "pass"]]
        with self.assertRaises(ValueError):
            HealthTracker.from_snapshot(payload, **FAST)

    def test_snapshot_without_a_ledger_still_loads(self) -> None:
        """Additive key: a payload written before the ledger existed loads."""
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        payload = tracker.snapshot()
        del payload["retired_ledger"]
        restored = HealthTracker.from_snapshot(payload, **FAST)
        self.assertEqual(restored.paired, 1)
        self.assertEqual(restored.retired_ledger, {})


class CompositionSplitTest(unittest.TestCase):
    """A shrinking bank must not silently redefine the level."""

    def test_no_retirement_means_no_composition_effect(self) -> None:
        tracker = HealthTracker(**FAST)
        for index in range(4):
            tracker.observe("v", "answer", f"q{index}", False, index < 2)
        report = tracker.report()
        self.assertEqual(report.score_held, report.score)
        self.assertEqual(report.composition_effect_pp, 0.0)
        self.assertEqual(report.retired_held, 0)

    def test_deleting_failing_questions_shows_up_as_composition(self) -> None:
        """The level reads 100 only because the hard questions were removed."""
        tracker = HealthTracker(**FAST)
        for index in range(4):
            tracker.observe("v", "answer", f"pass{index}", False, True)
        for index in range(2):
            tracker.observe("v", "answer", f"fail{index}", False, False)
        self.assertAlmostEqual(tracker.report().score, 100.0 * 4 / 6)
        for index in range(2):
            tracker.retire("v", "answer", f"fail{index}")
        report = tracker.report()
        self.assertEqual(report.score, 100.0)
        self.assertAlmostEqual(report.score_held, 100.0 * 4 / 6)
        self.assertAlmostEqual(report.composition_effect_pp, 100.0 - 100.0 * 4 / 6)
        self.assertEqual(report.retired_held, 2)
        self.assertEqual(report.retired, 2)

    def test_level_decomposes_exactly(self) -> None:
        """``score = score_held + composition_effect_pp``, with no residual."""
        tracker = HealthTracker(**FAST)
        for index in range(7):
            tracker.observe("v", "answer", f"p{index}", False, True)
        for index in range(5):
            tracker.observe("v", "answer", f"f{index}", False, False)
        for index in range(5):
            tracker.retire("v", "answer", f"f{index}")
        report = tracker.report()
        self.assertEqual(report.retired_held, 5)
        self.assertAlmostEqual(
            report.score - report.score_held - report.composition_effect_pp, 0.0, places=9
        )

    def test_a_fully_retired_view_is_still_held(self) -> None:
        """A view with no surviving pairs still carries its weight in the level."""
        tracker = HealthTracker(**FAST)
        tracker.observe("keep", "answer", "k1", False, True)
        tracker.observe("gone", "answer", "g1", False, False)
        tracker.retire("gone", "answer", "g1")
        report = tracker.report()
        self.assertEqual(report.score, 100.0)
        self.assertAlmostEqual(report.score_held, 50.0)
        self.assertAlmostEqual(report.composition_effect_pp, 50.0)
        self.assertEqual(report.retired_held, 1)

    def test_held_items_use_the_outcome_weight(self) -> None:
        tracker = HealthTracker(weights={"heavy": 3.0}, **FAST)
        tracker.observe("v", "heavy", "h1", False, True)
        tracker.observe("v", "light", "l1", False, False)
        tracker.retire("v", "light", "l1")
        report = tracker.report()
        self.assertEqual(report.score, 100.0)
        self.assertAlmostEqual(report.score_held, 75.0)  # 3*1 pass / (3*1 + 1*0)
        self.assertAlmostEqual(report.composition_effect_pp, 25.0)

    def test_held_items_add_to_a_cell_that_survived(self) -> None:
        """Retired items join their surviving cell's count, not replace it."""
        tracker = HealthTracker(
            categories={"v": "tier-1"}, category_weights={"tier-1": 3.0}, **FAST
        )
        tracker.observe("v", "answer", "a1", False, True)
        tracker.observe("v", "answer", "r1", False, False)
        tracker.retire("v", "answer", "r1")
        report = tracker.report()
        self.assertEqual(report.score, 100.0)
        self.assertAlmostEqual(report.score_held, 50.0)

    def test_an_unmeasured_retirement_has_nothing_to_hold(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.retire("v", "answer", "never-run")
        report = tracker.report()
        self.assertEqual(report.retired, 1)
        self.assertEqual(report.retired_held, 0)
        self.assertEqual(report.score_held, report.score)
        self.assertEqual(report.composition_effect_pp, 0.0)

    def test_the_overlay_respects_the_layer_policy_and_declared_tiers(self) -> None:
        """Held items keep their catalogue tier; live views keep the policy's weight.

        The only tiered view is fully retired, so no *live* cell declares a tier
        and ``tiered`` is False: the untagged live view is not re-defaulted into
        ``tier-2`` just because a retired item dragged a category into the
        overlay. The retired item itself keeps its declared tier-1 weight — that
        is catalogue truth and does not touch the headline.
        """
        tracker = HealthTracker(
            categories={"crit": "tier-1"},
            category_weights={"tier-1": 3.0, "tier-2": 2.0},
            **FAST,
        )
        tracker.observe("aux", "answer", "a1", False, True)
        tracker.observe("aux", "answer", "a2", False, True)
        tracker.observe("crit", "answer", "c1", False, False)
        tracker.retire("crit", "answer", "c1")
        report = tracker.report()
        self.assertFalse(report.tiered)
        self.assertEqual(report.score, 100.0)  # the live view alone, unweighted
        self.assertAlmostEqual(report.score_held, 40.0)  # (1*2 + 3*0) / (1*2 + 3*1)
        self.assertAlmostEqual(report.composition_effect_pp, 60.0)


class CoverageGateTest(unittest.TestCase):
    def test_coverage_gates_when_pending_share_is_high(self) -> None:
        tracker = HealthTracker(**FAST)
        for index in range(4):
            tracker.observe("v", "answer", f"q{index}", False, True)
        tracker.observe("v", "answer", "q9", False, None)
        report = tracker.report()
        self.assertEqual(report.coverage.pending, 1)
        self.assertTrue(report.coverage.complete)  # 1/5 = 20% <= 25%
        for index in range(4):
            tracker.observe("v", "answer", f"p{index}", False, None)
        report = tracker.report()
        self.assertFalse(report.coverage.complete)
        self.assertEqual(report.status, "insufficient")

    def test_critical_tier_pending_gates_immediately(self) -> None:
        tracker = HealthTracker(
            categories={"crit": "tier-1"},
            category_weights={"tier-1": 3.0, "tier-3": 1.0},
            **FAST,
        )
        tracker.set_category("aux", "tier-3")
        for index in range(20):
            tracker.observe("aux", "answer", f"q{index}", False, True)
        tracker.observe("crit", "answer", "q1", False, None)
        report = tracker.report()
        self.assertEqual(report.coverage.critical_unresolved, 1)
        self.assertEqual(report.status, "insufficient")


class SeedLifecycleTest(unittest.TestCase):
    """The first seed is drawn once, persisted, and reused for every update."""

    def test_seed_is_generated_when_not_supplied(self) -> None:
        tracker = HealthTracker(**FAST)
        self.assertIsInstance(tracker.seed, int)

    def test_seed_is_persisted_and_restored(self) -> None:
        tracker = HealthTracker(**FAST)
        seed = tracker.seed
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertEqual(restored.seed, seed)

    def test_seed_survives_updates_and_advance(self) -> None:
        tracker = HealthTracker(**FAST)
        seed = tracker.seed
        tracker.observe("v", "answer", "q1", False, True)
        tracker.report()
        tracker.advance()
        tracker.observe("v", "answer", "q1", True, False)
        tracker.report()
        self.assertEqual(tracker.seed, seed)

    def test_explicit_seed_is_honoured_on_restore(self) -> None:
        tracker = HealthTracker(seed=7, **FAST)
        restored = HealthTracker.from_json(tracker.to_json(), seed=99, **FAST)
        self.assertEqual(restored.seed, 99)


class MergeGuardTest(unittest.TestCase):
    def test_merge_rejects_different_epochs(self) -> None:
        primary = HealthTracker(**FAST, baseline_epoch="a", candidate_epoch="b")
        other = HealthTracker(**FAST, baseline_epoch="a", candidate_epoch="c")
        with self.assertRaises(ValueError):
            primary.merge(other)
        primary.merge(other, on_epoch_mismatch="ignore")

    def test_merge_prefer_other_moves_config_too(self) -> None:
        primary = HealthTracker(**FAST)
        other = HealthTracker(
            weights={"answer": 2.0},
            categories={"v": "tier-1"},
            category_weights={"tier-1": 3.0},
        )
        primary.merge(other)
        self.assertEqual(primary.weights, {"answer": 2.0})
        self.assertEqual(primary.category_of("v"), "tier-1")
        self.assertEqual(primary.category_weights, {"tier-1": 3.0})

    def test_merge_prefer_self_keeps_local_config(self) -> None:
        primary = HealthTracker(weights={"answer": 5.0}, **FAST)
        other = HealthTracker(weights={"answer": 2.0})
        primary.merge(other, prefer="self")
        self.assertEqual(primary.weights, {"answer": 5.0})

    def test_merge_keeps_a_real_value_over_an_error(self) -> None:
        primary = HealthTracker(**FAST)
        primary.observe("v", "answer", "q1", False, ERROR)
        secondary = HealthTracker(**FAST)
        secondary.observe("v", "answer", "q1", False, True)
        primary.merge(secondary)
        self.assertEqual(primary.errored, 0)
        self.assertEqual(primary.paired, 1)

    def test_merge_rejects_an_unknown_epoch_policy(self) -> None:
        with self.assertRaises(ValueError):
            HealthTracker().merge(HealthTracker(), on_epoch_mismatch="maybe")  # type: ignore[arg-type]


class TimelineTest(unittest.TestCase):
    """The tracker must keep enough history for :meth:`score_band`."""

    def test_advance_records_the_finished_epoch(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", False, False)
        tracker.advance(baseline_epoch="r1", candidate_epoch="r2")
        self.assertEqual(len(tracker.timeline), 1)
        self.assertAlmostEqual(tracker.timeline[0].score, 50.0)
        self.assertEqual(tracker.timeline[0].pairs, 2.0)
        self.assertEqual(tracker.timeline[0].epoch, "r1")
        self.assertEqual(tracker.baseline_epoch, "r1")
        self.assertEqual(tracker.candidate_epoch, "r2")

    def test_recording_is_optional_on_advance(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.advance(record=False)
        self.assertEqual(tracker.timeline, [])

    def test_an_empty_epoch_is_not_a_data_point(self) -> None:
        tracker = HealthTracker(**FAST)
        self.assertIsNone(tracker.record_score())
        tracker.observe("v", "answer", "q1", None, True)  # baseline still pending
        self.assertIsNone(tracker.record_score())
        self.assertEqual(tracker.timeline, [])

    def test_muted_cells_are_not_a_data_point(self) -> None:
        tracker = HealthTracker(weights={"answer": 0.0}, **FAST)
        tracker.observe("v", "answer", "q1", False, True)
        self.assertIsNone(tracker.record_score())

    def test_band_over_a_recorded_series(self) -> None:
        tracker = HealthTracker(**FAST)
        for epoch, passes in (("r1", 3), ("r2", 5), ("r3", 4)):
            for index in range(10):
                tracker.observe("v", "answer", f"q{index}", False, index < passes)
            tracker.advance(baseline_epoch=epoch, candidate_epoch=f"{epoch}+1")
        band = tracker.score_band()
        self.assertEqual(band.points, 3)
        self.assertAlmostEqual(band.center, (30.0 + 50.0 + 40.0) / 3.0)

    def test_empty_timeline_has_no_band(self) -> None:
        with self.assertRaises(ValueError):
            HealthTracker(**FAST).score_band()

    def test_timeline_survives_a_snapshot(self) -> None:
        tracker = HealthTracker(seed=0, **FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.advance(baseline_epoch="r1", candidate_epoch="r2")
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertEqual(restored.timeline, tracker.timeline)
        self.assertEqual(restored.snapshot(), tracker.snapshot())

    def test_recorded_points_carry_the_regression_base_rate(self) -> None:
        """The next epoch's prior calibration needs this epoch's pi_mean."""
        tracker = HealthTracker(seed=0, **FAST)
        tracker.observe("v", "answer", "q1", False, True)
        report = tracker.report()
        point = tracker.record_score()
        assert point is not None
        self.assertEqual(point.pi_mean, report.pi_mean)
        self.assertEqual(tracker.timeline[-1].pi_mean, report.pi_mean)

    def test_the_finished_epochs_base_rate_is_not_carried_forward(self) -> None:
        tracker = HealthTracker(seed=0, **FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.report()
        tracker.advance()
        tracker.observe("v", "answer", "q1", True, False)
        point = tracker.record_score()
        assert point is not None
        self.assertIsNone(point.pi_mean)

    def test_next_reports_calibration_uses_the_recorded_base_rate(self) -> None:
        tracker = HealthTracker(seed=0, **FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.report()
        tracker.advance()  # records the finished epoch, then clears the bank
        report = tracker.report(calibrate=True)
        self.assertEqual(report.calibration.history_points, 1)
        self.assertIn("pi_prior", report.calibration.derived)
        self.assertIsNotNone(report.calibration.observed_pi_mean)

    def test_report_offers_its_timeline_as_calibration_history(self) -> None:
        tracker = HealthTracker(seed=0, **FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.advance(baseline_epoch="r1", candidate_epoch="r2")
        tracker.observe("v", "answer", "q1", True, False)

        default = tracker.report()
        self.assertEqual(default.calibration.history_points, 1)
        self.assertFalse(default.calibration.enabled)
        self.assertFalse(default.calibration.calibrated)

        opted_in = tracker.report(calibrate=True)
        self.assertEqual(opted_in.calibration.history_points, 1)
        self.assertTrue(opted_in.calibration.enabled)
        # B1 plumbs the history; the derivations land with B2-B4, so with a
        # single point every decision number is still the default one.
        self.assertEqual(opted_in.rope_pp, default.rope_pp)
        self.assertEqual(opted_in.score, default.score)
        self.assertEqual(opted_in.delta_pp, default.delta_pp)
        self.assertEqual(opted_in.status, default.status)


class ColdStartTrackerTest(unittest.TestCase):
    """A first run (or a new view) has no baseline: it still has a level."""

    def test_candidate_only_epoch_reports_the_unpaired_level(self) -> None:
        tracker = HealthTracker(**FAST)
        for index in range(10):
            tracker.observe("v", "answer", f"q{index}", baseline=None, candidate=index < 7)
        report = tracker.report()
        self.assertEqual(report.status, "insufficient")
        self.assertIsNone(report.score)
        self.assertEqual(report.pending, 10)
        self.assertEqual(report.unpaired, 10)
        self.assertEqual(report.score_unpaired, 70.0)
        self.assertAlmostEqual(report.unresolved_share or 0.0, 1.0)

    def test_unpaired_level_disappears_once_the_baseline_arrives(self) -> None:
        tracker = HealthTracker(**FAST)
        for index in range(10):
            tracker.observe("v", "answer", f"q{index}", baseline=None, candidate=index < 7)
        for index in range(10):
            tracker.observe("v", "answer", f"q{index}", baseline=index < 4, candidate=index < 7)
        report = tracker.report()
        self.assertEqual(report.score, 70.0)
        self.assertEqual(report.unpaired, 0)
        self.assertIsNone(report.score_unpaired)
        self.assertEqual(report.unresolved_share, 0.0)

    def test_a_new_view_is_the_unpaired_part_of_a_mixed_report(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("old", "answer", "q1", False, True)
        tracker.observe("new", "answer", "n1", None, False)
        tracker.observe("new", "answer", "n2", None, True)
        report = tracker.report()
        self.assertEqual(report.score, 100.0)  # the paired view only
        self.assertEqual(report.unpaired, 2)
        self.assertEqual(report.score_unpaired, 50.0)  # the cold view

    def test_a_view_caused_failure_counts_against_the_unpaired_level(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", None, True)
        tracker.observe("v", "answer", "q2", None, FAILED)
        report = tracker.report()
        self.assertEqual(report.unpaired, 2)
        self.assertEqual(report.score_unpaired, 50.0)

    def test_paired_errored_and_retired_rows_are_not_unpaired(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "paired", False, True)
        tracker.observe("v", "answer", "errored", ERROR, True)
        tracker.observe("v", "answer", "retired", None, RETIRED)
        tracker.observe("v", "answer", "cold", None, True)
        cells = tracker.unpaired_cells()
        self.assertEqual(
            [(c.stratum, c.outcome, c.resolved, c.passed) for c in cells], [("v", "answer", 1, 1)]
        )

    def test_timeline_records_the_epochs_unresolved_share(self) -> None:
        tracker = HealthTracker(seed=0, **FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", True, None)
        tracker.advance()
        self.assertEqual(len(tracker.timeline), 1)
        # One of the two rows never resolved at the epoch boundary.
        self.assertAlmostEqual(tracker.timeline[0].unresolved_share or 0.0, 0.5)

    def test_unresolved_share_round_trips_through_a_snapshot(self) -> None:
        tracker = HealthTracker(seed=0, **FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", True, None)
        tracker.advance()
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertEqual(restored.timeline, tracker.timeline)


class ItemIdentityTest(unittest.TestCase):
    """The global item key ``(outcome, item)`` that the difficulty model needs.

    A row is identified by ``(stratum, outcome, item)``, but an *item* is
    identified by ``(outcome, item)``: the same question asked of two views is
    one item (the overlap that makes abilities comparable), and a renamed
    question is a different item.
    """

    def test_item_responses_are_paired_candidate_results(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", True, FAILED)
        tracker.observe("v", "answer", "q3", False, None)  # pending
        tracker.observe("v", "answer", "q4", False, ERROR)  # harness error
        tracker.observe("v", "answer", "q5", False, RETIRED)  # withdrawn
        responses = tracker.item_responses()
        self.assertEqual([r.item for r in responses], ["q1", "q2"])
        self.assertTrue(responses[0].passed)
        # A view-caused failure is a result: it arrives as a failed response.
        self.assertFalse(responses[1].passed)
        self.assertEqual(responses[0].stratum, "v")
        self.assertEqual(responses[0].outcome, "answer")

    def test_the_same_item_across_views_is_one_item(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("view_a", "answer", "q5", False, True)
        tracker.observe("view_b", "answer", "q5", True, False)
        responses = tracker.item_responses()
        self.assertEqual(len(responses), 2)
        self.assertEqual({r.key for r in responses}, {("answer", "q5")})
        self.assertEqual({r.stratum for r in responses}, {"view_a", "view_b"})

    def test_a_differing_item_string_is_a_different_item(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("view_a", "answer", "q5", False, True)
        tracker.observe("view_b", "answer", "q5_v2", True, False)
        keys = {r.key for r in tracker.item_responses()}
        self.assertEqual(keys, {("answer", "q5"), ("answer", "q5_v2")})

    def test_the_outcome_is_part_of_the_item_key(self) -> None:
        """One question under two metrics is two items, not one."""
        tracker = HealthTracker(**FAST)
        tracker.observe("view_a", "answer", "q5", False, True)
        tracker.observe("view_a", "sql_valid", "q5", False, True)
        keys = {r.key for r in tracker.item_responses()}
        self.assertEqual(keys, {("answer", "q5"), ("sql_valid", "q5")})

    def test_item_responses_are_sorted_for_determinism(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("view_b", "answer", "q2", False, True)
        tracker.observe("view_a", "answer", "q1", False, True)
        tracker.observe("view_a", "answer", "q1", False, True)  # replay overwrites
        responses = tracker.item_responses()
        self.assertEqual(
            [(r.stratum, r.item) for r in responses], [("view_a", "q1"), ("view_b", "q2")]
        )


class DifficultyConfigTest(unittest.TestCase):
    """IRT config is opt-in, validated, and persisted with the tracker."""

    def _tracker(self, **config: object) -> HealthTracker:
        tracker = HealthTracker(seed=0, **FAST, **config)  # type: ignore[arg-type]
        for index in range(6):
            tracker.observe("hard", "answer", f"anchor{index}", False, True)
            tracker.observe("easy", "answer", f"anchor{index}", False, True)
        for index in range(6):
            tracker.observe("hard", "answer", f"hard{index}", False, False)
            tracker.observe("easy", "answer", f"easy{index}", False, True)
        return tracker

    def test_config_round_trips_and_scores_the_same(self) -> None:
        reference = [("answer", f"anchor{index}") for index in range(6)]
        tracker = self._tracker(
            adjust_difficulty=True, sigma_theta=1.2, sigma_d=0.8, reference_items=reference
        )
        assert tracker.report().score_adjusted is not None
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertTrue(restored.adjust_difficulty)
        self.assertEqual(restored.sigma_theta, 1.2)
        self.assertEqual(restored.sigma_d, 0.8)
        self.assertEqual(restored.reference_items, tuple(reference))
        self.assertEqual(restored.snapshot(), tracker.snapshot())
        self.assertEqual(restored.report().model_dump(), tracker.report().model_dump())

    def test_the_config_is_the_report_default(self) -> None:
        self.assertIsNone(self._tracker().report().score_adjusted)
        self.assertIsNotNone(self._tracker(adjust_difficulty=True).report().score_adjusted)

    def test_per_call_override_does_not_touch_the_config(self) -> None:
        tracker = self._tracker()
        self.assertIsNone(tracker.report().score_adjusted)
        forced = tracker.report(adjust_difficulty=True)
        self.assertIsNotNone(forced.score_adjusted)
        self.assertFalse(tracker.adjust_difficulty)
        pinned = tracker.report(adjust_difficulty=True, reference_items=[("answer", "anchor0")])
        self.assertIsNotNone(pinned.score_adjusted)
        self.assertIsNone(tracker.reference_items)

    def test_a_payload_without_the_config_still_loads(self) -> None:
        restored = HealthTracker.from_snapshot(
            {"version": 3, "records": [["v", "answer", "q", False, True]]}
        )
        self.assertFalse(restored.adjust_difficulty)
        self.assertEqual(restored.sigma_theta, 1.5)
        self.assertEqual(restored.sigma_d, 1.5)
        self.assertIsNone(restored.reference_items)
        self.assertEqual(restored.snapshot()["version"], 3)

    def test_reference_item_order_does_not_change_the_snapshot(self) -> None:
        first = HealthTracker(seed=0, reference_items=[("answer", "q2"), ("answer", "q1")])
        second = HealthTracker(seed=0, reference_items=[("answer", "q1"), ("answer", "q2")])
        self.assertEqual(first.to_json(), second.to_json())

    def test_a_versioned_bank_round_trips_and_scores_the_same(self) -> None:
        probe = self._tracker()
        bank = ReferenceBank.from_responses(probe.item_responses(), name="standard", version="1")
        tracker = self._tracker(adjust_difficulty=True, reference_bank=bank)
        self.assertEqual(tracker.reference_bank, bank)
        self.assertIsNone(tracker.reference_items)
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertEqual(restored.reference_bank, bank)
        report = tracker.report()
        assert report.irt is not None and report.irt.reference is not None
        self.assertEqual(report.irt.reference.name, "standard")
        self.assertEqual(restored.report().model_dump(), report.model_dump())

    def test_a_per_call_bank_overrides_and_does_not_persist(self) -> None:
        tracker = self._tracker(adjust_difficulty=True)
        bank = ReferenceBank.from_responses(tracker.item_responses(), name="override", version="9")
        report = tracker.report(reference_bank=bank)
        assert report.irt is not None and report.irt.reference is not None
        self.assertEqual(report.irt.reference.name, "override")
        self.assertIsNone(tracker.reference_bank)

    def test_invalid_config_is_rejected(self) -> None:
        bank = ReferenceBank(name="s", version="1", items=(("answer", "q"),))
        for config in (
            {"adjust_difficulty": "yes"},
            {"sigma_theta": 0.0},
            {"sigma_d": -1.0},
            {"sigma_theta": "wide"},
            {"reference_items": []},
            {"reference_items": "answer/q1"},
            {"reference_items": [("answer",)]},
            {"reference_items": [(7, "q")]},
            {"reference_items": [("answer", "")]},
            {"reference_items": [("answer", "q"), ("answer", "q")]},
            {"reference_bank": {"name": "s"}},
            {"reference_bank": bank, "reference_items": [("answer", "q")]},
        ):
            with self.assertRaises(ValueError):
                HealthTracker(**config)  # type: ignore[arg-type]

    def test_a_per_call_bank_conflicts_with_a_configured_raw_pin(self) -> None:
        tracker = self._tracker(adjust_difficulty=True, reference_items=[("answer", "anchor0")])
        bank = ReferenceBank.from_responses(tracker.item_responses(), name="s", version="1")
        with self.assertRaises(ValueError):
            tracker.report(reference_bank=bank)

    def test_malformed_snapshot_config_is_rejected(self) -> None:
        for payload in (
            {"version": 3, "records": [], "adjust_difficulty": "yes"},
            {"version": 3, "records": [], "sigma_theta": "wide"},
            {"version": 3, "records": [], "sigma_d": 0.0},
            {"version": 3, "records": [], "reference_items": "nope"},
            {"version": 3, "records": [], "reference_items": []},
            {"version": 3, "records": [], "reference_items": [["answer"]]},
            {"version": 3, "records": [], "reference_bank": "nope"},
            {"version": 3, "records": [], "reference_bank": {"name": "s", "version": "1"}},
            {
                "version": 3,
                "records": [],
                "reference_bank": {"name": 7, "version": "1", "items": [["answer", "q"]]},
            },
            {
                "version": 3,
                "records": [],
                "reference_bank": {"name": "s", "version": "1", "items": []},
            },
        ):
            with self.assertRaises(ValueError):
                HealthTracker.from_snapshot(payload)


class DecisionConfigTest(unittest.TestCase):
    """A bad decision knob fails where it is set, not inside a draw."""

    def test_invalid_decision_config_is_rejected_at_construction(self) -> None:
        for config in (
            {"alpha": 0.9},
            {"alpha": 0.0},
            {"rope_pp": -1.0},
            {"max_unresolved_share": 1.5},
            {"prob_interval": 2.0},
            {"draws": 0},
            {"draws": 1},
            {"ppc_draws": 0},
            {"pi_points": 0},
            {"pi_prior": (0.0, 1.0)},
            {"a0": 0.0},
            {"b0": -1.0},
            {"a_d": 0.0},
            {"b_d": 0.0},
            {"alpha": "high"},
            {"draws": True},
            {"pi_prior": (1.0,)},
        ):
            with self.assertRaises(ValueError):
                HealthTracker(**config)  # type: ignore[arg-type]

    def test_report_overrides_are_validated_too(self) -> None:
        tracker = HealthTracker(seed=0, draws=96, ppc_draws=96)
        tracker.observe("v", "answer", "q0", False, True)
        for override in ({"draws": 0}, {"alpha": 0.9}, {"max_unresolved_share": 2.0}):
            with self.assertRaises(ValueError):
                tracker.report(**override)  # type: ignore[arg-type]


def _group_views() -> dict[str, str]:
    """Twelve views sharing one upstream table."""
    return {f"v{index}": "table_x" for index in range(12)}


def _fill_group(tracker: HealthTracker, *, improving: int) -> None:
    """Fill each view with ``improving`` improvements and the rest regressions."""
    for index in range(12):
        for q in range(20):
            tracker.observe(f"v{index}", "answer", f"q{q}", q >= improving, q < improving)


def _gate(report: object) -> list[tuple[object, ...]]:
    """The gate's per-incident reading, for comparing two runs."""
    return [
        (
            row.group,
            row.cells,
            row.pairs,
            row.improved,
            row.regressed,
            row.p_incident,
            row.flagged,
        )
        for row in report.groups  # type: ignore[attr-defined]
    ]


class GroupEpochInvariantTest(unittest.TestCase):
    """An incident is recomputed per epoch, and the group label survives the roll."""

    def test_a_finished_incident_does_not_leak_into_the_next_epoch(self) -> None:
        tracker = HealthTracker(seed=0, groups=_group_views(), candidate_epoch="e1", **FAST)
        _fill_group(tracker, improving=8)
        first = tracker.report()
        self.assertEqual([row.group for row in first.groups], ["table_x"])
        self.assertTrue(first.groups[0].flagged)
        self.assertEqual(first.status, "regressing")

        tracker.advance(baseline_epoch="e1", candidate_epoch="e2")
        self.assertEqual(tracker.group_of("v0"), "table_x")
        self.assertEqual(tracker.paired, 0)
        self.assertEqual([point.epoch for point in tracker.timeline], ["e1"])

        # Every view recovers: the closed epoch's incident must not be inherited.
        for index in range(12):
            for q in range(20):
                tracker.observe(f"v{index}", "answer", f"q{q}", True, True)
        second = tracker.report()
        self.assertEqual([row.group for row in second.groups], ["table_x"])
        self.assertFalse(second.groups[0].flagged)
        self.assertEqual(second.guardrail.flagged, [])
        self.assertNotEqual(second.status, "regressing")

    def test_a_new_shared_break_is_one_incident_again(self) -> None:
        tracker = HealthTracker(seed=0, groups=_group_views(), candidate_epoch="e1", **FAST)
        _fill_group(tracker, improving=8)
        tracker.advance(baseline_epoch="e1", candidate_epoch="e2")
        # The rolled baseline passes, the new candidate fails: a fresh incident.
        _fill_group(tracker, improving=0)
        second = tracker.report()
        self.assertEqual([row.group for row in second.groups], ["table_x"])
        self.assertTrue(second.groups[0].flagged)
        self.assertEqual(second.status, "regressing")


class GroupRetirementInvariantTest(unittest.TestCase):
    """Retirement removes evidence; it must not move the incident gate."""

    def _tracker(self) -> HealthTracker:
        tracker = HealthTracker(seed=0, groups=_group_views(), **FAST)
        _fill_group(tracker, improving=8)
        return tracker

    def test_retirement_leaves_the_incident_gate_on_the_remaining_pairs(self) -> None:
        retired = self._tracker()
        for index in range(12):
            for q in range(4):  # retire four improved items per view
                retired.retire(f"v{index}", "answer", f"q{q}")
        never = HealthTracker(seed=0, groups=_group_views(), **FAST)
        for index in range(12):
            for q in range(4, 20):  # the same evidence, never observed
                never.observe(f"v{index}", "answer", f"q{q}", q >= 8, q < 8)

        a, b = retired.report(), never.report()
        # Invariant 1: the surviving regressions stay flagged individually.
        self.assertEqual(a.status, "regressing")
        self.assertEqual(len(a.groups), 1)
        self.assertTrue(a.groups[0].flagged)
        self.assertTrue(all(row.verdict == "regressing" for row in a.contributions))
        # The incident row is computed on the remaining pairs only...
        self.assertEqual(_gate(a), _gate(b))
        self.assertEqual(
            [row.verdict for row in a.contributions],
            [row.verdict for row in b.contributions],
        )
        # ...so the layer's own numbers move only through the held overlay.
        self.assertEqual(a.score, b.score)
        self.assertEqual(a.delta_pp, b.delta_pp)
        self.assertEqual(a.retired, 48)
        self.assertEqual(a.retired_held, 48)
        assert a.composition_effect_pp is not None
        self.assertLess(a.composition_effect_pp, 0.0)


class FingerprintDriftTest(unittest.TestCase):
    """A content fingerprint guards global item identity, not the local row."""

    def test_the_first_fingerprint_is_remembered_and_replays_are_quiet(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True, "fp1_aaa")
        self.assertEqual(tracker.fingerprint_of("answer", "q1"), "fp1_aaa")
        self.assertIsNone(tracker.fingerprint_of("answer", "q2"))
        self.assertEqual(tracker.drift, [])
        tracker.observe("v", "answer", "q1", False, True, "fp1_aaa")
        self.assertEqual(tracker.drift, [])

    def test_changed_content_is_deduplicated_drift_and_keeps_the_first(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True, "fp1_aaa")
        tracker.observe("v", "answer", "q1", True, False, "fp1_bbb")
        tracker.observe("v2", "answer", "q1", True, False, "fp1_bbb")  # same change
        tracker.observe("v", "answer", "q1", False, False, "fp1_ccc")
        self.assertEqual(tracker.fingerprint_of("answer", "q1"), "fp1_aaa")
        self.assertEqual(
            [(row.expected, row.observed) for row in tracker.drift],
            [("fp1_aaa", "fp1_bbb"), ("fp1_aaa", "fp1_ccc")],
        )
        # Rows stay per-cell (v and v2); only the global identity drifts.
        self.assertEqual(tracker.paired, 2)

    def test_the_same_fingerprint_across_views_links_the_item(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v1", "answer", "q1", False, True, "fp1_aaa")
        tracker.observe("v2", "answer", "q1", False, True, "fp1_aaa")
        self.assertEqual(tracker.drift, [])

    def test_outcomes_own_their_fingerprints(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True, "fp1_aaa")
        tracker.observe("v", "sql_valid", "q1", False, True, "fp1_bbb")
        self.assertEqual(tracker.drift, [])

    def test_an_empty_fingerprint_is_rejected(self) -> None:
        tracker = HealthTracker(**FAST)
        with self.assertRaises(ValueError):
            tracker.observe("v", "answer", "q1", False, True, "")

    def test_observe_many_carries_the_fingerprint(self) -> None:
        tracker = HealthTracker(**FAST)
        tracker.observe_many(
            [
                PairRecord(
                    stratum="v",
                    outcome="answer",
                    item="q1",
                    baseline=False,
                    candidate=True,
                    fingerprint="fp1_aaa",
                )
            ]
        )
        self.assertEqual(tracker.fingerprint_of("answer", "q1"), "fp1_aaa")

    def test_advance_keeps_the_identity_guard(self) -> None:
        tracker = HealthTracker(seed=0, candidate_epoch="e1", **FAST)
        tracker.observe("v", "answer", "q1", False, True, "fp1_aaa")
        tracker.advance(baseline_epoch="e1", candidate_epoch="e2")
        self.assertEqual(tracker.paired, 0)
        tracker.observe("v", "answer", "q1", True, False, "fp1_bbb")
        self.assertEqual(tracker.fingerprint_of("answer", "q1"), "fp1_aaa")
        self.assertEqual(len(tracker.drift), 1)

    def test_snapshot_round_trips_fingerprints_and_drift(self) -> None:
        tracker = HealthTracker(seed=0, **FAST)
        tracker.observe("v", "answer", "q1", False, True, "fp1_aaa")
        tracker.observe("v2", "answer", "q1", True, False, "fp1_bbb")
        restored = HealthTracker.from_json(tracker.to_json(), **FAST)
        self.assertEqual(restored.snapshot(), tracker.snapshot())
        self.assertEqual(restored.fingerprint_of("answer", "q1"), "fp1_aaa")
        self.assertEqual(
            [(row.expected, row.observed) for row in restored.drift],
            [("fp1_aaa", "fp1_bbb")],
        )
        self.assertEqual(
            [(row.item, row.observed) for row in tracker.report().drift], [("q1", "fp1_bbb")]
        )

    def test_merge_keeps_the_first_fingerprint_and_records_the_disagreement(self) -> None:
        primary = HealthTracker(seed=0, **FAST)
        primary.observe("v", "answer", "q1", False, True, "fp1_aaa")
        secondary = HealthTracker(seed=0, **FAST)
        secondary.observe("v2", "answer", "q1", False, True, "fp1_bbb")
        primary.merge(secondary)
        self.assertEqual(primary.fingerprint_of("answer", "q1"), "fp1_aaa")
        self.assertEqual(
            [(row.expected, row.observed) for row in primary.drift],
            [("fp1_aaa", "fp1_bbb")],
        )

        # `prefer` decides the row, never the identity: the first binding stands.
        fresh = HealthTracker(seed=0, **FAST)
        fresh.observe("v", "answer", "q1", False, True, "fp1_aaa")
        other = HealthTracker(seed=0, **FAST)
        other.observe("v2", "answer", "q1", False, True, "fp1_bbb")
        fresh.merge(other, prefer="self")
        self.assertEqual(fresh.fingerprint_of("answer", "q1"), "fp1_aaa")
        self.assertEqual(len(fresh.drift), 1)

    def test_merge_fills_a_missing_fingerprint_without_drift(self) -> None:
        primary = HealthTracker(seed=0, **FAST)
        primary.observe("v", "answer", "q1", False, True)
        secondary = HealthTracker(seed=0, **FAST)
        secondary.observe("v2", "answer", "q1", False, True, "fp1_aaa")
        primary.merge(secondary)
        self.assertEqual(primary.fingerprint_of("answer", "q1"), "fp1_aaa")
        self.assertEqual(primary.drift, [])

    def test_a_drift_row_without_a_fingerprint_row_restores_its_canonical(self) -> None:
        tracker = HealthTracker.from_snapshot(
            {
                "version": 3,
                "records": [],
                "drift": [["answer", "q1", "fp1_aaa", "fp1_bbb"]],
            }
        )
        self.assertEqual(tracker.fingerprint_of("answer", "q1"), "fp1_aaa")
        self.assertEqual(len(tracker.drift), 1)

    def test_malformed_snapshot_fingerprints_and_drift_are_rejected(self) -> None:
        for payload in (
            {"version": 3, "records": [], "fingerprints": "nope"},
            {"version": 3, "records": [], "fingerprints": [["answer"]]},
            {"version": 3, "records": [], "fingerprints": [["answer", "q1"]]},
            {"version": 3, "records": [], "fingerprints": [["", "q1", "fp1_a"]]},
            {"version": 3, "records": [], "fingerprints": [["answer", "q1", 3]]},
            {"version": 3, "records": [], "drift": "nope"},
            {"version": 3, "records": [], "drift": [["answer", "q1", "fp1_a"]]},
            {"version": 3, "records": [], "drift": [["", "q1", "fp1_a", "fp1_b"]]},
            {"version": 3, "records": [], "drift": [["answer", "q1", "fp1_a", 3]]},
            {"version": 3, "records": [], "drift": [["answer", "q1", "fp1_a", "fp1_a"]]},
            {
                "version": 3,
                "records": [],
                "fingerprints": [["answer", "q1", "fp1_a"]],
                "drift": [["answer", "q1", "fp1_b", "fp1_c"]],
            },
            {
                "version": 3,
                "records": [],
                "fingerprints": [["answer", "q1", "fp1_a"], ["answer", "q1", "fp1_b"]],
            },
            {
                "version": 3,
                "records": [],
                "fingerprints": [["answer", "q1", "fp1_a"]],
                "drift": [
                    ["answer", "q1", "fp1_a", "fp1_b"],
                    ["answer", "q1", "fp1_x", "fp1_b"],
                ],
            },
        ):
            with self.assertRaises(ValueError):
                HealthTracker.from_snapshot(payload)


class RetrospectiveTrackerTest(unittest.TestCase):
    """The windowed read runs off the tracker's own timeline."""

    @staticmethod
    def _tracker(seed: int = 0) -> HealthTracker:
        tracker = HealthTracker(seed=seed, **FAST)
        tracker.timeline.extend(
            ScorePoint(score=40.0 + 2 * index, pairs=5000.0, epoch=f"e{index}")
            for index in range(6)
        )
        return tracker

    def test_the_timeline_is_pooled_into_one_reading(self) -> None:
        window = self._tracker().retrospective(window=6)
        self.assertEqual(window.points, 6)
        self.assertEqual(window.window, 6)
        self.assertEqual(window.status, "improving")
        self.assertGreater(window.score, 40.0)

    def test_rope_and_level_can_be_overridden(self) -> None:
        tracker = self._tracker()
        self.assertEqual(tracker.retrospective(window=6, rope_pp=100.0).status, "stable")
        self.assertEqual(tracker.retrospective(window=6, level=0.5).points, 6)
        with self.assertRaises(ValueError):
            tracker.retrospective(level=1.0)
        with self.assertRaises(ValueError):
            tracker.retrospective(window=0)

    def test_an_empty_timeline_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HealthTracker(**FAST).retrospective()

    def test_report_attaches_the_window_from_the_timeline(self) -> None:
        tracker = self._tracker()
        tracker.observe("v", "answer", "q1", False, True)
        report = tracker.report(retrospective_window=4)
        assert report.retrospective is not None
        self.assertEqual(report.retrospective.points, 4)
        self.assertEqual(report.retrospective.status, "improving")
        self.assertIsNone(tracker.report().retrospective)

    def test_advance_records_points_the_window_can_read(self) -> None:
        tracker = HealthTracker(seed=0, candidate_epoch="e0", **FAST)
        for epoch in range(4):
            for index in range(200):
                tracker.observe("v", "answer", f"q{index}", False, index < 100 + 20 * epoch)
            tracker.advance(baseline_epoch=f"e{epoch}", candidate_epoch=f"e{epoch + 1}")
        self.assertEqual(len(tracker.timeline), 4)
        window = tracker.retrospective(window=4)
        self.assertEqual(window.points, 4)
        self.assertEqual(window.status, "improving")


class BankCompositionTest(unittest.TestCase):
    """Timeline points carry the scored bank so attrition is not volatility."""

    def test_timeline_records_the_bank_and_counts(self) -> None:
        tracker = HealthTracker(seed=1, **FAST)
        for index in range(10):
            tracker.observe("v", "answer", f"q{index}", index < 5, index < 5)
        point = tracker.record_score(epoch="e1")
        assert point is not None
        self.assertIsNotNone(point.bank)
        self.assertEqual((point.paired, point.retired, point.failed), (10, 0, 0))
        self.assertEqual(point.score_held, point.score)

    def test_a_retired_bank_is_not_booked_as_volatility(self) -> None:
        tracker = HealthTracker(seed=1, **FAST)
        for index in range(100):
            tracker.observe("v", "answer", f"q{index}", index < 60, index < 60)
        tracker.record_score(epoch="e1")
        for index in range(20):
            tracker.retire("v", "answer", f"q{60 + index}")
        for index in range(100):
            if 60 <= index < 80:
                continue
            tracker.observe("v", "answer", f"q{index}", index < 60, index < 60)
        tracker.record_score(epoch="e2")
        band = tracker.score_band(window=2)
        self.assertEqual([point.score for point in tracker.timeline], [60.0, 75.0])
        self.assertEqual(tracker.timeline[1].retired, 20)
        self.assertIsNone(tracker.timeline[1].score_held)
        self.assertFalse(band.bank_stable)
        self.assertEqual(band.between_sd, 0.0)

    def test_a_stable_bank_keeps_the_old_band(self) -> None:
        tracker = HealthTracker(seed=1, **FAST)
        for index in range(100):
            tracker.observe("v", "answer", f"q{index}", index < 60, index < 60)
        tracker.record_score(epoch="e1")
        for index in range(100):
            tracker.observe("v", "answer", f"q{index}", index < 60, index < 75)
        tracker.record_score(epoch="e2")
        band = tracker.score_band(window=2)
        self.assertTrue(band.bank_stable)
        self.assertGreater(band.between_sd, 9.0)


if __name__ == "__main__":
    unittest.main()
