"""Streaming accumulation: partial refreshes, rollover, merges, snapshots."""

from __future__ import annotations

import json
import random
import unittest

from betapair import ERROR, FAILED, RETIRED, HealthTracker, PairRecord

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
        tracker = HealthTracker(**FAST)
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


if __name__ == "__main__":
    unittest.main()
