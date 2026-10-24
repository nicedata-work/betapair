"""Streaming accumulation: partial refreshes, rollover, merges, snapshots."""

from __future__ import annotations

import json
import random
import unittest

from pydfba import ERROR, HealthTracker, PairRecord

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
        self.assertEqual(payload["version"], 2)
        self.assertEqual(len(payload["records"]), 12)

    def test_snapshot_survives_a_json_round_trip_through_text(self) -> None:
        tracker = HealthTracker()
        tracker.observe("v", "answer", "q1", None, False)
        restored = HealthTracker.from_snapshot(json.loads(tracker.to_json()))
        self.assertEqual(restored.pending, 1)
        self.assertIsNone(restored._records[("v", "answer")]["q1"][0])

    def test_unknown_snapshot_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HealthTracker.from_snapshot({"version": 99, "records": []})

    def test_unversioned_snapshot_is_rejected(self) -> None:
        """Without a version there is nothing to check the shape against."""
        with self.assertRaises(ValueError):
            HealthTracker.from_snapshot({"records": []})

    def test_malformed_snapshot_is_rejected(self) -> None:
        for payload in (
            {"version": 2, "records": "nope"},
            {"version": 2, "records": [["v", "answer", "q1", "nonsense", True]]},
            {"version": 2, "records": [["v", "answer", "q1", True]]},
            {"version": 2, "records": [], "weights": {"answer": "heavy"}},
            {"version": 2, "records": [], "categories": {"v": 3}},
            {"version": 2, "records": [], "baseline_epoch": 7},
            {"version": 2, "records": [], "seed": "nope"},
        ):
            with self.assertRaises(ValueError):
                HealthTracker.from_snapshot(payload)

    def test_invalid_record_is_rejected(self) -> None:
        tracker = HealthTracker()
        with self.assertRaises(ValueError):
            tracker.observe("", "answer", "q1", True, False)


class SideValidationTest(unittest.TestCase):
    """A side is a real result, a pending ``None`` or :data:`ERROR` — nothing else."""

    def test_non_binary_sides_are_rejected(self) -> None:
        tracker = HealthTracker(**FAST)
        for junk in ("yes", "", 1, 0, "pending"):
            with self.assertRaises(ValueError):
                tracker.observe("v", "answer", "q1", junk, True)
            with self.assertRaises(ValueError):
                tracker.observe("v", "answer", "q1", True, junk)
        self.assertEqual(len(tracker), 0)

    def test_coverage_agrees_with_every_other_counter(self) -> None:
        """One record is paired, pending or errored in exactly one place."""
        tracker = HealthTracker(**FAST)
        tracker.observe("v", "answer", "q1", False, True)
        tracker.observe("v", "answer", "q2", False, None)
        tracker.observe("v", "answer", "q3", False, ERROR)
        coverage = tracker.coverage()[0]
        self.assertEqual(coverage.paired, tracker.paired)
        self.assertEqual(coverage.pending, tracker.pending)
        self.assertEqual(coverage.errored, tracker.errored)
        self.assertEqual(coverage.paired, sum(cell.pairs for cell in tracker.cells()))
        self.assertEqual(coverage.paired + coverage.pending + coverage.errored, len(tracker))


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


if __name__ == "__main__":
    unittest.main()
