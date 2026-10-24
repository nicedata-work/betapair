"""Streaming and batch accumulation of paired outcomes.

Observations do not arrive all at once. Items are evaluated one by one (and a
stratum may only be re-evaluated when something it depends on changes), so the
report has to be computable at any point in the stream without waiting for the
last stratum — and cheap enough to recompute after every arrival.

:class:`HealthTracker` holds the *item-level* paired outcomes, not just
counts, which buys three properties a service needs:

* **idempotent streaming** — keying by ``(stratum, outcome, item)`` means a
  retried or replayed item overwrites its own row instead of double-counting,
  so at-least-once delivery is safe;
* **partial data** — ``report()`` always works; items whose candidate side has
  not landed yet are reported as ``pending`` and excluded from the score, so a
  half-finished stream never reads as a regression;
* **epoch rollover** — :meth:`HealthTracker.advance` promotes the candidate
  epoch to baseline and clears the other side, so a periodic refresh reuses
  the same tracker with memory bounded by one epoch of items.

Batching is :meth:`observe_many` (many records at once) plus :meth:`merge`
(fold a second tracker's snapshot in, e.g. one produced by another worker).
Persistence is :meth:`snapshot`/:meth:`from_snapshot` — plain JSON, no pickle,
no database.

Categories are per **stratum**, not per item: pass
``categories={"model_a": "tier-1"}`` once, or bind one later with
:meth:`set_category`. ``category_weights`` then says how much each category
counts in the aggregate score (``{"tier-1": 3.0}``); an unlisted category
weighs 1.0. When some strata are categorised and others are not, the
uncategorised ones take ``default_category`` — a comparison with no category at
all stays uniform. Categories, weights and the default are persisted in the
snapshot so a resumed tracker scores the same way it did before the restart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydfba.hierarchy import DRAWS, PPC_DRAWS, Cell
from pydfba.report import DEFAULT_CATEGORY, HealthReport, build_report

SNAPSHOT_VERSION = 1

CellKey = tuple[str, str]
Pair = tuple[bool | None, bool | None]


@dataclass(frozen=True)
class PairRecord:
    """One item's paired outcome: baseline epoch vs candidate epoch.

    ``None`` means "not evaluated in that epoch" — an errored, undecided or
    not-yet-re-run item. ``True`` is PASS, ``False`` FAIL; the tracker never
    folds errors into a failure.
    """

    stratum: str
    outcome: str
    item: str
    baseline: bool | None
    candidate: bool | None


class HealthTracker:
    """Accumulates paired outcomes per (view, outcome) and reports the metric."""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        category_weights: dict[str, float] | None = None,
        categories: dict[str, str] | None = None,
        default_category: str | None = DEFAULT_CATEGORY,
        a0: float = 1.0,
        b0: float = 1.0,
        prob_interval: float = 0.95,
        draws: int = DRAWS,
        ppc_draws: int = PPC_DRAWS,
        seed: int = 0,
        alpha: float = 0.05,
        regression_prob: float = 0.95,
        baseline_epoch: str | None = None,
        candidate_epoch: str | None = None,
    ) -> None:
        self.weights = dict(weights or {})
        self.category_weights = dict(category_weights or {})
        self.categories = dict(categories or {})
        self.default_category = default_category
        for name, value in (*self.weights.items(), *self.category_weights.items()):
            if value < 0.0:
                raise ValueError(f"weight for {name!r} must be >= 0, got {value}")
        self.a0 = a0
        self.b0 = b0
        self.prob_interval = prob_interval
        self.draws = draws
        self.ppc_draws = ppc_draws
        self.seed = seed
        self.alpha = alpha
        self.regression_prob = regression_prob
        self.baseline_epoch = baseline_epoch
        self.candidate_epoch = candidate_epoch
        self._records: dict[CellKey, dict[str, Pair]] = {}

    def observe(
        self,
        stratum: str,
        outcome: str,
        item: str,
        baseline: bool | None,
        candidate: bool | None,
    ) -> None:
        """Record (or overwrite) one item's paired outcome.

        @param stratum: View/stratum the item belongs to.
        @param outcome: Outcome name within the view.
        @param item: Stable item key; re-observing the same key overwrites.
        @param baseline: PASS/FAIL in the baseline epoch, ``None`` if not run.
        @param candidate: PASS/FAIL in the candidate epoch, ``None`` if pending.
        @raise ValueError: If any of ``stratum``/``outcome``/``item`` is empty.
        """
        if not stratum or not outcome or not item:
            raise ValueError("stratum, outcome and item must be non-empty")
        cell = self._records.setdefault((stratum, outcome), {})
        cell[item] = (baseline, candidate)

    def set_category(self, stratum: str, category: str | None) -> None:
        """Bind (or clear) the category of one stratum.

        Categories often arrive after the first observations — from an external
        catalogue a stream may not have loaded yet — so binding is separate
        from observing and costs nothing.
        """
        if category is None:
            self.categories.pop(stratum, None)
        else:
            self.categories[stratum] = category

    def category_of(self, stratum: str) -> str | None:
        """The tier a view belongs to, if one is bound."""
        return self.categories.get(stratum)

    def observe_many(self, records: list[PairRecord] | tuple[PairRecord, ...]) -> int:
        """Batch ingest — the nightly-refresh path. Returns the record count."""
        for record in records:
            self.observe(
                record.stratum,
                record.outcome,
                record.item,
                record.baseline,
                record.candidate,
            )
        return len(records)

    def merge(self, other: HealthTracker, *, prefer: Literal["other", "self"] = "other") -> None:
        """Fold another tracker's state in (same schema, any epoch).

        Per item, a value already present wins over ``None`` from the other
        side; when both carry a value, ``prefer`` decides whose is kept.

        @param other: Tracker whose records and config are merged in.
        @param prefer: Whose value to keep when both sides have one;
            ``"other"`` (default) is the fresher in-flight evaluation.
        """
        for key, items in other._records.items():
            target = self._records.setdefault(key, {})
            for item, (other_baseline, other_candidate) in items.items():
                if item not in target:
                    target[item] = (other_baseline, other_candidate)
                    continue
                baseline, candidate = target[item]
                if prefer == "self":
                    target[item] = (
                        baseline if baseline is not None else other_baseline,
                        candidate if candidate is not None else other_candidate,
                    )
                else:
                    target[item] = (
                        other_baseline if other_baseline is not None else baseline,
                        other_candidate if other_candidate is not None else candidate,
                    )
        for outcome, weight in other.weights.items():
            self.weights.setdefault(outcome, weight)
        for stratum, category in other.categories.items():
            self.categories.setdefault(stratum, category)
        for category, weight in other.category_weights.items():
            self.category_weights.setdefault(category, weight)

    def advance(
        self,
        *,
        baseline_epoch: str | None = None,
        candidate_epoch: str | None = None,
    ) -> None:
        """Roll the epoch: candidate becomes baseline, the candidate side clears.

        Call this at a refresh boundary. Rows that carry no values at all are
        dropped, so memory stays proportional to one epoch's item count.
        """
        rolled: dict[CellKey, dict[str, Pair]] = {}
        for key, items in self._records.items():
            kept: dict[str, Pair] = {}
            for item, (_, candidate) in items.items():
                if candidate is not None:
                    kept[item] = (candidate, None)
            if kept:
                rolled[key] = kept
        self._records = rolled
        self.baseline_epoch = baseline_epoch or self.candidate_epoch
        self.candidate_epoch = candidate_epoch

    @property
    def pending(self) -> int:
        """Records waiting for the other epoch's outcome."""
        return sum(
            1
            for items in self._records.values()
            for baseline, candidate in items.values()
            if baseline is None or candidate is None
        )

    @property
    def paired(self) -> int:
        return sum(
            1
            for items in self._records.values()
            for baseline, candidate in items.values()
            if baseline is not None and candidate is not None
        )

    def cells(self) -> list[Cell]:
        """Current per-cell discordant counts, sorted for determinism."""
        cells: list[Cell] = []
        for (stratum, outcome), items in sorted(self._records.items()):
            pairs = improved = regressed = passed = 0
            for baseline, candidate in items.values():
                if baseline is None or candidate is None:
                    continue
                pairs += 1
                passed += int(bool(candidate))
                improved += int(not baseline and bool(candidate))
                regressed += int(bool(baseline) and not candidate)
            if pairs:
                cells.append(
                    Cell(
                        stratum=stratum,
                        outcome=outcome,
                        improved=improved,
                        regressed=regressed,
                        passed=passed,
                        pairs=pairs,
                        category=self.categories.get(stratum),
                    )
                )
        return cells

    def report(
        self,
        *,
        draws: int | None = None,
        ppc_draws: int | None = None,
        seed: int | None = None,
        alpha: float | None = None,
        regression_prob: float | None = None,
    ) -> HealthReport:
        """Compute the metric from whatever has arrived so far."""
        return build_report(
            self.cells(),
            weights=self.weights,
            category_weights=self.category_weights,
            default_category=self.default_category,
            a0=self.a0,
            b0=self.b0,
            prob_interval=self.prob_interval,
            draws=self.draws if draws is None else draws,
            ppc_draws=self.ppc_draws if ppc_draws is None else ppc_draws,
            seed=self.seed if seed is None else seed,
            alpha=self.alpha if alpha is None else alpha,
            regression_prob=self.regression_prob if regression_prob is None else regression_prob,
            pending=self.pending,
            baseline_epoch=self.baseline_epoch,
            candidate_epoch=self.candidate_epoch,
        )

    def snapshot(self) -> dict[str, object]:
        """JSON-ready state; sorted so equal state serialises byte-identically."""
        records: list[list[object]] = []
        for (stratum, outcome), items in sorted(self._records.items()):
            for item, (baseline, candidate) in sorted(items.items()):
                records.append([stratum, outcome, item, baseline, candidate])
        return {
            "version": SNAPSHOT_VERSION,
            "weights": dict(sorted(self.weights.items())),
            "category_weights": dict(sorted(self.category_weights.items())),
            "categories": dict(sorted(self.categories.items())),
            "default_category": self.default_category,
            "baseline_epoch": self.baseline_epoch,
            "candidate_epoch": self.candidate_epoch,
            "records": records,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, object], **config: object) -> HealthTracker:
        """Rebuild a tracker from :meth:`snapshot` output (JSON already parsed).

        @param payload: Parsed :meth:`snapshot` mapping.
        @param config: Extra keyword arguments forwarded to ``__init__``.
        @raise ValueError: If the snapshot version is not supported.
        """
        version = payload.get("version", SNAPSHOT_VERSION)
        if version != SNAPSHOT_VERSION:
            raise ValueError(f"unsupported snapshot version {version!r}")
        tracker = cls(
            weights=dict(payload.get("weights") or {}),  # type: ignore[arg-type]
            category_weights=dict(payload.get("category_weights") or {}),  # type: ignore[arg-type]
            categories=dict(payload.get("categories") or {}),  # type: ignore[arg-type]
            default_category=payload.get("default_category", DEFAULT_CATEGORY),  # type: ignore[arg-type]
            baseline_epoch=payload.get("baseline_epoch"),  # type: ignore[arg-type]
            candidate_epoch=payload.get("candidate_epoch"),  # type: ignore[arg-type]
            **config,  # type: ignore[arg-type]
        )
        for raw in payload.get("records") or []:  # type: ignore[union-attr]
            stratum, outcome, item, baseline, candidate = raw
            tracker.observe(str(stratum), str(outcome), str(item), baseline, candidate)
        return tracker

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.snapshot(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str, **config: object) -> HealthTracker:
        return cls.from_snapshot(json.loads(text), **config)

    def __len__(self) -> int:
        """Number of tracked items, paired or pending."""
        return sum(len(items) for items in self._records.values())
