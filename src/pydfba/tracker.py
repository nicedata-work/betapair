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
import random
from dataclasses import dataclass
from typing import Literal

from pydfba.hierarchy import DRAWS, PPC_DRAWS, Cell
from pydfba.report import DEFAULT_CATEGORY, CellCoverage, HealthReport, build_report

SNAPSHOT_VERSION = 2


class _Errored:
    """Sentinel for an evaluation that failed, distinct from pending ``None``.

    ``None`` means "not evaluated yet"; ``ERROR`` means "evaluated and the
    evaluation itself failed". Both are unresolved, but an error is a health
    signal in its own right and must survive a snapshot, so it gets its own
    state instead of being folded into pending.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "ERROR"


#: Marker for a side whose evaluation errored (as opposed to ``None`` = pending).
ERROR = _Errored()

Side = bool | None | _Errored
CellKey = tuple[str, str]
Pair = tuple[Side, Side]


#: One record's state. The single predicate every counter uses, so
#: ``coverage()``, ``cells()`` and the ``paired``/``pending``/``errored``
#: properties can never disagree about a record.
State = Literal["paired", "pending", "errored"]


def _errored(value: object) -> bool:
    return isinstance(value, _Errored)


def _state(baseline: Side, candidate: Side) -> State:
    """A record is ``paired`` only when both sides are real results."""
    if _errored(baseline) or _errored(candidate):
        return "errored"
    if baseline is None or candidate is None:
        return "pending"
    return "paired"


def _check_side(value: object, field: str) -> Side:
    """Narrow a caller-supplied side, rejecting anything off the type.

    @raise ValueError: If ``value`` is not ``True``, ``False``, ``None`` or
        :data:`ERROR`.
    """
    if value is None or isinstance(value, bool) or _errored(value):
        return value  # type: ignore[return-value]
    raise ValueError(f"{field} must be True, False, None or ERROR, got {value!r}")


def _serialise_side(value: Side) -> object:
    return "error" if _errored(value) else value


def _parse_side(value: object, field: str = "side") -> Side:
    if value == "error":
        return ERROR
    return _check_side(value, field)


def _text_or_none(value: object, field: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"snapshot {field} must be a string or null, got {value!r}")


def _number_mapping(value: object, field: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"snapshot {field} must be an object, got {value!r}")
    out: dict[str, float] = {}
    for name, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"snapshot {field}[{name!r}] must be a number, got {raw!r}")
        out[str(name)] = float(raw)
    return out


def _string_mapping(value: object, field: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"snapshot {field} must be an object, got {value!r}")
    out: dict[str, str] = {}
    for name, raw in value.items():
        if not isinstance(raw, str):
            raise ValueError(f"snapshot {field}[{name!r}] must be a string, got {raw!r}")
        out[str(name)] = raw
    return out


def _record_rows(value: object) -> list[list[object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"snapshot records must be a list, got {value!r}")
    rows: list[list[object]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 5:
            raise ValueError(
                "snapshot records must be [stratum, outcome, item, baseline, candidate], "
                f"got {raw!r}"
            )
        rows.append(list(raw))
    return rows


def _generate_seed() -> int:
    """A fresh seed for the Monte-Carlo draws.

    Generated once when a tracker is created and persisted in the snapshot, so
    every later refresh of the same tracker reuses the same random stream.
    Without that, a restart would redraw the intervals for unchanged data and
    a monitoring series would jitter for no reason.
    """
    return random.SystemRandom().randrange(0, 2**31)


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
    baseline: Side
    candidate: Side


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
        seed: int | None = None,
        alpha: float = 0.05,
        regression_prob: float | None = None,
        a_d: float = 1.0,
        b_d: float = 1.0,
        rope_pp: float = 1.0,
        pi_prior: tuple[float, float] = (1.0, 1.0),
        pi_points: int = 199,
        max_unresolved_share: float = 0.25,
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
        self.seed = seed if seed is not None else _generate_seed()
        self.alpha = alpha
        self.regression_prob = regression_prob
        self.a_d = a_d
        self.b_d = b_d
        self.rope_pp = rope_pp
        self.pi_prior = pi_prior
        self.pi_points = pi_points
        self.max_unresolved_share = max_unresolved_share
        self.baseline_epoch = baseline_epoch
        self.candidate_epoch = candidate_epoch
        self._records: dict[CellKey, dict[str, Pair]] = {}

    def observe(
        self,
        stratum: str,
        outcome: str,
        item: str,
        baseline: Side,
        candidate: Side,
    ) -> None:
        """Record (or overwrite) one item's paired outcome.

        @param stratum: View/stratum the item belongs to.
        @param outcome: Outcome name within the view.
        @param item: Stable item key; re-observing the same key overwrites.
        @param baseline: PASS/FAIL in the baseline epoch, ``None`` if not run,
            or :data:`ERROR` if the evaluation failed.
        @param candidate: PASS/FAIL in the candidate epoch, ``None`` if
            pending, or :data:`ERROR` if the evaluation failed.
        @raise ValueError: If any of ``stratum``/``outcome``/``item`` is empty.
        """
        if not stratum or not outcome or not item:
            raise ValueError("stratum, outcome and item must be non-empty")
        cell = self._records.setdefault((stratum, outcome), {})
        cell[item] = (
            _check_side(baseline, "baseline"),
            _check_side(candidate, "candidate"),
        )

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

    def merge(
        self,
        other: HealthTracker,
        *,
        prefer: Literal["other", "self"] = "other",
        on_epoch_mismatch: Literal["raise", "ignore"] = "raise",
    ) -> None:
        """Fold another tracker's state in.

        Per item, a value already present wins over ``None`` from the other
        side; when both carry a value, ``prefer`` decides whose is kept. The
        same ``prefer`` also decides whose weights and categories win, so the
        record and the aggregation can never disagree about whose state this
        is. Epochs must match unless ``on_epoch_mismatch="ignore"``: merging
        two different comparisons and calling the result one report would
        silently mix baselines.

        @param other: Tracker whose records and config are merged in.
        @param prefer: Whose value to keep when both sides have one;
            ``"other"`` (default) is the fresher in-flight evaluation.
        @param on_epoch_mismatch: What to do when the epochs differ.
        @raise ValueError: If the epochs differ and ``on_epoch_mismatch`` is
            ``"raise"`` (the default).
        """
        if on_epoch_mismatch not in ("raise", "ignore"):
            raise ValueError(f"unknown on_epoch_mismatch {on_epoch_mismatch!r}")
        if on_epoch_mismatch == "raise" and (
            self.baseline_epoch != other.baseline_epoch
            or self.candidate_epoch != other.candidate_epoch
        ):
            raise ValueError(
                "cannot merge trackers from different epochs: "
                f"self=({self.baseline_epoch!r}, {self.candidate_epoch!r}) "
                f"other=({other.baseline_epoch!r}, {other.candidate_epoch!r}); "
                "pass on_epoch_mismatch='ignore' to merge anyway"
            )
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
        if prefer == "other":
            self.weights.update(other.weights)
            self.categories.update(other.categories)
            self.category_weights.update(other.category_weights)
        else:
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

        @note: Only a real result can become the next baseline; a pending or
            errored candidate has nothing to promote.
        """
        rolled: dict[CellKey, dict[str, Pair]] = {}
        for key, items in self._records.items():
            kept: dict[str, Pair] = {}
            for item, (_, candidate) in items.items():
                if isinstance(candidate, bool):
                    kept[item] = (candidate, None)
            if kept:
                rolled[key] = kept
        self._records = rolled
        self.baseline_epoch = baseline_epoch or self.candidate_epoch
        self.candidate_epoch = candidate_epoch

    def _count(self, state: State) -> int:
        return sum(
            1
            for items in self._records.values()
            for baseline, candidate in items.values()
            if _state(baseline, candidate) == state
        )

    @property
    def pending(self) -> int:
        """Records waiting for an outcome that has not been produced yet."""
        return self._count("pending")

    @property
    def errored(self) -> int:
        """Records where an evaluation failed on either side."""
        return self._count("errored")

    @property
    def paired(self) -> int:
        return self._count("paired")

    def coverage(self) -> list[CellCoverage]:
        """Per-cell resolved/pending/errored counts, sorted for determinism."""
        out: list[CellCoverage] = []
        for (stratum, outcome), items in sorted(self._records.items()):
            counts: dict[State, int] = {"paired": 0, "pending": 0, "errored": 0}
            for baseline, candidate in items.values():
                counts[_state(baseline, candidate)] += 1
            out.append(
                CellCoverage(
                    stratum=stratum,
                    outcome=outcome,
                    category=self.categories.get(stratum),
                    paired=counts["paired"],
                    pending=counts["pending"],
                    errored=counts["errored"],
                )
            )
        return out

    def cells(self) -> list[Cell]:
        """Current per-cell discordant counts, sorted for determinism."""
        cells: list[Cell] = []
        for (stratum, outcome), items in sorted(self._records.items()):
            pairs = improved = regressed = passed = 0
            for baseline, candidate in items.values():
                if _state(baseline, candidate) != "paired":
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
        rope_pp: float | None = None,
        max_unresolved_share: float | None = None,
    ) -> HealthReport:
        """Compute the metric from whatever has arrived so far."""
        return build_report(
            self.cells(),
            weights=self.weights,
            category_weights=self.category_weights,
            default_category=self.default_category,
            a0=self.a0,
            b0=self.b0,
            a_d=self.a_d,
            b_d=self.b_d,
            prob_interval=self.prob_interval,
            draws=self.draws if draws is None else draws,
            ppc_draws=self.ppc_draws if ppc_draws is None else ppc_draws,
            seed=self.seed if seed is None else seed,
            alpha=self.alpha if alpha is None else alpha,
            regression_prob=self.regression_prob if regression_prob is None else regression_prob,
            rope_pp=self.rope_pp if rope_pp is None else rope_pp,
            pi_prior=self.pi_prior,
            pi_points=self.pi_points,
            pending=self.pending,
            coverage=self.coverage(),
            max_unresolved_share=(
                self.max_unresolved_share if max_unresolved_share is None else max_unresolved_share
            ),
            baseline_epoch=self.baseline_epoch,
            candidate_epoch=self.candidate_epoch,
        )

    def snapshot(self) -> dict[str, object]:
        """JSON-ready state; sorted so equal state serialises byte-identically."""
        records: list[list[object]] = []
        for (stratum, outcome), items in sorted(self._records.items()):
            for item, (baseline, candidate) in sorted(items.items()):
                records.append(
                    [
                        stratum,
                        outcome,
                        item,
                        _serialise_side(baseline),
                        _serialise_side(candidate),
                    ]
                )
        return {
            "version": SNAPSHOT_VERSION,
            "seed": self.seed,
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
        @raise ValueError: If the snapshot version is missing or not supported,
            or a record is not a five-element ``[stratum, outcome, item,
            baseline, candidate]`` list with valid sides.
        """
        version = payload.get("version")
        if version != SNAPSHOT_VERSION:
            raise ValueError(
                f"unsupported snapshot version {version!r}; expected {SNAPSHOT_VERSION}"
            )
        seed = config.pop("seed", payload.get("seed"))
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"snapshot seed must be an integer or null, got {seed!r}")
        tracker = cls(
            weights=_number_mapping(payload.get("weights"), "weights"),
            category_weights=_number_mapping(payload.get("category_weights"), "category_weights"),
            categories=_string_mapping(payload.get("categories"), "categories"),
            default_category=_text_or_none(
                payload.get("default_category", DEFAULT_CATEGORY), "default_category"
            ),
            seed=seed,
            baseline_epoch=_text_or_none(payload.get("baseline_epoch"), "baseline_epoch"),
            candidate_epoch=_text_or_none(payload.get("candidate_epoch"), "candidate_epoch"),
            **config,  # type: ignore[arg-type]
        )
        for raw in _record_rows(payload.get("records")):
            stratum, outcome, item, baseline, candidate = raw
            tracker.observe(
                str(stratum),
                str(outcome),
                str(item),
                _parse_side(baseline, "baseline"),
                _parse_side(candidate, "candidate"),
            )
        return tracker

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.snapshot(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str, **config: object) -> HealthTracker:
        return cls.from_snapshot(json.loads(text), **config)

    def __len__(self) -> int:
        """Number of tracked items, paired or pending."""
        return sum(len(items) for items in self._records.values())
