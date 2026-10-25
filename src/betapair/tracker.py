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
* **bank churn** — :meth:`HealthTracker.retire` withdraws an item from the
  bank. A retired item is not *pending* (nothing is outstanding) and not
  *errored* (nothing failed), so it does not gate coverage. Its last real
  outcome is kept in :attr:`HealthTracker.retired_ledger` and folded back into
  ``score_held``, so the part of the level caused by shrinking the bank
  (``composition_effect_pp``) is separable from the part caused by the views
  changing;
* **error attribution** — an evaluation can fail because *the view* failed
  (its SQL timed out, its query OOM'd the warehouse, its output crashed the
  judge) or because *the harness* failed (a worker died, an API returned 5xx).
  The two are not the same: a view-caused failure is a result, so it is passed
  as :data:`FAILED` and scored as a failure, keeping it in the level and the
  paired test. A harness-caused failure is genuinely unresolved and is passed
  as :data:`ERROR`, so it stays out of the score and gates coverage. Without
  the split, erroring out of the questions you were failing would raise the
  score exactly as deleting them would, and only one of those keeps a trace;
* **epoch rollover** — :meth:`HealthTracker.advance` promotes the candidate
  epoch to baseline and clears the other side, so a periodic refresh reuses
  the same tracker with memory bounded by one epoch of items, while
  :attr:`HealthTracker.timeline` keeps one :class:`~betapair.series.ScorePoint`
  per finished epoch for :meth:`HealthTracker.score_band`.

Batching is :meth:`observe_many` (many records at once) plus :meth:`merge`
(fold a second tracker's snapshot in, e.g. one produced by another worker).
Persistence is :meth:`snapshot`/:meth:`from_snapshot` — plain JSON, no pickle,
no database.

Item identity is local to a cell for streaming — ``(stratum, outcome, item)``
identifies a row, so a replayed item overwrites itself. The *global* identity a
difficulty model needs is different: ``(outcome, item)``, so the same question
observed under two views is **one** item. The caller owns that contract (the
``item`` string must be stable and shared across views); :meth:`item_responses`
exposes the resolved rows under it. See :mod:`betapair.irt`.

Categories are per **stratum**, not per item: pass
``categories={"model_a": "tier-1"}`` once, or bind one later with
:meth:`set_category`. ``category_weights`` then says how much each category
counts in the aggregate score (``{"tier-1": 3.0}``); an unlisted category
weighs 1.0. When some strata are categorised and others are not, the
uncategorised ones take ``default_category`` — a comparison with no category at
all stays uniform. Categories, weights and the default are persisted in the
snapshot so a resumed tracker scores the same way it did before the restart.

**Groups** are the second per-stratum label, beside the tier. Pass
``groups={"model_a": "orders_table"}`` (or bind one later with
:meth:`set_group`) to say which shared upstream dependency a view belongs to.
Unlike a category, a group carries no weight: it changes no number on its own.
It is carried onto every cell and report row so cells that fail *together* —
twenty views on one source table — can be recognised as one incident rather
than twenty independent regressions. Groups are persisted exactly like
categories.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from betapair.hierarchy import DRAWS, PPC_DRAWS, Cell
from betapair.irt import DEFAULT_SIGMA, ItemKey, ItemResponse, ReferenceBank
from betapair.report import (
    DEFAULT_CATEGORY,
    CellCoverage,
    HealthReport,
    UnpairedCell,
    build_report,
    weighting,
)
from betapair.series import ScoreBand, ScorePoint, score_band

#: Snapshot *format* version — internal to the JSON payload and independent
#: of the package version in ``pyproject.toml``. It moves when the payload
#: shape changes (v3 added the score ``timeline``); :meth:`from_snapshot`
#: rejects any other version. Within a version, additive keys are fine:
#: a v3 payload with no ``timeline`` still loads, defaulted to empty, and one
#: with no ``retired_ledger`` loads with an empty ledger. ``groups`` joined the
#: same way: absent means no view has a shared-dependency label.
SNAPSHOT_VERSION = 3


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


class _Retired:
    """Sentinel for an item withdrawn from the question bank.

    ``None`` means "not evaluated yet"; :data:`ERROR` means "evaluated and the
    evaluation itself failed"; ``RETIRED`` means "no longer part of the bank".
    A retired item is a *structural* change to the comparison rather than
    missing data, so it is excluded from the score and the paired test without
    counting as unresolved — and it is counted on its own so a shrinking bank
    is visible instead of silently redefining the denominator.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "RETIRED"


#: Marker for a side whose item has been withdrawn from the bank.
RETIRED = _Retired()


class _Failed:
    """Sentinel for a *view-caused* evaluation failure, scored as a failure.

    ``False`` means "evaluated and the item did not pass"; :data:`FAILED` means
    "evaluated and the evaluation itself failed because the view did". The two
    are kept apart so the failure stays visible in the counts, but they score
    the same: a view that times out on the questions it was failing must not
    improve its own level by removing them from the denominator. Because a
    failure is a real result, the record stays ``paired`` — it enters the level
    and the paired test with no ledger and no composition term.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "FAILED"


#: Marker for a side whose evaluation failed and the failure was the view's fault.
FAILED = _Failed()

Side = bool | None | _Errored | _Retired | _Failed
CellKey = tuple[str, str]
Pair = tuple[Side, Side]


#: One record's state. The single predicate every counter uses, so
#: ``coverage()``, ``cells()`` and the ``paired``/``pending``/``errored``/
#: ``retired`` properties can never disagree about a record.
State = Literal["paired", "pending", "errored", "retired"]


def _errored(value: object) -> bool:
    return isinstance(value, _Errored)


def _retired(value: object) -> bool:
    return isinstance(value, _Retired)


def _failed(value: object) -> bool:
    return isinstance(value, _Failed)


def _outcome(value: Side) -> bool | None:
    """The result a side *scores as*, or ``None`` when it has no result.

    :data:`FAILED` is a view-caused failure, so it scores as ``False``. That
    one line is what makes a failure enter the level instead of shrinking its
    denominator. ``None``, :data:`ERROR` and :data:`RETIRED` carry no result.
    """
    if isinstance(value, bool):
        return value
    if _failed(value):
        return False
    return None


def _last_result(*sides: Side) -> bool | None:
    """The first real outcome among ``sides``, or ``None`` if there is none.

    Callers pass the more recent side first, so a retirement keeps the latest
    thing actually known about the item rather than an older one. A
    view-caused failure counts as a real outcome: it last scored as ``False``.
    """
    for side in sides:
        result = _outcome(side)
        if result is not None:
            return result
    return None


def _state(baseline: Side, candidate: Side) -> State:
    """A record is ``paired`` only when both sides are real results.

    @note: Retirement outranks the other states. It is a statement about the
        item, not about an evaluation attempt, so it holds whichever side
        carries it. A harness error outranks a view-caused failure for the same
        reason in reverse: an unresolved record cannot be scored.
    """
    if _retired(baseline) or _retired(candidate):
        return "retired"
    if _errored(baseline) or _errored(candidate):
        return "errored"
    if baseline is None or candidate is None:
        return "pending"
    # A real result on both sides. A ``FAILED`` side is a result (it scores as
    # a failure), so it lands here as ``paired`` rather than gating coverage.
    return "paired"


def _check_side(value: object, field: str) -> Side:
    """Narrow a caller-supplied side, rejecting anything off the type.

    @raise ValueError: If ``value`` is not ``True``, ``False``, ``None``,
        :data:`ERROR`, :data:`FAILED` or :data:`RETIRED`.
    """
    if (
        value is None
        or isinstance(value, bool)
        or _errored(value)
        or _retired(value)
        or _failed(value)
    ):
        return value  # type: ignore[return-value]
    raise ValueError(f"{field} must be True, False, None, ERROR, RETIRED or FAILED, got {value!r}")


def _serialise_side(value: Side) -> object:
    if _errored(value):
        return "error"
    if _retired(value):
        return "retired"
    if _failed(value):
        return "failed"
    return value


def _parse_side(value: object, field: str = "side") -> Side:
    if value == "error":
        return ERROR
    if value == "retired":
        return RETIRED
    if value == "failed":
        return FAILED
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


def _positive_number(value: object, field: str) -> float:
    """Validate a positive numeric configuration value (rejects bools)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number, got {value!r}")
    if value <= 0.0:
        raise ValueError(f"{field} must be positive, got {value}")
    return float(value)


def _boolean(value: object, field: str) -> bool:
    """Validate a boolean configuration value (rejects the usual truthy ints)."""
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool, got {value!r}")
    return value


def _parse_reference_items(value: object) -> tuple[ItemKey, ...] | None:
    """Validate a pinned reference bank, or ``None`` for the observed union.

    The bank is a sequence of global ``(outcome, item)`` keys. ``None`` means
    each outcome's observed union; an explicitly empty bank is rejected,
    because it would equate onto nothing rather than onto the union.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"reference_items must be a list or null, got {value!r}")
    items: list[ItemKey] = []
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"reference item must be [outcome, item], got {entry!r}")
        outcome, item = entry
        if not isinstance(outcome, str) or not isinstance(item, str):
            raise ValueError(f"reference item must be [outcome, item] strings, got {entry!r}")
        if not outcome or not item:
            raise ValueError(f"reference item parts must be non-empty, got {entry!r}")
        items.append((outcome, item))
    if not items:
        raise ValueError("reference_items must not be empty; use None for the observed union")
    if len(set(items)) != len(items):
        raise ValueError("reference_items must not repeat an item")
    return tuple(sorted(items))


def _parse_reference_bank(value: object) -> ReferenceBank | None:
    """Validate a snapshot-encoded versioned reference bank.

    A bank is ``{"name", "version", "items": [[outcome, item], ...]}``. The
    item rows go through the same validation as ``reference_items`` (non-empty,
    no repeats) and :class:`~betapair.irt.ReferenceBank` re-checks its own
    invariants.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"reference_bank must be an object or null, got {value!r}")
    name = value.get("name")
    version = value.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError("reference_bank name and version must be strings")
    items = _parse_reference_items(value.get("items"))
    if items is None:
        raise ValueError("reference_bank items must be a non-empty list")
    return ReferenceBank(name=name, version=version, items=items)


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


def _ledger_rows(value: object) -> list[list[object]]:
    """Validate the retirement ledger's ``[stratum, outcome, item, last]`` rows."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"snapshot retired_ledger must be a list, got {value!r}")
    rows: list[list[object]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError(
                f"snapshot retired_ledger rows must be [stratum, outcome, item, last], got {raw!r}"
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

    ``None`` means "not evaluated in that epoch", :data:`ERROR` that the
    harness failed to evaluate it, :data:`FAILED` that the view caused the
    evaluation to fail, :data:`RETIRED` that the item has left the bank.
    ``True`` is PASS, ``False`` FAIL; the tracker never folds *harness* errors
    or retirement into a failure. A :data:`FAILED` side **is** scored as a
    failure, because a view that breaks on a question has answered it.

    ``item`` is the caller's stable, cross-view-shared identifier of the
    question, not of the row; see :mod:`betapair.irt` for the contract a
    difficulty model depends on.
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
        groups: dict[str, str] | None = None,
        default_category: str | None = DEFAULT_CATEGORY,
        a0: float = 1.0,
        b0: float = 1.0,
        prob_interval: float = 0.95,
        draws: int = DRAWS,
        ppc_draws: int = PPC_DRAWS,
        seed: int | None = None,
        alpha: float = 0.05,
        a_d: float = 1.0,
        b_d: float = 1.0,
        rope_pp: float = 1.0,
        pi_prior: tuple[float, float] = (1.0, 1.0),
        pi_points: int = 199,
        max_unresolved_share: float = 0.25,
        baseline_epoch: str | None = None,
        candidate_epoch: str | None = None,
        adjust_difficulty: bool = False,
        sigma_theta: float = DEFAULT_SIGMA,
        sigma_d: float = DEFAULT_SIGMA,
        reference_items: Sequence[ItemKey] | None = None,
        reference_bank: ReferenceBank | None = None,
    ) -> None:
        self.weights = dict(weights or {})
        self.category_weights = dict(category_weights or {})
        self.categories = dict(categories or {})
        self.groups = dict(groups or {})
        """Shared upstream dependency per stratum, beside :attr:`categories`.

        A label only: it is carried onto every cell and report row so cells
        that fail together can be grouped into one incident. It never changes
        the score, the weighting or the gate.
        """
        for stratum, group in self.groups.items():
            if not group:
                raise ValueError(f"group for {stratum!r} must be non-empty, got {group!r}")
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
        self.a_d = a_d
        self.b_d = b_d
        self.rope_pp = rope_pp
        self.pi_prior = pi_prior
        self.pi_points = pi_points
        self.max_unresolved_share = max_unresolved_share
        self.baseline_epoch = baseline_epoch
        self.candidate_epoch = candidate_epoch
        self.adjust_difficulty = _boolean(adjust_difficulty, "adjust_difficulty")
        """Whether :meth:`report` fits the difficulty model by default.

        Off by default: the adjustment changes what the headline level means,
        so it is opted into. Persisted, so a resumed tracker scores the same
        way it did before the restart.
        """
        self.sigma_theta = _positive_number(sigma_theta, "sigma_theta")
        self.sigma_d = _positive_number(sigma_d, "sigma_d")
        """Prior scales for the difficulty fit (see :func:`betapair.irt.fit_rasch`)."""
        self.reference_items = _parse_reference_items(reference_items)
        """Pinned reference bank, or ``None`` for each outcome's observed union."""
        if reference_bank is not None and not isinstance(reference_bank, ReferenceBank):
            raise ValueError(
                f"reference_bank must be a ReferenceBank or None, got {reference_bank!r}"
            )
        if reference_bank is not None and self.reference_items is not None:
            raise ValueError("pass reference_bank or reference_items, not both")
        self.reference_bank = reference_bank
        """Versioned pinned bank (``irt.ReferenceBank``), or ``None``.

        Preferred over :attr:`reference_items` for a long-lived standard bank:
        it carries a name, a version and a content digest, all published on the
        report's ``irt.reference`` so the adjusted score's meaning is auditable
        across epochs. Mutually exclusive with :attr:`reference_items`.
        """
        self._records: dict[CellKey, dict[str, Pair]] = {}
        self.timeline: list[ScorePoint] = []
        self._last_pi_mean: float | None = None
        """The share of cells that truly regressed in the last report built.

        Recorded onto the next :meth:`record_score` point so the following
        epoch's calibration can set its regression prior from it. Kept in
        memory only: a timeline point carries its own ``pi_mean``, and a fresh
        :meth:`report` recomputes this.
        """
        self.retired_ledger: dict[CellKey, dict[str, bool | None]] = {}
        """Last real outcome of every retired item, kept past :meth:`advance`.

        ``advance`` drops a retired row, so without this the bank could shrink
        with no trace. The ledger is what lets a composition-vs-change split
        tell a level move caused by dropping items apart from one caused by the
        views changing. ``None`` means the item was withdrawn before it was
        ever measured in a paired epoch.
        """

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
            Following the IRT contract, it must be identical for the same
            underlying question across views and epochs, because
            :meth:`item_responses` keys items globally by
            ``(outcome, item)`` — see :mod:`betapair.irt`.
        @param baseline: PASS/FAIL in the baseline epoch, ``None`` if not run,
            :data:`ERROR` if the harness failed to evaluate it, or
            :data:`FAILED` if the view itself failed.
        @param candidate: PASS/FAIL in the candidate epoch, ``None`` if
            pending, :data:`ERROR` if the harness failed to evaluate it, or
            :data:`FAILED` if the view itself failed.
        @raise ValueError: If any of ``stratum``/``outcome``/``item`` is empty.
        """
        if not stratum or not outcome or not item:
            raise ValueError("stratum, outcome and item must be non-empty")
        key = (stratum, outcome)
        baseline_side = _check_side(baseline, "baseline")
        candidate_side = _check_side(candidate, "candidate")
        self._records.setdefault(key, {})[item] = (baseline_side, candidate_side)
        if _retired(baseline_side) or _retired(candidate_side):
            self.retired_ledger.setdefault(key, {})[item] = _last_result(
                candidate_side, baseline_side
            )

    def retire(self, stratum: str, outcome: str, item: str) -> None:
        """Withdraw an item from the bank, keeping its last known outcome.

        A retired item is neither pending nor errored: it is structurally gone
        from the comparison, so it leaves the score and the paired test without
        counting as unresolved data — which is why it does not gate coverage.
        Its last real outcome is written to :attr:`retired_ledger`.

        Retiring an item the tracker has never seen is allowed and records
        ``None``, meaning "withdrawn before it was ever measured".

        @note: The tracker cannot infer retirement from silence — an item that
            is simply never observed again stays ``pending``, exactly as
            before. Only an explicit retirement makes the bank change visible.
        @raise ValueError: If any of ``stratum``/``outcome``/``item`` is empty.
        """
        if not stratum or not outcome or not item:
            raise ValueError("stratum, outcome and item must be non-empty")
        stored = self._records.get((stratum, outcome), {}).get(item)
        baseline, candidate = stored if stored is not None else (None, None)
        self.observe(stratum, outcome, item, _last_result(candidate, baseline), RETIRED)

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

    def set_group(self, stratum: str, group: str | None) -> None:
        """Bind (or clear) the shared-dependency group of one stratum.

        Mirrors :meth:`set_category`: the label may arrive from an external
        catalogue after the first observations, so binding is separate from
        observing. ``None`` clears it; the group changes no number by itself,
        it only lets cells that fail together be reported as one incident.

        @raise ValueError: If ``group`` is an empty string — ``None`` clears,
            an empty string is never a label.
        """
        if group is not None and not group:
            raise ValueError("group must be non-empty when set")
        if group is None:
            self.groups.pop(stratum, None)
        else:
            self.groups[stratum] = group

    def group_of(self, stratum: str) -> str | None:
        """The upstream dependency group a view belongs to, if one is bound."""
        return self.groups.get(stratum)

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
        @note: Only current records and config are folded in. Each tracker
            keeps its own :attr:`timeline`; merging histories is the caller's
            decision, because only the caller knows whether the two series
            describe the same epochs.
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
            self.groups.update(other.groups)
            self.category_weights.update(other.category_weights)
        else:
            for outcome, weight in other.weights.items():
                self.weights.setdefault(outcome, weight)
            for stratum, category in other.categories.items():
                self.categories.setdefault(stratum, category)
            for stratum, group in other.groups.items():
                self.groups.setdefault(stratum, group)
            for category, weight in other.category_weights.items():
                self.category_weights.setdefault(category, weight)

    def advance(
        self,
        *,
        baseline_epoch: str | None = None,
        candidate_epoch: str | None = None,
        record: bool = True,
    ) -> None:
        """Roll the epoch: candidate becomes baseline, the candidate side clears.

        Call this at a refresh boundary. Rows that carry no values at all are
        dropped, so memory stays proportional to one epoch's item count.

        @param baseline_epoch: Label of the epoch that just finished; defaults
            to the current candidate label.
        @param candidate_epoch: Label of the next candidate epoch.
        @param record: Append the finished epoch's score to :attr:`timeline`
            before rolling, so a chain of advances preserves the history a
            :meth:`score_band` needs.
        @note: Only a real result can become the next baseline. A
            :data:`FAILED` candidate is a real result and is promoted as
            ``False``; a pending, errored or retired one has nothing to
            promote.
        """
        if record:
            finished = self.candidate_epoch if baseline_epoch is None else baseline_epoch
            self.record_score(epoch=finished)
        rolled: dict[CellKey, dict[str, Pair]] = {}
        for key, items in self._records.items():
            kept: dict[str, Pair] = {}
            for item, (_, candidate) in items.items():
                # Only a real result can become the next baseline. Pending,
                # errored and retired rows are dropped here — a retired item is
                # already recorded in ``retired_ledger``, so the bank change
                # survives the rollover even though the row does not. A
                # view-caused failure is a real result and promotes as FAIL.
                outcome = _outcome(candidate)
                if outcome is not None:
                    kept[item] = (outcome, None)
            if kept:
                rolled[key] = kept
        self._records = rolled
        self.baseline_epoch = baseline_epoch or self.candidate_epoch
        self.candidate_epoch = candidate_epoch
        # The last report described the epoch that just closed; a point
        # recorded in the new epoch without a fresh report has no base rate
        # yet, and must not inherit the finished epoch's.
        self._last_pi_mean = None

    def _score_point(self, epoch: str | None = None) -> ScorePoint | None:
        """Current epoch's level with its effective pair count, or ``None``.

        Cheap on purpose: it re-uses the report's weighting policy but runs no
        sampling, so :meth:`advance` can record a point without paying for a
        full :meth:`report`.
        """
        cells = self.cells()
        if not cells:
            return None
        policy = weighting(
            cells,
            weights=self.weights,
            category_weights=self.category_weights,
            default_category=self.default_category,
        )
        pairs = sum(policy.weight(cell) * cell.pairs for cell in cells)
        if pairs <= 0.0:
            return None
        passed = sum(policy.weight(cell) * cell.passed for cell in cells)
        return ScorePoint(
            score=100.0 * passed / pairs,
            epoch=self.candidate_epoch if epoch is None else epoch,
            pairs=pairs,
            pi_mean=self._last_pi_mean,
            unresolved_share=self._unresolved_share(),
        )

    def record_score(self, *, epoch: str | None = None) -> ScorePoint | None:
        """Append the current epoch's score to :attr:`timeline`.

        Returns the recorded point, or ``None`` when there is nothing paired
        to score — an empty or fully pending epoch is not a data point. Use
        this for refresh-based series; :meth:`advance` calls it for you at an
        epoch boundary.
        """
        point = self._score_point(epoch)
        if point is not None:
            self.timeline.append(point)
        return point

    def score_band(self, *, window: int | None = None, level: float = 0.95) -> ScoreBand:
        """Reference band over :attr:`timeline`; see :func:`betapair.series.score_band`.

        @raise ValueError: If the timeline is empty (no epoch recorded yet).
        """
        return score_band(self.timeline, window=window, level=level)

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
        """Records where the *harness* failed to evaluate either side.

        These are genuinely unresolved: they are excluded from the score and
        gate coverage. A failure caused by the view is :attr:`failed` instead.
        """
        return self._count("errored")

    @property
    def failed(self) -> int:
        """Records scored as a failure because the *view* failed to evaluate.

        A subset of :attr:`paired`, not of :meth:`pending` or
        :attr:`errored`: a view that times out or crashes on a question has
        produced a result, and it counts against the level rather than leaving
        the denominator.
        """
        return sum(
            1
            for items in self._records.values()
            for baseline, candidate in items.values()
            if _state(baseline, candidate) == "paired" and (_failed(baseline) or _failed(candidate))
        )

    @property
    def paired(self) -> int:
        return self._count("paired")

    @property
    def retired(self) -> int:
        """Records withdrawn from the bank in the current epoch."""
        return self._count("retired")

    def held_counts(self) -> dict[CellKey, tuple[int, int]]:
        """Retired items per cell with a known outcome, as ``(items, passes)``.

        This is the input to the report's composition split: retired items are
        held at the result they last showed, so the level can be read with and
        without the bank change. An item retired before it was ever measured has
        no outcome to hold and is omitted — it still counts in
        :attr:`retired` and ``coverage().retired``.
        """
        out: dict[CellKey, tuple[int, int]] = {}
        for key, items in self.retired_ledger.items():
            known = [last for last in items.values() if last is not None]
            if known:
                out[key] = (len(known), sum(known))
        return out

    def unpaired_cells(self) -> list[UnpairedCell]:
        """Candidate results with no baseline yet, one entry per cell.

        These are the rows a cold start produces — the first run, or a view
        introduced mid-stream — where the candidate side has a real result but
        the baseline side has not run (``pending``, baseline ``None``). They
        are the evidence for the report's absolute ``score_unpaired`` level,
        which is *not* the paired score. A view-caused :data:`FAILED` candidate
        is a resolved result and counts as a failure.

        Rows that are paired, retired, or harness-errored are excluded: the
        first has a baseline, and the others are not a clean candidate-only
        reading. Sorted for determinism.
        """
        out: list[UnpairedCell] = []
        for (stratum, outcome), items in sorted(self._records.items()):
            resolved = passed = 0
            for baseline, candidate in items.values():
                if _state(baseline, candidate) != "pending":
                    continue
                result = _outcome(candidate)
                if result is None:
                    continue
                resolved += 1
                passed += int(result is True)
            if resolved:
                out.append(
                    UnpairedCell(
                        stratum=stratum,
                        outcome=outcome,
                        resolved=resolved,
                        passed=passed,
                        category=self.categories.get(stratum),
                        group=self.groups.get(stratum),
                    )
                )
        return out

    def _unresolved_share(self) -> float | None:
        """Share of tracked rows still unresolved, or ``None`` when empty.

        ``(pending + errored) / all rows``. Retirement is a deliberate bank
        change rather than missing data, so it is excluded on both sides, and
        a view-caused failure is resolved. This is the value recorded on the
        timeline for the coverage threshold to be calibrated from later.
        """
        total = len(self)
        if total == 0:
            return None
        return (self.pending + self.errored) / total

    def coverage(self) -> list[CellCoverage]:
        """Per-cell resolved/pending/errored counts, sorted for determinism."""
        out: list[CellCoverage] = []
        for (stratum, outcome), items in sorted(self._records.items()):
            counts: dict[State, int] = {
                "paired": 0,
                "pending": 0,
                "errored": 0,
                "retired": 0,
            }
            failed = 0
            for baseline, candidate in items.values():
                state = _state(baseline, candidate)
                counts[state] += 1
                # Only a paired record is scored, so only a paired record can
                # be a view-caused failure. Retirement outranks a failed side,
                # and such a record is not scored at all.
                if state == "paired" and (_failed(baseline) or _failed(candidate)):
                    failed += 1
            out.append(
                CellCoverage(
                    stratum=stratum,
                    outcome=outcome,
                    category=self.categories.get(stratum),
                    group=self.groups.get(stratum),
                    paired=counts["paired"],
                    pending=counts["pending"],
                    errored=counts["errored"],
                    retired=counts["retired"],
                    failed=failed,
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
                # ``_outcome`` maps FAILED to False, so a view-caused failure
                # counts as a paired FAIL rather than leaving the cell.
                before = _outcome(baseline)
                after = _outcome(candidate)
                pairs += 1
                passed += int(after is True)
                improved += int(before is False and after is True)
                regressed += int(before is True and after is False)
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
                        group=self.groups.get(stratum),
                    )
                )
        return cells

    def item_responses(self) -> list[ItemResponse]:
        """Resolved item responses, globally keyed for the difficulty model.

        One :class:`~betapair.irt.ItemResponse` per ``paired`` row — the
        candidate-epoch result of one stratum on one item. ``pending``,
        :data:`ERROR` and :data:`RETIRED` rows carry no response and are
        omitted; a view-caused :data:`FAILED` arrives as ``passed=False``,
        because it is scored as a failure rather than treated as a hole.

        Identity is ``(outcome, item)`` (:mod:`betapair.irt`): the same
        ``item`` string under two strata is one item seen twice, and a
        different string is a different item. Sorted for determinism.

        @return: Responses sorted by ``(stratum, outcome, item)``.
        """
        out: list[ItemResponse] = []
        for (stratum, outcome), items in sorted(self._records.items()):
            for item, (baseline, candidate) in sorted(items.items()):
                if _state(baseline, candidate) != "paired":
                    continue
                out.append(
                    ItemResponse(
                        stratum=stratum,
                        outcome=outcome,
                        item=item,
                        passed=_outcome(candidate) is True,
                    )
                )
        return out

    def report(
        self,
        *,
        draws: int | None = None,
        ppc_draws: int | None = None,
        seed: int | None = None,
        alpha: float | None = None,
        rope_pp: float | None = None,
        max_unresolved_share: float | None = None,
        calibrate: bool = False,
        adjust_difficulty: bool | None = None,
        reference_items: Sequence[ItemKey] | None = None,
        reference_bank: ReferenceBank | None = None,
    ) -> HealthReport:
        """Compute the metric from whatever has arrived so far.

        @param calibrate: Size the decision thresholds from this tracker's own
            :attr:`timeline` instead of the hand-set defaults (see
            :class:`~betapair.report.Calibration`). Off by default, so a
            report is unchanged until a caller opts in.
        @param adjust_difficulty: Fit the difficulty model and report an
            equated level (see :func:`betapair.report.build_report`). ``None``
            uses this tracker's own :attr:`adjust_difficulty` config; the
            responses come from :meth:`item_responses`.
        @param reference_items: Pinned reference bank for this call; ``None``
            uses :attr:`reference_items`, which itself may be ``None`` (each
            outcome's observed union).
        @param reference_bank: Versioned pinned bank for this call; ``None``
            uses :attr:`reference_bank`. Mutually exclusive with
            ``reference_items`` (see :func:`betapair.report.build_report`).
        @note: The report's ``pi_mean`` is remembered for the next recorded
            point, so the *following* epoch can calibrate its regression prior
            from this one.
        """
        adjust = self.adjust_difficulty if adjust_difficulty is None else adjust_difficulty
        report = build_report(
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
            rope_pp=self.rope_pp if rope_pp is None else rope_pp,
            pi_prior=self.pi_prior,
            pi_points=self.pi_points,
            pending=self.pending,
            failed=self.failed,
            retired=self.retired,
            coverage=self.coverage(),
            held_counts=self.held_counts(),
            max_unresolved_share=(
                self.max_unresolved_share if max_unresolved_share is None else max_unresolved_share
            ),
            baseline_epoch=self.baseline_epoch,
            candidate_epoch=self.candidate_epoch,
            history=self.timeline,
            calibrate=calibrate,
            unpaired_cells=self.unpaired_cells(),
            responses=self.item_responses() if adjust else None,
            adjust_difficulty=adjust,
            reference_items=(self.reference_items if reference_items is None else reference_items),
            reference_bank=(self.reference_bank if reference_bank is None else reference_bank),
            sigma_theta=self.sigma_theta,
            sigma_d=self.sigma_d,
        )
        self._last_pi_mean = report.pi_mean
        return report

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
            "groups": dict(sorted(self.groups.items())),
            "default_category": self.default_category,
            "baseline_epoch": self.baseline_epoch,
            "candidate_epoch": self.candidate_epoch,
            "records": records,
            "adjust_difficulty": self.adjust_difficulty,
            "sigma_theta": self.sigma_theta,
            "sigma_d": self.sigma_d,
            "reference_items": (
                None
                if self.reference_items is None
                else [[outcome, item] for outcome, item in self.reference_items]
            ),
            "reference_bank": (
                None
                if self.reference_bank is None
                else {
                    "name": self.reference_bank.name,
                    "version": self.reference_bank.version,
                    "items": [[outcome, item] for outcome, item in self.reference_bank.items],
                }
            ),
            "timeline": [point.snapshot() for point in self.timeline],
            "retired_ledger": [
                [stratum, outcome, item, last]
                for (stratum, outcome), items in sorted(self.retired_ledger.items())
                for item, last in sorted(items.items())
            ],
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
        adjust_difficulty = _boolean(
            config.pop("adjust_difficulty", payload.get("adjust_difficulty", False)),
            "adjust_difficulty",
        )
        sigma_theta = _positive_number(
            config.pop("sigma_theta", payload.get("sigma_theta", DEFAULT_SIGMA)), "sigma_theta"
        )
        sigma_d = _positive_number(
            config.pop("sigma_d", payload.get("sigma_d", DEFAULT_SIGMA)), "sigma_d"
        )
        reference_items = _parse_reference_items(
            config.pop("reference_items", payload.get("reference_items"))
        )
        reference_bank = _parse_reference_bank(
            config.pop("reference_bank", payload.get("reference_bank"))
        )
        tracker = cls(
            weights=_number_mapping(payload.get("weights"), "weights"),
            category_weights=_number_mapping(payload.get("category_weights"), "category_weights"),
            categories=_string_mapping(payload.get("categories"), "categories"),
            groups=_string_mapping(payload.get("groups"), "groups"),
            default_category=_text_or_none(
                payload.get("default_category", DEFAULT_CATEGORY), "default_category"
            ),
            seed=seed,
            baseline_epoch=_text_or_none(payload.get("baseline_epoch"), "baseline_epoch"),
            candidate_epoch=_text_or_none(payload.get("candidate_epoch"), "candidate_epoch"),
            adjust_difficulty=adjust_difficulty,
            sigma_theta=sigma_theta,
            sigma_d=sigma_d,
            reference_items=reference_items,
            reference_bank=reference_bank,
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
        for raw in _ledger_rows(payload.get("retired_ledger")):
            stratum, outcome, item, last = raw
            if last is not None and not isinstance(last, bool):
                raise ValueError(
                    f"snapshot retired_ledger outcome must be true, false or null, got {last!r}"
                )
            key = (str(stratum), str(outcome))
            tracker.retired_ledger.setdefault(key, {})[str(item)] = last
        raw_timeline = payload.get("timeline", [])
        if not isinstance(raw_timeline, list):
            raise ValueError(f"snapshot timeline must be a list, got {raw_timeline!r}")
        tracker.timeline.extend(ScorePoint.from_snapshot(point) for point in raw_timeline)
        return tracker

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.snapshot(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str, **config: object) -> HealthTracker:
        return cls.from_snapshot(json.loads(text), **config)

    def __len__(self) -> int:
        """Number of tracked rows in the current epoch, whatever their state."""
        return sum(len(items) for items in self._records.values())
