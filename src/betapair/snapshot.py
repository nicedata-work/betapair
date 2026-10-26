"""The tracker snapshot wire format: sides, states and their JSON encoding.

Extracted from :mod:`betapair.tracker` so serialisation is not tangled with
the streaming accumulator. This module owns the sentinels (:data:`ERROR`,
:data:`RETIRED`, :data:`FAILED`), the record :data:`State` predicate, and every
``_parse_*`` helper that turns a persisted payload back into domain values;
:class:`~betapair.tracker.HealthTracker` imports them and re-exports the
public names.

Nothing here knows about cells, scores or reports. The only betapair import is
:mod:`betapair.irt` for the reference-bank wire types, so the module can be
imported (and tested) on its own.
"""

from __future__ import annotations

import enum
from typing import Literal

from betapair.irt import ItemKey, ReferenceBank

__all__ = [
    "ERROR",
    "FAILED",
    "RETIRED",
    "SNAPSHOT_VERSION",
    "Marker",
    "Side",
    "State",
]

#: Snapshot *format* version — internal to the JSON payload and independent
#: of the package version in ``pyproject.toml``. It moves when the payload
#: shape changes (v3 added the score ``timeline``); :meth:`from_snapshot`
#: rejects any other version. Within a version, additive keys are fine:
#: a v3 payload with no ``timeline`` still loads, defaulted to empty, and one
#: with no ``retired_ledger`` loads with an empty ledger. ``groups`` joined the
#: same way: absent means no view has a shared-dependency label, and
#: ``fingerprints``/``drift`` are additive too — absent means none were
#: supplied or observed.
SNAPSHOT_VERSION = 3


class Marker(enum.Enum):
    """The three non-boolean states a side can carry, as one vocabulary.

    A side is normally ``True``/``False`` (a real result) or ``None`` (not
    evaluated yet). The remaining three states say *why* there is no boolean:

    * :attr:`ERROR` — evaluated, but the evaluation (the harness) failed;
    * :attr:`RETIRED` — the item has been withdrawn from the bank;
    * :attr:`FAILED` — evaluated, and the failure was the view's own fault,
      so it scores as ``False`` rather than shrinking the denominator.

    They were three look-alike singleton classes; one enum keeps them in a
    single namespace, makes membership and identity checks direct, and lets
    the wire form (:attr:`value`) be the serialised string itself.
    """

    ERROR = "error"
    RETIRED = "retired"
    FAILED = "failed"

    def __repr__(self) -> str:
        return self.name


#: Marker for a side whose evaluation errored (as opposed to ``None`` = pending).
ERROR = Marker.ERROR

#: Marker for a side whose item has been withdrawn from the bank.
RETIRED = Marker.RETIRED

#: Marker for a side whose evaluation failed and the failure was the view's fault.
FAILED = Marker.FAILED

Side = bool | None | Marker
CellKey = tuple[str, str]
Pair = tuple[Side, Side]


#: One record's state. The single predicate every counter uses, so
#: ``coverage()``, ``cells()`` and the ``paired``/``pending``/``errored``/
#: ``retired`` properties can never disagree about a record.
State = Literal["paired", "pending", "errored", "retired"]


def _errored(value: object) -> bool:
    return value is Marker.ERROR


def _retired(value: object) -> bool:
    return value is Marker.RETIRED


def _failed(value: object) -> bool:
    return value is Marker.FAILED


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

    Notes:
        Retirement outranks the other states. It is a statement about the
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

    Raises:
        ValueError: If ``value`` is not ``True``, ``False``, ``None``,
            :data:`ERROR`, :data:`FAILED` or :data:`RETIRED`.
    """
    if value is None or isinstance(value, (bool, Marker)):
        return value
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


def _fingerprint_rows(value: object) -> list[tuple[str, str, str]]:
    """Validate the snapshot's ``[outcome, item, fingerprint]`` rows."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"snapshot fingerprints must be a list, got {value!r}")
    rows: list[tuple[str, str, str]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError(
                f"snapshot fingerprints must be [outcome, item, fingerprint], got {raw!r}"
            )
        outcome, item, fingerprint = raw
        if not all(isinstance(part, str) and part for part in (outcome, item, fingerprint)):
            raise ValueError(f"snapshot fingerprint fields must be non-empty strings, got {raw!r}")
        rows.append((outcome, item, fingerprint))
    return rows


def _drift_rows(value: object) -> list[tuple[str, str, str, str]]:
    """Validate the snapshot's ``[outcome, item, expected, observed]`` rows."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"snapshot drift must be a list, got {value!r}")
    rows: list[tuple[str, str, str, str]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError(
                f"snapshot drift rows must be [outcome, item, expected, observed], got {raw!r}"
            )
        parts = raw
        if not all(isinstance(part, str) and part for part in parts):
            raise ValueError(f"snapshot drift fields must be non-empty strings, got {raw!r}")
        outcome, item, expected, observed = parts
        if expected == observed:
            raise ValueError(f"snapshot drift row needs two different fingerprints, got {raw!r}")
        rows.append((outcome, item, expected, observed))
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
