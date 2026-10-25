"""Parity check of the Rasch MAP against ``girth``'s JML fit (dev-only oracle).

``girth`` is the established MIT-licensed reference for Rasch/1PL estimation,
but it imports numpy and scipy, which the package's numeric core deliberately
does not. It is therefore an **optional oracle**, not a dependency: install it
with the extra and run::

    uv sync --extra oracle
    uv run python tools/irt_oracle.py

The script simulates one crossed view x item bank from a known seed, fits it
with both :func:`betapair.irt.fit_rasch` (penalised JML) and ``girth``'s
``rasch_jml`` (plain JML), and compares the centred item difficulties and view
abilities. The two estimators differ by the Gaussian priors, so exact equality
is not expected; agreement of the rank order is the claim being checked.

When ``girth``/``numpy`` are not installed the script prints a skip notice and
exits 0, so it is safe to run anywhere and is never part of CI's required
dependency set.
"""

from __future__ import annotations

import importlib
import random
import statistics
import warnings
from typing import Any

from betapair.irt import ItemResponse, fit_rasch, sigmoid

#: The simulated bank. Large enough that the priors are weak relative to the
#: likelihood, so the two estimators are comparable.
VIEWS = 20
ITEMS = 100
SIGMA = 1.5
SEED = 7

#: Agreement floors. Recovery is noisy, so these are deliberately below the
#: ~0.98/~0.9999 observed; they fail only on a real regression in the fit.
MIN_DIFFICULTY_CORRELATION = 0.95
MIN_ABILITY_CORRELATION = 0.98


def _load() -> tuple[Any, Any] | None:
    """Return ``(girth, numpy)``, or ``None`` when the oracle extra is absent."""
    try:
        return importlib.import_module("girth"), importlib.import_module("numpy")
    except ImportError:
        return None


def _simulate() -> tuple[list[ItemResponse], list[list[int]]]:
    """A crossed bank, plus the same responses as an ``items x views`` matrix."""
    rng = random.Random(SEED)
    abilities = [rng.gauss(0.0, SIGMA) for _ in range(VIEWS)]
    easiness = [rng.gauss(0.0, SIGMA) for _ in range(ITEMS)]
    responses: list[ItemResponse] = []
    matrix = [[0] * VIEWS for _ in range(ITEMS)]
    for view in range(VIEWS):
        for item in range(ITEMS):
            passed = rng.random() < sigmoid(abilities[view] + easiness[item])
            responses.append(ItemResponse(f"v{view}", "answer", f"q{item}", passed))
            matrix[item][view] = 1 if passed else 0
    return responses, matrix


def _centred(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    return [value - mean for value in values]


def main() -> int:
    loaded = _load()
    if loaded is None:
        print("girth/numpy not installed; skipping (run `uv sync --extra oracle` to enable)")
        return 0
    girth, numpy = loaded

    responses, matrix = _simulate()
    fit = fit_rasch(responses)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = girth.rasch_jml(numpy.asarray(matrix, dtype=bool))
        reference_difficulty = [float(value) for value in reference["Difficulty"]]
        reference_ability = [
            float(value)
            for value in girth.ability_eap(
                numpy.asarray(matrix, dtype=int), reference["Difficulty"], 1
            )
        ]

    our_difficulty = [-fit.difficulties[("answer", f"q{item}")] for item in range(ITEMS)]
    our_ability = [fit.abilities[f"v{view}"] for view in range(VIEWS)]

    difficulty = statistics.correlation(_centred(reference_difficulty), _centred(our_difficulty))
    ability = statistics.correlation(_centred(reference_ability), _centred(our_ability))
    print(f"difficulty correlation: {difficulty:.4f} (floor {MIN_DIFFICULTY_CORRELATION})")
    print(f"ability correlation:    {ability:.4f} (floor {MIN_ABILITY_CORRELATION})")
    if difficulty < MIN_DIFFICULTY_CORRELATION or ability < MIN_ABILITY_CORRELATION:
        print("FAIL: the penalised-JML fit disagrees with the girth reference")
        return 1
    print("OK: the penalised-JML fit agrees with girth's JML fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
