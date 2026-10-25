"""Item identity and the latent-difficulty (Rasch/IRT) model.

This module holds the **item-identity contract** that the difficulty model is
built on, and the model itself: a one-parameter logistic (Rasch) fit by
penalised joint maximum likelihood. Keeping it here — and out of
:mod:`betapair.tracker` and :mod:`betapair.report` — means the numeric core
never has to import the report layer, and the report can import this later
without a cycle.

The contract
------------

An IRT item is not a ``(stratum, item)`` row. It is a **question** identified
globally as:

    ``(outcome, item)``

so "the same question" asked for the ``answer`` metric and for ``sql_valid`` is
*two* items, while the same question asked of two views is *one* item shared by
both. That sharing is the identifiability backbone of the difficulty model:
without overlapping banks, ``theta_v + d_i`` is only pinned by the priors and
cross-view comparison rests on an assumption rather than on data.

Requirements on the caller
--------------------------

``item`` must be:

* **stable** — the same question keeps the same string across epochs, views and
  runs; and
* **shared** — every view that can evaluate the question must use that same
  string. A view that renames ``q5`` to ``q5_v2`` makes the banks look disjoint
  and the fit falls back to prior bridging instead of real linking.

Re-observing the same ``(stratum, outcome, item)`` row overwrites it, which is
what makes streaming idempotent; re-observing the same ``(outcome, item)`` under
a *different* stratum adds a second response to the same item, which is exactly
what links two views.

@note: Editing the *text* of a question while keeping its id pools two
    different questions under one difficulty — the classic item-drift bug. The
    content fingerprint that detects it is a separate, later task; until then
    the caller owns identity.

The model
---------

For one outcome at a time, each response ``y_{v,i} in {0, 1}`` is the
candidate-epoch result of view ``v`` on item ``i``:

    ``P(y_{v,i} = 1) = sigmoid(theta_v + d_i)``

with ``theta_v ~ N(0, sigma_theta^2)`` the view's ability (higher = healthier)
and ``d_i ~ N(0, sigma_d^2)`` the item's easiness (higher = easier). Reporting
uses difficulty ``b_i = -d_i``; the code carries ``d`` because the gradient is
cleanest that way.

The two Gaussian priors fix the additive non-identifiability (``theta + d`` is
invariant to ``theta + c, d - c``), keep perfectly separated views and items
finite, and — importantly — let two disconnected banks borrow strength from a
common distribution instead of diverging.

@note: The fit is **penalised JML** (joint maximum likelihood). The two
    parameter blocks are conditionally independent, so MAP is alternating 1-D
    Newton with no optimiser and no matrix solve. Its incidental-parameter
    bias is a small-sample effect that the ridge priors are the standard
    mitigation for. Uncertainty on ``theta`` comes from the Schur complement of
    the joint Hessian (items integrated out approximately); ``difficulty_sd``
    is the conditional item sd given ``theta`` and therefore understates the
    marginal, which is acceptable because items are nuisance here.

Equating and diagnostics
------------------------

A fitted model turns a view's pass rate on *its own* bank into
:func:`equated_rates` — the rate it would score on a common reference bank,
``mean_i sigmoid(theta_v + d_i)``. That is what makes two views on a hard and
an easy bank comparable.

Comparability is only as good as the **link**: views are connected only
through items they both ran. :class:`RaschDiagnostics` (carried on every
:class:`RaschFit`) reports the view-item graph's ``components``/``connected``
flags, the ``weakest_link`` shared-item count, and the views/items that are
perfectly separated (``prior_dominated``, ``separated_items``). Disjoint banks
are still fitted — the priors bridge them — but that is an assumption the
diagnostics surface rather than a measurement.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

#: Views above which the ``V x V`` Schur complement is not inverted; its
#: diagonal is used instead, which understates the ability uncertainty. The
#: pure-Python Cholesky is ``O(V^3)`` and fine to a few hundred views.
SCHUR_MAX_VIEWS = 300

#: Default Gaussian prior scale for both abilities and item easinesses.
DEFAULT_SIGMA = 1.5

#: Default Newton sweeps and convergence tolerance for the MAP.
DEFAULT_MAX_ITER = 200
DEFAULT_TOL = 1e-8

#: Halvings allowed before a Newton step is taken as-is (a safety net; the
#: penalised objective is concave, so it almost never triggers).
_MAX_HALVINGS = 20

#: Global identity of an IRT item: the outcome it targets and the caller's
#: stable item string. ``(outcome, item)`` — not the per-cell
#: ``(stratum, outcome, item)`` row.
ItemKey = tuple[str, str]


def item_key(outcome: str, item: str) -> ItemKey:
    """The global IRT key of an item, ``(outcome, item)``.

    @raise ValueError: If ``outcome`` or ``item`` is empty.
    """
    if not outcome or not item:
        raise ValueError("outcome and item must be non-empty")
    return (outcome, item)


@dataclass(frozen=True)
class ItemResponse:
    """One view's response to one globally identified item.

    ``passed`` is the candidate-epoch result of ``stratum`` on ``item``. The
    same :attr:`key` from several strata is the same item seen by several
    views, which is what links their ability estimates; a different ``item``
    string is a different item even when the underlying question is the same.

    @note: A view-caused :data:`betapair.tracker.FAILED` is already a failure
        by the time it reaches here, so it arrives as ``passed=False`` — a
        response, not a hole.
    """

    stratum: str
    outcome: str
    item: str
    passed: bool

    def __post_init__(self) -> None:
        if not self.stratum or not self.outcome or not self.item:
            raise ValueError("stratum, outcome and item must be non-empty")
        if not isinstance(self.passed, bool):
            raise ValueError(f"passed must be a bool, got {self.passed!r}")

    @property
    def key(self) -> ItemKey:
        """This response's global item key, ``(outcome, item)``."""
        return (self.outcome, self.item)


def bank_digest(items: Sequence[ItemKey]) -> str:
    """Short, content-based hash of a reference item set.

    Order- and duplicate-independent, so two banks that name the same
    questions hash the same. It is what makes an adjusted score's meaning
    auditable: two epochs whose digests differ were equated on a different
    item set, whatever their names or versions claim.
    """
    canonical = "\n".join(f"{outcome}\x1f{item}" for outcome, item in sorted(set(items)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ReferenceBank:
    """A named, versioned set of global items defining the adjusted scale.

    ``score_adjusted`` is the pass rate a view would score on *some* common
    bank. The observed union moves as the catalogue moves, so that number's
    meaning drifts with every edit; pinning the bank — ``name``, ``version``
    and the exact ``items`` — fixes the scale, and :attr:`digest` lets a
    consumer verify that two epochs were scored on the same set.

    ``items`` is canonicalised (sorted, de-duplicated, validated) in
    ``__post_init__``. ``version`` is the operator's label and is never
    parsed, so a bank can be re-versioned without its items changing; the
    content :attr:`digest` is what actually distinguishes two banks.
    """

    name: str
    version: str
    items: tuple[ItemKey, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("a reference bank needs a non-empty name and version")
        if not self.items:
            raise ValueError("a reference bank must not be empty")
        seen: set[ItemKey] = set()
        for key in self.items:
            outcome, item = key
            if not outcome or not item:
                raise ValueError(f"reference item parts must be non-empty, got {key!r}")
            if key in seen:
                raise ValueError(f"reference bank repeats item {key!r}")
            seen.add(key)
        object.__setattr__(self, "items", tuple(sorted(self.items)))

    @classmethod
    def from_responses(
        cls, responses: Sequence[ItemResponse], *, name: str, version: str
    ) -> ReferenceBank:
        """The distinct items those responses cover, as a pinned bank.

        The usual way to freeze "what we run today" into a standard bank: fit
        an epoch, capture its items, and pin the bank for later epochs.
        """
        return cls(name=name, version=version, items=tuple({r.key for r in responses}))

    @property
    def label(self) -> str:
        """``name@version``, for messages and logs."""
        return f"{self.name}@{self.version}"

    @property
    def digest(self) -> str:
        """Content hash of :attr:`items` (see :func:`bank_digest`)."""
        return bank_digest(self.items)


@dataclass(frozen=True)
class RaschDiagnostics:
    """What the fit can and cannot say about linking the views.

    The difficulty model compares views only through **shared items**. Two
    banks that share nothing are still fitted — the Gaussian priors make them
    exchangeable draws from a common distribution — but the comparison then
    rests on that assumption rather than on data. These fields are the honest
    caveat, in the spirit of the existing ``Guardrail``/``Coverage`` machinery:

    * ``components`` — connected components of the view-item graph;
    * ``connected`` — ``components <= 1``, i.e. every view is linked to every
      other by at least one shared item;
    * ``weakest_link`` — the fewest shared items over all view pairs (0 when
      there are fewer than two views, or when a pair shares nothing);
    * ``prior_dominated`` — views whose responses are all-pass or all-fail, so
      their ability estimate is the prior's, not the data's;
    * ``separated_items`` — how many items are in the same perfectly separated
      set (their difficulties are prior-driven too, but there are usually many
      of them, so only the count is reported).

    @note: A separated view or item still gets a finite estimate (the prior is
        part of the model), but it should not be read as measured. A
        ``connected=False`` fit with ``weakest_link == 0`` means the adjusted
        numbers are prior-bridged; the fix is to add shared anchor items.
    """

    components: int = 0
    connected: bool = True
    weakest_link: int = 0
    prior_dominated: tuple[str, ...] = ()
    separated_items: int = 0

    def __post_init__(self) -> None:
        if self.components < 0:
            raise ValueError(f"components must be non-negative, got {self.components}")
        if self.weakest_link < 0:
            raise ValueError(f"weakest_link must be non-negative, got {self.weakest_link}")
        if self.separated_items < 0:
            raise ValueError(f"separated_items must be non-negative, got {self.separated_items}")
        if self.connected != (self.components <= 1):
            raise ValueError("connected must be components <= 1")


@dataclass(frozen=True)
class RaschFit:
    """A fitted one-parameter logistic model for a single outcome.

    ``abilities`` maps a stratum to ``theta_v`` and ``difficulties`` maps a
    global :data:`ItemKey` to the easiness ``d_i`` (report difficulty as
    ``-d_i``). ``ability_sd`` and ``difficulty_sd`` are the Laplace posterior
    standard deviations; the latter is conditional on ``theta``.

    ``hyper`` is the resolved ``(sigma_theta, sigma_d)`` pair and ``outcomes``
    the outcome the fit covers (a one-tuple, or empty for no responses). The
    model is fitted **one outcome at a time** — items are outcome-specific — so
    a call that spans outcomes is rejected rather than silently pooled.

    ``diagnostics`` records how well the banks are actually linked (see
    :class:`RaschDiagnostics`); it is part of the fit, not a separate call, so
    an equated number can never be read without its caveat.

    @note: The fit is invariant to the additive split of the generating truth
        (``theta + c, d - c``): the likelihood sees only the sum, and the
        priors pin the decomposition, so the fitted probabilities are
        unchanged.
    """

    abilities: dict[str, float]
    ability_sd: dict[str, float]
    difficulties: dict[ItemKey, float]
    difficulty_sd: dict[ItemKey, float]
    hyper: tuple[float, float]
    outcomes: tuple[str, ...]
    diagnostics: RaschDiagnostics = RaschDiagnostics()
    _ability_factor: tuple[tuple[float, ...], ...] = field(default=(), repr=False, compare=False)
    """Lower factor ``G`` of ``Var(theta)``, empty when it was not built."""

    def probability(self, stratum: str, item: str) -> float:
        """Fitted ``P(pass)`` of ``stratum`` on ``item``, ``sigmoid(theta + d)``.

        @raise KeyError: If the stratum or the item is not in the fit, or the
            fit covers no outcome.
        """
        outcome = self.outcomes[0]
        return sigmoid(self.abilities[stratum] + self.difficulties[(outcome, item)])

    def sample_abilities(self, *, draws: int, seed: int) -> dict[str, list[float]]:
        """Seeded posterior draws of ``theta``, one list per stratum.

        Draws come from the Laplace posterior ``N(theta_hat, Var(theta))``.
        When the Schur complement factorised they are correlated through
        ``G`` (``G G^T = Var``), so shared-item dependence between views is
        carried into any aggregate; otherwise they are independent normals at
        the reported ``ability_sd`` (see :func:`_covariance_factor`).

        This is how ability uncertainty reaches a difficulty-adjusted score:
        each draw maps to an equated rate, and the spread of the resulting
        aggregate is its interval.

        @note: Deterministic for a given ``seed``; strata are drawn in sorted
            order.
        @raise ValueError: If ``draws`` is not positive.
        """
        if draws < 1:
            raise ValueError(f"draws must be at least 1, got {draws}")
        strata = sorted(self.abilities)
        if not strata:
            return {}
        rng = random.Random(seed)
        out: dict[str, list[float]] = {stratum: [] for stratum in strata}
        if self._ability_factor:
            factor = self._ability_factor
            size = len(strata)
            for _ in range(draws):
                standard = [rng.gauss(0.0, 1.0) for _ in range(size)]
                for row in range(size):
                    offset = 0.0
                    for column in range(row + 1):
                        offset += factor[row][column] * standard[column]
                    out[strata[row]].append(self.abilities[strata[row]] + offset)
        else:
            for stratum in strata:
                base = self.abilities[stratum]
                sd = self.ability_sd.get(stratum, 0.0)
                out[stratum] = [base + sd * rng.gauss(0.0, 1.0) for _ in range(draws)]
        return out


def log_sigmoid(value: float) -> float:
    """``log(sigmoid(value))``, stable in both tails."""
    return -log1p_exp(-value)


def sigmoid(value: float) -> float:
    """Logistic function, stable in both tails."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def log1p_exp(value: float) -> float:
    """``log(1 + exp(value))`` without overflow."""
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _log_likelihood(value: float, passed: int) -> float:
    """One response's log-probability at ``value = theta_v + d_i``."""
    return log_sigmoid(value) if passed else log_sigmoid(-value)


def _local_view(
    ability: float, observations: list[tuple[int, int]], easiness: list[float], prior: float
) -> float:
    """The objective terms that involve one view, at a trial ``ability``."""
    total = -0.5 * prior * ability * ability
    for item, passed in observations:
        total += _log_likelihood(ability + easiness[item], passed)
    return total


def _local_item(
    easiness: float, observations: list[tuple[int, int]], ability: list[float], prior: float
) -> float:
    """The objective terms that involve one item, at a trial ``easiness``."""
    total = -0.5 * prior * easiness * easiness
    for view, passed in observations:
        total += _log_likelihood(ability[view] + easiness, passed)
    return total


def _backtrack(value: float, step: float, objective: Callable[[float], float]) -> float:
    """Apply ``step``, halving it while the objective would decrease.

    The penalised objective is concave, so a full Newton step is normally
    accepted; this is the safety net for a pathological sweep.
    """
    current = objective(value)
    scale = 1.0
    for _ in range(_MAX_HALVINGS):
        trial = value + step * scale
        if objective(trial) >= current:
            return trial
        scale *= 0.5
    return value + step * scale


def _cholesky(
    matrix: list[list[float]], *, jitter: float = 1e-10, max_tries: int = 6
) -> list[list[float]]:
    """Lower Cholesky factor of a symmetric positive-definite matrix.

    A small diagonal jitter is added and retried if the matrix is only
    numerically positive-definite, which is the usual failure mode of a
    Schur complement built from floating-point weights.

    @raise ValueError: If the matrix stays non-positive-definite after
        ``max_tries`` attempts.
    """
    size = len(matrix)
    boost = 0.0
    for _ in range(max_tries):
        lower = [[0.0] * size for _ in range(size)]
        positive = True
        for row in range(size):
            for column in range(row + 1):
                total = matrix[row][column] + (boost if row == column else 0.0)
                for index in range(column):
                    total -= lower[row][index] * lower[column][index]
                if row == column:
                    if total <= 0.0:
                        positive = False
                        break
                    lower[row][column] = math.sqrt(total)
                else:
                    lower[row][column] = total / lower[column][column]
            if not positive:
                break
        if positive:
            return lower
        boost = jitter if boost == 0.0 else boost * 10.0
    raise ValueError("matrix is not positive definite")


def _inverse_from_cholesky(lower: list[list[float]]) -> list[list[float]]:
    """Full inverse of ``L L^T`` from its lower Cholesky factor ``L``.

    Used to build the covariance factor for ability draws; the diagonal-only
    path in :func:`_diagonal_of_inverse` avoids the extra storage when only
    the variances are needed.
    """
    size = len(lower)
    inverse = [[0.0] * size for _ in range(size)]
    for column in range(size):
        forward = [0.0] * size
        for row in range(column, size):
            total = 1.0 if row == column else 0.0
            for index in range(row):
                total -= lower[row][index] * forward[index]
            forward[row] = total / lower[row][row]
        backward = [0.0] * size
        for row in range(size - 1, -1, -1):
            total = forward[row]
            for index in range(row + 1, size):
                total -= lower[index][row] * backward[index]
            backward[row] = total / lower[row][row]
        for row in range(size):
            inverse[row][column] = backward[row]
    return inverse


def _diagonal_of_inverse(lower: list[list[float]]) -> list[float]:
    """Diagonal of ``(L L^T)^-1`` from its lower Cholesky factor ``L``.

    Only the diagonal is needed (the ability variances), so each column of the
    inverse is solved for and the rest discarded.
    """
    size = len(lower)
    result = [0.0] * size
    for column in range(size):
        forward = [0.0] * size
        for row in range(column, size):
            total = 1.0 if row == column else 0.0
            for index in range(row):
                total -= lower[row][index] * forward[index]
            forward[row] = total / lower[row][row]
        backward = [0.0] * size
        for row in range(size - 1, -1, -1):
            total = forward[row]
            for index in range(row + 1, size):
                total -= lower[index][row] * backward[index]
            backward[row] = total / lower[row][row]
        result[column] = backward[column]
    return result


def _inverse_diagonal(matrix: list[list[float]]) -> list[float]:
    """Diagonal of a symmetric matrix's inverse, with two safe fallbacks.

    Above :data:`SCHUR_MAX_VIEWS`, or when the Cholesky fails, the reciprocal
    of the diagonal is returned — documented as understating the uncertainty
    because ``(S^-1)_vv >= 1 / S_vv``.
    """
    size = len(matrix)
    if size == 0:
        return []
    if size <= SCHUR_MAX_VIEWS:
        try:
            return _diagonal_of_inverse(_cholesky(matrix))
        except ValueError:
            pass
    return [
        1.0 / matrix[index][index] if matrix[index][index] > 0.0 else math.inf
        for index in range(size)
    ]


def _covariance_factor(schur: list[list[float]]) -> list[list[float]]:
    """Lower factor ``G`` of ``Var(theta) = S^-1``, or ``[]`` when unavailable.

    Above :data:`SCHUR_MAX_VIEWS`, or if the Schur complement is not
    numerically positive definite, no factor is built and
    :meth:`RaschFit.sample_abilities` falls back to independent draws from the
    reported ``ability_sd`` — documented as understating the uncertainty of a
    *sum* of abilities, because it drops the cross-view correlation.
    """
    size = len(schur)
    if size == 0 or size > SCHUR_MAX_VIEWS:
        return []
    try:
        return _cholesky(_inverse_from_cholesky(_cholesky(schur)))
    except ValueError:
        return []


def _schur_complement(
    view_observations: list[list[tuple[int, int]]],
    item_observations: list[list[tuple[int, int]]],
    abilities: list[float],
    easiness: list[float],
    prior_theta: float,
    prior_d: float,
) -> tuple[list[list[float]], list[float]]:
    """Schur complement ``S = A - B C^-1 B^T`` and conditional item variances.

    The joint Hessian blocks are ``A`` (views), ``C`` (items) and ``B``, all
    built from the same ``w_vi = p_vi (1 - p_vi)``. The ``theta`` marginal is
    the Schur complement; the item variances are taken conditional on
    ``theta`` (``1 / C_ii``), the documented approximation for a nuisance
    parameter. Returning the matrix itself lets the caller build the ability
    covariance factor from the same numbers.
    """
    view_count = len(view_observations)
    item_count = len(item_observations)
    a_diagonal = [prior_theta] * view_count
    c_diagonal = [prior_d] * item_count
    weights: list[list[tuple[int, float]]] = [[] for _ in range(item_count)]
    for view in range(view_count):
        for item, _passed in view_observations[view]:
            probability = sigmoid(abilities[view] + easiness[item])
            weight = probability * (1.0 - probability)
            a_diagonal[view] += weight
            c_diagonal[item] += weight
            weights[item].append((view, weight))

    schur = [[0.0] * view_count for _ in range(view_count)]
    for view in range(view_count):
        schur[view][view] = a_diagonal[view]
    for item in range(item_count):
        inverse_c = 1.0 / c_diagonal[item]
        observed = weights[item]
        for view, weight in observed:
            schur[view][view] -= weight * weight * inverse_c
        for first in range(len(observed)):
            left_view, left_weight = observed[first]
            for second in range(first + 1, len(observed)):
                right_view, right_weight = observed[second]
                value = left_weight * right_weight * inverse_c
                schur[left_view][right_view] -= value
                schur[right_view][left_view] -= value

    item_variances = [1.0 / c_diagonal[item] for item in range(item_count)]
    return schur, item_variances


def _connected_components(view_observations: list[list[tuple[int, int]]], item_count: int) -> int:
    """Number of connected components of the bipartite view-item graph.

    Union-find over ``view_count + item_count`` nodes; every node has at least
    one edge by construction, so the number of distinct roots is the number of
    components.
    """
    view_count = len(view_observations)
    parent = list(range(view_count + item_count))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for view in range(view_count):
        for item, _passed in view_observations[view]:
            left, right = find(view), find(view_count + item)
            if left != right:
                parent[left] = right
    return len({find(node) for node in range(view_count + item_count)})


def _weakest_link(item_observations: list[list[tuple[int, int]]], view_count: int) -> int:
    """The fewest shared items over all view pairs (0 with fewer than two views)."""
    if view_count < 2:
        return 0
    shared = {
        (left, right): 0 for left in range(view_count) for right in range(left + 1, view_count)
    }
    for observations in item_observations:
        views = sorted(view for view, _passed in observations)
        for first in range(len(views)):
            for second in range(first + 1, len(views)):
                shared[(views[first], views[second])] += 1
    return min(shared.values())


def _diagnose(
    strata: list[str],
    item_count: int,
    view_observations: list[list[tuple[int, int]]],
    item_observations: list[list[tuple[int, int]]],
) -> RaschDiagnostics:
    """Summarise how the views are linked and what the priors are carrying."""
    components = _connected_components(view_observations, item_count)
    dominated = tuple(
        stratum
        for view, stratum in enumerate(strata)
        if len({passed for _item, passed in view_observations[view]}) == 1
    )
    separated_items = sum(
        1
        for item in range(item_count)
        if len({passed for _view, passed in item_observations[item]}) == 1
    )
    return RaschDiagnostics(
        components=components,
        connected=components <= 1,
        weakest_link=_weakest_link(item_observations, len(strata)),
        prior_dominated=dominated,
        separated_items=separated_items,
    )


def fit_rasch(
    responses: Sequence[ItemResponse],
    *,
    sigma_theta: float = DEFAULT_SIGMA,
    sigma_d: float = DEFAULT_SIGMA,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> RaschFit:
    """Fit the penalised-JML Rasch model to one outcome's responses.

    Alternating 1-D Newton over the two conditionally independent blocks
    (views, then items), with step-halving if a sweep would decrease the
    objective, until ``max|step| < tol`` or ``max_iter`` sweeps. ``sigma_theta``
    and ``sigma_d`` are the Gaussian prior scales that fix the additive
    non-identifiability and keep separated parameters finite.

    @note: One outcome per call — items are outcome-specific, so a fit that
        spans outcomes would silently pool unrelated items. The caller groups
        by outcome (the report's cells already are).
    @note: Deterministic: strata and items are processed in sorted order and no
        randomness enters the MAP, so the same responses give the same fit.
    @return: The :class:`RaschFit`; empty for no responses.
    @raise ValueError: If a prior scale is not positive, ``max_iter < 1``,
        ``tol`` is not positive, responses span outcomes, or a
        ``(stratum, item)`` row is repeated.
    """
    if sigma_theta <= 0.0:
        raise ValueError(f"sigma_theta must be positive, got {sigma_theta}")
    if sigma_d <= 0.0:
        raise ValueError(f"sigma_d must be positive, got {sigma_d}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")

    outcomes = sorted({response.outcome for response in responses})
    if len(outcomes) > 1:
        raise ValueError(f"fit_rasch fits one outcome at a time, got {outcomes}")
    if not outcomes:
        return RaschFit(
            abilities={},
            ability_sd={},
            difficulties={},
            difficulty_sd={},
            hyper=(sigma_theta, sigma_d),
            outcomes=(),
            diagnostics=RaschDiagnostics(),
        )
    outcome = outcomes[0]

    strata = sorted({response.stratum for response in responses})
    items = sorted({response.item for response in responses})
    view_index = {stratum: index for index, stratum in enumerate(strata)}
    item_index = {item: index for index, item in enumerate(items)}
    view_observations: list[list[tuple[int, int]]] = [[] for _ in strata]
    item_observations: list[list[tuple[int, int]]] = [[] for _ in items]
    seen: set[tuple[str, str]] = set()
    for response in responses:
        row = (response.stratum, response.item)
        if row in seen:
            raise ValueError(f"duplicate response for stratum={row[0]!r}, item={row[1]!r}")
        seen.add(row)
        view = view_index[response.stratum]
        item = item_index[response.item]
        passed = 1 if response.passed else 0
        view_observations[view].append((item, passed))
        item_observations[item].append((view, passed))
    # Canonical summation order: response order must not move the last bit.
    for observations in view_observations:
        observations.sort()
    for observations in item_observations:
        observations.sort()

    prior_theta = 1.0 / (sigma_theta * sigma_theta)
    prior_d = 1.0 / (sigma_d * sigma_d)
    abilities = [0.0] * len(strata)
    easiness = [0.0] * len(items)

    for _ in range(max_iter):
        largest = 0.0
        for view in range(len(strata)):
            observations = view_observations[view]
            gradient = 0.0
            curvature = prior_theta
            for item, passed in observations:
                probability = sigmoid(abilities[view] + easiness[item])
                gradient += passed - probability
                curvature += probability * (1.0 - probability)

            def view_objective(
                value: float, observations: list[tuple[int, int]] = observations
            ) -> float:
                return _local_view(value, observations, easiness, prior_theta)

            updated = _backtrack(abilities[view], gradient / curvature, view_objective)
            largest = max(largest, abs(updated - abilities[view]))
            abilities[view] = updated

        for item in range(len(items)):
            observations = item_observations[item]
            gradient = 0.0
            curvature = prior_d
            for view, passed in observations:
                probability = sigmoid(abilities[view] + easiness[item])
                gradient += passed - probability
                curvature += probability * (1.0 - probability)

            def item_objective(
                value: float, observations: list[tuple[int, int]] = observations
            ) -> float:
                return _local_item(value, observations, abilities, prior_d)

            updated = _backtrack(easiness[item], gradient / curvature, item_objective)
            largest = max(largest, abs(updated - easiness[item]))
            easiness[item] = updated

        if largest < tol:
            break

    schur, item_variances = _schur_complement(
        view_observations, item_observations, abilities, easiness, prior_theta, prior_d
    )
    ability_variances = _inverse_diagonal(schur)
    factor = _covariance_factor(schur)
    return RaschFit(
        abilities={strata[view]: abilities[view] for view in range(len(strata))},
        ability_sd={
            strata[view]: math.sqrt(max(ability_variances[view], 0.0))
            for view in range(len(strata))
        },
        difficulties={(outcome, items[item]): easiness[item] for item in range(len(items))},
        difficulty_sd={
            (outcome, items[item]): math.sqrt(max(item_variances[item], 0.0))
            for item in range(len(items))
        },
        hyper=(sigma_theta, sigma_d),
        outcomes=(outcome,),
        diagnostics=_diagnose(strata, len(items), view_observations, item_observations),
        _ability_factor=tuple(tuple(row) for row in factor),
    )


def equated_rates(
    fit: RaschFit, *, reference_items: Sequence[ItemKey] | None = None
) -> dict[tuple[str, str], float]:
    """True-score equating: each view's expected pass rate on a common bank.

    For every fitted view ``v`` and every reference item ``i``,

        ``equated_rate(v) = (1 / |R|) sum_i sigmoid(theta_v + d_i)``

    — what view ``v`` would score on the reference bank, rather than on the
    bank it happened to run. Two views of equal ability on a hard and an easy
    bank therefore converge, while their raw pass rates do not. The result is
    on the same 0-1 scale as a pass rate.

    ``reference_items`` is the bank ``R``, a sequence of global
    :data:`ItemKey` values. It defaults to every item in the fit (the union of
    the observed banks). Pass a pinned bank to keep the adjusted score's
    meaning stable as items are added or retired.

    @note: Equating is only as good as the link. When the fit is disconnected
        (``fit.diagnostics.connected`` is False) the reference items of one
        bank carry the *other* bank's difficulties from the prior, so the
        convergence is an assumption — surfaced on the diagnostics, never
        silently.
    @return: ``(stratum, outcome) -> equated pass rate``, in 0-1.
    @raise ValueError: If ``reference_items`` is empty, or names an item the
        fit does not cover.
    """
    if not fit.outcomes:
        if reference_items:
            raise ValueError("cannot equate an empty fit")
        return {}
    outcome = fit.outcomes[0]
    if reference_items is None:
        keys = list(fit.difficulties)
    else:
        keys = list(reference_items)
        if not keys:
            raise ValueError("reference_items must not be empty")
        unknown = [key for key in keys if key not in fit.difficulties]
        if unknown:
            raise ValueError(f"reference item not in the fit: {unknown[0]}")
    difficulties = [fit.difficulties[key] for key in keys]
    return {
        (stratum, outcome): sum(sigmoid(ability + value) for value in difficulties)
        / len(difficulties)
        for stratum, ability in fit.abilities.items()
    }


__all__ = [
    "DEFAULT_MAX_ITER",
    "DEFAULT_SIGMA",
    "DEFAULT_TOL",
    "SCHUR_MAX_VIEWS",
    "ItemKey",
    "ItemResponse",
    "RaschDiagnostics",
    "RaschFit",
    "ReferenceBank",
    "bank_digest",
    "equated_rates",
    "fit_rasch",
    "item_key",
    "log1p_exp",
    "log_sigmoid",
    "sigmoid",
]
