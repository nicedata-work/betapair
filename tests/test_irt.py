"""The item-identity contract and the penalised-JML Rasch fit."""

from __future__ import annotations

import math
import random
import statistics
import unicodedata
import unittest
from dataclasses import replace

from betapair import (
    ItemDrift,
    ItemResponse,
    RaschDiagnostics,
    RaschFit,
    ReferenceBank,
    bank_digest,
    equated_rates,
    fit_rasch,
    item_fingerprint,
    item_key,
)
from betapair.irt import (
    FINGERPRINT_CHARS,
    SCHUR_MAX_VIEWS,
    ItemKey,
    _backtrack,
    _cholesky,
    _covariance_factor,
    _diagonal_of_inverse,
    _inverse_diagonal,
    _inverse_from_cholesky,
    log_sigmoid,
    sigmoid,
)


def _simulate(
    seed: int,
    *,
    views: int = 8,
    items: int = 30,
    sigma: float = 1.5,
    shift: float = 0.0,
) -> tuple[dict[str, float], dict[str, float], list[ItemResponse]]:
    """A crossed view x item bank drawn from known abilities and easinesses."""
    rng = random.Random(seed)
    abilities = {f"v{index}": rng.gauss(0.0, sigma) + shift for index in range(views)}
    easiness = {f"q{index}": rng.gauss(0.0, sigma) - shift for index in range(items)}
    responses = []
    for view in range(views):
        for item in range(items):
            probability = sigmoid(abilities[f"v{view}"] + easiness[f"q{item}"])
            responses.append(
                ItemResponse(f"v{view}", "answer", f"q{item}", rng.random() < probability)
            )
    return abilities, easiness, responses


def _centred(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    return [value - mean for value in values]


def _rmse(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / len(left))


class ItemKeyTest(unittest.TestCase):
    def test_key_is_outcome_and_item(self) -> None:
        self.assertEqual(item_key("answer", "q5"), ("answer", "q5"))

    def test_key_rejects_empty_parts(self) -> None:
        with self.assertRaises(ValueError):
            item_key("", "q5")
        with self.assertRaises(ValueError):
            item_key("answer", "")


class ItemResponseTest(unittest.TestCase):
    def test_key_property_matches_the_module_helper(self) -> None:
        response = ItemResponse(stratum="v", outcome="answer", item="q5", passed=True)
        self.assertEqual(response.key, item_key("answer", "q5"))
        self.assertEqual(response.key, ("answer", "q5"))

    def test_key_ignores_the_stratum(self) -> None:
        """Only the outcome and the item string identify the item globally."""
        left = ItemResponse("view_a", "answer", "q5", True)
        right = ItemResponse("view_b", "answer", "q5", False)
        self.assertEqual(left.key, right.key)

    def test_outcome_is_part_of_the_key(self) -> None:
        left = ItemResponse("v", "answer", "q5", True)
        right = ItemResponse("v", "sql_valid", "q5", True)
        self.assertNotEqual(left.key, right.key)

    def test_rejects_empty_fields(self) -> None:
        with self.assertRaises(ValueError):
            ItemResponse("", "answer", "q5", True)
        with self.assertRaises(ValueError):
            ItemResponse("v", "", "q5", True)
        with self.assertRaises(ValueError):
            ItemResponse("v", "answer", "", True)

    def test_rejects_a_non_bool_pass(self) -> None:
        with self.assertRaises(ValueError):
            ItemResponse("v", "answer", "q5", 1)  # type: ignore[arg-type]

    def test_item_key_type_is_the_pair(self) -> None:
        key: ItemKey = ItemResponse("v", "answer", "q5", True).key
        self.assertEqual(key, ("answer", "q5"))


class RaschRecoveryTest(unittest.TestCase):
    """The MAP recovers the generating abilities and easinesses."""

    def test_recovers_a_crossed_bank(self) -> None:
        abilities, easiness, responses = _simulate(1)
        fit = fit_rasch(responses)
        names = sorted(abilities)
        true_theta = _centred([abilities[name] for name in names])
        fitted_theta = _centred([fit.abilities[name] for name in names])
        self.assertGreater(statistics.correlation(true_theta, fitted_theta), 0.9)
        self.assertLess(_rmse(true_theta, fitted_theta), 0.8)

        item_names = sorted(easiness)
        true_d = _centred([easiness[name] for name in item_names])
        fitted_d = _centred([fit.difficulties[("answer", name)] for name in item_names])
        self.assertGreater(statistics.correlation(true_d, fitted_d), 0.75)
        self.assertLess(_rmse(true_d, fitted_d), 1.0)

    def test_a_misspecified_prior_scale_degrades_gracefully(self) -> None:
        abilities, _, responses = _simulate(2)
        names = sorted(abilities)
        true_theta = _centred([abilities[name] for name in names])
        for sigma in (0.5, 5.0):
            fit = fit_rasch(responses, sigma_theta=sigma, sigma_d=sigma)
            fitted = _centred([fit.abilities[name] for name in names])
            self.assertGreater(statistics.correlation(true_theta, fitted), 0.85)
            self.assertTrue(all(math.isfinite(value) for value in fit.abilities.values()))


class IdentifiabilityTest(unittest.TestCase):
    """The priors pin the additive split; predictions do not follow it."""

    def test_an_additive_shift_does_not_move_predictions(self) -> None:
        _, _, base = _simulate(3)
        _, _, shifted = _simulate(3, shift=0.7)
        # theta + d is unchanged by the shift, so the responses are identical.
        self.assertEqual([r.passed for r in base], [r.passed for r in shifted])
        left = fit_rasch(base)
        right = fit_rasch(shifted)
        for view in range(8):
            for item in range(30):
                self.assertAlmostEqual(
                    left.probability(f"v{view}", f"q{item}"),
                    right.probability(f"v{view}", f"q{item}"),
                )

    def test_probability_is_sigmoid_of_ability_plus_easiness(self) -> None:
        _, _, responses = _simulate(4, views=3, items=4)
        fit = fit_rasch(responses)
        expected = sigmoid(fit.abilities["v0"] + fit.difficulties[("answer", "q0")])
        self.assertAlmostEqual(fit.probability("v0", "q0"), expected)
        with self.assertRaises(KeyError):
            fit.probability("missing", "q0")
        with self.assertRaises(KeyError):
            fit.probability("v0", "missing")


class SeparationTest(unittest.TestCase):
    """A perfectly separated view is finite because the prior is in the model."""

    def test_an_all_pass_view_stays_finite(self) -> None:
        responses = [
            ItemResponse("perfect", "answer", f"q{index}", True) for index in range(10)
        ] + [ItemResponse("mixed", "answer", f"q{index}", index % 2 == 0) for index in range(10)]
        fit = fit_rasch(responses)
        self.assertTrue(all(math.isfinite(value) for value in fit.abilities.values()))
        self.assertGreater(fit.abilities["perfect"], fit.abilities["mixed"])
        self.assertLess(fit.probability("perfect", "q0"), 1.0)
        self.assertTrue(all(value > 0.0 for value in fit.ability_sd.values()))

    def test_an_all_fail_view_stays_finite(self) -> None:
        responses = [
            ItemResponse("broken", "answer", f"q{index}", False) for index in range(10)
        ] + [ItemResponse("mixed", "answer", f"q{index}", index % 2 == 0) for index in range(10)]
        fit = fit_rasch(responses)
        self.assertGreater(fit.abilities["mixed"], fit.abilities["broken"])
        self.assertGreater(fit.probability("broken", "q0"), 0.0)


class DeterminismTest(unittest.TestCase):
    """No randomness enters the MAP, and response order is irrelevant."""

    def test_same_responses_give_the_same_fit(self) -> None:
        _, _, responses = _simulate(5)
        self.assertEqual(fit_rasch(responses), fit_rasch(list(responses)))

    def test_response_order_does_not_matter(self) -> None:
        _, _, responses = _simulate(6)
        shuffled = list(responses)
        random.Random(0).shuffle(shuffled)
        self.assertEqual(fit_rasch(responses), fit_rasch(shuffled))

    def test_a_loose_tolerance_stops_after_one_sweep(self) -> None:
        _, _, responses = _simulate(11, views=3, items=4)
        fit = fit_rasch(responses, tol=1e9)
        self.assertEqual(set(fit.abilities), {"v0", "v1", "v2"})
        self.assertTrue(all(math.isfinite(value) for value in fit.abilities.values()))

    def test_hyper_and_outcome_are_published(self) -> None:
        _, _, responses = _simulate(7, views=2, items=3)
        fit = fit_rasch(responses, sigma_theta=2.0, sigma_d=0.5)
        self.assertIsInstance(fit, RaschFit)
        self.assertEqual(fit.hyper, (2.0, 0.5))
        self.assertEqual(fit.outcomes, ("answer",))


class LaplaceTest(unittest.TestCase):
    """The Schur-complement posterior sd behaves like a posterior sd."""

    def test_more_items_narrow_the_ability_interval(self) -> None:
        _, _, few = _simulate(8, views=6, items=3)
        _, _, many = _simulate(8, views=6, items=60)
        few_fit = fit_rasch(few)
        many_fit = fit_rasch(many)
        self.assertLess(
            statistics.fmean(many_fit.ability_sd.values()),
            statistics.fmean(few_fit.ability_sd.values()),
        )

    def test_shared_items_narrow_the_item_intervals(self) -> None:
        """Two views on one shared bank are linked; on disjoint banks they are not.

        An item seen by two views has twice the evidence, so its difficulty
        interval is narrower. Disjoint banks still fit — the prior bridges
        them — but each item rests on one view.
        """
        rng = random.Random(9)
        shared = [
            ItemResponse(view, "answer", f"q{item}", rng.random() < 0.6)
            for view in ("a", "b")
            for item in range(40)
        ]
        disjoint = [
            ItemResponse("a", "answer", f"a{item}", rng.random() < 0.6) for item in range(40)
        ] + [ItemResponse("b", "answer", f"b{item}", rng.random() < 0.6) for item in range(40)]
        shared_fit = fit_rasch(shared)
        disjoint_fit = fit_rasch(disjoint)
        self.assertLess(
            statistics.fmean(shared_fit.difficulty_sd.values()),
            statistics.fmean(disjoint_fit.difficulty_sd.values()),
        )
        self.assertTrue(all(math.isfinite(value) for value in disjoint_fit.abilities.values()))

    def test_ability_and_item_sds_are_positive(self) -> None:
        _, _, responses = _simulate(10, views=4, items=6)
        fit = fit_rasch(responses)
        self.assertTrue(all(value > 0.0 for value in fit.ability_sd.values()))
        self.assertTrue(all(value > 0.0 for value in fit.difficulty_sd.values()))


class EdgeCaseTest(unittest.TestCase):
    """Empty, single, duplicated and mixed input."""

    def test_no_responses_yields_an_empty_fit(self) -> None:
        fit = fit_rasch([])
        self.assertEqual(fit.abilities, {})
        self.assertEqual(fit.ability_sd, {})
        self.assertEqual(fit.difficulties, {})
        self.assertEqual(fit.outcomes, ())
        self.assertEqual(fit.hyper, (1.5, 1.5))

    def test_one_view_and_one_item(self) -> None:
        fit = fit_rasch([ItemResponse("v", "answer", "q", True)])
        self.assertEqual(fit.outcomes, ("answer",))
        self.assertEqual(set(fit.abilities), {"v"})
        self.assertEqual(set(fit.difficulties), {("answer", "q")})
        self.assertTrue(math.isfinite(fit.abilities["v"]))

    def test_a_single_item_seen_by_two_views(self) -> None:
        fit = fit_rasch(
            [
                ItemResponse("a", "answer", "q", True),
                ItemResponse("b", "answer", "q", False),
            ]
        )
        self.assertGreater(fit.abilities["a"], fit.abilities["b"])

    def test_duplicate_rows_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_rasch(
                [
                    ItemResponse("v", "answer", "q", True),
                    ItemResponse("v", "answer", "q", False),
                ]
            )

    def test_outcomes_share_one_ability_per_view(self) -> None:
        # "answer" says a is much stronger than b; "spec" says the opposite.
        # The shared ability sits between the two independent estimates.
        responses = [ItemResponse("a", "answer", f"q{index}", True) for index in range(20)] + [
            ItemResponse("b", "answer", f"q{index}", False) for index in range(20)
        ]
        responses += [ItemResponse("a", "spec", f"s{index}", False) for index in range(2)]
        responses += [ItemResponse("b", "spec", f"s{index}", True) for index in range(2)]
        shared = fit_rasch(responses)
        answer = fit_rasch([r for r in responses if r.outcome == "answer"])
        spec = fit_rasch([r for r in responses if r.outcome == "spec"])
        self.assertEqual(shared.outcomes, ("answer", "spec"))
        shared_gap = shared.abilities["a"] - shared.abilities["b"]
        self.assertLess(shared_gap, answer.abilities["a"] - answer.abilities["b"])
        self.assertGreater(shared_gap, spec.abilities["a"] - spec.abilities["b"])
        self.assertIn(("spec", "s0"), shared.difficulties)
        self.assertIn(("answer", "q0"), shared.difficulties)

    def test_a_multi_outcome_fit_needs_the_outcome_to_predict(self) -> None:
        fit = fit_rasch(
            [
                ItemResponse("v", "answer", "q", True),
                ItemResponse("v", "spec", "q", True),
            ]
        )
        self.assertAlmostEqual(
            fit.probability("v", "q", "answer"), fit.probability("v", "q", "spec")
        )
        with self.assertRaises(ValueError):
            fit.probability("v", "q")
        with self.assertRaises(KeyError):
            fit_rasch([]).probability("v", "q")
        rates = equated_rates(fit)
        self.assertEqual(set(rates), {("v", "answer"), ("v", "spec")})

    def test_the_same_item_string_under_two_outcomes_is_two_items(self) -> None:
        fit = fit_rasch(
            [
                ItemResponse("v", "answer", "q", True),
                ItemResponse("v", "spec", "q", False),
            ]
        )
        self.assertEqual(set(fit.difficulties), {("answer", "q"), ("spec", "q")})
        self.assertGreater(fit.difficulties[("answer", "q")], fit.difficulties[("spec", "q")])

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        response = [ItemResponse("v", "answer", "q", True)]
        for kwargs in (
            {"sigma_theta": 0.0},
            {"sigma_theta": -1.0},
            {"sigma_d": 0.0},
            {"sigma_d": -1.0},
            {"max_iter": 0},
            {"tol": 0.0},
            {"tol": -1.0},
        ):
            with self.assertRaises(ValueError):
                fit_rasch(response, **kwargs)  # type: ignore[arg-type]


class NumericHelperTest(unittest.TestCase):
    """The stable link and the small dense linear algebra."""

    def test_sigmoid_and_log_sigmoid_are_stable_in_both_tails(self) -> None:
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertAlmostEqual(sigmoid(1000.0), 1.0)
        self.assertAlmostEqual(sigmoid(-1000.0), 0.0)
        self.assertAlmostEqual(log_sigmoid(0.0), math.log(0.5))
        self.assertAlmostEqual(log_sigmoid(3.0), math.log(sigmoid(3.0)))
        self.assertAlmostEqual(log_sigmoid(-1000.0), -1000.0)
        self.assertTrue(math.isfinite(log_sigmoid(-1e6)))

    def test_cholesky_factorises_a_positive_definite_matrix(self) -> None:
        matrix = [[4.0, 1.0], [1.0, 3.0]]
        lower = _cholesky(matrix)
        self.assertAlmostEqual(lower[0][0], 2.0)
        self.assertAlmostEqual(lower[0][1], 0.0)
        self.assertAlmostEqual(lower[1][0], 0.5)
        self.assertAlmostEqual(lower[1][1] * lower[1][1], 3.0 - 0.25)

    def test_cholesky_jitters_a_singular_diagonal(self) -> None:
        lower = _cholesky([[0.0]])
        self.assertGreater(lower[0][0], 0.0)
        self.assertLess(lower[0][0], 1e-3)

    def test_cholesky_rejects_a_non_positive_definite_matrix(self) -> None:
        with self.assertRaises(ValueError):
            _cholesky([[-1.0]])

    def test_diagonal_of_inverse_matches_the_analytic_inverse(self) -> None:
        matrix = [[4.0, 1.0], [1.0, 3.0]]
        diagonal = _diagonal_of_inverse(_cholesky(matrix))
        self.assertAlmostEqual(diagonal[0], 3.0 / 11.0)
        self.assertAlmostEqual(diagonal[1], 4.0 / 11.0)

    def test_inverse_from_cholesky_matches_the_analytic_inverse(self) -> None:
        matrix = [[4.0, 1.0], [1.0, 3.0]]
        inverse = _inverse_from_cholesky(_cholesky(matrix))
        self.assertAlmostEqual(inverse[0][0], 3.0 / 11.0)
        self.assertAlmostEqual(inverse[0][1], -1.0 / 11.0)
        self.assertAlmostEqual(inverse[1][0], -1.0 / 11.0)
        self.assertAlmostEqual(inverse[1][1], 4.0 / 11.0)

    def test_covariance_factor_multiplies_to_the_inverse_variance(self) -> None:
        schur = [[4.0, 1.0], [1.0, 3.0]]
        factor = _covariance_factor(schur)
        product = [
            [sum(factor[row][k] * factor[column][k] for k in range(2)) for column in range(2)]
            for row in range(2)
        ]
        self.assertAlmostEqual(product[0][0], 3.0 / 11.0)
        self.assertAlmostEqual(product[0][1], -1.0 / 11.0)
        self.assertAlmostEqual(product[1][1], 4.0 / 11.0)

    def test_covariance_factor_falls_back_when_unavailable(self) -> None:
        self.assertEqual(_covariance_factor([]), [])
        self.assertEqual(_covariance_factor([[-1.0]]), [])
        size = SCHUR_MAX_VIEWS + 1
        self.assertEqual(_covariance_factor([[0.0] * size for _ in range(size)]), [])

    def test_inverse_diagonal_falls_back_on_a_singular_matrix(self) -> None:
        self.assertEqual(_inverse_diagonal([]), [])
        self.assertEqual(_inverse_diagonal([[-1.0]]), [math.inf])

    def test_inverse_diagonal_falls_back_above_the_view_cap(self) -> None:
        size = SCHUR_MAX_VIEWS + 1
        matrix = [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]
        self.assertEqual(_inverse_diagonal(matrix), [1.0] * size)

    def test_backtrack_accepts_a_concave_step_and_halves_a_bad_one(self) -> None:
        concave = lambda value: -value * value + 2.0 * value  # noqa: E731
        self.assertAlmostEqual(_backtrack(0.0, 1.0, concave), 1.0)
        overshoot = lambda value: -(value**4)  # noqa: E731
        accepted = _backtrack(0.0, 2.0, overshoot)
        self.assertLess(abs(accepted), 0.01)


class SampleAbilitiesTest(unittest.TestCase):
    """Seeded posterior draws of ``theta``, the interval's raw material."""

    def test_draws_are_seeded_and_deterministic(self) -> None:
        _, _, responses = _simulate(3)
        fit = fit_rasch(responses)
        self.assertEqual(
            fit.sample_abilities(draws=5, seed=7), fit.sample_abilities(draws=5, seed=7)
        )
        self.assertNotEqual(
            fit.sample_abilities(draws=5, seed=7), fit.sample_abilities(draws=5, seed=8)
        )

    def test_draws_centre_on_the_estimate_with_the_measured_spread(self) -> None:
        _, _, responses = _simulate(4)
        fit = fit_rasch(responses)
        draws = fit.sample_abilities(draws=4000, seed=0)
        self.assertEqual(set(draws), set(fit.abilities))
        for stratum, values in draws.items():
            self.assertAlmostEqual(statistics.fmean(values), fit.abilities[stratum], places=1)
            self.assertAlmostEqual(statistics.pstdev(values), fit.ability_sd[stratum], delta=0.05)

    def test_a_factor_is_built_for_a_small_connected_fit(self) -> None:
        _, _, responses = _simulate(5)
        fit = fit_rasch(responses)
        self.assertTrue(fit._ability_factor)
        draws = fit.sample_abilities(draws=4, seed=0)
        self.assertTrue(all(len(values) == 4 for values in draws.values()))

    def test_without_a_factor_draws_are_independent_normals(self) -> None:
        _, _, responses = _simulate(6)
        fit = fit_rasch(responses)
        flat = replace(fit, _ability_factor=())
        draws = flat.sample_abilities(draws=4000, seed=0)
        for stratum, values in draws.items():
            self.assertAlmostEqual(statistics.pstdev(values), fit.ability_sd[stratum], delta=0.05)

    def test_empty_fit_has_no_draws(self) -> None:
        self.assertEqual(fit_rasch([]).sample_abilities(draws=3, seed=0), {})

    def test_draws_must_be_positive(self) -> None:
        _, _, responses = _simulate(7)
        with self.assertRaises(ValueError):
            fit_rasch(responses).sample_abilities(draws=0, seed=0)


def _hard_and_easy_banks(
    seed: int = 5, *, anchors: int = 30, bank: int = 20, views: int = 20
) -> list[ItemResponse]:
    """Equal-ability views split across a hard and an easy bank, linked by anchors."""
    rng = random.Random(seed)
    anchor_easiness = [rng.gauss(0.0, 0.6) for _ in range(anchors)]
    responses: list[ItemResponse] = []
    for stratum, easiness in (("hard", -1.5), ("easy", 1.5)):
        for view in range(views):
            for index, value in enumerate(anchor_easiness):
                responses.append(
                    ItemResponse(
                        f"{stratum}{view}",
                        "answer",
                        f"anchor{index}",
                        rng.random() < sigmoid(value),
                    )
                )
            for index in range(bank):
                responses.append(
                    ItemResponse(
                        f"{stratum}{view}",
                        "answer",
                        f"{stratum}{index}",
                        rng.random() < sigmoid(easiness),
                    )
                )
    return responses


def _raw_rate(responses: list[ItemResponse], stratum: str) -> float:
    rows = [response for response in responses if response.stratum == stratum]
    return sum(response.passed for response in rows) / len(rows)


class EquatingTest(unittest.TestCase):
    """True-score equating: the score a view would get on a common bank."""

    def test_equal_ability_views_on_hard_and_easy_banks_converge(self) -> None:
        responses = _hard_and_easy_banks()
        fit = fit_rasch(responses)
        rates = equated_rates(fit)
        raw_gap = abs(_raw_rate(responses, "hard0") - _raw_rate(responses, "easy0"))
        equated_gap = abs(rates[("hard0", "answer")] - rates[("easy0", "answer")])
        self.assertGreater(raw_gap, 0.15)
        self.assertLess(equated_gap, 0.05)
        self.assertLess(equated_gap, raw_gap)

    def test_default_reference_is_the_observed_union(self) -> None:
        _, _, responses = _simulate(12, views=4, items=6)
        fit = fit_rasch(responses)
        explicit = equated_rates(fit, reference_items=list(fit.difficulties))
        self.assertEqual(equated_rates(fit), explicit)

    def test_rate_is_the_mean_fitted_probability_over_the_reference(self) -> None:
        _, _, responses = _simulate(13, views=3, items=4)
        fit = fit_rasch(responses)
        keys = [("answer", "q0"), ("answer", "q2")]
        expected = sum(sigmoid(fit.abilities["v0"] + fit.difficulties[key]) for key in keys) / len(
            keys
        )
        self.assertAlmostEqual(equated_rates(fit, reference_items=keys)[("v0", "answer")], expected)

    def test_a_single_reference_item_is_its_fitted_probability(self) -> None:
        _, _, responses = _simulate(14, views=2, items=3)
        fit = fit_rasch(responses)
        key = ("answer", "q1")
        rates = equated_rates(fit, reference_items=[key])
        self.assertAlmostEqual(rates[("v0", "answer")], fit.probability("v0", "q1"))

    def test_an_empty_fit_equates_to_nothing(self) -> None:
        self.assertEqual(equated_rates(fit_rasch([])), {})
        with self.assertRaises(ValueError):
            equated_rates(fit_rasch([]), reference_items=[("answer", "q")])

    def test_an_empty_reference_is_rejected(self) -> None:
        _, _, responses = _simulate(15, views=2, items=3)
        with self.assertRaises(ValueError):
            equated_rates(fit_rasch(responses), reference_items=[])

    def test_an_unknown_reference_item_is_rejected(self) -> None:
        _, _, responses = _simulate(16, views=2, items=3)
        with self.assertRaises(ValueError):
            equated_rates(fit_rasch(responses), reference_items=[("answer", "missing")])


class DiagnosticsTest(unittest.TestCase):
    """How well the banks are actually linked, carried on every fit."""

    @staticmethod
    def _banks(anchors: int, unique: int) -> list[ItemResponse]:
        rng = random.Random(3)
        responses: list[ItemResponse] = []
        for index in range(anchors):
            for view in ("a", "b"):
                responses.append(ItemResponse(view, "answer", f"s{index}", rng.random() < 0.5))
        for view in ("a", "b"):
            for index in range(unique):
                responses.append(ItemResponse(view, "answer", f"{view}{index}", rng.random() < 0.5))
        return responses

    def test_disjoint_banks_are_flagged_and_widen(self) -> None:
        disjoint = fit_rasch(self._banks(0, 8))
        anchored = fit_rasch(self._banks(4, 8))
        self.assertEqual(disjoint.diagnostics.components, 2)
        self.assertFalse(disjoint.diagnostics.connected)
        self.assertEqual(disjoint.diagnostics.weakest_link, 0)
        self.assertEqual(disjoint.diagnostics.shared_items, 0)
        self.assertTrue(disjoint.diagnostics.prior_only)
        self.assertGreater(
            statistics.fmean(disjoint.ability_sd.values()),
            statistics.fmean(anchored.ability_sd.values()),
        )

    def test_anchors_connect_the_banks(self) -> None:
        anchored = fit_rasch(self._banks(4, 8))
        self.assertEqual(anchored.diagnostics.components, 1)
        self.assertTrue(anchored.diagnostics.connected)
        self.assertEqual(anchored.diagnostics.weakest_link, 4)
        self.assertEqual(anchored.diagnostics.shared_items, 4)
        self.assertFalse(anchored.diagnostics.prior_only)

    def test_weakest_link_is_the_sparsest_view_pair(self) -> None:
        responses = [
            ItemResponse(view, "answer", f"s{index}", True)
            for view in ("a", "b", "c")
            for index in range(3)
        ] + [ItemResponse("a", "answer", f"only{index}", True) for index in range(2)]
        diagnostics = fit_rasch(responses).diagnostics
        self.assertEqual(diagnostics.components, 1)
        self.assertEqual(diagnostics.weakest_link, 3)
        self.assertEqual(diagnostics.shared_items, 3)

    def test_separated_views_are_flagged(self) -> None:
        responses = [ItemResponse("perfect", "answer", f"q{i}", True) for i in range(6)]
        responses += [ItemResponse("broken", "answer", f"q{i}", False) for i in range(6)]
        responses += [ItemResponse("mixed", "answer", f"q{i}", i % 2 == 0) for i in range(6)]
        diagnostics = fit_rasch(responses).diagnostics
        self.assertEqual(diagnostics.prior_dominated, ("broken", "perfect"))
        self.assertEqual(diagnostics.separated_items, 0)

    def test_separated_items_are_counted(self) -> None:
        # q0/q1 are all-pass, q2 all-fail and q3 splits the two views, so only
        # the first three items are separated (and neither view is).
        responses = [
            ItemResponse("a", "answer", "q0", True),
            ItemResponse("b", "answer", "q0", True),
            ItemResponse("a", "answer", "q1", True),
            ItemResponse("b", "answer", "q1", True),
            ItemResponse("a", "answer", "q2", False),
            ItemResponse("b", "answer", "q2", False),
            ItemResponse("a", "answer", "q3", True),
            ItemResponse("b", "answer", "q3", False),
        ]
        diagnostics = fit_rasch(responses).diagnostics
        self.assertEqual(diagnostics.separated_items, 3)
        self.assertEqual(diagnostics.prior_dominated, ())

    def test_a_single_view_is_connected(self) -> None:
        responses = [ItemResponse("only", "answer", f"q{i}", i % 2 == 0) for i in range(5)]
        diagnostics = fit_rasch(responses).diagnostics
        self.assertEqual(diagnostics.components, 1)
        self.assertTrue(diagnostics.connected)
        self.assertEqual(diagnostics.weakest_link, 0)
        self.assertEqual(diagnostics.shared_items, 0)
        self.assertFalse(diagnostics.prior_only)  # no cross-view comparison to make
        self.assertEqual(diagnostics.prior_dominated, ())

    def test_shared_items_is_global_not_the_sparsest_pair(self) -> None:
        # a and b share three anchors; c is disjoint. weakest_link is 0 but
        # three items still carry cross-view evidence, so this is not prior-only.
        responses = [
            ItemResponse(view, "answer", f"s{index}", True)
            for view in ("a", "b")
            for index in range(3)
        ]
        responses += [ItemResponse("c", "answer", f"c{index}", True) for index in range(3)]
        diagnostics = fit_rasch(responses).diagnostics
        self.assertEqual(diagnostics.shared_items, 3)
        self.assertEqual(diagnostics.weakest_link, 0)
        self.assertFalse(diagnostics.connected)
        self.assertFalse(diagnostics.prior_only)

    def test_an_empty_fit_has_neutral_diagnostics(self) -> None:
        self.assertEqual(fit_rasch([]).diagnostics, RaschDiagnostics())

    def test_diagnostics_validate_their_fields(self) -> None:
        with self.assertRaises(ValueError):
            RaschDiagnostics(components=-1)
        with self.assertRaises(ValueError):
            RaschDiagnostics(weakest_link=-1)
        with self.assertRaises(ValueError):
            RaschDiagnostics(shared_items=-1)
        with self.assertRaises(ValueError):
            RaschDiagnostics(separated_items=-1)
        with self.assertRaises(ValueError):
            RaschDiagnostics(components=2, connected=True)
        self.assertFalse(RaschDiagnostics(components=2, connected=False).connected)


class ReferenceBankTest(unittest.TestCase):
    """A named, versioned item set defining the adjusted scale."""

    def test_bank_canonicalises_and_validates_its_items(self) -> None:
        bank = ReferenceBank(
            name="standard",
            version="2",
            items=(("answer", "q2"), ("answer", "q1"), ("spec", "q1")),
        )
        self.assertEqual(bank.items, (("answer", "q1"), ("answer", "q2"), ("spec", "q1")))
        self.assertEqual(bank.label, "standard@2")
        for kwargs in (
            {"name": "", "version": "1", "items": (("answer", "q"),)},
            {"name": "n", "version": "", "items": (("answer", "q"),)},
            {"name": "n", "version": "1", "items": ()},
            {"name": "n", "version": "1", "items": (("answer", ""),)},
            {"name": "n", "version": "1", "items": (("answer", "q"), ("answer", "q"))},
        ):
            with self.assertRaises(ValueError):
                ReferenceBank(**kwargs)  # type: ignore[arg-type]

    def test_from_responses_takes_the_distinct_global_items(self) -> None:
        _, _, responses = _simulate(1, views=2, items=4)
        bank = ReferenceBank.from_responses(responses, name="std", version="1")
        self.assertEqual(len(bank.items), 4)
        self.assertTrue(all(key[0] == "answer" for key in bank.items))

    def test_digest_ignores_order_duplicates_and_labels_but_not_items(self) -> None:
        keys = (("answer", "q1"), ("answer", "q2"))
        self.assertEqual(bank_digest(keys), bank_digest(tuple(reversed(keys))))
        self.assertEqual(bank_digest(keys), ReferenceBank("a", "1", keys).digest)
        self.assertEqual(ReferenceBank("a", "1", keys).digest, ReferenceBank("b", "9", keys).digest)
        self.assertNotEqual(
            ReferenceBank("a", "1", keys).digest,
            ReferenceBank("a", "1", (("answer", "q1"),)).digest,
        )


class FingerprintTest(unittest.TestCase):
    """A content fingerprint identifies meaning, not formatting."""

    TEXT = "What is the total revenue by region?"

    def test_format_is_versioned_and_short(self) -> None:
        fingerprint = item_fingerprint("answer", self.TEXT)
        self.assertTrue(fingerprint.startswith("fp1_"))
        self.assertEqual(len(fingerprint), len("fp1_") + FINGERPRINT_CHARS)
        self.assertTrue(all(char in "0123456789abcdef" for char in fingerprint[4:]))

    def test_formatting_does_not_mint_a_new_fingerprint(self) -> None:
        base = item_fingerprint("answer", self.TEXT)
        self.assertEqual(base, item_fingerprint("answer", f"  {self.TEXT}  "))
        self.assertEqual(base, item_fingerprint("answer", "what is the TOTAL revenue\nby region?"))
        self.assertEqual(base, item_fingerprint("answer", "What  is the total revenue\tby region?"))
        composed = "What is the total revenue by r\u00e9gion?"
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertNotEqual(composed, decomposed)
        self.assertEqual(
            item_fingerprint("answer", composed), item_fingerprint("answer", decomposed)
        )

    def test_wording_and_outcome_do_mint_a_new_fingerprint(self) -> None:
        base = item_fingerprint("answer", self.TEXT)
        self.assertNotEqual(base, item_fingerprint("answer", "What is the total revenue by month?"))
        self.assertNotEqual(base, item_fingerprint("answer", "What is the total revenue by region"))
        self.assertNotEqual(base, item_fingerprint("sql_valid", self.TEXT))

    def test_determinism_across_calls(self) -> None:
        self.assertEqual(
            item_fingerprint("answer", self.TEXT), item_fingerprint("answer", self.TEXT)
        )

    def test_empty_inputs_are_rejected(self) -> None:
        for outcome, text in (("", self.TEXT), ("answer", ""), ("answer", "   ")):
            with self.assertRaises(ValueError):
                item_fingerprint(outcome, text)


class ItemDriftTest(unittest.TestCase):
    """A drift row pairs the first fingerprint with a later, different one."""

    def test_key_and_fields(self) -> None:
        row = ItemDrift("answer", "q1", "fp1_aaa", "fp1_bbb")
        self.assertEqual(row.key, ("answer", "q1"))
        self.assertEqual((row.expected, row.observed), ("fp1_aaa", "fp1_bbb"))

    def test_empty_fields_are_rejected(self) -> None:
        for args in (
            ("", "q1", "fp1_a", "fp1_b"),
            ("answer", "", "fp1_a", "fp1_b"),
            ("answer", "q1", "", "fp1_b"),
            ("answer", "q1", "fp1_a", ""),
        ):
            with self.assertRaises(ValueError):
                ItemDrift(*args)

    def test_identical_fingerprints_are_not_drift(self) -> None:
        with self.assertRaises(ValueError):
            ItemDrift("answer", "q1", "fp1_aaa", "fp1_aaa")


if __name__ == "__main__":
    unittest.main()
