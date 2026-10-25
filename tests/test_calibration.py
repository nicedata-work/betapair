"""Operating-characteristic checks behind ``docs/validation.md``.

The full sweep lives in ``tools/calibration.py``. These are bounded versions
that run in CI and fail if the Bayesian regression gate drifts: under a
realistic global null it must stay quiet, criticality must raise power, and a
heterogeneous layer must never read as a clean improvement. The last class
pins the B5 evidence: re-centring the prior on a measured base rate raises the
power to flag a real regression.
"""

from __future__ import annotations

import random
import unittest

from betapair import Cell, ScorePoint, build_report, cell_verdicts, resolve_calibration


def _draw(rng: random.Random, n: int, q: float) -> tuple[int, int]:
    improved = sum(1 for _ in range(n) if rng.random() < q)
    return improved, n - improved


def _cell(name: str, improved: int, regressed: int, category: str | None = None) -> Cell:
    return Cell(
        stratum=name,
        outcome="answer",
        improved=improved,
        regressed=regressed,
        passed=improved,
        pairs=improved + regressed,
        category=category,
    )


class BayesianFdrTest(unittest.TestCase):
    def _null_flags(self, weighted: bool, trials: int = 150, seed: int = 0) -> tuple[float, float]:
        """Return (fraction of trials with a flag, flags per trial) under the null."""
        rng = random.Random(seed)
        flagged_trials = 0
        total_flags = 0
        for _ in range(trials):
            cells = []
            for index in range(40):
                improved, regressed = _draw(rng, 20, 0.5)
                category = (
                    "tier-1" if (weighted and index == 0) else ("tier-3" if weighted else None)
                )
                cells.append(_cell(f"v{index}", improved, regressed, category))
            weights = None
            if weighted:
                weights = {cell.key: (10.0 if cell.category == "tier-1" else 1.0) for cell in cells}
            verdicts = cell_verdicts(cells, weights=weights)
            flags = [result for result in verdicts.values() if result.verdict == "regressing"]
            total_flags += len(flags)
            flagged_trials += bool(flags)
        return flagged_trials / trials, total_flags / trials

    def test_global_null_rarely_flags(self) -> None:
        for weighted in (False, True):
            any_rate, per_trial = self._null_flags(weighted)
            self.assertLessEqual(any_rate, 0.05, f"weighted={weighted}")
            self.assertLessEqual(per_trial, 0.10, f"weighted={weighted}")


class PowerTest(unittest.TestCase):
    def _power(self, weighted: bool, trials: int = 150, seed: int = 0) -> float:
        rng = random.Random(seed)
        hits = 0
        for _ in range(trials):
            improved, regressed = _draw(rng, 20, 0.2)
            cells = [_cell("critical", improved, regressed, "tier-1" if weighted else None)]
            for index in range(40):
                aux_improved, aux_regressed = _draw(rng, 20, 0.5)
                cells.append(
                    _cell(
                        f"aux{index}", aux_improved, aux_regressed, "tier-3" if weighted else None
                    )
                )
            weights = None
            if weighted:
                weights = {cell.key: (10.0 if cell.category == "tier-1" else 1.0) for cell in cells}
            verdicts = cell_verdicts(cells, weights=weights)
            hits += verdicts[("critical", "answer")].verdict == "regressing"
        return hits / trials

    def test_criticality_weight_raises_power(self) -> None:
        self.assertGreater(self._power(True), self._power(False))


class PriorCalibrationEvidenceTest(unittest.TestCase):
    """B5: the prior table rests on a real power difference."""

    @staticmethod
    def _prior(rate: float) -> tuple[float, float]:
        resolved = resolve_calibration(
            [ScorePoint(score=60.0, pairs=50.0, pi_mean=rate)],
            calibrate=True,
            rope_pp=None,
            pi_prior=(1.0, 1.0),
            max_unresolved_share=0.25,
        )
        return resolved.pi_prior

    def _power(self, prior: tuple[float, float], trials: int = 150) -> float:
        """Flag rate for one true regressor among 40 null auxiliaries."""
        rng = random.Random(0)
        hits = 0
        for _ in range(trials):
            improved, regressed = _draw(rng, 20, 0.2)
            cells = [_cell("critical", improved, regressed)]
            for index in range(40):
                aux_improved, aux_regressed = _draw(rng, 20, 0.5)
                cells.append(_cell(f"aux{index}", aux_improved, aux_regressed))
            verdicts = cell_verdicts(cells, pi_prior=prior)
            hits += verdicts[("critical", "answer")].verdict == "regressing"
        return hits / trials

    def test_a_higher_measured_base_rate_raises_power(self) -> None:
        strict = self._prior(0.02)
        permissive = self._prior(0.8)
        self.assertLess(strict[0], permissive[0])
        self.assertGreater(self._power(permissive), self._power(strict))


class HeterogeneityTest(unittest.TestCase):
    def test_heterogeneous_layers_never_read_improving(self) -> None:
        rng = random.Random(0)
        for _ in range(30):
            cells = []
            for index in range(9):
                improved, regressed = _draw(rng, 20, 0.62)
                cells.append(_cell(f"v{index}", improved, regressed))
            improved, regressed = _draw(rng, 20, 0.15)
            cells.append(_cell("broken", improved, regressed))
            report = build_report(cells, draws=64, ppc_draws=64, seed=0)
            if report.guardrail.status == "heterogeneous":
                self.assertNotEqual(report.status, "improving")


if __name__ == "__main__":
    unittest.main()
