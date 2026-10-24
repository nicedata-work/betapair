"""Beta-distribution numerics — stdlib ``math`` and ``random`` only.

The model is conjugate, so every quantity needed downstream is a beta density,
a beta tail probability, a beta quantile or a beta-binomial marginal
likelihood. ``scipy``/``numpy``/``pandas`` are deliberately not imported: this
is a few hundred lines of ``math`` calls, which keeps the package installable
anywhere a Python interpreter runs.

Numerics follow the textbook recipes (Abramowitz & Stegun 26.5.8 for the
regularized incomplete beta continued fraction; Marsaglia-Tsang for gamma
sampling). The continued fraction is iterated to :data:`_FP_EPS` with a hard
iteration cap, which reproduces ``DFBA::dfba_mcnemar``'s published output to
every printed digit (see ``tests/test_mcnemar.py``).
"""

from __future__ import annotations

import math
import random

#: Continued-fraction convergence tolerance and denominator floor.
_FP_EPS = 3e-14
_FP_MIN = 1e-300
#: Hard iteration cap for the continued fraction.
_CF_ITMAX = 400
#: Bisection steps for :func:`beta_ppf`.
_PPF_ITMAX = 64
#: Smallest/largest rates :func:`beta_sample` returns — the limits of double
#: precision, so the tail keeps its spread and downstream log-odds stay finite.
_MIN_RATE = math.ulp(0.0)
_MAX_RATE = math.nextafter(1.0, 0.0)


def log_beta(a: float, b: float) -> float:
    """``ln B(a, b)`` — via ``lgamma`` so large shape parameters stay finite."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def log_choose(n: int, k: int) -> float:
    """``ln C(n, k)`` without ever materialising the binomial coefficient."""
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def beta_pdf(x: float, a: float, b: float) -> float:
    """Beta density; the closed-form limit is used at the two boundaries.

    @raise ValueError: If either shape parameter is not positive.
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"shape parameters must be positive, got a={a}, b={b}")
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0:
        if a < 1.0:
            return math.inf
        return float(b) if a == 1.0 else 0.0
    if x == 1.0:
        if b < 1.0:
            return math.inf
        return float(a) if b == 1.0 else 0.0
    return math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x) - log_beta(a, b))


def _beta_cf(x: float, a: float, b: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method).

    @note: Iterated to :data:`_FP_EPS` with a hard :data:`_CF_ITMAX` cap. A
        vanishing denominator is floored at :data:`_FP_MIN` instead of being
        divided through, which keeps the recursion finite on extreme shapes.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _FP_MIN else _FP_MIN)
    h = d
    for m in range(1, _CF_ITMAX + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        d = 1.0 / (d if abs(d) > _FP_MIN else _FP_MIN)
        c = 1.0 + numerator / c
        c = c if abs(c) > _FP_MIN else _FP_MIN
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        d = 1.0 / (d if abs(d) > _FP_MIN else _FP_MIN)
        c = 1.0 + numerator / c
        c = c if abs(c) > _FP_MIN else _FP_MIN
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _FP_EPS:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` — the beta CDF.

    @note: The continued fraction converges fastest on the smaller of ``x`` and
        ``1 - x``, so the larger side is evaluated by reflection about .5.
    @raise ValueError: If either shape parameter is not positive.
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"shape parameters must be positive, got a={a}, b={b}")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_cf(x, a, b) / a
    return 1.0 - math.exp(log_front) * _beta_cf(1.0 - x, b, a) / b


def beta_ppf(p: float, a: float, b: float) -> float:
    """Beta quantile by bisection on :func:`beta_cdf` (monotone, robust).

    @note: :data:`_PPF_ITMAX` halvings leave the bracket below double
        resolution, so the midpoint is as sharp as the CDF allows.
    @raise ValueError: If ``p`` is not strictly inside (0, 1).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    low, high = 0.0, 1.0
    for _ in range(_PPF_ITMAX):
        mid = 0.5 * (low + high)
        if beta_cdf(mid, a, b) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def log_beta_binomial(k: int, n: int, a: float, b: float) -> float:
    """Log marginal likelihood of ``k`` successes in ``n`` trials under Beta(a, b).

    This is the closed form the hierarchical model integrates cells with —
    no quadrature, no sampling, so a fit is cheap and deterministic.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid binomial counts k={k}, n={n}")
    return log_choose(n, k) + log_beta(a + k, b + n - k) - log_beta(a, b)


def _log_gamma_sample(rng: random.Random, shape: float) -> float:
    """``ln`` of a Gamma(shape, 1) draw — finite for any positive shape.

    @raise ValueError: If ``shape`` is not positive.

    @note: Sampling the log keeps the beta ratio well defined when the draw is
        subnormal: ``ln U**(1/shape)`` is finite even when ``U**(1/shape)``
        underflows to zero, which is what collapsed the direct gamma ratio onto
        a point mass for tiny shapes.
    @note: ``shape < 1`` uses the boost identity in log space,
        ``ln G(shape) = ln G(shape+1) + ln(U)/shape``, with ``log1p(-U)`` so
        the ``U == 0`` corner stays finite.
    """
    if shape <= 0.0:
        raise ValueError(f"gamma shape must be positive, got {shape}")
    if shape < 1.0:
        return _log_gamma_sample(rng, shape + 1.0) + math.log1p(-rng.random()) / shape
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0.0, 1.0)
        v = (1.0 + c * x) ** 3
        if v <= 0.0:
            continue
        u = rng.random()
        if math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
            return math.log(d) + math.log(v)


def _logistic(value: float) -> float:
    """Numerically stable logistic function; never overflows."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def beta_sample(rng: random.Random, a: float, b: float) -> float:
    """Beta(a, b) draw via the gamma ratio — used for posterior intervals.

    @note: The ratio is taken in log space and squashed by a stable logistic,
        so a subnormal gamma draw cannot collapse the sample onto a point
        mass. The result is clamped to (0, 1) only at the limits of double
        precision, which keeps downstream log-odds finite without discarding
        the spread of the tail.
    @raise ValueError: If either shape parameter is not positive.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"shape parameters must be positive, got a={a}, b={b}")
    log_ratio = _log_gamma_sample(rng, a) - _log_gamma_sample(rng, b)
    return min(max(_logistic(log_ratio), _MIN_RATE), _MAX_RATE)
