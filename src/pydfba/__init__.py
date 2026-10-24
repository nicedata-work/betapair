"""pydfba — dependency-light Bayesian analysis of paired binary outcomes.

This package ports the statistical core of ``DFBA::dfba_mcnemar()`` to Python
and extends it to many strata at once:

* :mod:`pydfba.beta` — stdlib beta / beta-binomial numerics (no numpy/scipy);
* :mod:`pydfba.mcnemar` — ``mcnemar()``, a faithful port of ``dfba_mcnemar``;
* :mod:`pydfba.hierarchy` — hierarchical (multi-strata) pooling and a
  posterior predictive homogeneity guardrail;
* :mod:`pydfba.report` — the assembled report models and ``build_report()``;
* :mod:`pydfba.tracker` — streaming/batch accumulation and JSON snapshots.

The intended surface is small::

    from pydfba import HealthTracker, PairRecord

    tracker = HealthTracker(weights={"answer": 1.0}, category_weights={"tier-1": 3.0})
    tracker.observe("model_a", "answer", "q1", baseline=False, candidate=True)
    tracker.observe_many([PairRecord("model_b", "answer", "q1", False, False)])
    report = tracker.report()
    report.score, report.delta_pp, report.contributions, report.guardrail

Nothing in the numeric core imports numpy/scipy/pandas; ``pydantic`` is used
only by :mod:`pydfba.report` for the serialisable report models.
"""

from pydfba.beta import beta_cdf, beta_pdf, beta_ppf
from pydfba.hierarchy import (
    Cell,
    CellPosterior,
    CellPosteriors,
    Fit,
    Homogeneity,
    cell_posteriors,
    fit,
    homogeneity_ppc,
    pooled_counts,
)
from pydfba.mcnemar import McNemarResult, mcnemar
from pydfba.report import (
    DEFAULT_CATEGORY,
    REPORT_VERSION,
    CategorySummary,
    CellContribution,
    CellCoverage,
    CellVerdict,
    Coverage,
    Guardrail,
    HealthReport,
    HierarchySummary,
    McNemarSummary,
    RegressionModel,
    Status,
    Weighting,
    build_report,
    cell_verdicts,
    delta_status,
    parse_category_weights,
    regression_model,
    verdicts_from_model,
    weighting,
)
from pydfba.tracker import ERROR, HealthTracker, PairRecord

__all__ = [
    "DEFAULT_CATEGORY",
    "ERROR",
    "REPORT_VERSION",
    "CategorySummary",
    "Cell",
    "CellContribution",
    "CellCoverage",
    "CellPosterior",
    "CellPosteriors",
    "CellVerdict",
    "Coverage",
    "Fit",
    "Guardrail",
    "HealthReport",
    "HealthTracker",
    "HierarchySummary",
    "Homogeneity",
    "McNemarResult",
    "McNemarSummary",
    "PairRecord",
    "RegressionModel",
    "Status",
    "Weighting",
    "beta_cdf",
    "beta_pdf",
    "beta_ppf",
    "build_report",
    "cell_posteriors",
    "cell_verdicts",
    "delta_status",
    "fit",
    "homogeneity_ppc",
    "mcnemar",
    "parse_category_weights",
    "pooled_counts",
    "regression_model",
    "verdicts_from_model",
    "weighting",
]
