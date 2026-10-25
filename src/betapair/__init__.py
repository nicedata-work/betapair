"""betapair — dependency-light Bayesian analysis of paired binary outcomes.

This package implements Bayesian analysis of paired binary outcomes and
extends it to many strata at once:

* :mod:`betapair.beta` — stdlib beta / beta-binomial numerics (no numpy/scipy);
* :mod:`betapair.mcnemar` — ``mcnemar()``, the Bayesian McNemar test;
* :mod:`betapair.hierarchy` — hierarchical (multi-strata) pooling and a
  posterior predictive homogeneity guardrail;
* :mod:`betapair.irt` — the global item-identity contract (``ItemResponse``,
  ``item_key``) for the latent-difficulty model;
* :mod:`betapair.report` — the assembled report models and ``build_report()``;
* :mod:`betapair.tracker` — streaming/batch accumulation and JSON snapshots.

The intended surface is small::

    from betapair import HealthTracker, PairRecord

    tracker = HealthTracker(weights={"answer": 1.0}, category_weights={"tier-1": 3.0})
    tracker.observe("model_a", "answer", "q1", baseline=False, candidate=True)
    tracker.observe_many([PairRecord("model_b", "answer", "q1", False, False)])
    report = tracker.report()
    report.score, report.delta_pp, report.contributions, report.guardrail

Nothing in the numeric core imports numpy/scipy/pandas; ``pydantic`` is used
only by :mod:`betapair.report` for the serialisable report models.
"""

from betapair.beta import beta_cdf, beta_pdf, beta_ppf, beta_sf
from betapair.hierarchy import (
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
from betapair.irt import (
    ItemKey,
    ItemResponse,
    RaschDiagnostics,
    RaschFit,
    ReferenceBank,
    bank_digest,
    equated_rates,
    fit_rasch,
    item_key,
)
from betapair.mcnemar import McNemarResult, SwitchEvidence, mcnemar, switch_evidence
from betapair.report import (
    DEFAULT_CATEGORY,
    REPORT_VERSION,
    Calibration,
    CategorySummary,
    CellContribution,
    CellCoverage,
    CellVerdict,
    Coverage,
    GroupSummary,
    Guardrail,
    HealthReport,
    HierarchySummary,
    IrTSummary,
    McNemarSummary,
    ReferenceBankSummary,
    RegressionModel,
    Status,
    UnpairedCell,
    Weighting,
    build_report,
    cell_verdicts,
    delta_status,
    parse_category_weights,
    regression_model,
    resolve_calibration,
    verdicts_from_model,
    weighting,
)
from betapair.series import ScoreBand, ScorePoint, score_band
from betapair.tracker import ERROR, FAILED, RETIRED, HealthTracker, PairRecord

__all__ = [
    "DEFAULT_CATEGORY",
    "ERROR",
    "FAILED",
    "REPORT_VERSION",
    "RETIRED",
    "Calibration",
    "CategorySummary",
    "Cell",
    "CellContribution",
    "CellCoverage",
    "CellPosterior",
    "CellPosteriors",
    "CellVerdict",
    "Coverage",
    "Fit",
    "GroupSummary",
    "Guardrail",
    "HealthReport",
    "HealthTracker",
    "HierarchySummary",
    "Homogeneity",
    "IrTSummary",
    "ItemKey",
    "ItemResponse",
    "McNemarResult",
    "McNemarSummary",
    "PairRecord",
    "RaschDiagnostics",
    "RaschFit",
    "ReferenceBank",
    "ReferenceBankSummary",
    "RegressionModel",
    "ScoreBand",
    "ScorePoint",
    "Status",
    "SwitchEvidence",
    "UnpairedCell",
    "Weighting",
    "bank_digest",
    "beta_cdf",
    "beta_pdf",
    "beta_ppf",
    "beta_sf",
    "build_report",
    "cell_posteriors",
    "cell_verdicts",
    "delta_status",
    "equated_rates",
    "fit",
    "fit_rasch",
    "homogeneity_ppc",
    "item_key",
    "mcnemar",
    "parse_category_weights",
    "pooled_counts",
    "regression_model",
    "resolve_calibration",
    "score_band",
    "switch_evidence",
    "verdicts_from_model",
    "weighting",
]
