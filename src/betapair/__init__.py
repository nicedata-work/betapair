"""betapair — dependency-light Bayesian analysis of paired binary outcomes.

This package implements Bayesian analysis of paired binary outcomes and
extends it to many strata at once:

* :mod:`betapair.beta` — stdlib beta / beta-binomial numerics (no numpy/scipy);
* :mod:`betapair.mcnemar` — ``mcnemar()``, the Bayesian McNemar test;
* :mod:`betapair.hierarchy` — hierarchical (multi-strata) pooling and a
  posterior predictive homogeneity guardrail;
* :mod:`betapair.irt` — the global item-identity contract (``ItemResponse``,
  ``item_key``) for the latent-difficulty model;
* :mod:`betapair.gate` — the dependence-aware regression gate and the ROPE
  layer verdict;
* :mod:`betapair.models` — the serialisable report schema and the pure-data
  policy objects (the only pydantic users);
* :mod:`betapair.report` — ``build_report()``, the decision policy and the
  assembly steps;
* :mod:`betapair.series` — score-over-time dispersion and windowed reads;
* :mod:`betapair.snapshot` — the tracker snapshot wire format (sentinels,
  record state, ``_parse_*`` helpers);
* :mod:`betapair.tracker` — streaming/batch accumulation and JSON snapshots.

The intended surface is small::

    from betapair import HealthTracker, PairRecord

    tracker = HealthTracker(weights={"answer": 1.0}, category_weights={"tier-1": 3.0})
    tracker.observe("model_a", "answer", "q1", baseline=False, candidate=True)
    tracker.observe_many([PairRecord("model_b", "answer", "q1", False, False)])
    report = tracker.report()
    report.score, report.delta_pp, report.contributions, report.guardrail

Nothing in the numeric core imports numpy/scipy/pandas; ``pydantic`` is used
only by :mod:`betapair.models` for the serialisable report schema.
"""

from importlib.metadata import PackageNotFoundError, version

from betapair.beta import beta_cdf, beta_pdf, beta_ppf, beta_sf
from betapair.hierarchy import (
    Cell,
    CellPosterior,
    CellPosteriors,
    HierarchyFit,
    Homogeneity,
    cell_posteriors,
    fit,
    homogeneity_ppc,
    pooled_counts,
)
from betapair.irt import (
    ItemDrift,
    ItemKey,
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
    ReportOptions,
    RetrospectiveWindow,
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
from betapair.series import ScoreBand, ScorePoint, retrospective, score_band
from betapair.tracker import ERROR, FAILED, RETIRED, HealthTracker, PairRecord

try:
    __version__ = version("betapair")
except PackageNotFoundError:  # pragma: no cover - a source checkout never installed
    __version__ = "0.0.0"

#: The top-level surface. Every name is declared in the ``__all__`` of the
#: module that owns it; ``tests/test_surface.py`` asserts this list matches the
#: submodule declarations, so it cannot silently drift.
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
    "GroupSummary",
    "Guardrail",
    "HealthReport",
    "HealthTracker",
    "HierarchyFit",
    "HierarchySummary",
    "Homogeneity",
    "IrTSummary",
    "ItemDrift",
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
    "ReportOptions",
    "RetrospectiveWindow",
    "ScoreBand",
    "ScorePoint",
    "Status",
    "SwitchEvidence",
    "UnpairedCell",
    "Weighting",
    "__version__",
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
    "item_fingerprint",
    "item_key",
    "mcnemar",
    "parse_category_weights",
    "pooled_counts",
    "regression_model",
    "resolve_calibration",
    "retrospective",
    "score_band",
    "switch_evidence",
    "verdicts_from_model",
    "weighting",
]
