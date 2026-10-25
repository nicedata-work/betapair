# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `FAILED`: a *view-caused* evaluation failure (a timeout, a crash) is now a
  result rather than an absence. It is scored as a failure and stays in the
  level and the paired test, so a view cannot raise its own score by breaking
  on the questions it was failing. `ERROR` keeps its meaning — a *harness*
  failure is genuinely unresolved, stays out of the score and gates coverage.
- `HealthTracker.failed`, `CellCoverage.failed`, `Coverage.failed` and
  `HealthReport.failed` expose the view-caused failures alongside the existing
  `pending`/`errored`/`retired` accounting. `failed` is a subset of `paired`.

### Changed

- `REPORT_VERSION` is now 5 (added `failed` to the report JSON). The tracker
  snapshot format is unchanged at v3: `"failed"` is an additive side encoding,
  and a payload without it still loads.

## [0.1.0] - 2026-10-25

Initial release.

### Added

- `mcnemar()` and `switch_evidence()`: Bayesian McNemar test on the discordant
  counts, with posterior mean/median, equal-tail credible interval, point and
  interval Bayes factors, and `P(phi > .5)`.
- `betapair.beta`: dependency-free beta / beta-binomial numerics on the
  standard library (`math`, `random`) — no numpy or scipy.
- `betapair.hierarchy`: hierarchical pooling across strata with a posterior
  predictive homogeneity guardrail.
- `HealthTracker`: streaming and batch accumulation with `observe()`,
  `observe_many()`, `advance()`, `pending`/`errored` accounting, weighted
  categories, and JSON snapshot round-tripping.
- `HealthReport`: overall and per-cell scores, deltas, verdicts, coverage and
  guardrail flags, plus a Bayesian spike-and-slab regression gate for
  FDR-controlled regression detection.
- `score_band()` / `HealthTracker.score_band()`: separate per-epoch sampling
  noise from real movement in a score series.
- Typed package (`py.typed`) with strict mypy, ruff lint/format, and CI across
  Python 3.10–3.13.

[Unreleased]: https://github.com/nicedata-work/betapair/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nicedata-work/betapair/releases/tag/v0.1.0
