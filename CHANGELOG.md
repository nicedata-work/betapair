# Changelog

All notable changes are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-26

First release.

### Added

- **Say whether a change is real.** `mcnemar()` takes paired before/after
  results and reports how likely the difference is genuine rather than luck,
  plus a likely range for the true improvement rate.
- **Watch many groups at once.** `HealthTracker` collects results as they
  arrive and `HealthReport` scores each group, so one group getting much worse
  cannot hide behind a good average.
- **Borrow strength between groups.** Small groups are pooled with the rest
  through a hierarchical fit (`betapair.hierarchy`), so a group with few
  results still gets a sensible score.
- **Filter out false alarms.** A Bayesian gate on the per-group regressions
  keeps the false-alarm rate down to a level you choose, instead of flagging
  every group that wiggles.
- **Adjust for how hard the questions are.** `fit_rasch()` learns how hard each
  question is, and `equated_rates()` puts scores from different question sets
  on the same scale, so an easier test is not mistaken for progress.
- **Tell noise from real movement.** `score_band()` separates ordinary
  sampling noise from a genuine shift, and `series.retrospective()` gives a
  steadier read of a jumpy run of results.
- **Notice when a question itself changed.** Item fingerprints record what a
  question asked, and `HealthReport.drift` lists any that were edited or reused
  under the same id.
- **Failures count against the thing that failed.** A crash or timeout is
  recorded as `FAILED` and stays in the score, so breaking on the hard
  questions is not a way to look better.
- **Thresholds can learn themselves.** With `calibrate=True`, the decisions
  above are set from the tracker's own history instead of fixed defaults.
- **Save and resume.** Tracker snapshots and reports round-trip through JSON.
- Only `pydantic` is needed at runtime; the maths uses the standard library.

[Unreleased]: https://github.com/nicedata-work/betapair/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nicedata-work/betapair/releases/tag/v0.1.0
