# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `betapair.irt.fit_rasch()` and `RaschFit`: the latent-difficulty model — a
  one-parameter logistic (Rasch) fit by penalised joint maximum likelihood.
  The two parameter blocks are conditionally independent, so the MAP is
  alternating 1-D Newton (no optimiser, no matrix solve); Gaussian priors fix
  the additive non-identifiability and keep perfectly separated views/items
  finite. `RaschFit` exposes `abilities`, `ability_sd`, `difficulties`,
  `difficulty_sd`, the resolved `hyper` priors and `probability(view, item)`.
  Ability uncertainty is the **Schur complement** of the joint Hessian
  (items integrated out) inverted by a small dense Cholesky with diagonal
  jitter; above 300 views, or if the Cholesky fails, the reciprocal diagonal
  is used and documented as understating the uncertainty. One outcome per fit;
  duplicate rows and mixed outcomes are rejected, and response order cannot
  change the result. Exported from `betapair` alongside `ItemResponse`.
- `betapair.irt.equated_rates()`: true-score equating — the pass rate a view
  would score on a common reference bank, `mean_i sigmoid(theta_v + d_i)`. The
  reference defaults to every item in the fit (the union of the observed
  banks) and can be pinned with global `ItemKey` values, so the adjusted
  score's meaning can be held stable as items are added or retired. Two views
  of equal ability on a hard and an easy bank converge on the reference bank
  while their raw rates do not.
- `betapair.irt.RaschDiagnostics`, carried on every `RaschFit`: how well the
  banks are actually linked, so an equated number is never read without its
  caveat. `components`/`connected` are the view-item graph's connectivity,
  `weakest_link` the fewest shared items over all view pairs, and
  `prior_dominated`/`separated_items` the perfectly separated views and items
  whose estimates are the prior's rather than the data's. Disjoint banks are
  still fitted — the priors bridge them — but that is surfaced as an
  assumption, never silently.
- `betapair.irt.log_sigmoid`, `sigmoid` and `log1p_exp`: stable logistic link
  helpers on the standard library (no numpy/scipy in the numeric core).
- `tools/irt_oracle.py`: an optional parity check of the fit against `girth`'s
  plain JML on a shared simulated bank (difficulty corr ≈ 0.985, ability corr
  ≈ 0.9999). Install it with the new `oracle` extra; without it the script
  skips and exits 0. CI runs it in its own job so the main dependency set
  stays numpy-free.
- `FAILED`: a *view-caused* evaluation failure (a timeout, a crash) is now a
  result rather than an absence. It is scored as a failure and stays in the
  level and the paired test, so a view cannot raise its own score by breaking
  on the questions it was failing. `ERROR` keeps its meaning — a *harness*
  failure is genuinely unresolved, stays out of the score and gates coverage.
- `HealthTracker.failed`, `CellCoverage.failed`, `Coverage.failed` and
  `HealthReport.failed` expose the view-caused failures alongside the existing
  `pending`/`errored`/`retired` accounting. `failed` is a subset of `paired`.
- `HealthReport.score_low` / `HealthReport.score_high`: the headline `score`
  now ships with its own credible interval — the equal-tail bounds of the
  `Beta(1 + k, 1 + n - k)` posterior over the weighted effective pairs, whose
  mode is exactly `score`. `ScorePoint.from_report` carries the bounds into a
  timeline, so `score_band` sizes sampling noise on the same micro-average.
- `betapair.irt` with the global item-identity contract: `ItemResponse`,
  `ItemKey` and `item_key(outcome, item)`. An IRT item is `(outcome, item)`,
  so the same question under two views is one item and the same question under
  two outcomes is two. `HealthTracker.item_responses()` exposes the resolved
  responses under that key (paired rows only; `FAILED` arrives as a failure).
- `Calibration` on `HealthReport`: the resolved decision thresholds (ROPE,
  regression prior, coverage gate) and their provenance. `build_report(...)`
  and `HealthTracker.report(...)` accept `history=`/`calibrate=`; the tracker
  offers its own `timeline` as history. Opt-in and additive — with no history,
  or with `calibrate` off, every existing number is bit-identical, and no
  threshold is derived until enough history is available.
- Self-calibrating ROPE: with `calibrate=True` and at least
  `ROPE_MIN_POINTS` scored epochs, `rope_pp = ROPE_BAND_FACTOR *
  score_band(history).between_sd`, so a volatile metric is no longer judged
  against a fixed 1pp dead zone. `build_report(rope_pp=None)` falls back to
  `1.0` (or the caller's value) whenever the band is too short or measures no
  between-epoch movement; the resolved value and its provenance are published
  on `HealthReport.calibration`.
- `ScorePoint.pi_mean`: a timeline point now records the epoch's estimated
  share of cells that truly regressed, captured from the last report the
  tracker built. It is an additive snapshot key (`SNAPSHOT_VERSION` stays 3)
  and the empirical-Bayes evidence for the next epoch's regression prior.
- Self-calibrating regression prior: with `calibrate=True` and history that
  carries a `pi_mean`, `pi_prior` is re-centred on the measured base rate while
  keeping the default prior strength, and clipped away from 0 and 1 so it
  cannot collapse. `Calibration.observed_pi_mean` and `derived=("pi_prior",)`
  publish the measurement and its use.
- Self-calibrating coverage gate: with `calibrate=True` and at least
  `COVERAGE_MIN_POINTS` epochs carrying an `unresolved_share`, the gate is set
  to the observed mean plus `COVERAGE_MARGIN`, clamped to
  `[COVERAGE_MIN, COVERAGE_MAX]`. Fewer epochs keep the default (a cold-start
  guard: early runs are mostly unresolved and must not set their own gate).
- `ScorePoint.unresolved_share`: a timeline point records the epoch's
  unresolved share, the evidence the coverage gate is calibrated from. Additive
  snapshot key (`SNAPSHOT_VERSION` stays 3).
- `docs/validation.md` now publishes the evidence for all three calibrated
  thresholds, regenerated from `tools/calibration.py`: **verdict stability as
  volatility changes** (a grid of true improvements × `between_sd`, default vs
  calibrated ROPE), the **self-calibrated regression prior** (power to flag a
  true regression as the measured base rate moves from 0.02 to 0.8), and the
  **self-calibrated coverage gate** (a 23%-unresolved epoch gated under a
  healthy history, passed under a stalled one). The null stays quiet at every
  prior — FDR control is the posterior rule's job, not the prior's.
- Cold-start level: `HealthReport.score_unpaired` / `score_unpaired_low` /
  `score_unpaired_high` and `unpaired`, plus `UnpairedCell` and
  `HealthTracker.unpaired_cells()`, describe the candidate-epoch pass rate for
  rows whose baseline has not run yet (the first run, or a view introduced
  mid-stream). It is an absolute level only — never the paired `score`, the
  delta, the verdict or the gate — so a brand-new view has a number while the
  comparison is still incomplete. `HealthReport.unresolved_share` publishes the
  epoch's coverage pressure.
- Difficulty-adjusted level: `build_report(..., responses=...,
  adjust_difficulty=True, reference_items=..., sigma_theta=..., sigma_d=...)`
  fits `betapair.irt.fit_rasch` one outcome at a time and equates every active
  cell onto a common reference bank (each outcome's observed union by default),
  publishing `HealthReport.score_adjusted` / `score_adjusted_low` /
  `score_adjusted_high` and `difficulty_adjustment_pp = score_adjusted - score`.
  The aggregation mirrors `score` exactly, the raw score is never overwritten,
  and the adjusted interval comes from seeded posterior draws of `theta` — so
  ability uncertainty sizes the level. `HealthReport.irt` carries `IrTSummary`
  (`sigma_theta`/`sigma_d`, `connected`, `components`, `weakest_link`,
  `prior_dominated`, `separated_items`, `observed_pairs`, `items`, `outcomes`),
  the per-outcome fits summarised conservatively. `CellContribution` gains
  `equated_pass_rate` and `difficulty_shift_pp`, whose pair-mass-weighted mean
  reproduces the layer adjustment. Off by default and inert when off.
- `RaschFit.sample_abilities(draws=, seed=)`: seeded Laplace-posterior draws of
  `theta`. The Schur complement is factorised once at fit time, so the draws
  carry the shared-item correlation between views; above `SCHUR_MAX_VIEWS`, or
  on a Cholesky failure, they fall back to independent normals at the reported
  `ability_sd` (documented as understating the uncertainty of a sum).
- `HealthTracker` IRT config: `adjust_difficulty`, `sigma_theta`, `sigma_d`,
  `reference_items` and `reference_bank` are validated, stored, used by
  `report()` by default, and round-tripped through `snapshot()` as additive v3
  keys (`SNAPSHOT_VERSION` stays 3 — a payload without them still loads,
  defaulted to no adjustment). `report(adjust_difficulty=..., reference_items=...,
  reference_bank=...)` overrides per call.
- `betapair.irt.ReferenceBank`: a named, versioned set of global items that
  pins what `score_adjusted` is a pass rate *on*, so the adjusted level no
  longer drifts as the observed union shrinks with the catalogue. Items are
  canonicalised (sorted, validated); `label` is ``name@version`` and `digest`
  is a content hash (`bank_digest`) that identifies the item set regardless of
  its label. `ReferenceBank.from_responses()` freezes the items an epoch ran.
  `build_report(..., reference_bank=...)` (mutually exclusive with raw
  `reference_items`) equates on the intersection of the bank and the fitted
  items, so a temporarily absent anchor narrows the reference set instead of
  failing the report; `IrTSummary.reference` (`ReferenceBankSummary`) publishes
  the name, version, digest, declared size and used size. `HealthTracker`
  stores, validates and persists the bank like the rest of its IRT config.
- `tools/calibration.py` and `docs/validation.md` now cover the difficulty
  adjustment: interval coverage of `score_adjusted_low`/`score_adjusted_high`
  (at or above the 95% nominal level, slightly wider than the raw interval
  because it carries the shared-item component), cross-view comparability
  error under shared / partially linked / disjoint banks (RMSE 6.7 / 15.8 /
  20.6pp, bias ~0 — weak linking costs precision, not accuracy), and the bias
  of the raw vs equated score as a view's own bank shifts (+19.7pp vs +6.1pp
  at a three-logit offset). `README.md` gained a "Comparing views on different
  question banks" section covering the opt-in flag, `ReferenceBank` pinning
  and the linking caveats.
- `uv run python tools/calibration.py --check` and a `calibration` CI job: the
  seeded tables are regenerated and compared byte-for-byte with the committed
  `docs/validation.md` on every push and pull request, failing with a unified
  diff when they drift — so no statistical change can land without its evidence
  being regenerated.
- `group`: a second per-stratum label beside `category`, naming the shared
  upstream dependency a view belongs to (source table, domain, metric family).
  `HealthTracker(groups={...})`, `set_group()`/`group_of()`, and the snapshot's
  additive `"groups"` key mirror the category API, and the label is carried on
  `Cell`, `CellCoverage`, `UnpairedCell` and `CellContribution`, resolved
  per cell by `Weighting.group()`. Unlike a category it carries no weight and
  no gate: nothing in `score`, `delta_pp`, the guardrail or the regression
  model depends on it. It is the groundwork for reading cells that fail
  *together* as one incident (roadmap Direction E).
- Dependence-aware regression gate: when any cell declares a `group`, the
  spike-and-slab gate is fitted a second time with the **group as the unit** —
  its members' evidence pooled (discordant counts summed, weight their
  pair-mass weighted mean) — and that fit's FDR-controlled flag set is
  published as `RegressionModel.flagged_groups` / `group_incident`, with
  `bfdr_groups`. Counting twenty correlated views on one source table as one
  test is the denominator fix: a shared break is one incident, not a flag
  storm, and mild group-wide regressions no single cell can prove are caught
  by the pooled test (`docs/validation.md`, "Shared-dependency incidents").
  The per-cell fit is untouched and its flags are unioned with the flagged
  groups, so a group can never hide an individual regression.
- `HealthReport.groups` (`GroupSummary`): one row per declared group with the
  pooled counts, `score`/`baseline_score`/`delta_pp`, the group's additive
  `delta_pp_contribution` (so the layer's movement decomposes exactly into
  group rows plus ungrouped cells), `p_incident` and `flagged`. The per-cell
  `contributions` still carry `group`, so an incident expands into the views
  responsible — the same attribution used to investigate a downstream metric.
- `RegressionModel.muted`: the cells whose effective weight is 0. They are
  excluded from the score, from their own flag and from their group's
  incident, so callers can tell "not flagged on the evidence" apart from "not
  admissible at all". Empty when no weight is 0, keeping an unweighted report
  bit-identical.

### Changed

- `REPORT_VERSION` is now 11 (added `HealthReport.groups`, the per-incident
  roll-up). v10 added `CellContribution.group`, the shared-dependency label;
  v9 added `irt.reference`, the resolved reference bank, to the report JSON;
  v8 added `score_adjusted`/
  `difficulty_adjustment_pp`/`irt` and the per-cell equated rate; v7 added
  `score_unpaired`/`unpaired` and `unresolved_share`; v6 added
  `score_low`/`score_high`; v5 added `failed`. The tracker snapshot format is
  unchanged at v3: `"failed"`, `"pi_mean"`, `"unresolved_share"`, the IRT
  config (`"adjust_difficulty"`, `"sigma_theta"`, `"sigma_d"`,
  `"reference_items"`, `"reference_bank"`) and `"groups"` are additive
  encodings, and a payload without them still loads.

### Fixed

- A group incident no longer pools a muted (weight-0) member's discordance or
  implicates that member in its flag. A muted view could previously be dragged
  to `regressing` by its group's incident — and a group of muted cells could
  manufacture the incident that drove the layer's `status` — contradicting "a
  cell with weight 0 is muted everywhere". The group gate now admits only
  unmuted members (counts and pair-mass weight), a fully muted group is not
  tested (`p_incident = 0.0`, `flagged = False`), and verdict propagation
  skips muted cells. Pinned by the E4 cross-checks
  (`GroupInvariantTest`, `GroupEpochInvariantTest`,
  `GroupRetirementInvariantTest`) across tiers, epochs and retirement.

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
