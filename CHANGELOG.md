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
- `betapair.irt.item_fingerprint(outcome, text)`: a short, versioned content
  fingerprint (`fp1_<16 hex>`) of a question. The digest covers Unicode-NFC,
  case-folded, whitespace-collapsed text plus the outcome it targets, so
  formatting differences collapse while wording, punctuation and the outcome
  do not. A reworded question gets a new fingerprint, and a fingerprint change
  under an unchanged id is detectable drift.
- `betapair.irt.ItemDrift(outcome, item, expected, observed)`: one id observed
  with two different fingerprints. `HealthTracker.observe(...,
  fingerprint=...)` (and `PairRecord.fingerprint`) records the first
  fingerprint seen per global `(outcome, item)`; later mismatches are
  deduplicated and exposed as `HealthTracker.drift`, with
  `HealthTracker.fingerprint_of(outcome, item)` returning the binding. The
  first fingerprint survives `advance`, `merge`, snapshots and restarts, so an
  edited question cannot be silently re-baselined, and a merge disagreement
  becomes a drift row rather than rewriting identity.
- `HealthReport.drift`: the content-fingerprint mismatches, sorted and
  de-duplicated. Reported, not gated — the score, delta and verdict are
  unchanged — because identity is the caller's contract and silently re-keying
  the row would hide the mixed evidence. `REPORT_VERSION` moved to 12; the
  snapshot keys `"fingerprints"`/`"drift"` are additive (`SNAPSHOT_VERSION`
  stays 3).
- `betapair.irt.RaschDiagnostics.shared_items` and `prior_only`: the global
  count of item keys observed by more than one view, and the explicit caveat
  for a multi-view fit that shares no item. Unlike `weakest_link` (the
  pair-wise minimum), the global count is not hidden by one disjoint view
  pair. `prior_only` is false for a single-view fit, where there is no
  cross-view comparison to make. `IrTSummary` (and so `HealthReport.irt`)
  carries both (summed and any-of across the fitted outcomes), so an adjusted
  report states when its comparison rests on the prior rather than on data.
- `RetrospectiveWindow` and `series.retrospective()`: a stabilised read of
  the most recent `window` score epochs. The level is the pair-mass pooled
  score with the headline's Beta interval, `delta_pp` compares the pooled first
  and second halves, `trend_pp` is the pair-weighted slope, and `flips` counts
  sign changes between consecutive movements. The verdict is level-only: a
  credible movement clears the ROPE, `stable` means the central movement is
  inside it, and `insufficient` means the window cannot place the movement. It
  never overrides a credible per-cell regression.
  `HealthTracker.retrospective(window=...)` runs it off `timeline`, and
  `HealthTracker.report(retrospective_window=...)` attaches it as
  `HealthReport.retrospective` (opt-in; the headline status is unchanged).

- `ScorePoint.score_held`/`paired`/`retired`/`failed`/`bank`: timeline
  points can now record the bank they were scored on. `bank` is a digest of
  the active `(outcome, item)` keys, the counts come from the tracker/report,
  and all keys are additive in `ScorePoint.snapshot()` (snapshot v3 still
  loads). `ScoreBand.bank_stable` is `True` when every point names the same
  bank, `False` when they differ (or only some carry one), and `None` for
  legacy points. When it is `False`, `score_band` estimates between-epoch
  dispersion within each bank only, so retiring hard items (60% on 100 pairs →
  75% on 80) reads `between_sd = 0.0` instead of booking attrition as
  volatility; a genuine move on a stable bank is unchanged.
- `fit_rasch` accepts **all outcomes at once**: item difficulties stay
  outcome-specific, but each view has one shared ability informed by every
  outcome it ran. The multi-facet fit is the same alternating Newton, so no
  optimiser enters; `RaschFit.probability` takes an explicit `outcome` for a
  multi-outcome fit, and `equated_rates` groups the reference bank by outcome
  and equates each on its own items with the shared ability.
  `report(adjust_difficulty=True)` now fits once and draws the shared `theta`
  once, so a view's cells move together instead of pretending to be
  independent fits. `docs/validation.md` measures the borrowing: a thin
  outcome's ability RMSE falls from 0.881 to 0.472 (2 items) and 0.622 to
  0.430 (8 items) against a 20-item outcome.

### Added

- `betapair.__version__`, read from the installed distribution metadata
  (`importlib.metadata`), and `betapair.HierarchyFit` — the hierarchical-pooling
  fit renamed from the too-generic `Fit` and now the exported name. The old
  `betapair.hierarchy.Fit` still resolves as an alias.
- `HealthReport.load_json()`: parse persisted report JSON and reject a payload
  whose `version` is not `REPORT_VERSION`. `model_validate_json` alone cannot
  tell an old same-shaped payload from a current one; this is the load path
  callers should use.

### Changed

- Docstrings across the package moved from epytext-ish ``@param``/``@raise``/
  ``@note`` tags to Google-style ``Args``/``Returns``/``Raises``/``Notes``
  sections, so Sphinx/napoleon and IDEs can read them. `ruff` now runs the
  `pydocstyle` rules under ``convention = "google"`` (magic methods and
  ``__init__`` are exempted; tests and tools are out of scope).
- Every public module declares an ``__all__``. A new
  ``tests/test_surface.py`` asserts the top-level ``betapair.__all__`` matches
  those declarations, so the exported surface cannot drift silently.
- The three record sentinels (``ERROR``, ``RETIRED``, ``FAILED``) are now
  members of one ``betapair.snapshot.Marker`` enum instead of three look-alike
  singleton classes. The module-level constants are unchanged (and still
  serialise to the same strings).
- `.gitignore` no longer ignores all of ``docs/``; markdown there is visible,
  so the tracked, CI-diffed ``docs/validation.md`` is no longer trapped in an
  ignored tree.
- The persisted pydantic models now enforce their own contract:
  `extra="forbid"` rejects unknown keys, `frozen=True` makes a parsed report
  immutable, scores/rates/counts carry `ge`/`le` bounds, and a
  `model_validator` rejects half-present credible bounds (`score_low` without
  `score_high`, a lone `delta_pp_low`, …). A malformed or edited payload is
  now a read error instead of silently dropped or nonsensical data.
- Decision knobs are validated where they are set, not where they are used.
  A shared `betapair.models.validate_decision_config()` (also used by
  `ReportOptions`) now backs `HealthTracker.__init__`, so
  `HealthTracker(alpha=0.9)`, `draws=0`, `prob_interval=2.0` and `pi_points=0`
  fail at construction; per-call `report()` overrides are validated too.
- The tracker snapshot wire format moved to `betapair.snapshot`: the
  `ERROR`/`RETIRED`/`FAILED` sentinels, the record-state predicate and the
  `_parse_*` helpers. `HealthTracker` imports and re-exports the public names,
  so `betapair.tracker.ERROR` and friends are unchanged; `tracker.py` lost
  about 300 lines of serialisation code.
- `report.py` no longer owns the schema, the regression gate and the
  assembler at once. The pydantic models and the pure-data policy objects
  (`Calibration`, `RetrospectiveWindow`, `Weighting`, `CellCoverage`,
  `UnpairedCell`, `Status`, `REPORT_VERSION`, `DEFAULT_CATEGORY`) moved to
  `betapair.models`, and the spike-and-slab gate plus `delta_status` moved to
  `betapair.gate`. The move breaks the old `report <-> series` import cycle:
  `series` now imports `models`, and `report` imports `series` at module
  level (the two function-local imports and the `TYPE_CHECKING` block are
  gone). Existing `betapair.report` and `betapair` imports keep working —
  `report` re-exports the moved names.
- `build_report`'s assembly is decomposed into `_resolve_inputs`,
  `_score_levels`, `_fit_hierarchy` and `_assemble`, with a single
  `_insufficient_report` helper for the two cold-start returns. Configuration
  is bundled in a new exported, frozen `ReportOptions` dataclass, validated on
  construction and shared by `build_report` and `HealthTracker.report`;
  `build_report(cells, **kwargs)` remains as a compatibility facade over it.
- The stable logistic helpers (`sigmoid`, `log1p_exp`) moved to
  `betapair.beta`, so `betapair.irt` and `betapair.gate` share one
  implementation instead of maintaining byte-identical copies. `irt.sigmoid`
  and `irt.log1p_exp` still resolve (re-exported).
- `REPORT_VERSION` is now 14 (added `HealthReport.retrospective`, the opt-in
  windowed level/verdict). v13 added `irt.shared_items`/`irt.prior_only`, the
  cross-view linking diagnostic; v12 added `HealthReport.drift`, the
  content-fingerprint mismatch rows; v11 added `HealthReport.groups`, the
  per-incident roll-up; v10 added `CellContribution.group`, the
  shared-dependency label; v9 added `irt.reference`, the resolved reference
  bank, to the report JSON; v8 added `score_adjusted`/
  `difficulty_adjustment_pp`/`irt` and the per-cell equated rate; v7 added
  `score_unpaired`/`unpaired` and `unresolved_share`; v6 added
  `score_low`/`score_high`; v5 added `failed`. The tracker snapshot format is
  unchanged at v3: `"failed"`, `"pi_mean"`, `"unresolved_share"`, the IRT
  config (`"adjust_difficulty"`, `"sigma_theta"`, `"sigma_d"`,
  `"reference_items"`, `"reference_bank"`), `"groups"`, `"fingerprints"` and
  `"drift"` are additive encodings, and a payload without them still loads;
  the new composition keys are additive within each `timeline` point.
- `fit_rasch` no longer rejects mixed outcomes (F1): the one-ability-per-view
  model supersedes the per-outcome call. Single-outcome calls are unchanged;
  `fit_rasch([]).probability(...)` still raises and a multi-outcome
  `probability` now requires the outcome. The `REPORT_VERSION` stays 14 — the
  report shape did not change, only the values of an opted-in adjustment.

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
