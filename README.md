![betapair](docs/betapair.png)

# betapair

One question, answered:

> We ran the same set of items before and after a change. Did anything really
> get better or worse, and which groups are responsible?

`betapair` takes paired pass/fail results (each item measured twice), works out
whether the change is real, and points at the groups that moved. It is small
and dependency-light: the maths uses only Python's built-in `math` and
`random`, and `pydantic` is used just to save reports as JSON.

## What it is for

* **Comparing two versions** of a test, a model, a lesson or a prompt on the
  same items.
* **Watching many groups at once** — classes, models, datasets, customers.
* **Not letting a good average hide a bad group**: if one group gets much worse
  while the rest improve, `betapair` says so.
* **Streaming**: add results as they arrive and ask for a fresh report at any
  time.

## Install

```bash
pip install betapair
# or with uv:
uv add betapair
```

From a checkout:

```bash
uv sync --extra dev
```

## Count the changes

If you already have the totals, `mcnemar()` turns them into an answer.

```python
from betapair import mcnemar

# 12 items went from fail to pass; 3 went from pass to fail.
result = mcnemar(n_improved=12, n_regressed=3)

print("chance it really improved:", round(result.p_improved, 4))
print("likely range             :", round(result.eti_low, 3), "to", round(result.eti_high, 3))
```

```
chance it really improved: 0.9894
likely range             : 0.544 to 0.927
```

`p_improved` is the chance the change is a genuine improvement rather than
luck. The range is where the true improvement rate most likely sits. Items that
did not change are ignored — only the ones that moved tell you about the
change.

## Many groups at once

Usually you have several groups and several items. `HealthTracker` collects the
results and `report()` sums them up.

<details>
<summary>Show example</summary>

```python
from betapair import HealthTracker

tracker = HealthTracker()

for q in range(10):
    # group_a: many items got better
    tracker.observe("group_a", "quiz", f"q{q}", baseline=(q >= 6), candidate=(q >= 2))
    # group_b: some items got better
    tracker.observe("group_b", "quiz", f"q{q}", baseline=(q >= 6), candidate=(q >= 4))

report = tracker.report()

print("score      :", round(report.score, 1))
print("range      :", round(report.score_low, 1), "to", round(report.score_high, 1))
print("baseline   :", round(report.baseline_score, 1))
print("delta (pp) :", round(report.delta_pp, 2))
print("status     :", report.status)
print("guardrail  :", report.guardrail.status)

for c in report.contributions:
    print(f"  {c.stratum:8s} {c.verdict:10s} change={c.delta_pp_contribution:+5.1f}pp")
```

```
score      : 70.0
range      : 47.8 to 85.4
baseline   : 40.0
delta (pp) : 30.0
status     : improving
guardrail  : ok
  group_a  improving  change=+20.0pp
  group_b  stable     change=+10.0pp
```

</details>

* `score` is the pass rate **after** the change, `baseline` the pass rate
  **before**, and `delta_pp` the change in percentage points.
* `score_low`/`score_high` are the credible bounds on `score`: the same
  `Beta(1 + k, 1 + n - k)` posterior whose mode is the headline number, over
  the **weighted effective** pairs, so the range shrinks as more items run and
  describes exactly the micro-average it belongs to.
* `status` is the overall verdict: `improving`, `stable`, `regressing`,
  `mixed`, or `insufficient` (not enough data yet).
* Each contribution is one group. The `change` values add up to `delta_pp`, so
  you can see exactly who moved the number.
* `group_b` is `stable` even though it improved: the improvement is real but
  not yet strong enough on its own to call.

## The first run, and new views

A paired score needs a before **and** an after. On the very first run there is
no before, so `score` is `None` and the status is `insufficient` — but the
candidate results already measured are not thrown away. `score_unpaired`
reports them as an absolute level, so a dashboard has something to show while
the baseline is still missing:

<details>
<summary>Show example</summary>

```python
for q in range(10):
    tracker.observe("group_a", "quiz", f"q{q}", baseline=None, candidate=q < 7)

report = tracker.report()
print(report.status)  # insufficient - no comparison yet
print(report.score)  # None - the paired score needs a baseline
print(report.score_unpaired)  # 70.0 - the candidate epoch's own level
print(report.unpaired)  # 10 rows behind it
```

</details>

The same applies to a view introduced mid-stream: it contributes
`score_unpaired` until its first baseline exists, and then joins the paired
score. `score_unpaired` is an absolute level only — it never enters
`delta_pp`, the verdict or the `regression gate`, so it can never be mistaken
for a comparison.

## Catching a group that got worse

A good average can hide a broken group. `betapair` checks for that and names the
group.

<details>
<summary>Show example</summary>

```python
from betapair import HealthTracker

tracker = HealthTracker()
for q in range(10):
    tracker.observe("group_a", "quiz", f"q{q}", baseline=False, candidate=True)  # better
    tracker.observe("group_b", "quiz", f"q{q}", baseline=True, candidate=False)  # worse

report = tracker.report()

print("score     :", round(report.score, 1))
print("delta (pp):", round(report.delta_pp, 2))
print("status    :", report.status)
print("guardrail :", report.guardrail.status, report.guardrail.flagged)
```

```
score     : 50.0
delta (pp): 0.0
status    : regressing
guardrail : heterogeneous ['group_b/quiz']
```

</details>

The average did not move at all, but `group_b` clearly got worse, so the report
is `regressing` and `flagged` names the culprit. This is the most important
safety feature.

## Watching the score move over time

A single report says what the score is now. Across epochs, `advance()` keeps
one score per finished epoch, and `score_band()` turns that history into a
range: how much the score moves once sampling noise is taken out.

<details>
<summary>Show example</summary>

```python
from betapair import HealthTracker

tracker = HealthTracker()
for rate in (0.70, 0.71, 0.69, 0.70):
    for q in range(200):
        tracker.observe("group_a", "quiz", f"q{q}", baseline=False, candidate=q < rate * 200)
    tracker.advance()

band = tracker.score_band(window=3)
print("avg score   :", round(band.center, 1))
print("likely range:", round(band.low, 1), "to", round(band.high, 1))
print("noise       :", round(band.within_sd, 2), "pp")
print("movement    :", round(band.between_sd, 2), "pp")
```

```
avg score   : 70.0
likely range: 63.7 to 76.3
noise       : 3.22 pp
movement    : 0.0 pp
```

</details>

`within_sd` is the per-epoch sampling noise — it shrinks as more items run —
and `between_sd` is real epoch-to-epoch movement. When the whole observed
spread is explained by noise, `between_sd` is 0 and the band is just the
noise; a volatile metric widens it. `window=3` keeps the band responsive to a
recent regime change. A series can also be passed to `score_band()` directly,
as percentages or as reports:

<details>
<summary>Show example</summary>

```python
from betapair import score_band

band = score_band([70.0, 72.0, 69.5, 85.0])
print(round(band.low, 1), "to", round(band.high, 1))  # 59.8 to 88.5
```

</details>

You can record a point without rolling the epoch with `tracker.record_score()`,
and the timeline survives `to_json()`/`from_json()`.

**See which thresholds the report used.** Every decision threshold has a
hand-set default, and the report publishes the resolved set. Passing the
tracker's own `timeline` as calibration history lets the report size those
thresholds from the measured series rather than a fixed constant:

<details>
<summary>See which thresholds the report used</summary>

```python
report = tracker.report(calibrate=True)
print(report.calibration.rope_pp)  # the resolved ROPE, in pp
print(report.calibration.derived)  # thresholds taken from the series
```

</details>

Three thresholds are sized this way. The **ROPE** becomes the metric's own
`between_sd` once three epochs are on record (a move smaller than the metric's
normal epoch-to-epoch movement is not a change), so volatile views stop reading
as `improving` on ordinary noise. The **regression prior** is re-centred on the
base rate of true regressions the previous epoch measured — recorded on each
timeline point as `pi_mean` — and clipped away from 0 and 1 so it cannot
collapse. The **coverage gate** becomes the epochs' observed unresolved share
plus a margin (recorded as `unresolved_share`), so a pipeline that normally
resolves almost everything is not waved through at 25% pending — and a cold
start, where nearly everything is unresolved, keeps the default until enough
epochs exist to size it. Calibration is opt-in and additive: with no history —
or with `calibrate` left off — every number is unchanged, and a threshold is
only taken from the series when there is enough history to measure it.

The seeded operating characteristics behind those three thresholds — verdict
stability as volatility changes, the prior's power against the measured base
rate, and the gate's behaviour on a healthy vs stalled pipeline — are
generated into [`docs/validation.md`](docs/validation.md) by
`uv run python tools/calibration.py --write`.

## Everyday extras

**Add many results at once** with `PairRecord`:

<details>
<summary>Add many results at once</summary>

```python
from betapair import HealthTracker, PairRecord

tracker = HealthTracker()
tracker.observe_many(
    [
        PairRecord("group_a", "quiz", "q1", False, True),  # fail -> pass
        PairRecord("group_a", "quiz", "q2", True, True),  # stayed passing
    ]
)
```

</details>

**Items with no result yet.** Use `None`; they are counted as `pending` and
left out, so a half-finished run never looks like a regression:

<details>
<summary>Items with no result yet</summary>

```python
tracker.observe("group_a", "quiz", "q3", baseline=False, candidate=None)
print(tracker.pending)  # 1
```

</details>

**A failed evaluation.** Attribute it to whoever failed. `ERROR` means the
*harness* could not evaluate the item (a worker died, an API returned 5xx): it
is genuinely unresolved, so it is left out of the score and gates coverage.

<details>
<summary>A failed evaluation</summary>

```python
from betapair import ERROR

tracker.observe("group_a", "quiz", "q4", baseline=False, candidate=ERROR)
print(tracker.errored)  # 1
```

</details>

`FAILED` means the *view* could not answer the item (its SQL timed out, its
query OOM'd the warehouse, its output crashed the judge). A view-caused failure
is a result, so it is scored as a failure and stays in the score and the paired
test — a view cannot raise its own level by breaking on the questions it was
failing:

<details>
<summary>Show example</summary>

```python
from betapair import FAILED

tracker.observe("group_a", "quiz", "q6", baseline=True, candidate=FAILED)
print(tracker.failed)  # 1, counted against the score rather than dropped from it
```

</details>

**A question leaves the bank.** Retire it, so the score is not quietly
redefined by the ones that remain:

<details>
<summary>A question leaves the bank</summary>

```python
tracker.retire("group_a", "quiz", "q5")
print(tracker.retired)  # 1

report = tracker.report()
print(report.score)  # the surviving questions only
print(report.score_held)  # ...with the retired ones held at their last result
print(report.composition_effect_pp)  # how much of the score is the bank shrinking
```

</details>

A retired question is not `pending` (nothing is outstanding) and not `ERROR`
(the harness did not fail), so it does not gate coverage. `composition_effect_pp`
is the part of the score explained by questions *leaving* the bank rather than
by the views changing, so deleting the questions you were failing shows up as a
number instead of an improvement. (Erroring out of them with `FAILED` needs no
such bookkeeping: the failure is already in the score.)

**Name questions consistently.** A row is identified by
`(group, outcome, item)`, but an *item* is identified globally by
`(outcome, item)`. Two views that use the same `item` string are recorded as
seeing the **same** question — that overlap is what lets a difficulty model
compare their levels later — so use one stable name per question across views
and epochs. A renamed question is a different item:

<details>
<summary>Name questions consistently</summary>

```python
from betapair import item_key

item_key("quiz", "q5")  # ('quiz', 'q5')
```

</details>

The resolved rows are available under that key with `tracker.item_responses()`;
pending, `ERROR` and retired rows are omitted.

## Comparing views on different question banks

`score` is a pass rate on whatever questions each view happened to run. If
`group_b` ran an easier bank than `group_a`, it looks healthier even when the
two views are equally good. `adjust_difficulty=True` fits a latent difficulty
model (Rasch/IRT) to the candidate responses and reports the level each view
would score on a common **reference bank**, so the two become comparable:

<details>
<summary>Show example</summary>

```python
from betapair import HealthTracker

tracker = HealthTracker(adjust_difficulty=True)
for q in range(4):  # shared anchors: both views pass
    tracker.observe("group_a", "quiz", f"anchor{q}", baseline=False, candidate=True)
    tracker.observe("group_b", "quiz", f"anchor{q}", baseline=False, candidate=True)
for q in range(8):  # group_a ran a hard bank and is still failing it
    tracker.observe("group_a", "quiz", f"hard{q}", baseline=False, candidate=False)
for q in range(8):  # group_b ran an easy bank and passes all of it
    tracker.observe("group_b", "quiz", f"easy{q}", baseline=False, candidate=True)

report = tracker.report()
print("raw       :", round(report.score, 1))
print("adjusted  :", round(report.score_adjusted, 1))
print("shift (pp):", round(report.difficulty_adjustment_pp, 1))
```

```
raw       : 66.7
adjusted  : 63.3
shift (pp): -3.3
```

</details>

* `score_adjusted` is the same weighted micro-average as `score`, but over
each view's equated rate, and `score_adjusted_low`/`score_adjusted_high` are
its interval. The raw `score` is never replaced: both numbers are always
present, and `difficulty_adjustment_pp` is the difference between them.
* The paired `delta_pp` is **not** adjusted — both epochs run the same items,
so difficulty already cancels in the discordant pairs. Only the level needs
equating.
* It is opt-in. With `adjust_difficulty` off (the default) every number is
bit-identical to a report built without it.

**Pin the reference bank.** By default the reference bank is every item
observed, which shrinks as questions are retired and shifts what the adjusted
score means. A named, versioned bank keeps the number comparable across epochs:

<details>
<summary>Pin the reference bank</summary>

```python
from betapair import HealthTracker, ReferenceBank

bank = ReferenceBank.from_responses(tracker.item_responses(), name="standard", version="2026-07")
tracker = HealthTracker(adjust_difficulty=True, reference_bank=bank)
```

</details>

`report.irt.reference` publishes which bank was used — the declared `name` and
`version`, a content `digest` of the items actually used, and the declared vs
`available` counts — so two epochs are only comparable when their digests
match. The bank is saved with the tracker's snapshot, so a resumed tracker
scores the same way.

**Read the linking caveat.** Views are connected only through items they both
ran. `report.irt` reports `connected` and `weakest_link` (the fewest shared
items over any view pair), plus `prior_dominated` for views whose responses
were perfectly separated. With no shared items the fit still runs — the priors
bridge the banks — but that is an assumption, not a measurement.

The seeded evidence behind the adjusted interval's coverage, the cost of weak
linking, and how much bank bias equating removes is generated into
[`docs/validation.md`](docs/validation.md) by `tools/calibration.py`.

**Some groups matter more.** Tag them and weight the tags:

<details>
<summary>Some groups matter more</summary>

```python
tracker = HealthTracker(
    categories={"group_a": "tier-1"},
    category_weights={"tier-1": 3.0, "tier-2": 1.0},
)
```

</details>

**Which groups share a dependency.** A *group* here is the second label a view
can carry: the upstream thing it depends on — a source table, a domain, a
metric family. Twenty views on one table are not twenty independent tests, so
the regression gate reads the whole group as **one** test and reports an
incident instead of a flag storm:

<details>
<summary>Which groups share a dependency</summary>

```python
tracker = HealthTracker(groups={"sv_orders": "orders_table", "sv_lines": "orders_table"})
tracker.set_group("sv_lines", "orders_table")  # or bind it later
tracker.group_of("sv_orders")  # "orders_table"

report.groups
# [GroupSummary(group='orders_table', views=2, cells=8, p_incident=0.98, flagged=True, ...)]
```

</details>

The group changes nothing about the score or the delta, and no individual view
is hidden: the per-cell gate still runs, so a single regressing view is flagged
whether or not its group is. Each `GroupSummary` carries its additive share of
the layer's movement (`delta_pp_contribution`), and the per-cell
`contributions` (each tagged with `group`) expand an incident into the views
responsible — the same contribution decomposition used to investigate a
downstream metric. Groups are persisted in the snapshot exactly like
categories.

**Start a new round.** `advance()` moves the "after" results into "before" and
clears the slate:

<details>
<summary>Start a new round</summary>

```python
tracker.advance(candidate_epoch="round_2")
```

</details>

**Save and load.** State is plain JSON, including the random seed, so a
restored tracker gives the same numbers for the same data:

<details>
<summary>Save and load</summary>

```python
text = tracker.to_json()
tracker = HealthTracker.from_json(text)
```

</details>

## Common terms and what they mean

| Term | What it means |
| --- | --- |
| **Pair** | One item measured twice: before and after. |
| **Changed item** | A pair that went fail→pass or pass→fail. Items that stayed the same carry no signal and are ignored. |
| **Switch rate** | Of the changed items, the fraction that improved. Above 0.5 means more got better than worse. |
| **`p_improved`** | The chance the change is a real improvement. |
| **Credible interval** | A range of likely values for a number, given the data (shown as "likely range"). |
| **Bayes factor** | How much more likely the data are under "improved" than under "no change". |
| **`mu`** | The model's overall switch rate across all groups. |
| **`kappa`** | How much the groups move together; big means they agree. |
| **Guardrail** | A check that stops a good average from hiding a bad group. |
| **Tier / weight** | A label saying a group matters more, so it counts more in the score and the verdict. |
| **Group** | A label naming the upstream dependency a view shares with others. The regression gate reads the group as one test, so a shared break is one incident, not a flag storm; it never changes the score. |
| **Incident** | One flagged group: the shared dependency broke and its views regressed together. Reported in `report.groups`, with each view's own contribution still in `report.contributions`. |
| **Equated score** | The level a view would score on a common reference bank, so views that ran different questions can be compared. |
| **Reference bank** | The set of questions an adjusted score is measured on; pin it (named and versioned) so the number keeps its meaning. |

## What it cannot do

* It only understands pass/fail, not "how wrong" an answer was.
* Small samples can only catch big changes.
* Cells in the same group are pooled into one test only if you say so; two
  views on the same table that you leave ungrouped are still treated as
  independent and will look slightly more certain than they are.
* The difficulty-adjusted score compares views only through questions they both
  ran; banks with no overlap are bridged by an assumption (the priors), which
  `report.irt` flags as `connected=False` rather than measuring.
* If too much of the comparison has no result yet, the status is
  `insufficient` rather than a guess.

## Development

```bash
uv sync --extra dev     # set up the environment
uv run pytest           # run the tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy             # type-check
```

The seeded operating characteristics in [`docs/validation.md`](docs/validation.md)
are regenerated with `uv run python tools/calibration.py --write`. CI runs
`uv run python tools/calibration.py --check`, which fails with a diff if that
document is stale, so a statistical change and its evidence land together.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
