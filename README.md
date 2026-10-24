# pydfba

One question, answered:

> We ran the same set of items before and after a change. Did anything really
> get better or worse, and which groups are responsible?

`pydfba` takes paired pass/fail results (each item measured twice), works out
whether the change is real, and points at the groups that moved. It is small
and dependency-light: the maths uses only Python's built-in `math` and
`random`, and `pydantic` is used just to save reports as JSON.

## What it is for

* **Comparing two versions** of a test, a model, a lesson or a prompt on the
  same items.
* **Watching many groups at once** — classes, models, datasets, customers.
* **Not letting a good average hide a bad group**: if one group gets much worse
  while the rest improve, `pydfba` says so.
* **Streaming**: add results as they arrive and ask for a fresh report at any
  time.

## Install

Not published on PyPI yet, so install from a git URL or a checkout:

```bash
uv add "pydfba @ git+https://github.com/<owner>/pydfba"
# or, inside a clone:
uv sync --extra dev
```

## Count the changes

If you already have the totals, `mcnemar()` turns them into an answer.

```python
from pydfba import mcnemar

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

```python
from pydfba import HealthTracker

tracker = HealthTracker()

for q in range(10):
    # group_a: many items got better
    tracker.observe("group_a", "quiz", f"q{q}", baseline=(q >= 6), candidate=(q >= 2))
    # group_b: some items got better
    tracker.observe("group_b", "quiz", f"q{q}", baseline=(q >= 6), candidate=(q >= 4))

report = tracker.report()

print("score      :", round(report.score, 1))
print("baseline   :", round(report.baseline_score, 1))
print("delta (pp) :", round(report.delta_pp, 2))
print("status     :", report.status)
print("guardrail  :", report.guardrail.status)

for c in report.contributions:
    print(f"  {c.stratum:8s} {c.verdict:10s} change={c.delta_pp_contribution:+5.1f}pp")
```

```
score      : 70.0
baseline   : 40.0
delta (pp) : 30.0
status     : improving
guardrail  : ok
  group_a  improving  change=+20.0pp
  group_b  stable     change=+10.0pp
```

* `score` is the pass rate **after** the change, `baseline` the pass rate
  **before**, and `delta_pp` the change in percentage points.
* `status` is the overall verdict: `improving`, `stable`, `regressing`,
  `mixed`, or `insufficient` (not enough data yet).
* Each contribution is one group. The `change` values add up to `delta_pp`, so
  you can see exactly who moved the number.
* `group_b` is `stable` even though it improved: the improvement is real but
  not yet strong enough on its own to call.

## Catching a group that got worse

A good average can hide a broken group. `pydfba` checks for that and names the
group.

```python
from pydfba import HealthTracker

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

The average did not move at all, but `group_b` clearly got worse, so the report
is `regressing` and `flagged` names the culprit. This is the most important
safety feature.

## Everyday extras

**Add many results at once** with `PairRecord`:

```python
from pydfba import HealthTracker, PairRecord

tracker = HealthTracker()
tracker.observe_many([
    PairRecord("group_a", "quiz", "q1", False, True),   # fail -> pass
    PairRecord("group_a", "quiz", "q2", True, True),    # stayed passing
])
```

**Items with no result yet.** Use `None`; they are counted as `pending` and
left out, so a half-finished run never looks like a regression:

```python
tracker.observe("group_a", "quiz", "q3", baseline=False, candidate=None)
print(tracker.pending)   # 1
```

**A failed evaluation.** Use `ERROR` (different from "not run yet"):

```python
from pydfba import ERROR

tracker.observe("group_a", "quiz", "q4", baseline=False, candidate=ERROR)
print(tracker.errored)   # 1
```

**Some groups matter more.** Tag them and weight the tags:

```python
tracker = HealthTracker(
    categories={"group_a": "tier-1"},
    category_weights={"tier-1": 3.0, "tier-2": 1.0},
)
```

**Start a new round.** `advance()` moves the "after" results into "before" and
clears the slate:

```python
tracker.advance(candidate_epoch="round_2")
```

**Save and load.** State is plain JSON, including the random seed, so a
restored tracker gives the same numbers for the same data:

```python
text = tracker.to_json()
tracker = HealthTracker.from_json(text)
```

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

## What it cannot do

* It only understands pass/fail, not "how wrong" an answer was.
* Small samples can only catch big changes.
* Groups are treated as independent; two outcomes that are really the same
  thing will look slightly more certain than they are.
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

The `mcnemar()` function is a Python port of the R package `DFBA`'s
`dfba_mcnemar()`. The decisions behind the report, and the simulations that
check them, are documented in `docs/validation.md`.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
