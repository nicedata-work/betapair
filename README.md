# pydfba

A small Python package for one question:

> We ran the same set of items before and after a change. Did anything really
> get better or worse, and which groups are responsible?

`pydfba` gives you a clear answer using a well-known method called **McNemar's
test**, plus a Bayesian version of it that can compare many groups at once. It
needs no numpy, scipy or pandas — just plain Python.

---

## 1. What is this for?

Imagine you have a quiz with 20 questions. You try a new way of teaching, then
give the same quiz again. For each question you know if the student got it
right **before** and **after**. That is a "pair".

`pydfba` looks only at the questions that *changed*:

* ones that went from **fail to pass** (an improvement), and
* ones that went from **pass to fail** (a regression).

Questions that stayed the same tell us nothing about whether the change helped,
so they are ignored. This is what makes McNemar's test work even on a small
quiz.

### Good use cases

* **Comparing two versions** of a test, a model, or a lesson on the same items.
* **Watching many groups at once** (classes, models, datasets, customers).
* **Not letting a good average hide a bad group** — if one group gets much
  worse while the rest improve, `pydfba` flags it.
* **Streaming results** — you can add items one at a time and get a fresh
  report whenever you like.
* **Weighting groups** — you can say some groups matter more than others.

---

## 2. Install

```bash
uv add pydfba      # or: pip install pydfba
```

The only dependency is `pydantic` (used to save reports as JSON). The math
itself uses only Python's built-in `math` and `random`.

---

## 3. Toy example: one comparison

The simplest function takes the two change counts and tells you what they mean.

```python
from pydfba import mcnemar

# 12 questions went from fail to pass; 3 went from pass to fail.
result = mcnemar(n_improved=12, n_regressed=3)

print("chance the new version is better:", round(result.p_improved, 4))
print("95% range for the switch rate   :", round(result.eti_low, 3), "to", round(result.eti_high, 3))
```

Output:

```
chance the new version is better: 0.9894
95% range for the switch rate   : 0.544 to 0.927
```

**How to read it**

* **Switch rate** means: of the questions that changed, what fraction changed
  for the better? Here 12 out of 15, which is 0.8.
* `p_improved = 0.9894` means there is a 98.9% chance the new version is
  really better (not just lucky).
* The **95% range** `0.544 to 0.927` is a range of likely values for the true
  switch rate. Because it stays above 0.5, it points to real improvement.

---

## 4. Toy example: many groups at once

Usually you have several groups and several questions. `HealthTracker` collects
the results and `report()` sums them up.

```python
from pydfba import HealthTracker

tracker = HealthTracker(baseline_epoch="before", candidate_epoch="after")

for q in range(10):
    # group_a: many questions got better
    tracker.observe("group_a", "quiz", f"q{q}", baseline=(q >= 6), candidate=(q >= 2))
    # group_b: some questions got better
    tracker.observe("group_b", "quiz", f"q{q}", baseline=(q >= 6), candidate=(q >= 4))

report = tracker.report()

print("score      :", round(report.score, 1))
print("baseline   :", round(report.baseline_score, 1))
print("delta (pp) :", round(report.delta_pp, 2))
print("status     :", report.status)
print("guardrail  :", report.guardrail.status)

for c in report.contributions:
    print(f"  {c.stratum:8s} {c.verdict:10s} delta={c.delta_pp:6.1f} share={c.share:+.2f}")
```

Output:

```
score      : 70.0
baseline   : 40.0
delta (pp) : 30.0
status     : improving
guardrail  : ok
  group_a  improving  delta=  40.0 share=+0.67
  group_b  stable     delta=  20.0 share=+0.33
```

**How to read it**

* `score` is the pass rate **after** the change (70 out of 100).
* `baseline_score` is the pass rate **before** (40 out of 100).
* `delta_pp` is the change in **percentage points**. 70 − 40 = 30, so the two
  numbers always match exactly.
* `status` is the overall verdict: `improving`, `stable`, `regressing`, or
  `insufficient` (not enough data yet).
* Each line under `contributions` is one group. `share` says how much of the
  total change came from that group — here `0.67` and `0.33`, which add up to
  1.00.
* `group_b` is marked `stable` even though it improved, because 20 points from
  only a few changed questions is not yet strong enough evidence on its own.

---

## 5. Toy example: the guardrail

A simple average can be tricked. If nine groups improve and one collapses, the
average may still look fine. `pydfba` checks for this and warns you.

```python
from pydfba import HealthTracker

tracker = HealthTracker()
for q in range(10):
    tracker.observe("group_a", "quiz", f"q{q}", baseline=(q >= 6), candidate=(q >= 2))  # better
    tracker.observe("group_b", "quiz", f"q{q}", baseline=(q >= 4), candidate=(q >= 9))  # worse

report = tracker.report()

print("score     :", round(report.score, 1))
print("delta (pp):", round(report.delta_pp, 2))
print("status    :", report.status)
print("guardrail :", report.guardrail.status, report.guardrail.flagged)
```

Output:

```
score     : 45.0
delta (pp): -5.0
status    : regressing
guardrail : heterogeneous ['group_b/quiz']
```

**How to read it**

* `group_a` improved, but `group_b` got much worse, so the whole thing is
  `regressing`.
* `guardrail` says `heterogeneous`, which means "the groups are not moving
  together".
* `flagged` names the group that caused it: `group_b/quiz`.

This is the most important safety feature: a good average cannot hide a broken
group.

---

## 6. Other useful things

**Add many results at once.** Use `PairRecord` and `observe_many`:

```python
from pydfba import HealthTracker, PairRecord

tracker = HealthTracker()
tracker.observe_many([
    PairRecord("group_a", "quiz", "q1", False, True),   # fail -> pass
    PairRecord("group_a", "quiz", "q2", True, True),    # stayed passing
])
```

**Skip items that were not decided.** Use `None` for a side that has no result
yet. Those items are counted as `pending` and left out of the score, so a
half-finished run never looks like a regression:

```python
tracker.observe("group_a", "quiz", "q3", baseline=False, candidate=None)
print(tracker.pending)   # 1
```

**Make some groups count more.** Give a group a category and give categories a
weight. Weights change the overall score but never the statistical tests:

```python
tracker = HealthTracker(
    categories={"group_a": "important"},
    category_weights={"important": 3.0, "normal": 1.0},
)
```

**Start a new round.** `advance()` moves the "after" results into "before" and
clears the slate:

```python
tracker.advance(candidate_epoch="round_2")
```

**Save and load.** State is plain, sorted JSON:

```python
text = tracker.to_json()
tracker = HealthTracker.from_json(text)
```

**Want the raw numbers?** The building blocks are exported too:

```python
from pydfba import fit, homogeneity_ppc, cell_verdicts

# fit() gives mu (overall switch rate) and kappa (how much groups agree)
# homogeneity_ppc() gives the guardrail p-value on its own
# cell_verdicts() gives each group's own verdict
```

---

## 7. Words you need to know

| Word | Plain meaning |
| --- | --- |
| **Pair** | One item measured twice: before and after. |
| **Discordant** | A pair that changed (fail→pass or pass→fail). |
| **Switch rate** | Of the changed pairs, the fraction that got better. |
| **`mu`** | The overall (pooled) switch rate across all groups. Above 0.5 means more improved than regressed. |
| **`kappa`** | How much the groups move together. Big means they agree. |
| **`p_improved`** | The chance that the result is a real improvement. |
| **Credible interval** | A range of likely values for a number (like a confidence interval). |
| **Guardrail** | A check that stops a good average from hiding a bad group. |

---

## 8. What it cannot do

* If **nothing changed**, it reports `insufficient` rather than "healthy".
  That is on purpose: there is no evidence about direction.
* It treats each group × outcome as independent. If two outcomes are really the
  same thing, the ranges will look a little too confident.
* Small quizzes can only catch big changes.
* It only knows pass/fail, not "how wrong" an answer was.
* Ranges and the guardrail p-value use a fixed random seed, so they are the
  same every time — but they are estimates, not exact values.

---

## 9. Development

```bash
uv sync                # set up the environment
uv run pytest          # run the tests (97 of them)
uv run ruff check .    # lint
uv run ruff format .   # format
```

## 10. Where it came from

`pydfba` is a cleaned-up, standalone copy of the statistics from the `jello`
health metric. The `mcnemar()` function is a Python port of the R package
`DFBA`'s `dfba_mcnemar()`, and it matches that package's published numbers to
the last digit. Two bugs found during review are fixed here:

1. the math counts only the **changed** pairs, not all pairs;
2. the regression warning is corrected for testing many groups at once, so a
   wide comparison does not raise a false alarm.
