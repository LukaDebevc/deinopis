**013 · Every free signal measures the same thing** · 2026-08-25 · `chess trace`
Asked whether the re-search rate from 011 — free, oracle-free, ~20x more
decisions than there are positions — can serve as an objective in place of
regret. Swept `c_rank` over the 012 range, 12-position bench corpus at 60k
nodes, and put it beside the regret and depth already measured on 4000 labelled
positions.

| `c_rank` | 0 | 125 | 250 *(def)* | 375 | 500 | 625 | 750 | 1000 | 1500 |
|---|---|---|---|---|---|---|---|---|---|
| regret (cp) | +1.30 | +1.15 | 0 | **+0.18** | +0.97 | +0.95 | +1.86 | +3.21 | +5.68 |
| depth | 8.00 | 8.86 | 9.81 | 10.56 | 11.06 | 11.41 | 11.72 | 12.23 | 12.69 |
| re-search rate | 0.55% | 1.54% | 2.23% | 2.24% | 3.15% | 3.24% | 3.73% | 4.49% | 5.07% |
| priced out | 0.1% | 27.8% | 47.9% | 58.1% | 62.8% | 65.8% | 67.8% | 70.1% | 72.6% |

**Result — negative, and structural.** Re-search rate is **monotone** in pruning
aggressiveness over the whole range, as are depth and priced-out rate. Regret is
**U-shaped**, with its minimum at 250-375. So re-search rate is not an objective:
minimising it selects `c_rank = 0`, no pricing at all, which is 007's
80-150 Elo hole. It is a third gauge on the same axis as depth, not a second
opinion about quality.
**The general statement, which is the useful part:** the price list's live
parameters trace out an essentially **one-dimensional** aggressiveness axis, and
every signal available without a teacher is monotone along it. Free signals tell
you *where on the axis you are*; only a signal that knows the right answer
(regret, or Elo) can say *where on the axis to stop*. The teacher cannot be
eliminated, only made cheaper.
**Consequence for the optimiser:** with two live parameters and a 13.5 s
evaluation, a grid dominates any stochastic search — it returns the whole
landscape for the same money, and 012's landscape is piecewise constant over
wide regions (`max_base`), where a gradient estimator reads exactly zero. SPSA
becomes the right tool only when the price function is learned and the dimension
is too high to grid, and then only with a **resampled corpus per step**: the
objective is bit-deterministic, so its error is corpus bias, not zero-mean
noise, and descent methods fit it (LEDGER 009: 0.35 cp gained on train, 0.10 on
held-out against ±1.65).

---
