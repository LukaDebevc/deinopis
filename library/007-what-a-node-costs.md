# What a node costs

Measured 2026-08-25 on the 12-core box, idle, single-threaded, `bench` corpus
(12 FENs), min-of-5 per cell, cleared TT. Raw data and the sweep harness are in
the session scratchpad; the patch is three lines (a `QNODES` counter at the
`quiescence` entry) and was never committed.

## The question

If a price feature calls qsearch (004), a "node" stops being a unit of work and
every node-denominated result silently changes meaning. Is wall clock better?
Is a component cost model better than either?

## 1. Wall clock is biased, not merely noisy

`bench 11`, node count bit-identical across all runs (1318702 every time).

| condition | mean nps | vs idle | CV |
|---|---|---|---|
| idle | 4,350,000 | — | **1.9%** |
| 4 competing spinners | 4,295,000 | −1.3% | 1.1% |
| 8 competing spinners | 3,508,000 | **−19.4%** | **14.0%** |
| 11 competing spinners | 2,699,000 | −38.0% | 7.3% |

12 physical cores, so 4 spinners is free and 8 is not — the knee is at the core
count, not at the thread count. The point is that the shift is a *bias*: it
depends on what else was running when that arm ran, and a tuning loop that runs
arms sequentially will absorb it. Repetition beats down the 14% noise; it does
not touch the 19% bias. Node counts are immune to both.

`perf` is unavailable here (`perf_event_paranoid=4`); instruction counts would
need `sudo sysctl kernel.perf_event_paranoid=1`. They would fix the frequency
and co-tenancy problem but not the cache-miss problem, so they would misprice
exactly the TT-heavy changes we care about. Not pursued.

## 2. Node count is a 2x-wrong unit, and it is wrong by position

ns per node, `bench 11`:

| | ns/node | q-fraction |
|---|---|---|
| middlegame positions (0, 8) | 232–237 | 0.69–0.72 |
| tactical (1, 3, 5, 7) | 200–219 | 0.67–0.74 |
| endgames (9, 10, 11) | **111–140** | 0.47–0.64 |

2.1x spread across positions, but only **±9% within a position across depths
8–12**. So cost-per-node is a property of the position, not of the depth. A
paired within-position comparison cancels almost all of it; a pooled
cross-position average does not.

## 3. The observational fit gets the sign backwards

Fitting `t = c_m·N_main + c_q·N_q` over the 60 (position, depth) cells:

    t = 2 ns·N_main + 264 ns·N_q       -> a q node costs 132 main nodes

That is absurd, and it is absurd for a nameable reason. `corr(N_main, N_q) =
0.97` across the corpus, and the residual variation is confounded: endgames have
both fewer q-nodes-per-main-node *and* cheaper nodes, because both track piece
count. The regression attributes the piece-count effect to the q-fraction.

The controlled version breaks the confound by varying the mix *inside* a fixed
position. Sweeping one pricing knob, `bench 11`, aggregate over the corpus:

| arm | q-frac | nodes | true time |
|---|---|---|---|
| baseline | 0.694 | — | — |
| `skip_below = -2000` | 0.819 | **+54.8%** | **+17.8%** |
| `skip_below = 0` | 0.561 | **+5.9%** | **+43.7%** |

Same position, opposite conclusion: more qsearch is *cheaper* per node. This is
a Simpson reversal on our own data. The interventional estimate is the one to
believe.

## 4. One coefficient, fitted on the contrast it will be used for

Fit on the 132 paired within-position contrasts, minimising the error of
(weighted-node ratio) as a predictor of (time ratio):

    a qsearch node costs 0.275 main-search nodes     SSE-band [0.19, 0.40]

| predictor of the true time ratio | median | p90 | max |
|---|---|---|---|
| raw node ratio | 4.66% | 26.3% | 52.1% |
| weighted node ratio (r = 0.275) | 4.59% | **18.0%** | **27.6%** |

Halves the tail; leaves the median alone, because most arms barely move the mix.
Fitting the *same* two-coefficient model on the pooled corpus instead makes the
paired prediction **worse** (p90 36%) — the pooled fit is dominated by variance
the paired design already removes. Fit the cost model on the contrast you will
use it for.

Residual ~18% at p90 is the honest ceiling for two counters. The worst arms
after correction are the ones that change tree *size* by 2–4x (`c_rank = 0`,
`c_rank_depth = 0`), where TT pressure and locality move too.

## 5. What to do

Budget in weighted nodes, normalised so nothing existing is invalidated:

    weighted = (N_main + 0.275 * N_q) / 0.497

0.497 = (1 − 0.694) + 0.275 × 0.694 is the baseline mix, so a weighted node
equals a raw node at default params **exactly**. Every existing node budget and
the `4x nodes ≈ 4.5 cp` ladder carry over unchanged at the default config; only
*differences in mix between arms* get repriced. Do this before the 004 feature
swap, not after.

Recalibrate r with this sweep (≈2 minutes) whenever the hot path changes.

## 6. Where this actually bites

`tune::search_once` budgets by raw node limit (`Limits::nodes`) and `compare`
runs both arms on the same board — paired, so the median error is 4.7% and most
past results are safe. It bites on exactly one axis: **`skip_below`**, i.e.
late-move pruning aggressiveness, where raw nodes misprice by 30–38 percentage
points in *both* directions. That is the pruning-aggressiveness direction, which
is where a proxy win is most likely to fail to convert to Elo. See 005 on
single-axis metrics having degenerate maximisers — this is the same failure
wearing a different hat: a policy can buy proxy improvement by shifting work
into a node type the budget undercharges.
