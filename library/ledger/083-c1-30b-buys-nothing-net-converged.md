# 083 · c1: 30 more billion buys nothing, the net has converged

_2026-09-10. `m1-c1`: m1-b1 trained 30B further on the same lc0 target
(`nnue/runs/c1-20260910/job3.sh`: same 1.48B unique positions, seed 2 for a
fresh presentation order, peak lr 2.2e-4 like b1 — a clean repeat of b1's
manipulation with only the parent changed). SPRT [0,5] @8+0.08 conc 5,
m1-c1 vs m1-b1, O1 binary both sides._

## Result

| games | score | Elo | LLR |
|---|---|---|---|
| 3000 | +468 =2050 −438 | **+2.4 [−4.5, +9.3]** | −0.06, inconclusive |

Log `nnue/runs/c1sprt-20260910/sprt.log`. Faded from +9 (LLR ~1.0 at 950
games) to +2 — the early lead was noise. b1 bought +22.6 over a1 on the same
recipe (072/ORDERING log); c1 buys ~0 over b1. ~20 more passes over positions
already seen ~20 times each teach nothing further.

## Decision

- m1-b1 stays the net. Do not run a c2 on this data.
- The net-side lever is **unique games, not more passes** (LEDGER 021's
  ~106 GB of unused positions — never bought). Next training spend goes
  there: extract + relabel fresh lc0 games, then a full-length run.
