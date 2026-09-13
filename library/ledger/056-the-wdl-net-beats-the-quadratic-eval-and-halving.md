## 056 — The WDL net beats the quadratic eval, and halving the accumulator is free

_2026-09-06. `tools/sprt3.sh`, `nnue/runs/sprt3-20260906-092758`, commit
`f3e0707` + working tree._

**Setup.** Four 10B-position arms, cluster GPUs 2–3, two phases, lr 6.7e-4,
batch 16,384, wd 1e-2, clip 1.98, one arm differing from the reference in
exactly one thing (`other/run10b.sh`). A `gate` 512 wide, bneck 16 · B the same
with wd 0 · C `gateh`, accumulator halved to 256 · D `gateb8`, waist halved to
8. All four completed rc=0.

Cross-run comparability was checked before any of this was believed: the line
`teacher alone on val: ce_out 0.64714 onll 0.56100` is identical across all 30
cluster logs, so these val numbers sit on the same instrument as every earlier
run. That is the LEDGER 052 trap — a val set that quietly changed underneath
the numbers — and it is clear here.

**B is off the Pareto frontier and was dropped.** Same architecture as A, worse
val (0.72678 vs 0.72582), and its 699,462 nps against A's 688,748 is a
different tree, not a faster kernel. wd 0 also leaves an unbounded weight tail
(max |w| 5.306). So the LR cut alone is *not* enough; the decay is carrying
part of the fix.

**Result.** Each of the three frontier arms, exported to int8, played the
incumbent quadratic eval. Same binary both sides, frozen as a copy, both evals
named explicitly (`quad` verified at 334,110 bench nodes, so not the PeSTO
fallback). SPRT [0, 5], alpha = beta = 0.05, 10+0.1, concurrency 4 of 12, run
**sequentially** — three at once would put contention into the exact quantity
being measured, which is biased about −19% under load.

| arm | shape | Elo vs quad | games | val ce_out | nps | size |
|---|---|---|---|---|---|---|
| `gate-A` | 512, bneck 16 | **+84.6 [+61, +109]** | 402 | 0.72582 | 688,748 | 2.11 MB |
| `gateh-C` | 256 wide | **+96.5 [+71, +123]** | 384 | 0.73270 | 870,471 | 1.19 MB |
| `gateb8-D` | bneck 8 | **+53.9 [+35, +73]** | 624 | 0.72847 | 790,550 | 1.92 MB |

All three accepted H1.

**The sign flip is the kernel, not the net.** LEDGER 052 measured this design
at −127.7 Elo [−156, −101] over 324 games. The architecture did not change;
removing a bounds check from the matvec inner loop took it from 7.0x slower
than the quadratic eval to 3.14x (LEDGER 055). One manipulation, opposite sign.
**052's −127.7 is corrected at the source rather than left standing.**

**Val loss ranked them backwards for the third time.** `gateh` is 0.0069 worse
on val — the widest gap in the sweep — and leads on Elo. Same failure as
`gatew1024` over `gate` in 052. Halving the accumulator costs almost nothing
the search can feel and buys 26% more nps, which is the trade the engine is
actually making while it is eval-bound. Do not rank these on val.

**What this does not settle.** The three are ranked only transitively through
quad and their intervals overlap, so the shape question is open;
`tools/netcmp.sh` with `gate-A` as reference plays them head to head at equal
nodes and equal time. `gateb8` losing 31 Elo to `gate-A` while running *faster*
is the one clean statement: the waist, unlike the accumulator width, is load
bearing.
