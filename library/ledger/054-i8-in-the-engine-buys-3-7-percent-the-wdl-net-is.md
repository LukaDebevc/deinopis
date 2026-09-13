# 054 — i8 in the engine buys 3-7%, because the layer weights were never cold

_2026-09-06. `src/wdleval.rs` version-9 net file, `nnue/export_wdl.py --int8`.
Ryzen 5 5500 (Zen 3), `target-cpu=native`, `chess bench`, single thread._

## What was built

A version-9 net file: the five bucketed layers (`down`, `mid`, `up`, `l2`,
`head`) stored as i8 with one f32 scale per (bucket, output column).
Activations and accumulation stay f32, so there is no overflow to reason about
and no activation scale to calibrate. `ft` and `psqt` stay f32 (LEDGER 053).
Version 8 still loads, because the f32 file is the control arm.

Correct: `nnue/verify_wdl.py` on 1500 positions gives 0 feature mismatches,
0 bucket mismatches, logits matching torch to **1.29e-05**, cp to 0.4994.
`cargo test --release` 33 pass, `perft-suite` ALL PASS, default-eval bench
byte-identical at 334,110 nodes.

## The measurement

**First attempt was confounded.** Comparing `gate-3ep.nnue` against
`gate-3ep-q8.nnue` gave 303k -> 312k nps, but the two nets score positions
differently and therefore search different trees — 359,219 nodes against
396,299. A node is not a unit of work (`library/007`), so nps across two
different trees is not a comparison.

The controlled version exports the **same quantised weights** twice, once as a
v8 f32 file and once as a v9 i8 file. Identical evaluations, identical tree,
identical node count; only the kernel differs. Three runs each, medians:

| depth | nodes | f32 kernel | i8 kernel | gain |
|---|---|---|---|---|
| 9 | 396,299 | 303.0k nps | 312.3k nps | **+3.1%** |
| 11 | 998,910 | 302.6k nps | 322.5k nps | **+6.6%** |

For scale, the WDL net needs about **7x** to reach the quadratic eval:
2,145,709 nps against 302,116 nps at depth 9. It got 1.07x.

## Why the footprint argument does not hold here

`CHESS_ACC_OFF=1` rebuilds the accumulator from scratch at every eval, which
adds roughly 90 KB of ft-table traffic per evaluation. It costs 297k -> 202k
nps, i.e. **1584 ns/node**, which puts the effective rate at ~50-60 GB/s.

The i8 file cuts 49 KB from the per-evaluation weight read (65.5 -> 16.4 KB).
At that same rate a *cold* 49 KB would be worth about 1040 ns/node, or +45%.
It was worth ~100-200 ns. **So the layer weights were already in cache and the
capacity argument was counting bytes that nothing was evicting.**

The likely mechanism, a guess with a test attached: the bucket indices are king
squares and material counts, and both change slowly within a subtree, so
consecutive evaluations re-read the same two `down` buckets. Testable by
logging how often each bucket index changes between consecutive evals; not yet
done, because the decision does not depend on it.

That the i8 gain grows from 3.1% at depth 9 to 6.6% at depth 11 is consistent
with a small real footprint effect that gets slightly larger as the tree grows.
It is not consistent with the weight read being the bottleneck.

## Corrections

- **STATE's "i8 the `down` read — this is the Elo lever. Nothing else buys back
  128 Elo" is wrong.** It buys 3-7% nps. Corrected at the source.
- **LEDGER 040's kernel benchmark does not transfer to the engine.** Its 512 KB
  intervening-traffic column forces an eviction that the real search does not
  perform between evaluations. 040's headline — that the arithmetic speed-up is
  invisible — still holds; what does not hold is the implied "and the footprint
  cut is therefore the win". Both were small. 040 measured `deepeval`'s l1 and
  this measured `wdleval`, so 040 is not falsified, only its generalisation.
- The remaining ~2850 ns/node of WDL overhead is **not** the weight read and
  **not** attributable from this measurement. `wdleval` has no `CHESS_ABL`
  equivalent; ROADMAP 4b is now the blocking item, not a nice-to-have.

## Decision

Keep the i8 path: it is free accuracy-wise (LEDGER 053), correct, and 3-7% is
still 3-7%. But it is **not** the lever, and no SPRT of q8 against f32 is worth
running — 5% of speed is a handful of Elo and needs thousands of games to
resolve.

The open question the WDL net actually needs answered is whether a net this
shape can be made ~7x cheaper in *operations*. One evaluation is ~16.6k
multiply-adds across ten matvecs; that is what has to shrink, and bytes are not
the axis. Build the profiler first (ROADMAP 4b) so the next attempt aims at a
measured cost rather than an assumed one.
