**023 · Bucketed quadratic: -18.3% val loss, and it is mostly material** ·
2026-08-27
Luka's design: `eval = psqt(x) + c + <act(V x + B[b]), R[b]>`, where the bucket
`b` is a hard function of the position. The rule that makes it deployable is
that the bucket may only change the **read**, never the accumulator: `V x`
stays bucket-free and incrementally updatable, and a capture that changes the
bucket costs a different lookup, not a rebuild.
With `act = square` and one bucket this is *exactly* the deployed `Quadratic`
(`W = V^T diag(R) V`), so the baseline is nested inside the family and any win
is new capacity rather than a different model.
**Setup:** 894M-position pool, 8000 steps x 65536 = 524M positions (0.59
epoch), r=512, warm-started from the same PSQT, identical batches in identical
order per arm. Bar = **base r=768 x1, the full-rank single-bucket maximum**;
beating r=512 alone would only show that buckets beat a rank-deficient model.

| arm | val | vs bar | buckets | table |
|---|---|---|---|---|
| base r=512 x1 | 0.021095 | +0.2% | 1 | - |
| **base r=768 x1 (bar)** | **0.021051** | - | 1 | - |
| bishops read+pre | 0.020179 | -4.1% | 16 | 32 KB |
| bishops **read only** | 0.020198 | -4.1% | 16 | 16 KB |
| queens | 0.020068 | -4.7% | 9 | 18 KB |
| kings | 0.019934 | -5.3% | 4096 | 8.4 MB |
| count | 0.019423 | -7.7% | 8 | 16 KB |
| material | 0.018269 | -13.2% | 576 | 1.18 MB |
| **all five** | **0.017200** | **-18.3%** | 4705 | 9.6 MB |

**Calibration measured in the same run:** +50% rank buys 0.2%, 5x compute buys
0.5%. So 18.3% is not on the same axis as anything we had been tuning.
**Crossing beats summing.** `material` = rooks(3) x light bishop(2) x dark
bishop(2) x queen(2) per side, crossed. Its factors *separately* are worth 4.1%
and 4.7%; crossed with rooks they are worth 13.2%. A rook ending with opposite
bishops is its own regime, not the sum of two corrections. This is the
degree-3-and-up capacity the cubic idea was after, for 1.18 MB instead of a
150 MB tensor.
**`count` is not a new idea, it is a deleted one.** `eval.rs:207` tapers PeSTO
between mid- and endgame tables by remaining material. The quadratic net has
**no tapering at all** -- one matrix, opening to endgame -- and nobody noticed
because it still gained +354 Elo. 8 piece-count buckets recover 7.7%, which is
the price we had been paying silently.
**The pre-activation shift `B` buys nothing with ONE family and pays with
several.** bishops alone: read-only 0.020198 vs read+pre 0.020179, 0.09% --
noise. material+count, measured in-run so there is no cross-invocation confound:
read-only 0.017842 vs read+pre **0.017685, 0.88%** -- ~9x the noise floor
(`runs/shift.log`). **Corrected at the source:** this entry previously concluded
"every deployable configuration should drop it", generalising a single-family
null to the stacked case. Wrong. Each family contributes its own shift vector
and they sum, so the shift path gains capacity as families are added, while the
single-family test had none to gain. Cost of keeping it: +43% parameters
(992,514 vs 693,506 at material+count), 0.6 -> 0.85 MB, one extra gather and add
on the read-out copy only. It still never enters the accumulator, so `V x` stays
incrementally updatable.
**Corrected, twice, at the source:** kings' train-below-val was read here as
memorising. That reading is dead. First correction: `queens` has 460x fewer
bucket parameters and shows the same sign flip. Second and final: the logged
`train` number is `l.item()` of the **last single batch** (65,536 positions,
`train.py:399`), while `val` is 1M positions. A one-batch readout against a 1M
one carries no generalisation signal in either direction, so the comparison
should never have been made. Do not read train-vs-val in these logs; if we want
an over-fitting diagnostic, log a running mean over the last N batches.
What stands without that claim: kings buys 0.6% over queens for 460x the
memory, is 8.4 MB of the 9.6 MB total, and cut training throughput from
1.25M to 789k pos/s.
**This is a proxy.** Val loss on a game-level split, not Elo. The proxy rule
applies and nothing here is deployed. Deploying *any* of it also requires an
accumulator, which the engine does not have (LEDGER 019 removed the need for
one); the r=512 form needs `V x` maintained incrementally.

**Subset screen** (same pool, steps, rank and warm start; `runs/groups.log`).
Which families actually carry the gain, all read-only:

| configuration | val | vs bar | table int16 | pos/s |
|---|---|---|---|---|
| base r=768 x1 (bar) | 0.021051 | -- | -- | -- |
| material+count x584 | 0.017827 | -15.3% | 0.6 MB | 1,232k |
| +bishops+queens x609 | 0.017809 / 0.017804 | -15.4% | 0.62 MB | 1,057k |
| +kings x4705 | 0.017443 | **-17.1%** | 4.8 MB | 989k |

**Noise floor measured, not assumed: ~0.1%.** Two replicate pairs.
`+bishops+queens` run twice inside one invocation: 0.017809 / 0.017804 (0.03%).
`material+count` run as arm 0 of two separate invocations: 0.017827 / 0.017842
(0.086%). Corrected: the first pair alone was quoted here as a 0.03% floor,
which was one draw read as a bound. Both gaps are the same mechanism --
nondeterministic GPU reductions in the embedding backward, compounded over 8000
optimiser steps, so identical code diverges chaotically -- and the honest floor
is the larger, ~0.1%. Anything under that is nothing.
Also established: arms draw identical batches regardless of arm *position*, so
the shared-RNG bug that made later arms see different data is genuinely fixed,
and within-run comparisons are clean.
**bishops and queens are already inside material.** They add 0.1% on top of
material+count -- i.e. exactly at the noise floor, indistinguishable from zero. `material` crosses
rooks x light-bishop x dark-bishop x queen per side, so the standalone 4.1% and
4.7% were never additive with material's 13.2%: same information, counted twice.
Do not stack sub-factors of a crossed family alongside it.
**kings is worth 2.05%** over material+count -- ~20x the noise floor, so real --
for 8x the table.
**Corrected: kings' table size is not a cache argument.** This entry earlier
treated 4.8 MB as disqualifying. It is not. The king-bucket index changes only
when a king moves, so within a subtree the same 512-value row is gathered over
and over and stays in L1 whatever the total table size. Big tables cost resident
RAM and cold misses, not bandwidth, whenever the index is slow-moving. The same
argument covers `material` (changes only on a capture or promotion) and fails
for any future family keyed on something that changes every ply.
**Shift ablation, all four arms in one invocation** (`runs/shift.log`), so no
cross-invocation confound anywhere in this table:

| configuration | val | vs bar | shift gain |
|---|---|---|---|
| material+count x584 read only | 0.017842 | -15.2% | -- |
| material+count x584 read+pre | 0.017685 | -16.0% | 0.88% |
| material+count+kings x4680 read only | 0.017466 | -17.0% | -- |
| **material+count+kings x4680 read+pre** | **0.017227** | **-18.2%** | **1.37%** |

The shift grows with the family set: 0.09% at one family, 0.88% at two, 1.37%
at three. That is the predicted shape -- each family adds its own shift vector
and they sum -- and it reproduces the 1.4% seen across invocations for all five.
**Deployable configuration: `material+count+kings`, r=512, read+pre.** 0.017227
is **-18.2% vs the bar**, statistically the same as the five-family read+pre
0.017200 (0.16% apart, at the noise floor). So the whole -18.3% is reachable
without bishops and queens. That is bishops+queens measured as worthless a
third time, in a third invocation.
Cost: two 4680x512 tables, 4.8 MB each in int16, 9.6 MB total; three read
gathers and three shift gathers per eval. Both indices are slow-moving (material
changes on a capture or promotion, kings on a king move), so the gathered rows
stay hot in L1 and the size is a RAM cost, not a bandwidth one.
**Nothing here is deployed.** Val loss, not Elo, and the engine still needs an
accumulator.

---
