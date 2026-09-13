## 045 — Annealed pawn predicates beat the best magic; the merge is the ceiling

**Setup.** Same arm shape as 044 (`PurePSQTBucket`: base PSQT + one 768-dim
PSQT delta per bucket, no hidden layer), 1.0B positions/arm, `all.data`,
15258 x 65536, lr 1e-3, seed 0, warm start `ckpt-big/psqt_768-_64-_16-_1.pt`.
Only the 64-bucket pawn rule changes. Rule family: 10 predicates
`popcount(our_pawns & MASK_j) > k_j` packed to 1024 states, then merged to 64.
`nnue/attic/anneal_rules.py`, `runs/anneal_full.log`, `runs/rule_arm.log`.

| arm (64 buckets) | val loss | vs `rand64` |
|---|---|---|
| `rand64` (no chess in it) | 0.028502 | control |
| incumbent magic | 0.027814 | −2.41% |
| `newmagic` | 0.027577 | −3.25% |
| annealed rules, 300 proposals | 0.027442 | −3.72% |
| **annealed rules, 20,000 proposals** | **0.027432** | **−3.75%** |
| `king6` (not a pawn rule) | 0.027335 | −4.09% |

**The family wins.** A readable-family rule beats the best magic hash by 0.53%
of val loss — ~30x the noise floor of 044 — and closes most of the gap to
`king6`, the strongest single feature we have.

**Determinism, verified three times.** `rand64` seed 0 was re-run inside each
invocation and reproduced **bit-identically at every logged step**, train and
val (2000: 0.028741/0.028573, 6000: 0.028584/0.028558, 15258: 0.028536/
0.028502). Only throughput moved (3.4 -> 5.6M pos/s, warm page cache). So arms
may be dropped and compared across invocations: `train.py` reseeds and resets
the batcher per arm, and `--only` was added to exploit it.

**Negative, and the important one: 67x more search bought ~nothing.**
300 proposals/chain gives 0.027442; 20,000 gives 0.027432 — 0.036% of val loss,
against a floor of 0.004-0.018%. Real, but 1/15th of the gap the family already
had over `newmagic`. Offline it is the same story: the 1024-state score rose
5.65 -> 6.87 (+21%) while the merged k=64 score rose 3.881 -> 3.911 (+0.8%) and
I(B;G) rose 3.563 -> 3.569. **The greedy 1024->64 merge is the ceiling, not the
search.** Predicted in advance from the offline numbers (0.02742-0.02746) and
it landed at 0.027432 — so the offline proxy predicted a training outcome.

**Negative: the Renyi-2 collision objective is the wrong functional.** On
40,000 held-out games / 2M pairs, two independent draws:

| rule | collision score | I(B;G) | refresh |
|---|---|---|---|
| `rand64` | 3.498 / 3.502 | 3.133 / 3.129 | 0.272 |
| incumbent | 3.657 / 3.659 | 3.313 / 3.310 | 0.237 |
| `newmagic` | 3.657 / 3.668 | 3.458 / 3.460 | 0.206 |
| annealed k=64 | 3.911 | 3.569 | 0.216 |

Estimator noise ~0.010 on the collision score, ~0.003 on I(B;G). **The
collision score cannot separate the two magics** (gap 0.000 and 0.009, inside
its own noise) even though training separates them by 0.85% = ~50x the training
floor. **I(B;G) separates them by 0.145 (~50x its noise) and orders all four
pawn rules monotonically with trained val loss** — the first offline proxy here
checked against real training on more than one contrast. [INFERENCE] Renyi-2 is
dominated by the largest buckets: incumbent's flatter bucket distribution
(H 5.988 vs 5.795) lowers its collision-by-chance term and offsets its weaker
within-game concentration; Shannon MI weights each bucket by its own mass.
Finalist selection now uses I(B;G) (`anneal_rules.py`). The anneal itself still
optimises collision, because over 1024 states at 32 plies/game the Miller-Madow
correction is ~0.70 bits — the size of the signal. Changing that needs its own
test.

**Negative: the family's stated advantage does not exist.** It was built
because a popcount does not scramble — a pawn moving inside its own mask is a
no-op — so it should refresh less often than a magic. Measured on the identical
slice: **0.216 vs `newmagic`'s 0.206**. It is slightly *worse*. Stability has
to be put into the objective or it does not appear. Nor is the result readable:
the winning masks are scattered 17-24 square blobs with thresholds 1-2, i.e.
"at least 2 pawns among these ~20 scattered squares" — a soft material count,
matching the 54.5% of its I(B;G) that is pawn count alone.

**No saturation on the pawn axis.** Marginal k=64 score per added bucket bit:
+0.628 (2->3), +0.753 (3->4), +0.843 (4->5), +0.886 (5->6) — *increasing*
returns, still rising at 6 bits. King routing saturates by 121-256 buckets
(LEDGER 035); the pawn axis does not. 7-8 pawn bits is the cheapest open test.

**Method fixes that changed numbers.** `merge_states` originally fit the
1024->64 LUT on the same games it scored on; refit on slice A and applied to B,
the bias was **0.18 at k=64 — four times the whole spread across 12 chains**
(3.867-3.911). A->B shrinkage over the finalists is only +0.0147, so the anneal
does not overfit its slice; the overfitting was in the merge. Trainer and
annealer pawn bitboards agree on **200,000 records, 0 mismatches**.

**Decision.** Keep the popcount family; drop the stability and readability
claims made for it. Next, in order: (1) more pawn bits, 7-8, since the curve is
still climbing; (2) a better compression step, since that is the measured
ceiling; (3) `king6 x pawn_rule` scored against `king6` alone. Do not spend
more on annealing proposals.
