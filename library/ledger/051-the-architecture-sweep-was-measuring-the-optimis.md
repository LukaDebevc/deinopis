# 051 — The architecture sweep was measuring the optimiser; the activation is where the loss is

**Date** 2026-09-04/05 · **Corpus** `all.data`, 893,500,985 records, val = last 40M
strided by 40 → 1M positions, identical in every run (`teacher alone on val:
ce_out 0.64714 onll 0.56100`, base-rate outcome entropy 1.04095)
**Budget** 250M positions, batch 16,384, seed 0, unless stated · **Noise** ~0.001
(from four seed pairs)

## Setup

Batches 5–9 of the WDL architecture sweep, plus a batch-size study on the
cluster. Metric is `ce_out`, the three-way cross-entropy against the actual game
outcome. Lower is better. Cluster runs use a 300M-record file built as
`[first 260M] ++ [last 40M]` so the validation set is byte-identical to local;
cluster and local `ce_out` are NOT comparable to each other (different training
slices) and are anchored by running the same arm on both.

## Result

### The learning rate was wrong, and it confounded everything

`combo`, cluster, 250M:

| lr | ce_out |
|---|---|
| 1e-3 (used for all ~50 earlier arms) | 0.74967 |
| **2e-3** | **0.74712** |
| 4e-3 | 0.74824 |
| 8e-3 | 0.81169 (diverged) |
| 1.6e-2 | 0.89237 (diverged) |

The sweep ran 0.0026 below its own optimum. That matters beyond the lost
accuracy: `BucketLinear` stores `W_base + ΔW_b`, and Adam normalises each
parameter to ~lr regardless of gradient size, so an over-parameterised arm moves
its shared direction at roughly twice the rate. **Redundant parameterisation
acts as a learning-rate multiplier**, so part of what the sweep measured was
which arm accidentally had a better effective lr. Every effect below ~0.002 in
batches 1–4 is uninterpretable for this reason.

### The hash control: no information, and the effect was the optimiser

Second layer varied, king bucket held on the first:

| arm | second layer | lr | ce_out |
|---|---|---|---|
| `nol2` s0/s1 | one plain matrix | 1e-3 | 0.75763 / 0.75897 |
| `hashm` s0/s1 | 54 matrices, Zobrist-random index | 1e-3 | 0.75648 / 0.75682 |
| `hash2` s0/s1 | **2** matrices, Zobrist-random index | 1e-3 | 0.75679 / 0.75656 |
| `nol2` | one plain matrix | **2e-3** | **0.75468** |

Two random buckets equal fifty-four random buckets, and simply raising the
learning rate beats both. The partition was never doing anything — confirming
the permutation-null result that the index carries no information (LEDGER 050) —
and the apparent 0.0016 gain was the reparameterisation acting as a higher lr.

**Correction to the batch-5 reading:** material bucketing buys 0.00102 over
`nol2`, which is *less* than an information-free 54-way split buys. Material
buckets are not earning their 54 matrices. Queen-square buckets (`qbuck`,
0.75646 vs `base` 0.75661) buy nothing either. King buckets survive: 0.00564
over no buckets, and `hashk` bought nothing (−0.00055, wrong sign).

### The nonlinearity is where the loss actually is

Full 2³ over crelu positions in `acc → [a0] → king layer → [a1] → queen layer →
[a2] → head`, identical shapes and parameter counts within each column:

| crelu at | 256→32 | 64→64 |
|---|---|---|
| nowhere | 0.83193 | 0.83280 |
| `acc` only | **0.76863** | 0.78537 |
| `B` only | 0.78109 | 0.77697 |
| `C` only | 0.78174 | 0.77752 |
| `acc`+`B` | 0.76034 | 0.77384 |
| `acc`+`C` | **0.75911** | 0.77155 |
| `B`+`C` | 0.77485 | **0.76978** |
| all three | 0.75945 | 0.76990 |

Removing every nonlinearity costs **0.063** — ten times any architecture effect
in the whole sweep. Two are enough; the third is worth ~0.0001. Which two
depends on the shape, and the two columns disagree: at 256→32 the accumulator
crelu is the most valuable single one, at 64→64 the least. [INFERENCE] a clamp
is worth most where the vector is widest, which is the accumulator in the first
column and not in the second.

### The accumulator activation, and Stockfish's fold

`combo` at lr 2e-3, only the accumulator activation changed:

| arm | accumulator | into L1 | params | ce_out |
|---|---|---|---|---|
| `gate` | 512, signed × (0,1) gate | 256 | 890,520 | **0.74033** |
| `pair` | 512, both clamped (0,1) | 256 | 890,520 | 0.74117 |
| `act01w512` | 512, clamp(0,1) | 512 | 1,156,760 | 0.74185 |
| `gateh` | 256, gated fold | 128 | 560,280 | 0.74566 |
| `pairh` | 256, folded | 128 | 560,280 | 0.74648 |
| `combo` | 256, clamp(0,1) | 256 | 693,400 | 0.74712 |
| `act11` | 256, clamp(−1,1) | 256 | 693,400 | 0.76337 |

- **clamp(−1,1) costs 0.0163** at identical shape. [INFERENCE] it is the
  identity on most of the accumulator's range, so it is closer to no
  nonlinearity at all; 0.0163 is about a quarter of the 0.063 that going fully
  linear costs.
- **The fold is a cost win, not a loss win.** Widening 256→512 with a plain
  clamp buys 0.00527 on its own; widening *and* folding buys 0.00595. The fold's
  own contribution over the width-matched control is 0.00068, inside noise.
  What it buys is price: `pair` matches `act01w512` with 23% fewer parameters
  and **half the L1 input width**, which is the engine's actual bottleneck.
- **The gated fold beats the symmetric one at both shapes**, +0.00084 at 512 and
  +0.00082 at 256. One noise unit each, but the two shapes agree to 0.00002.
  A signed × non-negative product can represent negative interactions that two
  clamped (0,1) halves cannot.
- **`foldkeep`** (accumulator 256 folded to 128, bottleneck 16→32) reaches
  0.74430 against `combo`'s 0.74712 **at an identical 4,096 weights read per
  evaluation**. Folding and doubling the layer that reads the accumulator is
  free.

### Bucket controls that came back empty

| arm | what it tests | ce_out | vs base 0.75661 |
|---|---|---|---|
| `kingadd` | 64×hidden additive offset instead of bucketing L1 | 0.76230 | +0.0057 |
| `kingboth` | both | 0.75686 | +0.0002 |
| `king4` / `king16` | king at 4 / 16 buckets | 0.76051 / 0.75889 | dose-response is monotone with *increasing* returns |
| `qbuck` | queen square instead of material on L2 | 0.75646 | −0.0002 |
| `skip` in a linear stack | any bypass to the logits | 0.83279 vs 0.83280 | exactly nothing |

The skip result has a proof, not just a measurement: in a linear stack the
bypass term `B·S` and the main path `B·(Q + ΔQ_c)·H` are both linear in `B`, so
`S` only shifts the shared `QH` and adds no representable function. The
product form `(K + ΔK_b)(Q + ΔQ_c) = KQ + K·ΔQ_c + ΔK_b·Q + ΔK_b·ΔQ_c` already
supplies the constant, king-only, queen-only and king×queen terms; the only
thing it cannot do is vary them independently.

### HalfKA, inconclusive

`acc → clamp → logits` with nothing between (250M): `flat0` (no king) 0.79304,
`flatkb` (64 output matrices) 0.79222, `flatka` (HalfKA input, 3,195,395 params)
0.79133. In the deep net, `gateka` (mirrored HalfKA, 32 king squares,
13,473,432 params) scored 0.74597 against `gate`'s 0.74033. **Not a verdict**:
each HalfKA row is seen far less often, and 250M positions is plausibly too few
to fill a 13M-parameter sparse table. Retest at a larger budget.

### Batch size and hardware

`combo`, 250M, cluster, one arm per row:

| batch | lr | warmup | ce_out |
|---|---|---|---|
| 16,384 | 1e-3 | 2% | **0.74967** |
| 262,144 | 4e-3 (√16) | 2% | 0.75448 |
| 262,144 | 4e-3 | 10% | 0.75505 |
| 262,144 | 1.6e-2 (×16) | 2% | 0.75718 |
| 262,144 | 1e-3 | 2% | 0.77147 |

A 16× batch with lr unchanged costs **0.0218**. √16 lr scaling is the right rule
and recovers most of it, but even at the best lr the large batch still loses
0.0048 while buying only 1.23× throughput. **Keep batch 16,384.** More warmup did
not help, so the reduced step count was not the binding constraint.

Throughput, measured not assumed: A100 666k pos/s at batch 16k against the
RTX 4060 Ti's ~470k — **1.4×**, not 10×. Two processes on one A100 give 748k
against 666k for one, **1.12×**; memory is not the limit (2.5 GB per process)
but kernel launches and bandwidth are. Spread across cards, do not co-locate,
and do not build a stacked-weights trainer for this.

## Decision

- **lr 2e-3, batch 16,384** are the new defaults.
- **Adopt the gated fold.** `gate` at 0.74033 against the sweep's starting
  `base` 0.75661 — and 0.0154 of that 0.0163 is the learning rate plus the
  activation, not architecture.
- **Drop clamp(−1,1), material buckets and queen buckets.** Keep king buckets.
- **Stop sweeping buckets.** Everything outside "have a nonlinearity" and
  "bucket by king" is inside the noise, and the two largest wins found in this
  entry were both optimisation, not structure.
- The next number that means anything is Elo, not `ce_out`.

## Files

`nnue/wdlnet.py` (`act` ∈ crelu/sym/pair/gate, `ft_mode` ∈ plain/halfka/halfkam,
`stack` ∈ ""/kq/flat), `nnue/wdlarms.py` batches 5–9, `nnue/runs/wdlnet/`
sweeps 5–8, cluster `<cluster scratch>`.
