# 053 — int8 weight-only is free everywhere except the ft table

_2026-09-06. `nnue/quant.py`, `nnue/qat.py`. Net `ckpt-wdlnet/gate-3ep.pt`
(width 512 `gate`, bneck 16, hidden 32). Frozen val set, n = 409,600._

## Setup

Weight-only, symmetric, round-to-nearest, one scale per (bucket, output
column); activations and accumulation stay f32. That choice follows LEDGER 040:
the integer *arithmetic* speed-up is invisible under real memory pressure and
the *footprint* cut is not, so what is being bought is bytes. It also removes
the whole overflow question — nothing accumulates in an integer.

Everything is measured on the **folded** net (`base + delta` collapsed), which
is what `export_wdl.py` writes. Folding is exact: val CE drifts 0.0 to every
digit.

f32 bytes read per evaluation, two buckets for each per-side layer:

| | down | l2 | mid | up | head | skiph | total |
|---|---|---|---|---|---|---|---|
| KB/eval | 32.0 | 16.0 | 8.0 | 8.0 | 0.8 | 0.8 | **65.5** |

`ft` is not in that column — the accumulator is incremental, so its cost is the
1.57 MB table's residency, not a per-node read.

## Result

Round-to-nearest, no training at all. Baseline fp32 `ce_out` 0.729461; the
repo's ranking threshold is ~0.0004 (README: nothing under ~0.05%).

| quantised | Δ ce_out | cp RMSE vs fp32 | cp p99 |
|---|---|---|---|
| down mid up l2 head psqt skiph @ int8 | **−0.00002** | **3.7** | 14.1 |
| ft @ int8 | +0.00018 | 10.8 | 35.8 |
| everything @ int8 | +0.00017 | 11.4 | 38.1 |
| everything @ int6 | +0.00431 | 45.9 | 152.7 |
| everything @ int4 | +0.06424 | 170.4 | 550.8 |

**The 65.5 KB per-evaluation read goes to int8 for nothing.** It is −0.00002
val CE, i.e. below the noise floor and on the good side of zero. 65.5 KB →
16.4 KB, which fits Zen 3's 32 KB L1D. That is the lever ROADMAP #4 names.

`nets/gate-3ep-i8core.nnue` is that net, exported and passing `verify_wdl.py`:
0/1500 feature mismatches, 0/1500 bucket mismatches, engine−model 0.50 cp max.

## Why `ft` is the hard one

Mechanism, not speculation. Per output column the ft table's mean |w| is 13.1%
of its max |w| — mean |w| 0.192, max 6.53, p99 0.908. A typical weight
therefore uses about an eighth of int8's range, so the bulk of the table is
quantised at roughly five effective bits. On top of that the accumulator is a
sum of ~17.7 rows, so the per-row errors add.

## Clipping does not fix it, and is not needed

Clipping the ft table before quantising, then measuring end to end:

| clip | int8 Δce | int8 cp RMSE | int6 Δce | int4 Δce |
|---|---|---|---|---|
| none | +0.00018 | 10.8 | +0.00375 | +0.05515 |
| ±4 | +0.00019 | 10.4 | — | — |
| ±2 | +0.00046 | 15.2 | +0.00272 | — |
| ±1 | +0.00966 | 67.3 | +0.01081 | +0.02816 |

At int8 clipping is neutral at ±4 and a loss below that. QAT at ±2 recovers
15.2 → 10.4 cp over 3000 steps, i.e. back to the *unclipped* baseline and no
further. **A clip is only needed once the accumulator itself is an integer**,
and this net was not trained with one (`wdlarms.py --clip` exists and was off).

For that day: measured on 212,992 held-out positions, the accumulator |z| has
median 0.47, p99.99 9.2 and **max 13.3**. An i16 accumulator with ±16 of range
covers it with room to spare; ±32 is generous.

## QAT helps int4 and nothing else — and the control says why

Straight-through fake quantiser, KL to the fp32 net's logits, 3000 steps at
lr 3e-4 with a cosine schedule.

| arm | RTN | after QAT | Δce after |
|---|---|---|---|
| down mid up l2 head @ int4 | 61.2 cp | 39.3 cp | +0.00251 |
| down mid up l2 head @ int8 | 3.7 cp | 2.9 cp | +0.00002 |
| **control: nothing quantised** | 0.0 cp | **0.9 cp** | +0.00000 |

The control is the point. Fine-tuning with no quantiser attached still moves
the net 4.9 cp away from itself at step 500 and settles at 0.9 cp — so cp RMSE
against a fixed fp32 reference measures *any* weight movement, not only
quantisation error, and the int8 "improvement" from 3.7 to 2.9 is within that.
Every arm also drops sharply at step 3000 because the cosine LR anneals to
zero, control included; the mid-run rows are schedule, not convergence.

**int4 is not free anywhere.** Even after QAT it costs +0.0025 val CE, six
times the ranking threshold, for 16.4 → 8.2 KB. Input-axis grouping at 32 gets
RTN from ~62 to 51 cp and does not change the verdict.

## Decision

- Deploy int8 weight-only on the five per-evaluation layers. No training step
  is required for it: round-to-nearest is already free.
- Leave `ft` at f32 or i16 for now. Its int8 cost is 4x the rest combined, its
  per-node read is zero, and its win is L2 residency rather than L1D.
- Do not pursue int4.
- **All of the above is proxy.** The next measurement is an SPRT of
  `nets/gate-3ep-i8core.nnue` against `nets/gate-3ep.nnue` — both f32 files, so
  it prices the accuracy cost alone, before any integer kernel exists to be
  wrong. The footprint win needs the v9 file and the i8 matvec in
  `src/wdleval.rs` and is a separate SPRT after that.
