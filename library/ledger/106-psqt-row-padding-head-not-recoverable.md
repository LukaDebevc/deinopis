# 106 · psqt row padding is +1.6% nps; head is not recoverable; evalprof's stage column is noisy

_2026-09-19, desk box, HEAD `c48b20c` + this change, m1-b1 (`ce1f2656`).
ROADMAP 4c.3 ranked `head` and `psqt` as "10% of the search, left over from
LEDGER 055 which fixed `down` and stopped". One half paid; the other is
closed. **This entry was rewritten the same day** — the first version argued
from `evalprof`'s per-stage column and over-claimed; see the last section._

## What was shipped

`psqt` was `[NROWS][3]` f32, so row `i` starts at byte `12i` and **12 of every
64 rows straddle a 64-byte cache line**, costing two line touches. At stride 4
every row is 16-byte aligned and lies inside one line. The pad float is never
read, so every sum keeps its terms and its order.

| | before | after |
|---|---|---|
| nps, depth 12, **14 interleaved pairs** | 969,370 | **984,610** |
| node count, all 14 pairs | 827,059 | 827,059 |
| bench fingerprint | 219718 | 219718 |

- [FACT] **+1.57% mean, +1.35% median, n=14 paired.** Bit-identical, 55 tests
  pass. An 8-pair sample of the same thing read +2.67%; it was a fluke.

## How much each stage is actually worth, measured by nps

The honest instrument here is a **paired nps run with the work doubled** — the
extra copy is thrown away, so the eval result and therefore the node count are
untouched, and the delta is the marginal cost of one pass, end to end.

| probe | nps delta | pairs |
|---|---|---|
| one extra `head` + `psqt` pass | −2.6% | 6 |
| one extra **`psqt` gather alone** | **−3.31%** | 12 |

- [FACT] `psqt`'s gather is worth about **3.3% of the search**. It reads 27
  features x 2 perspectives = 54 random rows out of a 768-row table on every
  evaluation.
- [INFERENCE] `head`'s marginal cost is **~0** — doubling both costs no more
  than doubling `psqt` alone, within the noise of the smaller sample.

## The conversion hypothesis was wrong

`stepq` widens one i8 weight per output per input (`row[k] as f32`). At
`dout >= 8` the inner loop is a whole AVX2 vector and the widening vectorises;
at `dout = 3` it stays scalar. A standalone benchmark of the exact 64->3 shape,
with the real 54-bucket working set, said **53.8 ns i8 against 27.0 ns f32**.

Widening `qw` to f32 once at load (bit-identical, bench 219718 exact) moved nps
by **−0.2% against a ±1.5% spread, n=5 paired**. **Reverted.**

- [FACT] `skiph` is the other half of the `head` stage, is **already f32**, is
  also `dout = 3`, and is just as slow. That falsified the story from the
  source before any benchmark was written.
- [FACT] The benchmark also lied outright once: it first reported 0.71 ns for
  192 MACs — 80 MACs/cycle, where AVX2 tops out at 16 — because the call had
  been hoisted out of the timing loop.

## 🔴 `evalprof`'s per-stage column is far noisier than its total

Eight runs of **the same binary**, `taskset -c 5`:

| | min | max | spread | mean |
|---|---|---|---|---|
| head | 71.5 | 88.4 | 24% | 77.9 |
| psqt | 37.3 | 56.0 | **50%** | 48.9 |
| **eval (total)** | 879.7 | 920.5 | **4.6%** | 892.6 |

- [FACT] The stage column is a **difference of two prefix timings**, so a 4.6%
  error on each end becomes 25-50% on a stage worth 5-8% of the total. The
  total is trustworthy; a single stage reading is not.
- [DECISION] **Rank stages with evalprof; never price a change with it.** A
  change gets a paired nps run with the node count held identical. Every
  number in this entry that survived is one of those.
- This retires the arithmetic that opened the item: "0.189 and 0.453 ns/MAC
  against 0.035-0.049" is one draw from a distribution 25-50% wide. The
  *ordering* it implies is real and reproduced across all eight runs; the sizes
  were not, and "10% of the search" was built on the sizes.

## Decision

- **Shipped:** the psqt padding, on its nps number.
- **Closed, not shipped:** `head`. Not conversion-bound, and its marginal cost
  by nps is ~0. ROADMAP 4c.3 updated.
- **Next:** `psqt`'s 3.3% is real and still unclaimed — see LEDGER 107, which
  tries the obvious fix and fails.
