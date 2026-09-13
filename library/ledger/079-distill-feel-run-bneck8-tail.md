# 079 · Distill feel-run: bneck-8 tail on frozen m1-b1 accumulator

_2026-09-10. `nnue/distill_tail.py` (new): student keeps m1-b1's ft +
ft_bias exactly (copied, frozen — 394,240 params) and trains everything
downstream fresh on KL to the teacher's own logits. No labels — agreement
is the target. Student is bneck 8, hidden 32, everything else the
teacher's combo cfg (export-checked before training). Local binpack,
`--pool-end 853500985`, seq/stride 8, batch 16384, AdamW lr 3e-4 cosine +
2% warmup, wd 0. `nnue/runs/distill/tail-b8.pt` (3000 steps ≈ 49M
positions, ~335k pos/s on the 4060 Ti, ~2.5 min)._

## Result

Val (213k frozen positions; teacher ce_out 0.72136):

| step | train KL | val KL | cp rmse | cp mae | cp p99 | ce_out |
|---|---|---|---|---|---|---|
| 250 | 0.800 | 0.797 | 213.9 | 121.1 | 830.1 | 0.797 |
| 500 | 0.760 | 0.763 | 159.3 | 85.7 | 628.6 | 0.763 |
| 1000 | 0.750 | 0.755 | 144.2 | 75.3 | 569.6 | 0.754 |
| 2000 | 0.749 | 0.750 | 130.9 | 67.4 | 518.1 | 0.749 |
| 3000 | 0.748 | 0.749 | 129.4 | 66.2 | 510.4 | 0.748 |

Fast drop to step ~500, then a slow grind (mae 86 → 66 over 6x steps),
still falling at the end. For scale: the rung-1 tier that lost −245
disagreed at **MAE 520**; this tail is at **66 after 50M positions from
a fresh init** — 8x closer, with the curve pointing further down.

## Supply side (same binary, net swapped via `$CHESS_WDL`)

- Bench: m1-b1 199091 nodes @~688k nps vs tail-b8 standalone 197207
  (−1.0% tree) @~759k nps (**+10.3%** as a full replacement).
- `evalprof`: 1107 ns/eval → 826 ns (**1.34x per eval**). The per-stage
  split shows mid/up/head ≈ 0 ns for the b8 net — timing misattribution,
  not skipped layers (see verify below); quote only the total.
- Dual-use estimate (small tail on q-evals only, ~82% of evals, eval
  64.5% of node time): ≈ **1.15x on time** before tree growth, worth
  roughly +12–17 Elo at doubling ≈ 80. Against that, a 66-MAE
  *structured* residual — the probe priced σ=25 *white* noise at −12
  (078), and structured can cut either way.

## Fidelity checks

- `export_wdl.py --int8` writes it (1.92 MB standalone; a dual file
  sharing one ft would be ~1.5 MB — format work not done).
- `verify_wdl.py`: features 0/2000, buckets 0/2000, logit mean 7.8e-3,
  engine−model max 14.1 cp — FAILs the 0.6 cp bar, but so does the
  teacher on this tree (max 35.0 cp, mean −1.54). The bar assumes f32
  both sides and has been stale since the i16 accumulator shipped;
  072 recorded the same class of offset (12.7 cp max then) as
  non-blocking. No student-specific structural bug: a skipped layer
  would read hundreds of cp, and the student's gap is *smaller* than
  the teacher's. Features/buckets exact means the b8 shape is
  implemented correctly in the engine.

## Reading

Thin-green-light territory, same shape as 078: demand (66 MAE and
falling, 8x better than the thing that lost −245) against supply
(~1.15x, ~+15 Elo). Two cheap improvements not yet tried: l2 / head /
skiph / psqt have **identical shapes** in teacher and student and can
be warm-started by copy (this run was fully fresh), and 50M positions
is ~3 minutes — 1B is under an hour locally.

## Decision

- Feel-run closed; `distill_tail.py` is the rung-3 instrument.
- Next: warm-started 1B run, then the dual-tail export format (one ft,
  two tails, tier per node through the existing seam), then the
  real-dual SPRT — which remains the only judge.
