# 080 · Warm distill, 800M: MAE 38.7, converged

_2026-09-10. Follow-up to 079. `tail-b8-warm.pt`: same bneck-8 student on
the frozen m1-b1 accumulator, but `--warm-start` (l2/head/skiph/psqt have
identical shapes and were copied as init, still trainable) and 800M
positions (48,828 steps, inside the 853M unique budget — no repeats).
Same lr 3e-4, KL to the teacher. `nnue/runs/distill/tail-b8-warm.{pt,csv,log}`._

## Init experiment (negative, recorded so it stays dead)

Waist-slicing — init the 8-wide waist from the teacher's first 8 units
per side — measured **init KL 1.065, worse than random init's 1.01**,
and ×2 dropout-style rescaling made it worse (1.073). The waist units
co-adapted; half a trained waist is broken, not half as good. The
surgery was reverted out of `distill_tail.py`, not shipped. Only
`--warm-start` (same-shape copy, init KL 0.90) rode into the long run.

## Result

Val (213k frozen; teacher ce_out 0.72136):

| pos | val KL | cp rmse | cp mae | cp p99 | ce_out |
|---|---|---|---|---|---|
| 131M | 0.7366 | 82.5 | 44.0 | 323.0 | 0.7342 |
| 262M | 0.7351 | 77.1 | 41.2 | 303.4 | 0.7321 |
| 393M | 0.7344 | 73.9 | 39.7 | 287.0 | 0.7315 |
| 524M | 0.7341 | 72.5 | 39.1 | 284.2 | 0.7312 |
| 655M | 0.7339 | 72.0 | 38.8 | 281.1 | 0.7309 |
| 800M | 0.7338 | 71.6 | 38.7 | 279.3 | 0.7309 |

Flat since ~700M — converged at this capacity, not under-trained.
16x the feel-run's positions bought mae 66 → 39.

## Supply side (paired, back-to-back — see lesson)

Standalone via `$CHESS_WDL`, same binary: m1-b1 199091 nodes @~825k
nps vs warm tail **211089 (+6.0% tree) @~925k (+12% nps)**. Eval is
~42% of node time in the qstore tree (reconciles bench +12% with
evalprof's 1.34x per eval; the 64.5% number is pre-qstore). Dual-use
(small tail on q-evals only, ~82% of evals) then prices at **≈1.06–
1.10x on time after tree growth, ~+7–12 Elo** — tighter than 079's
~+15 sketch.

Lesson: the feel-run's absolutes (688k/759k) were box-depressed; this
run's paired ratio (+12%) agrees with the feel-run's (+10%). Quote
paired ratios, never absolutes across sessions.

## Fidelity

`verify_wdl.py`: features 0/2000, buckets 0/2000, logit mean 2.4e-2,
max engine−model 32.4 cp — same class as the teacher's own 35.0 cp on
this tree (stale 0.6 cp bar, see 079). Exported int8, 1.92 MB
standalone; the shared-ft dual file (~1.5 MB) is still format work.

## Reading

Demand (38.7 MAE structured, converged) against supply (~1.1x,
~+10 Elo) with the probe pricing σ=25 white at −12 (078). The race is
tighter than the feel-run suggested and MAE will not settle it —
structured residual vs white noise is an unknown mapping in both
directions. The rung-1 failure (MAE 520, −245) is 13x further out than
this tail, so the stranded question is only the last mile.

## Decision

- The 800M tail is the rung-3 candidate. No more training at this
  shape: converged, and the remaining gap to the teacher (ce_out
  0.7309 vs 0.7214) is capacity, not steps.
- Next: dual-tail export format (one ft, two tails, tier per node
  through the existing seam), then the **real-dual SPRT** — the only
  judge. If it fails, the fallback is a wider student (bneck 12?)
  before any rethink; the seam serves whatever tail wins.
