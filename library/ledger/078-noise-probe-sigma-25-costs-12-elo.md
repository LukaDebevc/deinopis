# 078 · Noise probe: σ=25 on q-evals costs −12 Elo

_2026-09-10. Fixed 400 @8+0.08 conc 5, same frozen binary both sides
(`/tmp/qnoise/chess`, bench re-verified 199091), A `+--qnoise 25`
vs B base, m1-b1 both arms. Log `/tmp/qnoise/qn25.log`._

## Result

| games | score | Elo |
|---|---|---|
| 400 | +52 =282 −66 | **−12.2 [−31, +7]** |

Deterministic key-seeded Gaussian on every q-eval of the full net, zero speed
difference between arms — this is pure disagreement cost. Bench: σ=25 grows
the tree +11.5% (221994 nodes, bit-identical across repeats); σ=100 grows it
+81% (measured, no games — almost certainly underwater, not run).

## Reading

Demand (−12 point, [−31, +7]) against supply (rung-1 measured 1.29x on time,
worth ~+25–30 Elo at doubling ≈ +80): net ≈ **+15 at point estimates, interval
wide**, resting on white-noise ≈ distilled-residual. A distilled tail's error
is structured (position-correlated), which can cut either way against white
noise — the probe bounds the question, it does not settle it.

## Decision

- **Thin green light for rung 3**: distill a small tail from m1-b1 against
  the teacher's own logits (agreement is the target, not absolute quality),
  sharing the frozen ft/accumulator — one file, two tails, tier selected per
  node through the existing seam. Judge it by SPRT of the real dual, not by
  MAE.
- **Large-net top tier deferred, not rejected.** Leverage concentrates near
  the root while cost concentrates at the leaves, so spending 2–3x eval on a
  few percent of nodes is the right shape — but "larger" must mean "better",
  and the 20B round robin says width is not the lever (1024-wide s1 finished
  last, −50). Train better, not bigger; if a longer lc0-label run beats m1-b1
  it enters through the same seam as the top tier. No separate bigger-net
  project now.
