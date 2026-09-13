## 047 — Whitening the router logits works as a TRAINING pressure, and `--eigfloor` is dead

`nnue/losslab/runeig300.log`, `runspec.log`, `runspec2.log`, `runnone.log`,
2026-09-03. All arms n=1, 300 steps, 12 bits, batch 16384, no penalty,
`--no-table`. The 6104-step rows are from `runwhite.log`, same config, n=1.

**The claim under test was mine and it was wrong.** After measuring that ZCA
whitening of the router logits gives 4095/4096 occupied buckets for nothing,
I explained its flip2 cost as `Cov^-1/2` dividing each direction by its own
sigma and so promoting near-degenerate directions to full bits that flip every
move -- and proposed `--eigfloor`, which caps that amplification, as a
continuous dial between "even buckets" and "don't promote noise". The premise
was never measured. It is false.

| arm | val | flip2 | eff | occ | I(B;game) | max/mean | min/mean | condition |
|---|---|---|---|---|---|---|---|---|
| none | 0.029266 | 49.8% | 54.9 | 247 | 1.68 | 7.42 | 0.0436 | **170** |
| center | 0.029202 | 63.7% | 445 | 3909 | 3.78 | 6.39 | 0.0916 | **70** |
| white | 0.029311 | 74.1% | 3090 | 4096 | 5.65 | 1.68 | 0.627 | **2.7** |

The trained whitened router has **no flat directions**: its smallest
eigenvalue is 63% of the mean, so an eigenvalue floor below 0.627 is an exact
no-op and one above it squashes directions carrying real variance. Measured
directly: eigfloor 0, 0.1 and the 1e-2 default are the same arm (eff 3084,
3092, 3090); eigfloor 1 drops eff to 1389 and moves flip2 by less than the
noise. **Do not run the eigfloor ladder.**

**What whitening actually does.** The spectrum column is the mechanism. An
unnormalised router puts 7.4x the mean variance in one direction and lets the
bottom four collapse to 4-13% of it -- the correlated cigar -- and names 247
of 4096 buckets. Centring removes part of that anisotropy, whitening removes
essentially all of it, and the resulting logits are isotropic, which means the
exported transform is near-identity. So whitening is not a correction applied
to a bad spectrum at export; it is a gradient-path pressure that stops the bad
spectrum forming. The "it folds into the rule matrix and the engine pays
nothing" claim now has a mechanism, not just the algebra.

**And the flip2 cost is not noise.** I(B;game) rises 1.68 -> 3.78 -> 5.65
across the same three arms, so the extra buckets carry position information.
Whitening's +18 points of flip2 is the honest price of a router using 12
near-independent bits, every one of which can change when a piece moves. No
transform on the logits removes it; only a training term on the bits can, and
that is `persist`, which at 0.1 already costs more than the table is worth
(6104 steps: val 0.029268 against the 0.028562 pure-PSQT floor, flip2 31.1%).

**Instrument notes, both of which invalidate earlier readings.**
- The `% vs none` column printed by `arm.py` uses a hardcoded 6104-step
  reference (val 0.025805, flip2 42.2%, `nnue/losslab/arm.py:410`). Every
  300-step arm reports "+13%" against it. That number is the step-count gap,
  not an effect. Only val differences BETWEEN arms at equal steps are readable.
- The 300-step norm screen quoted on 2026-09-02 (`none` 131 buckets / eff 93 /
  flip2 54.9%) is in no sweep record and is not reproducible under these flags;
  `none` here is 247 / 54.9 / 49.8%. Whitening's evenness gain over `none` is
  **56x**, not 33x.
- flip2 varies by **0.8 points at FIXED SEED** on this config (GPU
  nondeterminism; two back-to-back identical runs). val reproduces to five
  digits. Any flip2 difference under ~2 points at 300 steps is unreadable.

**Harness.** `--router-eigfloor` is now plumbed CLI-to-module (it existed in
`LogitNorm` and nothing passed it). `LogitNorm` tracks the covariance in every
mode, and a new `stat` mode returns the logits untouched while tracking them,
so any arm can report its spectrum; `stat` reproduces `none` to six digits of
val and exactly on eff/occ, which is what makes the table above readable.

---
