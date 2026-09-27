# 111 · The 43-line book understates every SE by up to 2.2x, and the AA null cannot see it

_2026-09-20. No new games: re-analysis of the surviving PGNs of LEDGER 108
(`<cluster scratch>`, 20 cells) and 109
(`<cluster scratch>`, 2 cells). Scripts `bookclust.py`,
`dupgames.py`, `deffall.py` in the same directory._

## The mechanism

`matchplay.rs` assigns opening `book[i % 43]` to pair `i`, so a 12,000-game
cell replays each of the 43 built-in lines ~140 times. The harness computes
`SE = sqrt(var(pair scores) / n_pairs)` — correct for colour, which the pair
already differences out, and **wrong for openings**, which it assumes are
independent draws.

They are not, whenever the change under test is worth different amounts in
different openings. Write a pair score as `mu + a_g + e` with `Var(a) = tau^2`
between openings and `Var(e) = sigma^2` within. With `N` openings repeated
`m = n/N` times the true variance of the mean is `tau^2/N + sigma^2/n`, but the
harness reports `(tau^2 + sigma^2)/n`. The ratio is the design effect,
`deff = 1 + (m-1)*rho` with `rho = tau^2/(tau^2+sigma^2)`.

**A null match has `tau^2 = 0` by construction** — identical binaries cannot
have an opening-specific effect — so `AA` reads `deff ~ 1` however bad the
book is. That is why θ and the AA controls have never fired on this.

## Result

`deff` measured per cell as `Var_cluster / Var_naive`, where the cluster
estimate treats each opening's mean as one observation. Every cell also
reports a **shuffled-label control**: pairs regrouped at random, which must
read ~1 or the estimator rather than the book is what is being measured.

| cell | n pairs | Elo | SE naive | SE honest | deff | ICC | shuffled |
|---|---|---|---|---|---|---|---|
| R1-AA | 3000 | +1.62 | 2.26 | 1.83 | 0.65 | −0.005 | 0.78 |
| R2-AA | 6000 | −2.14 | 1.92 | 2.25 | 1.37 | +0.003 | 1.01 |
| R3-AA | 6000 | +3.27 | 2.15 | 1.77 | 0.67 | −0.002 | 0.76 |
| R4-AA | 6000 | −3.71 | 2.36 | 2.22 | 0.88 | −0.001 | 1.02 |
| **R1-FUT** | 3000 | +14.25 | 2.36 | **4.88** | **4.27** | +0.047 | 1.05 |
| **R2-FUT** | 6000 | +9.44 | 2.00 | **4.44** | **4.92** | +0.028 | 0.89 |
| R3-FUT | 6000 | +10.80 | 2.18 | 3.27 | 2.26 | +0.009 | 1.30 |
| R4-FUT | 6000 | +10.83 | 2.40 | 3.24 | 1.83 | +0.006 | 0.72 |
| R1-EASY | 3000 | −2.03 | 2.33 | 3.70 | 2.51 | +0.022 | 1.34 |
| R2-EASY | 6000 | +1.07 | 1.95 | 2.98 | 2.34 | +0.010 | 0.63 |
| R3-EASY | 6000 | −2.29 | 2.17 | 2.76 | 1.62 | +0.005 | 1.48 |
| R4-EASY | 6000 | −1.74 | 2.40 | 2.24 | 0.87 | −0.001 | 0.86 |
| R1-SEE | 750 | −8.80 | 4.86 | 5.93 | 1.49 | +0.030 | 0.94 |
| R2-SEE | 1500 | −22.85 | 3.99 | 5.41 | 1.84 | +0.025 | 0.78 |
| R3-SEE | 1500 | −30.65 | 4.55 | 6.45 | 2.01 | +0.030 | 1.45 |
| R4-SEE | 1500 | −34.74 | 5.04 | 5.55 | 1.22 | +0.006 | 1.65 |
| R1-CP34 | 750 | +43.30 | 4.79 | 6.76 | 1.99 | +0.061 | 0.99 |
| R2-CP34 | 1500 | +72.85 | 3.97 | 5.77 | 2.11 | +0.033 | 0.78 |
| R3-CP34 | 1500 | +85.78 | 4.38 | 5.12 | 1.36 | +0.011 | 1.16 |
| R4-CP34 | 1500 | +104.24 | 4.63 | 4.51 | 0.95 | −0.002 | 0.90 |
| 109 split R3 | 6000 | +6.08 | 2.18 | 2.49 | 1.30 | +0.002 | 1.00 |
| 109 split R1 | 3000 | −1.22 | 2.31 | 3.54 | 2.35 | +0.019 | 0.96 |

- [FACT] **All four AA nulls read `deff` ≤ 1.37. Every cell with a real change
  reads 1.2–4.9.** Shuffled controls span 0.63–1.65, mean ≈ 1.0.
- [FACT] **`rho` rises monotonically as the TC slows** — FUT reads 0.006 →
  0.009 → 0.028 → 0.047 across R4→R1; CP34 −0.002 → 0.011 → 0.033 → 0.061.
- [INFERENCE] Mechanism: at slow TC the random component collapses (draws
  51% → 71%) while the opening's systematic contribution does not, so it
  becomes a larger share of a smaller variance. **The damage is worst exactly
  at R1, the confirm rung — the only rung allowed into LEDGER and
  CHECKPOINTS.**
- [FACT] It is **not** duplicate games. 98.3% of games are unique at both R3
  and R1. The one exception is its own bug: `book[15]`, the French Advance,
  produced the same 24-ply repetition draw **146 times out of 280** and drew
  80%. Fixed in `de938c0`, chosen by measurement over four candidate lines.

## What this corrects

- **109's screen/confirm disagreement mostly dissolves.** +6.08 ± 2.49 against
  −1.22 ± 3.54 is **1.69σ**, not the 2.3σ recorded. No "sign fails to
  transfer" is needed. Corrected at the source in 109.
- **108's single-stretch-factor test is at risk.** Recomputing the R3/R1 Elo
  ratios (FUT +0.76, SEE +3.48, CP34 +1.98):

  ```
  naive    common ratio 1.26   chi2(2 dof) = 16.61   p = 0.0002
  cluster  common ratio 1.42   chi2(2 dof) =  7.23   p = 0.0270
  ```

  [INFERENCE] This does not reproduce 108's own χ² = 10.76 (it used the Δ
  scale), so take the **factor**, not the absolute: honest SEs cut χ² by
  ~2.3x, which lands 10.76 near 4.7, **p ≈ 0.09 — no longer significant**.
  "Magnitude does not transfer" may be an artifact of the book. The *sign*
  conclusion is untouched. 108's own script has to settle it. Flagged in 108.
- **LEDGER 110's screens**: A09 `rfp_margin` drops from 3.7σ to 2.8σ and out
  of the pre-registered confirm band. The other three winners survive easily.
- **A02_GATE's +15.36 ± 2.37 becomes ± ~4.9.** Its 3.7σ against 103's +0.90
  survives untouched: same change, same book, same TC, so the opening effects
  are common and cancel in the difference.

## Decision

- 🔴 **Widen every SE in this repo that was measured on the 43-line book and
  carries a real effect.** Nulls are unaffected. The rule of thumb: R1 ×2.1,
  R2 ×2.2, R3 ×1.5, R4 ×1.35.
- 🔴 **Distinguish the two questions.** "Is this change worth more than zero?"
  needs the honest SE. "Do these two measurements of the same change on the
  same book agree?" does not — the opening effects cancel.
- **The fix is a book with more lines than the match has pairs.** `deff ≤ 1.1`
  needs `N ≥ ~1500` at both rungs. EPD/FEN book support shipped in `9de99c6`
  because every book worth using is distributed that way.
- **Which book is an open measurement, not an argument.** Bake-off running:
  FUT at R1 and R3, builtin-43 vs `lich.epd` (4000 lines), same binary, judged
  on cluster-robust SE per hour. Nothing on disk is both large and balanced —
  `Pohl.epd` reads median **+149 cp** on our eval at depth 10 (n=400), i.e.
  UHO-class, which 096 rejected. Sourcing a large *balanced* book is open.
- 🔴 **Never delete a match PGN again.** The 110 batch deleted 30 of them and
  therefore cannot be corrected, only estimated.
