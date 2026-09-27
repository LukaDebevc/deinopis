# 109 · The SR-1 split is neutral at the anchor; futility is +12.3, not +4.6

_Gate resolved in LEDGER 110; the screen/confirm disagreement resolved in 111._

_2026-09-20. Run dir `<cluster scratch>`, driver `split.sh`.
One manipulation: `ced4727` (parent) vs `0e8ac04` (the SR-1 split), fut flag
**OFF in both**. Match driver `chess-head` and net `m1-b1`, conc 32, hash 64 —
identical harness to LEDGER 108, so θ and Elo pool with the ladder. Zero
errors, zero forfeits._

## Why this was run

104 pooled the gate (103, +0.9 over 4000) with two flag-on/off cells (+10.1,
+13.2) to get "futility is +4.6 [−0.2, +9.4], on the [0,5] bound". 108's
R1-FUT cell is that same h64 experiment at 6x the games and reads **+14.25 ±
2.37**. The four one-binary measurements of the flag are homogeneous
(χ² = 1.31 on 3 dof, p ≈ 0.73) and pool to **+12.28 ± 1.50**; adding the gate
pushes χ² to 12.47 on 4 dof (p ≈ 0.014), with the gate contributing 11.2 of it.

So the gate does not belong in that pool. 104's stated reason for pooling was
an nps control — the split is 0.33% from cp-0005 against a ±0.8% within-build
spread. **That rules out a speed difference, not a play difference**, and
bench-exactness on one position set is not behaviour-identity everywhere. The
untested possibility was that the split itself costs about −11.

## Result

Bench re-verified first: **226828 nodes, identical between `ced4727` and
`0e8ac04`** (only the nps line differs). So the split is node-exact.

| cell | TC | n | Elo | Δ (σ) | θ | draws |
|---|---|---|---|---|---|---|
| R3 screen | 1.4+0.014 | 12000 | **+6.08 [+1.81, +10.35]** | +0.028 ± .010 | 0.700 | 51.6% |
| R1 confirm | 11.5+0.115 | 6000 | **−1.22 [−5.74, +3.31]** | −0.008 ± .015 | 1.071 | 71.6% |

- [FACT] **The split is neutral at the anchor: −1.22 ± 2.31.** The −11
  hypothesis is dead. 104's conclusion that the split is free was right, and
  is now supported on play and not only on speed.
- [FACT] θ reads 0.700 at R3 and 1.071 at R1 against ladder references of
  0.703–0.708 and 1.048–1.069. The TC did not drift; both cells are
  comparable to 108.

## The gate is still unreconciled

All three on the 8+0.08 anchor:

| | Elo | SE |
|---|---|---|
| flag alone, pooled n=20000 | +12.28 | 1.50 |
| split alone, R1 n=6000 | −1.22 | 2.31 |
| **⇒ predicted gate (split + flag)** | **+11.06** | 2.75 |
| observed gate 103, n=4000 | +0.90 | 3.06 |
| **discrepancy** | **+10.16** | 4.12 (**2.5σ**) |

- 🔴 **RESOLVED 2026-09-20 by LEDGER 110.** The gate was re-run at R1 outside
  `tools/checkpoint.sh`: **+15.36 [+10.71, +20.00]**, 6,000 games. That is
  1.2σ from the +11.06 predicted here and **3.7σ from 103's +0.90**. The 3.7σ
  holds under LEDGER 111 because both measure the same change on the same
  book at the same TC, so the opening effects cancel in the difference. The
  suspicion below is confirmed: it was the gate's own machinery.
- [INFERENCE] The components do not add up to the gate. Hash is excluded
  (104's h32 cell reads +10.1). The split is excluded (above). What remains
  untested is the gate **itself** — it was run by `tools/checkpoint.sh` during
  the window in which the exit-code bug of 103 was live, and that bug is known
  to have mis-handled one result already.
- [GUESS] The gate's own machinery is the prime suspect, not any engine
  change. Not established; 2.5σ is also just 2.5σ.

## The screen and the confirm disagreed

- 🔴 **RESOLVED 2026-09-20 by LEDGER 111.** Both SEs were too small: the
  design effect of the 43-line book is **1.30 on this R3 cell and 2.35 on this
  R1 cell**. Honest, the screen is +6.08 ± 2.49 and the confirm −1.22 ± 3.54,
  a difference of **+7.30 ± 4.33 = 1.69σ** — ordinary noise. No "sign fails to
  transfer" is required, and the extra 2 h of games proposed below is no
  longer the way to settle it; fixing the book is.
- [FACT, SE understated] The R3 screen read **+6.08, LOS 99.7%** where the R1
  anchor reads null. Difference **+7.30 ± 3.17 (2.3σ)** in Elo on naive SEs.
- [INFERENCE] This was the first use of screen-then-confirm and it flagged
  immediately — correctly, as it turns out, but not for the reason assumed.
  The instrument was miscalibrated, not the ruler. Neither "the screen's LOS
  is untrustworthy" nor "sign fails to transfer" is needed to explain it.
- **Screen-then-confirm survives.** What does not survive is quoting either
  rung's printed interval without the book correction of LEDGER 111.

## Decision

- **`fut_max_depth=3` is worth +12.3 ± 1.5 at 8+0.08, not +4.6.** It clears
  the [0,5] bound on its own and does not need a bundle partner. 104 is
  corrected at the source.
- **Done 2026-09-20:** the deliverable — tree(flag on) vs the cp-0005 binary
  — was re-run at R1 and reads **+15.36 ± 4.9 honest** (LEDGER 110). cp-0006
  is unblocked, and must be tagged **through** `tools/checkpoint.sh` rather
  than around it, since that script is the accused and a clean pass is also
  its acquittal.
