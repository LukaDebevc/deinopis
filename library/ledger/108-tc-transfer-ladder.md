# 108 · The latent gap is not TC-invariant; 099's 2+0.02 decision is overturned

_2026-09-19. TC-transfer ladder on the GPU cluster, run dir
`<cluster scratch>`, driver `tcx.sh`, binary `chess-head`
(a873e77) and `chess-cp-0003` / `chess-cp-0004`, net `m1-b1`, conc 32,
hash 64, `nice 19`. 5 arms x 4 rungs, **147,000 games in 8.6 h**, zero engine
errors and zero forfeits in all 20 cells._

## The question

099 validated 2+0.02 against 8+0.08 on **one pair** (cp3 vs cp4) at n=300 and
explicitly refused 0.5+0.01 pending "shape evidence (multiple diverse pairs)".
This is that evidence. Arms span mechanism, not just effect size: `AA` (null),
`FUT` (`fut_max_depth=3`, depth-gated pruning), `EASY` (`easy_stable=0`, time
management), `SEE` (`c_see=0`, large pruning), `CP34` (cp-0003 vs cp-0004).

Cluster TCs are desk TCs x 1.436 (desk 863,357 nps native; cluster 601,000 nps
under 32-way concurrency). Ratio base/inc = 100 on every rung.

## Elo — the stretching ruler

🔴 **Every ± below is the harness's naive SE and is too small** — LEDGER 111 measures the design effect of the 43-line book on these exact
cells at 1.2–4.9 (nulls ~1.0). Multiply by ~2.1 at R1, 2.2 at R2, 1.5 at R3,
1.35 at R4 before using any of them to test a hypothesis.

| arm | R1 ~8+0.08 | R2 ~2+0.02 | R3 ~1+0.01 | R4 ~0.5+0.005 |
|---|---|---|---|---|
| AA | +1.62 ± 2.27 | −2.14 ± 1.92 | +3.27 ± 2.15 | −3.71 ± 2.36 |
| FUT | +14.25 ± 2.37 | +9.44 ± 2.00 | +10.80 ± 2.18 | +10.83 ± 2.40 |
| EASY | −2.03 ± 2.34 | +1.07 ± 1.95 | −2.29 ± 2.17 | −1.74 ± 2.40 |
| SEE | −8.80 ± 4.90 | −22.85 ± 4.02 | −30.65 ± 4.59 | −34.74 ± 5.09 |
| CP34 | +43.30 ± 4.91 | +72.85 ± 4.16 | +85.78 ± 4.66 | +104.24 ± 5.07 |

## Latent gap Δ (σ) — 099's instrument, pair-corrected

Probit fit `N(Δ,1)` cut at ±θ, per cell from W/D/L. SEs are bootstrap,
inflated so the implied Elo SE matches the harness's own pair-aware interval.

| arm | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| AA | +0.010 ± .014 | −0.011 ± .010 | +0.015 ± .010 | −0.016 ± .010 |
| FUT | **+0.090** ± .015 | +0.049 ± .010 | +0.050 ± .010 | +0.046 ± .010 |
| EASY | −0.013 ± .015 | +0.006 ± .010 | −0.011 ± .010 | −0.007 ± .010 |
| SEE | −0.053 ± .030 | −0.115 ± .020 | −0.137 ± .020 | −0.141 ± .021 |
| CP34 | **+0.269** ± .031 | +0.368 ± .021 | +0.385 ± .021 | +0.427 ± .021 |

θ falls 1.07 → 0.89 → 0.71 → 0.55 across the rungs and is stable **within**
each rung (AA/FUT/EASY agree to ~0.02, across R1's four hours) — so the node
was quiet and no rung's effective TC drifted.

## The anchor holds

- [FACT] R1 reproduces the desk on both arms that have a desk comparison:
  CP34 **Δ = +0.269 vs 099's +0.271**; FUT **Δ = +0.090 vs 104's h64 cell
  +0.082**. The ladder is anchored, so its shape is not a cluster artefact.
- [FACT] 099's ruler-stretch model is confirmed. Decomposing R1→R4 Elo change
  into stretch × gap change gives a stretch of **1.52 / 1.48 / 1.49** for
  CP34 / SEE / FUT — uniform, exactly as 099 said.

## What is new: Δ itself does not transfer

- 🔴 **CORRECTED 2026-09-20 by LEDGER 111 — this claim may not survive.** The
  SEs below are the harness's, which assume pairs are independent. They are
  not: the 43-line book repeats each opening ~140 times and the design effect
  on these very cells is **1.2–4.9** (R1-FUT 4.27, R1-CP34 1.99). Recomputing
  the R3/R1 Elo ratios with cluster-robust SEs cuts χ² by ~2.3x, which lands
  10.76 near **4.7, p ≈ 0.09 — not significant**. Re-run this test with honest
  SEs before relying on it. **The sign conclusion below is unaffected.**
- [FACT, SE understated] A single common stretch factor for Δ from R1→R2
  across the three non-null arms is **rejected: χ² = 10.76 on 2 dof,
  p ≈ 0.005**. FUT breaks it at −3.0σ. R1→R2 ratios are CP34 1.37x, SEE 2.17x, **FUT 0.54x** — no
  monotone distortion of the ruler moves three arms in two directions.
- [FACT] R2, R3 and R4 agree with each other for every arm (χ² 0.09–5.57 on
  2 dof). **The break is between the anchor and everything shorter**, not
  spread along the ladder.
- [FACT] Sign transferred in all 12 non-null cells.
- [INFERENCE] A short TC is a valid **screen** and not a valid **measuring
  stick**. The Elo conversion factor against 8+0.08 ranges 0.76x (FUT) to
  1.98x (CP34) and is not predictable from the mechanism class before
  measuring: FUT and SEE are both pruning changes and sit at opposite ends.

## Throughput

| rung | games/s | vs desk | h to SE 2.0 in 8+0.08-equivalent Elo |
|---|---|---|---|
| desk 8+0.08 | 0.27 | 1x | 8.9 |
| R1 | 1.27 | 4.7x | 1.7 |
| R3 (24k games) | 10.25 | 38x | **0.65** |
| R4 | 19.97 | 74x | — |

- [FACT] R3 beats R1 by ~2.7x in resolution per hour **even after paying the
  worst observed transfer penalty in full**, and beats the desk by ~14x.

## Decision

- [DECISION] **099's "TU-5 runs at 2+0.02, transfer evidence clean" is
  overturned.** That rested on Δ = 0.29 vs 0.27 at n=300, SE ≈ 0.09, which
  could not have seen this. At n=3000 the same comparison reads **+0.368 ±
  0.021 vs +0.269 ± 0.031, a 0.099 ± 0.038 gap (2.6σ)**. 099 is corrected at
  the source.
- [DECISION] **TU-5 campaign A is blocked**, parked at pair 362. Its SPSA
  objective is match results at 2+0.02, and FUT is a worked example of a
  search parameter worth 1.9x more at 8+0.08 than at 2+0.02 — the class it
  is tuning. Resuming it optimises the wrong objective.
- [DECISION] **θ is a standing instrument check.** Record it with every match.
  At R3 it must read 0.70 ± 0.05; off that means the node was loaded and the
  effective TC drifted, so the run does not pool with anything.
- The ladder screened **no new ideas** — all five arms were already-known
  quantities. It bought the ruler, not Elo.
