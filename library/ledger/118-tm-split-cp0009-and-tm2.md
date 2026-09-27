# 118 — The TM split holds at R1 (+59.3) and ships as cp-0009; a growth-predicted TM loses

**Date:** 2026-09-26/27 · **Commit:** `71be205` (tag `cp-0009`) · **Net:** `m1-b1.nnue` (sha `ce1f2656`)
· **Run dir:** `the GPU cluster:<cluster scratch>` (`cells.log`, PGNs, `s7`/`s8` + patches)
**Change shipped:** `tm_hard_pct` 100 → 500, `tm_instab` 0 → 100, `tm_fall` 0 → 20
(LEDGER 117), plus a TT fix (a qsearch Exact store no longer displaces a deeper
same-age entry; bench-exact, game-visible) and the hunt's dormant flags.

## The time manager, written out

`d = max(5, min(t/24 + 0.9·inc, t/3))` ms. Hard stop `H = max(d, min(5·d, t/3))`.
After completed iteration k, start the next only if elapsed ≤
`0.5·d · e_k · (1 + c_k) · (1 + min(1, 0.02·f_k))`, with `c_k = 0.5·c_{k−1} +
[best move changed]`, `f_k` = cp the root score fell, `e_k` = 0.7 on an easy
move. The gate therefore ranges 0.35·d to just under 3·d; the old code never
passed d.

## Results

All cells: same binary both sides, conc 32 on an idle cluster. A/A null
**−0.87 [−4.95, +3.21]** at R3.

| cell | A vs B | TC, n | Elo [95%] |
|---|---|---|---|
| **R1_HUNT_VSOLD** | hunt TM vs cp-0008 TM | 8+0.08, 6000 | **+59.3 [+54.4, +64.2]** |
| gate (`checkpoint.sh`) | cp-0009 vs cp-0008 | 8+0.08, SPRT [0,10], desk conc 5 | +44.9 [+22, +68], H1 at 296 games |
| E1 / R1_EASYOFF | easy move off vs on | R3 12k / R1 6k | −2.2 [−6.2, +1.8] / −3.65 [−8.5, +1.2] |
| M1 / M2 | + material moves-to-go ×0.5 / ×0.3 | R3, 12k | **+4.9** [+0.9, +9.0] / **−25.8** |
| T1 | TM2 (below) vs hunt TM | R3, 12k | **−82.0** |
| T2 / T3 | TM2, mtg fixed 24 / material ×0.5 | R3, 12k | −69.1 / −54.9 |
| T4 | TM2 vs cp-0008 TM | R3, 12k | −7.8 |

**Quote +59.3 (fixed n), not the gate's +44.9**, which stops at a boundary.
R3 → R1 shrink is **×1.33** (78.6 → 59.3), not the ~×2 ruler of 108/114. No
time forfeits in any file. The hunt TM spends 10.75 s/game vs 9.38 of ~12 s
available at R1 (1.39 vs 1.18 of ~1.5 at R3): part of the gain is clock
cp-0008 left unspent.

**Easy move:** no sign it turns negative at the longer TC; both TCs lean
(n.s.) towards keeping it.

**Material moves-to-go (×0.5 of 1/3/3/5/9, both sides, floor 10):** moves time
out of moves 1–10 into 10–40 and reads +4.9; ×0.3 does the reverse and reads
−25.8. Candidate only; needs an R1 run and a ×0.7 neighbour.

## TM2 — why the growth-predicted manager loses [negative result]

TM2 (`tm_mode=1`, dormant): target `T = (t + M·inc)/M`, M = material; start
the next iteration iff `elapsed + dt_last²/dt_prev ≤ min(1.5·T, t/2)`; hard stop
`min(4·T, t/2)`. With `tm2_mat=0` (M fixed at 24) it spends **exactly** the
hunt TM's clock — 1.389 vs 1.392 s/game, same per-move profile — and still
loses 69. So the loss is the stopping rule, not the budget.

Probe (156 positions, `go depth 15`, cost = nodes per iteration, 1,404
predictions of depth n+1 from n ≤ 14): a single growth ratio has log-sd 0.79.

| predictor of next iteration's cost | median error | p90 | under by >2x |
|---|---|---|---|
| `c_n²/c_{n−1}` (ratio clamped ≥ 1) | ×2.17 | ×8.2 | 25% |
| `c_n · 1.69` (mean ratio) | ×1.65 | ×3.9 | 17% |
| `(r−1) · elapsed` | **×1.54** | **×3.0** | 13% |

Extrapolating the last ratio is the worst predictor; one that averages all
history is the best — and "start while elapsed < g·d" **is** that predictor.
Under-predicting by >2x a quarter of the time means starting iterations the
hard stop then throws away. Caveats: fresh TT, fixed depth, book/bench
positions, nodes not time.

## Status

cp-0009 is current, bench 151742. Not gauntleted.
