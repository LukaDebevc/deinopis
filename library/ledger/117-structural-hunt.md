# 117 — Structural hunt: 45 dormant flags at R3, the time manager is the only big one

**Date:** 2026-09-26 · **Code:** dormant flags, committed in `cp-0009` (`71be205`)
· **Net:** `m1-b1.nnue` (sha `ce1f2656`) · **Run dir:** `the GPU cluster:<cluster scratch>`
(`cells.log`, `stage.log`, every PGN kept, binaries `s1`..`s6` + `s*.patch`)

## Setup

Every cell: R3 = 1+0.01, 12000 games, 4000-line `lich.epd`, hash 64, candidate
`--set` flags vs the same binary at defaults. Conc 96 until 08:52, 32 after.
`stager.py` ran pre-registered rounds: round 1 = one flag per cell; round 2 =
stack of the non-overlapping winners ≥ +4, a replication of each member, a
leave-one-out (LOO) of each, and a ±1 fan of each member's value; round 3 = the
final stack, six replicates. **Four A/A nulls: +1.51, +1.59, 0.00, +1.45**; the
six final replicates spread 93.7–98.4, matching their ±4.2 intervals.

## Result — the ones that live

| flag | round 1 | replication | LOO (stack without it) | fan |
|---|---|---|---|---|
| TM soft/hard split (`tm_hard_pct=250,tm_instab=100,tm_fall=20`) | **+68.7** | +68.4 | −73.6 | hard 125 +19.6 · **500 +78.6** · instab 50 +69.8 / 200 +51.0 · fall 10 +69.0 / 40 +60.6 |
| `iir_min_depth=4` | +7.8 | +7.4 | −11.8 | **3: +12.4** · 5: +6.3 |
| `hmc_scale=200` | +5.6 | +4.1 [−0.1, +8.3] | −4.1 | 100: +5.3 · 400: +0.3 |
| `nmp_eval_div=200` | +4.5 | **−1.4** | +2.2 | dropped — winner's curse, caught by the replication |

Hard stop alone: `tm_hard_pct` 200 **+41.4**, 300 **+43.7**. Instability alone
−0.9, falling score alone +2.9: the stretches only pay once the hard stop lets
the iteration they start actually finish. Mechanism [FACT, from the code]: an
iteration cut by the hard stop is discarded whole (`if self.stopped && depth > 1
{ break }` runs before `best` is updated), so with hard = soft every iteration
started between ~0.35·d and 0.5·d was mostly wasted time.

**Final stack** `tm_hard_pct=500,tm_instab=100,tm_fall=20,iir_min_depth=3,hmc_scale=200`:
**+95.7** mean of six replicates vs `s6`, +89.5 vs the cp-0008 binary. R3 only.
The fan winners (500, depth 3) are single max-of-fan cells and read high.

## Result — the ones that do not (all R3, n=12000, ±~4.2)

| flag | Elo |
|---|---|
| `--corr pawn` / bound-aware correction | **−33.4** / −29.3 (085's "neutral" at 586 games was blind) |
| LC-2 qsearch gap, five variants / `c_gap=2000` | −15.3 to **−94.0** / −24.2 |
| cut-node reduction 500 / 250 | −23.4 / −10.6 |
| not-improving 500 / neg, before the root fix / after | −18.8, −16.9 / −10.8, −9.2 |
| SE multicut, both signs | −14.1, −16.1 |
| `conth2` · `se_min_depth=8` · TT capture 500 · check ext 500 | −4.5 · −4.3 · −4.8 · −3.4 |
| razoring 2/3 · ProbCut · q-checks · recapture ext · capture-hist 4/8 · `c_conth` 6/12 · improving-RFP | null (within ±3) |

🔴 **`tt_eval_adj` 1 / 2 read −775 / −872 (11,748 of 12,000 lost). That is a
bug in the flag, not a measurement of the idea.** Do not cite it as one.

## Decision

- The TM split went to an R1 confirm and shipped as **cp-0009** (LEDGER 118).
- IIR (depth 3) and `hmc_scale=200` each need their own R1 run and gate.
- Every negative above stays dormant in the code, as a re-runnable record.
