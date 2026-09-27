# 114 · The 3x night: TM transfers, `rfp_margin` is refuted, and cp-0006 is +61 not +37

_2026-09-21. Three concurrent streams on the cluster, ~198 engine processes
where 65 had been the norm: `20260921-hunt/` (21 knob cells + 3 nulls),
`20260921-tmconfirm/` (B0–B3), `20260921-dm/` (C0–C5). All on net `m1-b1`
(sha `ce1f2656`), book `lich.epd` 4000 lines, conc 32, hash 64, `nice 19`.
R3 = `1+0.01`, R1 = `8+0.08`. Every cell ran to its full n; nothing was
truncated by the 08:00 deadline. `delta_margin`, the headline of the night,
has its own entry — see 113._

## Three corrections to things on record

**1. cp-0006 is +61.1 over cp-0005, not +37.4.** `tools/checkpoint.sh:54`
carried a hardcoded net default that had gone stale, so cp-0006's 448-game
gate played `gate-3ep-q8.nnue` on *both* sides. That is a valid comparison of
the search bundle, but at an eval ~58 Elo worse than the one we deploy, and it
is the number that went into CHECKPOINTS.md. Cell C4 is the measurement that
was never made: the assembled cp-0006 against cp-0005, both on the shipping
net, fixed n, same TC/conc/hash as the original gate, with the net as the only
manipulation.

**+61.08 [+55.39, +66.79]** over 6,000 games at 8+0.08.

CHECKPOINTS.md and STATE.md corrected at the source. The fix to
`checkpoint.sh` resolves the net from `publish/net.sha256` and hash-checks it,
so a gate can no longer name one eval and play another.

Note this cuts *against* the standing stopping-rule discount: the 448-game
SPRT read +37.4 and STATE.md advised reading it as "+20 to +30". The honest
number is nearly double the discounted one. The discount was right in
principle and swamped in practice by the net confound.

**2. `rfp_margin` 75→40 is refuted.** LEDGER 110 screened it at +8.0 (2.8σ)
and STATE.md carried it as a bundling candidate. Cell H11 at 12,000 games on
the 4000-line book reads **−1.16 [−5.84, +3.53]**. The +8.0 was a 43-line-book
artefact (111): inflated SE on a clustered design that no A/A null could see.
Dropped, not demoted.

**3. The R3→R1 transfer ratio is ~2.0x, not 2–3x.** Four measurements now:

| change | R3 | R1 | ratio |
|---|---|---|---|
| `tm_moves_to_go=24` | +21.68 | +14.25 | 1.52 |
| `tm_inc_pct=90` | +14.57 | +7.59 | 1.92 |
| `delta_margin=400` | +49.02 | +24.83 | 1.97 |
| `delta_margin=1200` | +84.06 | +40.02 | 2.10 |

The old 2–3x was measured on the 43-line book, where R1 was the worst-hit rung
(111). With 4000 lines the ratio tightens and the rung ordering stops mattering
as much.

## Controls: the book fix shows up in the nulls

Four A/A nulls at R3, 12,000 games each, spread across all three streams:

| cell | Elo |
|---|---|
| H00_AA (opens the hunt) | +1.27 [−3.42, +5.97] |
| H50_AA_mid | +4.29 [−0.28, +8.85] |
| H99_AA_end | −0.61 [−5.14, +3.93] |
| C3_AA (stream C) | −2.78 [−7.49, +1.93] |

Mean +0.54, spread (SD 3.05) about **1.3x** the nominal per-cell SE of ~2.35.
That is the honest overdispersion factor at R3 on the 4000-line book, against
1.5x on the 43-line book at the same rung (111). The book bake-off's promise
(112) holds up under a 3x-loaded box.

**The R1 null is the one to watch.** `B0_AA` at 8+0.08, n=4000, reads
**−4.69 [−11.02, +1.64]** — 1.45σ, not significant, but the only R1 null of
the night and it sits negative. If it is real, every R1 number from these
streams is understated by ~5 Elo, which is the conservative direction. Not
corrected for anywhere; recorded so it is not rediscovered.

## Time management: it transfers, and the feared ambiguity did not happen

The screens (H01–H05) ran at 1x load; the confirms (B0–B3) at 3x. Both TC and
box load changed between screen and confirm, on the knob most sensitive to
contention there is. The pre-registered worry was that a small confirm — say
+4 against a +15 screen — would leave "does not transfer to 8+0.08" and "3x
load ate it" both live, with no way to separate them tonight.

It did not happen. R1, 6,000 games each:

| cell | Elo |
|---|---|
| B1 `tm_moves_to_go=24` | **+14.25** [+8.92, +19.59] |
| B2 `tm_inc_pct=90` | +7.59 [+2.42, +12.75] |
| B3 both | **+18.20** [+12.76, +23.64] |

Sub-additive (14.25 + 7.59 = 21.8 → 18.20) but clearly positive, and the
transfer ratios sit in line with everything else in the table above. No 1x
re-run is needed to interpret the sign or the rough size.

It is still **not in cp-0007**. The confound is narrowed, not closed: these ran
at 3x with a null at −4.69, and bundling them would put that inside the
delta_margin result. TM gets a 1x confirm and its own checkpoint.

## The rest of the fan: cp-0006 is at a local optimum

21 cells, 12,000 games each, cp-0006 against itself one `--set` apart. Beyond
`delta_margin` (113) and the TM pair, nothing moved:

| cell | Elo | reading |
|---|---|---|
| H20_GIVCHK | **−13.76** [−18.50, −9.03] | live and wrong-signed |
| H12_NMP1 (`nmp_min_depth=1`) | −7.09 [−11.87, −2.32] | cp-0006's 2 is right |
| H14_ASP12 | −4.57 [−9.36, +0.21] | null |
| H10_CSEE3K / H09_CSEE1K | −3.68 / −3.13 | **`c_see=2000` is a real optimum** |
| H19_INCHK | −3.53 [−8.24, +1.18] | null |
| H15_ASPG300 | −3.16 [−7.93, +1.61] | null |
| H16_QMAX32 | −2.90 [−7.62, +1.83] | null |
| H04_EASYOFF | −2.78 [−7.44, +1.88] | easy-move is not hurting |
| H13_COFFPVN | −2.72 [−7.51, +2.07] | null |
| H18_FUT5 / H17_FUT2 | −2.32 / +2.17 | **`fut_max_depth=3` is right** |
| H11_RFP40 | −1.16 [−5.84, +3.53] | refuted, see above |
| H03_GATE65 | +2.26 [−2.28, +6.80] | null |

Three of cp-0006's five bundled changes are confirmed as local optima by their
own neighbours (`c_see` from both sides, `fut_max_depth` from both sides,
`nmp_min_depth` from below). `c_offpv` was already known live and wrong-signed
(110) and is untouched here.

The useful summary: **after cp-0006 the classical knob surface is flat
everywhere except the one place nobody had pushed far enough.** 21 cells to
find that out is cheap; the cost of not knowing is a year of tuning a surface
with no gradient on it.
