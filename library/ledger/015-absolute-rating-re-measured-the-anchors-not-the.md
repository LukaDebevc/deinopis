**015 · Absolute rating re-measured; the anchors, not the games, are the limit** · 2026-08-25
Full ladder gauntlet on the current build (price list + 008), 200 games per
opponent at 10+0.1, five CCRL-rated anchors. Tantabus and Blunder were re-run on
an idle box after the first pass overlapped a parameter sweep; the numbers below
are the all-clean set.

| opponent | CCRL | score | implied |
|---|---|---|---|
| BBC 1.1 | 2019 | 95.0% | 2530 |
| Goldfish 2.1.1 | 2252 | 86.0% | 2567 |
| Cinnamon 2.4 | 2326 | 73.2% | 2501 |
| Tantabus 2.0.0 | 2555 | 41.5% | 2495 |
| Blunder 8.5.5 | 2664 | 44.2% | 2624 |

**Result: about 2554 ± 56 on the CCRL Blitz scale**, against **2551 ± 48** for
the pre-price-list build under the same combiner. **Δ = +3.** The whole
budget-search rebuild — LMR, LMP and the 64x64 reduction table deleted and
replaced by the price list — is **Elo-neutral end to end**, as the two
head-to-head SPRTs already implied (-12.7 then +10.4).
**The error is anchor-limited, not game-limited.** chi2/dof = **4.98**: Tantabus
says 2495 [2447, 2544] and Blunder says 2624 [2582, 2666] and those intervals
are **disjoint**. There is no trend with opponent strength (2530, 2567, 2501,
2495, 2624 is scatter, not a slope), so it is per-opponent style mismatch, which
a single rating cannot represent. Doubling to 400 games per anchor would move
the statistical term from ±25 to ±18 and leave ±56 essentially unchanged.
**Stop buying games for the absolute number** — only more or better-matched
anchors would help.
**Contamination check — negative.** Tantabus moved -38.4 -> -59.6 and Blunder
-81.4 -> -40.1 between the loaded and the idle pass: **opposite signs**, both
inside the ±54 that a difference of two 200-game matches carries. Sharing the
box produced no detectable bias, and no game in any PGN was lost on time. The
process rule still stands (a timed measurement owns the machine), but the worry
was unfounded as a matter of fact. Useful calibration in passing: **two
nominally identical 200-game matches differed by 21 and 41 Elo.** A 200-game
match is worth about ±38; a 30-Elo difference from one is not a result.
**Tool fix:** `tools/ladder.sh` reported only the unscaled statistical error and
merely *warned* above chi2/dof 2.5. It now always scales by sqrt(chi2/dof),
PDG-style, and prints both. Re-running the 24 Aug data through the fixed
combiner turns 2551 ±23 into 2551 ±48 — LEDGER 004's hand-written ±60 was
closer to right than the tool was.
**Methodological consequence:** the gauntlet moved +3 ± 74 across a change the
paired SPRTs resolved to ±20. It is **~4x less sensitive to a delta** than a
direct SPRT, because five separate matches against five opponents carry five
uncancelled systematic offsets. Use the gauntlet for *where we are*, never for
*did this patch help*. That is what `tools/checkpoint.sh` is for.

---
