# 115 — cp-0007 on the CCRL scale: 3293 ± 29, with the ladder finally bracketing us

**Date:** 2026-09-21/22 · **Commit:** `8cfffd6` · **Net:** `m1-b1.nnue` (sha `ce1f2656`)
**Conditions:** 10+0.1, 200 games per anchor, concurrency 5, `lich.epd` (4000 lines),
32 MB hash, desk box idle. 1600 games total, **zero engine errors**.

## Result

| opponent | CCRL | score | diff | implied | interval |
|---|---|---|---|---|---|
| byte-knight 4.0.0 | 2859 ±17 | 86.2% | +319 | 3178 | [3105, 3251] |
| 4ku 5.1 | 3057 ±14 | 78.0% | +220 | 3277 | [3225, 3329] |
| Inanis 1.6.0 | 3087 ±13 | 74.2% | +184 | 3271 | [3224, 3318] |
| Simbelmyne 1.10.0 | 3239 ±11 | 58.6% | +61 | 3300 | [3272, 3327] (400 games) |
| Frozenight 6.0.0 | 3362 ±9 | 34.8% | −109 | 3253 | [3212, 3293] |
| Stash 37.0 | 3419 ±11 | 37.2% | −91 | 3328 | [3290, 3367] |
| Marvin 6.3.0 | 3458 ±10 | 33.0% | −123 | 3335 | [3298, 3372] |

**rating 3293 ± 29** (95%, scaled for anchor spread; ±15 unscaled).
χ²/dof = 3.83 over 7 opponents — the anchors still disagree, so the scaled
interval is the one to quote.

Three anchors were added for this run (Frozenight, Stash, Marvin), each cloned
at the exact CCRL-rated tag and built with native flags. They are the first
opponents on the ladder that are *stronger* than the engine.

## What it changes

**cp-0004 was 3246 ± 33 (LEDGER 088, four anchors, same TC and game count).
cp-0007 is +47.** The self-play ledger for the same three steps claims
+7.5 (cp-0005) + 61.1 (cp-0006) + 40.0 (cp-0007) = **+108.6**. Roughly half of
self-play Elo survived contact with foreign opponents. [FACT for both endpoints;
the two gauntlets share four anchors, so their errors are partly correlated and
the ±44 on the difference is conservative.]

**The anchor-strength bias is now resolved instead of extrapolated.** Fit the
implied rating against the anchor's own rating:

| ladder | slope | self-consistent point |
|---|---|---|
| 4 anchors, all below us (2859–3239) | **+0.327** | 3352 |
| 7 anchors, bracketing (2859–3458) | **+0.198** | **3294** |

With anchors on both sides the correction collapses onto the plain combine
(3294 vs 3293), which is the point of bracketing: the estimate stops depending
on which extrapolation you believe. The four-anchor fixed point of 3352 was
**+58 too high**, and every rating this project has quoted came off a ladder
whose members were all weaker than the engine. [INFERENCE — a linear fit on 7
points, but two independent estimators agreeing to 1 Elo is the strongest form
this argument can take here.]

**Largest residual:** we score better against Stash 3419 (37.2%) than against
Frozenight 3362 (34.8%) — a 75-point inversion between adjacent anchors, −54
against the fit. n=200 each, so ~1.5σ. Recorded, not chased.

## The instrument defect this run exposed (noted, not fixed)

Two gauntlet legs against Simbelmyne on 2026-09-21 returned 113–115 "engine
error" games out of 200 and had to be discarded. The mechanism, from the PGNs
and `src/matchplay.rs`:

1. Across **every match PGN on record — 10,210 games — there are exactly three
   distinct hang events** (`timed out waiting for output`). Everything else
   labelled "engine error" is fallout. The hang is the opponent's: round 80
   died at ply 48 with normal move times and our side reporting +9.99 the ply
   before, and the harness's grace is `clock + 15 s` floored at 30 s, so it is
   not a tight-timeout artefact. [FACT]
2. **The blast radius is ours.** Each worker in `matchplay.rs:866` spawns its
   opponent once and reuses it for every game, with no respawn after an error.
   One hang poisons that worker permanently — every later `send` hits a dead
   pipe. Worse, a poisoned worker's games fail *instantly*, so it wins the race
   on the shared job counter and eats nearly the whole remaining queue: one
   hang became 113 of the last 126 games rather than one fifth of them.
   [INFERENCE from the code, but it is the only thing that fits 113/126.]

A dead-end worth recording: the first hypothesis was that `Stdio::null()` on
the opponent's stderr caused it, because an instrumented build with
`Stdio::inherit()` ran clean. The control — repo binary, `Stdio::null()`,
identical everything else — also ran clean (+54.29 [+18.4, +91.4], 200 games).
**Stderr handling was never the variable**; the comparison that suggested it had
no manipulation in it. The two clean runs pool to the Simbelmyne row above.

At one hang per ~3400 games the expected cost is ~6% of legs, and re-running a
leg is 18 minutes. The fix — respawn on error, drop the affected pair from the
stats — is cheap but was not worth the hour before a rating existed. It belongs
in `ROADMAP.md`, not in front of the next measurement.

## Also fixed here

`tools/ladder.sh` resolved no net, so a gauntlet silently played whatever
`chess bench` would have found — on 2026-09-21 one did, returning 3047 on a
stale August `quad.nnue`. It now resolves `$CHESS_WDL` from `publish/net.sha256`
the way `tools/checkpoint.sh` does, hash-checks it, refuses to run without one,
and writes the net and its sha into the results file. A rating whose eval
nobody recorded is not reproducible.

## Files

- `.ladder/gauntlet-20260921-192054.tsv` — byte-knight, 4ku, Inanis (clean rows)
- `.ladder/gauntlet-20260921-233835.tsv` — Frozenight, Stash, Marvin
- Simbelmyne row pooled from the two clean 200-game runs of the stderr control
- `.ladder/VOID-wrong-net-gauntlet-20260921-180910.tsv` — the stale-net run, kept as the negative
