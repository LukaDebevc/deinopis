# 116 — Time management survives the 1x confirm: cp-0008

**Date:** 2026-09-22 · **Commit:** `4104862` (tag `cp-0008`) · **Net:** `m1-b1.nnue` (sha `ce1f2656`)
**Change:** `tm_moves_to_go` 30 → 24, `tm_inc_pct` 75 → 90 (two defaults in `search.rs`, nothing else).

## The question

LEDGER 114 measured this pair at **+18.20 [+12.8, +23.6]**, n=6000, R1 (8+0.08)
— but on a box running three streams at once, and its own A/A null on that box
read **−4.69 [−11.0, +1.6]**. Time management is the knob most exposed to
contention there is: every arm's think time is measured in wall clock, and wall
clock is biased −19% under load (LEDGER 105). So the +18.2 could have been an
artefact of one arm handling a loaded box better than the other, which measures
the box, not the engine.

The confirm had to be a **single stream on an idle desk box**, same TC.

## Result

`tools/checkpoint.sh`, 8+0.08, concurrency 5, SPRT H0 +0 / H1 +10, nothing else
on the machine. **H1 accepted at 508 games in 31 minutes:**

```
508 games: +114 =316 -78   +24.7 Elo  [+9, +41]   LOS 99.9%   LLR 2.95
```

**The contention confound did not eat the effect.** [FACT] The 1x interval
contains 114's 3x point estimate, so the two measurements agree and there is no
evidence the load was doing the work.

**Quote +18.2, not +24.7.** The gate stops when the LLR crosses, so its point
estimate sits at a boundary the walk was climbing and overstates. 114's
n=6000 fixed-n number is the unbiased one. No fixed-n rerun was played here:
it would sharpen the magnitude and change no decision.

Bench is **151742**, identical to cp-0007 — the right answer for a change that
cannot move a fixed-depth node count, and the check that the two arms really
differed only in time management. Perft suite exact, `cargo test` clean.

## Status

**cp-0008 is the current checkpoint.** It has not been gauntleted; the
3293 ± 29 of LEDGER 115 is cp-0007's number and cp-0008 sits above it by
roughly +18 of self-play Elo, of which history says about half transfers.
