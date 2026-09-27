# 101 · SR-2 singular (se_min_depth=8): −2.5 over 3000, inconclusive, not shipped

_2026-09-17/18. P4 SPRT [0,5] @8+0.08 conc 5, cap 3000, hash 64, builtin
book. Same frozen binary both arms (`~/chess-runs/20260917-sr2/`, cp-0005
tree + draw-12 rule, bench 219718, sha-verified post-reboot): A
`--set se_min_depth=8` (variant a: min depth 8, margin 3cp/ply
`se_margin`, slack 3 — both defaults) vs B base. m1-b1 both sides.
The match ran through the night to cap; the box was shut down after it
finished, so no resume was needed._

## Result

| games | score (A) | Elo | LLR |
|---|---|---|---|
| 3000 | +448 =2082 −470 | **−2.5 [−9, +4]** | inconclusive at cap |

PGN re-score agrees to the digit (−2.55 [−9.0, +3.9], pairs=1500).
Zero engine errors, zero forfeits.

## Reading

- [FACT] The early +17 lean at ~200 games (LLR +0.48) regressed to −2.5 —
  the third time this project watched a small-sample lead evaporate
  (089's fade, the ladder anchor's +94→+48). Nothing to learn beyond the
  standing rule: no hand-reads before the bound or the cap.
- [FACT] Proxy-blind as predicted (2/4000 positions differ at 15k nodes):
  the rule fires only at game depths. The SPRT was the instrument and it
  says "not ±5-decisive at 8+0.08".
- [INFERENCE] −2.5 ± 6.5 excludes neither a small win nor a small loss.
  This is MS-3 territory ([0,5] cannot ship ±3 effects), not a kill of
  the mechanism: variants b–d (double extension, negative extension /
  multi-cut) are untested, and the map's LTC fallback (flat at STC →
  rerun 20+0.2) was written for this case.

## Decision

- **Not shipped.** `se_min_depth` stays 0 (dormant, bench-exact).
- No LTC rerun now: TU-5 and SR-1 own the box queue. The variant line in
  the map stands as written for a quiet-box future.
