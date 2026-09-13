# 085 · Correction history phase 1 killed by hand: 0.0 Elo over 586 games

_2026-09-12. library/015 phase 1: per-thread pawn-key table (`corr[stm][key
& 16383]`), read as `corr_eval` in the RFP and NMP gates only, updated from
`best_score − static_eval` at main-search non-check nodes. SPRT [0,5]
@8+0.08 conc 5, same frozen binary both arms, A `--corr pawn` vs B base,
m1-b1 net both sides, builtin book. Bench 199091 off / 174057 on
(−12.6% nodes: it does prune more). Stopped by hand at LLR −0.20
(inconclusive by the letter), log + PGN `/tmp/corr/`._

## Result

| games | score | Elo | LLR |
|---|---|---|---|
| 586 | +88 =410 −88 | **−0.0 [−15, +15]** | −0.20 |

Zero engine errors. PGN re-score identical (pairs=293). Trajectory faded
steadily: +25 at 166g → +13 at 340g → 0.0 at 586g; the early lead was noise
(width ±30), not a lead lost.

## Why it reads zero while pruning 12% more

On time, fewer nodes per depth should buy depth. Net zero means the depth
bought is worth exactly the accuracy lost — the table's "learning" prunes
good and bad alike. Two suspects, in order: (i) entries never converge —
16k entries per side learned within one search and cleared after, a handful
of visits each, so `corr_eval` is mostly noise; (ii) the RFP/NMP gates are
not the binding constraint, so even true residuals would not move the game.
No diagnostic separates them yet, and building one costs box time against a
mechanism whose upper edge (+15) already excludes the 20–40 prior.

## Decision

- **Phase 1 killed. Variants (exact-only, quiet-gated updates) NOT run:**
  each is another ~600-game sink to rescue a mechanism reading exactly zero
  with a weakened story (less volume → less convergence, not more).
- **Phase 2 (stand-pat use) stays gated** behind 015's own rule — phase 1
  was the mechanism proof and it failed. Do not build it on these residuals.
- Code stays dormant in the tree (bench-exact, one relaxed load per node);
  the pawn key serves the next attempt. If correction is ever revisited, the
  first step is a convergence diagnostic (entry visit counts / residual
  variance within one search), not another SPRT — measure whether the table
  learns anything before pricing what it learned.
- Box pivots to `c0=200`, the only live proxy hint (−1.36 ± 1.63, 084).
