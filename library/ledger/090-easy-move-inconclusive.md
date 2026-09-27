# 090 · Easy-move time management: +4.1 over 3000, inconclusive, not shipped

_2026-09-13. Easy move (`easy_stable`, `easy_margin=15`, `easy_depth=8`,
`easy_frac=70` in `search::Params`): same best move with a stable score over
consecutive completed iterations past depth 8 → the next iteration starts
only below 0.35 of the deadline instead of 0.50. SPRT [0,5] @8+0.08 conc 5,
frozen `easy_stable=2` binary vs base (one-line default diff, bench 199091
both — TM cannot move a node-limited bench; timed smoke test clean), m1-b1
net both sides, builtin book. Ran to the 3000-game cap; log + PGN
`/tmp/sprt/s2-easy.{log,pgn}`._

## Result

| games | score | Elo | LLR |
|---|---|---|---|
| 3000 | +400 =2235 −365 | **+4.1 [−2, +10]** | 0.76, inconclusive |

PGN re-score identical (pairs=1500). Zero engine errors.

## Reading

Same verdict class as continuation history (089): small-positive,
unresolvable at [0,5] within 3000 games, not shippable. The mechanism is
safe (bestmove always legal, only the stop decision moves) so there is no
correctness reason to remove it — but a +4 effect is below what this
project ships, and the box time it would take to resolve (~6000+ games)
exceeds the prize.

## Decision

- **Not shipped.** Params stay dormant in the tree (`easy_stable=0`).
- If time management is revisited, the next form is PV-instability
  *extension* (spend more when the score jumps), not a bigger easy-move
  discount — this SPRT says the savings side is nearly empty.
- Box moves to S3 futility skip-quiets.
