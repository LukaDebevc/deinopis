# 089 · Continuation history killed by hand: +3.2 over 2424 games

_2026-09-12/13. Full-form continuation history
(`conth[prev_piece][prev_to][piece][to]`, ~147k entries, same gravity update
as plain history, added to the history term for quiet moves; `--conth` /
`$CHESS_CONTH`, dormant by default, bench 199091 off / 219718 on). SPRT
[0,5] @8+0.08 conc 5, same frozen binary both arms, A `--conth` vs B base,
m1-b1 net both sides, builtin book. Stopped by hand before the 3000-game
cap; log + PGN `/tmp/sprt/s1-conth.{log,pgn}`._

## Result

| games | score | Elo | LLR |
|---|---|---|---|
| 2424 | +358 =1730 −336 | **+3.2 [−4, +10]** | 0.24 |

PGN re-score identical (pairs=1212). Zero engine errors.

Trajectory faded the whole way: +27 at 168g → +15 at 326g → +21 at 476g →
+15 at 630g → +14 at 788g → +10 at 940g → +10 at 1090g → +7 at 1250g →
+6.7 at 1400g → +4.7 at 1554g → +1.8 at 1704g (LLR −0.17) → +3.2 at 2424g.
The early lead was noise, the same shape as correction history's fade in
085 (+25 → +13 → 0.0).

## Reading

The +10–30 literature prior does not survive contact: the full form reads
~0–5 here, and the upper edge (+10) excludes nearly all of it. Two suspects,
both consistent with the earlier kills: (i) O1's SEE split already takes
the ordering that matters, so a better quiet order has little left to buy;
(ii) rank feeds the price list, so a "better" order also reprices the tree
against itself (ORDERING.md interaction warning — O5 shrank the tree 8.7%
and lost).

This also retires the ordering project as a whole for now: O1 shipped, and
O2–O12 plus the full continuation form are all dead. What is untried is
position-dependence from a refutation-aware teacher (per-child corpus),
which 014 already gates on.

## Decision

- **Killed.** `--conth` stays dormant in the tree (bench-exact, one relaxed
  load per quiet scored); the q-branch `path[ply].mv` store stays — it
  restores the "written on the way down" invariant and is behavior-neutral.
- Box moves to S2 easy-move time management.
