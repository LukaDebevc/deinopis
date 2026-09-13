## The price was measured on the wrong positions

Luka: *"we have to account for how search works. move ordering, alpha beta
pruning... highly possible that king moves aren't evaluated often because of
the pruning. i think that it might be best to spin up engine and see what are
moves that actually get evaluated."*

Right that the estimator was wrong. Wrong about which way, and the way it is
wrong is more interesting than the guess.

Up to here, a rule was priced by taking a **game** position, enumerating its
legal moves, and reweighting the per-move rates by the tree's move-CLASS mix
(69% quiet / 30% capture / 1% promo, from `chess bench`). A class mix cannot
express move ordering, so `chess movedump` samples the real thing: parent FEN
and child FEN for one in every 150 made moves, 1000 real game roots at 60k
nodes each, 56.6M made moves. `--rand` writes a control from the *same*
sampled positions with a move drawn uniformly from the legal list, which is
what separates the two effects. `chess expand` does the same on game positions.

| moved piece | game pos, legal | tree node, legal | tree node, played |
|---|---|---|---|
| pawn | 16.65% | 18.54% | 18.48% |
| knight | 10.78% | 8.35% | 9.05% |
| bishop | 16.90% | 15.43% | 15.98% |
| rook | 28.41% | 21.71% | 21.94% |
| queen | 15.80% | 14.02% | 14.37% |
| **king** | **11.45%** | **21.96%** | **20.17%** |

Ordering does suppress king moves — and by **8%**, 21.96 to 20.17. The effect
that matters is the one nobody was looking for: **tree nodes carry 16.6 pieces
against a game position's 19.4**, so the king is a much larger share of a
shorter legal list. Getting into the tree nearly doubles the king-move rate;
ordering then shaves a twelfth off it.

Prices per ply across both accumulators, on the real stream, with the two
effects separated (`x pos` = tree vs game positions at uniform legal moves,
`x order` = the tree's choice vs uniform at the same positions):

| rule | buckets | on record | tree, real | x pos | x order |
|---|---|---|---|---|---|
| castle state 4^2 | 16 | — | 6.41% | 1.85 | 0.94 |
| count x8 | 8 | — | 12.88% | 0.85 | 3.86 |
| king region 11^2 | 121 | — | 15.32% | 2.23 | 0.97 |
| phase x8 | 8 | — | 17.23% | 2.35 | 4.39 |
| **material 24^2** | 576 | 8.00% | **22.55%** | 2.46 | 4.36 |
| **merged K32xM64->256** | 256 | 7.79% | **24.31%** | 2.10 | 1.58 |
| merged K64xM128->512 | 512 | 9.40% | 28.69% | 1.92 | 1.52 |
| shelter (castle x cover)^2 | 256 | — | 32.12% | 1.36 | 1.00 |
| **kings 64^2 (HalfKP)** | 4,096 | 16.20% | **40.34%** | 1.92 | 0.92 |

The `x order` column is the whole argument in one column. Rules that read
material sit at **4.3** — the tree plays captures far out of proportion, which
is what the class reweighting was for, and it only ever caught part of it.
Rules that read the king sit at **0.92-0.97** — ordering makes them slightly
cheaper, exactly as Luka predicted, and it is a rounding error next to `x pos`.

Not a bug in the new code path. HalfKP's bucket changes exactly when a king
moves, so its price per ply must be `2 x P(king move)`. Uniform-legal gives
22.99% king moves; the pricer says 45.99%. Exact. Both FENs come from the
engine, so nothing in the pricer reimplements en passant, castling or
promotion — which is where the last pricing bug lived.

**What this costs us.** Every price on record is 2-3x low, so the budget has to
be re-derived in these units. The hand-rule ranking survives (HalfKP against
material 24^2 goes from 2.03 to 1.72), but the designed rule does not:
**`merged256` flips from cheaper than material 24^2 to more expensive**, 7.79
vs 8.00 becoming 24.31 vs 22.55. The merge search was minimising a cut computed
on the wrong distribution, and some of its apparent cheapness was that.

**Known gap, unmeasured.** A made move is not an accumulator update. A real
engine updates lazily, so a child pruned before `evaluate()` is ever called
never pays for its accumulator. That scales every number here down, and not by
a constant — quiescence children almost always stand-pat, main-search children
often do not. Measuring it means hooking the eval call sites rather than the
make-move sites, which `src/evalstats.rs` already has the shape for.
