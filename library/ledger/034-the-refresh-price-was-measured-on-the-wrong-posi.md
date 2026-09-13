## 034 — The refresh price was measured on the wrong positions (`chess movedump`)

**Setup.** Everything up to 031 priced a routing rule on *game* positions: take
a position, enumerate its legal moves, and reweight the per-move rates by the
tree's move-CLASS mix (69% quiet / 30% capture / 1% promo, from `chess bench`).
Luka's objection: move ordering and pruning decide which moves the search
actually plays, and a class mix cannot express that — king moves are
history-poor and ordered late, so HalfKP might be priced too high.

New tool `chess movedump` (`--features movedump`) samples real made moves out
of the search and writes parent FEN + child FEN, so the pricing never has to
reimplement en passant, castling or promotion. `--rand` adds the control that
separates the two effects: the SAME sampled position, one move drawn uniformly
from the legal list. `chess expand` does the same for game positions.
`nnue/treeprice.py` prices any rule on either stream, per ply across both
accumulators. n = 1000 real game roots at 60k nodes, 42.6M nodes, 56.6M made
moves, sampled 1-in-150.

**Result — the objection is right, the mechanism is not.**

| moved piece | game pos, legal | tree node, legal | tree node, played |
|---|---|---|---|
| king | 11.45% | 21.96% | **20.17%** |
| rook | 28.41% | 21.71% | 21.94% |
| pawn | 16.65% | 18.54% | 18.48% |

Move ordering costs king moves 8% of their share (21.96 -> 20.17). Moving from
game positions into the tree nearly **doubles** it, because tree nodes are
thinner: **16.6 pieces against 19.4**. Fewer pieces, larger king share of the
legal list. The position effect is ~2x and was missed entirely; the ordering
effect is ~0.9x for king rules and ~4.3x for material rules, and the class
reweighting only ever caught part of the latter.

Per ply, both accumulators, on the real stream:

| rule | on record | tree, real | x positions | x ordering |
|---|---|---|---|---|
| material 24^2 | 8.00% | **22.55%** | 2.46 | 4.36 |
| merged K32xM64->256 | 7.79% | **24.31%** | 2.10 | 1.58 |
| kings 64^2 (HalfKP) | 16.20% | **40.34%** | 1.92 | 0.92 |
| king region 11^2 | — | 15.32% | 2.23 | 0.97 |
| count x8 | — | 12.88% | 0.85 | 3.86 |

Not a bug in the new path: HalfKP's bucket changes exactly when a king moves,
so its price must be `2 x P(king move)`. Uniform-legal gives 22.99% king moves
and the pricer says 45.99%. Exact.

**Consequences.** (a) Every price on record is ~2-3x low, so the budget must be
re-derived in these units — the "7%" was already void, and the scale is wrong
too. (b) The ranking of hand rules survives (HalfKP/material 2.03 -> 1.72), but
**`merged256` flips from cheaper than material 24^2 to more expensive**
(7.79 vs 8.00 becomes 24.31 vs 22.55). The merge search bought its cheapness
partly from the biased price.

**Known gap, unmeasured.** A made move is not an accumulator update: an engine
updates lazily, so a child pruned before `evaluate()` never pays. That scales
everything down, and not uniformly — quiescence children almost always
stand-pat. Measuring it means hooking the eval call sites, not the make-move
sites.

**Decision.** `nnue/attic/refresh.py`, `coarsen.py`, `stagedrules.py`, `rulesearch.py`
and `merge2.py` all price against the class-mix weights `W`; that estimator is
superseded. Price on the dump.
