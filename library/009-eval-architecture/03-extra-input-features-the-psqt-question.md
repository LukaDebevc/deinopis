## 3. Extra input features — the "psqt++" question (`runs/extras.log`)

Features that enter the accumulator, so the quadratic form multiplies them
against every piece. Bar 0.021086.

| family | val | vs bar | rebuild | pawn mv | knight mv | king mv | train pos/s |
|---|---|---|---|---|---|---|---|
| bar (768 only) | 0.021086 | — | 32 | 2.00 | 2.00 | 2.00 | 1.41M |
| +pawnpair | 0.020003 | −5.14% | 7 | 1.73 | **0.00** | **0.00** | — |
| +pawnfile | 0.020099 | −4.68% | 8 | 1.68 | **0.00** | **0.00** | — |
| +tile22 | 0.019963 | −5.33% | 98 | 6.44 | 6.85 | 4.27 | 0.30M |
| +tile23 | 0.020063 | −4.85% | 48 | 4.29 | 4.86 | 3.18 | 0.42M |
| +centre | 0.020781 | −1.45% | 2 | 0.42 | 0.49 | 0.00 | — |
| +kingpair | 0.020846 | −1.14% | 1 | 0.00 | 0.00 | 1.00 | — |
| **+pawnfile+pawnpair** | **0.019599** | **−7.05%** | 15 | 3.4 | 0.00 | 0.00 | 0.97M |
| + centre+kingpair too | 0.019208 | −8.91% | 18 | 3.8 | 0.49 | 1.00 | 0.87M |

Update costs measured over 1,012 pawn / 198 knight / 11 king quiet moves on 4
real FENs.

**`tile22` and `pawnpair` are 0.2% apart on loss but not close on cost** —
tile22 costs 6.85 gathers on a knight move against pawnpair's 0.00, where the
entire base feature set costs 2.00. Training throughput says the same thing
independently (0.30M vs 0.97M pos/s). **The tiles' information is real but it
is the same information the pawn files carry, and the pawn version is the one
that is incrementally free.**

`centre` and `kingpair` are the weakest alone (−1.45%, −1.14%) yet add another
−1.86% on top of the pawn pair for 3 gathers. What is weak alone is not what
is redundant.

**Caveat:** the four-family arm reports train 0.018946 below val 0.019208.
Every other arm has train above val, which is what unique-data training gives
(train is a window average including earlier, worse steps). That inversion is
the first overfitting signal in this series, at 0.59 epoch. It still wins on
val. A second seed would settle whether it is real or a val-slice accident.
