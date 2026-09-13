**016 · A quadratic form on piece-square occupancy beats any PSQT by 23%** · 2026-08-27
The eval bet: `eval = x^T W x + b^T x`, x the 768-dim piece-square occupancy
vector. W symmetric factors as `sum_k lam_k (v_k . x)^2`, so the model is a
768 -> r -> 1 network with a *square* activation — full rank at r = 768,
a rank-r approximation below it.
**Data:** Leela T79 May-2022, Stockfish-rescored (`linrock/test79` binpack),
250M positions after the standard filter (54.0% of 463M raw entries pass).
Records are stm-canonical: board mirrored and colours swapped so the side to
move is always White at the bottom. Validation is a contiguous 1M **game-level**
tail — a random position split would leak, since every position in a game
carries one result label.
**Loss:** `(sigmoid(f/K) - [lam*sigmoid(s/K) + (1-lam)*z])^2`, lam = 0.9.
**K fitted by maximum likelihood on this data = 258.7 cp**, not the 400 that
circulates in NNUE pipelines. K sets where the sigmoid spends the model's
capacity, so this is not a cosmetic difference.
**Result** (8000 steps x 65536 = 524M positions, ~2 epochs, AdamW + cosine,
lr 1e-3, output scale = K):

| arm | params | val loss |
|---|---|---|
| PSQT (best possible model linear in x) | 50,257 | 0.031451 |
| quadratic r=64 | 50,050 | 0.024076 |
| ReLU h=64 | 49,281 | 0.023571 |
| quadratic r=256 | 197,890 | 0.022417 |
| ReLU h=256 | 197,121 | 0.021500 |

**Decision:** proceed. The quadratic form is a **23% loss reduction over the
best PSQT** at matched parameters. The degree-2 ceiling is real but modest —
ReLU leads by 2.1% at width 64 and 4.1% at 256, and the gap *widens* with
capacity, which is what composition being the missing ingredient looks like.
**Not yet Elo.** This is a proxy and CLAUDE.md's rule applies: nothing here is
a result until it plays games.

---
