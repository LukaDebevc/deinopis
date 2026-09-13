## 9. Luka's proposed architecture — what the arithmetic says

```
(quadratic psqt -> 2x128, y) -> (relu(2x128) -> 64, z: bucketed(256->1))
  -> relu(bucketed(64->16)) -> (bucketed(16->1) + y + z)
```
router: `(bishop_light x bishop_dark x queen x knight x rook[0,1,2+] x
king_region)^2 x 2`

**The router is affordable — better than the declared count suggests.**
Implemented exactly (king regions `{abc},{fgh} x {12,34,56,78}` + `{de} x
{123,45,678}` = **11**, not 10, so 528^2 = 278,784 before the x2):

| family | declared | perplexity | train rows per effective bucket |
|---|---|---|---|
| **full** | 278,784 | **1,518** | 345,000 |
| minus king regions | 2,304 | 131 | 4.0M |
| minus knight | 69,696 | 794 | 660,000 |
| material (existing) | 576 | 66 | 7.9M |

Perplexity is stable across sample sizes (1,459 on 1M val, 1,518 on 19.7M
train), so it is not a sampling artefact. `z` is 256 params per bucket, so
~1,350 training rows per parameter. **Not data-starved.** 7.8% of positions
sit in buckets seeing under 10,000 rows.

**Compaction is free and large.** Only **19.6%** of declared buckets are ever
reached in 19.7M positions; 9,959 buckets cover 96.6% of positions, 2,344
cover 84.3%. The full stack is ~1,312 params/bucket, so **366M declared** but
**~26M compacted** to 20k live buckets.

**The quadratic feature transformer does not work, and not because of the 75M
parameters.** If `a_k = x^T W_k x`, moving a piece from s to t needs, for every
k, `dA_k = W_k[t,t] - W_k[s,s] + 2 sum_j (W_k[t,j] - W_k[s,j])`. That reads row
t and row s of all 128 matrices: 128 x 768 x 2 B = **196 KB per square
touched**, ~393 KB for a quiet move — a third of the *entire* current deployed
model, per move. At ~50 GB/s that is ~8 us against a budget of ~100 ns,
**roughly 80x too slow**. No layout change helps; it is the volume.

**The fix keeps the idea and costs nothing: factorise.** `a_k =
(v_k . x)(u_k . x)` — a product of two linear accumulator entries. 768 x 2m
parameters instead of 768^2 x m, an ordinary incremental accumulator, m
multiplies at read time. This is what the deployed quadratic already is in
factored form, and what Stockfish's pairwise-multiply layer does. Being tested
as `pairs` (section 10). Note `square` is only the diagonal case `a_k^2`.

**Two smaller things.** The `x2` for black/white-to-move is **not reachable**
from `all.data` — records are stm-canonical and absolute colour was never
stored — and section 2 argues it should be worth zero anyway. And the router
has **no pawn axis**, while `count x8` alone bought −7.7%.

**On "sparse MoE with built-in knowledge":** the analogy holds, with one
disanalogy — MoE routers are learned *and* load-balanced. This one is fixed and
chess supplies the load, which is skewed (top bucket 6.25%, 84% of positions in
2,344 of 278,784 slots). Perplexity is the honest count.

**On `A + buckets` (trainable shared dense layer + bucketed correction):**
purely additive shared+bucket is **exactly redundant** in function space —
`R_shared + R_b` absorbs into `R_b` — so any gain is optimisation, not
capacity. That is not a dismissal (LEDGER 017 measured exactly this being
worth a lot). It becomes genuinely non-redundant when the per-bucket part is
**constrained**: low-rank `A + u_b v_b^T` costs 4 x (256+64) = 1,280 params per
bucket at rank 4 instead of 16,384, so 1,024 buckets cost 1.3M instead of
16.8M, with ~16% more compute. Rare buckets then shrink toward `A`
automatically — the soft version of merging, with no offline analysis and no
training steps spent.

---
