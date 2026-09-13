## 6. The NNUE head (`runs/deep.log`)

`acc(2 x width) -> act -> Linear(2w, h) -> crelu -> bucketed h x h -> crelu ->
bucketed h -> 1`, plus psqt skip.

| arm | params | val | vs quadratic |
|---|---|---|---|
| quadratic r=512 (reference) | 395,010 | 0.021096 | — |
| **deep 128x2 -> 32, crelu, x1** | **108,515** | **0.018644** | **−11.62%** |
| deep 128x2 -> 32, square, x1 | 108,515 | 0.019054 | −9.68% |
| deep 128x1 -> 32, crelu (no perspective) | 206,947 | 0.019271 | −8.65% |
| deep 128x2 -> 32, crelu, x material+count | 743,402 | **0.017410** | **−17.47%** |
| same, bucket curriculum (40% warm) | 743,402 | 0.017961 | −14.86% |

**The head beats the quadratic form with 3.6x fewer parameters.** For scale,
+50% parameters on the quadratic (r=512->768) buys −0.21%. This is
architecture, not capacity.

**The perspective tie is positively good here, not just neutral.** Checked by
parameter count rather than label: `128x2` is one 769x128 table read twice
(98,432) + 10,083 of head; `128x1` is a 769x256 table (196,864) + the identical
head. Same 256-wide accumulator into the same `Linear(256,32)`. The tied
version uses **half the table and is 3.36% better**. Plausible mechanism
(untested): tying doubles the gradient reaching each row, and the nonlinear
head has more to fit, so the constraint acts as regularisation.

**crelu beats square by 2.15%.** The entire quadratic line used `square`.

**The bucket curriculum lost, but the test was not fair to it.** +3.16% worse.
The curriculum arm spent 3,200 of 8,000 steps in bucket 0, so it got 4,800
steps of bucketed training against 8,000, and never closed the gap after
release. Honest statement: *at a fixed step budget, 40% warm on material+count
costs more than it buys*. And material+count (584 buckets, perplexity 67) is
not sparse, which is the case the idea was for. `shrink` tests `kings`
(4,096 buckets, perplexity 238) at warm=0.25.
