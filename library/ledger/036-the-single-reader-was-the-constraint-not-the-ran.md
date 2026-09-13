## 036 — The single reader was the constraint, not the rank (`runs/rankbuckets.log`)

**Setup.** `persp` found r=512 -> 1024 worth only −0.19%, and every arm there had
ONE reader. If the reader is what binds, extra rank has nothing to do and the
flat curve says nothing about rank. Two curves over the same ranks, differing
only in read conditioning. 8000 steps, lr 1e-3, K = 258.7, warm-started.

| rank | x1 (one reader) | x584 (material+count) |
|---|---|---|
| 128 | 0.021581 | 0.019181 |
| 256 | 0.021201 | 0.018313 |
| 512 | 0.021091 | 0.017690 |
| 768 | 0.021058 | 0.017444 |
| 1024 | 0.021054 | **0.017344** |
| 128 -> 1024 | **−2.4%** | **−9.6%** |

**Result.** Unbucketed, the curve is flat — r=768 to r=1024 buys 0.02%.
Bucketed, the same step buys **0.57%**, and the whole range buys four times as
much. The single reader was the binding constraint.

**Mechanism, and a correction to the comment in `train.py`.** That comment
called 768 a hard algebraic ceiling: with `act=square` and one reader the model
is `x^T (V^T diag(R) V) x`, and every symmetric 768x768 W decomposes in 768
terms. True — for ONE bucket, and the x1 column confirms it. It does not extend
to many: the buckets share one V, each bucket's `W_b = V^T diag(R_b) V` is still
768x768, but V has to serve 584 different diagonal reweightings at once, so more
than 768 shared directions is expressible. r=1024 winning only in the bucketed
column is exactly that, not over-parameterisation.

**0.017344 is the best number in the study.** Deploying it needs the
accumulator; see 037.
