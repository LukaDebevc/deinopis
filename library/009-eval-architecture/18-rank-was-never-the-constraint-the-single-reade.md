## Rank was never the constraint; the single reader was

`persp` measured r=512 -> 1024 at −0.19% and it was read as rank saturation.
Every arm there had one reader.

| rank | x1 | x584 (material+count) |
|---|---|---|
| 128 | 0.021581 | 0.019181 |
| 512 | 0.021091 | 0.017690 |
| 768 | 0.021058 | 0.017444 |
| 1024 | 0.021054 | **0.017344** |
| 128 -> 1024 | −2.4% | **−9.6%** |

The algebraic ceiling argument — `act=square` with one reader gives
`x^T (V^T diag(R) V) x`, and a symmetric 768x768 W needs at most 768 terms —
is correct for one bucket and **does not extend to many**. The buckets share
one V; each `W_b = V^T diag(R_b) V` is still 768x768, but V must serve 584
different diagonal reweightings at once, so more than 768 shared directions is
expressible. r=1024 wins only in the bucketed column, which is exactly that.
