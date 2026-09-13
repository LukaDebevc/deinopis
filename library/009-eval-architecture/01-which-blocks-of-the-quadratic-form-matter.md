## 1. Which blocks of the quadratic form matter (`runs/sides.log`)

`eval = x^T W x`; split `x` into ours/theirs and ask which blocks are needed.

| arm | val | vs bar | share of the quadratic gain |
|---|---|---|---|
| **tied** (deployed: one accumulator, one reader) | **0.021085** | — | 100% |
| all three blocks, untied coefficients | 0.021133 | +0.23% | 99.5% |
| same-side only (ours x ours + theirs x theirs) | 0.025715 | +21.96% | **54.3%** |
| cross only (ours x theirs) | 0.026257 | +24.53% | **49.0%** |
| ours x ours only | 0.028942 | +37.26% | 22.5% |
| theirs x theirs only | 0.028614 | +35.71% | 25.7% |
| tied, orientation scrambled | 0.021998 | +4.33% | — |

**Both sides matter and split almost exactly evenly** — 54.3% same-side,
49.0% cross, summing to 103%, so the two are additive to within 3%. Untying
the block coefficients is *worse* (+0.23%), so the single shared accumulator
is the right shape.

Orientation scrambling costs +4.33%, an **upper bound** on what abandoning the
stm-canonical frame would cost (that arm sees random orientations, so it also
halves effective data per frame).

Unexplained: `theirs` beats `ours` by 1.55%, which is 15x the noise floor.
