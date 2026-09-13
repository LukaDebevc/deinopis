## 035 — A designed coarsening loses to a one-line hand rule (`runs/merged.log`)

**Setup.** The decisive test of the whole rule-search direction: train the
coarsenings `merge2.py` designed, against the best hand rule, under identical
conditions. Six arms, 8000 steps, lr 1e-3, K = 258.7, block 2M, warm-started
from `ckpt-big/psqt_768-_64-_16-_1.pt`. Prices below are the honest ones from
034, per ply across both accumulators; the parenthesised ones are what the arms
were selected on.

| arm | val loss | vs base | price/ply (real) | (as selected) |
|---|---|---|---|---|
| base, no buckets | 0.021075 | — | 0 | 0 |
| **material 24^2** | **0.018264** | **-13.34%** | **22.55%** | 8.00% |
| merged K32xM64->256 | 0.018583 | -11.83% | 24.31% | 7.79% |
| merged K64xM128->512 | 0.018553 | -11.97% | 28.69% | 9.40% |
| merged K64xM128->1024 | 0.018545 | -12.00% | 31.35% | 11.14% |
| kings 64^2 (HalfKP) | 0.019871 | -5.71% | 40.34% | 16.20% |

**Result — negative, and clean. The hand rule dominates every designed rule on
both axes**: lower loss and lower price. `merged256` has 184.6 effective
buckets against material 24^2's ~62 and still fits worse. This is the arm that
was supposed to justify the machinery, and it does not.

The merged curve is **flat**: 256 -> 1024 moves the loss by 0.2% (0.018583 ->
0.018545) while the price goes 24.31% -> 31.35%. So the effective bucket count
is not the binding constraint — which coordinate you read is.

**Mechanism.** `merge2.merge` implements only half of the theory. The stated
criterion was `lambda * 2 T_ab - dP_ab`: buy cut reduction, pay in predictive
power. Only the `T` half was ever coded — the rarest live cell joins its
strongest connection, pure cut minimisation, no value term anywhere. So the
merge has no way to prefer a coordinate that matters.

Measured, on tree nodes, as `I(group; axis) / H(axis)`:

| rule | reads kings | reads material |
|---|---|---|
| merged K32xM64->256 | 56.5% | 68.3% |
| material 24^2 | 39.2% | 84.0% |
| kings 64^2 (HalfKP) | 100.0% | 46.2% |

The merge splits its buckets between the two coordinates roughly evenly. The
hand rule spends them almost entirely on material. King routing is already
known to saturate near -5.8% (HalfKP here: -5.71% for 4,096 buckets and the
highest price on the board), so every bucket the merge spends on the king is
close to wasted. *(A prediction that it would destroy the material coordinate,
since captures are the heaviest edges, is falsified: it keeps 68.3% of it.)*

**Decision.** The cut-only merge is dead; do not train another rule from it.
The untested claim is now specific and small: put `dP` back in the criterion
and see whether a power-aware merge beats `material 24^2` at a lower price. If
a one-line hand rule still wins, learned routing on this axis is finished and
the eval study should move to the deployable item (psqt x material+count).
