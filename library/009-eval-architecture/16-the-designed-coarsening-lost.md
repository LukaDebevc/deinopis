## The designed coarsening lost (`runs/merged.log`)

The test the machinery was built for. Six arms, identical conditions, 8000
steps at lr 1e-3, warm-started from the big PSQT checkpoint. Prices are the
honest ones from the section above; the arms were *selected* on the old ones,
shown in brackets.

| arm | val loss | vs base | price/ply | (as selected) |
|---|---|---|---|---|
| base, no buckets | 0.021075 | — | 0 | 0 |
| **material 24^2** | **0.018264** | **-13.34%** | **22.55%** | (8.00%) |
| merged K32xM64->256 | 0.018583 | -11.83% | 24.31% | (7.79%) |
| merged K64xM128->512 | 0.018553 | -11.97% | 28.69% | (9.40%) |
| merged K64xM128->1024 | 0.018545 | -12.00% | 31.35% | (11.14%) |
| kings 64^2 (HalfKP) | 0.019871 | -5.71% | 40.34% | (16.20%) |

**The hand rule dominates on both axes** — lower loss and lower price than
every designed rule. `merged256` carries 184.6 effective buckets against
material 24^2's ~62 and still fits worse.

And the merged curve is **flat**: 256 to 1024 buys 0.2% of loss for 29% more
price. So the effective bucket count was never the binding constraint. Which
coordinate you read is.

### Why it lost

`merge2.merge` implements half the theory. The criterion was supposed to be

    merge the connected pair maximising   lambda * 2 T_ab  -  dP_ab

— buy cut reduction, pay in predictive power, sweep lambda for the frontier.
Only the `T` half was ever written: the rarest live cell joins its strongest
connection. That is pure cut minimisation. Nothing in it can prefer a
coordinate that matters.

The obvious guess for what goes wrong — captures are the heaviest edges, so a
merge that joins across heavy edges should destroy the material coordinate — is
**wrong**. Measured as `I(group; axis) / H(axis)` on tree nodes:

| rule | reads kings | reads material |
|---|---|---|
| merged K32xM64->256 | 56.5% | 68.3% |
| merged K64xM128->512 | 62.0% | 73.8% |
| merged K64xM128->1024 | 68.2% | 77.9% |
| material 24^2 | 39.2% | 84.0% |
| kings 64^2 (HalfKP) | 100.0% | 46.2% |
| king region 11^2 | 55.6% | 25.1% |

The merge keeps 68.3% of the material coordinate. What it does is *split*: it
spends roughly as much capacity on the king as on material. The hand rule
spends almost everything on material. King routing saturates near -5.8%
(HalfKP: -5.71% for 4,096 buckets, at the highest price on the board), so
capacity spent on the king is close to burnt.

That is a cost-only optimiser behaving exactly as specified. It was never told
that one axis is worth more than the other.

### What survives

One specific, cheap claim: put `dP` back in the criterion. `power.py` already
computes it additively over cells, which is what makes the sweep tractable. If
a power-aware merge still cannot beat a rule a human wrote in one line, learned
routing on this axis is finished.
