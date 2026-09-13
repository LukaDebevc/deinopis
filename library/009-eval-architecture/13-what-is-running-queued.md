## What is running / queued

Chain (`run_night2.sh` .. `run_night8.sh`, each waiting on the previous via
`pgrep`). Done: `kingcoarse` (section 7), `ranksweep`, `extras2`.

| run | what | status |
|---|---|---|
| `ranksweep` | r = 256/512/768/1024/2048 unbucketed | **done**, see below |
| `extras2` | pawnrow, kingzone, flanks, centre, randcell control | done 15:48 |
| `shrink` | `kings` at warm=0.25, curriculum variants | running |
| `ratio` | `PerspSplit` allocation sweep. **Prediction on record: flat** | queued |
| `rankbuckets` | rank x bucketing. **Prediction: unbucketed flat past 256, bucketed keeps falling** | queued |
| `pairs` | pairwise-multiply reads | queued |
| `vbuck` | **the decisive one**: conditioning the accumulator itself | queued |
| `rules` | graph vs hand coarsenings; our factorisation vs HalfKP under 7% | queued |

`ranksweep` result, and it confirms the prediction for the unbucketed side:
0.021203 / 0.021082 / 0.021059 / 0.021056 / 0.021032 for r = 256 / 512 / 768 /
1024 / 2048. Flat past 512 -- 0.24% over a 4x rank increase. r = 2048 cannot
add capacity (`V diag(R) V^T` is still a symmetric 768x768 W), so the 0.13%
above r = 768 is optimisation, not function class.

`vbuck` arms: for each of material 24^2 and kings 11^2, read+pre at r=512;
read+pre at r=528 unbucketed (width control); read+pre at r=512 plus 16
**bucketed** accumulator entries. Plus two random-router arms (hash of the
position, same bucket count and table size, no chess in it) because the
bucketed arm has 8x the parameters and could win on capacity alone. A null
result is the decisive one: it would mean no rule belongs in front of the
accumulator, every rule goes behind it for free, and the entire refresh study
prices a cost with no benefit.

`rules` arms: material 24^2 (anchor), kings 11^2 (hand), kings graph-121,
kings graph-64, kings 64^2 HalfKP, kings11 + material, kgraph64 + material,
HalfKP + material. Note that two families in `Bucketed` are ADDITIVE readers,
not a cross. **Luka's prediction on record: graph-121 beats hand-121, because
it has more effective differentiation (perplexity 26 against 17).**

Not on the GPU: the `stagedrules.py` sweep (500k positions, seven
configurations) was still building transition graphs when this was written.
