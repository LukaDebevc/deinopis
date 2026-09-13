## Tools built this session

| file | what |
|---|---|
| `nnue/rulestats.py` | scores a routing rule on balance / completeness / stability |
| `nnue/attic/refresh.py` | refresh rate per legal move AND per made move, via `chess moves`. SUPERSEDED by `treeprice.py`: it prices on game positions, and the tree's are 3 pieces thinner |
| `nnue/verify.py` | engine vs model, features + eval + bucket index; `CHESS_PFAMS` for version-4 nets |
| `nnue/treeprice.py` | prices a rule on the moves the search really makes, per ply across both accumulators |
| `src/movedump.rs`, `chess movedump` | samples real made moves out of the search: parent FEN + child FEN, plus a uniform-legal control from the same position |
| `src/main.rs` `chess expand` | every legal move of every given position, same format — the game-position control |
| `nnue/attic/phaseloss.py` | per-position loss binned by phase / piece count, for two or more models |
| `nnue/attic/power.py` | held-out loss reduction from bucketing a rule, without training it |
| `nnue/attic/rulesearch.py` | the 20-predicate pool scored on power and stability together |
| `nnue/attic/coarsen.py` | spectral coarsening of a rule from its transition graph |
| `nnue/attic/stagedrules.py` | Luka's staged prune, VOID: half-price, and merged across the lightest edge |
| `nnue/attic/merge2.py` | the same idea corrected: per-ply price across both accumulators, rarest cell joins its strongest connection |
| `nnue/attic/jointpower.py` | scores a joint rule against hand rules and a random-router control |
| `nnue/fen.py` | independent FEN -> feature implementation for tests |
| `nnue/test_features.py`, `test_round2.py` | second dumb implementation of every feature; ALL PASS |
| `nnue/attic/feature_stats.py`, `feature_cost.py` | visit perplexity, per-move update cost |
| `nnue/runs/summarise.py` | `% vs bar` from a log |
| `src/main.rs` `chess moves` | dumps legal moves per FEN as `from:to:flag` |
| `src/search.rs` `ThreadData::movemix` | counts made moves by class; `chess bench` prints the mix |
| scratchpad `merge.py`, `merge2.py`, `lukabuckets.py` | merge and router studies |
| scratchpad `gather.rs` | **built, NOT run** — see below |

`train.py` gained: `SideQuadratic`, `Perspective`, `PerspSplit`, `Antisym`,
`Deep`, pairwise-multiply reads, `Extras` (all extra input families), coarsened
king families (`kings16`/`kings11`/`kings4`), and experiments `sides`, `persp`,
`psqtbuckets`, `extras`, `deep`, `ranksweep`, `shrink`, `ratio`, `rankbuckets`,
`pairs`, `kingcoarse`.
