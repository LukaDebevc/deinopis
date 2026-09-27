# 091 · An eval cache for the tuner: 38–96% hit rates, ~1.4–3.4x, 3.7x ceiling

_2026-09-15. Probe, not a feature. A wrapper evaluator (`ProbeEval`, patch in
`library/017-long-run-map/probes/probe-eval-tune.patch`) sits around
`DefaultEval` inside `tune::search_once`, forwards every call unchanged, and
records (a) every key that reaches `evaluate` and (b) how many incremental
accumulator updates a **lazy** accumulator would need (update only when an
eval actually misses, walking back to the last computed ancestor). Built in a
scratch copy of the working tree (cp-0004 behaviour, dormant flags). Corpus
`fishpack-1k-see.labels`, 15k nodes/position, m1-b1 net, 6 threads._

## Instrument checks

- Probe binary vs real binary, same pass: regret 37.17 ± 1.86, depth 8.11,
  nodes 11071 on both. The wrapper does not move the tree.
- Pass replayed against its own key set: **100.0%** hits, 0 lazy updates.

## Result

Pass A (defaults): 7,846,544 evals, 7,462,587 unique keys, 14,228,252
accumulator pushes. Only 4.9% of evals repeat a key seen earlier *in the same
pass*: the corpus positions share almost nothing. The cache has to come
from earlier passes.

| pass B (cache = keys of pass A) | eval hit rate | lazy updates / pushes |
|---|---|---|
| no cache (lazy accumulator only) | 0 | **66.1%** |
| `c_hist` 100→150 | **96.3%** | 2.7% |
| `c_see` 4000→4500 | 87.3% | 8.9% |
| `nmp_base_reduction` 3→4 | 79.9% | 14.2% |
| `rfp_margin` 75→85 | 49.5% | 35.2% |
| `c_rank` 250→281 (descent-sized step) | 37.7% | 43.5% |
| `c_rank` 250→375 | 32.2% | 47.7% |
| 4-knob simultaneous step (SPSA shape) | 39.8% | 42.0% |
| same step, cache = union of 5 earlier passes | **58.4%** | 30.2% |
| `c_rank` 300, cache = union of 5 | 53.7% | 33.4% |

Knobs that reshape the budget everywhere (`c_rank`, RFP) change most of the
deep tree; knobs that act rarely leave it almost intact. The hit rate climbs as
the cache accumulates passes.

## Converting to speed

`chess nodeprof` on bench, same tree, pinned core 5, one run: eval **55.8%**
(q 45.6 + main 10.2), accumulator push **14.3%**, rest ~30%. A tune pass has
more evals per node (0.71 vs 0.51) and pushes (1.29 vs 0.96), which re-weights
to eval ~60%, push ~16%, rest ~23% — consistent with the measured 2.2 s pass.
Model `T = 0.23 + 0.60(1-h) + 0.16u + 0.04` (last term: a cache lookup, a
[GUESS] of ~70 ns per eval call):

| case | speed-up |
|---|---|
| lazy accumulator alone | 1.07x |
| SPSA-shaped step, 1 pass cached | 1.43x |
| same, 5 passes cached | 1.76x |
| `c_see`-type knob | 2.8x |
| `c_hist`-type knob | 3.4x |
| ceiling, every eval a hit | **3.7x** |

So "at least 4x" is above the ceiling: ~23% of a pass is search code a cache
cannot touch. Realistic for an optimiser is 1.4x rising toward ~2x as the
cache fills.

## Why no re-pricing is needed

The eval is a pure function of the position at default draw value (i16
accumulator is modular, so incremental == refresh exactly), and the tuner's
budgets count nodes or work *calls*. A cache changes wall time and nothing the
objective reads. Two conditions: the work meter must charge a full eval on a
hit, and the key must include net identity and any root-dependent eval
setting (draw value, contempt).

## Side result: a lazy accumulator for the engine

In a tune pass, **33.9% of accumulator pushes are never read** (9.40M lazy
updates for 14.23M pushes). With push at 14.3% of bench time that is ~4–5% of
search time before bookkeeping. Not measured in games or on bench; the tune
pass has a fresh TT per position, so bench's fraction may differ.

## Decision

- The tuner cache is worth building only where passes are many: iterative
  optimisers on the proxy, large corpora, the per-child oracle. It is not the
  bottleneck for single screens (a pass is 2–3 s). Map items EC-1..EC-6.
- Lazy accumulator goes on the speed track (map SP-1), priced by an SPRT.
