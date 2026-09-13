# 001 — Minimax pathology: is it real, and what fixes it

**Date:** 2026-08-24 · **Code:** `examples/pathology.rs` · **Status:** probe done, direction picked

## Setup
Depth-10, b=3 uniform tree. Leaf values generated hierarchically; `leaf_frac` = share of
total variance injected at the last level (1.0 -> iid leaves = Nau's setup; 0.1 = uniform
per level, chess-like). True node values = exact backup. A depth-K search backs up K plies
using a noisy static eval at the frontier: `eval(n) = true(n) + sigma*N(0,1)`.
Paired design: all operators see the same trees and the same noise draws. 300 trials.

## Result 1 — pathology is governed by one ratio: sigma_eval / sibling_spread(K)

| leaf_frac | sigma | spread@K=1 | spread@K=10 | regret K=1 -> K=10 |
|---|---|---|---|---|
| 1.00 (iid) | 0.5 | 0.038 | 1.000 | 0.043 -> 0.043 (flat, = random) |
| 1.00 (iid) | 1.5 | 0.038 | 1.000 | 0.043 -> 0.043 (flat) |
| 0.10 | 0.5 | 0.355 | 0.978 | 0.088 -> 0.018 |
| 0.10 | 1.5 | 0.335 | 0.972 | 0.226 -> 0.030 |
| 0.10 | 3.0 | 0.349 | 0.979 | 0.349 -> 0.065 |

Chess is not pathological because sibling spread is LARGE, not because minimax is sound.
Prediction: positions with genuinely small sibling spread (closed/fortress/zugzwang) sit
in the bad regime. Testable once we have a search.

## Result 2 — "endless flipping" reproduced. P(root move changes from K-1 to K):

| setting | K=2 | K=5 | K=8 | K=10 |
|---|---|---|---|---|
| iid, sigma=1.5 | 0.640 | 0.647 | 0.647 | 0.680 |  <- never settles
| 0.10, sigma=0.5 | 0.437 | 0.310 | 0.223 | 0.183 |
| 0.10, sigma=3.0 | 0.677 | 0.617 | 0.490 | 0.320 |

PV instability is a direct readout of sigma/spread. (Stockfish already uses
`bestMoveChanges` for time management — it is measuring exactly this ratio.)

## Result 3 — backup operators (regret; lower better)

leaf_frac=0.10, sigma=1.5:

| K | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|
| max | .2056 | .1529 | .1317 | .0936 | .0690 | .0304 |
| soft tau=1 | .1861 | .1483 | **.0902** | **.0560** | **.0367** | .0345 |
| double | .2843 | .2930 | .2519 | .2564 | .2708 | .1893 |

* **Soft backup (softmax-weighted mean, tau=1) wins 35-47% at mid depth and high noise**,
  loses slightly at full depth / low noise. Exactly the bias-variance shape expected.
  Flip rate also collapses: 0.087 vs 0.253 at K=10 (sigma=1.5). Optimal tau should track
  sigma/spread -> a learnable quantity.
* **Double-estimator backup is clearly WORSE** (negative result). It removes the E[max] >
  max E bias but doubles selection variance, and at the root we care about argmax quality,
  not value calibration. It fixes the wrong thing here.
* soft tau=0.3 ~= max, as expected (sanity check passed).

## Result 4 — Beal drift / odd-even effect reproduced
Mean root score, leaf_frac=0.10 sigma=1.5, K=1..10:
`+0.684 -0.309 +0.831 -0.327 +0.699 -0.407 +0.647 -0.446 +0.675 -0.480`
Persistent, does not decay with depth, amplitude grows with sigma.

## The blocker, and what it implies
**Soft backup is incompatible with alpha-beta pruning.** A soft average needs every child's
value; alpha-beta's whole b^(d/2) advantage comes from cutting after one refutation. This
is precisely why MCTS averages and alpha-beta maxes — the backup operator determines
whether you can prune.

Narrowed proposal: **apply soft backup only at PV nodes**, which PVS searches with a full
window over all moves anyway. Near-zero cost, and PV nodes are the ones that determine the
root move. Secondary route: keep max backup, use a learned variance estimate to set
*pruning margins* (RFP/NMP/futility), which leaves alpha-beta intact.

## Caveats
Synthetic uniform tree; no alpha-beta, no transpositions, no terminal nodes / traps, b=3
fixed, 300 trials. Real trees have mates that anchor the backup. Treat magnitudes as
indicative, the sigma/spread mechanism as the durable finding.
