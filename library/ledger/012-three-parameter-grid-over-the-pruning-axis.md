**012 · Three-parameter grid over the pruning axis** · 2026-08-25
150 points, `max_base` x `skip_below` x `c_rank`, paired against the default on
4000 freshly labelled positions at 60k student nodes, plus a 6-point extension
of `c_rank` past the grid edge. Depth is now reported by `tune compare` (it was
computed and discarded) and read as the constraint LEDGER 010 requires.
**Result:** the price list has one live shape parameter, one live threshold, and
one dead clamp.
- **`max_base` is saturated.** Values 0, 500, 1000 and 2000 give *bit-identical*
  results in almost every cell — same regret, same `differ` count. The ceiling
  stops binding somewhere between -1000 and 0. 008 moved it from -2000 to +1000
  and got the entire effect in the first quarter of that move; the rest was
  inert. Effectively 150 grid points bought 59 distinct experiments.
- **`skip_below` has an interior optimum and it is already there** (-128). Mean
  regret over the slice: -640 +1.28, -256 +1.15, **-128 +0.73**, 0 +2.78,
  +128 +7.24. Above it recall collapses; below it depth collapses (8.93 vs 9.63
  plies) because budget goes to hopeless moves. **First interior optimum this
  project has found in the pruning direction** — the boundary-solution failure
  `library/005` predicted does not occur on this axis.
- **`c_rank` is the whole story, and it too turns over** once pushed past the
  grid edge:

| `c_rank` | 0 | 125 | 250 (def) | 375 | 500 | 625 | 750 | 1000 | 1500 |
|---|---|---|---|---|---|---|---|---|---|
| regret vs def | +1.30 | +1.15 | 0 | **+0.18** | +0.97 | +0.95 | +1.86 | +3.21 | +5.68 |
| stderr | .53 | .49 | - | .53 | .54 | .57 | .57 | .59 | .62 |
| depth | 8.00 | 8.86 | 9.81 | **10.56** | 11.06 | 11.41 | 11.72 | 12.23 | 12.69 |

  Depth rises monotonically to +2.88 plies; regret is flat to `c_rank=625` and
  then separates at 3.3 sigma (750) and 5.4 sigma (1000). So the optimum is
  interior, around **375-500**, and the regret instrument *can* resolve it —
  but only once the arm is pushed ~2x past it.
**Decision:** `c_rank = 375` is the candidate — **+0.75 plies at +0.18 ± 0.53 cp**,
i.e. free depth on the one axis that shapes the tree. This is exactly the shape
008 had (depth bought, regret unable to see it), and 008 needed the SPRT to
confirm, so it goes through `tools/checkpoint.sh` before it is believed. Regret
alone is not a result. `max_base` should be pinned at 0 and dropped from every
future sweep; it is a clamp that no longer clamps.

---
