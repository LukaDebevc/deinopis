# 102 · SR-1 futility skip-quiets (fut_max_depth=3): +24.6 over 806, H1

_2026-09-18. P4 SPRT [0,5] @8+0.08 conc 5, cap 3000, hash 64, builtin
book. Same frozen binary both arms (`~/chess-runs/20260917-sr1/`, split
commit `0e8ac04`, bench 219718): A `--set fut_max_depth=3` (pure
skip-quiets form after the 100 split; margin stays 100) vs B base.
m1-b1 both sides._

## Result

| games | score (A) | Elo | LLR |
|---|---|---|---|
| 806 | +138 =587 −81 | **+24.6 [+13, +37]** | 2.96 (bounds ±2.94) |

H1 accepted, 2896 s wall. PGN re-score +25.2 [+13, +37] over 816 games
(5 in-flight pairs) — agrees. Zero engine errors, zero forfeits.

## Reading

- [FACT] Proxy said −0.3 ± 0.7 (100): neutral, 881/4000 differ. Games say
  +24.6. The proxy is blind to this mechanism (quiet-move pruning shifts
  which subtrees get searched — a second-order allocation effect at 15k
  nodes, first-order at game depths). Regret may reject; games promote —
  and did.
- [FACT] This is the pure form. The as-bundled arm (whole-node gate
  included) read +10.4 worse on proxy; the split earned its keep before
  a single game was played.
- [INFERENCE] +24.6 carries the usual H1-stopping upward bias (096/097).
  The gate re-measures it; quote the gate, not this.
- [FACT] Ship vehicle: flip `fut_max_depth` default 0→3 (margin 100
  unchanged). Bench moves (skip-quiets changes node counts); persist and
  easy-move precedents say node-limited tools stay valid. `fut_whole_node`
  stays 0 — the killed form rides dormant, unchanged.

## Decision

- **Ship candidate.** Flip the default, record the new bench fingerprint,
  run `tools/checkpoint.sh` vs cp-0005. This SPRT is not the gate.
