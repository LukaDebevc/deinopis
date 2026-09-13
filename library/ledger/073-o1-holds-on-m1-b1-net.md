# 073 · O1 holds on the m1-b1 net: +21.8, H1 over 1164 games

_2026-09-09. `nnue/runs/order-b1-o1sprt-20260909-200039/`. A = O1 binary
(current tree: SEE-split, losing captures below killers) + m1-b1 net; B =
cp-0002 binary + m1-b1 net. Same net both sides, so this is the pure ordering
delta on the new net. TC 8+0.08, conc 5, hash 32, SPRT [0,5], frozen binaries
(`/tmp/order/b1-o1`, `/tmp/order/b1-base`). One pairing, sequential._

## Result

```
1164 games: +236 =765 -163   +21.8 Elo  [+11, +33]   LOS 100.0%   LLR 2.98
SPRT: H1 accepted
```

(A's perspective.) Bench on m1-b1: base 224306 nodes, O1 243605 (+8.6%
tree) — on m1-a1 O1 *shrank* the tree (219127 -> 190559) and won +25.5, here
it *grows* the tree and still wins +21.8. Fifth bench≠Elo instance for the
ordering project: never drop or keep on bench alone.

## Decision

- **Winner is A: m1-b1 net with O1.** The net-only candidate (B) is dropped —
  same net, −21.8 behind. A goes into `tools/checkpoint.sh --net m1-b1`
  (LEDGER 074), whose gate SPRT re-measures this same contrast at [0,10].
