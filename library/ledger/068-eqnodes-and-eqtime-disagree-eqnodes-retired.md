# 068 · Equal nodes and equal time disagree; equal nodes is retired

_2026-09-08. `tools/netcmp.sh --games 400` (before the eqnodes removal),
m1-a1 as reference vs m2-a2, both RR-exported int8 files, one frozen binary.
400 games at nodes=100000, then 400 at 10+0.1, concurrency 4.
`nnue/runs/netcmp-20260908-155426/`._

## Result

```
eqnodes (100000 nodes/move, 400 games):
RESULT games=400 w=88 l=61 d=251 elo=+23.49 lo=+0.68 hi=+46.50 los=0.9782
eqtime (10+0.1, 400 games):
RESULT games=400 w=70 l=84 d=246 elo=-12.17 lo=-32.79 hi=+8.37 los=0.1228
```

(w/d/l from m2-a2's perspective.) Per-node quality says m2-a2, the clock
leans m1-a1 (pooled with the RR head-to-head, m1 leads ~+8 on time —
same direction twice, neither significant).

Mechanism for the split: at the same bench depth m2-a2 grows a **+17%
bigger tree** (255,808 vs 219,127 nodes). Its per-node edge does not convert
on the clock because it searches more nodes to get there.

## Decision

- **Ship on the clock: m1-a1 stays the candidate.** Quality-per-node is
  information, not a ship criterion.
- **The equal-nodes condition is deleted** from `tools/netcmp.sh` (condition,
  `--nodes`, `--skip-*` flags) and the `--nodes` option from
  `tools/roundrobin.sh`. Rationale, also in the script header: games are spent
  on questions somebody acts on; quality is read off val loss for free. This
  mirrors the standing `wdlmatch.sh` policy ("EQUAL TIME, on purpose").
- Stated doubt, kept on record: "val ranks quality" is the weak half — val has
  ranked these arms backwards three times (LEDGER 052, STATE). The policy does
  not rest on it; it rests on the clock ruling regardless.
- The follow-up ship match (m1-a1 vs deploy net, time only) is LEDGER 069.
  The aborted eqnodes run it replaced is `nnue/runs/netcmp-20260908-170352/`
  (partial log, ignore).
