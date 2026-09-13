# 072 · m1-b1: 30B continuation of m1-a1 on the lc0 target

_2026-09-09. `nnue/runs/b1-20260909/` (`job2.sh`, `b1.log`). Warm restart
from `m1-a1.pt` (the cp-0002 net): weights only, fresh Adam, fresh cosine,
peak lr cut to a third (6.7e-4 -> 2.2e-4, parent annealed to zero at its
recipe). Batch 16384, wd 1e-2, clip 1.98, block 8M, seed 1, everything else
the parent's recipe. `--pool-end 1484364508` (every non-held-out record:
1.484B unique, ~20 passes, `--allow-repeat`); the val tail does not move with
`pool-end`, so ce_out is directly comparable to a1's. Ran on the cluster
(`<cluster scratch>`); 1,831,054 steps x 16384 = 30B positions._

## Result

Val `ce_out` **0.72281** vs a1's 0.72411 (Δ −0.0013; onll 0.59220, ce_mix
0.72476). Export int8 v9, sha256 `ce1f265647c060a2` — re-exported from the
`.pt` bit-identical. Same 512-wide combo topology as m1-a1, K 288.5.

Games (`nnue/runs/netcmp-20260909-191200/`, O1 binary both sides, equal time
only per LEDGER 068), 400 games at 10+0.1, conc 4:

```
400 games: +77 =272 -51   +22.6 Elo  [+4, +42]   LOS 99.0%
```

(m1-b1's perspective vs m1-a1.) Bench on the O1 binary: m1-a1 190559 nodes,
m1-b1 243605 (+27.9% tree) — the net costs nodes and still wins on the clock.

## Caveats

- `verify_wdl.py` reports a systematic engine-vs-torch offset (f32: mean
  5.9e-3 logits, max 12.7 cp) that reproduces byte-identically on the clean
  cp-0002 baseline binary and on m1-a1 — it predates this net and this tree
  (the i16/kernel state as shipped), affects both nets equally, and moves no
  game result. Features/buckets 0 mismatches; re-export identical. Not
  blocking; needs its own entry.
- The net delta was measured on the O1 binary; the O1-on-m1-b1 delta is
  LEDGER 073's SPRT, same net both sides.

## Decision

- **m1-b1 is the net.** It rides into the gate as
  `--net nnue/runs/b1-20260909/m1-b1.nnue`, sha `ce1f265647c060a2`.
