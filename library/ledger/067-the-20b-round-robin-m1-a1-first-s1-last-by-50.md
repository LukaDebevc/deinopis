# 067 · The 20b round robin: m1-a1 first, s1 last by 50

_2026-09-08. `tools/roundrobin.sh --games 300 --tc 10+0.1 --concurrency 4`
over `nnue/ckpt-20b/{s0,m1-a1,m2-a2,m3-a3,s1}.pt`, exported to int8 once.
One frozen binary both sides; the only difference between arms is `--wdl`.
10 pairings x 300 games = 3000 games. `nnue/runs/rr-20260908-103255/`._

Provenance gap: the training regime behind these five ckpts is not recorded
in any ledger entry. File dates: s0/s1 00:31, m2-a2 05:16, m3-a3 06:02,
m1-a1 09:49. s0/m1-a1/m2-a2 are 512-wide (2.1 MB); m3-a3/s1 are 1024-wide
(3.9 MB). Bench nps: 702k/707k/721k vs 528k/525k — the wide nets cost ~25%
on the clock, which is part of what equal-time measures.

## Result

Ratings (anchor s0 = 0, +- one SE). Fit chi2 8.0 on 6 dof — consistent, no
intransitivity flag.

```
net                  elo       +-   95% interval
m1-a1              +30.1      7.6   [+15, +45]
m2-a2              +12.9      7.4   [-2, +27]
m3-a3              +10.1      7.6   [-5, +25]
s0                   0.0        -   (anchor)
s1                 -52.8      7.8   [-68, -38]
```

Head-to-heads (first-arm perspective): s0-m1-a1 -36.0 [-60,-12];
s0-m2-a2 -10.4 [-34,+13]; s0-m3-a3 -17.4; s0-s1 +63.2 [+40,+87];
m1-m2 -2.3 [-27,+22] (tied); m1-m3 +27.9 [+5,+51]; m1-s1 +93.7 [+68,+120];
m2-m3 -10.4 [-33,+12]; m2-s1 +58.5; m3-s1 +50.1.

## Decision

- **m1-a1 is the points leader**, but m1 vs m2 is unresolved: tied
  head-to-head, ratings gap ~17 Elo at ~1.6 SE. Settled in LEDGER 068.
- s1 is last by a distance (−53, loses to all four). Nothing there ships.
- The round robin retires the transitive-through-quad ranking and STATE's
  standing caveat about it.
