# 100 · SR-1a as-bundled is dead on arrival (+10.4 proxy); split shipped

_2026-09-17. Pre-SPRT proxy screen on `fishpack-4k-cp5.labels` @15k nodes:
`--set fut_max_depth=3` (map SR-1 variant a) vs default._

## Result

```
4000 positions @ 15000 nodes: b - a = +10.403 +/- 1.011 cp (|t| > 10)
```

[FACT] The arm as specified measures the bundle of skip-quiets plus the
084-killed whole-node return-static gate (both shared `fut_max_depth`),
and the killed form dominates. This reproduces 084's kill (+4.6..+22cp on
the old labels) on the new teacher — the mirror works.

## Decision

- [DECISION] No SPRT of the bundle. Split shipped instead (commit
  `0e8ac04`, bench-exact 219718, all tests pass): `fut_max_depth` now
  enables ONLY the skip-quiets form; the whole-node gate additionally
  needs `fut_whole_node=1` (default 0, never SPRT on).
- [FACT] The pure skip-quiets form (`fut_max_depth=3`, whole-node off)
  reads **−0.322 ± 0.739 cp** on the same corpus — proxy-neutral
  (|t| < 1), 881/4000 positions differ. Not refuted; SPRT-worthy.
  Frozen split binary in `~/chess-runs/20260917-sr1/`; SPRT launches
  after SR-2 (or after reboot — same command either way, P4 with
  `--set fut_max_depth=3` on one arm).
