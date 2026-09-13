# Checkpoints

Every row is a version that passed the gate in `tools/checkpoint.sh`: correct
(perft + tests exact), and beaten-the-previous-checkpoint at an SPRT, played
between the two binaries over the same UCI path. Each row has a matching
annotated git tag, so any of them can be rebuilt and re-tested:

```
tools/checkpoint.sh -m "what changed"     run the gate; push only if it passes
tools/checkpoint.sh --restore-baseline    rebuild the reference from the last tag
git tag -l 'cp-*'                         every kept version
```

A failed gate writes nothing and pushes nothing. Failures are not recorded
here — they belong in `LEDGER.md`, with the same care as the successes.

| tag | date | what changed | bench nodes | strength vs previous |
|---|---|---|---|---|
| cp-0001 | 2026-09-06 | WDL eval by default, 2.2x faster kernel, contempt as a readout | 396299 | first checkpoint — no reference to compare against |
| cp-0002 | 2026-09-08 | i16 accumulator, c_see pricing, m1-a1 net | 219127 | 380 games: +95 =229 -56   +35.8 Elo  [+16, +56]   LOS 100.0%   LLR 2.96 [-2.94, 2.94] |
| cp-0003 | 2026-09-09 | o1 SEE-split ordering, m1-b1 net | 243605 | 684 games: +135 =455 -94   +20.9 Elo  [+7, +35]   LOS 99.8%   LLR 2.99 [-2.94, 2.94] |
| cp-0004 | 2026-09-12 | quiescence TT sharing | 199091 | 246 games: +65 =159 -22   +61.4 Elo  [+34, +90]   LOS 100.0%   LLR 2.95 [-2.94, 2.94] |
