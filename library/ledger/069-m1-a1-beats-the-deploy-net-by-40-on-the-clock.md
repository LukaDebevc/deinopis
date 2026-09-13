# 069 · m1-a1 beats the deploy net by 40 Elo on the clock

_2026-09-08. `tools/netcmp.sh --games 400` (eqtime-only, post-068),
`nnue/nets/gate-3ep-q8.nnue` as reference vs RR-exported `m1-a1.nnue`.
400 games at 10+0.1, concurrency 4, one frozen binary, hash 32.
`nnue/runs/netcmp-20260908-171454/`._

This measurement never existed before: the RR ranked the 20b nets only
against each other. Same 512-wide shape both sides, bench nps within 5%
(670k vs 703k), so this is eval quality converting on the clock, not a
speed artefact.

## Result

```
chess vs chess   10+0.1   2400s
400 games: +98 =250 -52   +40.1 Elo  [+19, +62]   LOS 100.0%
RESULT games=400 w=98 l=52 d=250 elo=40.13 lo=18.59 hi=61.99 los=0.9999
```

(m1-a1's perspective.)

## Decision

- **m1-a1 is the net.** It went into the gate as
  `--net nnue/runs/rr-20260908-103255/m1-a1.nnue`, sha256 `2cf1c2a57579fef9`
  in the cp-0002 tag message (LEDGER 070).
- The deploy file `nnue/nets/gate-3ep-q8.nnue` was not overwritten; nets are
  gitignored and the tag pins the sha instead.
