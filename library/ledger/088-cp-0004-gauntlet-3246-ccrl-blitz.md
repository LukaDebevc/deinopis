# 088 · cp-0004 gauntlet: about 3246 on the CCRL Blitz scale

_2026-09-12. `tools/ladder.sh run --games 200 --tc 10+0.1 --concurrency 5
--only byteknight,4ku,inanis,simbelmyne` — same protocol and anchor set as
071/076, so the number compares directly. Playing binary
`.checkpoints/best` (the cp-0004 tag build), net `m1-b1.nnue` via `$CHESS_WDL`,
bench verified 199091 before launch. `.ladder/gauntlet-20260912-205119.tsv`._

## Result

```
opponent       CCRL    score     diff   implied   95% interval
byteknight     2859    86.2%     +319      3178   [3107, 3249]
4ku            3057    76.2%     +203      3260   [3205, 3315]
inanis         3087    69.8%     +145      3232   [3186, 3279]
simbelmyne     3238    54.5%      +31      3269   [3230, 3309]

rating  3246  ±33   (95%, scaled for anchor spread)
spread  chi2/dof = 1.81 over 4 opponents, error x1.34
```

Zero engine-error games on all four anchors. All four pairings won.

## Reading

Delta vs 076's 3199 ±56 is **+47 against a combined error of ~±65** — the
gauntlet does not resolve the cp-0004 jump on its own; the gate SPRT (+61.4
same-net over cp-0003, 087) carries that claim, and this number is consistent
with it rather than confirmation of it.

The combination is the cleanest yet (chi2/dof 1.81 vs 5.73 in 076, 3.43 in
071). Byteknight still reads lowest (3178 vs 3232–3269) — fourth time — but
the gap narrowed and no longer dominates the error. The anchor-set rule
stands: if the set changes, drop byteknight first.

## Decision

- Quote **about 3246 on the CCRL Blitz scale**, never bare. Same systematic
  offset as always (book, tablebases, TC, pool) on top of the ±33.
