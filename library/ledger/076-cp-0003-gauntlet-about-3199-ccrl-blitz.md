# 076 · cp-0003 gauntlet: about 3199 on the CCRL Blitz scale

_2026-09-09. `tools/ladder.sh run --games 200 --tc 10+0.1 --concurrency 5
--only byteknight,4ku,inanis,simbelmyne` — same protocol and anchor set as
071, so the number compares directly. Playing binary `.checkpoints/best`
(the cp-0003 tag build), net `m1-b1.nnue` via `$CHESS_WDL`, bench verified
243605 before launch. `.ladder/gauntlet-20260909-221721.tsv`._

## Result

```
opponent       CCRL    score     diff   implied   95% interval
byteknight     2859    83.8%     +285      3144   [3075, 3213]
4ku            3057    72.2%     +166      3223   [3172, 3274]
inanis         3087    57.2%      +51      3138   [3094, 3182]
simbelmyne     3238    51.5%      +10      3248   [3210, 3287]

rating  3199  ±56   (95%, scaled for anchor spread)
spread  chi2/dof = 5.73 over 4 opponents, error x2.39
```

Zero engine-error games on all four anchors. All four pairings won.

## Reading

Delta vs 071's 3176 ±42 is **+23 against a combined error of ~±70** — the
gauntlet does not resolve the cp-0003 jump; the SPRTs carry that claim
(code +20.9 same-net, net +22.6 same-code), not this number. The +23 is
consistent with the legs, not confirmation of them.

The anchor spread is the worst yet (chi2/dof 5.73 vs 3.43 in 071, 3.71 in
run2). Byteknight is low for the third time (3144 vs 3138–3248 elsewhere);
inanis dipped this time (3138, was 3172 in 071) while simbelmyne rose
(3248, was 3207). Same direction as before, larger: the combination is
suspect by its own metric, and per-opponent estimates moved ±35 between
runs without any code change on those pairings — that is anchor noise,
not signal.

## Decision

- Quote **about 3199 on the CCRL Blitz scale**, never bare. Same systematic
  offset as always (book, tablebases, TC, pool) on top of the ±56.
- Anchor-set rule stands and extends: if the set changes, drop byteknight
  first; inanis is now second on the list.
