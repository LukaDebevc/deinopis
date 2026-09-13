# 071 · cp-0002 gauntlet: about 3176 on the CCRL Blitz scale

_2026-09-08. `tools/ladder.sh run --games 200 --tc 10+0.1 --concurrency 5
--only byteknight,4ku,inanis,simbelmyne`, fixed 200 each (session decision:
SPRT per opponent was rejected — the even opponents would run to the cap and
there is no combined rating at the end of it). Playing binary
`.checkpoints/best` (the cp-0002 tag build, not the NOHIST-dirty tree), net
`m1-a1.nnue` via `$CHESS_WDL`, verified 219127 before launch.
`.ladder/gauntlet-20260908-191341.tsv`._

## Result

```
opponent       CCRL    score     diff   implied   95% interval
byteknight     2859    79.5%     +235      3094   [3036, 3152]
4ku            3057    68.0%     +131      3188   [3140, 3236]
inanis         3087    62.0%      +85      3172   [3130, 3214]
simbelmyne     3238    45.5%      -31      3207   [3168, 3245]

rating  3176  ±42   (95%, scaled for anchor spread)
spread  chi2/dof = 3.43 over 4 opponents, error x1.85
```

Run2 (Sep 7, same protocol, 8 opponents) said 3049 ±46. Delta +127 against
~±62 combined error — consistent with the cp-0002 legs (code +35.8 same-net,
net +40.1 same-code), not a surprise on top of them.

## Decision

- Quote **about 3176 on the CCRL Blitz scale**, never bare. Systematic offset
  (book, tablebases, TC, pool) still applies on top of the ±42.
- The byteknight anchor (3094) disagrees with the other three (~3172-3207),
  same pattern as run2's chi2/dof 3.71. Combination is suspect by its own
  metric; if the anchor set changes, drop byteknight first.
- Weak anchors (blunder and below) were excluded up front — they saturate
  near 100% and contribute nothing but runtime.
