# 064 · The first search feature to pass an SPRT: +51 Elo from SEE in the price

_2026-09-07. `chess match`, two frozen binaries in /tmp, both run as external
UCI subprocesses, both handed `--wdl nnue/nets/gate-3ep-q8.nnue` by name.
8+0.08, 5 concurrent, 32 MB hash, 43 openings. Engine b986f5b + working tree._

## Result

```
RESULT games=484 w=131 l=60 d=293 elo=51.34 lo=33.31 hi=69.64 los=1.0000 pairs=242
SPRT: H1 accepted
```

**+51.34 Elo [+33.31, +69.64]**, 484 games, SPRT [0,5], LLR 2.95.

The two arms are the **same source tree**, differing only in the default of
`Pricing::c_see` — 0 against 4000. Same net, named explicitly to both sides,
which is the trap that STATE records and that a bare binary in a bare directory
walks into every time: `chess-base bench` in /tmp reads **567321**, the PeSTO
fingerprint, because `quad.nnue` is not beside it. With `--wdl` both arms read
396299 and 227236 respectively.

## What the change is

One term in the price list: a move that loses material by static exchange
evaluation is charged 4000 milli-plies. Past ~9000 the term saturates against
the `max_base + max_slope*depth` ceiling, so the large-coefficient limit is
"an SEE-losing move gets the largest reduction the ceiling allows", which for
most of them is below `skip_below`. **It is SEE pruning in the main search, a
rule this engine only had inside quiescence.**

`library/004` measured that the existing gap feature — `best_score -
static_eval - move_gain` — explains **r² = 0.032** of the gap it is supposed to
predict, and that an SEE veto takes that to 0.080. It read `c_gap`'s measured
optimum of zero as falsifying the *feature*, not the allocation argument, and
predicted that fixing the feature would make the gap term large and positive.
This is that prediction, cashed, and then confirmed in games.

## The proxy predicted the sign, and badly overstated the size

| | |
|---|---|
| tuner, n=1000 @ 15000 nodes | **−4.62 ± 1.80 cp** |
| LEDGER 007 exchange rate | 1 cp ~ 25-30 Elo, so ~115-140 Elo |
| **games, 484 @ 8+0.08** | **+51.3 [+33, +70] Elo** |

The exchange rate over-predicts by roughly 2.3x. That is its second test: in
LEDGER 008 it predicted +12-14 and delivered +10.4 [−8, +29]. **Record the
exchange rate as biased high on this axis rather than as broken** — it has now
been right in direction twice and wrong in magnitude once, and one of the
reasons is measurable:

**Neither budget can see what the feature costs.** `c_see=4000` searches 43%
fewer nodes to depth 9 (227236 vs 396299) but runs **10% slower per node**
(646,374 vs 728,580 nps, deploy config), because `see()` now runs on every
priced child. A node budget prices that at zero. So does the work meter:
`work.rs` charges the main `price` zone **4.29 ns**, calibrated when that zone
did `move_gain` plus a history read and no SEE. The q-side `delta+SEE` zone is
7.12 ns for comparable work *including* a `see()` call.

So both currencies flatter this change by the same mechanism, and only the wall
clock charges it. The fix is mechanical: re-emit `DEPLOY_NS` with
`--features nodeprof` now that the zone does more work.

## Decision

- **`Pricing::c_see` default 0 → 4000.** New fingerprints: `chess bench`
  **234598** (quadratic), **227236** (WDL 512). The bit-identical match between
  the refactored platform build and the frozen binary that played the match is
  what says the Elo transfers to the refactored code.
- The corpus is **relabelled** from the promoted engine. A stronger engine's
  opinion at the same node count is a different opinion, and the label cache
  keys on the binary hash so mixing the two is impossible rather than merely
  discouraged (`library/012`).
- Re-emit the work price table before quoting any work number across this
  change.
- Not done: this was measured at one time control on one box. 8+0.08 is fast;
  a term that trades nodes for a per-node cost may scale differently at longer
  TC, where nps matters less relative to depth.
