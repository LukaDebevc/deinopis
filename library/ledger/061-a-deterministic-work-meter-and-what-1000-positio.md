# 061 — A deterministic work meter, and what 1000 positions can resolve

**Setup.** Two things built and measured together: a cost model that replaces
the wall clock with a deterministic count, and a position corpus drawn from a
binpack instead of from our own games. Commit `b986f5b` plus the working tree;
eval `nnue/nets/gate-3ep-q8.nnue`; box Ryzen 5 5500.

## The work meter

`--features work` counts each non-recursive leaf operation inside a node and
prices it from the frozen table in `src/work.rs`
(`work_ns = Σ calls[z]·PRICE_NS[z] + nodes·CONTROL_NS_PER_NODE`). It is a
budget as well as a readout: `Limits::work` stops a search at a work ceiling,
accumulated in integer picoseconds so it is bit-reproducible.

| check | result |
|---|---|
| does it change the tree? | bench **334110 / 396299 / 308930**, all exact |
| deterministic? | counts **bit-identical across 5 runs** |
| immune to load? | 6 CPU spinners: wall clock moved **2.06×**, the modelled number moved **0 digits** (491.4 ms both, to the last digit) |
| honest against the clock? | −0.6% to −3.5% across runs |
| two computation paths agree? | incremental counter vs batch model **+0.0000%** |
| does it cost anything? | 727,784 nps counting against 714,166 plain |

**What it cannot see.** It prices *calls*, not cycles, so it is blind to cache
and bandwidth. The i16 accumulator — the largest measured lever in the engine —
performs exactly the same number of pushes and would read as **zero change**
here. Behaviour changes go to this model; implementation changes still go to an
interleaved pinned wall-clock A/B.

A bug worth recording: a fresh `ThreadData` started with an all-zero price
table, so `work_ps` never incremented, a work ceiling never bound, and the
search ran to `MAX_PLY` instead of erroring. The test had missed it by
installing a price table by hand — arranging a condition the real caller cannot
arrange. `Tally::default()` now prices itself, and the test no longer installs
anything.

## Fixed work vs fixed nodes: no difference yet

A quiescence node costs 0.275 of a main-search node (`library/007`), so a fixed
*node* budget should credit a price list for buying cheap nodes. Tested on the
direction where that should bite hardest, `c_rank` 250 → 1000 (LEDGER 014's
known-bad over-pruning arm), n=1000, budget 15000:

| budget | b − a | nodes a → b |
|---|---|---|
| fixed nodes | **+2.49 ± 2.06 cp** | 10551 → 11265 (+6.8%) |
| fixed work | **+2.23 ± 2.09 cp** | 10583 → 11487 (+8.5%) |

The mix shift is real and visible in the node column, but the verdict moved by
**0.25 cp against a 2.06 cp standard error** — an order of magnitude inside the
noise. Same on `delta_margin` 100 → 400 (0.11 cp). **Fixed work is a
correctness fix that changes no answer at this operating point.** It is kept
because it costs nothing and is load-immune, not because it bought anything.

## What 1000 positions resolve

Paired standard error on this corpus at 15000 is **2.06 cp**, so the smallest
resolvable difference at |t| = 2 is **4.1 cp**:

| effect | positions needed |
|---|---|
| 4.1 cp | 1000 |
| 2.5 cp (`c_rank` 250→1000) | 2700 |
| 1.37 cp (deleting the whole price list, LEDGER 010) | 9000 |
| 1.0 cp | 16900 |

LEDGER 014 resolved the `c_rank=1000` arm at n=4000; at n=1000 it sits at
|t| = 1.2 and is not resolved. **1000 is nonetheless the chosen size**: at
LEDGER 007's exchange rate a 4.1 cp floor is a ~100 Elo floor, and the project
is shopping for structural wins of that order, not +1 Elo patches. A change
that does not resolve at 1k is a small change. The corpus sample is nested in
`n`, so raising it later costs only the new positions.

**Correction to `tune.rs`'s pairing argument.** The module claims pairing
removes most of the variance because two price lists usually choose the same
move. On this arm **436 of 1000 positions differ** and pairing bought only
**1.4×** over independent means. The claim holds for the small steps a descent
takes, not for a large A/B; stated in `library/012`.

## The corpus

`nnue-extract --mark --game-stride 401 --fen-dump` read 2.56 billion positions
from 21.4 million games of `fishpack32.binpack` in 137 s and wrote 6,379,672
FENs. 1000 sampled by hash, labelled at 400k nodes × top 5 in 554 s.

Against the old PGN-drawn set at the same budget: regret **41.5 cp vs ~23.2**,
off-list **11.7% vs 4.6–10.5%**, and **0 of 1000** positions have a mate as
their best move against the ~5% `tune.rs` documents. These positions are harder
and cleaner, and **numbers from this corpus are not comparable to any earlier
ledger number.**

## Decision

The framework is the instrument for proposing; nothing about strength is
claimed here. `tools/checkpoint.sh` still decides. Usage in `library/012`.
