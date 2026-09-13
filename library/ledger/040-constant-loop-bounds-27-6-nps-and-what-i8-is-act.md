## 040 — Constant loop bounds: +27.6% nps, and what i8 is actually for

Every inner loop in `deepeval.rs` bounded by `self.width` or `self.hidden` —
runtime values from the net file, so LLVM could not keep the destination in
registers or size its vector body. Monomorphised on the dimension and
dispatched over the shapes the trainer emits, with the old loop as a fallback.

`chess bench 12 --deep`: **478,000 → 609,880 nps (+27.6%)** at 2,041,983 nodes
/ 1,764,016 evals / 115 rebuilds — **all three identical before and after**,
which is the whole point: same operations, same order, same f32 rounding. Quad
and PeSTO fingerprints unchanged, 33 tests pass, `perft-suite` ALL PASS.

`CHESS_ABL`, ns/node, total 2105 → 1651: extras gather **423 → 189**, l2 108 →
90, **l1 728 → 708 — no change**, and l1 is the biggest single item.

**Why, measured rather than inferred.** 038 guessed l1 was footprint-bound from
a coincidence (32.8 KB of weights, 32 KB L1D). A standalone kernel benchmark
with a controlled cache walk between calls confirms it, ns per l1 call:

| intervening traffic | runtime dims | const dims | avx2 int8 |
|---|---|---|---|
| none | 927 | **243** | 77 |
| 64 KB | 910 | 420 | 94 |
| 512 KB | 1559 | **1486** | 1012 |

With clean caches, constant bounds are worth 3.8x on l1. At 512 KB of
intervening traffic — roughly what a real node does, between a TT probe into
256 MB and a 128 KB accumulator stack — they are worth **nothing**, which is
exactly what the engine measured. The arithmetic was never the bottleneck.

**This conclusion was wrong — see LEDGER 055.** What it said was: i8 is
motivated by footprint, not arithmetic (32.8 KB → 8.2 KB), the kernel speed-up
will mostly not be visible, and the column that survives is the one that fits
in L1D.

Two later measurements kill it. LEDGER 054: in a real search the layer weights
are never cold, because the bucket indices are king squares and material and
neither moves much inside a subtree — so the 512 KB-of-intervening-traffic
column above is not what a node does, and i8 in the engine bought 3-7%, not the
3.4x its byte count implied. LEDGER 055: the constant bounds above were
necessary but not sufficient, because `&w[j * H..j * H + H]` left a conditional
panic *inside* the loop and the accumulator therefore could not stay in a
register. Removing it bought **+41.6% on this very net** (574,076 → 813,140
nps, bit-identical output) — in the engine the table above predicted would show
nothing.

The table itself stands as measured. What does not stand is reading its bottom
row as the operating point.
