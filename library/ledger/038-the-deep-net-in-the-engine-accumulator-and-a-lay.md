## 038 — The deep net in the engine: accumulator, and a layout bug worth 2.5x

**Setup.** `nnue/nets/deep.bin` (v5, f32) is `stack`'s best arm: `deep
128x2->32 +pawnfile+pawnpair x material+count`, val **0.016827**, 823,898
parameters. `src/deepeval.rs` evaluates it; `nnue/export_deep.py` writes it;
`nnue/verify_deep.py` rebuilds the model from the CHECKPOINT and compares, so
the exporter is on trial too. Bench is `chess bench 9`, AMD Ryzen 5 5500
(Zen 3, AVX2, no AVX-512), `target-cpu=native`, SPRT running on other cores
throughout — so every nps here is contended, and only ratios from interleaved
runs are quoted.

**Verification: exact.** 400 real tree positions, **0 feature mismatches**,
engine − model mean **−0.019 cp**, max **0.5000 cp** — precisely the rounding
bound. That covers the feature convention, the perspective flip, both derived
pawn families, the bucket indices, the weight layout, the summed-not-crossed
family rule, and the file format.

**Speed, and where it actually went.** First measurement was **138 knps**
against the deployed quadratic's **1.75 Mnps** — 12.7x down, which failed the
pre-registered gate (an SPRT there measures the slowdown, not the net).

| change | knps | vs quad |
|---|---|---|
| from scratch, f32, row-major | 138 | 12.7x |
| + incremental accumulator | 147 | 11.9x |
| + l1/l2 stored **input-major** | 365 | 4.7x |
| + bucket-weight cache | **411** | **4.2x** |

The accumulator was worth **1.13x**, not the 2x an operation count predicted.
The layout change was worth **2.48x**. `s += row[j] * acc[j]` is a reduction
into one scalar and f32 addition is not associative, so LLVM must emit it as a
serial chain of scalar FMAs however good the target CPU is. Storing the weights
input-major and swapping the loops makes the inner loop `h` independent
accumulators, which vectorises. **This is a data-layout property, not a
quantisation one** — the earlier claim that i8 was the only way out was wrong.

**A rough attribution is not a measurement.** Marginal costs were first
estimated by running one stage twice (`CHESS_ABL`, idempotent, node counts
unchanged). That said the bucket-weight sum was 27.5% of a node. Caching it at
a **71.1% hit rate** then bought **+6.1%** measured interleaved, not +20%. The
doubling trick ranks stages; it does not size them. The cache is kept.

**Negative:** skipping `acc[j] == 0.0` in l1 — the standard NNUE sparsity trick
— measured **neutral** (395.6 vs 395.9 knps, n=3 interleaved). crelu does not
zero enough of this accumulator to pay for the branch. Removed.

**Correctness of the incremental path.** `push` is told only the new board and
XORs the 12 piece bitboards against the parent's, so quiet moves, captures,
castling, en passant and promotion share one mechanism with no move-type switch.
Accumulators are keyed by perspective **colour**, not side to move, which makes
a null move free. `CHESS_ACC_CHECK=1` recomputes every eval from scratch and
compares: this found a real bug (levels are seeded by `evaluate`, but the root
skips its static eval in check and behind a TT hit, so the first `push` diffed
against nothing) worth **17,986 mismatches / 116,436 evals, worst 846 cp**.
After the fix: **8 mismatches / 111,233, worst 1 cp**, which is f32
non-associativity between the two summation orders. `CHESS_ACC_OFF=1` runs the
from-scratch path in the same binary; node counts are identical either way
(395,106), so the accumulator reproduces the search exactly.

**Fingerprints.** `chess bench 9` = **334,110** and **334,948**, both
unchanged; 33 tests pass; `chess perft-suite` ALL PASS. (The labels on those
two were checked in 2026-09-06 and the PeSTO one is wrong for the current
tree: today PeSTO benches **567,321** and the quadratic net **334,110**.
`qeval::net()` silently falls back to PeSTO when `quad.nnue` is not beside the
executable, so a bench number without the eval named is ambiguous.)

**Result: no SPRT yet.** 4.2x down still fails the 2x gate. Remaining costs are
now roughly level — l1, the extras gather (30 random rows, 15 KB), and what is
left of the bucket read.

**Corrected (LEDGER 055):** the closing sentence used to say i8 for l1 was
motivated by footprint rather than arithmetic, on the strength of l1's 32.8 KB
of weights sitting just over Zen 3's 32 KB L1D. That coincidence was not the
mechanism. The l1 kernel was leaving its accumulator in memory because of a
bounds check inside the loop; fixing that alone took this net from 574,076 to
813,140 nps with bit-identical output, and the weights were cache-resident the
whole time (LEDGER 054).
