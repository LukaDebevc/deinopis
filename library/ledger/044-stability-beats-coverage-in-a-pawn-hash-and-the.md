## 044 — Stability beats coverage in a pawn hash, and the first noise floor

**Setup.** Five arms, identical model, only the bucket function differs:
`PurePSQTBucket` = base PSQT + one 768-dim PSQT delta per bucket, no hidden
layer, so there is no quadratic path to confound the bucket's contribution.
1.0B positions per arm (`all.data`, 15258 steps x 65536, lr 1e-3, warm start
`ckpt-big/psqt_768-_64-_16-_1.pt`). Run twice, seeds 0 and 1 — the seed changes
both init and batch order; the val set is the last `n_val` records and is
identical across seeds. `nnue/runs/magic_arms.log`.

| arm | seed 0 | seed 1 | seed spread | vs `rand64` |
|---|---|---|---|---|
| pure psqr only | 0.028543 | 0.028541 | 0.007% | — |
| pure psqr + `rand64` | 0.028502 | 0.028499 | 0.011% | (control) |
| pure psqr + `king6` | 0.027335 | 0.027334 | 0.004% | **−4.09%** |
| pure psqr + `newmagic` `0xdc0041000103ca4f` | 0.027577 | 0.027576 | 0.004% | **−3.24%** |
| pure psqr + `incumbent` `0x2c0c8fcfcfedecbb` | 0.027814 | 0.027809 | 0.018% | **−2.42%** |

**Noise floor — first one measured in this project.** Run-to-run spread is
≤0.018%, typically 0.004%. Two things follow. The 0.85% gap between the two
magics is ~50x the largest spread seen, so it is real. And retroactively: the
1B adapter study's 0.7% span was signal, the 2.5B joint arms' 0.026% span was
not, and should not have been read as a ranking.

**The control matters.** `rand64` — 64 buckets with no chess in them, assigned
by a random hash — already gains 0.14% on its own, just from extra capacity and
from splitting the data. So the honest contrast for any rule is (rule − rand64),
not (rule − psqr). Every "vs" column above is against `rand64` for this reason.

**Result.** The more *stable* magic wins. `newmagic` was selected for low
refresh rate, not for coverage, and it beats the incumbent by 0.85% of val loss
— 34% more captured signal against the control (3.24% vs 2.42%). This was
predicted in advance.

**What `newmagic` throws away.** Square-sensitivity probe over n=87,060 distinct
pawn structures (`nnue/attic/magic_explain.py`): per square, P(index changes | toggle
that square). Incumbent mean 0.966, `newmagic` mean 0.737. `newmagic` is blind
to the **entire 4th rank** (0.00–0.07), to a2/b2/c2 (0.03/0.07/0.16) and to
a5/b5 — about 13 of the 48 pawn-legal squares.

**Negative — "the discarded rank must be redundant" is false.** The obvious
defence of `newmagic` is that rank 4 is predictable from the rest, so dropping
it costs nothing. Measured directly (`rank_info.py`, n=4,000,000 positions,
211,488 distinct structures): H(rank r | all other ranks), in bits, ranks 2–7 =
2.768 / 2.888 / **2.913** / 1.584 / 0.431 / 0.138. Rank 4 is the *most*
expensive rank to discard, not the least. `newmagic` deleted the most
informative rank and won anyway.

**[INFERENCE] why that is not a contradiction.** The 768-feature PSQT already
sees every pawn on every square, linearly. The bucket is not there to describe
the position — it is there to name the regime the position is in, and pick the
matching PSQT. A regime label wants to be coarse and to hold still while the
position develops. Information about the pawn structure and usefulness as a
regime label are different quantities, and this is the first measurement in
this project that separates them.

**Decision.** Score candidate rules by stability, not by how much of the board
they read. Keep the king at 6 bits (043); the pawn side gets a designed rule
family — `popcount(pawns & MASK) > k` predicates, several of them compressed
down to a small state count — annealed against a game-level objective. Next
arms: `king6 x pawn_rule` against `king6` alone, `newmagic`, `pawncount` and
`rand64`, so that the cross is measured against the strongest single feature it
contains rather than against psqr.
