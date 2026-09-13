## 043 — Negative: the 32 learned "rules" are king squares (`nnue/runs/km*.log`)

**Setup.** 6 bits of king square x 6 bits of a pawn-bitboard magic hash =
4,096 contexts; train `psqr + 4096 x psqr` (PurePSQTBucket, base PSQT plus a
per-context PSQT delta, so only the *relative* difference is learned); k-means
the 4,096 learned adapters into 32 groups; retrain `(rule x psqr) -> 256`
end to end and compare against HalfKA. Contra 033, this consults the fine model
only to pick the partition and then retrains — it does not inherit weights.

**Adapters, 1.0B positions (1.12 epochs of all.data).** psqr only 0.028543;
+ king_magic 4096 **0.026869 (-5.86%)**; + `rand4096` (a hash of the position,
same table size, no chess) **0.027795 (-2.62%)**; bucketed psqr+quad 0.020113.
**The control ate 45% of the gain.** King x magic over a random 4,096-way
partition is worth only -3.33%.

**Additive head-to-head, 1.0B.** halfka 128 **0.015633**; km32 **0.015646**;
king6 alone **0.015525**; magic6 alone 0.015613. The 32 learned rules lose to
the single simplest rule available — the king square, on its own.

**Joint head-to-head, 2.5B.** halfka (king,piece) 64x768x256, 12,715,779 params
-> **0.015151**; km32 (rule,piece) 32x768x256, ~6.4M params -> **0.015155**. A
dead heat at half the table. This is the one defensible positive.

**Why the rules lost: they are the king.** MI(cluster; king6) = **3.439 of
H(cluster) = 3.921 bits, 87.7%**; MI(cluster; magic6) = 0.530, 13.5%.
Visit-weighted modal-king purity **0.822**, modal-magic purity 0.114. Cluster
sizes pile up at 63 ~ one king square's worth of contexts. k-means on the
adapters rediscovered king routing, which 035 already showed saturates at
-5.8%. The pawn half contributed almost nothing to the partition.

**Two flaws in how it was produced.** (1) The k-means was **unweighted**: 880 of
4,096 contexts get <10 visits per million and 159 are never visited, yet each
fitted point counted as much as an 8,195-visit context (corr(log freq,
|adapter|) = 0.482). (2) The 2.5B run is **2.80 epochs** over 893M unique
positions with `order=seq --allow-repeat`, so pass two replays pass one's exact
batches — against the one-epoch rule.

**The magic search behind it did not do what it claimed.** Independent check,
n=300 random magics on 500k positions / 300k pairs of `tono_games.data`: the
chosen magic scores H 5.9317, stability 0.9266, H*stab^2 5.0926 — 100th
percentile, z = 6.09. But **random magics already score H p50 = 5.9448 and the
chosen one is below that**. Entropy is free for any scramble, so the
free-energy objective `H/n - beta*E/n` collapsed to a pure stability search and
beta never bound. The stability ceiling is set by chess, not by the hash: the
pawn bitboard is **identical in 78.7%** of same-side move pairs, so any
scrambler scores 1 - 0.5*0.213 = 0.8934, which is the measured random median
(0.8927).

**Also:** the "halfka" label is unearned in the additive experiment — that arm
is PSQT plus a 128-row king bias, not HalfKA; and the joint version is
single-perspective where real HalfKA keeps two accumulators.

**Decision.** Drop k-means-over-adapters as a way to find rules. It recovers
the strongest single feature already in the index and calls it 32 rules. Keep
the king at full 6 bits; the pawn side is what needs a better rule. Replace
`H * stab^2` and the free-energy form with an objective that can actually bind
— see 044.
