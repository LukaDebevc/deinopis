## Bugs found and fixed this session

0. **The refresh price was half, and asymmetric rules were free.** The single
   most damaging bug of the study: refresh was measured in the mover's
   perspective only, so a rule reading the opponent's half measured 0.00%. It
   voided the king-coarsening frontier and every staged rule. Found by Luka
   from a conservation argument, not from a sample. Fixed in `nnue/attic/merge2.py`
   by mirroring the record and counting both accumulators.
0b. **The merge joined across the lightest edge.** Lowest combined mass is the
   pair with the *weakest* transition, so each merge saved the least refresh
   available. Material 2,916 -> 128 removed 13% of the cut; joining across the
   heaviest edge instead removes 62%.
1. **Silent cold start.** `main()` gated the warm start on an `isinstance`
   list, so *every* model class added after it was written was cold-started
   while historical bars were warm. Caught in the `deep` smoke test. Now
   `hasattr(model, "init_psqt")`.
2. `release_buckets` wrote into a tensor while reading an overlapping expanded
   view — added `.clone()`.
3. `feature_cost.py` measured three wrong things in succession (only moved a
   knight; *added* a second king rather than moving one; an 8-piece backdrop
   left region masks mostly zero).
4. The capture column in `rulestats.py` allowed capturing the enemy king,
   which is why every rule read ~11%.
5. Test FENs without kings; a test importing another test's module body.
6. **A per-bucket ridge scaled by each bucket's own data.** A five-sample
   bucket got a five-sample prior, stayed underdetermined, and its read
   exploded on the held-out half. Every rule scored NEGATIVE and the
   correlation with training was -0.77. One absolute prior on a per-sample
   scale fixed it. Looks right, is not.
7. **A control that swapped the accumulator also swapped the prediction**, so
   every rule "repaired" a broken model by ~27% and the control measured
   nothing.
8. **An (N, 32, 32) int64 intermediate in the passed-pawn test** -- 4.8 GB at
   580k moves, killed by the OOM killer. Chunked, and int8.
9. **Phase bins were half the width they claimed.** `_phase` maxes at 12 per
   army, 24 for the position, but the bin divisor was 49. `phase x8` silently
   produced 4 bins and `phase x4` produced 2. Everything phase-related in
   section 8 is post-fix; the pre-fix numbers in an earlier draft of this file
   were for half the granularity and 2-3x too cheap.
10. **Rating rules by their legal-move refresh rate.** Not a code bug, a
    measurement design bug, and the one that mattered most — see section 8.

---
