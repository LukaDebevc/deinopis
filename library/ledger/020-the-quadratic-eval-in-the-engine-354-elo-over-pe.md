**020 · The quadratic eval in the engine: +354 Elo over PeSTO** · 2026-08-27
First real games. `src/qeval.rs` implements `Evaluator` with `eval = x^T M x + c`
computed from scratch — the upper triangle of the active submatrix, ~496
gathers — from the i16 file written by `nnue/export.py`.
**Setup:** 200 games, 10+0.1, 64 MB hash, concurrency 4, colour-reversed pairs,
built-in book. **Both sides are the same binary**; the only difference is
whether `quad.nnue` sits beside it, so nothing but the eval differs. Net is the
r=768 checkpoint from LEDGER 019, trained on 250M positions.
**Result: +354.5 Elo [+295, +437], +160 =34 -6, LOS 100.0%.**
**Cost:** bench 3.58 Mnps -> 2.52 Mnps, a **30% slowdown** — far cheaper than
feared, because eval is not called at every node. Node count moves (567321 ->
483608) as it must: a different eval is a different search.
**What this is NOT.** It is +354 against *our own previous build*, at one time
control, with 200 games. Elo does not add across a gap that size — LEDGER 004
measured 131 Elo of disagreement between anchors over a much smaller range — so
"2554 + 354" is not a rating and must not be written down as one. The absolute
number needs `tools/ladder.sh` re-run against the CCRL anchors.
**Two export bugs found by the cross-check, both silent in play:**
- the scalar bias was stored rounded to whole centipawns (-22.54 -> -23), a
  systematic **-0.46 cp on every eval**;
- each of the three terms was divided and rounded separately, up to 0.5 cp each.
  One rounding at the end took the engine-vs-model error from mean -0.78 /
  max 1.40 cp to mean **-0.068 / max 0.549** — unbiased, and now dominated by
  `Score` being whole centipawns rather than by i16 quantisation.
**Verification.** `chess evalfen` prints eval and feature indices for a FEN;
`nnue/verify.py` re-derives the features from the FEN independently and
evaluates the same set in float torch. **0 feature mismatches** on 14 positions
spanning openings, middlegames, endgames and both sides to move. With no net
present, `bench` is **567321 nodes — byte-identical** to the PeSTO build, which
is what makes the wiring believable.

---
