**022 · The eval is not called on quiet positions — 83.6% have a capture** ·
2026-08-27
The training filter (`nnue/extract::keep`, the Stockfish/bullet filter) keeps a
position only if its **best move is a quiet non-capture**, plus not-in-check,
`|score| <= 10000` and `ply >= 16`. 54.0% of binpack entries pass. The question
this entry answers is whether that matches where `evaluate()` is actually
called.
**Setup:** `--features evalstats` (compiled out otherwise) counts, at each of
the two eval call sites, whether the position has a capture available and what
SEE says about the best one. `chess bench` at depth 9 with the 894M net,
n = 302747 eval calls. Node count **334110, identical to the uninstrumented
build**, so the instrumentation is behaviour-neutral.

| site | calls | share | any capture | SEE>=0 | SEE>=100 |
|---|---|---|---|---|---|
| qsearch stand-pat | 202099 | 66.8% | 84.6% | 70.7% | 63.0% |
| main-search static | 100648 | 33.2% | 81.8% | 69.9% | 62.5% |

**Result: 83.6% of eval calls have a capture available and 62.8% have one that
wins at least a pawn by SEE.** The training set is 0% by construction.
**Mechanism, not coincidence.** `evaluate()` is the qsearch stand-pat, computed
at *every* qsearch node before captures are generated (`search.rs:851`), and
again at every non-check interior node for pruning (`search.rs:628`). A qsearch
node is *reached by* making a capture, so a recapture is usually available. The
selection effect is the whole story.
**Why this is not obviously a bug in the filter.** The filter's justification is
about the *label*, not the input: a search score taken mid-exchange describes
the tactic, which a static function may not be able to represent. But a
quadratic form on piece-square pairs is a piece-*interaction* model and can
encode some attack/defend relations that a PSQT cannot, so it may have capacity
for the static part of a tactic. Open.
**Next test (pre-registered):** extract a second pool at the same stride with
the capture clause dropped (keep check / mate-score / ply), train the same
model for one epoch on each, and play the two nets head to head. Val loss
cannot decide it — the two pools have different validation distributions, so
the comparison has to be Elo.

---
