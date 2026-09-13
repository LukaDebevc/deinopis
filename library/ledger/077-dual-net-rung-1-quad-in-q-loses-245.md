# 077 · Dual net rung 1: quad in quiescence loses 245 Elo

_2026-09-10. Stateless tiering: `--dual` routes quiescence to the cheap quad
net (`nnue/nets/base.nnue`, v3) while the main search keeps m1-b1. Same frozen
binary both sides, only the flag differs. Killed by hand at 56 games,
unambiguous._

## Result

SPRT [0,5] @8+0.08 conc 5, A dual vs B base, same m1-b1 + same quad file:

| games | score | Elo | LLR |
|---|---|---|---|
| 56 | +0 =22 −34 | **−244.7 [−374, −156]** | −0.99, heading to H0 |

Log `/tmp/dual/dual-sprt.log`, PGN alongside. Patch
`/tmp/dual/dual.patch` (trait `set_small` default no-op, tier flag in
`DefaultEval`, one gated branch per node in `search()`, `--dual`/`$CHESS_DUAL`
parsed in `qeval::init_dual` wired into `crate::init`).

## Speed (what was bought)

Bench, m1-b1: base 199091 nodes @848k nps (0.235s) vs dual 211008 nodes (+6.0%)
@1.155M nps (+36%) (0.183s) — **1.285x on time**. PeSTO bench (no nets):
194203 both modes, bit-identical — the plumbing is exact when tiers coincide.

Rung 0 (nodeprof, base): main eval 18k calls / 9.9% of search, **q eval 83k
calls / 46.0%**. Quad evals run 255ns vs WDL 983ns (3.9x, not 10x).

## Diagnosis (why it lost)

Quad and WDL are different functions, not two fidelities of one: on 1500 tree
positions, quad−WDL bias **−98cp**, MAE **520cp**, sd 1465 vs 983, corr 0.889.
A 520cp-MAE tier at 86k q-nodes corrupts every stand-pat window, and Q-store
carries it into the main search via TT-eval reuse (kept deliberately — see
below). No affine fix: best-case residual sd ≈ 450cp.

Two design notes, both measured:

- **TT-eval exclusion tripled main evals** (18k → 54k on bench): Q-store makes
  Q_DEPTH entries abundant, so refusing them in the main search rebuys half
  the win. Dropped before the SPRT — main reuses cheap evals, same code path
  as base. The SPRT above prices the fast variant.
- `base.nnue` is the ancient v3 deploy fallback, not a distilled small. A
  tier must be trained to agree with the big net (rung 3: shared-accumulator
  distilled tail); an unrelated weak net is not a fidelity level.

## Decision

- Rung 1 (unrelated small net) is **closed**. Do not retry with another
  off-the-shelf weak net — the failure is disagreement scale (520cp), not the
  particular net.
- Next is the **noise-tolerance probe**, which separates demand from supply:
  big net everywhere, plus deterministic hash-seeded Gaussian noise σ on
  q-evals only (`$CHESS_QNOISE`). Fixed 400 games, not an SPRT — the probe
  carries no speed benefit so it can only lose or tie; the number wanted is
  the magnitude. σ=25 first (bench: +11.5% tree; σ=100 inflates the tree
  +81%, almost certainly underwater). Proceed to distillation iff the σ=25
  loss is well under what the 1.29x speedup is worth (~+25–30 Elo at
  doubling ≈ +80).
- The `--dual` seam stays in the tree (dormant, base-exact) while the probe
  runs. If the probe fails too, revert the seam fully.

## Incident: the restructure that silently left the trait (2026-09-10)

Extracting the quad path into `quad_eval` closed `impl Evaluator for
DefaultEval` early, leaving `observe`/`push`/`pop`/`set_small` in an inherent
`impl DefaultEval` block. Trait calls then resolve to the no-op defaults:
the accumulator was never pushed (every eval rebuilt — nps 848k → ~660k),
the tier never engaged. **Node counts were bit-identical** (rebuilds return
the same values), so bench fingerprints, tests and determinism all stayed
green — only nps and `wdl acc: … rebuilds` (→ ~100%) showed it. Numbers in
this entry are unaffected (the SPRT and the 211008 bench predate the
restructure); the σ=100 bench readings taken on the broken binary were
discarded, not recorded. Fixed at the source by rejoining the trait block and
keeping `quad_eval` inherent with a comment saying why.

Lesson written on the seam: **`chess bench`'s `wdl acc: … rebuilds` line is
the canary for evaluator wiring — read it after every eval change.** And
`strings` on the binary beats re-reading source when behavior disagrees with
code: the missing `DBG` literal proved the binary predated the source before
the structural cause was found. (A stale-binary theory was considered and
rejected: the binary faithfully compiled broken-but-value-identical source.)
