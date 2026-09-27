# 119 — An uncertainty head on the frozen net predicts RFP fail-lows well and loses every game test

**Date:** 2026-09-27 · **Base:** cp-0009 (`71be205`) · **Net:** `m1-b1.nnue` (sha `ce1f2656`), frozen
· **Run dirs:** `the GPU cluster:<cluster scratch>` (dump, labels, fits),
`…/20260927-rfpl/` (screen: `cells.log`, PGNs, binaries `rl`/`rl2`)
· **Scripts + outputs:** `library/017-long-run-map/probes/rfp-learn-20260927/`
(`p0.sh`, `p1.py`, `p2b.py`, `p2c.py`, `fitexport.py`, `chk.py`, `*.out`,
and `rfp-learn.patch` = the losing search arm, applies to `fc743f0`).

## Question

Luka's proposal: lock the net and add two outputs, (a) uncertainty about its
own eval, (b) P(best move is a capture), and see whether the search can use them.

## What was built (kept, dormant, bench 151742)

- `--features nodedump` (`src/nodedump.rs`): samples nodes at the whole-node
  pruning block (non-PV, not in check, main search) from real games.
- `chess nodelabel`: per node, qsearch value, one 64k-node search (the score of
  every completed iteration, best move) and `WdlNet::inner` (post-mid 32,
  post-up 64, post-l2 64, logits 3) as f32.
- P0: 2000 self-play games at 1+0.01 (dump build vs itself: −0.35 [−10.3, +9.6]),
  **493,182 nodes**, 54% at depth 1. Split by dump file (disjoint games):
  train / early-stop / held-out.

## Probe results (held-out, 91k rows)

**σ (pre-registered):** target ln(|v_d − static| + 1), `v_d` = the node's own-depth iteration.

| inputs | r² | Laplace NLL gain |
|---|---|---|
| free hand features (material, depth, \|static\|, D, improving, ply) | 0.078 | +0.017 |
| + net l2 (64) | 0.135 | +0.045 |
| + all 163 net values | **0.194** | +0.100 |
| + \|q − static\| (costs a qsearch) | 0.549 | +0.426 |

Passes its +0.02 bar. p1's own Laplace fits never converged (bias started at
σ = 1 cp); `p2b.py` refits exactly. **But symmetric σ is the wrong object for
RFP**: q ≥ static by stand pat, so the error is one-sided. Nodes with q > static
(n 197,772) have mean error **+532 cp** and P(error < −100) **0.005**; q = static
(n 277,599): mean +10, P(< −100) **0.154**. Large σ̂ marks the *safest* nodes to
prune. A σ-scaled margin was worse than today's in replay (p1, P2).

**P(best move is a capture):** AUC 0.607 hand → **0.697** with the net → 0.912
with the q gap. The net adds a little; qsearch dominates. Not taken further.

**One-sided classifier P(v_d < β)** on candidates (static ≥ β, depth ≤ 7),
replayed against today's `static − 75·d ≥ β` (43,239 prunes, 334 fail low):

| arm | AUC | wrong at today's prune count | prunes at today's wrong count |
|---|---|---|---|
| margin only (control ≈ today) | 0.828 | +1.8% | −0.4% |
| H: margin + free features (no D) | — | −30% | +12% |
| Z: H + net l2 (the head) | 0.895 | −61% | +21% |

## Games (R3, 1+0.01, 12,000 each, conc 32, both sides one binary, flag only)

Rust logits checked equal to Python's on 68 held-out rows (`chk.py`, max diff 0.0000).

| cell | rule | bench | Elo [95%] |
|---|---|---|---|
| H_SAMEP | H, threshold = today's prune count | 156754 (+3.3%) | **−6.8** [−10.9, −2.7] |
| H_SAMEW | H, threshold = today's wrong count | — | **−18.4** [−22.5, −14.3] |
| H_PERD | H, today's prune count *at each depth* | 160548 (+5.8%) | **−17.9** [−22.0, −13.9] |
| Z_SAMEP | Z, today's prune count | 164078 (+8.1%), nps −9% | **−24.3** [−28.4, −20.1] |
| Z_PERD | Z, per depth | 186033 (+22.6%) | **−27.1** [−31.3, −22.9] |

## Reading

1. **The proxy failed its severity test.** Fewer fail-lows at the same prune
   count, and at the same count per depth (`p2c.out`: depth-weighted wrong
   prunes −11% H, −56% Z at identical Σ1.88^d), yet every arm loses. A prune is
   not worth `EBF^d`: the classifier refuses to prune exactly the tactical nodes
   whose subtrees are large for their depth, so the tree grows (bench +5.8% at
   equal per-depth counts). [INFERENCE — the per-node subtree size was not
   measured.] The margin rule's fail-lows are evidently cheap ones.
2. **A head is not free.** Z's l2 read costs −9% nps, mostly the re-evaluation
   at nodes whose static eval came from the hash.
3. **What this does NOT kill:** a pruning rule trained on the *cost* of an error
   (the change in the parent's result, per unit of subtree saved), which is what
   LC-1's per-child corpus labels. Fail-low-at-own-depth is the wrong label.
   Nor does it kill σ in the *price list*, where it enters as (σ/gap)² with a
   sign fixed by `library/004`; that was not tested here.

## Decision

`rfp_learn` removed from `search.rs` (patch kept). The dump/label tooling stays.
Do not re-run "better RFP by fail-low classification" without a cost-weighted label.
