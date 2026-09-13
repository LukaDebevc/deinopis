# 063 · Quiescence depth is a property of the position, not of the budget

_2026-09-07. `chess tune compare`, 1000 fishpack positions, paired SE ±2.02 cp
at 15000 nodes and ±0.73 at 60000. Engine b986f5b + working tree. Run to decide
one design question before writing any code for it._

## The question

The plan was to **fold quiescence into the priced search**: one node function,
one currency, with a q-node charged the same `fare` as any other node so the
budget starts binding below the main search's horizon. That is the aesthetically
principled version — quiescence stops being a separate hand-written routine and
becomes the same node at a small budget.

It rests on an assumption nobody had checked: **that quiescence depth has slack
worth allocating.** If quiescence runs to 15 plies and most of that is waste, a
budget can take it back. If it self-terminates at 5, a budget has nothing to
spend and the fold can only share code, not currency.

`Params::q_max_ply` caps quiescence at N plies below the point the main search
ran out of budget. 64 is unreachable and reproduces an uncapped quiescence
exactly (`bench` 334110, unchanged).

## The answer, at two budgets 4x apart

| cap | 15000 nodes: b − a | differ | 60000 nodes: b − a | differ |
|---|---|---|---|---|
| 1 | +2.234 ± 1.788 | 315 | — | — |
| 2 | +2.395 ± 1.458 | 193 | — | — |
| 3 | +0.833 ± 0.889 | 90 | — | — |
| 4 | +0.087 ± 0.457 | 34 | **−1.145 ± 0.726** | 71 |
| 6 | −0.027 ± 0.035 | 2 | −0.227 ± 0.202 | 10 |
| **8** | **+0.000 ± 0.000** | **0** | +0.012 ± 0.012 | 1 |
| 12 | +0.000 ± 0.000 | 0 | +0.000 ± 0.000 | 0 |
| 16 | — | — | +0.000 ± 0.000 | 0 |

**Quiescence deeper than 8 plies does not exist.** Capping there is
bit-identical over 1000 positions at both budgets. All the value is in plies
1-3; plies 4 and beyond are worth −0.03 cp and touch 2 positions in 1000.

The main search's horizon grew 2.62 plies between the two budgets (7.03 →
9.65). **The quiescence tail did not move at all.** That is the finding: q-depth
is set by how many captures are available on the contested squares, which is a
property of the position, and not by how much compute the search has.

Capping also frees no nodes — cap 4 at 15000 moves the node count by −0.1%.
Deep quiescence is a thin tail, not a bulk cost.

## What this does to the fold

**Corrected after review.** The first version of this entry said "the ambitious
fold is dead", which is broader than the measurement supports. What is measured
is one thing only: **charging a fare to limit quiescence *depth* has nothing to
allocate.** Every setting that changes anything (cap ≤ 3) is measurably worse,
and every setting that is safe (cap ≥ 8) changes nothing. That kills one design
— a budget that binds on q-depth — and it says nothing about the rest.

In particular it does **not** kill the idea that the quiescence boundary is
*information*. That is a different proposal and it is untested:

0. **Whether a child would drop into quiescence is itself a feature.** The
   boundary at zero budget is a **cliff** — a child left with 1 unit gets a
   full recursive node, one left with 0 gets a capture-only search — and
   nothing in the price list knows the cliff is there, while **27.9% of priced
   children land on the far side of it** (`library/008`). A term that pulls
   marginal children back over it, or pushes them across decisively, is a
   knob nobody has turned. `Pricing::c_qdrop`.

What is left besides that is worth doing and is a different change again:

1. **Fold the move-admission rules, not the depth.** Quiescence's real
   decisions are which moves it lets through — the SEE ≥ 0 filter and delta
   pruning — and both are already price-shaped. LEDGER 062 is the evidence
   this is the right direction: the SEE filter, expressed as a price and let
   loose in the *main* search, is the largest single-parameter win measured on
   this objective. The rule that only lived in quiescence turned out to be
   worth budget above it too.
2. **Fold the hash.** Quiescence never probes or stores, and 26.9% of q-nodes
   repeat a key quiescence itself already visited (LEDGER 059). One node
   function gets that for free. This is a speed change, not a control change,
   and it is measured with a wall clock, not with this objective.

## Loose end worth pulling

At 60000 nodes, capping quiescence at **4** plies measures as an *improvement*:
**−1.145 ± 0.726 cp**, |t| = 1.6, on 71 positions. It is not resolved and the
sign flipped from +0.087 at 15000, so treat it as a hint rather than a result.
If it holds at larger `--n`, it says deep quiescence is actively harmful — the
search chasing a long forcing line with no static-eval sanity check anywhere
along it. That is cheap to settle: the sample is nested, so raising `--n` costs
only the new labels.

## Decision

- `q_max_ply` stays in the registry at 64. It changes nothing and it is the
  instrument that answers this question again after any change to the eval or
  the filters.
- The fold proceeds as **shared move admission + shared hash + the boundary as
  a feature**, not as shared currency. The budget stops at zero, where it
  already stopped — but where it stops is now something the price can see.
- Re-run this sweep after the SEE price ships. It changes which moves reach
  quiescence, so it can change how deep quiescence goes.
