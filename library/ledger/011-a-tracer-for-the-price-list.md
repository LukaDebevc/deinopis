**011 · A tracer for the price list** · 2026-08-25 · `src/trace.rs`, `chess trace`
Per-decision record of what the price list actually did to every child: price
charged, budget inherited, rank, whether it was priced out, whether its reduced
search failed high and had to be redone. All of it is computed from what
alpha-beta already knows — no oracle, no labels, no reference search. Bench is
byte-identical (567321 at depth 9) with it compiled in.
**Result:** four things, none of which needed a teacher.
1. **Late move pruning is not gone — 008 turned it back on by accident.**
   `skip_below` fires in **0.00%** of priced children at `max_base = -2000` and
   **59.57%** at 008's `+1000`. STATE.md and LEDGER 007 both said "0 of 14000
   positions"; both were measured before 008 and are now corrected at source.
   008's -0.443 cp was mostly LMP returning, not a clamp loosening.
2. **30.45% of all nodes are re-search waste** (14.80% at 007's settings): a
   priced-down child whose reduced search failed high and had to be redone at
   full budget. The search proves its own price wrong, for free. Largest
   recoverable pool measured in this engine.
3. **The mispricing lives at ranks 2-7.** Re-search rate by rank: 0.05% (r1),
   **4.41% (r2-3)**, 1.46% (r4-7), 0.30% (r8-15), 0.18% (r16+). Rank 1 is never
   mispriced; rank 16+ is already correctly refused. A better price for rank 30
   is worth nothing.
4. **The ordering confound has a number.** Cutoff-at-first-move moves
   63.75% -> 57.05% between arms — comparable to the effect being measured, so
   it must be reported alongside any allocation result.
**Hypothesis raised and killed:** that `c_gap` is structurally dead because
`gap` is negative for every *root* child and `price()` takes `max(gap,0)`.
Measured across the whole tree: gap > 0 in **50-89%** of priced children. The
feature is live; `c_gap = 0` stands for the reason `library/004` gives (r2=0.032
noise), not for a coding reason. Write-up: `library/008-what-the-price-list-does.md`.

---
