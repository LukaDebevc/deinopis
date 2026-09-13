# 005 · What to measure: metrics for search allocation

_2026-08-25. Written against LEDGER 007/008, while the 008 SPRT was in flight.
Companion to `004-what-the-price-should-carry.md`, which measures the features;
this one is about the objectives._

## The hole in 007, named

LEDGER 007's negative result is usually stated as "the objective is
asymmetric". It is worth restating in the standard terms, because the name
tells you what is missing:

> **Root regret is a recall metric. This project has no precision metric.**

Treat each priced child as a classification. A child is **relevant** if
searching it fully would change the parent's conclusion — it raises alpha or
causes a cutoff.

|  | we spent budget | we priced it out |
|---|---|---|
| **relevant** | correct | **miss** — over-pruning |
| **irrelevant** | **waste** — under-pruning | correct |

* **Recall** = relevant children we actually searched. Failing it makes the
  search play the wrong move. Regret sees this loudly: scaling coefficients by
  2.4 measures +4.38 ± 0.33 cp, 13 sigma.
* **Precision** = searched children that were relevant. Failing it costs nodes
  and nothing else at the root of an easy position. Regret is blind: deleting
  the *entire* pricing policy measures +0.006 ± 0.283 cp, for something worth
  80-150 Elo.

007's mechanism explanation is right and this is the same statement: at fixed
nodes under-pruning only makes the search shallower, and a shallower search
still agrees about the root move because that choice is dominated by the first
few plies, the TT, ordering and quiescence.

**The useful consequence: nearly every metric that costs nothing is a precision
metric, and precision is exactly what is missing.** The cheapest fix is not a
better label set. It is to report a second number the search already computes.

**And a rule that follows from the table:** every single-axis metric here has a
trivial degenerate maximiser. Precision alone is maximised by searching one
move everywhere; recall alone by searching everything. **Never optimise one
without constraining the other.** Any objective quoted as a single number is
either a pair in disguise or a mistake.

## What that predicts about the sweep

A recall-only objective cannot find an interior optimum in the pruning
direction. It can only push toward more pruning until recall starts to fail, so
the price list it selects sits **on the boundary of what the metric tolerates**,
not at an optimum.

That is falsifiable with the sweep that already exists: take another step in
the same direction (`max_base` past 1000, or all coefficients scaled up) and
see whether the proxy keeps improving while the SPRT does not. If it does, the
boundary is confirmed and every number the proxy produced in that direction is
an upper bound rather than an estimate.

## The catalogue

`gran` = signal per position. `O` = over-pruning visible, `U` = under-pruning
visible.

| metric | gran | oracle | O | U | gamed by |
|---|---|---|---|---|---|
| **A · free, no oracle** | | | | | |
| depth reached at fixed nodes | 1 | none | – | **yes** | prune everything |
| effective branching factor | 1 | none | – | **yes** | prune everything |
| node utilisation | 10^5 | none | – | **yes** | search one move everywhere |
| cutoff-at-first-move rate | 10^4 | none | – | weak | (measures ordering, a confound) |
| re-search / aspiration failure rate | 10 | none | – | weak | wider windows |
| PV instability across iterations | 10 | none | weak | – | never change your mind |
| backup (Bellman) residual along the PV | 10 | none | weak | – | never change your mind |
| **B · cheap oracle (labels we already make)** | | | | | |
| root regret (current) | 1 | 400k/move | **yes** | **no** | match the oracle's mistakes |
| top-k agreement | 1 | 400k/move | yes | no | as above, plus near-tie noise |
| depth-sensitivity-**weighted** regret | 1 | 2 budgets | yes | **yes** | — (but see the trap below) |
| settling budget `b*(m)` | ~8 | node ladder | **yes** | **yes** | — |
| nodes-to-fixed-fidelity | 1 | ladder | yes | yes | — (the only Elo-convertible one) |
| **C · full-width oracle** | | | | | |
| minimal-tree precision / recall | 10^5 | full-width `V_d` | **yes** | **yes** | — |
| counterfactual pruning audit (ROC) | 10^5 | full-width `V_d` | **yes** | **yes** | — |
| bound tightness at ALL nodes | 10^4 | full-width `V_d` | yes | yes | — |
| **D · external** | | | | | |
| tactical suite nodes-to-solution | 1 | free (EPD) | yes | yes | overfits the suite |
| stronger engine as labeller | 1 | free (`.ladder/bin`) | yes | yes | **wrong oracle — see below** |
| **E · perturbation** | | | | | |
| decision variance under ordering perturbation | 1 | none | — | — | diagnostic only |
| budget-perturbation sensitivity | 1 | none | — | — | insensitivity |

### Notes on the ones that carry weight

**Depth at fixed nodes** is the direct complement of regret and it is already in
`SearchResult.depth`. Under-pruning *is* "less depth for the same nodes"; there
is no proxy step. Plumbing it into `tune::Report` alongside `regret` closes
007's blindness for the cost of a struct field. It is trivially gamed on its
own, which is the point — it is the constraint, and regret is the objective.

**Node utilisation** — the fraction of searched children whose returned value
raised alpha or caused a cutoff — is the free, dense stand-in for minimal-tree
precision. A few lines in the move loop, one counter, no labels, 10^5 samples
per position. Under-pruning tanks it directly.

**Minimal-tree precision/recall** is the flagship because it decomposes the two
failure modes at the node level rather than the position level, and because its
absolute value is the **headroom**: |actual| / |minimal| bounds what any
allocation policy can win at a fixed evaluator. The Knuth-Moore minimal tree
assumes perfect ordering, so it is a floor no real search reaches — precision
will never be 1 and the number to watch is the ratio between price lists, with
the absolute value read only as "how much room is left".

**Settling budget** `b*(m)` = the smallest budget after which a root move's
value stays within ε of the reference. From one node ladder you get, per move,
`max(0, b* − b_spent)` (under-spend, causes misses) and
`max(0, b_spent − b*)` (over-spend, pure waste). **Both Pareto axes from one
measurement, in budget units**, and the ladder tool that computes it already
exists (`examples/gapstat.rs` in the 004 scratch lab).

**A stronger engine is the wrong oracle here**, and the reason is worth
recording because the option is tempting and the binaries are already in
`.ladder/bin`. Search control's job is to find the best move *under our own
evaluator*. Labelling against Stockfish conflates search-control error with
evaluator error and would tune the search to compensate for eval bugs — which
NNUE will then remove, invalidating the fit. It is the right oracle for eval
tuning and the wrong one for this. This is also the positive argument for the
full-width `V_d` oracle: it is correct precisely *because* it uses our
evaluator, and it is order- and policy-independent besides.

**Tactical suites** are genuinely external and free, and engines that ace WAC
are famously not stronger. Guardrail, never objective: if a price list makes
tactics collapse, that is information; if it improves them, that is not.

## Three traps

**1. Filtering on depth-sensitivity breaks the exchange rate.** 007's proposed
fix — label at two budgets, keep positions where the reference's best move
changed — restores sensitivity by changing the position distribution. The
`1 cp ≈ 25-30 Elo` conversion was derived on the game distribution and does not
transfer to a filtered subsample. **Weight rather than filter**: keep every
position, weight by depth-sensitivity, and the population mean stays
recoverable, so the units still convert. If filtering is preferred for
simplicity, the exchange rate must be re-derived on the filtered set before any
cp number is quoted as Elo.

**2. Node counts stop being an honest currency if pricing calls quiescence.**
004 proposes sourcing the gap from a qsearch of the child (r² 0.71 against
0.03). If that lands, a "node" in the priced search is no longer the same unit
of work as a node in the old one, and *every* node-denominated result —
including 007's `4x nodes ≈ 4.5 cp` — silently changes meaning. Switch to time,
or to a **weighted node count** that charges qsearch nodes at their measured
relative cost, which keeps determinism. Decide before the feature lands, not
after.

**3. Ordering is a confound in every allocation experiment.** A price list
changes which moves get searched, which changes history updates, which changes
ordering, which changes prices. Cutoff-at-first-move rate should be reported
alongside any allocation result so that "better allocation" can be
distinguished from "accidentally better ordering".

## Recommended order

1. **Plumb `depth` into `tune::Report` and refuse any candidate that regresses
   it.** One struct field. Closes 007's blindness today. Report the pair
   `(regret, depth)` at fixed nodes; neither number is an objective alone.
2. **Node utilisation counter in the move loop.** Free, dense, and it is the
   precision metric the project does not have.
3. **Settling budget from the existing ladder.** Gives under-spend and
   over-spend separately, in budget units, on the real position distribution.
4. **Weighted (not filtered) depth-sensitivity** in the label set, with the
   exchange rate re-derived.
5. **Minimal-tree precision/recall** from a full-width `V_d` corpus, when there
   is a day for it. This is also the headroom number: **under about 2x, the
   project's central bet has no room and that should change the roadmap.**
6. Tactical suite as a guardrail. Decision-variance decomposition when a result
   needs explaining rather than detecting.

## One correction to the 008 plan

008 proposes fixing the gap feature with SEE — "a one-line change and a rerun".
004 measured the candidates directly, and **SEE is not enough**: it moves r²
from 0.032 to 0.080, which will still measure at approximately zero and will
cost a labelling run to discover. The predictor that works is a **quiescence
search of the child, r² = 0.712** (a 1000-node search reaches 0.825, so qsearch
captures 86% of what more nodes buy). Gate it on remaining depth — near the
leaves a qsearch per child costs more than the subtree it prices — and consider
the free intermediate first: **use the TT value of the child when one exists,
`move_gain` otherwise.**
