# 008 · What the price list actually does

_2026-08-25. `chess trace` — `src/trace.rs`, one position at a time, every
pricing decision recorded. Built because LEDGER 007/009/010 are a sequence of
discoveries that the one-number objective was measuring something other than
what we thought, and the way you catch that class of error is by looking._

All metrics here are **free and oracle-free** — computed from what alpha-beta
already knows. `library/005` argued the existing objective is a recall metric
and that nearly every precision metric costs nothing. This is that claim cashed.

## The correction: late move pruning is not gone. It came back in 008.

`STATE.md` and LEDGER 007 said `skip_below` fires in **0 of 14000 positions**.
That was true when it was measured. It is no longer true, and nothing recorded
the change.

Same position, same everything, sweeping only the ceiling:

| `max_base` | priced out | dropped to qsearch | nodes | score |
|---|---|---|---|---|
| −2000 (007's) | **0.00%** | 43.50% | 54878 | cp −1 |
| 0 | 53.89% | 33.68% | 40069 | cp −18 |
| **1000 (008's, current default)** | **59.57%** | 27.94% | 36886 | cp −18 |

LEDGER 008 loosened `max_base` from −2000 to +1000 and read the result as "the
old LMR clamp was the binding constraint and it was too tight." That reading is
incomplete. The ceiling is what stops a price from consuming a child's whole
budget, so raising it is precisely the thing that lets a child fall below
`skip_below`. **008's headline change silently switched late move pruning back
on**, and the −0.443 cp it bought is mostly that, not a gentler reduction curve.

Roadmap item 3 — "give the price a term that can consume a whole budget, so LMP
exists again" — is therefore already done, by accident, and needs rewriting into
"decide whether the LMP we now have is the LMP we want."

## The asymmetry, a third independent time

4000 labelled positions, 60k student nodes, reverting only the ceiling:

```
b − a = +0.259 ± 0.523 cp     depth 9.81 → 9.00 (−0.80 plies)
```

Depth sees an 0.80-ply effect. Regret does not resolve it at all. 007 found this
with `nopolicy`, 010 found it across a budget ladder, and here it is on the one
parameter change the project actually shipped. **The thing 008 bought was depth,
and the objective it was selected on could not see it.**

## Nearly a third of the search is spent searching subtrees twice

When a child is priced down and its reduced search fails high, the search must
redo it at full budget. The price was wrong, the search proved it itself, and we
paid twice to find out. No teacher, no labels, no oracle — this is a
**per-decision precision signal sitting in the move loop already.**

Bench corpus, depth 9, whole tree:

| | `max_base` = −2000 | `max_base` = 1000 (current) |
|---|---|---|
| re-search rate | 0.11% | **2.25%** |
| **re-search waste** | 14.80% | **30.45%** |
| utilisation | 4.89% | 6.08% |
| priced out | 0.00% | 57.04% |
| cutoff at first move | 63.75% | 57.05% |

The current price list saves 19.1% of nodes across the corpus and burns
**30.45% of what remains on re-searches**. That is the largest single pool of
recoverable waste anyone has measured in this engine, and it needed no oracle
to find.

## Where the price is wrong — a narrow band, not everywhere

Re-search rate by rank, one tactical position, depth 9:

| rank | children | re-searched | rate |
|---|---|---|---|
| 1 | 12743 | 6 | 0.05% |
| **2–3** | 20551 | 907 | **4.41%** |
| 4–7 | 35572 | 520 | 1.46% |
| 8–15 | 68779 | 207 | 0.30% |
| 16–31 | 108419 | 200 | 0.18% |
| 32+ | 15845 | 30 | 0.19% |

Monotone decay after a sharp peak at ranks 2–3. Rank 1 is essentially never
mispriced; ranks 16+ are already correctly priced out. **All the headroom is in
ranks 2–7**, which is where the ordering is genuinely uncertain and where a real
deviation feature would pay. A better price for rank 30 is worth nothing.

## The ordering confound is real and now has a number

`library/005` warned that prices change history, history changes ordering, and
ordering is what alpha-beta's efficiency rests on — so every allocation
experiment is partly an ordering experiment. Cutoff-at-first-move moves
**63.75% → 57.05%** between the two arms above. That is not a small confound
riding along; it is comparable to the effect being measured. Report it alongside
any allocation result.

## One hypothesis raised and killed

The root's per-child table shows `gap` negative for **every** root move, and
`price()` computes `log2(max(gap,0)/gap_unit + 1)`, so a non-positive gap zeroes
the `c_gap` term whatever `c_gap` is set to. That suggested `c_gap = 0` was not
a measurement of a weak feature but of dead code.

Measured across the tree: **gap > 0 in 50–89% of priced children** (mostly
70–85%). The term is live. The root is special — `is_pv` holds for every root
child and `best_score` is still moving — and it is ~35 children out of ~84000.
`c_gap = 0` stands exactly as `library/004` explains it: a live feature that
predicts r² = 0.032 of the true gap, i.e. noise.

## Using it

```
chess trace --fen <fen> --depth 8            # one position, full report
chess trace --depth 9 --ply 2                # bench corpus, per-child to ply 2
chess trace --set max_base=-2000 --vs max_base=1000
                                             # two arms, side by side, and
                                             # whether the answer changed
```

The two-arm mode answers the question that matters for a pruning change — *did
the cheaper search still find the same move and the same score* — rather than
only how many nodes it saved:

```
TOTAL (12 positions)   701400   567321   -19.1%
best move changed in 1/12;  pv diverges early in 5/12
|score delta|: median 7 cp, max 62 cp
```

Cost: one predictable not-taken branch in the move loop. **Bench is byte-identical
at 567321 nodes** with the hooks compiled in, which is the only reason it was
allowed into `search.rs` at all.
