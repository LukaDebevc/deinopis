# 065 · Five features measure at zero — and so does every existing knob

_2026-09-07. `chess tune compare`, n=1000, 15000 nodes, paired SE ±1.96 cp, on
the corpus **relabelled from the promoted `c_see` engine** (teacher hash
d775c104, 400k nodes, top 5). Baseline: regret 39.21 ± 1.96, depth 7.77,
11043 nodes. Engine b986f5b + working tree._

## The five features

All declared through `features!`, all screened one at a time against the
default, both signs, several magnitudes. **None pays.**

| feature | group | best value tried | b − a |
|---|---|---|---|
| `c_qdrop` (the quiescence cliff) | shaping | −1500 | +1.626 ± 1.674 |
| `c_ply` (plies elapsed from the root) | Cost | +30 | +1.890 ± 1.748 |
| `c_depth` (standalone `log2` budget left) | Cost | −300 | +1.615 ± 1.692 |
| `c_incheck` (node is in check) | Cost | −500 | −0.086 ± 0.969 |
| `c_gives_check` (move gives check) | Cost | −1500 | +1.906 ± 1.549 |

`c_gives_check` deserves its own line because the obvious excuse was tested and
failed. It could have been redundant with the node-level `check_extension`,
which fires one ply later on nearly the same set. **It is not**: with
`check_extension = 0` on both sides, `c_gives_check` still measures at zero
(+0.284, +0.270, −0.290 at −500/−1000/−2000, all inside noise). The node-level
extension meanwhile is still worth **+2.326 ± 1.516** to delete.

[INFERENCE] The structural reason a move-level check term cannot do the
node-level extension's job: **a feature priced in the move loop can never
affect rank 0.** The main line pays only the fare and is never priced, so any
information that should also apply to the PV has to enter as a node-level term,
not as a `features!` entry. That is a real limit of the platform and is now in
`library/013`.

## The control, which is the actual finding

Before recording five negatives, the boring explanation was tested: **is the
default simply a local optimum?** Same corpus, same budget, moving the knobs
that were already there:

| knob | down | up |
|---|---|---|
| `c0` | — | +0.900 ± 1.767 (200) |
| `c_rank` | +4.737 (150) | +3.062 (400) |
| `c_rank_depth` | +2.998 (100) | +0.201 (350) |
| `c_hist` | **−0.534 ± 0.400** (0) | −0.202 ± 0.759 (300) |
| `skip_below` | +5.574 (−256) | +5.888 (0) |
| `pv_discount` | +1.151 (500) | +1.638 (2000) |
| `check_extension` | +2.326 (0) | — |
| `max_base` | +3.30 (0) | **inert** ≥1000 |

**Every direction is worse.** The only exception is `c_hist`, at −0.53 ± 0.40
with a standard error that small because almost no positions change.

So the honest statement is not "five features failed". It is: **at this
operating point the objective cannot find an improvement in any direction it
can resolve, old knob or new feature alike.** That is the 4.1-cp floor from
LEDGER 061 biting, exactly where `library/004` said it would.

### `max_base` is inert, and was already documented as such

`max_base` at 1000 and at 20000 give **bit-identical** output — same regret,
same depth, same node count. Checked as an instrument fault first (`c_rank` as
a positive control does move the numbers, so the override plumbing is fine),
and checked with `c_see` both on and off, so it is not a `c_see` effect. The
answer was already in the field's own doc comment: *"raising it further changes
nothing at all, because the price formula saturates below it."* At 0 it does
bind (+3.30). Nothing new; recorded so the next person does not re-chase it.

## The self-referential teacher does not explain it

The teacher is now the student's own price list at 26x the nodes, which should
pull the objective's minimum toward the default and make every deviation look
bad — precisely the pattern above. One manipulation, student identical, only
the teacher's policy changed:

| arm | teacher **without** `c_see` | teacher **with** `c_see` |
|---|---|---|
| `c_qdrop=-1500` | +3.018 ± 1.689 | +1.626 ± 1.674 |
| `c_ply=100` | +2.392 ± 1.781 | +2.349 ± 1.779 |
| `c_depth=-300` | +2.728 ± 1.673 | +1.615 ± 1.692 |
| `c_incheck=-500` | −1.176 ± 0.941 | −0.086 ± 0.969 |
| `c_gives_check=-1500` | −0.055 ± 1.443 | +1.906 ± 1.549 |
| `c0=200` | +1.693 ± 1.748 | +0.900 ± 1.767 |

If self-agreement drove the result the right column would be systematically
more positive. Four of six went the other way. **No systematic direction, so
the confound is not producing a detectable effect here.** It remains real in
principle (STATE "Next", `library/012`) and this is not a licence to ignore it
— but it is not the explanation for this sweep.

Corroborating that from the other side: `c_see` won by **−4.62 ± 1.80** against
a teacher that did **not** have it, and by −5.42 against one that did. If the
objective merely rewarded agreeing with the teacher, the first number could not
exist.

## Decision

- Five negatives recorded. All stay declared at zero — a dormant feature costs
  less than the box can measure (`library/013`), and a negative result you can
  re-run in three seconds is worth keeping.
- **Single-coordinate screening on root regret is exhausted at this operating
  point.** This is the strongest evidence yet for STATE's per-child
  supervision item: the next move is not another feature, it is an objective
  that can resolve more than one thing at a time.
- Do not read this as "the feature idea does not work". `c_see` came through
  the same pipe and is worth **+51.3 Elo**. What is exhausted is the
  *instrument*, at this point, not the approach.
