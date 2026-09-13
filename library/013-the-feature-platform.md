# 013 — The feature platform

How a piece of information gets wired into the search, how it is screened, and
what the screen can and cannot decide. `library/004` says *what* the price
should carry; this says *how you add one and find out*.

## The idea in one paragraph

The search does not count plies, it spends a **budget**. Every child is charged
a price in milli-plies, and every hand-written search heuristic — LMR, LMP,
extensions, the SEE filter — is a special case of that one number. So bringing
new information to the search means adding a **term to the price**, and the
project's bet is that we will keep doing that: is this a check, was the last
move a capture, how far from the PV are we, is this child about to fall into
quiescence. That only stays cheap if adding one is a single line.

## Adding a feature

One declaration in the `features!` block in `src/search.rs`:

```rust
/// Doc comment. This is what `chess features` and `chess params` print.
c_see: Gap, Lin, Gated, 4000, -4000, 20000,
      |c| !crate::eval::see(c.b, c.mv, 0);
```

That generates the coefficient, its box, its registry entry, its UCI spin under
`--features tune`, the extraction, the lazy gating and the price term. Nothing
else needs editing. `chess features` lists the surface.

| column | meaning |
|---|---|
| **group** | `Sigma`, `Gap` or `Cost` — which term of the allocation law the feature estimates |
| **kind** | `Lin` (coefficient x value; an indicator is `Lin` over 0/1) or `Log` (x `log2`, in sixteenths) |
| **cost** | `Free` = register arithmetic, always computed. `Gated` = expensive, computed **only** while the coefficient is non-zero |

### The group is the principled part

`library/004` derives the budget a child deserves as

```text
b ~ (PLY/gamma) * [ log2(sigma) - log2(gap) + log2(C) ]
```

with gamma = 0.126/ply measured independently. So a feature earns its place by
estimating **sigma** (how wrong a shallow search of this child will be), the
**gap** (how far behind the running best), or **C** (what an error here costs
at the root) — and its group **fixes its sign before it is measured**. That is
what makes a screen a test rather than a fishing trip: a feature whose measured
sign fights its group is telling you the model is wrong somewhere, and that is
information too.

It also says the coefficients are not independent. They are tied together
through one constant, which is why a feature is a claim and not just a number.
Nothing in the code enforces the group. It is documentation, and it is the
documentation that matters most.

### What is deliberately *not* a feature

- **`c_qdrop`** is a function of the *price*, not of the position: it fires
  when the price lands a child at or below zero budget. Features answer "what
  is this move like"; shaping answers "where did the price land". It is applied
  as a separate stage in `price()`.
- **The five legacy terms** (`c_rank`, `c_gap`, `c_hist`, `c_rank_depth`,
  `pv_discount`) share **one** rounding shift. Moving them into `features!`
  would round each separately and change the tree, so they stay hand-written
  and bit-frozen. They are not conceptually special — `c_rank` is a `Gap` term
  under the same taxonomy.

## What a dormant feature costs — measured

The coefficients are runtime fields; the tuner and UCI mutate them, so the
compiler **cannot** fold a dormant feature away and is not expected to. A
`Free` feature costs a multiply-add on a register; a `Gated` one costs a
compare-and-branch on a field that is always zero, perfectly predicted, body
never executed.

Interleaved pinned A/B, `taskset -c 5`, deploy config, node counts identical
(396299 both), 4 dormant features vs 7 plus the whole macro platform:

| pair | 1 | 2 | 3 | 4 | 5 | mean |
|---|---|---|---|---|---|---|
| diff | −2.60% | +0.48% | +0.17% | −0.47% | +3.69% | **+0.25%** |

**Unresolvable** against this box's ±2-6% spread. Declare features freely; the
cost of carrying a dormant one is below the noise floor. Literal zero would
need compile-time specialisation and this says it is not worth it.

## Screening a feature

```bash
chess tune compare --labels <f> --nodes 15000 --wdl <net> --a c_x=0 --b c_x=V
```

Three seconds a pass at n=1000, paired SE ±2.0 cp. Four rules, each of which
has already caught something:

1. **Sweep, do not spot-check.** A monotone trend over four or five values is
   far harder for noise to fake than one `|t| = 1.5` point. `c_see` was
   monotone over five and then plateaued; `c_offpv` was not monotone at all,
   and that was the tell.
2. **Compare against a *tuned* baseline, not the default.** `c0` is a flat
   price carrying no information and it is worth **−2.2 cp on its own** — the
   price list has never been fitted (ROADMAP 1b), so some of any candidate's
   win is just that. Sweep `c0` first and beat the best point on it.
3. **Do not match depth.** An earlier version of this protocol did, and it was
   wrong: depth is an *output* of the search, not an input. If a feature works
   by pruning junk and searching deeper, holding depth fixed deletes the
   mechanism. Hold the *budget* fixed — that is the resource — and let depth
   land where it lands. See LEDGER 062, corrected.
4. **Read both channels.** Regret and depth-at-fixed-nodes disagree usefully.
   Deleting the check extension **gains 0.49 plies and loses 4.5 cp**, which is
   the sharpest available proof that this objective is not merely rewarding
   depth.

## A feature in the move loop can never reach rank 0

The main line pays only the fare and is never priced, so **a `features!` entry
cannot affect the first move at a node.** Any information that should also
apply to the PV has to enter as a node-level term in `Params` — the way
`check_extension` does — not as a feature.

This is not theoretical. `c_gives_check` measures at zero while the node-level
`check_extension` is worth +2.3 cp to delete, and the two fire on nearly the
same set one ply apart. The obvious excuse (they are redundant) was tested and
failed: with `check_extension = 0` on both sides, `c_gives_check` still buys
nothing. LEDGER 065.

## What the screen cannot decide, and the two ways it lies

**It cannot price the feature's own cost.** `c_see` searches 43% fewer nodes at
depth 9 and runs **10% slower per node**, because SEE now executes on every
priced child. A node budget cannot see that at all. Neither can the work meter
as it stands: `work.rs` prices the main `price` zone at **4.29 ns**, calibrated
when that zone did `move_gain` and a history read and nothing else. Both
currencies are blind, in the same direction. Re-emit the price table with
`--features nodeprof` after any change that alters what an operation does.

**It cannot resolve many parameters at once.** `library/004`: root regret gives
one scalar per position, zero wherever two price lists agree, and its noise
floor does not improve as parameters are added. Twelve was already the edge and
the surface is now larger than that. **Screen one at a time; do not joint-fit.**
Joint fitting needs the dense per-child corpus (STATE "Next", `library/005`
category C), and that is the precondition for the feature set ever growing past
a handful — not a faster way to do the same thing.

**Right now it cannot find anything at all.** On the relabelled corpus at the
current default, *every* single-coordinate move is worse — five new features
and six already-fitted knobs, both directions, only `c_hist` very slightly
negative and tiny. The self-referential teacher was tested as the explanation
and is not producing a detectable effect. So the instrument is at its floor
here, and the next move is the dense per-child corpus rather than another
feature. LEDGER 065.

**And a proxy winning is not a result.** `c_see` is the first feature to go all
the way: −4.6 cp on the proxy, then **+51.3 Elo [+33, +70]** over 484 games at
8+0.08 (LEDGER 064). That is the bar. Everything else in `chess features` that
reads `dormant` has not cleared it.
