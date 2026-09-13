# 003 — Search as budget allocation

**Date:** 2026-08-25 · **Code:** `search::Pricing`, `src/tune.rs` · **Status:** platform built, defaults untuned

## The reframe

The proposal was to stop building the search as "minimax, then cut away what we
do not want" and instead give each iteration a budget, charging a penalty for
stepping off the main line, with a floor and a ceiling on the penalty.

The first thing worth saying is that this is not as far from the existing engine
as it looks. **Late move reduction already is a budget allocator.** The currency
is log-depth, the price of a step off the main line is

```
r = 0.77 + ln(depth)*ln(rank) / 2.36
```

pulled from a hand-fit 64×64 table, and the floor and ceiling already exist as
`r.clamp(0, depth - 2)`. Late move pruning is the same price list with the
ceiling pushed past the remaining budget. So the change is not a new species of
search. It is the same machinery with the pricing made **explicit, continuous
and named** — which is the difference between a heuristic and a policy you can
tune, and between a table with no gradient and a function with one.

That framing also sets the honest expectation: this is not free Elo. The table
it replaces is hand-fit but not stupid, and the first version of a smooth
replacement should be expected to land near it, not above it.

## What the engine does now

Depth is no longer an integer. The search spends a **budget** in units of
`PLY = 128`, and each child of a node is charged

```
spend = fare + clamp(price, floor, max_base + max_slope*depth)
price = c0 + c_rank*L(rank) + c_gap*L(1 + gap/gap_unit)
           + c_hist*H + c_rank_depth*L(rank)*L(depth) - pv_discount      (L = log2)
```

inheriting `parent_budget - spend`, strictly less than the parent's. A child
left with a non-positive budget drops into quiescence; one left below
`skip_below` is not searched at all.

Twelve parameters, all in `Pricing`, all named in `Pricing::NAMES`, all tunable.

### Three things that are deliberate

**The bound is not for sale.** Beta cutoffs still terminate the move loop and
the transposition table still returns on a sufficient bound. Alpha-beta's power
is the *proof* that a subtree cannot affect the root; no allocation policy buys
that back, and softening it costs 5–10× in nodes immediately. This is the
specific reason the 1990s metareasoning engines lost — Russell & Wefald's MGSS\*
(1991), Baum & Smith's BPIP (1997), McAllester's conspiracy numbers (1988),
Korf & Chickering's best-first minimax (1996) all allocate computation by
expected value of information, and all pay per-node bookkeeping that exceeds
what they save when a node costs 200ns. MCTS/PUCT is the branch of that family
that won, and it won only because a GPU-scale evaluation makes a node worth
~1000× more, which pays for the bookkeeping. On six cores with PeSTO, a
best-first frontier is dead on arrival for exactly the 1990s reason.

So: **depth-first alpha-beta stays the executor; the budget is only the
allocator.** Pricing decides how much a relevant move gets, never whether a
provably irrelevant one is visited. Bookkeeping cost of the whole scheme is one
`move_gain` call and a handful of shifts per move.

**The gap is in centipawns, not in rank.** `L(rank)` is the old LMR feature and
it is a proxy — rank is informative only because a good ordering correlates it
with value. `gap` is the quantity itself: how far below the running best score
this move is predicted to land, from `eval::move_gain` (one piece-square delta
plus the victim's value, ~5 table lookups, no exchange sequence, no phase
taper).

**The ceiling grows with depth**, which is what `r.clamp(0, depth-2)` was, and
letting it push a child's budget past zero is *meant* to be what late move
pruning was. **It is not — measured.** `skip_below` fires in 0 of 14000 test
positions, and raising the ceiling from `depth-2` to `depth+1` and to `depth+2`
gives byte-identical results, which means the price never reaches the ceiling at
all: the formula's dynamic range tops out near 65% of the remaining budget at
rank 40, so a child always keeps a ply or more. Loosening the ceiling is worth
−0.44 ± 0.26 cp and then saturates.

So late move pruning was removed and **not** replaced, and no setting of the
existing twelve parameters brings it back — it needs a price term that can grow
with rank fast enough to consume a whole budget. That is a design hole, not a
tuning problem, and it is part of what LEDGER 006's −12.7 Elo bought.

### What was removed

| gone | replaced by |
|---|---|
| `lmr_table()`, the 64×64 hand-fit reduction table | `c_rank_depth` — the `L(rank)*L(depth)` interaction term is what that table consisted of |
| `lmr_min_depth`, `lmr_min_moves` | nothing: `L(rank)` is zero at rank 1, so the second move is free by construction rather than by a gate |
| `lmp_max_depth`, `lmp_base` | `skip_below` plus the ceiling — **in principle only, and it does not work.** See below. |
| "do not reduce captures" | nothing: a good capture has a small gap and therefore a low price |

Four parameters and a table out, twelve named parameters in. Kept, because they
are bound tests rather than allocation decisions and are orthogonal to the
question: null-move pruning, reverse futility, the transposition table, killers
and history, quiescence with delta pruning, aspiration windows.

## Why the price should be linear in log(gap)

This is the part that makes the functional form a prediction rather than a
guess, and it is not something we chose — it falls out of a standard result.

Allocating a fixed sampling budget across alternatives so as to maximise the
probability of correctly selecting the best is **optimal computing budget
allocation** (Chen et al.). Its solution allocates samples in proportion to

```
n_i  ∝  (sigma_i / delta_i)^2
```

where `delta_i` is alternative `i`'s gap to the best and `sigma_i` its sampling
noise. Search nodes are roughly exponential in depth, `n ≈ b^d`, so

```
d_i = log_b(n_i) = const + 2*log_b(sigma_i) - 2*log_b(delta_i)
```

and therefore the **reduction** — the depth a move gives up relative to the main
line — should be **linear in log(gap)**, with a coefficient that also carries
the eval noise. That is `c_gap * L(1 + gap/gap_unit)`.

Two things follow that are worth writing down before measuring:

1. It explains why the hand-fit `log(rank)` table worked at all. Under a move
   ordering whose value gaps grow roughly exponentially in rank, `log(rank)` is
   a stand-in for `log(gap)` — a proxy for the right quantity, which is why it
   is good but not optimal.
2. It ties directly to [001](001-minimax-pathology.md). The coefficient on
   `log(gap)` should scale with `sigma_eval`, and 001 found that the entire
   pathology story is governed by the single ratio `sigma_eval /
   sibling_spread`. The same quantity that predicts when minimax is unreliable
   should be setting the price of exploring. If a tuned `c_gap` turns out to
   want to vary with position type in the direction 001 predicts, that is two
   independent lines of evidence meeting, and it is the first real sign that
   there is something here.

## Tuning: making the shallow search agree with the deep one

Twelve parameters cannot be tuned against Elo. A 5-Elo effect needs thousands of
games and a gradient step needs tens of evaluations; that is a throughput
problem we lose by two orders of magnitude, and SPSA on Elo — the standard
answer — is still measured in machine-days per run on six cores.

So `src/tune.rs` implements the cheap objective:

1. Take positions from real games (ours, from the gauntlet PGNs).
2. Label each **once**, expensively: search the top few root moves at a large
   node limit and record what each is worth. That is "depth n+k".
3. Score a candidate price list by searching at a **small** node limit — "depth
   n" — and asking what the reference thinks its choice cost. That is the
   **regret**, in centipawns.

Minimising regret at fixed nodes is exactly "make the cheap search choose what
the expensive search would have chosen", which is what search control is *for*.
Labels are computed once and reused, so an evaluation costs `positions × nodes`
and nothing else.

**How much cheaper than games it actually is, measured rather than assumed, is
in the next section. The answer is "less than it looks".**

### What the objective is worth, measured

Three things had to be established before descending on it, and the third one
changed the plan.

**It is calibrated.** Regret falls monotonically with search effort, on 1500
labelled positions at a 600k-node reference:

| nodes | regret (cp) | agree |
|---|---|---|
| 5 000 | 27.4 ± 1.3 | 41.5% |
| 15 000 | 23.0 ± 1.2 | 45.3% |
| 60 000 | 17.8 ± 1.1 | 50.1% |
| 240 000 | 13.2 ± 0.9 | 57.5% |

So **4× the nodes buys about 4.5 cp of regret**. Four times the nodes at a fast
control is worth roughly 100–140 Elo, which puts the exchange rate at
**1 cp of regret ≈ 25–30 Elo**. That number is what makes the rest of this
section possible: it converts the proxy into units we can reason about.

**The mean is the wrong statistic.** A single price list's mean regret carries a
standard error of ~1.05 cp over 1500 positions, because positions differ from
each other enormously. But two price lists choose the *same move* in most
positions, and those contribute exactly zero to a difference. `tune compare`
therefore scores both on the same positions and reports the paired difference —
the same reason the match runner scores colour-reversed pairs rather than
independent games. On the extreme comparison (the default price list against no
pricing at all) that tightens ±1.05 to **±0.80**, with 420 of 1500 positions
disagreeing.

**And that is still not enough.** The extreme comparison itself measures
**+1.11 ± 0.80 cp** — deleting the entire pricing policy is a bit over one
standard error. Worse, repeating it at 10k/20k/40k/100k nodes gives +1.31,
−0.08, −0.47, +1.14: the sign is not stable. At 1500 positions this objective
cannot resolve the effect of *removing the whole policy*, let alone the effect
of moving one coefficient.

Put in the calibrated units: ±0.80 cp is ±20–25 Elo, which is about what a
600-game match resolves. **The claim that this proxy is orders of magnitude
cheaper than playing games is wrong, and was wrong when it was made.** Per unit
of resolution at 1500 positions it is roughly a wash. What it actually buys is:

* resolution scales as `1/sqrt(positions)` against a **one-time** labelling
  cost (241s for 1500 positions on six cores), whereas games have no such
  amortisation;
* it is deterministic, so a difference of zero is *exactly* zero rather than a
  draw-heavy match's worth of noise;
* **lower node budgets are strictly more efficient.** Effect size did not shrink
  from 60k nodes down to 10k (+1.11 vs +1.31) while cost is linear in nodes, so
  halving the nodes and doubling the positions improves the standard error by
  `sqrt(2)` for the same compute. The objective should be run shallow and wide,
  not deep and narrow.

The working conclusion is that the objective is real and calibrated but needs
**~10⁴–10⁵ positions**, not 10³, and that this was worth two hours of measuring
rather than discovering after a descent run produced a fitted price list that
lost an SPRT.

### What the objective actually says: it is a rail, not a fitness function

With 14000 labelled positions the paired standard error falls to ±0.26 cp,
exactly as `1/sqrt(n)` predicts from the 1500-position runs. Sweeping one
coefficient at a time against the default, at 60k nodes (`b − a`, negative means
the change is better):

| change | Δ regret (cp) | positions differing |
|---|---|---|
| `c_rank` 250 → 0 | +0.410 ± 0.244 | 3660 |
| `c_rank` 250 → 750 | +0.315 ± 0.248 | 3926 |
| `c_rank_depth` 204 → 0 | +0.256 ± 0.264 | 3953 |
| `c_rank_depth` 204 → 400 | +0.363 ± 0.245 | 3968 |
| `c_gap` 300 → 0 | **−0.248 ± 0.204** | 1818 |
| `c_gap` 300 → 900 | +0.094 ± 0.203 | 1908 |
| `c_hist` 100 → 0 | −0.082 ± 0.089 | **234** |
| `pv_discount` 1000 → 0 | +0.273 ± 0.237 | 3735 |
| `max_slope` 1000 → 600 (tighter ceiling) | +0.674 ± 0.268 | 4063 |
| `max_base` −2000 → 0 (ceiling = depth) | −0.274 ± 0.257 | 4253 |
| `max_base` −2000 → 1000 (ceiling = depth+1) | **−0.443 ± 0.264** | 4340 |
| `max_base` −2000 → 2000 (ceiling = depth+2) | −0.443 ± 0.264 | 4340 |
| all coefficients ×1.6, loose ceiling | +1.404 ± 0.291 | 5279 |
| all coefficients ×2.4, loose ceiling | **+4.383 ± 0.330** | 6040 |

Five things fall out, and the last one is the important one.

1. **`c_rank` and `c_rank_depth` are already at a local optimum** — both
   directions are worse. The hand-fit table's shape was right.
2. **The ceiling was the binding constraint and it was too tight.** Loosening it
   from `depth-2` to `depth+1` is the single largest available improvement.
   Going further to `depth+2` is *byte-identical*, which proves the price never
   reaches the ceiling at all — the formula saturates around 65% of the
   remaining budget at rank 40.
3. **`c_hist` is nearly inert**: 234 of 14000 positions change. A history term
   that moves 1.7% of decisions is not a feature, it is rounding.
4. **`c_gap` does not earn its place.** This is the OCBA-motivated term, the
   whole reason the price is a function of centipawns rather than of rank, and
   turning it off is *better* by 0.25 ± 0.20 cp while turning it up is worse.
   The prediction in the section above does not survive contact with real
   positions as formulated. See the caveats — `move_gain` is crude, and this
   falsifies *this* gap feature, not the OCBA argument.
5. **The objective is asymmetric, and that is what it is for.** Over-pricing is
   detected loudly and monotonically — 1.0× → 1.6× → 2.4× gives 0 → +1.40 →
   +4.38, the last at 13 standard errors. Under-pricing is nearly invisible:
   deleting the *entire* policy measures +0.006 ± 0.283 at 15k nodes and +0.445
   ± 0.278 at 60k, for a policy worth 80–150 Elo in real play.

The mechanism behind (5) is straightforward once stated. Over-pruning makes the
search miss the right move, which is exactly what regret measures. Under-pruning
only makes the search shallower, and at a fixed node count a shallower search
still agrees with the reference about the root move most of the time — the root
choice is dominated by the first few plies, the transposition table, move
ordering and quiescence, all of which both arms share. What late move reduction
actually buys is depth in the minority of positions where depth decides the
game, plus everything that compounds across a whole game, and none of that is
visible in single-position root-move agreement.

**So this objective is a safety rail, not a fitness function.** It will reliably
tell you that a candidate prunes too hard. It cannot tell you that a candidate
prunes too little, and it must never be used to argue that a conservative price
list is good. Anything it proposes still has to survive `checkpoint.sh`, and
"the proxy improved" is specifically not evidence in the under-pruning
direction.

### Two implementation points that matter

**The node limit had to be made real.** `go nodes N` was parsed into
`Limits::nodes` and then silently ignored — the search only ever stopped on the
clock or the stop flag. Fixed, and checked every node rather than every 2048:
the objective must be bit-reproducible, and a 2048-node slop window would make
it depend on where the counter happened to land. Every search in the tuner also
runs from a cleared transposition table, so a candidate's score cannot depend on
which position was visited before it.

**Full finite-difference gradients, not SPSA.** SPSA exists because each
evaluation is expensive enough that you can afford exactly two, so it settles
for one noisy random projection of the gradient. When an evaluation is cheap,
`2n+1` evaluations buys the actual gradient — a direction that descends, rather
than one that descends on average. Note the measured caveat above: "cheap" has
to mean cheap *at a position count that resolves the effect*, and at 1500
positions it did not.

**A held-out split, not the whole set.** `tune search` fits on 60% and reports
the other 40% every iteration without ever steering on it. With a dynamic range
of ~1 cp and a standard error of the same order, a descent on the full set will
buy improvements that exist only in that sample; the held-out half is the only
thing that distinguishes finding policy from finding noise.

### The awkward part, stated plainly

Search decisions are discrete, so for a single position the regret is piecewise
constant in the parameters and its true gradient is **zero almost everywhere**.
What rescues it is averaging over a thousand positions: the sum is still a step
function, but with a thousand small steps instead of one large one, and a finite
difference over a coarse enough perturbation sees a real slope. Hence
`h = max(|θ|/8, 25)` — deliberately large. A small `h` reports "no gradient" for
a parameter that matters.

### Three ways this proxy lies

Recorded here because the project rule is that a proxy winning is not a result,
and these are the specific failure modes to check before believing a number.

* **The labels come from the same engine.** A price list that steers the shallow
  search toward the deep search's *mistakes* scores perfectly. The labels are a
  strong opinion, not ground truth.
* **The positions come from our own games**, so the distribution is the one this
  engine already reaches — not the one a stronger engine would reach.
* **Regret at fixed nodes is not Elo.** It ignores time management, the value of
  a stable PV, and everything that happens across moves rather than within one.
  The 25–30 Elo/cp exchange rate above is derived from the node-scaling column
  and assumes the two respond alike; that assumption is itself untested.

Its job is to *propose*. `tools/checkpoint.sh` still decides.

## Where it stands

The refactor was done in two verified steps. Converting the currency to a
fine-grained budget with pricing set to exactly the old LMR table reproduced
`bench` at **630367 nodes — byte-identical**, which is the only real proof that
the restructuring was behaviour-neutral. Replacing the table with the price list
and deleting LMP moved bench to 661128 nodes (+4.9%) and cost
**−12.7 Elo [−32, +7]** over 600 games at 10+0.1 against the previous build —
consistent with zero, on coefficients that were guessed rather than fitted.

That is the intended state: the platform is in place at approximately no cost in
strength, and none of its twelve parameters has been tuned yet.

## Open questions

* **The objective is blind in the under-pruning direction, and that is the thing
  to fix next.** The most promising route: label each position at *two* node
  budgets and keep only those where the reference's best move changed between
  them. Those are the positions where depth decides, which is precisely where
  reductions earn their Elo, and they are currently diluted to invisibility by
  the majority of positions whose best move is obvious at any depth.
* `c_gap` measured at zero. That falsifies **this** gap feature, not the OCBA
  argument — `move_gain` has no phase taper and no exchange sequence, so a move
  that hangs a piece two plies later looks fine to it. Does an SEE-based gap
  behave differently? That is a one-line change to the feature and a rerun.
* `c_hist` moves 1.7% of decisions. Either the scaling is wrong by an order of
  magnitude or history does not belong in the price at all; the sweep cannot
  tell those apart.
* `move_gain` is a first-order estimate with no phase taper and no exchange
  sequence. Is the gap signal good enough, or does the price need SEE?
* The labelling pass clears a 32 MB table per child search, which dominates its
  cost. A smaller table for the cheap ranking pass would speed it up several
  fold — the ranking pass only selects which moves to label, so it does not need
  the isolation the labelling pass does. This matters more now that the position
  count needs to go up by one to two orders of magnitude.
* The ceiling is linear in depth. OCBA says the allocation should also carry
  `sigma_eval`; there is no term for that yet, and 001 says it is the quantity
  that matters most.
