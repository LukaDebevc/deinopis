# What this engine is

One file describing what the engine does, with the actual formulas. If the
code disagrees with this file, the code is right and this file needs fixing.

Fingerprints (which engine you are running — `chess bench` prints one):

| bench nodes | eval |
|---|---|
| 199091 | WDL net `m1-b1` (`--wdl m1-b1.nnue`) — the `cp-0004` engine |
| 233103 | PeSTO piece-square tables — means **no net file was found**, not a second engine |

## The outer loop: iterative deepening with aspiration

The engine searches depth 1, then 2, then 3…, reusing the transposition table
and move ordering from each pass for the next. Depths 1–4 use a full window;
deeper iterations search a narrow window around the previous score:

```
alpha, beta = prev - 25, prev + 25        (aspiration_window = 25 cp)
loop:
    score = search(budget = depth * 128, alpha, beta)
    if score <= alpha:  beta = (alpha + beta) / 2; alpha = score - delta
    elif score >= beta: beta = score + delta
    else: return score
    delta += delta / 2
```

A new iteration starts unless half the clock is gone or mate is proven.
The point of the shallow passes: they are faster than the depth they buy,
because ordering compounds.

## The core idea: search spends a budget, not plies

There is no depth counter that says "reduce by 1 here, prune at depth 3
there". Each node holds a **budget** in units of 1/128 ply (`PLY = 128`).
Every child pays a **fare** plus a **price** that grows the worse the move
looks, and inherits what is left:

```
spend  = fare + clamp(price, floor, max_base + max_slope * depth)
child  = parent - spend        (and never more than parent - 1)
child <= 0      -> quiescence (captures only, see below)
child < skip_below -> move not searched at all (pruned)
```

The first move at every node pays only the fare. The price of the rest:

```
price = c0 + c_rank * log2(rank) + c_gap * log2(1 + gap / gap_unit)
           + c_hist * H - c_rank_depth * log2(rank) * log2(depth)
           - pv_discount (at PV nodes only)
           + feature terms (table below)
```

All coefficients are integers in **milli-plies** (1000 = one ply). Current
defaults: fare 128, c0 0, c_rank 250, c_gap **0** (measured useless, not
missing), gap_unit 32, c_hist 100, c_rank_depth 204, pv_discount 1000,
floor 0, max_base 1000, max_slope 1000, skip_below −128.

Two things this replaces: the old 64×64 reduction table (that is
`c_rank` + `c_rank_depth`, now tunable) and late-move pruning (that is
`skip_below`, now a price instead of a move index). The budget strictly
decreases — a price below −fare is clamped — because a tuned price list
that can hand a child more than the parent held is a search that never
returns. Alpha-beta cutoffs and hash cutoffs are untouched: the budget
decides how much a *relevant* move gets, never whether a *proven
irrelevant* one is visited.

Cheap search, then verify (principal variation search): moves after the
first are searched with a zero-width window at their priced budget. If one
beats alpha anyway, the price was wrong, so it is searched again at the
full budget — first narrow, then full window if needed. About 30% of nodes
are these re-searches; they are the precision mechanism, not overhead.

## Extra information: the feature table

One line per thing the search may charge for. Group = what it claims to
measure (Sigma: this child is hard to judge shallowly; Gap: this move is
behind; Cost: an error here matters at the root). Gated features cost
nothing while their coefficient is zero — the expensive check is skipped,
not just multiplied by zero.

| feature | claims | cost to compute | default | state |
|---|---|---|---|---|
| c_see (move loses material by static exchange) | Gap | SEE call, gated | **4000 — LIVE** | +51 Elo over 484 games |
| c_recap (recapture on the last capture square) | Sigma | free | 0 | fires too rarely to measure |
| c_offpv (off-PV decisions above this node) | Cost | free | 0 | loses to no-information price, closed |
| c_ply (plies from root) | Cost | free | 0 | measures ≤ 0 |
| c_depth (log of budget remaining) | Cost | free | 0 | measures ≤ 0 |
| c_incheck (side to move in check) | Cost | free | 0 | measures ≤ 0 |
| c_gives_check (the move gives check) | Cost | make+check, gated | 0 | measures ≤ 0 |

`chess features` prints this surface. New information goes here, never as
an inline `if` in the move loop.

## Whole-node pruning (three rules, all tunable)

Skipped in check, at PV nodes, and inside quiescence. `depth` below is
budget in whole plies, rounded up.

- **Reverse futility.** Ahead by more than the rest of the search can
  plausibly lose: `static_eval - 75 * depth >= beta` at depth ≤ 7 →
  return the static eval.
- **Null move.** Passing is a lower bound on any real move (except in
  zugzwang, so: only with non-pawn material, only when already above
  beta, at depth ≥ 3). Reduction `R = 3 + depth / 4` plies; if the
  pass-and-give-up-R-plies search still beats beta, return it (beta
  itself for unproven mates).
- **Check extension.** The one place a node gets money *back*: side to
  move in check → budget += 1 ply. Forced lines are cheap and
  catastrophic to judge statically.

Mate-distance pruning: alpha = max(alpha, mated-in-ply),
beta = min(beta, mate-in-ply+1); return alpha if crossed.

## Quiescence: captures only, standing pat allowed

When the budget runs out, the search does not return the static eval —
the side to move might be hanging a queen. It searches captures only
(every evasion when in check, or mates would go unseen):

```
stand pat:  s = eval; if s >= beta return s; alpha = max(alpha, s)
for each capture (best first):
    delta pruning: static + victim_value + 100 < alpha, not a promo -> skip
    SEE < 0 -> skip (loses material on the spot)
    full-window search, one q-ply deeper; keep the best
```

Quiescence shares the main hash with one restriction: a quiescence score
bounds a different function (captures only, stand pat allowed), so score
cutoffs cross only between quiescence nodes — marked with depth 255
(`tt::Q_DEPTH`), which main-search probes refuse at any depth. The stored
move and static eval are position properties and cross freely, which is
also what orders quiescence by TT move now. Quiescence stores twice: the
stand-pat fail-high (a Lower bound, no move), and the loop result with the
usual bound. A deep main-search entry survives a quiescence store; a
main-search store always displaces a quiescence one (`tt.rs` compares
depths semantically). Bench 243605 → 199091 nodes (−18%), nps +29%;
Elo pending SPRT. Capped at 64 q-plies, which never binds.
## Move ordering

TT move first, then captures/promotions by MVV-LVA, then two killers per
ply, then history:

```
capture score = 2^22 + 16 * victim_value - attacker_value + promo_value
killer = 2^21 (+1 for the first), history = table[side][from][to]
```

History update on a quiet cutoff, with saturation instead of overflow
(old news decays rather than dominating):

```
bonus = min(depth^2, 1200)
good move: h += bonus - h * bonus / 16384      (saturates toward +16384)
tried quiets that failed: same with -bonus
```

## Transposition table

Lockless, keyed by zobrist hash. Each store: best move, score, the static
eval (so later nodes can skip evaluating), depth proved (**rounded down**
— never advertise work not done), bound (exact / lower / upper). A probe
returns only outside quiescence, only off-PV, only if stored depth covers
the depth claimed and the bound fits the window.

## Eval: a WDL distribution, read as centipawns

The deployed net (`m1-b1`, width 512, bottleneck 16, hidden 32) outputs win/draw/loss
probabilities through: one 768→512 table, then per-king-square down,
symmetric mid, per-material up, per-material second layer, head — plus a
skip connection and a piece-square passthrough. Reported to the search as:

```
cp = 288.5 * logit(W + D / 2)
```

Contempt is a readout, not a retrain: `--draw-value v` scores a draw as
`W + v * D` from the root side's view (default 50 = normal chess; 0 =
a draw counts as a loss). Symmetric contempt costs ~190 Elo in self-play
(fixed 400 @8+0.08, +94 =14 −292: it fights draws it should take).
Asymmetric form: `--contempt-b` adds `b * weight * tanh(a * gap)` in cp,
stm-relative so it needs no root tracking — behind seeks (up to +b), ahead
fights (down to −b), parity and (unless flat) decisive positions get
nothing. `weight` is the normalised draw mass, or identically 1 with
`--contempt-flat`, so the bonus keys on the win-vs-loss difference alone
(`D` is normalised, not the raw draw logit — raw logits are shift-variant
and meaningless). A state-dependent `v`-blend was tried first and deleted:
numerically probed globally incoherent (a 2% drawish position reads +418cp
while its 18% neighbour reads −418 — min-loss and max-win do not compare
across the boundary). tanh(100, 1.0) vs plain fixed-400 running 2026-09-10.

Fallback chain when the net file is missing: deep net → quadratic net
(`quad.nnue`) → PeSTO tables. **A binary that silently falls back plays a
~100–350 Elo weaker eval with the same name.** Every match script must
name the eval explicitly (`--wdl <file>` or `$CHESS_WDL`); `chess bench`
fingerprints which one ran (table at the top).

Cost: ~910 ns per eval, 64% of search time; the accumulator update on
every made move is another ~20%. Everything the search itself does —
movegen, make/unmake, hash, ordering, pricing — is ~5% together.

## Clock: simple time management

```
allotment = time_left / moves_to_go + 3/4 * increment, capped at time_left / 3
movetime mode: movetime - 20 ms.   moves_to_go defaults to 30.
```

## Parallel search: lazy SMP

N threads each search the whole tree from the root over one shared hash
table; the speedup (2.3× effective on 6 threads) comes from threads
reading each other's cutoffs, not from splitting work. Thread 0's move is
played. Non-deterministic by design — benchmarks and tuning stay
single-threaded. Built, never measured in games.

## The work meter: wall-clock time without a clock

`src/work.rs`. Count every bracketed operation (counts are bit-identical
run to run), multiply by a frozen per-operation price from `nodeprof`,
add a flat per-node remainder:

```
work_ns = sum over zones (calls * PRICE_NS) + nodes * CONTROL_NS
1 work unit = 1232.3 ns (one main-search node at the deploy config, frozen)
```

Same binary + position = same number to the digit, under any machine
load (verified: +7% wall clock under spinner load, work unchanged at
240522 units). Use differences, never absolutes (±1.5% systematic bias).
Blind to kernel/speed work (halving bytes moved reads as zero) — those
need a real clock. Tuner budgets: `--work N` instead of `--nodes N`.

## The tuner: regret against a deep search, not games

`chess tune`: label 1000 positions with a 400k-node reference (top 5 root
moves), then score a price list by **regret** — how much worse its move
is than the reference's, in cp, capped at 200 (mates would be the whole
objective otherwise):

```
regret = reference_cp(best reference move) - reference_cp(cheap move)
```

1000 positions resolve 4.1 cp. Rough exchange: 1 cp ≈ 25–30 Elo (biased
high ~2× — right direction twice, wrong size once). Score at ~15k
nodes/position; 1k nodes has the *wrong sign* on over-pruning. The root
channel saturates near ~1.7 cp at any reference depth — single-knob
screening there is exhausted; the headroom is per-child labels.

## What is deliberately NOT in here

Singular extensions. Continuation, countermove, or correction history
(plain side×from×to only). Futility pruning at normal nodes (reverse
only, plus delta in quiescence). A reduction table (it is the price
list now). Tablebases, an opening book of its own, pondering, Chess960.
Quiescence hash reads. A tuned price list — `c0` alone is worth −2.2 cp
untouched, so the list is still largely unfitted. Lazy-SMP and contempt
Elo numbers. Multi-opponent Elo combination in `chess elo` (hand-combined;
see the ladder script).
