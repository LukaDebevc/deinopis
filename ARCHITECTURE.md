# Architecture

Why the code is shaped the way it is. Changes rarely; read `STATE.md` for what
is actually happening right now.

## Layering

```
types, bitboard, attacks      geometry. no game state, no allocation.
zobrist, board, chess_move    position representation, copy-make
movegen                       legal move generation
eval                          static evaluation  (trait: Evaluator)
tt, search                    transposition table, alpha-beta, budget pricing
game, san                     rules of game-end; notation for humans
uci, gui                      front-ends
perft                         movegen correctness harness
book, sprt, matchplay         the measurement harness
tune                          fitting the price list without playing games
```

Each layer depends only on the ones above it.

## Decisions that are load-bearing

**The search spends a budget, it does not count plies.** Depth is a real number
held as an integer in units of `PLY = 128`. Every child of a node is charged a
fare plus a *price* that depends on how unpromising it looks, and inherits what
is left; a child left with nothing drops into quiescence, one left below
`skip_below` is not searched at all. Late move reduction and the "don't reduce
captures" rule are the same object in this scheme and were deleted in favour of
it.

Late move pruning was deleted too, and — measured rather than assumed — is
**not** reproduced by the price list. With the current coefficients the price
saturates well below a child's remaining budget, so `skip_below` fires in
**0 of 14000 test positions**. The mechanism exists; the parameterisation that
would use it does not. That is a real hole and part of what LEDGER 006 cost.

Two constraints hold the design up. **The bound is not for sale**: beta cutoffs
and TT bound-returns are untouched, because alpha-beta's power is the proof that
a subtree cannot matter and no allocation policy buys that back — that is the
specific reason the 1990s best-first metareasoning engines lost, and why the
budget is only the *allocator* while depth-first alpha-beta stays the
*executor*. And **the budget strictly decreases**, enforced at the use site
rather than trusted to the parameters, since a price list is tuned by machine
and a negative net cost is a search that never returns.

Why bother, given the table it replaced was not bad: a 64×64 integer table has
no gradient and cannot be fitted. A continuous function of named features can,
which is what makes roadmap #5 a fitting problem rather than a rewrite. Full
rationale, including why the price is linear in `log(gap)` rather than in the
gap, is in `library/003-budget-search.md`.

**Copy-make, not make/unmake.** `Board` is `Copy`, ~180 bytes, and
`make_move` returns a new board. Copying costs under 2% of a node (~2 Elo). In
exchange the search recursion has no state to unwind, which removes the entire
class of "forgot to restore a field" bugs and makes the search restructurable —
which for an engine we intend to keep rewriting is worth far more than 2%.

Consequence this was written for: an NNUE accumulator (4 KB) is far too big to
copy per node, so it would need its own ply-indexed stack with incremental
update — which is why `Evaluator` has `push`/`pop` and takes `&mut self`.

**The eval we actually trained does not need any of that.** `eval = x^T W x` on
the 768 piece-square occupancy bits is a sum over *pairs of pieces on the
board*, so lifting a piece off square `p` costs

    delta = -2 * sum_{i in S} W[p][i] + W[p][p]

where `S` is the ~32 occupied features. That is **~32 gathers per toggle**,
about **4 x 32 per move** at the worst (castling, promotion-capture), and there
is **no accumulator and no per-ply state at all**. `a = Wx` is an NNUE idiom: it
exists because a nonlinearity sits downstream and has to see every hidden unit
before it can be applied. A quadratic form has no such layer, so the
accumulator is pure overhead — computing it would cost 768 adds per toggle to
save work that was never needed.

Two things follow. Cost **scales with the piece count** — ~64 ops per quiet move
in the opening, ~12 in a king-and-pawn ending — where an NNUE costs the same
everywhere. And `push`/`pop` stay on the trait for a future evaluator that does
need them, but the quadratic evaluator implements them as no-ops. See LEDGER 019.

**On-demand attacks, not incremental attack tables.** Maintaining a
"who attacks what" table sounds attractive — instant recaptures, cheap SEE — and
Rebel did exactly that in the 90s. It lost, for a specific reason: **modern
search is lazy and incremental tables are eager.** With staged generation and a
~90% first-move cutoff rate, most nodes never generate a full move list, so an
incremental table pays full update cost for queries that are never made.

`Board::attackers_to(sq, occ)` gets the same information in ~5-10ns, only where
the search asks. SEE, check detection, recapture detection and defended-square
ordering are all one call to it. The explicit `occ` parameter is what lets SEE
peel attackers off one at a time and see the x-rays behind them.

**Legal movegen, not pseudo-legal + filter.** Three masks computed once per
node — `check_mask`, `pinned`, and king-danger-with-our-king-removed. The search
never makes a move to discover it was illegal, and `MoveList::len() == 0` is
directly checkmate-or-stalemate.

**Magic bitboards with magics searched at startup.** Costs a few ms and keeps
128 unexplainable constants out of the source. Zen 3 has fast PEXT, so a PEXT
variant is a five-line swap later — both index the same per-square sub-tables —
but it is worth ~1%, so one code path wins for now.

## Seams built for what comes later

These exist now, unused, because retrofitting them is expensive:

| seam | for |
|---|---|
| `Shared` vs `ThreadData` | Lazy SMP. Shared = TT + stop + node count; ThreadData = killers, history, PV. Spawning N threads over one `Shared` needs no restructuring. |
| `tt::TranspositionTable` atomics | already lockless (Hyatt XOR checksum, `Relaxed` ordering = a plain `mov` on x86). This is where C++ engines commit deliberate UB and Rust does not have to. |
| `eval::Evaluator` | the trained eval. Implement the trait, keep the search. The quadratic form needs no accumulator, so this is a smaller change than the seam was built for. |
| `search::Params` | **every** pruning and reduction constant lives in one struct. This is what makes learned search control a swap instead of a rewrite, and lets a hand-tuned control arm be reproduced exactly. Declared through the `params!` macro, which generates the field, its default, its search box and the `NAMES`/`get`/`set`/`doc` registry from one line — so a constant cannot be added without becoming tunable. **30 names**, one flat namespace across two levels of nesting; `chess params` lists them, `--features tune` exposes them over UCI. |
| `search::Pricing` | the price list that allocates the search budget. It replaced the hand-fit `lmr_table()`; a learned controller replaces it in turn. Nested inside `Params`, which sees through it. |
| `features!` + `search::Ctx` | **the feature platform** — how new information reaches the search. One declaration gives a coefficient, its box, its docs, its registry entry, its extraction and its price term; `Ctx` is the single borrow of node state every feature reads. Each declares a **group** (`Sigma` / `Gap` / `Cost`, from the allocation law in `library/004`, which fixes its sign before measurement) and a **cost class** — `Gated` features do not execute while their coefficient is zero. Carrying a dormant feature is **unmeasurable** at ±2-6% box noise, so declare freely. `chess features` lists the surface; `library/013` is the manual. |
| `tune.rs` | the cheap objective a controller is fitted against, and the place a learned one would be trained. `library/012` is the operating manual. |
| `work.rs` + `Limits::work` | a search budget denominated in **work** rather than nodes or seconds, so a controller that moves effort between main search and quiescence is charged for it (a q-node costs 0.275 of a main node). Deterministic and load-immune, which a wall clock is not. Behind `--features work`; a playing build has no counter and refuses a work budget rather than ignoring it. |

## Things that are deliberately crude

- Eval is PeSTO piece-square tables. It is the **control arm**, not an attempt
  at a good eval — everything later gets measured against a search that is
  otherwise identical.
- Time management is `time/movestogo + 3/4 inc`, capped at a third of the clock.
- Castling rights are a 4-bit mask, not rook files, so Chess960 is not supported.
- A single repetition inside the tree counts as a draw (three are required in
  the rules). Every engine does this: it prunes an enormous amount of tree, and
  a line that repeats once can be repeated again by either side.

**Two different draw rules, on purpose.** `search::is_draw` treats a *single*
repetition inside the tree as a draw. That is not the rule of chess — three are
required — and it is what every engine does, because a line that repeats once
can be repeated again by either side and detecting it a repetition early prunes
an enormous amount of tree. `game::status` is the actual rule, and is what
adjudicates real games in the match runner and on the web board. Confusing the
two would either lose the pruning or claim draws that never happened, so they
are separate functions in separate modules with the reason written on both.

**The measurement harness is in the engine, not beside it.** `chess match`
plays engine-vs-engine games, in-process or over a pipe, and reports Elo with
pentanomial error bars and an SPRT verdict. The alternative — cutechess-cli —
is a better tool with more features, and was still the wrong choice here: the
measurement is the thing this project's claims rest on, and keeping it in-repo
means it is versioned with the engine, tested by `cargo test`, and reproducible
from a git tag alone. It is ~600 lines of `std`.

Consequence worth knowing: the runner can play the engine under test
in-process or as a subprocess, and those are different code paths. Measured at
400 games, the difference is +4.3 Elo [-18, +27] — consistent with zero but
only to ±22. So the checkpoint gate runs *both* sides as subprocesses, where it
cancels by construction. See `library/002-measuring-strength.md` §7.

**The web board streams.** `/api/think` emits one JSON object per completed
iteration on an open connection rather than returning a move at the end. This
is the difference between a toy and an instrument: a score that oscillates
between iterations, a PV whose first move keeps changing, a node count that
explodes at one depth, are all visible live and none of them are visible from
the final move. The server stays stateless — every request carries the whole
game — so stepping back through a game needs no server-side undo stack, and the
engine always sees the real repetition history.

## Testing

`cargo test --release` runs in under a second and covers:
perft to depth 4 on all six standard positions, FEN round-trip, incremental
zobrist vs from-scratch after every move, mate distances, quiescence refuting a
poisoned capture, a full legal self-play game, and search determinism.

`chess perft-suite` runs the full published depths (minutes).
`chess bench` prints a fixed-budget node count (9 ID iterations, top budget
9·PLY — "depth 9" is legacy naming, there are no fixed plies) — a
deterministic fingerprint of the search. **Run it after any change that was
supposed to be behaviour-neutral; if the node count moved, the change was not
neutral.** It is not a quality metric: node count is not time (tree
composition moves nps, so report nodes/nps or work), and fixed-node scoring
discounts cheap-node tricks (see `work.rs`). It is also the line
OpenBench parses.

`chess tune eval` is the fast, deterministic proxy the price list is fitted
against — mean centipawn regret against a reference search, over a labelled
position set. It is a proxy and it is allowed to propose, never to decide; the
three specific ways it can mislead are written down in `library/003`.

`tools/checkpoint.sh` is the gate that decides a version is worth keeping:
tests, full perft suite, bench, then an SPRT against the previous checkpoint's
binary. It commits, tags and pushes only on H1; a failure writes nothing. The
kept versions are listed in `CHECKPOINTS.md`, each with a git tag, so any of
them can be rebuilt and re-tested — which is also how the reference binary is
restored when `.checkpoints/` is missing.
