# 015 · Correction history (scalar tables)

The eval is systematically wrong in ways that persist across a search: same
pawn structure, same misevaluation, every node. History remembers which
* moves* worked; correction history remembers where the *eval* was wrong,
indexed by pawn structure rather than by move. Stockfish credits the full
set at ~20–40 Elo; this note specs the scalar subset against the current
tree, cheapest table first, each table separately measurable.

## What it is

One array per thread: `corr[side][pawn_key & 16383]`, entries `i16`.
At a main-search node, after `static_eval` is known:

```
corr_eval = static_eval + ((corr[stm][idx] as i32 * 100) >> 8).clamp(-256, 256)
```

`corr_eval` replaces `static_eval` in pruning decisions. At the end of the
node, the residual corrects the table:

```
err   = best_score - static_eval          // what the search thought, minus what the eval said
bonus = err * weight(depth)               // deeper nodes earn more say
*e   += bonus - *e * bonus.abs() / 16384  // gravity, the same form as the history update
```

The update is the history update's shape (`search.rs:1856`) with a signed
bonus: entries saturate instead of overflowing, old information decays.

## Where it hooks in (current tree)

All line numbers against `00d3f2c` + the threads-a/b working tree.

**Needs a pawn key.** `Board` carries none (`board.rs:46-59`). Maintenance
is two gated XORs: in `add_piece`/`remove_piece` (`board.rs:201-220`),
`if pawn { self.pawn_key ^= zobrist::pawn_key_component(color, sq) }`.
The component helper already exists and names this use
(`zobrist.rs:77-80`). FEN parse rebuilds it in the from-scratch path
(`board.rs:~481`); `make_null` is untouched (no pieces move). Perft is
untouched (it counts nodes, never reads keys); the incremental-vs-scratch
zobrist test gains a pawn-key leg. Cost when the feature is off: one
predicted branch per piece toggle, no node-count effect — bench-exact.

**Read path** — one lookup per main-search node, right after `static_eval`
is bound (`search.rs:1290-1297`). Phase 1 uses `corr_eval` in exactly two
places, both pure pruning decisions whose output is never returned as a
score:

- RFP: `static_eval - margin*depth >= beta` (`search.rs:1305-1310`).
- NMP gate: `static_eval >= beta` (`search.rs:1315-1318`).

Phase 2 (only if phase 1 passes) extends to value-returning uses: qsearch
stand-pat and its beta cutoff (`search.rs:1280-1289`) and q futility
(`search.rs:1429`). Phase 2 is where most of the Elo should live — the
stand-pat is the score a quarter of q-nodes return — but it also changes
returned values, so it is gated behind phase 1 proving the mechanism.

**Write path** — end of `search`, next to the TT store (`search.rs:1706-
1731`). Gate: `!in_check && !in_q`. No quiet-best-move gate in phase 1:
the residual is meaningful whenever the node's value is searched, and
gating on move properties is the first variant if the SPRT is flat
(exact-bound-only is variant two). Weight: `weight(depth) = min(depth, 16)`?
Start linear in depth capped small — deeper is more trustworthy, and a
cap keeps one deep node from pinning an entry. Tune nothing by hand that
an SPRT can rank; this number is a guess with a measurement attached.

**TT discipline: store raw, correct at use.** The store at `search.rs:1716`
keeps writing raw `static_eval` (it already does). Old entries stay valid
across the flag flip and across main/q sharing (the Q-store line depends
on stored evals flowing freely); the correction is applied wherever the
stored/probed eval is *read*. This is also what makes A/B share one PGN
pool honestly — same TT contents, different readouts.

**SMP: per-thread tables, no sharing.** They live in `ThreadData` next to
`history` (`search.rs:753-757`), cleared in `clear()` (`search.rs:830-838`).
Same precedent as killers/history: 64 KB per thread, no atomics, no false
sharing to reason about.

**Switch: POT-style static, default off.** `static CORR: OnceLock<mode>` +
`CORR_ANY: AtomicBool`, parsed from `--corr` / `$CHESS_CORR` in
`crate::init`, following `init_pot` (`search.rs:715-735`). Off means the
read path early-outs and the write path never fires: bit-identical bench,
zero-cost dormant carriage (the contempt precedent, STATE 2026-09-10).
Same binary plays both SPRT arms (`--corr pawn` vs absent). Enabling by
default after H1 moves the bench fingerprint — record the new number,
don't argue with it.

## Why this order

- Pawn first because the key is one field and the structure truly persists:
  pawns move rarely, so the same entry serves many nodes, which is what
  amortises the learning. Non-pawn material, minor/major splits need the
  same treatment with a second maintained key — same two XORs, different
  predicate — and continuation correction (by previous moves) needs only
  `PathInfo.mv`, which already exists (`search.rs:670`). All deferred
  until the pawn table proves entries converge to something useful.
- Not a `features!` price term: this adjusts the eval, not the budget. No
  group, no price — wrong platform, applied at the wrong point.
- The proxy cannot screen it (065: single-coordinate screening is
  exhausted at this operating point, and regret never sees pruning
  decisions cleanly). Straight to SPRT.

## Measurement

1. Land dormant: tests pass, bench `199091` exact with the flag off
   (m1-b1 net), pawn-key scratch test added.
2. SPRT `[0,5]` @8+0.08 conc 5, same frozen binary both arms,
   A `--corr pawn` vs B base, m1-b1 net both sides. The claim being tested
   is the mechanism (residuals converge, RFP/NMP fire better), not a size.
3. H1 → phase 2 SPRT (stand-pat), then consider the non-pawn key.
   Flat → variant ladder (exact-only updates; quiet-gated updates), each
   one SPRT, then kill. A killed scalar table stays dormant in the tree;
   it costs nothing and the pawn key serves the next attempt.

## Open points (guesses, labelled)

- `weight(depth)` shape and the `100>>8` readout scale are copied instincts,
  not measurements. If H1, fit them; if flat, vary the gate before the scale.
- Update-on-fail-high/low uses bounds as targets and biases entries toward
  the window centre. Kept in phase 1 for volume (16k entries need samples);
  exact-only is the named fallback.
- Qsearch nodes never update (their stand-pat *is* the eval — updating from
  it is the self-referential-teacher trap from STATE item 9). Main search
  only, deliberately.
