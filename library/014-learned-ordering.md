# 014 — Learned ordering

A position-dependent move-ordering score, built up step by step: PSQT
baseline first, then a small bilinear model distilled from our own deep
search. Games decide at every step; nothing here lands without an SPRT.

## Why

Static ordering tables are saturated: capture history (−3.5), killer
hygiene (−17.4), countermove dead twice, piece-aware history −20.9
(`ORDERING.md` Rounds 2–3). None of them sees the position — history is
`side×from×to` shared across every node. The missing ingredient is
position-dependence, and the hand-built cousin (Stockfish-form
continuation history) is still untried. This is the learned version of
that idea, and a first step toward ROADMAP 5's non-linear policy — with
a parameter count root-regret-free screening could never fit, which is
fine, because ordering is judged in games, not by the tuner
(`ORDERING.md` interaction warning).

## Why not an lc0 policy teacher (original objection, then the revival)

Considered and rejected first, two reasons. The big net's policy head is
attention-based and was deliberately never transcribed (`nnue/lc0/net.py:18`);
rebuilding it unverified (no lc0 binary to check against) is days of
work with no verification path. And the objective is wrong: lc0's prior
optimises MCTS visits under a 100M-param net, while alpha-beta ordering
wants "the move a deeper alpha-beta search would pick". The teacher with
the exactly-right objective already exists — the 400k-node reference
search behind `chess-data/tune/*.labels`, in our own move encoding, so
there is no policy-index mapping to build.

Revival (2026-09-09, Luka's call): the 3.5k root labels cannot supervise
even a 12k-param model (three probes, all below MVV-LVA held-out — see
Phase 4 status in `ORDERING.md`), and dense full-policy labels multiply
every position by ~35 targets. So: a SMALL classic net, not the big one.
`maia-1900` (6x64 SE-resnet, `POLICY_CONVOLUTION`, 1.2 MB) transcribed from
lc0 source (`nnue/lc0/policy.py`): tower + BN folding as in `net.py` but
relu throughout (no activation flag = relu default), policy head = 3x3
conv 64->64 + relu, 3x3 conv 64->80 + bias no activation, gather through
`kConvPolicyMap` translated to NHWC-flat indices. Format-flag meanings
pinned against `proto/net.proto`, forward against the cudnn/blas/TF
backends (all three agree here), mapping against `encoder.cc`'s
`kMoveStrs` (1858 strings, `/tmp/bilinear/lc0moves.txt`).

Verification that replaces the missing reference binary, strongest first:
legal-move mass 0.992 on 4k real positions (a scrambled mapping scores
exactly the chance rate 0.019 — this check caught one such scramble, a
missing square-major→NHWC translation in the gather); sane top moves
(startpos d4/e4/Nf3/c4; after 1.e4 d5/e5/c5); value-head cross-check
(bare kings D 0.997, isolating tower from head); deep-search top-1
agreement 0.304 at MRR 0.487, 3x MVV-LVA zero-shot. The objective mismatch
stands and is handled by role, not by trust: lc0 supervises (abundant),
deep-search labels gate (honest), games decide.

## Phase 1 — PSQT baseline (hours)

Replace the MVV-LVA term in `score_moves` (`src/search.rs:1674`,
`victim*16 − attacker`) with `move_gain` (PST delta + victim,
`src/eval.rs:424`, ~5 lookups, no SEE call). This also deletes O1's
per-capture `see()` from the ordering path, so it tests two things at
once: position-beats-material, and whether SEE's signal in ordering is
worth its cost. Assumption (flagged, cheap to reverse): full
`move_gain` swap rather than MVV-LVA-minus-SEE. Bench must move
(record, don't optimise); verdict is a 200-game screen vs O1 on the
clock under the `ORDERING.md` protocol. If this can't beat MVV-LVA+SEE,
Phase 2's bar is set honestly.

## Phase 2 — bilinear ordering (days)

Score = `φ(pos) · ψ(piece,from,to)`: position embedding `φ` is the literal
2x8 — own-occupancy 384 bits -> 8, opp-occupancy 384 bits -> 8, concatenated
— dotted with a `6×64×64×16` move table (~405k params total, full table, not
factorised). Per move: one 16-gather + 16 MACs — the same price class as the
rest of `score_moves`. Promotions share the pawn `(from,to)` slot (0.13% of
teacher-bests, note and move on); castling rides as king from→to.
Assumptions: maia-1900 full-policy teacher (not deep search — see above),
all-moves scope (teacher-best is a capture 16% of the time, so
captures-only caps at 1 position in 6; history stays until all-moves proves
out), raw T=1 policy renormalised over legal moves (the ~0.8% illegal mass
is dropped — the only filtering).

1. **Labels**: 12.78M FENs (`fishpack-gs200.fens`, game-stride 200, every
   ply, no position filter) x maia policy (`nnue/lc0/teach.py`, sharded
   .npz of per-legal (idx, prob)). Gate FENs (4k + 1k-see labels) excluded
   from training at prep time.
2. **Training**: softmax over legal moves, KL(teacher || model), 1 epoch,
   Adam. `/tmp/bilinear/prep.py` + `train.py`.
3. **Offline gate before any Rust**: top-1 agreement + MRR on the held-out
   4k deep-search labels vs MVV-LVA (0.114/0.231), move_gain (0.118/0.240)
   and maia itself (0.304/0.487). Result: width is binding, everything else
   is noise —
   full16-split 0.138/0.284, mixed16 0.142/0.290, wide64 0.172/0.335,
   wide128 0.193/0.358 (+lr drops → wide128c 0.211/0.382, fidelity 0.463),
   wide256 worse (0.179 — overfits/optimises worse at fixed lr).
   16-dim distilling wide128 stalls at 0.149 (capacity wall, not target
   sharpness — T=2 softening also stalls at 0.139). Absolute-frame phi
   (deployment-motivated, incremental-trivial) plateaus at 0.165 vs 0.211
   rel-frame — the mirror correspondence is worth 0.05, keep rel-frame
   with dual accumulators instead. Capture-conditional: 0.739 vs 0.630
   MVV-LVA among captures. int8 (per-tensor W, per-row psi) is gate-lossless
   (0.211/0.382 exact) — the screen runs int8 at fair price.
   Bar cleared 1.8x over MVV-LVA at 64% of teacher; remaining gap is future
   work (nonlinear phi, stronger teacher), not a ship-blocker — games decide.
4. **Rust**: dual-frame int8 phi (`white-frame` + `black-frame`, `2×128`
   i16 accumulators per ply, eager 4–8 toggles per descend, null move =
   frame swap by copy) + int8 table (3.1 MB) with f32-compute dots.
   TT-first and killers stay, O1 SEE-split stays; learned score replaces
   the MVV-LVA term captures-only first (history untouched — smallest diff;
   all-moves scope needs the dot price measured first). Bench moves →
   200-game screen vs O1 → SPRT star.
5. **Quantize after proof**: f32 table is 1.5MB; i8/i16 once games say
   yes (same playbook as the accumulator, LEDGER 053/070).

## Phase 3 — serious run (only if Phase 2 plays)

It did not. Screens, both run 2026-09-09 at 8+0.08 ×200 vs O1
(`ORDERING.md` log O11/O12, one binary with `--order-table` off/on):

- O11 captures-only (learned dot replaces MVV-LVA, split/killers/history
  stay): bench 189839 (−0.4% tree, −10% nps), games **−24.4 [−48,−0]**.
  Rust scores verified == Python (≤0.12), so the loss is the idea, not the
  port. Residual MVV-LVA+α·learned interpolates monotonically offline
  (0.630→0.706, no sweet spot) — no screen.
- O12 all-moves (learned for captures and quiets, history off, split stays):
  bench 210797 (+10.6% tree, −16% nps), games **−20.9 [−49,+8]**.

Fourth and fifth bench≠Elo instances. Verdict: move-plausibility agreement
(0.739 capture-conditional, 0.211 overall vs 0.114) does not convert to
alpha-beta strength — MVV-LVA's biggest-victim-first finds refutations
faster than the teacher's favourite move, and the price is structural
(128-dim needed for quality; 16-dim gates 0.14; absolute-frame phi costs
0.05). Ledger entry on ship or surprise, same as any ordering result —
this one is filed as a surprise with a negative sign.

## Standing risks, and the reopen condition

Ordering feeds `c_rank` pricing, so a "better" order can search more
or less — proxy metrics (KL, agreement, bench size) never promote;
only the clock does. That risk is now a measured loss twice over.

This direction reopens on exactly one thing: a **refutation-aware
teacher** — labels of which child refutes, one per priced child, i.e. the
per-child corpus (STATE Next 8). Plausibility teachers (lc0 policy,
deep-search top-1) are exhausted as ordering supervision: two scopes,
two screens, both negative. The pipeline is kept (`nnue/lc0/policy.py`,
`teach.py`, `/tmp/bilinear`: probe/train/prep/export); the Rust
(`src/orderlearn.rs` + hooks) is reverted, per the no-measured-win rule.
