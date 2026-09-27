//! Alpha-beta search.
//!
//! Structure notes, because these are the seams that matter later:
//!
//! * `Shared` holds everything threads must agree on (TT, stop flag, node
//!   counter). `ThreadData` holds everything they must not share (killers,
//!   history, PV). Lazy SMP is then "spawn N threads over the same `Shared`" —
//!   no restructuring required.
//! * Every pruning and reduction decision reads its constants from `Params`
//!   rather than hardcoding them. That is what makes the learned-search-control
//!   experiment a swap rather than a rewrite, and it lets a hand-tuned control
//!   arm be reproduced exactly.
//! * The backup is plain negamax `max`. The soft-backup experiment from
//!   notes/001 would slot in at PV nodes only, since alpha-beta pruning is
//!   incompatible with a backup that needs every child's value.

use crate::board::Board;
use crate::chess_move::{Move, MoveList};
use crate::eval::{self, Evaluator, Score, DRAW, INFINITY, MATE_BOUND};
use crate::movegen::{generate, GenType};
use crate::types::PieceType;
use crate::tt::{Bound, TranspositionTable};
#[cfg(feature = "prof")]
use crate::nodeprof::Z;
use crate::nodeprof::zone;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const MAX_PLY: usize = 128;

/// One ply, in budget units.
///
/// The search does not count depth in whole plies. It spends a **budget**, and
/// one ply of nominal search costs `PLY`. The fine grain exists so the price of
/// a move can be a continuous function of how unpromising it looks, rather than
/// an integer pulled from a table — see `Pricing`. `PLY` is a power of two so
/// every conversion is a shift.
pub const PLY: i32 = 128;

/// Every magic number the search uses. Grouped here so an experiment can vary
/// one and keep the rest fixed, and so a learned controller can replace the
/// whole struct.
///
/// `pricing` is how the search allocates its budget — the part we are actually
/// researching, kept as its own struct because a learned controller replaces
/// it and leaves the rest of `Params` alone. `get`, `set` and `all_names` see
/// straight through it, so the tunable surface is one flat namespace of
/// twenty names regardless of how it is grouped in memory.
/// Declare a parameter struct and its registry in one place.
///
/// The point is that they cannot drift apart. Before this, `Pricing` carried a
/// hand-written `NAMES`/`get`/`set` triple and `Params`' own eight fields
/// carried none at all — so half the search's constants were invisible to the
/// tuner and to UCI, and adding a parameter meant remembering to edit four
/// places. Here the field, its doc, its default and its search box are one
/// declaration and everything else is generated, so a parameter cannot be
/// added without becoming tunable.
///
/// The box is not decoration. `set` clamps to it, which is where `fare`'s and
/// `gap_unit`'s old hand-written `.max(1)` guards went, and it is what a
/// stochastic optimiser needs handed to it: SPSA wants a range, not a name.
///
/// A struct may carry one nested parameter struct, declared as `> field:
/// Type;`. `get`/`set`/`NAMES` see through it, so the whole search-control
/// surface is one flat namespace no matter how it is grouped in memory.
macro_rules! params {
    (
        $(#[$sm:meta])*
        $sv:vis struct $S:ident {
            $(> $nf:ident : $NT:ty;)?
            $(
                $(#[doc = $d:literal])*
                $fv:vis $f:ident = $def:expr, $lo:expr, $hi:expr;
            )*
        }
    ) => {
        $(#[$sm])*
        $sv struct $S {
            $(pub $nf: $NT,)?
            $(
                $(#[doc = $d])*
                $fv $f: i32,
            )*
        }

        impl Default for $S {
            fn default() -> $S {
                $S {
                    $($nf: <$NT>::default(),)?
                    $($f: $def,)*
                }
            }
        }

        impl $S {
            /// This struct's own parameters, in declaration order. Use
            /// `all_names()` for the surface including any nested struct.
            pub const NAMES: &'static [&'static str] = &[$(stringify!($f)),*];
            /// `(low, high)` per name, same order. The tuner's search box.
            pub const BOUNDS: &'static [(i32, i32)] = &[$(($lo, $hi)),*];
            /// One line of prose per name, same order. `chess params` reads
            /// these, so a parameter documents itself exactly once.
            pub const DOCS: &'static [&'static str] = &[$(concat!($($d),*)),*];

            /// Every tunable name reachable from here, nested struct included.
            pub fn all_names() -> Vec<&'static str> {
                let mut v: Vec<&'static str> = Self::NAMES.to_vec();
                $(v.extend(<$NT>::all_names()); let _ = stringify!($nf);)?
                v
            }

            pub fn get(&self, name: &str) -> Option<i32> {
                match name {
                    $(stringify!($f) => Some(self.$f),)*
                    $(_ => self.$nf.get(name),)?
                    #[allow(unreachable_patterns)]
                    _ => None,
                }
            }

            /// Clamped to the declared box. Returns false for an unknown name,
            /// so a typo in a tuner config fails loudly instead of quietly
            /// tuning nothing.
            pub fn set(&mut self, name: &str, v: i32) -> bool {
                match name {
                    $(stringify!($f) => { self.$f = v.clamp($lo, $hi); true })*
                    $(_ => self.$nf.set(name, v),)?
                    #[allow(unreachable_patterns)]
                    _ => false,
                }
            }

            pub fn bounds(name: &str) -> Option<(i32, i32)> {
                if let Some(i) = Self::NAMES.iter().position(|n| *n == name) {
                    return Some(Self::BOUNDS[i]);
                }
                $(let _ = stringify!($nf); if let Some(b) = <$NT>::bounds(name) { return Some(b); })?
                None
                // NOTE: `bounds` and `doc` recurse through the nested type, so
                // they reach any depth. `all_names` has to call `all_names`
                // rather than `NAMES` for the same reason.
            }

            /// Prose for one name, for the `chess params` listing.
            pub fn doc(name: &str) -> &'static str {
                if let Some(i) = Self::NAMES.iter().position(|n| *n == name) {
                    return Self::DOCS[i];
                }
                $(let _ = stringify!($nf); if <$NT>::all_names().contains(&name) { return <$NT>::doc(name); })?
                ""
            }
        }
    };
}

/// The `eval_k` value that reproduces the net file's own scale exactly:
/// `static_eval * 2885 / 2885` is the identity for every `Score`.
pub const EVAL_K_UNITY: Score = 2885;

params! {
#[derive(Clone, Copy, Debug)]
pub struct Params {
        > pricing: Pricing;

        /// Null move pruning is off below this depth.
        pub nmp_min_depth = 2, 1, 12;
        /// Plies the null-move search gives up before the depth term.
        pub nmp_base_reduction = 3, 0, 8;
        /// ...plus `depth / this`.
        pub nmp_depth_divisor = 4, 1, 16;
        /// Reverse futility pruning applies at or below this depth.
        pub rfp_max_depth = 7, 0, 16;
        /// Centipawns per ply of depth conceded before RFP will prune.
        pub rfp_margin = 75, 0, 600;
        /// Futility pruning applies at or below this depth. 0 disables it
        /// (default: off until measured). The mirror of RFP on the alpha
        /// side: if even gaining `fut_margin * depth` leaves us below alpha,
        /// no quiet move reaches alpha at this depth. This depth enables the
        /// move-loop skip-quiets form below, which still searches every
        /// capture and is NOT refuted (084 killed only the whole-node gate,
        /// which now needs `fut_whole_node` as well).
        pub fut_max_depth = 3, 0, 16;
        /// Whole-node return-static gate: a second consumer of the futility
        /// condition that returns the static eval for the entire node instead
        /// of skipping quiets. PROXY-KILLED in 084 (+4.6..+22cp, monotone —
        /// it forfeits the captures that recover), so this stays 0; it exists
        /// only so the killed form and the live form can be told apart.
        /// Never SPRT this on.
        pub fut_whole_node = 0, 0, 1;
        /// Centipawns per ply of depth granted before futility gives up on
        /// reaching alpha.
        pub fut_margin = 100, 0, 600;
        /// Half-width of the first aspiration window, centipawns.
        pub aspiration_window = 25, 4, 400;
        /// Internal iterative deepening: minimum budget, in whole plies
        /// rounded up, for a reduced-budget ramp on a TT miss at a PV node.
        /// 0 disables it (default: off until measured). When enabled (try 4),
        /// a node with no TT move climbs probe budgets 1, 2, 4, ... plies
        /// capped at `budget - iid_reduction`, then searches at full budget.
        /// Each step orders the next through the TT; the ramp recurses, so
        /// a deep TT-empty subtree bootstraps in steps rather than searching
        /// blind at full width.
        pub iid_min_depth = 0, 0, 16;
        /// Same gate for non-PV nodes. 0 disables (default: off until
        /// measured). Needs its own threshold because the trade is
        /// different: a non-PV probe may take TT cutoffs so it is cheaper,
        /// but TT misses fire far more often off the PV. Try 6.
        pub iid_min_nonpv = 0, 0, 16;
        /// Plies of budget the IID probe gives up. 2 is the classic shape.
        pub iid_reduction = 2, 1, 8;
        /// Budget refunded, in milli-plies, at a node whose side to move is in
        /// check. This was a hardcoded `budget += PLY`; 1000 reproduces it
        /// exactly. It is here because an extension and a reduction are the
        /// same quantity with opposite signs, and only one of the two was
        /// tunable.
        pub check_extension = 1000, 0, 4000;
        /// Delta pruning slack in quiescence, centipawns. **1200 is the top
        /// of the box, which is the rule switched off in all but name** — it
        /// still fires only when the static eval is twelve pawns below alpha.
        /// That is deliberate: the rule does not pay for itself. The level fan
        /// is monotone over the whole box and never turns around (LEDGER 113,
        /// 12000 games per cell at 1+0.01, vs this same binary at 200):
        /// 150 −19.0 · 300 +22.4 · 400 +49.0 · 600 +67.6 · 800 +80.8 ·
        /// 1200 +84.1, plateau from 600 up. Confirmed at the gate's own time
        /// control: **+40.02 [35.02, 45.03]** over 6000 games at 8+0.08.
        ///
        /// The mechanism is not that pruning less buys accuracy at a price —
        /// it is that the rule made the tree *bigger*. At fixed depth 9 on
        /// m1-b1 the bench goes 179364 → 151742 nodes (−15.4%) at unchanged
        /// nps, because qsearch does the same work (q captures +0.7%) and
        /// returns better bounds, so the main search cuts off sooner (main
        /// quiet −16%, main capture −22%).
        ///
        /// Kept as a tunable rather than deleted: the deletion is a different
        /// change and nobody has played a game on it.
        pub delta_margin = 1200, 0, 1200;
        /// Plies of quiescence allowed below the point the main search runs
        /// out of budget. 64 is unreachable in practice and reproduces an
        /// uncapped quiescence exactly.
        ///
        /// This exists to answer one question before quiescence is folded into
        /// the priced search: **is quiescence depth worth what it costs?** A
        /// folded search charges a q-node the same fare as any other node, so
        /// the budget would start binding down here. If capping costs nothing,
        /// there is slack for a price to spend; if it costs a lot, the fold
        /// can only share code and the hash, not the currency.
        pub q_max_ply = 64, 1, 64;
        /// Lazy SMP diversification. 0 = every helper thread runs the same
        /// iterative-deepening schedule as the main thread and diverges only
        /// through the TT; 1 = helpers skip iterations on the classic
        /// phase/size table, so at any instant the threads sit at different
        /// depths.
        ///
        /// Measured at 0, on time-to-depth 16 over 24 searches at 6 threads,
        /// n=3 runs each: 2.70x with no skipping against 2.38x with it, and
        /// 5.6x nps against 5.3x. Skipping makes the helpers spend their time
        /// on budgets the main thread has not reached yet, which is worth less
        /// than spending it on the one it is on. This is the same conclusion
        /// Stockfish reached when it deleted its skip table. Time-to-depth is
        /// a proxy; no SPRT has been run on this.
        pub smp_skip = 0, 0, 1;
        /// Easy move: consecutive completed iterations with the same best
        /// move and a stable score before the clock trusts the move. 0
        /// disables it (default 2 since the 094 bundle gate). Time
        /// management never touches a node-limited search, so this is
        /// bench-exact at any setting — only games price it.
        pub easy_stable = 2, 0, 8;
        /// Score drift allowed between those iterations, centipawns.
        pub easy_margin = 15, 0, 100;
        /// Minimum completed depth before the easy-move shortcut applies.
        pub easy_depth = 8, 1, 64;
        /// When easy, the next iteration starts only below this percent of
        /// the normal half-deadline gate (70 = 0.35 instead of 0.50).
        pub easy_frac = 70, 5, 100;
        /// Order in-check evasions by SEE instead of killer/history. 0 =
        /// current ordering (default: off until measured). Captures, killers
        /// and the TT move keep their slots; only the quiet-evasion order
        /// changes, so the tree moves but the mechanism is one lookup.
        pub qevade_see = 0, 0, 1;
        /// Singular extensions: minimum depth for the verification search. 0
        /// disables (default: off until measured). When a hash move looks
        /// that much better than every alternative — a reduced search
        /// without it fails low by more than `se_margin * depth` — the move
        /// gets an extra ply. The one extension this engine has ever had is
        /// checks; tactics that hang on one move are the other classical
        /// case. Bench-exact at 0.
        pub se_min_depth = 0, 0, 16;
        /// Centipawns per ply of fail-low margin the verification must show
        /// before a move counts as singular.
        pub se_margin = 3, 0, 50;
        /// Singular extensions: the stored entry's depth must be within this
        /// many plies of the current depth. A stale bound from a much
        /// shallower search says nothing about how singular the move is now.
        pub se_depth_slack = 3, 0, 8;
        /// Full-window search up to this depth; deeper iterations start with
        /// a narrow window around the previous score and re-search on fail.
        pub asp_full_depth = 4, 1, 12;
        /// Aspiration window growth, percent: after each fail,
        /// `delta = delta * this / 100`. 150 reproduces the old
        /// `delta += delta / 2` exactly (must stay above 100, or the window
        /// never grows and the loop never ends).
        pub asp_growth = 150, 110, 400;
        /// History bonus ceiling: `bonus = min(depth^2, this)`. The shape
        /// (quadratic in depth) stays hardcoded — TU-7 owns shape variants —
        /// this is the cap.
        pub hist_cap = 1200, 0, 5000;
        /// History gravity divisor: entries saturate toward ±this instead of
        /// overflowing, so old information decays rather than dominating.
        /// Shared with continuation history, which learns on the same events.
        pub hist_grav = 16384, 1024, 65536;
        /// Quiescence SEE threshold: after delta pruning passes a capture,
        /// skip it unless it wins at least this much material by static
        /// exchange. 0 is "don't drop exchanges".
        pub qsee_thresh = 0, -500, 500;
        /// Time manager: assumed moves to go when UCI sends no `movestogo`.
        pub tm_moves_to_go = 24, 1, 100;
        /// Increment share, percent of the increment added to the allotment.
        pub tm_inc_pct = 90, 0, 100;
        /// Never spend more than `time_left / this` on one move.
        pub tm_cap_div = 3, 2, 8;
        /// Movetime mode: keep this many milliseconds in hand (`movetime` is
        /// a ceiling the GUI enforces, not a suggestion).
        pub tm_movetime_margin = 20, 0, 500;
        /// Start the next iteration only below this percent of the deadline.
        /// 50 is "don't start what you can't finish".
        pub tm_gate_pct = 50, 5, 100;
        /// Eval readout scale, in tenths of a cp per K unit: 2885 is the net
        /// file's own K = 288.5, so the default is the identity. A
        /// search-side multiplier on every static eval — margins stay put,
        /// so tuning this rescales every margin at once (TU-9). Does not
        /// touch the net file, the TT's stored scores' meaning (they are
        /// stored scaled, consistently), or the contempt/draw readouts in
        /// `eval.rs`, which stay in net units.
        pub eval_k = 2885, 1000, 6000;
        /// SR-5a: RFP concedes `rfp_margin * (depth - improving * this/1000)`
        /// — milli-plies of depth taken off the margin when the side to move
        /// has improved on its static eval of two plies ago. 1000 is
        /// Stockfish's shape. 0 disables (bench-exact).
        pub rfp_improving = 0, 0, 4000;
        /// SR-7: null-move reduction grows by `min((eval - beta) / this,
        /// nmp_eval_max)` plies. 0 disables (bench-exact).
        pub nmp_eval_div = 0, 0, 2000;
        /// Cap on that extra null-move reduction, plies.
        pub nmp_eval_max = 3, 0, 8;
        /// SR-4 razoring: at non-PV nodes with `depth <= this` and
        /// `eval + raz_margin * depth <= alpha`, a null-window quiescence
        /// that also fails low is returned. Unlike 084's whole-node
        /// futility it *searches* the captures that recover. 0 disables.
        pub raz_max_depth = 0, 0, 8;
        /// Centipawns per ply of depth for the razoring gate.
        pub raz_margin = 250, 0, 2000;
        /// SR-6 ProbCut: minimum depth. 0 disables (bench-exact). A capture
        /// whose SEE clears `beta + pc_margin - eval`, confirmed first by
        /// quiescence and then by a search `pc_reduction` plies shallower,
        /// against `beta + pc_margin`, cuts the node.
        pub pc_min_depth = 0, 0, 16;
        pub pc_margin = 200, 0, 2000;
        pub pc_reduction = 4, 1, 8;
        /// Internal iterative *reduction*: a node with no hash move and at
        /// least this depth loses a ply. 0 disables (bench-exact). Not IID
        /// (074, which searched more); this searches less where ordering is
        /// blind and lets the next visit find a hash move.
        pub iir_min_depth = 3, 0, 16;
        /// 1 = IIR only at PV and expected-cut nodes (Stockfish's gate).
        pub iir_cut_only = 0, 0, 1;
        /// SR-2d multi-cut: when the singular verification fails *high*
        /// above beta, some other move also beats beta — return that bound.
        pub se_multicut = 0, 0, 1;
        /// SR-2d negative extension: when the verification fails high but
        /// the hash score is at least beta, the hash move is searched this
        /// many milli-plies shallower. 0 disables.
        pub se_neg_ext = 0, 0, 3000;
        /// Pruning eval refined by the hash bound: a lower bound above the
        /// static eval, or an upper bound below it, replaces it for RFP,
        /// null move and razoring. 0 off; 1 main-search entries only; 2 any
        /// entry including quiescence ones.
        pub tt_eval_adj = 0, 0, 2;
        /// SR-15a: static eval is scaled by `(this - halfmove) / this`, so a
        /// shuffling line drifts toward a draw score. 0 disables.
        pub hmc_scale = 200, 0, 1000;
        /// Capture history: `[piece][to][victim]`, learned on cutoffs like
        /// history, added to MVV-LVA as `capthist / this`. 0 disables both
        /// the update and the read (bench-exact).
        pub capt_hist_div = 0, 0, 64;
        /// Correction history (`--corr pawn`) updates only when the bound
        /// agrees with the residual's sign. 0 = the phase-1 rule (every node).
        pub corr_bound_aware = 0, 0, 1;
        /// LC-2 two-stage gap: at nodes with at least this depth, every
        /// non-hash child gets a null-window-ish quiescence search before the
        /// loop (004: r² 0.03 for `move_gain`'s gap, 0.71 for this one). 0
        /// disables (bench-exact). What the values feed is picked below.
        pub lc2_min_depth = 0, 0, 16;
        /// 1 = demote to the bottom of the list every move whose quiescence
        /// value is `lc2_hang` below `min(static eval, alpha + 1)` — a
        /// quiescence-strength SEE for quiet moves; the order among the rest
        /// is untouched. (Ranking *by* the value instead was 7.7x the bench
        /// nodes: it throws away history and killers.)
        pub lc2_order = 0, 0, 1;
        pub lc2_hang = 100, 0, 2000;
        /// 1 = the price's gap is `best - qvalue` instead of the static
        /// `move_gain` prediction. Needs `c_gap` non-zero to reach the price.
        pub lc2_price = 0, 0, 1;
        /// The probe window is `[alpha - this, alpha + 1]` from the parent's
        /// side; values below it read as `alpha - this`. Width is the cost:
        /// at 800 a probe was ~12 quiescence nodes (bench 5.4x), at 100 ~4.
        pub lc2_win = 100, 50, 4000;
        /// Follow-up history: continuation history keyed on *our* previous
        /// move (two plies up), `[its piece][its to][piece][to]`, added to
        /// quiet-move ordering and learned on the same events. 0 disables
        /// both the update and the read (bench-exact).
        pub conth2 = 0, 0, 1;
        /// Hard limit as a percent of the soft allotment, still capped at
        /// `time_left / tm_cap_div`. 100 = the old single deadline (an
        /// iteration started just under the gate is usually cut unfinished).
        pub tm_hard_pct = 500, 100, 600;
        /// Soft-gate stretch per recent best-move change, percent. Changes
        /// decay by half each iteration (Stockfish's `totBestMoveChanges`).
        /// 0 disables.
        pub tm_instab = 100, 0, 300;
        /// Soft-gate stretch for a falling root score: per-mille of extra
        /// time per centipawn dropped since the previous iteration, capped
        /// at +100%. 0 disables.
        pub tm_fall = 20, 0, 100;
        /// 1 = predictive time manager (TM2): per-move target
        /// `T = (t + mtg·inc) / mtg` with `mtg` = material on the board
        /// (1/3/3/5/9, both sides, 78 at the start); the next iteration
        /// starts only if `elapsed + dt_last²/dt_prev` fits under
        /// `min(tm2_go_pct·T, t/tm2_cap_div)`; hard stop at
        /// `min(tm2_hard_pct·T, t/tm2_cap_div)`. Ignores the gate, the
        /// easy move and the stretches above. 0 = the soft/hard manager.
        pub tm_mode = 0, 0, 1;
        /// TM2: 1 = moves-to-go from material, 0 = `tm_moves_to_go`.
        pub tm2_mat = 1, 0, 1;
        /// TM2: floor on the material moves-to-go (KR v K would be 5).
        pub tm2_mtg_min = 10, 1, 80;
        /// TM2: moves-to-go is this percent of the material sum.
        pub tm2_mat_pct = 100, 20, 200;
        /// Soft/hard manager: if > 0, moves-to-go is this percent of the
        /// material sum (floor `tm2_mtg_min`) instead of `tm_moves_to_go`.
        /// 0 disables.
        pub tm_mat_pct = 0, 0, 200;
        /// TM2: start the next iteration only if its predicted finish is
        /// under this percent of the target.
        pub tm2_go_pct = 150, 50, 400;
        /// TM2: hard stop, percent of the target.
        pub tm2_hard_pct = 400, 100, 1000;
        /// TM2: never plan past `time_left / this`, gate or hard stop.
        pub tm2_cap_div = 2, 2, 8;
        /// TM2: iterations up to this depth run without the prediction.
        pub tm2_free_depth = 4, 2, 12;
        /// SR-10: at the first quiescence ply (not in check), also search
        /// quiet moves that give check and do not lose material by SEE.
        /// 0 disables (bench-exact).
        pub q_checks = 0, 0, 1;
        /// SR-11: milli-plies of extension for a capture that recaptures on
        /// the square the previous move captured on. Node-level, so unlike
        /// the `c_recap` price (065) it reaches the first move. 0 disables.
        pub recap_ext = 0, 0, 2000;
    }
}

/// The price list — the object this engine is now built around.
///
/// Every child of a node is charged
///
/// ```text
///   spend  = fare + clamp(price, floor, max_base + max_slope*depth)
///   price  = c0 + c_rank*L(rank) + c_gap*L(1 + gap/gap_unit)
///               + c_hist*H + c_rank_depth*L(rank)*L(depth) - pv_discount
/// ```
///
/// with `L = log2`, and inherits `parent_budget - spend`. A child left with a
/// non-positive budget drops into quiescence; one left below `skip_below` is
/// not searched at all. Coefficients are in **milli-plies** so they stay
/// integers a tuner can step.
///
/// Four things are deliberate.
///
/// **The bound is not for sale.** A beta cutoff still terminates the move loop,
/// and the transposition table still returns on a sufficient bound. Alpha-beta's
/// power is the *proof* that a subtree cannot affect the root, and no allocation
/// policy buys that back — softening it costs 5-10x in nodes immediately.
/// Pricing decides how much a relevant move gets, never whether a provably
/// irrelevant one is visited.
///
/// **The gap is in centipawns, not in rank.** `L(rank)` is the old LMR feature
/// and it is a proxy: rank informs only because a good ordering correlates it
/// with value. `gap` is the quantity itself — how far below the running best
/// this move is predicted to land, from `eval::move_gain`. Optimal-computing-
/// budget-allocation theory says samples should go as `(sigma/gap)^2`; since
/// nodes are exponential in depth, that makes the *reduction* linear in
/// `log(gap)`, which is why the term is logged. It also explains why the
/// hand-fit `log(rank)` table worked at all: under a roughly exponential
/// ordering, `log(rank)` is a stand-in for `log(gap)`.
///
/// **The ceiling grows with depth.** This is the old `r.clamp(0, depth-2)`,
/// made explicit — and letting the ceiling push a child's budget past zero is
/// what late-move pruning was, made continuous.
///
/// **Nothing here is gated on move type.** There is no "don't reduce captures"
/// rule; a good capture simply has a small gap and therefore a low price. That
/// unification is the point, and it is also the main thing that could go wrong.
params! {
#[derive(Clone, Copy, Debug)]
pub struct Pricing {
        > feat: Feat;

        /// Charged to every child for descending one level. Floored at 1: a
        /// zero fare makes the budget non-decreasing and the search
        /// non-terminating.
        pub fare = PLY, 1, 4 * PLY;
        /// Intercept, milli-plies.
        pub c0 = 0, -4000, 4000;
        /// Milli-plies per unit of `log2(rank)`.
        pub c_rank = 250, -4000, 4000;
        /// Milli-plies per unit of `log2(1 + gap/gap_unit)`. Zero, and that is
        /// a result rather than an omission: on 14000 labelled positions
        /// turning it off is worth -0.25 +/- 0.20 cp while turning it up is
        /// worth +0.09. It does not earn its place as formulated. `move_gain`
        /// and the plumbing stay; see LEDGER 007.
        pub c_gap = 0, -4000, 4000;
        /// Centipawns per unit of gap, before the log. Floored at 1; zero
        /// divides by zero.
        pub gap_unit = 32, 1, 2000;
        /// Milli-plies per unit of scaled history (16 units = one full bin).
        pub c_hist = 100, -4000, 4000;
        /// Milli-plies per unit of `log2(rank)*log2(depth)` — the interaction
        /// term the old `ln(d)*ln(m)` table consisted of.
        pub c_rank_depth = 204, -4000, 4000;
        /// Subtracted at PV nodes, milli-plies.
        pub pv_discount = 1000, -8000, 8000;
        /// Floor on the price. Negative values buy extensions.
        pub floor = 0, -4000, 4000;
        /// Ceiling is `max_base + max_slope * depth`, milli-plies. `depth + 1`
        /// plies, not `depth - 2`: the old LMR clamp was the binding
        /// constraint and it was too tight. Loosening it is worth -0.44 +/-
        /// 0.26 cp; raising it further changes nothing at all, because the
        /// price formula saturates below it.
        pub max_base = 1000, 0, 20000;
        /// Slope of that ceiling, milli-plies per ply of depth.
        pub max_slope = 1000, 0, 20000;
        /// A child left below this budget is not searched at all. Budget
        /// units, normally negative. Late move pruning with a price tag.
        pub skip_below = -PLY, -16 * PLY, PLY;
        /// Milli-plies charged to a move that loses material by static
        /// exchange evaluation. A **gap** term: `move_gain` is a PST delta
        /// plus the raw victim value and resolves no exchange, which is why it
        /// explains 3% of the variance in the gap it predicts; adding an SEE
        /// veto takes that to 8% (`library/004`). Zero by default, and the
        /// SEE call is skipped entirely while it is zero, so a playing build
        /// Milli-plies charged to a move that recaptures on the square the
        /// previous move captured on. A **sigma** term: mid-exchange
        /// positions are where a shallow search is least reliable, so the
        /// Milli-plies per off-PV decision on the path from the root — the
        /// number of times the search took a move other than the first at a
        /// node above this one. A **cost-of-error** term: it is the log of the
        /// probability that this subtree's value ever reaches the root
        /// decision. `pv_discount` is the binary version of this and the two
        /// Milli-plies applied to a child that would land at or below zero
        /// budget — i.e. one that drops into quiescence.
        ///
        /// The boundary at zero is currently a **cliff**: a child left with 1
        /// budget unit gets a full recursive node, one left with 0 gets a
        /// capture-only search, and nothing in the price list knows the cliff
        /// is there. 27.9% of priced children land on the far side of it
        /// (`library/008`). Negative pulls marginal children back over;
        /// positive pushes them decisively across.
        ///
        /// Applied after the base price, so "would it drop" is evaluated once
        /// against the price without this term rather than being circular.
        pub c_qdrop = 0, -4000, 4000;
    }
}

/// Everything a feature is allowed to look at.
///
/// One borrow of the node's state, handed to every feature expression. A
/// feature that needs something not in here does not silently reach for it —
/// the field gets added, on purpose, and every feature can then see it.
pub struct Ctx<'a> {
    pub b: &'a Board,
    pub mv: Move,
    /// Position in the ordered move list. 0 is the main line and never priced.
    pub rank: usize,
    /// Budget *remaining*, in plies, rounded up.
    pub depth: i32,
    /// Plies already searched from the root. Not the same thing as `depth`.
    pub ply: usize,
    /// Centipawns this move is predicted to land below the running best.
    pub gap: Score,
    /// Raw history score, quiet moves only.
    pub hist: i32,
    pub is_pv: bool,
    pub in_check: bool,
    /// The parent's budget after the fare.
    pub full: i32,
    /// How the search got to this node. See `PathInfo`.
    pub prev: PathInfo,
    /// The side to move has improved on its static eval of two plies ago.
    pub improving: bool,
    /// There was an eval two plies ago to compare with (not at plies 0-1,
    /// not when either node was in check). Without it `improving` is false
    /// by default, not by measurement.
    pub impr_known: bool,
    /// This node is expected to fail high (non-PV, reached as the kind of
    /// child whose first move should cut).
    pub cut_node: bool,
    /// The hash move at this node is a capture.
    pub tt_capture: bool,
    /// Continuation-history score of this move (quiet moves only), the
    /// half of the ordering signal the legacy `c_hist` term never saw.
    pub conth: i32,
}

/// Declare a search feature once, and generate everything else.
///
/// This is the seam the project is built around: the plan is to keep bringing
/// new information to the search — is it a check, was the last move a capture,
/// how far from the PV are we — and to keep whichever terms measure well. That
/// only stays cheap if adding one is a single line, so this macro generates
/// the coefficient, its box, its documentation, the registry entry, the
/// extraction, the lazy gating and the price term from one declaration.
///
/// ```text
/// name : GROUP, KIND, COST, default, lo, hi, |c| expression;
/// ```
///
/// * **GROUP** is which quantity in the allocation law the feature estimates —
///   `Sigma` (how wrong a shallow search of this child will be), `Gap` (how far
///   behind the running best), or `Cost` (what an error here costs at the
///   root). `library/004` derives `b ~ (PLY/gamma)*[log2(sigma) - log2(gap) +
///   log2(C)]`, so the group is what fixes a feature's sign *before* it is
///   measured. It is documentation, not code: nothing enforces it, and a
///   feature whose measured sign fights its group is telling you something.
/// * **KIND** is how it enters the price. `Lin` multiplies the coefficient by
///   the value (an indicator is just `Lin` over 0/1); `Log` multiplies by
///   `log2` of it, in sixteenths.
/// * **COST** is what computing it costs. `Free` is register arithmetic and is
///   always computed. `Gated` is expensive — a `see()` call, a movegen — and
///   is computed **only when its coefficient is non-zero**, so a feature that
///   has not earned its place costs one predictable not-taken branch.
///
/// The gate is a branch on a runtime field, not a compile-time constant: these
/// are tunable at runtime by design (`chess tune`, and UCI under `--features
/// tune`), so the optimiser cannot fold them away and is not expected to. What
/// it buys is that the *expensive* half never runs. See `library/013`.
macro_rules! features {
    (
        $(
            $(#[doc = $d:literal])*
            $name:ident : $grp:ident, $kind:ident, $cost:ident, $def:expr, $lo:expr, $hi:expr,
                | $c:ident | $body:expr;
        )*
    ) => {
        params! {
            /// Coefficients for the declared feature set, in milli-plies.
            /// Generated by `features!`; nested inside `Pricing` so the tuner,
            /// `chess params` and UCI see one flat namespace.
            #[derive(Clone, Copy, Debug)]
            pub struct Feat {
                $( $(#[doc = $d])* pub $name = $def, $lo, $hi; )*
            }
        }

        /// One computed value per declared feature. Indicators are 0/1.
        #[derive(Clone, Copy, Debug, Default)]
        pub struct Feats {
            $( pub $name: i32, )*
        }

        impl Feats {
            /// Compute every feature for one child.
            #[inline(always)]
            pub fn extract(p: &Feat, c: &Ctx) -> Feats {
                Feats {
                    $( $name: features!(@compute $cost, p.$name, $c, c, $body), )*
                }
            }
        }

        impl Feat {
            /// Which of sigma / gap / cost-of-error each feature estimates.
            pub const GROUPS: &'static [&'static str] = &[$( stringify!($grp) ),*];
            /// How each feature enters the price: `Lin` or `Log`.
            pub const KINDS: &'static [&'static str] = &[$( stringify!($kind) ),*];
            /// `Free` or `Gated`. See the macro docs.
            pub const COSTS: &'static [&'static str] = &[$( stringify!($cost) ),*];

            /// The feature terms, summed, in milli-plies.
            #[inline(always)]
            pub fn sum(&self, v: &Feats) -> i32 {
                let mut mp = 0i32;
                $( mp += features!(@term $kind, self.$name, v.$name); )*
                mp
            }
        }
    };

    // A Free feature is always computed; a Gated one only when it is paid for.
    (@compute Free,  $coef:expr, $c:ident, $ctx:expr, $body:expr) => {{ let $c = $ctx; $body as i32 }};
    (@compute Gated, $coef:expr, $c:ident, $ctx:expr, $body:expr) => {
        if $coef != 0 { let $c = $ctx; $body as i32 } else { 0 }
    };

    (@term Lin, $coef:expr, $v:expr) => { $coef * $v };
    (@term Log, $coef:expr, $v:expr) => { ($coef * Pricing::log2_q4($v.max(0) as u32)) >> 4 };
}

features! {
    /// The move loses material by static exchange evaluation.
    ///
    /// `move_gain` is a PST delta plus the raw victim value and resolves no
    /// exchange, which is why it explains 3% of the variance in the gap it
    /// predicts; an SEE veto takes that to 8% (`library/004`). The one
    /// quantitative prediction that theory made, and the first feature to
    /// pass an SPRT: LEDGER 062/064.
    c_see: Gap, Lin, Gated, 2000, -4000, 20000,
          |c| !crate::eval::see(c.b, c.mv, 0);
    /// The move recaptures on the square the previous move captured on.
    /// Mid-exchange positions are where a shallow search is least reliable, so
    /// the prior is a discount. Fires in 22-68 of 1000 corpus positions —
    /// unmeasured at n=1000, not measured at zero (LEDGER 062).
    c_recap: Sigma, Lin, Free, 0, -4000, 4000,
          |c| c.prev.mv != Move::NONE && c.prev.mv.is_capture() && c.prev.mv.to() == c.mv.to();
    /// Off-PV decisions on the path from the root — how many times the search
    /// took a move other than the first above this node.
    ///
    /// **Measured and closed.** It loses to a flat information-free price by
    /// 3.2 +/- 1.9 cp at matched depth. The mechanism: the budget is already
    /// `root - sum of prices on the path`, so accumulated distance from the PV
    /// is carried additively by the currency and a plain count adds nothing.
    /// Kept declared because a negative result is worth being able to re-run.
    c_offpv: Cost, Lin, Free, 0, -1000, 1000,
          |c| c.prev.off_pv.min(32) as i32;
    /// We have left our believed-best at least once on the path from the root
    /// (sticky bit: the "us off" square of the 2x2). The `c_offpv` count was
    /// closed as redundant with the budget currency; this splits *what kind*
    /// of path it was instead of *how long* (LEDGER 062), starting with the
    /// coarsest such split: us vs them.
    c_usoff: Cost, Lin, Free, 0, -1000, 1000,
          |c| c.prev.off_us as i32;
    /// They have left their believed-best at least once on the path from the
    /// root (the "them off" square). Screen together with `c_usoff`; the
    /// (off, off) square is the sum of the two until an interaction term says
    /// otherwise.
    c_themoff: Cost, Lin, Free, 0, -1000, 1000,
          |c| c.prev.off_them as i32;
    /// This move itself leaves our believed-best (rank > 0 at one of our
    /// nodes). The transition into the "us off" square, charged once, rather
    /// than the state `c_usoff` which charges every node below it.
    c_usleave: Cost, Lin, Free, 0, -1000, 1000,
          |c| (c.rank > 0 && c.ply % 2 == 0) as i32;
    /// This move itself leaves their believed-best (rank > 0 at one of their
    /// nodes). The transition into the "them off" square, charged once.
    c_themleave: Cost, Lin, Free, 0, -1000, 1000,
          |c| (c.rank > 0 && c.ply % 2 == 1) as i32;
    /// Us leaving while them still on: (on,on) -> (off,on). One of the four
    /// directed edges of the 2x2; a potential table is the special case where
    /// opposite paths agree (c_usleave_on + c_themleave_off ==
    /// c_themleave_on + c_usleave_off).
    c_usleave_on: Cost, Lin, Free, 0, -2000, 2000,
          |c| (c.rank > 0 && c.ply % 2 == 0 && c.prev.off_them == 0) as i32;
    /// Us leaving while them already off: (on,off) -> (off,off).
    c_usleave_off: Cost, Lin, Free, 0, -2000, 2000,
          |c| (c.rank > 0 && c.ply % 2 == 0 && c.prev.off_them != 0) as i32;
    /// Them leaving while us still on: (on,on) -> (on,off).
    c_themleave_on: Cost, Lin, Free, 0, -2000, 2000,
          |c| (c.rank > 0 && c.ply % 2 == 1 && c.prev.off_us == 0) as i32;
    /// Them leaving while us already off: (off,on) -> (off,off).
    c_themleave_off: Cost, Lin, Free, 0, -2000, 2000,
          |c| (c.rank > 0 && c.ply % 2 == 1 && c.prev.off_us != 0) as i32;
    /// Plies already searched from the root. Distinct from `depth`, which is
    /// budget remaining: two nodes with the same money left are not the same
    /// node if one is 2 plies from the root and the other is 12.
    c_ply: Cost, Lin, Free, 0, -1000, 1000,
          |c| c.ply.min(64) as i32;
    /// `log2` of the budget remaining, on its own rather than only interacted
    /// with rank. `library/004` derives a standalone `log2(C)` term and notes
    /// that `Pricing` carried only the rank x depth interaction, so this is
    /// the one term the allocation law asks for that was never in the list.
    c_depth: Cost, Log, Free, 0, -4000, 4000,
          |c| c.depth.max(1);
    /// The side to move is in check at this node.
    c_incheck: Cost, Lin, Free, 0, -8000, 8000,
          |c| c.in_check;
    /// The move gives check. Costs a `make_move` plus a check test, so it is
    /// gated: nothing pays for it until it earns a coefficient.
    c_gives_check: Cost, Lin, Gated, 0, -8000, 8000,
          |c| c.b.make_move(c.mv).in_check();
    /// SR-5b: the side to move has NOT improved on its static eval of two
    /// plies ago (false in check or without an eval to compare). Moves at a
    /// worsening node are less likely to lift alpha, so the sign is a
    /// higher price.
    c_notimpr: Gap, Lin, Free, 0, -4000, 4000,
          |c| c.impr_known && !c.improving;
    /// SR-8: this is an expected cut node. Its first move should cut, so a
    /// later child's value is less likely to matter at the root.
    c_cutnode: Cost, Lin, Free, 0, -4000, 4000,
          |c| c.cut_node && !c.is_pv;
    /// The hash move is a capture and this move is quiet: the position's
    /// business is tactical, so a quiet alternative is less likely best.
    c_ttcapt: Gap, Lin, Free, 0, -4000, 4000,
          |c| c.tt_capture && c.mv.is_quiet();
    /// Continuation history in the price, scaled like `c_hist`'s `h`
    /// (`-score/64`, clamped to 256) but without its `>> 4`, so a
    /// coefficient of ~6 weighs it the same as `c_hist = 100` weighs
    /// history. Negative history costs more.
    c_conth: Gap, Lin, Free, 0, -400, 400,
          |c| (-c.conth / 64).clamp(-256, 256);
}

impl Pricing {
    /// `log2(x)` in sixteenths. Exact at powers of two, linear in the mantissa
    /// between them: monotone, four instructions, and far more precision than a
    /// price needs.
    #[inline]
    pub fn log2_q4(x: u32) -> i32 {
        if x == 0 {
            return 0;
        }
        let n = 31 - x.leading_zeros();
        let frac = if n >= 4 { (x >> (n - 4)) & 15 } else { (x << (4 - n)) & 15 };
        n as i32 * 16 + frac as i32
    }

    /// What this child costs on top of the fare, in budget units.
    ///
    /// Three stages, and they are separate on purpose.
    ///
    /// 1. **The legacy terms.** `rank`, `gap`, `hist` and the rank x depth
    ///    interaction share **one** rounding shift. Splitting them into
    ///    `features!` would round each separately and change the tree, so they
    ///    stay hand-written and bit-frozen until something forces a re-fit.
    ///    They are not special otherwise; `c_rank` is a `Gap` term and
    ///    `c_rank_depth` a `Cost` one under the same taxonomy.
    /// 2. **The declared features**, summed by generated code.
    /// 3. **Shaping**, which is a function of the price rather than of the
    ///    position — currently only `c_qdrop`. This is why the quiescence
    ///    boundary is not a `features!` entry: a feature answers "what is this
    ///    move like", and `c_qdrop` answers "where did the price land".
    #[inline]
    pub fn price(&self, f: &Feats, rank: usize, depth: i32, gap: Score, hist: i32, is_pv: bool, full: i32) -> i32 {
        let l_rank = Self::log2_q4(rank as u32);
        let l_depth = Self::log2_q4(depth.max(1) as u32);
        let l_gap = Self::log2_q4((gap.max(0) / self.gap_unit.max(1)) as u32 + 1);
        // History runs to +/-16384; scale it to +/-16.0 in Q4. Negative history
        // means the move has failed before, so it should cost *more*.
        let h = (-hist / 64).clamp(-256, 256);

        let mut mp = self.c0
            + ((self.c_rank * l_rank
                + self.c_gap * l_gap
                + self.c_hist * h
                + self.c_rank_depth * ((l_rank * l_depth) >> 4))
                >> 4);
        if is_pv {
            mp -= self.pv_discount;
        }

        mp += self.feat.sum(f);

        // The quiescence cliff. Evaluated against the price so far, so "would
        // this child drop into quiescence" has one answer rather than
        // depending on its own answer.
        if self.c_qdrop != 0 && full - (mp.clamp(self.floor, i32::MAX / 2) * PLY / 1000) <= 0 {
            mp += self.c_qdrop;
        }

        let ceiling = (self.max_base + self.max_slope * depth).max(self.floor);
        mp = mp.clamp(self.floor, ceiling);
        mp * PLY / 1000
    }
}

/// Material on the board at 1/3/3/5/9, both sides: 78 at the start. The
/// time managers read it as a moves-to-go estimate.
fn material(b: &Board) -> u32 {
    const VAL: [u32; 5] = [1, 3, 3, 5, 9];
    [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen]
        .iter()
        .zip(VAL)
        .map(|(&pt, v)| b.pieces(pt).count() * v)
        .sum()
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Limits {
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    /// A ceiling in **work units** rather than nodes -- one unit is one
    /// main-search node at the deploy configuration. A node is not a unit of
    /// work (a quiescence node costs 0.275 of one, `library/007`), so a
    /// fixed-*node* budget silently discounts any candidate that shifts effort
    /// into quiescence, and quiescence is already 44.6% of search time. This
    /// is the budget the tuner should spend. `--features work` only; ignored
    /// by a playing build, which has no counter to test.
    pub work: Option<u64>,
    pub movetime: Option<u64>,
    pub time: [Option<u64>; 2],
    pub inc: [u64; 2],
    pub movestogo: Option<u32>,
    pub infinite: bool,
}

impl Limits {
    /// Returns the soft deadline. Simple but not silly: reserve for the moves
    /// still to come, add most of the increment, never spend more than a third
    /// of what is left.
    fn allotment(&self, stm: usize, p: &Params) -> Option<Duration> {
        if self.infinite {
            return None;
        }
        if let Some(mt) = self.movetime {
            return Some(Duration::from_millis(mt.saturating_sub(p.tm_movetime_margin.max(0) as u64)));
        }
        let t = self.time[stm]?;
        let inc = self.inc[stm];
        let mtg = self.movestogo.unwrap_or(p.tm_moves_to_go.max(1) as u32).max(1) as u64;
        let budget = t / mtg + (inc * p.tm_inc_pct.max(0) as u64) / 100;
        Some(Duration::from_millis(budget.min(t / p.tm_cap_div.max(2) as u64).max(5)))
    }
}

pub struct Shared {
    pub tt: TranspositionTable,
    pub stop: AtomicBool,
    /// Every thread's node count, summed. Published in blocks of 2048 from
    /// `check_time`, so the hot path pays one relaxed `lock xadd` per 2048
    /// nodes and nothing else. `ThreadData::nodes` stays the per-thread
    /// count, which is what `bench` fingerprints.
    pub nodes: AtomicU64,
}

impl Shared {
    pub fn new(hash_mb: usize) -> Shared {
        Shared {
            tt: TranspositionTable::new(hash_mb),
            stop: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
        }
    }
}

/// What the path from the root into a node looked like, one entry per ply.
///
/// This is the seam for every feature that is a property of the *route* rather
/// than of the move being priced — recaptures, distance from the PV, and later
/// continuation and correction history, which need the same thing. Written on
/// the way down and never read above the current ply, so it costs one store
/// per made move.
#[derive(Clone, Copy)]
pub struct PathInfo {
    /// The move made to reach this node. `Move::NONE` at the root and after a
    /// null move.
    pub mv: Move,
    /// How many times the search took a move other than the first at a node
    /// above this one.
    pub off_pv: u32,
    /// Sticky bits splitting `off_pv` by side: `off_us` went 0->1 the first
    /// time the search took a non-first move at one of *our* nodes (even ply
    /// from the root), `off_them` at one of *theirs*. Together they are the
    /// 2x2 state (on/on, off/on, on/off, off/off); each bit only moves one
    /// way going down the tree.
    pub off_us: u8,
    pub off_them: u8,
    /// Deviation *counts* per side, capped at 3 — the coordinates of the
    /// potential table below. Bits answer "did we ever leave"; counts answer
    /// "how far off are we".
    pub off_us_n: u8,
    pub off_them_n: u8,
}

impl Default for PathInfo {
    fn default() -> PathInfo {
        PathInfo { mv: Move::NONE, off_pv: 0, off_us: 0, off_them: 0, off_us_n: 0, off_them_n: 0 }
    }
}

/// The 2-D potential table: `POT[x][y]` is the value of sitting `x` us-
/// deviations and `y` them-deviations off the believed-best path, in
/// milli-plies. A priced child pays `POT[next] - POT[current]`, so a whole
/// path telescopes to `POT[final] - 0` however it got there. Row 0 and column
/// 0 are anchored at zero (single-sided deviations pay nothing here — that
/// shape is what the `c_usleave*` scalars cover); only doubly-off squares
/// carry value.
///
/// Deliberately NOT in `Params`: it is a table, not a scalar, so it does not
/// fit the registry, the tuner cannot see it, and the proxy cannot screen it.
/// It is set per-process from `--pot` / `$CHESS_POT` (nine comma-separated
/// integers, row-major over x,y in 1..=3) so one binary plays N arms in a
/// tournament. Zero by default, in which case the move loop below is
/// bit-identical (bench-exact).
static POT: std::sync::OnceLock<[[i32; 4]; 4]> = std::sync::OnceLock::new();
static POT_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Parse `--pot v,...` / `$CHESS_POT`. Called once from `crate::init`.
pub fn init_pot() {
    let spec = std::env::args().skip_while(|a| a != "--pot").nth(1)
        .or_else(| | std::env::var("CHESS_POT").ok());
    let spec = match spec {
        Some(s) => s,
        None => return,
    };
    let vs: Vec<i32> = spec.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    if vs.len() != 9 {
        eprintln!("--pot needs 9 comma-separated integers, got {}", vs.len());
        return;
    }
    let mut t = [[0i32; 4]; 4];
    for (k, v) in vs.iter().enumerate() {
        t[1 + k / 3][1 + k % 3] = (*v).clamp(-8000, 8000);
    }
    if t.iter().any(|r| r.iter().any(|&v| v != 0)) {
        POT_ANY.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let _ = POT.set(t);
}

#[inline]
fn pot_term(ply: usize, rank: usize, prev: &PathInfo) -> i32 {
    if !POT_ANY.load(std::sync::atomic::Ordering::Relaxed) {
        return 0;
    }
    let t = POT.get_or_init(|| [[0i32; 4]; 4]);
    let xu = prev.off_us_n.min(3) as usize;
    let xt = prev.off_them_n.min(3) as usize;
    let nu = (xu + (rank > 0 && ply % 2 == 0) as usize).min(3);
    let nt = (xt + (rank > 0 && ply % 2 == 1) as usize).min(3);
    // Rank 0 needs no special case: no deviation means next == current and
    // the difference is zero by construction — the main line pays the fare.
    t[nu][nt] - t[xu][xt]
}

/// Correction history (`library/015`), phase 1: a per-thread pawn-structure
/// table of eval residuals. Off by default; `--corr pawn` / `$CHESS_CORR=pawn`
/// switches it on. Off means one relaxed load per node and bit-identical
/// search (bench-exact), the same dormant-carriage deal as POT above.
static CORR_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Parse `--corr <mode>` / `$CHESS_CORR`. Called once from `crate::init`.
/// The only mode is `pawn`; anything else leaves the feature off.
pub fn init_corr() {
    let spec = std::env::args().skip_while(|a| a != "--corr").nth(1)
        .or_else(| | std::env::var("CHESS_CORR").ok());
    if matches!(spec.as_deref().map(str::trim), Some("pawn") | Some("1")) {
        CORR_ANY.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

#[inline]
fn corr_enabled() -> bool {
    CORR_ANY.load(std::sync::atomic::Ordering::Relaxed)
}

/// Continuation history: what the previous move was predicts what replies
/// well here. `[prev_piece][prev_to][piece][to]`, learned within one search
/// like plain history. On by default since the 094 bundle gate;
/// `--no-conth` / `$CHESS_CONTH=0` switches it off (the off-ramp the next
/// gate's B arm needs). Off means one relaxed load per quiet move scored
/// and bit-identical search (bench-exact), the same dormant-carriage deal
/// as CORR above — except the default is on, so the bench moves with it.
///
/// The cheap O5/O6 forms (countermove `[side][prev_to]`, piece/to history)
/// are both measured dead (ORDERING.md); this is the full Stockfish-shaped
/// form, which is untried here.
static CONTH_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Parse `--no-conth` / `$CHESS_CONTH=0` (on by default). Called once from
/// `crate::init`. `--conth` / `$CHESS_CONTH=1` are accepted as no-ops, so
/// the 094 command line keeps working; an explicit off beats them.
pub fn init_conth() {
    let off = std::env::args().any(|a| a == "--no-conth")
        || matches!(std::env::var("CHESS_CONTH").as_deref(), Ok("0"));
    if !off {
        CONTH_ANY.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

#[inline]
fn conth_enabled() -> bool {
    CONTH_ANY.load(std::sync::atomic::Ordering::Relaxed)
}

/// History persistence across moves (FX-1): `go_parallel` used to build a
/// fresh `ThreadData` for every `go`, so history, killers, continuation
/// history and the correction table started empty on each move. The UCI
/// `Engine` now owns one `ThreadData` per thread and reuses it across moves;
/// this flag picks the move-start rule. 0 rebuilds fresh tables every move;
/// 1 persists everything unchanged (the default since the 094 bundle gate);
/// 2 persists but halves history and conth at the start of each move; 3
/// persists with killers cleared (killers are per-ply and refer to the
/// previous root, so they may be noise). Node-limited searches always behave
/// as 0 — bench stays exact and the tuner stays reproducible — whatever this
/// says. Same binary for every arm of the SPRT; only the flag differs.
static PERSIST_HIST: std::sync::OnceLock<u8> = std::sync::OnceLock::new();

/// Parse `--persist-hist=N` / `$CHESS_PERSIST_HIST=N`, clamped to 0..=3.
/// Parsed once per process; a match arm sets it on the engine command line.
pub fn persist_mode() -> u8 {
    *PERSIST_HIST.get_or_init(|| {
        let mut mode: u8 = std::env::var("CHESS_PERSIST_HIST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        for a in std::env::args().skip(1) {
            if let Some(v) = a.strip_prefix("--persist-hist=") {
                if let Ok(n) = v.parse() {
                    mode = n;
                }
            }
        }
        mode.min(3)
    })
}

/// Per-thread state. Nothing here is shared, so Lazy SMP needs no locking.
pub struct ThreadData {
    pub id: usize,
    killers: [[Move; 2]; MAX_PLY],
    /// `[side][from][to]`, incremented on quiet-move cutoffs.
    history: Box<[[[i32; 64]; 64]; 2]>,
    /// Continuation history: `[prev_piece][prev_to][piece][to]`, ~147k
    /// entries of gravity-bounded scores like `history`. Updated on the
    /// same cutoff/malus events, read for quiet moves in `score_moves`.
    /// Zero unless enabled (on by default; `--no-conth`).
    conth: Box<[[[[i32; 64]; 6]; 64]; 6]>,
    /// Pawn-structure correction history (`library/015`): `[side][pawn_key &
    /// 16383]`, entries in ~cp Q8. Like history, learned within one search and
    /// forgotten after — no locking, no sharing, cleared per search.
    corr: Box<[[i16; 16384]; 2]>,
    /// Capture history `[piece][to][victim]` (victim 6 = none, a quiet
    /// promotion). Zero and never touched unless `capt_hist_div > 0`.
    capth: Box<[[[i32; 7]; 64]; 6]>,
    /// Follow-up history (`conth2`), same shape as `conth`.
    conth2: Box<[[[[i32; 64]; 6]; 64]; 6]>,
    /// Expected-cut flag per ply, written by the parent before it recurses
    /// (same-ply re-entries — IID, singular, razoring — share it).
    cut: [bool; MAX_PLY],
    /// Static eval (with correction) per ply, `-INFINITY` in check. Read two
    /// plies up for the improving flag.
    ss_eval: [Score; MAX_PLY],
    pv: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
    /// See `PathInfo`. Boxed: 128 entries is 2 KB and `ThreadData` is already
    /// large enough that the search stack notices.
    path: Box<[PathInfo; MAX_PLY]>,
    /// Position keys along the current line, for repetition detection. The
    /// prefix is the game history supplied by the caller.
    keys: Vec<u64>,
    root_ply: usize,
    pub nodes: u64,
    pub sel_depth: usize,
    /// Set to record every pricing decision. `None` in every normal search, so
    /// the hot path pays one predictable not-taken branch and the bench
    /// fingerprint cannot move. See `trace.rs`.
    pub trace: Option<Box<crate::trace::Trace>>,
    /// Move mix actually made in the tree, as `[main, q] x [quiet, capture,
    /// promotion]`. Counted, not sampled: it decides how often a routing rule
    /// that reads material would have to rebuild an accumulator, and the
    /// legal-move mix is not the made-move mix. Six increments do not move the
    /// node count, so the bench fingerprint is unaffected.
    pub movemix: [u64; 6],
    /// Node utilisation (library/005 item 2): priced children searched, and
    /// how many of them raised alpha or cut off. The free precision number —
    /// regret is blind to under-pruning and this is where it shows. Two
    /// integer adds next to the trace hook, no control flow, so the bench
    /// fingerprint cannot move (same argument as `movemix` above). Main
    /// search only: quiescence has no pricing decisions.
    pub util_searched: u64,
    pub util_useful: u64,
    /// Hysteresis-rule rebuild tallies. The rule's bucket depends on the path
    /// from the root, so it has to be counted here rather than priced from a
    /// dump of edges. See `hyst.rs`.
    #[cfg(feature = "hyst")]
    pub hyst: Box<crate::hyst::Tally>,
    /// Per-zone cycle tallies. Boxed for the same reason `hyst` is: it is a
    /// measurement build's state and has no business widening `ThreadData` in
    /// a playing one. See `nodeprof.rs`.
    #[cfg(feature = "prof")]
    pub prof: Box<crate::nodeprof::Tally>,
    /// P2 probe recursion guard: refutation replies are ordinary qsearches
    /// and must not re-probe, or the probe explodes exponentially. Zero in
    /// every normal search.
    #[cfg(feature = "checkstats")]
    pub probe_guard: u32,
}

impl ThreadData {
    pub fn new(id: usize) -> ThreadData {
        ThreadData {
            id,
            killers: [[Move::NONE; 2]; MAX_PLY],
            history: Box::new([[[0; 64]; 64]; 2]),
            conth: Box::new([[[[0; 64]; 6]; 64]; 6]),
            corr: Box::new([[0; 16384]; 2]),
            capth: Box::new([[[0; 7]; 64]; 6]),
            conth2: Box::new([[[[0; 64]; 6]; 64]; 6]),
            cut: [false; MAX_PLY],
            ss_eval: [-INFINITY; MAX_PLY],
            pv: [[Move::NONE; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            path: Box::new([PathInfo::default(); MAX_PLY]),
            keys: Vec::with_capacity(512),
            root_ply: 0,
            movemix: [0; 6],
            util_searched: 0,
            util_useful: 0,
            #[cfg(feature = "hyst")]
            hyst: Box::new(crate::hyst::Tally::default()),
            #[cfg(feature = "prof")]
            prof: Box::new(crate::nodeprof::Tally::default()),
            #[cfg(feature = "checkstats")]
            probe_guard: 0,
            nodes: 0,
            sel_depth: 0,
            trace: None,
        }
    }

    pub fn clear(&mut self) {
        self.killers = [[Move::NONE; 2]; MAX_PLY];
        *self.history = [[[0; 64]; 64]; 2];
        *self.conth = [[[[0; 64]; 6]; 64]; 6];
        *self.corr = [[0; 16384]; 2];
        *self.capth = [[[0; 7]; 64]; 6];
        *self.conth2 = [[[[0; 64]; 6]; 64]; 6];
        #[cfg(feature = "checkstats")]
        {
            self.probe_guard = 0;
        }
        *self.path = [PathInfo::default(); MAX_PLY];
    }

    /// FX-1 mode 2: halve the learned move-ordering tables at the start of
    /// each move. Last move's ordering is a prior, not a verdict — halving
    /// keeps the ranking while letting this move's cutoffs overwrite it.
    /// The correction table is untouched: entries are position-keyed cp
    /// values, not move scores, so there is no reason to decay them here.
    pub fn halve_learned(&mut self) {
        for s in self.history.iter_mut() {
            for f in s.iter_mut() {
                for x in f.iter_mut() {
                    *x /= 2;
                }
            }
        }
        for p in self.conth.iter_mut() {
            for t in p.iter_mut() {
                for q in t.iter_mut() {
                    for x in q.iter_mut() {
                        *x /= 2;
                    }
                }
            }
        }
    }

    /// FX-1 mode 3: forget last move's killers while keeping the learned
    /// tables. Killers name moves that cut off at a ply of the *previous*
    /// root position; after the opponent's reply the position moved on.
    pub fn clear_killers(&mut self) {
        self.killers = [[Move::NONE; 2]; MAX_PLY];
    }
}

pub struct SearchResult {
    pub best_move: Move,
    pub score: Score,
    pub depth: u32,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

/// Called once per completed iteration so the front-end can print `info` lines.
pub type InfoFn<'a> = &'a mut dyn FnMut(&SearchResult, Duration, usize);

pub struct Searcher<'a, E: Evaluator> {
    shared: &'a Shared,
    pub td: ThreadData,
    eval: E,
    params: Params,
    start: Instant,
    deadline: Option<Duration>,
    /// Where `check_time` stops the search. Equal to `deadline` unless
    /// `tm_hard_pct > 100`.
    hard: Option<Duration>,
    node_limit: Option<u64>,
    /// The work ceiling in picoseconds, so the test is one integer compare.
    #[cfg(feature = "prof")]
    work_limit: Option<u64>,
    stopped: bool,
    /// How much of `td.nodes` has already been added to `Shared::nodes`.
    reported: u64,
}

impl<'a, E: Evaluator> Searcher<'a, E> {
    pub fn new(shared: &'a Shared, td: ThreadData, eval: E, params: Params) -> Searcher<'a, E> {
        Searcher {
            shared,
            td,
            eval,
            params,
            start: Instant::now(),
            deadline: None,
            hard: None,
            node_limit: None,
            #[cfg(feature = "prof")]
            work_limit: None,
            stopped: false,
            reported: 0,
        }
    }

    /// Iterative deepening with aspiration windows.
    ///
    /// Iterative deepening is not a compromise: the shallow iterations fill the
    /// TT and the killer tables, so depth N with the previous iterations is
    /// reached FASTER than depth N from scratch, because move ordering is that
    /// much better.
    /// The learned RFP's logit of P(this node fails low at its depth), LEDGER
    /// 119. Features and weights are `fitexport.py`'s, fitted on 493k nodes
    /// sampled at this very block (`--features nodedump`) and labelled by the
    /// node's own depth-`d` iteration of a 64k-node search. `g = eval - beta`,
    /// `>= 0` here. `stale` says the static eval came from the hash, so the
    /// evaluator's l2 buffer describes some other board: arm 2 re-evaluates.
    /// A full-window quiescence search of `root`, from its side to move, with
    /// `root`'s side as the contempt side. For `chess nodelabel`, which needs
    /// the value the search would stand on before it spends any depth.
    pub fn qsearch_value(&mut self, root: &Board) -> Score {
        eval::set_root(root.stm());
        self.stopped = false;
        self.node_limit = None;
        self.deadline = None;
        self.hard = None;
        self.td.nodes = 0;
        self.td.keys.clear();
        self.td.root_ply = 0;
        self.search(root, 0, -INFINITY, INFINITY, 0, false, true, 0, Move::NONE)
    }

    pub fn go(
        &mut self,
        root: &Board,
        history: &[u64],
        limits: &Limits,
        mut info: Option<InfoFn>,
    ) -> SearchResult {
        self.start = Instant::now();
        // Contempt is read from the ROOT side's point of view, so the search
        // has to say whose search this is. Every helper thread sets the same
        // value, which is why it can be a plain global.
        eval::set_root(root.stm());
        self.deadline = limits.allotment(root.stm().index(), &self.params);
        if self.params.tm_mat_pct > 0 && limits.movetime.is_none() && limits.movestogo.is_none() && !limits.infinite {
            let us = root.stm().index();
            if let Some(t) = limits.time[us] {
                let p = &self.params;
                let mtg = (material(root) * p.tm_mat_pct as u32 / 100).max(p.tm2_mtg_min.max(1) as u32) as u64;
                let budget = t / mtg + (limits.inc[us] * p.tm_inc_pct.max(0) as u64) / 100;
                self.deadline = Some(Duration::from_millis(budget.min(t / p.tm_cap_div.max(2) as u64).max(5)));
            }
        }
        self.hard = self.deadline;
        if self.params.tm_hard_pct > 100 && limits.movetime.is_none() {
            if let (Some(d), Some(t)) = (self.deadline, limits.time[root.stm().index()]) {
                let cap = Duration::from_millis(t / self.params.tm_cap_div.max(2) as u64);
                self.hard = Some(d.mul_f64(self.params.tm_hard_pct as f64 / 100.0).min(cap).max(d));
            }
        }
        // TM2's start limit, when it is on. It replaces the gate below and
        // sets its own hard stop.
        let mut tm2_go: Option<Duration> = None;
        if self.params.tm_mode == 1 && limits.movetime.is_none() && !limits.infinite {
            let us = root.stm().index();
            if let Some(t) = limits.time[us] {
                let p = &self.params;
                let mtg = if p.tm2_mat == 1 {
                    (material(root) * p.tm2_mat_pct.max(1) as u32 / 100).max(p.tm2_mtg_min.max(1) as u32)
                } else {
                    p.tm_moves_to_go.max(1) as u32
                } as f64;
                let target_ms = ((t as f64 + mtg * limits.inc[us] as f64) / mtg).max(5.0);
                let cap_ms = t as f64 / p.tm2_cap_div.max(2) as f64;
                let ms = |x: f64| Duration::from_secs_f64(x.max(1.0) / 1000.0);
                tm2_go = Some(ms((target_ms * p.tm2_go_pct as f64 / 100.0).min(cap_ms)));
                self.hard = Some(ms((target_ms * p.tm2_hard_pct as f64 / 100.0).min(cap_ms)));
            }
        }
        self.node_limit = limits.nodes;
        // The counter runs from zero each search, so the ceiling is absolute
        // rather than relative to whatever a previous search left behind.
        #[cfg(feature = "prof")]
        {
            self.td.prof.work_ps = 0;
            self.work_limit =
                limits.work.map(|u| (u as f64 * crate::work::NS_PER_UNIT * 1000.0) as u64);
        }
        self.stopped = false;
        self.td.nodes = 0;
        // Per-search tallies, like the node counter above: a reused
        // `ThreadData` (FX-1 persistence) must not carry last move's mix and
        // utilisation into this move's numbers. No-op on a fresh table.
        self.td.movemix = [0; 6];
        self.td.util_searched = 0;
        self.td.util_useful = 0;
        self.td.keys.clear();
        self.td.keys.extend_from_slice(history);
        self.td.root_ply = self.td.keys.len();
        self.reported = 0;
        // One age bump per search, not one per thread: under Lazy SMP every
        // helper enters `go` too, and N bumps would age out entries this
        // search is about to want.
        if self.td.id == 0 {
            self.shared.tt.new_search();
        }

        let mut best = SearchResult {
            best_move: Move::NONE,
            score: 0,
            depth: 0,
            nodes: 0,
            pv: Vec::new(),
        };

        // A legal move is guaranteed before search starts, so `bestmove` is
        // never empty even if we are stopped during the first iteration.
        let mut root_moves = MoveList::new();
        generate(root, GenType::All, &mut root_moves);
        if root_moves.is_empty() {
            return best;
        }
        best.best_move = root_moves.get(0);

        let max_depth = limits.depth.unwrap_or(MAX_PLY as u32 - 1);
        let mut prev_score = 0;
        // Easy move: how many consecutive completed iterations agree on the
        // best move with a stable score. Dormant while `easy_stable` is 0.
        let mut em_move = Move::NONE;
        let mut em_score = 0;
        let mut em_stable: u32 = 0;
        // Instability: best-move changes, halved every iteration.
        let mut changes = 0.0f64;
        let mut last_move = Move::NONE;
        // TM2: wall time of the last two completed iterations, seconds.
        let mut iter_end = 0.0f64;
        let mut dt_last = 0.0f64;

        for depth in 1..=max_depth {
            // Lazy SMP diversification: a helper skips some iterations so the
            // threads are spread across depths rather than all grinding the
            // same one. Cheap, and it is the only thing besides TT races that
            // makes the threads do different work.
            if self.skip_iteration(depth) {
                continue;
            }
            self.td.sel_depth = 0;
            let score = if depth <= self.params.asp_full_depth as u32 {
                self.negamax(root, depth as i32 * PLY, -INFINITY, INFINITY, 0, true)
            } else {
                self.aspiration(root, depth as i32 * PLY, prev_score)
            };

            if self.stopped && depth > 1 {
                break;
            }
            prev_score = score;

            let pv: Vec<Move> = (0..self.td.pv_len[0]).map(|i| self.td.pv[0][i]).collect();
            if let Some(&m) = pv.first() {
                best.best_move = m;
            }
            best.score = score;
            best.depth = depth;
            best.nodes = self.td.nodes;
            best.pv = pv;

            if let Some(f) = info.as_deref_mut() {
                f(&best, self.start.elapsed(), self.td.sel_depth);
            }

            // A forced mate is proven; searching deeper cannot improve it.
            if eval::is_mate_score(score) && depth as Score >= MATE_BOUND - score.abs() {
                break;
            }
            // Easy move: the same best move at sufficient depth with a score
            // that is not drifting trusts the move and spends less clock.
            // Only completed iterations count — a stopped one proved nothing.
            if self.params.easy_stable > 0
                && depth >= self.params.easy_depth as u32
                && best.best_move == em_move
                && (score - em_score).abs() <= self.params.easy_margin
            {
                em_stable += 1;
            } else {
                em_stable = 0;
            }
            if depth > 1 {
                changes = changes * 0.5 + (best.best_move != last_move) as u32 as f64;
            }
            last_move = best.best_move;
            let fall = if depth > 1 { (em_score - score).max(0) } else { 0 };
            em_move = best.best_move;
            em_score = score;
            if let Some(go) = tm2_go {
                // Predict the next iteration from the growth of the last two:
                // dt_next = dt_last · (dt_last / dt_prev), never shrinking.
                let now = self.start.elapsed().as_secs_f64();
                let dt_prev = dt_last;
                dt_last = now - iter_end;
                iter_end = now;
                if depth >= self.params.tm2_free_depth as u32 {
                    let pred = dt_last * (dt_last / dt_prev.max(1e-6)).max(1.0);
                    if now + pred > go.as_secs_f64() {
                        break;
                    }
                }
                continue;
            }
            // Don't start an iteration we have no realistic chance of finishing.
            if let Some(d) = self.deadline {
                let base = self.params.tm_gate_pct as f64 / 100.0;
                let mut stretch = 1.0 + changes * self.params.tm_instab as f64 / 100.0;
                stretch *= 1.0 + (fall as f64 * self.params.tm_fall as f64 / 1000.0).min(1.0);
                let gate = if self.params.easy_stable > 0 && em_stable >= self.params.easy_stable as u32 {
                    d.mul_f64(base * stretch * self.params.easy_frac as f64 / 100.0)
                } else {
                    d.mul_f64(base * stretch)
                };
                if self.start.elapsed() > gate {
                    break;
                }
            }
        }
        best
    }

    /// Re-search with a narrow window around the previous score. Most
    /// iterations land inside it, and a narrow window prunes far more.
    fn aspiration(&mut self, root: &Board, budget: i32, prev: Score) -> Score {
        let mut delta = self.params.aspiration_window;
        let mut alpha = (prev - delta).max(-INFINITY);
        let mut beta = (prev + delta).min(INFINITY);
        loop {
            let score = self.negamax(root, budget, alpha, beta, 0, true);
            if self.stopped {
                return score;
            }
            if score <= alpha {
                beta = (alpha + beta) / 2;
                alpha = (score - delta).max(-INFINITY);
            } else if score >= beta {
                beta = (score + delta).min(INFINITY);
            } else {
                return score;
            }
            delta = delta * self.params.asp_growth / 100;
        }
    }

    /// Should this thread skip iterative-deepening iteration `depth`?
    ///
    /// Only helpers ever skip — thread 0 runs every depth, because it is the
    /// one whose result is played and whose `info` lines are printed. The
    /// table is the one Stockfish used: thread `i` searches in runs of
    /// `SIZE[i]` consecutive depths and then skips `SIZE[i]`, offset by
    /// `PHASE[i]`. The point is not the schedule itself but that no two
    /// threads share one, so they arrive at each depth with different TT
    /// contents and different move orders.
    #[inline]
    fn skip_iteration(&self, depth: u32) -> bool {
        const SIZE: [u32; 20] = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4];
        const PHASE: [u32; 20] = [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6];
        if self.td.id == 0 || self.params.smp_skip == 0 {
            return false;
        }
        let i = (self.td.id - 1) % 20;
        ((depth + PHASE[i]) / SIZE[i]) % 2 != 0
    }

    #[inline]
    fn check_time(&mut self) -> bool {
        // Polling the clock is a syscall; every 2048 nodes is frequent enough
        // to respond in well under a millisecond and rare enough to be free.
        if self.td.nodes & 2047 == 0 {
            let delta = self.td.nodes - self.reported;
            self.reported = self.td.nodes;
            self.shared.nodes.fetch_add(delta, Ordering::Relaxed);
            if self.shared.stop.load(Ordering::Relaxed) {
                self.stopped = true;
            } else if let Some(d) = self.hard {
                if self.start.elapsed() >= d {
                    self.stopped = true;
                }
            }
        }
        // The node ceiling is tested every node, not every 2048. A node-limited
        // search is the tuner's unit of work, and the whole method rests on it
        // being *exactly* reproducible — a 2048-node slop window would make the
        // objective depend on where the counter happened to land.
        if let Some(max) = self.node_limit {
            if self.td.nodes >= max {
                self.stopped = true;
            }
        }
        // The work ceiling, for the same reason and on the same schedule. The
        // counter is an integer sum of integer prices, so this is exactly
        // reproducible in the way a float accumulation would not be.
        #[cfg(feature = "prof")]
        if let Some(max) = self.work_limit {
            if self.td.prof.work_ps >= max {
                self.stopped = true;
            }
        }
        self.stopped
    }

    /// Draw by threefold repetition or the fifty-move rule.
    ///
    /// A *single* repetition inside the search tree is treated as a draw. That
    /// is technically wrong (three are required) but is what every engine does:
    /// if a line repeats once, either side can repeat again, so the position is
    /// a draw by agreement in all but pathological cases — and detecting it one
    /// repetition early prunes an enormous amount of tree.
    fn is_draw(&self, b: &Board) -> bool {
        if b.halfmove() >= 100 {
            return true;
        }
        let n = self.td.keys.len();
        let limit = b.halfmove() as usize;
        let mut i = 2;
        while i <= limit && i <= n {
            if self.td.keys[n - i] == b.key() {
                return true;
            }
            i += 2; // only same-side-to-move positions can repeat
        }
        false
    }

    /// The main search below the root. Kept so callers read the intent;
    /// the work is in `search` with `in_q = false`.
    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        b: &Board,
        budget: i32,
        alpha: Score,
        beta: Score,
        ply: usize,
        is_pv: bool,
    ) -> Score {
        self.td.cut[ply] = false;
        self.search(b, budget, alpha, beta, ply, is_pv, false, 0, Move::NONE)
    }

    /// One recursive function for both searches above. `in_q` picks the branch:
    /// full moves, pricing and the hash outside quiescence; captures, stand pat
    /// and the SEE filter inside it. Same tree as the two functions this
    /// replaces — `bench` proves it, not the comments.
    ///
    /// `excluded` removes one move from the loop. Only the singular-extension
    /// verification uses it: the bound proved without a legal move is not a
    /// bound on the real position, so an excluded search neither stores to
    /// the hash nor extends further. `Move::NONE` everywhere else.
    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        b: &Board,
        mut budget: i32,
        mut alpha: Score,
        beta: Score,
        ply: usize,
        is_pv: bool,
        in_q: bool,
        qd: i32,
        excluded: Move,
    ) -> Score {
        if !in_q {
            self.td.pv_len[ply] = 0;
        }
        if ply >= MAX_PLY - 1 {
            return self.eval.evaluate(b);
        }
        if in_q && qd >= self.params.q_max_ply {
            return self.eval.evaluate(b);
        }
        self.td.sel_depth = self.td.sel_depth.max(ply);

        // Q-tier (`--dual`: cheap net in quiescence; `--qnoise`: perturbed
        // full net). Gated on one flag so the base path pays a single
        // predictable branch and stays bit-identical.
        let tier = crate::qeval::dual_enabled() || crate::qeval::qnoise_armed();
        if tier {
            self.eval.set_small(in_q);
        }

        if !in_q && ply > 0 {
            if zone!(self, Z::Draw, self.is_draw(b)) {
                return eval::draw_score(b.stm());
            }
            // Mate-distance pruning: if we already have a mate at least this
            // fast, nothing deeper can beat it.
            let a = alpha.max(eval::mated_in(ply));
            let bt = beta.min(eval::mate_in(ply + 1));
            if a >= bt {
                return a;
            }
            alpha = a;
        }

        let in_check = b.in_check();
        // Extra budget when in check: forced lines are cheap to search and
        // disastrous to judge statically. The one place a node gets money back.
        if !in_q && in_check {
            budget += self.params.check_extension * PLY / 1000;
        }

        if !in_q && budget <= 0 {
            return self.search(b, 0, alpha, beta, ply, false, true, 0, Move::NONE);
        }

        // Two ply-valued views of the budget, and they round opposite ways on
        // purpose. `depth` scales margins and indexes tables, so it rounds UP:
        // a node with a third of a ply left should be treated as shallow, not
        // as depth zero, or every margin collapses to nothing. `tt_depth` is
        // what we claim to have proved, so it rounds DOWN — never advertise
        // work that was not done. On whole-ply budgets the two coincide, which
        // is what makes the identity pricing byte-identical to the old search.
        let mut depth = (budget + PLY - 1) / PLY;
        let mut tt_depth = budget / PLY;
        let cut_node = self.td.cut[ply];

        self.td.nodes += 1;
        #[cfg(feature = "prof")]
        {
            self.td.prof.hit_node(in_q as usize);
        }
        if self.check_time() {
            return alpha;
        }

        // Measurement only: what would a hash probe in quiescence have found?
        // Counted and dropped, so the tree is bit-identical with or without
        // it. Only runs in the quiet branch (see the probe cost note that
        // used to live on the old standalone quiescence).
        #[cfg(feature = "qprobe")]
        if in_q {
            self.td.prof.qtt[0] += 1;
            let k = b.key();
            self.td.prof.q_witness(k);
            if let Some(h) = self.shared.tt.probe(b.key(), ply) {
                if h.eval != 0 {
                    self.td.prof.qtt[1] += 1;
                }
                let usable = match h.bound {
                    Bound::Exact => true,
                    Bound::Lower => h.score >= beta,
                    Bound::Upper => h.score <= alpha,
                    Bound::None => false,
                };
                if usable {
                    self.td.prof.qtt[2] += 1;
                }
            }
        }

        // ---- transposition table. The main search and quiescence share one
        // table, but a quiescence score is a different function (captures
        // only, stand pat allowed), so score cutoffs cross only between
        // quiescence nodes — marked with depth Q_DEPTH. The stored move and
        // static eval are properties of the position and cross freely, which
        // is also what unlocks TT-move ordering inside quiescence.
        let mut tt_move = Move::NONE;
        let mut tt_eval = None;
        // Saved for the singular-extension check below: the entry's own
        // score, bound and proven depth. Probe-adjusted for ply already.
        let mut tt_score = 0;
        let mut tt_bound = crate::tt::Bound::None;
        let mut tt_stored: i32 = -1;
        {
            let hit = zone!(self, Z::TtProbe, self.shared.tt.probe(b.key(), ply));
            if let Some(h) = &hit {
                tt_move = h.mv;
                tt_score = h.score;
                tt_bound = h.bound;
                tt_stored = h.depth as i32;
                // Deliberately NOT tiered: a quiescence store's eval is the
                // cheap net's, and the main search reuses it anyway — exactly
                // as the base build reuses quiescence evals. Recomputing a
                // full eval on every such hit tripled the main-search eval
                // count (18k -> 54k on bench) and ate half the win
                // (measured, nodeprof). Bounds and moves cross freely as
                // before; only the static-eval approximation is tiered, and
                // games price it.
                if h.eval != 0 {
                    tt_eval = Some(h.eval);
                }
                // Main search: quiescence bounds never satisfy a probe, at any
                // depth — they bound the wrong function.
                let deep_enough =
                    if in_q { h.depth == crate::tt::Q_DEPTH } else { h.depth != crate::tt::Q_DEPTH && h.depth as i32 >= tt_depth };
                if !is_pv && deep_enough {
                    let usable = match h.bound {
                        Bound::Exact => true,
                        Bound::Lower => h.score >= beta,
                        Bound::Upper => h.score <= alpha,
                        Bound::None => false,
                    };
                    if usable {
                        return h.score;
                    }
                }
            }
        }

        // Internal iterative reduction (dormant unless `iir_min_depth > 0`).
        // No hash move means ordering is blind here; search a ply shallower
        // and let the next visit find the move this one stores.
        if !in_q
            && ply > 0
            && excluded == Move::NONE
            && self.params.iir_min_depth > 0
            && tt_move == Move::NONE
            && depth >= self.params.iir_min_depth
            && (self.params.iir_cut_only == 0 || is_pv || cut_node)
        {
            budget -= PLY;
            depth = (budget + PLY - 1) / PLY;
            tt_depth = budget / PLY;
        }

        // What the hash stores: the evaluator's own number, before any
        // search-side adjustment (the 50-move scaling depends on the path,
        // not the key, so storing it scaled would compound on re-read). At
        // default parameters it equals the static eval exactly.
        let mut raw_eval: Score = 0;
        let static_eval = if in_check {
            #[cfg(feature = "checkstats")]
            if self.td.probe_guard == 0 {
                crate::checkstats::record(b);
            }
            -INFINITY
        } else if in_q {
            #[cfg(feature = "evalstats")]
            crate::evalstats::record(b, 0);
            #[cfg(feature = "checkstats")]
            if self.td.probe_guard == 0 {
                crate::checkstats::record(b);
            }
            // A stored static eval is the same deterministic function the
            // evaluator computes — whoever stored it (main or quiescence)
            // paid the ~910 ns already.
            let s = match tt_eval {
                Some(e) => e,
                None => zone!(self, Z::QEval, self.eval.evaluate(b)),
            };
            raw_eval = s;
            let s = self.hmc_adjust(b, s);
            // Standing pat: we are not obliged to capture, so the static eval
            // is a lower bound on what this node is worth. Store it — a later
            // quiescence node at this key with beta below it cuts off without
            // evaluating.
            if s >= beta {
                zone!(
                    self,
                    Z::TtStore,
                    self.shared.tt.store(b.key(), Move::NONE, s, raw_eval, crate::tt::Q_DEPTH, Bound::Lower, ply)
                );
                return s;
            }
            alpha = alpha.max(s);
            s
        } else {
            #[cfg(feature = "evalstats")]
            crate::evalstats::record(b, 1);
            raw_eval = match tt_eval {
                Some(e) => e,
                None => zone!(self, Z::Eval, self.eval.evaluate(b)),
            };
            self.hmc_adjust(b, raw_eval)
        };
        // The eval readout scale (TU-1 `eval_k`, default = identity): every
        // static eval the search sees is in these units — stored evals,
        // stand-pat bounds, pruning comparands — so one multiplier rescales
        // every margin at once. Mate-distance and in-check sentinels pass
        // through unchanged in value (the scale is exact at default).
        let static_eval = static_eval * self.params.eval_k / EVAL_K_UNITY;

        // Correction history (library/015), phase 1: the eval plus what this
        // pawn structure has taught us about the eval's bias. Main search
        // only, never in check (where static_eval is -INFINITY), and never
        // returned as a score — only the two pruning decisions below read it.
        // Off by default: one relaxed load, bit-identical bench.
        let corr_eval = if !in_q && !in_check && corr_enabled() {
            let idx = (b.pawn_key() & 16383) as usize;
            let e = self.td.corr[b.stm().index()][idx] as i32;
            static_eval + ((e * 100) >> 8).clamp(-256, 256)
        } else {
            static_eval
        };

        // Improving: better than our own static eval two plies ago. Written
        // for every main node that gets this far, read two plies down.
        let mut improving = false;
        let mut impr_known = false;
        if !in_q {
            self.td.ss_eval[ply] = if in_check { -INFINITY } else { corr_eval };
            impr_known = !in_check && ply >= 2 && self.td.ss_eval[ply - 2] != -INFINITY;
            improving = impr_known && corr_eval > self.td.ss_eval[ply - 2];
        }
        // The pruning eval: the corrected static eval, optionally tightened
        // by a hash bound that says the true value lies beyond it.
        let prune_eval = if self.params.tt_eval_adj > 0
            && !in_q
            && !in_check
            && tt_bound != Bound::None
            && !eval::is_mate_score(tt_score)
            && (self.params.tt_eval_adj == 2 || tt_stored != crate::tt::Q_DEPTH as i32)
            && ((tt_bound == Bound::Lower && tt_score > corr_eval)
                || (tt_bound == Bound::Upper && tt_score < corr_eval)
                || tt_bound == Bound::Exact)
        {
            tt_score
        } else {
            corr_eval
        };

        // ---- whole-node pruning, all disabled in check, in PV nodes,
        // and inside quiescence
        if !in_q && !is_pv && !in_check {
            #[cfg(feature = "nodedump")]
            if excluded == Move::NONE {
                crate::nodedump::record(b, depth, static_eval, alpha, beta, improving, ply);
            }
            // Reverse futility: if we are so far ahead that even conceding
            // `margin * depth` leaves us above beta, assume the opponent has no
            // way to claw it back at this depth.
            if depth <= self.params.rfp_max_depth
                && prune_eval
                    - self.params.rfp_margin
                        * (depth * 1000 - improving as i32 * self.params.rfp_improving)
                        / 1000
                    >= beta
                && !eval::is_mate_score(beta)
            {
                return static_eval;
            }

            // Null-move pruning. Disabled without non-pawn material, which is
            // exactly the zugzwang-prone case where "passing" is not a valid
            // lower bound on what a real move achieves.
            if depth >= self.params.nmp_min_depth
                && prune_eval >= beta
                && b.has_non_pawn_material(b.stm())
            {
                let mut r = self.params.nmp_base_reduction + depth / self.params.nmp_depth_divisor;
                if self.params.nmp_eval_div > 0 {
                    r += ((prune_eval - beta) / self.params.nmp_eval_div).min(self.params.nmp_eval_max);
                }
                self.td.cut[ply + 1] = !cut_node;
                let nb = b.make_null();
                self.td.path[ply + 1] = PathInfo { mv: Move::NONE, off_pv: self.td.path[ply].off_pv, off_us: self.td.path[ply].off_us, off_them: self.td.path[ply].off_them, off_us_n: self.td.path[ply].off_us_n, off_them_n: self.td.path[ply].off_them_n };
                self.td.keys.push(nb.key());
                let score =
                    -self.search(&nb, budget - (r + 1) * PLY, -beta, -beta + 1, ply + 1, false, false, 0, Move::NONE);
                self.td.keys.pop();
                if score >= beta {
                    // Don't return unproven mate scores from a null move.
                    return if eval::is_mate_score(score) { beta } else { score };
                }
            }

            // Razoring (SR-4, dormant unless `raz_max_depth > 0`): hopelessly
            // below alpha at low depth — ask quiescence, which does search
            // the captures that recover, and believe it only if it agrees.
            if self.params.raz_max_depth > 0
                && depth <= self.params.raz_max_depth
                && prune_eval + self.params.raz_margin * depth <= alpha
                && !eval::is_mate_score(alpha)
            {
                let v = self.search(b, 0, alpha, alpha + 1, ply, false, true, 0, Move::NONE);
                if self.stopped {
                    return alpha;
                }
                if v <= alpha {
                    return v;
                }
            }

            // ProbCut (SR-6, dormant unless `pc_min_depth > 0`): a good
            // capture that beats beta by a margin at reduced depth is taken as
            // proof the full-depth search would too.
            let pbeta = beta + self.params.pc_margin;
            if self.params.pc_min_depth > 0
                && depth >= self.params.pc_min_depth
                && excluded == Move::NONE
                && !eval::is_mate_score(beta)
                && !eval::is_mate_score(pbeta)
                && !(tt_stored != crate::tt::Q_DEPTH as i32
                    && tt_bound != Bound::None
                    && tt_stored >= depth - self.params.pc_reduction
                    && tt_score < pbeta)
            {
                let mut caps = MoveList::new();
                generate(b, GenType::Captures, &mut caps);
                self.score_moves(b, &mut caps, tt_move, ply, false);
                let child_budget = budget - self.params.pc_reduction * PLY;
                for i in 0..caps.len() {
                    let mv = caps.pick_best(i);
                    if !eval::see(b, mv, pbeta - static_eval) {
                        continue;
                    }
                    let nb = b.make_move(mv);
                    self.shared.tt.prefetch(nb.key());
                    self.td.path[ply + 1] = PathInfo {
                        mv,
                        off_pv: self.td.path[ply].off_pv + 1,
                        off_us: self.td.path[ply].off_us | ((ply % 2 == 0) as u8),
                        off_them: self.td.path[ply].off_them | ((ply % 2 == 1) as u8),
                        off_us_n: (self.td.path[ply].off_us_n + (ply % 2 == 0) as u8).min(3),
                        off_them_n: (self.td.path[ply].off_them_n + (ply % 2 == 1) as u8).min(3),
                    };
                    self.td.cut[ply + 1] = !cut_node;
                    self.td.keys.push(nb.key());
                    self.eval.push(&nb);
                    let mut v = -self.search(&nb, 0, -pbeta, -pbeta + 1, ply + 1, false, true, 0, Move::NONE);
                    if v >= pbeta && child_budget > 0 {
                        v = -self.search(&nb, child_budget, -pbeta, -pbeta + 1, ply + 1, false, false, 0, Move::NONE);
                    }
                    self.eval.pop();
                    self.td.keys.pop();
                    if self.stopped {
                        return alpha;
                    }
                    if v >= pbeta {
                        self.shared.tt.store(
                            b.key(),
                            mv,
                            v,
                            raw_eval,
                            (tt_depth - self.params.pc_reduction + 1).clamp(0, 254) as u8,
                            Bound::Lower,
                            ply,
                        );
                        return v;
                    }
                }
            }

            // Futility pruning (dormant: `fut_max_depth` defaults to 0, which
            // no main-search node reaches, so the tree is bit-identical with
            // it off). Checked last so the stronger prunes fire first.
            // PROXY-KILLED 2026-09-12 as a whole-node return-static rule:
            // fut_max_depth=1..5 reads +4.6..+22cp on the relabelled 1k-see
            // corpus @15k nodes (|t|>2.5 everywhere, mildest is m=200 at
            // depth 1: +4.57 +/- 1.81). Larger margins only asymptote to 0
            // from above — monotonically bad, no interior optimum. Mechanism:
            // a node failing low is exactly where the search must find the
            // tactic that gets back, and returning static forfeits captures
            // too. Gated behind `fut_whole_node` (default 0, never SPRT on)
            // so the skip-quiets form below can be measured pure. The
            // standard form (skip quiets, still search captures)
            // is a different mechanism and is NOT refuted by this — but it
            // needs move-loop surgery, not this gate. Do not SPRT this form.
            if self.params.fut_max_depth > 0
                && self.params.fut_whole_node > 0
                && depth <= self.params.fut_max_depth
                && corr_eval + self.params.fut_margin * depth <= alpha
                && !eval::is_mate_score(alpha)
            {
                return static_eval;
            }
        }

        // ---- internal iterative deepening: on a TT miss at sufficient
        // budget, climb a geometric ramp of budgets (1, 2, 4, ... plies,
        // capped below the full budget) instead of searching blind at full
        // width. Each step orders the next through the TT; node count is
        // exponential in budget so the short steps cost ~nothing next to
        // the deepest probe. A bound struck at any step ends the ramp: the
        // ordering so far is kept and the full search below takes over —
        // the probe score is never returned, a shallow fail-high proves
        // nothing about the full budget. The ramp recurses through this
        // same branch, but nesting self-suppresses: after the first probe
        // stores a TT move the deeper same-node calls see tt_move != NONE
        // and skip straight to their full search.
        let iid_gate = if is_pv {
            self.params.iid_min_depth > 0 && depth >= self.params.iid_min_depth
        } else {
            self.params.iid_min_nonpv > 0 && depth >= self.params.iid_min_nonpv
        };
        if !in_q && ply > 0 && tt_move == Move::NONE && iid_gate {
            // Strictly smaller, always — measured against what the probe
            // will actually hold, not what is passed. The same node re-adds
            // the check extension on entry, so `cap < budget` alone lets a
            // 1-ply reduction in check recurse forever at an equal budget.
            // This is the same strict-decrease rule the move loop enforces
            // with `.min(budget - 1)`.
            let ext = if in_check { self.params.check_extension * PLY / 1000 } else { 0 };
            let cap = budget - self.params.iid_reduction * PLY;
            if cap > 0 && cap + ext < budget {
                let mut step = PLY;
                loop {
                    let probe = if step < cap { step } else { cap };
                    let ps = self.search(b, probe, alpha, beta, ply, is_pv, false, 0, excluded);
                    if self.stopped {
                        return alpha;
                    }
                    // Never leak a probe PV: on a fail-low below, the move
                    // loop must start from empty, not from a shallow line.
                    self.td.pv_len[ply] = 0;
                    if let Some(h) = self.shared.tt.probe(b.key(), ply) {
                        tt_move = h.mv;
                    }
                    if ps <= alpha || ps >= beta || probe >= cap {
                        break;
                    }
                    step *= 2;
                }
            }
        }

        // ---- singular extension: one move far better than the rest earns
        // more budget. Dormant while `se_min_depth` is 0 (bench-exact). A
        // lower-bound hash entry proven near this depth says the hash move
        // failed high; a half-budget search *without* it that still fails
        // low by `se_margin * depth` says nothing else comes close. Excluded
        // searches never nest, and mate scores stay out on both sides.
        let mut se_ext = false;
        let mut se_neg = false;
        if !in_q
            && ply > 0
            && excluded == Move::NONE
            && self.params.se_min_depth > 0
            && depth >= self.params.se_min_depth
            && tt_move != Move::NONE
            && tt_bound == crate::tt::Bound::Lower
            && tt_stored >= depth - self.params.se_depth_slack
            && !eval::is_mate_score(tt_score)
            && !eval::is_mate_score(beta)
        {
            let sbeta = tt_score - self.params.se_margin * depth;
            let vs = self.search(b, (budget / 2).max(1), sbeta - 1, sbeta, ply, false, false, 0, tt_move);
            if self.stopped {
                return alpha;
            }
            // Never leak a probe PV: same reason as the IID ramp above.
            self.td.pv_len[ply] = 0;
            se_ext = vs < sbeta;
            if !se_ext {
                // Multi-cut: another move clears sbeta, and sbeta clears beta,
                // so two moves beat beta — the node is a cut either way.
                if self.params.se_multicut > 0 && sbeta >= beta {
                    return sbeta;
                }
                se_neg = self.params.se_neg_ext > 0 && tt_score >= beta;
            }
        }

        // ---- move loop. Quiescence looks at captures only (every move
        // when in check, or a mate would go unnoticed); the main search
        // looks at everything.
        let mut moves = MoveList::new();
        if in_q {
            zone!(
                self,
                Z::QGen,
                generate(
                    b,
                    if in_check { GenType::All } else { GenType::Captures },
                    &mut moves,
                )
            );
            if in_check && moves.is_empty() {
                return eval::mated_in(ply);
            }
            if !in_check && qd == 0 && self.params.q_checks > 0 {
                let mut all = MoveList::new();
                generate(b, GenType::All, &mut all);
                for i in 0..all.len() {
                    let mv = all.get(i);
                    if mv.is_quiet() && eval::see(b, mv, 0) && b.make_move(mv).in_check() {
                        moves.push(mv);
                    }
                }
            }
            zone!(self, Z::QOrder, self.score_moves(b, &mut moves, tt_move, ply, in_check));
        } else {
            zone!(self, Z::Gen, generate(b, GenType::All, &mut moves));
            if moves.is_empty() {
                return if in_check { eval::mated_in(ply) } else { eval::draw_score(b.stm()) };
            }
            zone!(self, Z::Order, self.score_moves(b, &mut moves, tt_move, ply, in_check));
        }
        // LC-2: quiescence value of every non-hash child, from our side.
        let mut lc2_vals: Vec<(Move, Score)> = Vec::new();
        if !in_q
            && self.params.lc2_min_depth > 0
            && depth >= self.params.lc2_min_depth
            && !in_check
            && excluded == Move::NONE
            && alpha > -MATE_BOUND
            && alpha < MATE_BOUND
        {
            let w = self.params.lc2_win;
            let (lo, hi) = (alpha - w, alpha + 1);
            lc2_vals.reserve(moves.len());
            for i in 0..moves.len() {
                let mv = moves.get(i);
                if mv == tt_move {
                    continue;
                }
                let nb = b.make_move(mv);
                self.td.path[ply + 1] = PathInfo { mv, ..self.td.path[ply] };
                self.td.keys.push(nb.key());
                self.eval.push(&nb);
                let v = -self.search(&nb, 0, -hi, -lo, ply + 1, false, true, 0, Move::NONE);
                self.eval.pop();
                self.td.keys.pop();
                if self.stopped {
                    return alpha;
                }
                lc2_vals.push((mv, v.clamp(lo, hi)));
                if self.params.lc2_order > 0 && v + self.params.lc2_hang < static_eval.min(hi) {
                    moves.set_score(i, moves.score(i) - (1 << 23));
                }
            }
        }

        if in_q {
            // Quiet branch: captures that lose on the spot or cannot reach
            // alpha never make it past the filter; the rest get a full
            // window, one ply deeper into quiescence.
            let q_entry_alpha = alpha;
            let mut best = static_eval;
            let mut best_move = Move::NONE;
            for i in 0..moves.len() {
                let mv = moves.pick_best(i);

                if !in_check {
                    // Delta pruning: if capturing the target outright still leaves
                    // us far below alpha, the whole branch is hopeless. Then SEE,
                    // which is the expensive half — both are inside one zone
                    // because they are one decision.
                    let keep = zone!(self, Z::QSee, {
                        let victim = b.piece_at(mv.to());
                        let gain = if victim.is_some() {
                            eval::SEE_VALUE[victim.piece_type().index()]
                        } else {
                            eval::SEE_VALUE[0]
                        };
                        if static_eval + gain + self.params.delta_margin < alpha && !mv.is_promotion() {
                            false
                        } else {
                            // Skip captures that lose material outright.
                            eval::see(b, mv, self.params.qsee_thresh)
                        }
                    });
                    if !keep {
                        continue;
                    }
                }

                self.td.movemix[3 + if mv.is_promotion() { 2 } else if mv.is_capture() { 1 } else { 0 }] += 1;
                let nb = zone!(self, Z::QMake, b.make_move(mv));
                // The main loop writes a full `PathInfo` on descent; quiescence
                // never did, leaving `path[ply+1].mv` stale from an earlier
                // main-search node at the same ply. Anything reading the route
                // (continuation history today) would then index on a move from
                // a different position. One move store per q-node fixes the
                // invariant "path is written on the way down" everywhere; the
                // off-PV fields at q plies stay zero and nobody reads them.
                self.td.path[ply + 1].mv = mv;
                #[cfg(feature = "movedump")]
                crate::movedump::record(b, &nb, mv, 1);
                #[cfg(feature = "hyst")]
                self.td.hyst.record(b, &nb, ply, 1);
                // Quiescence makes 10.5% of all made moves but is 35% of all
                // accumulator work (LEDGER 038), so it needs the incremental path
                // more than the main search does, not less.
                zone!(self, Z::QPush, self.eval.push(&nb));
                let score = -self.search(&nb, 0, -beta, -alpha, ply + 1, false, true, qd + 1, Move::NONE);
                zone!(self, Z::QPop, self.eval.pop());
                if self.stopped {
                    return alpha;
                }
                if score > best {
                    best = score;
                    best_move = mv;
                    if score > alpha {
                        alpha = score;
                        if score >= beta {
                            break;
                        }
                    }
                }
            }
            if !in_check && static_eval != -INFINITY && best < beta {
                zone!(self, Z::QObserve, self.eval.observe(b, static_eval, best, true));
            }
            // Share with later quiescence nodes at this key (see the probe
            // above): same restricted function, so the bound is valid there
            // and nowhere else — the Q_DEPTH marker keeps main-search probes
            // off it. In-check nodes store a 0 eval sentinel, mirroring the
            // main search: -INFINITY is not an eval.
            let q_bound = if best >= beta {
                Bound::Lower
            } else if best > q_entry_alpha {
                Bound::Exact
            } else {
                Bound::Upper
            };
            zone!(
                self,
                Z::TtStore,
                self.shared.tt.store(
                    b.key(),
                    best_move,
                    best,
                    if in_check { 0 } else { raw_eval },
                    crate::tt::Q_DEPTH,
                    q_bound,
                    ply,
                )
            );
            // P2 probe: would a quiet check refute this return? Upper bound
            // on what generating quiet checks in quiescence can buy. Runs at
            // nodes that searched captures (stand-pat failed high returns
            // above, so they are out of scope — checks only matter when the
            // static eval did not already fail high).
            #[cfg(feature = "checkstats")]
            if !in_check {
                self.qrefute_probe(b, best, beta, alpha, ply, qd);
            }
            return best;
        }

        let orig_alpha = alpha;
        let mut best_score = -INFINITY;
        let mut best_move = Move::NONE;
        // A heap allocation per interior node that tries a quiet move, and it
        // looks like an obvious thing to put on the stack. Measured: a fixed
        // `[Move; 64]` is **+1.2% nps, n=3, inside the run-to-run spread** —
        // the allocator caches this size and the list is short. Left as a Vec
        // because the array version needs a cap, and a cap is a behaviour
        // change (the history penalty stops accumulating past it) for no
        // measured gain. See LEDGER 041.
        let mut quiets_tried: Vec<Move> = Vec::new();
        let ch_on = self.params.capt_hist_div > 0;
        let mut capts_tried: Vec<Move> = Vec::new();
        let tt_capture = tt_move != Move::NONE && tt_move.is_capture();
        let use_conth_price = self.params.pricing.feat.c_conth != 0 && conth_enabled();
        let n = moves.len();

        let price = self.params.pricing;
        let full = budget - price.fare;
        if let Some(t) = self.td.trace.as_mut() {
            t.agg.interior += 1;
            // Iterative deepening re-enters the root once per iteration and
            // once per aspiration window. The aggregate should accumulate over
            // all of them; the per-event list should show the last one, or it
            // is unreadable.
            if ply == 0 {
                t.events.clear();
            }
        }

        for i in 0..n {
            let mv = moves.pick_best(i);
            let quiet = mv.is_quiet();

            // Singular verification searches one move down.
            if mv == excluded {
                continue;
            }

            // Futility, skip-quiets form (shares `fut_max_depth`/`fut_margin`
            // with the proxy-killed whole-node gate above, which additionally
            // needs `fut_whole_node` — 0 disables both, so this is
            // bench-exact by default). A quiet that cannot reach
            // alpha at this depth is not searched, but every capture still
            // is: the tactic that gets back survives. Never skips the first
            // move (something must set best_score) and never at PV nodes,
            // in check, or against mate scores.
            if i > 0
                && quiet
                && !is_pv
                && !in_check
                && !in_q
                && self.params.fut_max_depth > 0
                && depth <= self.params.fut_max_depth
                && static_eval + self.params.fut_margin * depth <= alpha
                && !eval::is_mate_score(alpha)
                && best_score > -MATE_BOUND
            {
                continue;
            }

            // ---- price the move, before paying to make it
            let (mut ev_gap, mut ev_hist) = (0, 0);
            let cost = if i == 0 {
                // The main line steps nowhere, so it pays only the fare.
                0
            } else {
                // How far below the running best is this move predicted to
                // land? Meaningless in check (there is no static eval to build
                // on) and meaningless against a mate score, so the term drops
                // out in both cases rather than being clamped to nonsense.
                let (c, g, h) = zone!(self, Z::Price, {
                    let gap = if in_check
                        || eval::is_mate_score(best_score)
                        || best_score < -MATE_BOUND
                    {
                        0
                    } else if self.params.lc2_price > 0 && !lc2_vals.is_empty() {
                        match lc2_vals.iter().find(|e| e.0 == mv) {
                            Some(&(_, v)) => best_score - v,
                            None => best_score - static_eval - eval::move_gain(b, mv),
                        }
                    } else {
                        best_score - static_eval - eval::move_gain(b, mv)
                    };
                    let hist = if quiet {
                        self.td.history[b.stm().index()][mv.from().index()][mv.to().index()]
                    } else {
                        0
                    };
                    // One context, handed to every feature. Gated features
                    // (SEE, gives-check) do not run while their coefficient is
                    // zero, so the declared-but-unproven surface costs one
                    // predictable not-taken branch each.
                    let f = Feats::extract(&price.feat, &Ctx {
                        b, mv, rank: i, depth, ply, gap, hist, is_pv, in_check,
                        full, prev: self.td.path[ply],
                        improving, impr_known, cut_node, tt_capture,
                        conth: if quiet && use_conth_price { self.conth_bonus(b, mv, ply) } else { 0 },
                    });
                    (price.price(&f, i, depth, gap, hist, is_pv, full), gap, hist)
                });
                ev_gap = g;
                ev_hist = h;
                c
            };
            // The 2-D potential table (zero unless --pot was passed, so this
            // is bit-identical by default). Paid on top of the price, never
            // instead of the fare the main line pays.
            let cost = cost + pot_term(ply, i, &self.td.path[ply]);
            // Strictly less than the parent's, always. `floor` is allowed to
            // go negative so the tuner can buy extensions, and a price below
            // `-fare` would otherwise hand a child more money than its parent
            // had — a search that never terminates. `MAX_PLY` would eventually
            // catch it, but only after the move was already lost on time.
            let child = (full - cost).min(budget - 1);
            // Singular move: the verification above proved nothing else comes
            // close, so spend an extra ply here. Clamped into the same strict
            // decrease — an extension that could hand back the full budget
            // is how searches stop terminating.
            let child = if se_ext && mv == tt_move {
                (child + PLY).min(budget - 1)
            } else if se_neg && mv == tt_move {
                child - self.params.se_neg_ext * PLY / 1000
            } else {
                child
            };
            let child = if self.params.recap_ext > 0
                && mv.is_capture()
                && self.td.path[ply].mv != Move::NONE
                && self.td.path[ply].mv.is_capture()
                && self.td.path[ply].mv.to() == mv.to()
            {
                (child + self.params.recap_ext * PLY / 1000).min(budget - 1)
            } else {
                child
            };

            // Priced out of the search entirely. This is late move pruning,
            // except the threshold is a price rather than a move index — and it
            // is the same ceiling that produces reductions, pushed one step
            // further.
            if child < price.skip_below && !is_pv && !in_check && quiet && best_score > -MATE_BOUND {
                if let Some(t) = self.td.trace.as_mut() {
                    t.child(crate::trace::Event {
                        ply: ply as u8, rank: i as u16, mv, parent_budget: budget,
                        cost, child_budget: child, gap: ev_gap, hist: ev_hist, is_pv,
                        priced_out: true, researched: false, raised_alpha: false,
                        cutoff: false, dropped_to_q: false, nodes: 0, score: 0,
                    });
                }
                continue;
            }

            let nodes_before = self.td.nodes;
            let alpha_before = alpha;
            let mut researched = false;
            let mut research_nodes = 0u64;

            self.td.movemix[if mv.is_promotion() { 2 } else if mv.is_capture() { 1 } else { 0 }] += 1;
            let nb = zone!(self, Z::Make, b.make_move(mv));
            #[cfg(feature = "movedump")]
            crate::movedump::record(b, &nb, mv, 0);
            #[cfg(feature = "hyst")]
            self.td.hyst.record(b, &nb, ply, 0);
            zone!(self, Z::Prefetch, self.shared.tt.prefetch(nb.key()));
            self.td.path[ply + 1] = PathInfo {
                mv,
                off_pv: self.td.path[ply].off_pv + (i > 0) as u32,
                // Even ply from the root is our move, odd ply theirs. Sticky:
                // once either side has left its believed-best, the subtree
                // stays in the off square of the 2x2.
                off_us: self.td.path[ply].off_us | ((i > 0 && ply % 2 == 0) as u8),
                off_them: self.td.path[ply].off_them | ((i > 0 && ply % 2 == 1) as u8),
                off_us_n: (self.td.path[ply].off_us_n + (i > 0 && ply % 2 == 0) as u8).min(3),
                off_them_n: (self.td.path[ply].off_them_n + (i > 0 && ply % 2 == 1) as u8).min(3),
            };
            self.td.keys.push(nb.key());
            zone!(self, Z::Push, self.eval.push(&nb));

            // Expected node types, Stockfish's convention: the first child of
            // a PV node is PV; the first child of a cut node is an all node
            // and vice versa; a reduced null-window probe expects to cut.
            let mut score;
            if i == 0 {
                self.td.cut[ply + 1] = !is_pv && !cut_node;
                score = -self.search(&nb, child, -beta, -alpha, ply + 1, is_pv, false, 0, Move::NONE);
            } else {
                // Search at the price we set; if it beats alpha anyway the price
                // was wrong, so buy the full search back.
                self.td.cut[ply + 1] = cost > 0 || !cut_node;
                score = -self.search(&nb, child, -alpha - 1, -alpha, ply + 1, false, false, 0, Move::NONE);
                if score > alpha && cost > 0 {
                    // The price was wrong and the search just proved it. Pay
                    // again at full budget. This is the free precision signal.
                    researched = true;
                    let n0 = self.td.nodes;
                    self.td.cut[ply + 1] = !cut_node;
                    score = -self.search(&nb, full, -alpha - 1, -alpha, ply + 1, false, false, 0, Move::NONE);
                    research_nodes = self.td.nodes - n0;
                }
                if score > alpha && score < beta {
                    self.td.cut[ply + 1] = false;
                    score = -self.search(&nb, full, -beta, -alpha, ply + 1, is_pv, false, 0, Move::NONE);
                }
            }

            zone!(self, Z::Pop, self.eval.pop());
            self.td.keys.pop();

            if self.stopped {
                return alpha;
            }

            let mut did_cutoff = false;
            if score > best_score {
                best_score = score;
                best_move = mv;
                if score > alpha {
                    alpha = score;
                    self.update_pv(ply, mv);
                    if score >= beta {
                        if quiet {
                            zone!(
                                self,
                                Z::Hist,
                                self.update_quiet_heuristics(b, mv, depth, ply, &quiets_tried)
                            );
                        }
                        if ch_on {
                            self.update_capture_history(b, mv, depth, &capts_tried);
                        }
                        did_cutoff = true;
                    }
                }
            }

            let spent = self.td.nodes - nodes_before;
            if let Some(t) = self.td.trace.as_mut() {
                let e = crate::trace::Event {
                    ply: ply as u8, rank: i as u16, mv, parent_budget: budget,
                    cost, child_budget: child, gap: ev_gap, hist: ev_hist, is_pv,
                    priced_out: false, researched, raised_alpha: score > alpha_before,
                    cutoff: did_cutoff, dropped_to_q: child <= 0, nodes: spent, score,
                };
                t.agg.research_nodes += research_nodes;
                if i == 0 { t.main_line(e) } else { t.child(e) }
            }
            // Same event, always counted: rank 0 pays only the fare, priced-out
            // children never reach here (they `continue` above), so this is
            // exactly trace's `priced - priced_out` denominator with the
            // `raised_alpha + cutoffs` numerator. This block is main-search
            // only — the quiescence branch returned further up.
            if i > 0 {
                self.td.util_searched += 1;
                if score > alpha_before || did_cutoff {
                    self.td.util_useful += 1;
                }
            }

            if did_cutoff {
                break;
            }
            if quiet {
                quiets_tried.push(mv);
            } else if ch_on {
                capts_tried.push(mv);
            }
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > orig_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        // An excluded (singular-verification) search bounds the wrong
        // position — one legal move down — so it stores nothing. The probe
        // above still reads: entries stored with the move available stay
        // valid bounds for the search without it.
        if excluded == Move::NONE {
            zone!(
                self,
                Z::TtStore,
                self.shared.tt.store(
                    b.key(),
                    best_move,
                    best_score,
                    if in_check { 0 } else { raw_eval },
                    tt_depth.clamp(0, 255) as u8,
                    bound,
                    ply,
                )
            );
        }
        // Runtime adaptation: only when we have an exact value (window not clipped),
        // otherwise best_score is a bound, not a value.
        if !in_check && bound == Bound::Exact {
            zone!(self, Z::Observe, self.eval.observe(b, static_eval, best_score, false));
        }
        // Correction-history update (library/015): credit the residual between
        // what the search found and what the eval said to this pawn structure.
        // Main search only, never in check. A qsearch stand-pat IS the eval,
        // so updating from it would teach the table to agree with itself.
        // Mate scores carry no residual information. The TT keeps the raw
        // eval; the correction applies at read time, so old entries stay valid.
        // `corr_bound_aware = 1` skips the updates whose sign the bound cannot
        // vouch for: a fail-high below the eval, or a fail-low above it. A
        // fail-soft bound is only a bound, and without this an all-node's
        // upper bound (often far below the truth) teaches the table that the
        // eval is too optimistic.
        let corr_ok = self.params.corr_bound_aware == 0
            || !((bound == Bound::Lower && best_score <= static_eval)
                || (bound == Bound::Upper && best_score >= static_eval));
        if corr_enabled() && corr_ok && excluded == Move::NONE && !in_check && !in_q && !eval::is_mate_score(best_score) {
            let idx = (b.pawn_key() & 16383) as usize;
            let e = &mut self.td.corr[b.stm().index()][idx];
            // Deeper nodes earn more say; one update stays a few percent of
            // the entry range so single nodes cannot pin it. Gravity bounds
            // entries near +/-512 units (~+/-200 cp readout) the way history
            // saturates near +/-16384.
            let w = depth.clamp(1, 8);
            let bonus = ((best_score - static_eval).clamp(-512, 512) * w) / 64;
            let cur = *e as i32;
            *e = (cur + bonus - cur * bonus.abs() / 512).clamp(-2048, 2048) as i16;
        }
        best_score
    }

    // Quiet search used to be a second function here. It is now the `in_q`
    // branch of `search` above — same moves, same filter, same counters.

    /// P2 probe (behind `--features checkstats`, never in a playing build).
    ///
    /// Of the qsearch nodes that searched captures without failing high, how
    /// many would a quiet check refute — i.e. a checking quiet whose full
    /// qsearch reply beats `best`? That is the upper bound on what Q2 (one ply
    /// of quiet checks) can buy. Unpruned by SEE on purpose: Q2 would prune,
    /// so this overestimates it, which is what an upper bound is for.
    ///
    /// Recursion-guarded by `probe_guard`: the replies are ordinary qsearches
    /// and re-probing inside them would blow bench up exponentially.
    #[cfg(feature = "checkstats")]
    fn qrefute_probe(&mut self, b: &Board, best: Score, beta: Score, alpha: Score, ply: usize, qd: i32) {
        if self.td.probe_guard > 0 {
            return;
        }
        let mut moves = MoveList::new();
        generate(b, GenType::All, &mut moves);
        // Availability first (cheap: make + check test), replies after — one
        // Vec, probe-only, never in the hot path.
        let mut checks: Vec<(Move, bool)> = Vec::new();
        for i in 0..moves.len() {
            let mv = moves.get(i);
            if !mv.is_quiet() {
                continue;
            }
            if b.make_move(mv).in_check() {
                checks.push((mv, eval::see(b, mv, 0)));
            }
        }
        crate::checkstats::record_probe(qd, !checks.is_empty());
        if checks.is_empty() {
            return;
        }
        self.td.probe_guard += 1;
        let mut top = best;
        let mut aborted = false;
        for (mv, sok) in &checks {
            let nb = b.make_move(*mv);
            // Full-window reply, mirroring how a generated check would be
            // searched if Q2 put it in the move list.
            let s = -self.search(&nb, 0, -beta, -alpha, ply + 1, false, true, qd + 1, Move::NONE);
            if self.stopped {
                aborted = true;
                break;
            }
            crate::checkstats::record_reply(*sok);
            if s > top {
                top = s;
            }
        }
        self.td.probe_guard -= 1;
        if !aborted {
            crate::checkstats::record_refuted(qd, top > best);
        }
    }

    /// Move ordering. Getting this right is worth more than any pruning
    /// heuristic: alpha-beta visits `b^(d/2)` nodes with perfect ordering and
    /// `b^d` with none, so ordering is the difference between depth 8 and 16.
    fn score_moves(&self, b: &Board, moves: &mut MoveList, tt_move: Move, ply: usize, in_check: bool) {
        const TT_BONUS: i32 = 1 << 24;
        const CAPTURE_BONUS: i32 = 1 << 22;
        const KILLER_BONUS: i32 = 1 << 21;
        // O1: exchanges that lose material sort here — below killers, above
        // history. MVV-LVA alone puts a losing QxP above every quiet.
        const BAD_CAPTURE_BONUS: i32 = 1 << 20;
        // Q1 (`qevade_see`): quiet evasions that hold material sort here —
        // above history, below losing captures. Losing evasions sort at the
        // bottom, below all of history.
        const EVADE_OK_BONUS: i32 = 1 << 19;
        const EVADE_BAD_BONUS: i32 = -(1 << 19);
        let qevade = self.params.qevade_see > 0 && in_check;
        let use_conth = conth_enabled();
        let ch_div = self.params.capt_hist_div;
        let use_conth2 = self.params.conth2 > 0;

        for i in 0..moves.len() {
            let mv = moves.get(i);
            let s = if mv == tt_move {
                TT_BONUS
            } else if mv.is_capture() || mv.is_promotion() {
                // MVV-LVA: most valuable victim, least valuable attacker.
                // Index 6 of SEE_VALUE is zero — the "no victim" slot, which a
                // quiet promotion needs since its target square is empty.
                let victim = if mv.is_ep() {
                    PieceType::Pawn.index()
                } else {
                    let p = b.piece_at(mv.to());
                    if p.is_some() { p.piece_type().index() } else { 6 }
                };
                let attacker = b.piece_at(mv.from()).piece_type().index();
                let promo = if mv.is_promotion() {
                    eval::SEE_VALUE[mv.promo_piece().index()]
                } else {
                    0
                };
                let ch = if ch_div > 0 { self.td.capth[attacker][mv.to().index()][victim] / ch_div } else { 0 };
                CAPTURE_BONUS + eval::SEE_VALUE[victim] * 16 - eval::SEE_VALUE[attacker] + promo + ch
                    + if eval::see(b, mv, 0) { 0 } else { BAD_CAPTURE_BONUS - CAPTURE_BONUS }
            } else if mv == self.td.killers[ply][0] {
                KILLER_BONUS + 1
            } else if mv == self.td.killers[ply][1] {
                KILLER_BONUS
            } else if qevade {
                if eval::see(b, mv, 0) { EVADE_OK_BONUS } else { EVADE_BAD_BONUS }
            } else {
                let h = self.td.history[b.stm().index()][mv.from().index()][mv.to().index()];
                h + if use_conth { self.conth_bonus(b, mv, ply) } else { 0 }
                    + if use_conth2 { self.conth2_bonus(b, mv, ply) } else { 0 }
            };
            moves.set_score(i, s);
        }
    }

    /// Continuation bonus for a quiet move: what replies to the previous
    /// move scored here before. Zero unless enabled. Missing pieces (never
    /// at a real node — the previous move's target always holds our piece —
    /// but cheap to guard) read as no information, never as a crash.
    #[inline]
    fn conth_bonus(&self, b: &Board, mv: Move, ply: usize) -> i32 {
        let prev = self.td.path[ply].mv;
        if prev == Move::NONE {
            return 0;
        }
        let (pp, pc) = (b.piece_at(prev.to()), b.piece_at(mv.from()));
        if pp.is_none() || pc.is_none() {
            return 0;
        }
        self.td.conth[pp.piece_type().index()][prev.to().index()][pc.piece_type().index()][mv.to().index()]
    }

    /// Our previous move (two plies up) and the piece that made it, if that
    /// piece is still ours on its target square.
    #[inline]
    fn followup_key(&self, b: &Board, ply: usize) -> Option<(usize, usize)> {
        if ply < 1 {
            return None;
        }
        let m2 = self.td.path[ply - 1].mv;
        if m2 == Move::NONE {
            return None;
        }
        let p = b.piece_at(m2.to());
        if p.is_none() || p.color() != b.stm() {
            return None;
        }
        Some((p.piece_type().index(), m2.to().index()))
    }

    #[inline]
    fn conth2_bonus(&self, b: &Board, mv: Move, ply: usize) -> i32 {
        match (self.followup_key(b, ply), b.piece_at(mv.from())) {
            (Some((pp, pt)), pc) if pc.is_some() => self.td.conth2[pp][pt][pc.piece_type().index()][mv.to().index()],
            _ => 0,
        }
    }

    /// A quiet move that caused a cutoff is likely to cause one again — in a
    /// sibling node (killers) or anywhere with the same from/to (history).
    fn update_quiet_heuristics(
        &mut self,
        b: &Board,
        mv: Move,
        depth: i32,
        ply: usize,
        tried: &[Move],
    ) {
        if self.td.killers[ply][0] != mv {
            self.td.killers[ply][1] = self.td.killers[ply][0];
            self.td.killers[ply][0] = mv;
        }
        let side = b.stm().index();
        let bonus = (depth * depth).min(self.params.hist_cap);
        let h = &mut self.td.history[side][mv.from().index()][mv.to().index()];
        // Gravity: entries saturate toward ±`hist_grav` instead of
        // overflowing, so old information decays instead of dominating forever.
        *h += bonus - *h * bonus / self.params.hist_grav;
        // Penalise the quiets that were tried and failed, or history would only
        // ever record which moves are common, not which are good.
        for &q in tried {
            let h = &mut self.td.history[side][q.from().index()][q.to().index()];
            *h += -bonus - *h * bonus / self.params.hist_grav;
        }
        if self.params.conth2 > 0 {
            if let Some((pp, pt)) = self.followup_key(b, ply) {
                let g = self.params.hist_grav;
                let pc = b.piece_at(mv.from());
                if pc.is_some() {
                    let e = &mut self.td.conth2[pp][pt][pc.piece_type().index()][mv.to().index()];
                    *e += bonus - *e * bonus / g;
                }
                for &q in tried {
                    let qpc = b.piece_at(q.from());
                    if qpc.is_some() {
                        let e = &mut self.td.conth2[pp][pt][qpc.piece_type().index()][q.to().index()];
                        *e += -bonus - *e * bonus / g;
                    }
                }
            }
        }
        // Continuation history learns on the same events, keyed by the reply
        // pair. Skipped unless enabled (one relaxed load above).
        if conth_enabled() {
            let prev = self.td.path[ply].mv;
            if prev != Move::NONE {
                let (pp, pc) = (b.piece_at(prev.to()), b.piece_at(mv.from()));
                if pp.is_some() && pc.is_some() {
                    let e = &mut self.td.conth[pp.piece_type().index()][prev.to().index()][pc.piece_type().index()][mv.to().index()];
                    *e += bonus - *e * bonus / self.params.hist_grav;
                    for &q in tried {
                        let qpc = b.piece_at(q.from());
                        if qpc.is_some() {
                            let e = &mut self.td.conth[pp.piece_type().index()][prev.to().index()][qpc.piece_type().index()][q.to().index()];
                            *e += -bonus - *e * bonus / self.params.hist_grav;
                        }
                    }
                }
            }
        }
    }

    /// SR-15a: scale an eval toward zero as the 50-move counter climbs.
    /// Identity while `hmc_scale` is 0.
    #[inline]
    fn hmc_adjust(&self, b: &Board, e: Score) -> Score {
        let k = self.params.hmc_scale;
        if k == 0 || eval::is_mate_score(e) {
            return e;
        }
        e * (k - (b.halfmove() as i32).min(k)) / k
    }

    /// Capture-history index: `[moving piece][to][victim]`, victim 6 for a
    /// promotion onto an empty square.
    #[inline]
    fn capth_idx(b: &Board, mv: Move) -> (usize, usize, usize) {
        let victim = if mv.is_ep() {
            PieceType::Pawn.index()
        } else {
            let p = b.piece_at(mv.to());
            if p.is_some() { p.piece_type().index() } else { 6 }
        };
        (b.piece_at(mv.from()).piece_type().index(), mv.to().index(), victim)
    }

    /// Capture history learns on every cutoff: the move that cut (if a
    /// capture) gains, the captures tried before it lose. Same bonus shape and
    /// gravity as quiet history.
    fn update_capture_history(&mut self, b: &Board, mv: Move, depth: i32, tried: &[Move]) {
        let bonus = (depth * depth).min(self.params.hist_cap);
        let g = self.params.hist_grav;
        if !mv.is_quiet() {
            let (p, t, v) = Self::capth_idx(b, mv);
            let e = &mut self.td.capth[p][t][v];
            *e += bonus - *e * bonus / g;
        }
        for &q in tried {
            let (p, t, v) = Self::capth_idx(b, q);
            let e = &mut self.td.capth[p][t][v];
            *e += -bonus - *e * bonus / g;
        }
    }

    fn update_pv(&mut self, ply: usize, mv: Move) {
        self.td.pv[ply][0] = mv;
        let child_len = self.td.pv_len[ply + 1];
        for i in 0..child_len {
            self.td.pv[ply][i + 1] = self.td.pv[ply + 1][i];
        }
        self.td.pv_len[ply] = child_len + 1;
    }
}

/// Format a score the way UCI wants it: centipawns, or mate distance in moves.
pub fn score_to_uci(s: Score) -> String {
    if eval::is_mate_score(s) {
        let plies = eval::MATE - s.abs();
        let moves = (plies + 1) / 2;
        format!("mate {}", if s > 0 { moves } else { -moves })
    } else {
        format!("cp {s}")
    }
}

// ------------------------------------------------------------------ Lazy SMP

/// Run `threads` searchers over one `Shared`, and return the main thread's
/// result.
///
/// This is Lazy SMP, and the name is honest: there is no work decomposition,
/// no split points and no synchronisation beyond the transposition table. Every
/// thread searches the *whole* tree from the root. The speed-up comes from
/// them not searching it the same way — one thread's TT entry is another
/// thread's instant cutoff, so the tree each one has left to walk keeps
/// shrinking under it.
///
/// Two reasons this is the right design here rather than a fallback. The
/// alternative (Young Brothers Wait, DTS) needs a work queue over subtrees, and
/// a subtree's *value* depends on the alpha-beta window it inherits, which
/// changes while it is queued — so the shared state is far larger than a TT and
/// far harder to keep correct. And this engine's TT is already lockless and
/// race-tolerant (Hyatt XOR checksum, relaxed atomics), which is exactly and
/// only what Lazy SMP requires.
///
/// The races are real and are load-bearing, not tolerated: threads read each
/// other's half-written tables on purpose. What the checksum guarantees is the
/// only thing that has to hold — a probe never returns a *different position's*
/// entry. A stale or slightly-wrong score for the right position is already
/// something the search handles, because that is what an aged TT is.
///
/// Determinism is gone, and it cannot be bought back: two runs will visit
/// different nodes and can return different moves. So `bench`, the tuner and
/// any node-limited search stay single-threaded — a node limit is the tuner's
/// unit of work and it has to be exactly reproducible.
/// Make `td` ready for one more `go` with `threads` searchers under persist
/// `mode` (see `persist_mode`): grow one entry per thread, then apply the
/// mode's move-start rule. Mode 0 drops everything and rebuilds fresh —
/// today's behaviour, and what node-limited searches always get. Modes 1..3
/// reuse the tables the last search learned. Pure over the vec, so tests can
/// check every mode without running a search.
pub fn prepare_td(td: &mut Vec<ThreadData>, threads: usize, mode: u8) {
    let n = threads.max(1);
    match mode {
        0 => {
            td.clear();
            for i in 0..n {
                td.push(ThreadData::new(i));
            }
        }
        1 => {
            while td.len() < n {
                let id = td.len();
                td.push(ThreadData::new(id));
            }
        }
        2 => {
            while td.len() < n {
                let id = td.len();
                td.push(ThreadData::new(id));
            }
            for t in td.iter_mut().take(n) {
                t.halve_learned();
            }
        }
        _ => {
            while td.len() < n {
                let id = td.len();
                td.push(ThreadData::new(id));
            }
            for t in td.iter_mut().take(n) {
                t.clear_killers();
            }
        }
    }
}

pub fn go_parallel<E, F>(
    shared: &Shared,
    root: &Board,
    history: &[u64],
    limits: &Limits,
    params: Params,
    threads: usize,
    make_eval: F,
    info: Option<InfoFn>,
    td: &mut Vec<ThreadData>,
) -> SearchResult
where
    E: Evaluator,
    F: Fn() -> E + Sync,
{
    let threads = threads.max(1);
    // `prepare_td` is the caller's job (the UCI engine applies the persist
    // mode there); this only guarantees the indexing below is in range, so a
    // caller that manages its own tables cannot go out of bounds.
    while td.len() < threads {
        let id = td.len();
        td.push(ThreadData::new(id));
    }
    shared.stop.store(false, Ordering::Relaxed);
    shared.nodes.store(0, Ordering::Relaxed);

    if threads == 1 {
        let owned = std::mem::replace(&mut td[0], ThreadData::new(0));
        let mut s = Searcher::new(shared, owned, make_eval(), params);
        let r = s.go(root, history, limits, info);
        td[0] = s.td;
        return r;
    }

    // Helpers get no depth ceiling. If the caller asked for `go depth 20` the
    // helpers should keep filling the TT right up to the moment thread 0
    // finishes depth 20, not stop alongside it and leave the last iteration
    // unassisted. They keep the deadline as a backstop so a wedged main thread
    // cannot leave them running.
    let helper_limits = Limits { depth: None, nodes: None, ..*limits };
    let root = *root;

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads - 1);
        // Disjoint halves, so the helpers can borrow their tables while the
        // main thread uses its own — no locking, same deal as before, except
        // the tables now survive the search instead of being dropped with it.
        let (first, rest) = td.split_at_mut(1);
        for slot in rest.iter_mut().take(threads - 1) {
            let eval = make_eval();
            let hl = helper_limits;
            handles.push(scope.spawn(move || {
                let owned = std::mem::replace(slot, ThreadData::new(0));
                let mut s = Searcher::new(shared, owned, eval, params);
                s.go(&root, history, &hl, None);
                *slot = s.td;
            }));
        }

        let owned = std::mem::replace(&mut first[0], ThreadData::new(0));
        let mut main = Searcher::new(shared, owned, make_eval(), params);
        let res = main.go(&root, history, limits, info);
        first[0] = main.td;

        // The helpers have no reason of their own to stop.
        shared.stop.store(true, Ordering::Relaxed);
        for h in handles {
            let _ = h.join();
        }
        res
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::START_FEN;
    use crate::eval::PstEval;

    fn dirty_td() -> Vec<ThreadData> {
        let mut td = vec![ThreadData::new(0)];
        let t = &mut td[0];
        t.history[0][0][0] = 100;
        t.history[1][63][63] = -7;
        t.conth[0][0][0][0] = 50;
        t.corr[0][0] = 12;
        t.killers[0] = [crate::chess_move::Move::NONE; 2];
        td
    }

    fn table_sums(t: &ThreadData) -> (i64, i64, i64) {
        let h = t.history.iter().flatten().flatten().map(|&x| x as i64).sum();
        let c = t
            .conth
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .map(|&x| x as i64)
            .sum();
        let k = t.corr.iter().flatten().map(|&x| x as i64).sum();
        (h, c, k)
    }

    #[test]
    fn prepare_td_mode0_rebuilds_fresh() {
        let mut td = dirty_td();
        prepare_td(&mut td, 2, 0);
        assert_eq!(td.len(), 2);
        for (i, t) in td.iter().enumerate() {
            assert_eq!(t.id, i);
            assert_eq!(table_sums(t), (0, 0, 0), "mode 0 must drop everything");
        }
    }

    #[test]
    fn prepare_td_mode1_persists_everything() {
        let mut td = dirty_td();
        prepare_td(&mut td, 1, 1);
        assert_eq!(table_sums(&td[0]), (93, 50, 12));
        // Grows with fresh ids when the thread count rises.
        prepare_td(&mut td, 3, 1);
        assert_eq!(td.len(), 3);
        assert_eq!(td[2].id, 2);
        assert_eq!(table_sums(&td[0]), (93, 50, 12));
    }

    #[test]
    fn prepare_td_mode2_halves_ordering_tables_only() {
        let mut td = dirty_td();
        prepare_td(&mut td, 1, 2);
        // Rust integer division truncates toward zero: -7/2 == -3.
        assert_eq!(table_sums(&td[0]), (100 / 2 - 7 / 2, 25, 12));
        assert_eq!(td[0].history[0][0][0], 50);
        assert_eq!(td[0].history[1][63][63], -3);
    }

    #[test]
    fn prepare_td_mode3_clears_killers_keeps_tables() {
        use crate::chess_move::{Move, MoveList};
        use crate::movegen::{generate, GenType};
        let mut td = dirty_td();
        // Plant a real killer so the test checks removal, not just NONEs.
        let b = Board::from_fen(START_FEN).unwrap();
        let mut list = MoveList::new();
        generate(&b, GenType::All, &mut list);
        let m = list.get(0);
        assert_ne!(m, Move::NONE);
        td[0].killers[3] = [m, m];
        prepare_td(&mut td, 1, 3);
        assert!(td[0].killers.iter().flatten().all(|&k| k == Move::NONE));
        assert_eq!(table_sums(&td[0]), (93, 50, 12));
    }

    #[test]
    fn search_learns_into_reused_tables() {
        crate::init();
        let b = Board::from_fen(START_FEN).unwrap();
        let shared = Shared::new(16);
        let limits = Limits { depth: Some(4), ..Default::default() };
        let mut td = Vec::new();
        prepare_td(&mut td, 1, 0);
        let r = go_parallel(&shared, &b, &[], &limits, Params::default(), 1, || PstEval, None, &mut td);
        assert_ne!(r.best_move, crate::chess_move::Move::NONE);
        // A real search cuts off, and cutoffs write history — so a nonzero
        // table afterwards proves the search learned into the reused vec.
        let learned: i64 = td[0]
            .history
            .iter()
            .flatten()
            .flatten()
            .map(|&x| x.abs() as i64)
            .sum();
        assert!(learned > 0, "depth-4 startpos search should learn history");
        // The second search runs on the same tables without complaint, and
        // mode 0 afterwards drops all of it again.
        let r2 = go_parallel(&shared, &b, &[], &limits, Params::default(), 1, || PstEval, None, &mut td);
        assert_ne!(r2.best_move, crate::chess_move::Move::NONE);
        prepare_td(&mut td, 1, 0);
        assert_eq!(table_sums(&td[0]), (0, 0, 0));
    }
}
