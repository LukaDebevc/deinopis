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

params! {
#[derive(Clone, Copy, Debug)]
pub struct Params {
        > pricing: Pricing;

        /// Null move pruning is off below this depth.
        pub nmp_min_depth = 3, 1, 12;
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
        /// no quiet move reaches alpha at this depth. Captures can, so the
        /// node returns the static eval as a fail-soft upper bound rather
        /// than pruning outright — safe at a non-PV node, where any value
        /// below alpha keeps the bound logic intact; the only cost of being
        /// wrong is strength, not correctness.
        pub fut_max_depth = 0, 0, 16;
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
        /// Delta pruning slack in quiescence, centipawns.
        pub delta_margin = 100, 0, 1200;
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
    c_see: Gap, Lin, Gated, 4000, -4000, 20000,
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
    fn allotment(&self, stm: usize) -> Option<Duration> {
        if self.infinite {
            return None;
        }
        if let Some(mt) = self.movetime {
            return Some(Duration::from_millis(mt.saturating_sub(20)));
        }
        let t = self.time[stm]?;
        let inc = self.inc[stm];
        let mtg = self.movestogo.unwrap_or(30).max(1) as u64;
        let budget = t / mtg + inc * 3 / 4;
        Some(Duration::from_millis(budget.min(t / 3).max(5)))
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

/// Per-thread state. Nothing here is shared, so Lazy SMP needs no locking.
pub struct ThreadData {
    pub id: usize,
    killers: [[Move; 2]; MAX_PLY],
    /// `[side][from][to]`, incremented on quiet-move cutoffs.
    history: Box<[[[i32; 64]; 64]; 2]>,
    /// Pawn-structure correction history (`library/015`): `[side][pawn_key &
    /// 16383]`, entries in ~cp Q8. Like history, learned within one search and
    /// forgotten after — no locking, no sharing, cleared per search.
    corr: Box<[[i16; 16384]; 2]>,
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
            corr: Box::new([[0; 16384]; 2]),
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
        *self.corr = [[0; 16384]; 2];
        #[cfg(feature = "checkstats")]
        {
            self.probe_guard = 0;
        }
        *self.path = [PathInfo::default(); MAX_PLY];
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
        self.deadline = limits.allotment(root.stm().index());
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

        for depth in 1..=max_depth {
            // Lazy SMP diversification: a helper skips some iterations so the
            // threads are spread across depths rather than all grinding the
            // same one. Cheap, and it is the only thing besides TT races that
            // makes the threads do different work.
            if self.skip_iteration(depth) {
                continue;
            }
            self.td.sel_depth = 0;
            let score = if depth <= 4 {
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
            // Don't start an iteration we have no realistic chance of finishing.
            if let Some(d) = self.deadline {
                if self.start.elapsed() > d.mul_f64(0.5) {
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
            delta += delta / 2;
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
            } else if let Some(d) = self.deadline {
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
        self.search(b, budget, alpha, beta, ply, is_pv, false, 0)
    }

    /// One recursive function for both searches above. `in_q` picks the branch:
    /// full moves, pricing and the hash outside quiescence; captures, stand pat
    /// and the SEE filter inside it. Same tree as the two functions this
    /// replaces — `bench` proves it, not the comments.
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
            return self.search(b, 0, alpha, beta, ply, false, true, 0);
        }

        // Two ply-valued views of the budget, and they round opposite ways on
        // purpose. `depth` scales margins and indexes tables, so it rounds UP:
        // a node with a third of a ply left should be treated as shallow, not
        // as depth zero, or every margin collapses to nothing. `tt_depth` is
        // what we claim to have proved, so it rounds DOWN — never advertise
        // work that was not done. On whole-ply budgets the two coincide, which
        // is what makes the identity pricing byte-identical to the old search.
        let depth = (budget + PLY - 1) / PLY;
        let tt_depth = budget / PLY;

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
        {
            let hit = zone!(self, Z::TtProbe, self.shared.tt.probe(b.key(), ply));
            if let Some(h) = &hit {
                tt_move = h.mv;
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
            // Standing pat: we are not obliged to capture, so the static eval
            // is a lower bound on what this node is worth. Store it — a later
            // quiescence node at this key with beta below it cuts off without
            // evaluating.
            if s >= beta {
                zone!(
                    self,
                    Z::TtStore,
                    self.shared.tt.store(b.key(), Move::NONE, s, s, crate::tt::Q_DEPTH, Bound::Lower, ply)
                );
                return s;
            }
            alpha = alpha.max(s);
            s
        } else {
            #[cfg(feature = "evalstats")]
            crate::evalstats::record(b, 1);
            match tt_eval {
                Some(e) => e,
                None => zone!(self, Z::Eval, self.eval.evaluate(b)),
            }
        };

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

        // ---- whole-node pruning, all disabled in check, in PV nodes,
        // and inside quiescence
        if !in_q && !is_pv && !in_check {
            // Reverse futility: if we are so far ahead that even conceding
            // `margin * depth` leaves us above beta, assume the opponent has no
            // way to claw it back at this depth.
            if depth <= self.params.rfp_max_depth
                && corr_eval - self.params.rfp_margin * depth >= beta
                && !eval::is_mate_score(beta)
            {
                return static_eval;
            }

            // Null-move pruning. Disabled without non-pawn material, which is
            // exactly the zugzwang-prone case where "passing" is not a valid
            // lower bound on what a real move achieves.
            if depth >= self.params.nmp_min_depth
                && corr_eval >= beta
                && b.has_non_pawn_material(b.stm())
            {
                let r = self.params.nmp_base_reduction + depth / self.params.nmp_depth_divisor;
                let nb = b.make_null();
                self.td.path[ply + 1] = PathInfo { mv: Move::NONE, off_pv: self.td.path[ply].off_pv, off_us: self.td.path[ply].off_us, off_them: self.td.path[ply].off_them, off_us_n: self.td.path[ply].off_us_n, off_them_n: self.td.path[ply].off_them_n };
                self.td.keys.push(nb.key());
                let score =
                    -self.search(&nb, budget - (r + 1) * PLY, -beta, -beta + 1, ply + 1, false, false, 0);
                self.td.keys.pop();
                if score >= beta {
                    // Don't return unproven mate scores from a null move.
                    return if eval::is_mate_score(score) { beta } else { score };
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
            // too. The standard form (skip quiets, still search captures)
            // is a different mechanism and is NOT refuted by this — but it
            // needs move-loop surgery, not this gate. Do not SPRT this form.
            if self.params.fut_max_depth > 0
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
                    let ps = self.search(b, probe, alpha, beta, ply, is_pv, false, 0);
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
            zone!(self, Z::QOrder, self.score_moves(b, &mut moves, tt_move, ply));
        } else {
            zone!(self, Z::Gen, generate(b, GenType::All, &mut moves));
            if moves.is_empty() {
                return if in_check { eval::mated_in(ply) } else { eval::draw_score(b.stm()) };
            }
            zone!(self, Z::Order, self.score_moves(b, &mut moves, tt_move, ply));
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
                            eval::see(b, mv, 0)
                        }
                    });
                    if !keep {
                        continue;
                    }
                }

                self.td.movemix[3 + if mv.is_promotion() { 2 } else if mv.is_capture() { 1 } else { 0 }] += 1;
                let nb = zone!(self, Z::QMake, b.make_move(mv));
                #[cfg(feature = "movedump")]
                crate::movedump::record(b, &nb, mv, 1);
                #[cfg(feature = "hyst")]
                self.td.hyst.record(b, &nb, ply, 1);
                // Quiescence makes 10.5% of all made moves but is 35% of all
                // accumulator work (LEDGER 038), so it needs the incremental path
                // more than the main search does, not less.
                zone!(self, Z::QPush, self.eval.push(&nb));
                let score = -self.search(&nb, 0, -beta, -alpha, ply + 1, false, true, qd + 1);
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
                    if in_check { 0 } else { static_eval },
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

            let mut score;
            if i == 0 {
                score = -self.search(&nb, child, -beta, -alpha, ply + 1, is_pv, false, 0);
            } else {
                // Search at the price we set; if it beats alpha anyway the price
                // was wrong, so buy the full search back.
                score = -self.search(&nb, child, -alpha - 1, -alpha, ply + 1, false, false, 0);
                if score > alpha && cost > 0 {
                    // The price was wrong and the search just proved it. Pay
                    // again at full budget. This is the free precision signal.
                    researched = true;
                    let n0 = self.td.nodes;
                    score = -self.search(&nb, full, -alpha - 1, -alpha, ply + 1, false, false, 0);
                    research_nodes = self.td.nodes - n0;
                }
                if score > alpha && score < beta {
                    score = -self.search(&nb, full, -beta, -alpha, ply + 1, is_pv, false, 0);
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
            }
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > orig_alpha {
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
                best_score,
                if in_check { 0 } else { static_eval },
                tt_depth.clamp(0, 255) as u8,
                bound,
                ply,
            )
        );
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
        if corr_enabled() && !in_check && !in_q && !eval::is_mate_score(best_score) {
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
            let s = -self.search(&nb, 0, -beta, -alpha, ply + 1, false, true, qd + 1);
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
    fn score_moves(&self, b: &Board, moves: &mut MoveList, tt_move: Move, ply: usize) {
        const TT_BONUS: i32 = 1 << 24;
        const CAPTURE_BONUS: i32 = 1 << 22;
        const KILLER_BONUS: i32 = 1 << 21;
        // O1: exchanges that lose material sort here — below killers, above
        // history. MVV-LVA alone puts a losing QxP above every quiet.
        const BAD_CAPTURE_BONUS: i32 = 1 << 20;

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
                CAPTURE_BONUS + eval::SEE_VALUE[victim] * 16 - eval::SEE_VALUE[attacker] + promo
                    + if eval::see(b, mv, 0) { 0 } else { BAD_CAPTURE_BONUS - CAPTURE_BONUS }
            } else if mv == self.td.killers[ply][0] {
                KILLER_BONUS + 1
            } else if mv == self.td.killers[ply][1] {
                KILLER_BONUS
            } else {
                self.td.history[b.stm().index()][mv.from().index()][mv.to().index()]
            };
            moves.set_score(i, s);
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
        let bonus = (depth * depth).min(1200);
        let h = &mut self.td.history[side][mv.from().index()][mv.to().index()];
        // Gravity: entries saturate toward +/-16384 instead of overflowing, so
        // old information decays instead of dominating forever.
        *h += bonus - *h * bonus / 16384;
        // Penalise the quiets that were tried and failed, or history would only
        // ever record which moves are common, not which are good.
        for &q in tried {
            let h = &mut self.td.history[side][q.from().index()][q.to().index()];
            *h += -bonus - *h * bonus / 16384;
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
pub fn go_parallel<E, F>(
    shared: &Shared,
    root: &Board,
    history: &[u64],
    limits: &Limits,
    params: Params,
    threads: usize,
    make_eval: F,
    info: Option<InfoFn>,
) -> SearchResult
where
    E: Evaluator,
    F: Fn() -> E + Sync,
{
    let threads = threads.max(1);
    shared.stop.store(false, Ordering::Relaxed);
    shared.nodes.store(0, Ordering::Relaxed);

    if threads == 1 {
        let mut s = Searcher::new(shared, ThreadData::new(0), make_eval(), params);
        return s.go(root, history, limits, info);
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
        for id in 1..threads {
            let eval = make_eval();
            let hl = helper_limits;
            handles.push(scope.spawn(move || {
                let mut s = Searcher::new(shared, ThreadData::new(id), eval, params);
                s.go(&root, history, &hl, None);
            }));
        }

        let mut main = Searcher::new(shared, ThreadData::new(0), make_eval(), params);
        let res = main.go(&root, history, limits, info);

        // The helpers have no reason of their own to stop.
        shared.stop.store(true, Ordering::Relaxed);
        for h in handles {
            let _ = h.join();
        }
        res
    })
}
