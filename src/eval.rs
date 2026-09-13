//! Static evaluation.
//!
//! Tapered material + piece-square tables (PeSTO's tuned values). This is a
//! placeholder with a deliberate purpose: it is the *control arm*. Everything
//! we do later — NNUE, the learned search-control work — has to be measured
//! against a search that is otherwise identical, and that requires a baseline
//! eval that is fast, deterministic, and good enough to play real chess.
//!
//! The seam for NNUE is `Evaluator`: swap the implementation, keep the search.
//! Note NNUE will need incremental state that a `&Board` alone cannot provide,
//! which is why the trait takes `&mut self` — the accumulator stack lives in
//! the evaluator, not in `Board`.

use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::board::Board;
use crate::types::{Color, PieceType, Square};

/// Score in centipawns, from the perspective of the side to move.
pub type Score = i32;

pub const MATE: Score = 32_000;
/// Any score at least this large is a forced mate, and its distance is
/// `MATE - score` plies. The search uses that to prefer shorter mates.
pub const MATE_BOUND: Score = MATE - 256;
pub const DRAW: Score = 0;
pub const INFINITY: Score = 32_001;

// ---------------------------------------------------------------- contempt
//
// What a draw is WORTH to the side that started the search, on the 0..1 win
// scale: 0.5 is a normal engine, 0.0 makes a draw count as a loss. It is one
// number because a WDL net makes it one number — the eval already estimates
// P(win), P(draw) and P(loss) separately, so changing what a draw is worth is
// a change to how the three are collapsed into a score, not a retrain and not
// a bonus bolted onto a scalar. `E = W + v*D` instead of `W + D/2`.
//
// It is ROOT-RELATIVE, and that is not a detail. A negamax score is read from
// the side to move at that node, so a draw penalty that ignored whose search
// this is would make both sides draw-averse — a position would then evaluate
// differently depending on the parity of the ply it was reached at, and the
// search would be inconsistent with itself. Zero-sum fixes it: if a draw is
// worth `v` to us it is worth `1 - v` to the opponent, so `logit(E)` still
// negates across a ply exactly as it must.
//
// Asymmetric contempt does NOT live here as a state-dependent `v`: blending
// `behind -> ahead` in the root's log-odds was tried and is globally
// incoherent — in drawish territory a 2%-win position reads +418cp while its
// 18%-win neighbour reads −418, because min-loss and max-win are different
// objectives and no blend of them compares across the boundary (probed
// numerically 2026-09-10, then deleted). The coherent form is one fixed
// objective plus a bounded additive nudge: `contempt_bonus` below.

static DRAW_VALUE: AtomicU32 = AtomicU32::new(0x3f00_0000); // 0.5f32
static ROOT_IS_WHITE: AtomicBool = AtomicBool::new(true);

/// What a draw is worth to the root side, 0..1. 0.5 is a normal engine.
#[inline]
pub fn draw_value() -> f32 {
    f32::from_bits(DRAW_VALUE.load(Ordering::Relaxed))
}

pub fn set_draw_value(v: f32) {
    DRAW_VALUE.store(v.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

/// Called once at the top of every search. Lazy SMP threads all share a root,
/// so this is global rather than per-thread.
pub fn set_root(c: Color) {
    ROOT_IS_WHITE.store(c == Color::White, Ordering::Relaxed);
}

/// Whether `stm` is the side that started this search.
#[inline]
pub fn stm_is_root(stm: Color) -> bool {
    (stm == Color::White) == ROOT_IS_WHITE.load(Ordering::Relaxed)
}

/// What a draw is worth to `stm`, 0..1 — `v` for the root side, `1 - v` for
/// the opponent.
#[inline]
pub fn draw_value_for(stm: Color) -> f32 {
    let v = draw_value();
    if stm_is_root(stm) {
        v
    } else {
        1.0 - v
    }
}

// Asymmetric contempt as an additive bonus, in centipawns:
//
//   bonus = b * weight * tanh(a * gap)
//
// with `gap = l_logit - w_logit` (side-to-move-relative) in nats and `weight`
// the draw probability `D` — or identically 1 when `flat` is on, so the
// bonus keys on the win-vs-loss difference alone. Behind (gap > 0) the draw
// up to +b; ahead it looks bad by down to −b; parity and decisive positions
// get nothing. It is stm-relative, so it negates across a ply by
// construction — no root tracking needed: at the opponent's node the same
// formula reads their gap, and their draw-seeking is our draw-avoidance.
//
// `D` is the normalised probability, not the raw draw logit: raw logits are
// defined up to a shared additive shift, so the raw value is meaningless and
// only the normalised mass (or its own logit) can carry weight. `flat`
// drops the draw term (weight identically 1): the bonus then keys on the
// win-vs-loss difference alone, decisive positions included. Flat hits
// harder for the same `b` — no `D` attenuation — so it wants a smaller `b`.
static CONTEMPT_A: AtomicU32 = AtomicU32::new(0x3f80_0000); // 1.0f32
static CONTEMPT_B: AtomicU32 = AtomicU32::new(0x0000_0000); // 0.0f32
static CONTEMPT_FLAT: AtomicU32 = AtomicU32::new(0x0000_0000); // 0.0f32

/// Steepness of the contempt bonus in the logit gap. 1.0 saturates around
/// 2-3:1 odds; 0.0 would pay the middle everywhere (pointless — leave 1.0
/// until games say otherwise).
#[inline]
pub fn contempt_a() -> f32 {
    f32::from_bits(CONTEMPT_A.load(Ordering::Relaxed))
}

/// Scale of the contempt bonus in centipawns. 0.0 is off — the default, and
/// what keeps the default build bit-identical.
#[inline]
pub fn contempt_b() -> f32 {
    f32::from_bits(CONTEMPT_B.load(Ordering::Relaxed))
}

pub fn set_contempt_a(v: f32) {
    CONTEMPT_A.store(v.max(0.0).to_bits(), Ordering::Relaxed);
}

pub fn set_contempt_b(v: f32) {
    CONTEMPT_B.store(v.max(0.0).to_bits(), Ordering::Relaxed);
}

/// Flat weighting (weight 1 instead of `D`) when nonzero. Off by default.
#[inline]
pub fn contempt_flat() -> bool {
    CONTEMPT_FLAT.load(Ordering::Relaxed) != 0
}

pub fn set_contempt_flat(on: bool) {
    CONTEMPT_FLAT.store((if on { 1.0f32 } else { 0.0f32 }).to_bits(), Ordering::Relaxed);
}

/// Pure so it can be unit-tested; `cp_from_logits_asym` must agree with it.
#[inline]
pub fn contempt_bonus(d_prob: f64, gap: f64, a: f32, b: f32) -> f64 {
    if b == 0.0 {
        return 0.0;
    }
    b as f64 * d_prob * (a as f64 * gap).tanh()
}

/// `--draw-value <0..100>` or `$CHESS_DRAW_VALUE`, as a percent. Read once at
/// startup so it reaches `serve` and `play` as well as the UCI loop; over UCI
/// the `DrawValue` option sets the same thing and wins, being later.
pub fn init_from_args() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    if let Some(v) = flag("--draw-value")
        .or_else(|| std::env::var("CHESS_DRAW_VALUE").ok())
        .and_then(|v| v.parse::<f32>().ok())
    {
        set_draw_value(v / 100.0);
    }
    // `--contempt-a` is a percent (100 = 1.0, the default steepness),
    // `--contempt-b` is centipawns (0 = off). Same read-once-at-startup path
    // as `--draw-value`, so engine-arm strings in match scripts just work.
    if let Some(v) = flag("--contempt-a")
        .or_else(|| std::env::var("CHESS_CONTEMPT_A").ok())
        .and_then(|v| v.parse::<f32>().ok())
    {
        set_contempt_a(v / 100.0);
    }
    if let Some(v) = flag("--contempt-b")
        .or_else(|| std::env::var("CHESS_CONTEMPT_B").ok())
        .and_then(|v| v.parse::<f32>().ok())
    {
        set_contempt_b(v);
    }
    // `--contempt-flat` takes no value: presence turns it on. Same for
    // `$CHESS_CONTEMPT_FLAT=1` and the UCI `ContemptFlat` spin.
    if args.iter().any(|a| a == "--contempt-flat")
        || std::env::var("CHESS_CONTEMPT_FLAT").is_ok_and(|v| v != "0")
    {
        set_contempt_flat(true);
    }
}

/// The score of a position that IS drawn, from `stm`'s point of view.
///
/// `K * logit(v)` so that a terminal draw and a position the eval merely
/// believes is drawish are priced on the same scale, clamped well short of
/// `MATE_BOUND` so it can never be mistaken for a mate score. At the default
/// 0.5 this is exactly 0 and every node behaves as it did before.
#[inline]
pub fn draw_score(stm: Color) -> Score {
    let v = draw_value_for(stm);
    if v == 0.5 {
        return DRAW;
    }
    let v = v.clamp(1e-4, 1.0 - 1e-4);
    ((288.5 * (v / (1.0 - v)).ln()) as Score).clamp(-2500, 2500)
}

#[inline(always)]
pub fn mate_in(ply: usize) -> Score {
    MATE - ply as Score
}

#[inline(always)]
pub fn mated_in(ply: usize) -> Score {
    -MATE + ply as Score
}

#[inline(always)]
pub fn is_mate_score(s: Score) -> bool {
    s.abs() >= MATE_BOUND
}

// Midgame/endgame piece values, PeSTO.
const MG_VALUE: [Score; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUE: [Score; 6] = [94, 281, 297, 512, 936, 0];

/// Phase weight per piece; the sum over all pieces at the start is 24.
const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];
const MAX_PHASE: i32 = 24;

/// Simple piece values for SEE and move ordering. Deliberately NOT the tapered
/// values above: SEE needs a single fixed scale so exchange sequences are
/// order-independent.
pub const SEE_VALUE: [Score; 7] = [100, 320, 330, 500, 900, 20_000, 0];

// PSTs are written in reading order (a8 first) so they can be eyeballed against
// a board. A white piece on `sq` therefore indexes `sq ^ 56`.
#[rustfmt::skip]
const MG_PST: [[Score; 64]; 6] = [
    [ // pawn
          0,   0,   0,   0,   0,   0,   0,   0,
         98, 134,  61,  95,  68, 126,  34, -11,
         -6,   7,  26,  31,  65,  56,  25, -20,
        -14,  13,   6,  21,  23,  12,  17, -23,
        -27,  -2,  -5,  12,  17,   6,  10, -25,
        -26,  -4,  -4, -10,   3,   3,  33, -12,
        -35,  -1, -20, -23, -15,  24,  38, -22,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    [ // knight
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    [ // bishop
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,   5,  19,  50,  37,  37,   7,  -2,
         -6,  13,  13,  26,  34,  12,  10,   4,
          0,  15,  15,  15,  14,  27,  18,  10,
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    [ // rook
         32,  42,  32,  51,  63,   9,  31,  43,
         27,  32,  58,  62,  80,  67,  26,  44,
         -5,  19,  26,  36,  17,  45,  61,  16,
        -24, -11,   7,  26,  24,  35,  -8, -20,
        -36, -26, -12,  -1,   9,  -7,   6, -23,
        -45, -25, -16, -17,   3,   0,  -5, -33,
        -44, -16, -20,  -9,  -1,  11,  -6, -71,
        -19, -13,   1,  17,  16,   7, -37, -26,
    ],
    [ // queen
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50,
    ],
    [ // king
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ],
];

#[rustfmt::skip]
const EG_PST: [[Score; 64]; 6] = [
    [ // pawn
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    [ // knight
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    [ // bishop
        -14, -21, -11,  -8,  -7,  -9, -17, -24,
         -8,  -4,   7, -12,  -3, -13,  -4, -14,
          2,  -8,   0,  -1,  -2,   6,   0,   4,
         -3,   9,  12,   9,  14,  10,   3,   2,
         -6,   3,  13,  19,   7,  10,  -3,  -9,
        -12,  -3,   8,  10,  13,   3,  -7, -15,
        -14, -18,  -7,  -1,   4,  -9, -15, -27,
        -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    ],
    [ // rook
         13,  10,  18,  15,  12,  12,   8,   5,
         11,  13,  13,  11,  -3,   3,   8,   3,
          7,   7,   7,   5,   4,  -3,  -5,  -3,
          4,   3,  13,   1,   2,   1,  -1,   2,
          3,   5,   8,   4,  -5,  -6,  -8, -11,
         -4,   0,  -5,  -1,  -7, -12,  -8, -16,
         -6,  -6,   0,   2,  -9,  -9, -11,  -3,
         -9,   2,   3,  -1,  -5, -13,   4, -20,
    ],
    [ // queen
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ],
    [ // king
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
];

/// The eval seam. Implementations own whatever incremental state they need.
pub trait Evaluator: Send {
    /// Score from the perspective of the side to move.
    fn evaluate(&mut self, b: &Board) -> Score;
    /// Called when the search descends/ascends, so an incremental evaluator can
    /// keep an accumulator stack. The PST evaluator ignores it.
    fn push(&mut self, _b: &Board) {}
    fn pop(&mut self) {}
    /// Q-tier switch, set once per search node. `true` means this node is in
    /// quiescence, where the engine may apply its q-tier policy: the cheap net
    /// (`--dual`), deterministic noise on the full net (`--qnoise`), or both
    /// off. Default no-op, so every existing evaluator is full-fidelity
    /// everywhere and the search is bit-identical unless an evaluator
    /// overrides it. Implementations must make it a plain store.
    fn set_small(&mut self, _small: bool) {}
    /// Runtime learning: predicted vs actual (backed) score at this node.
    /// Default no-op; adaptive eval overrides it.
    fn observe(&mut self, _b: &Board, _predicted: Score, _actual: Score, _is_q: bool) {}
}

#[derive(Default, Clone)]
pub struct PstEval;

impl Evaluator for PstEval {
    fn evaluate(&mut self, b: &Board) -> Score {
        evaluate_pst(b)
    }
}

pub fn evaluate_pst(b: &Board) -> Score {
    let mut mg = [0 as Score; 2];
    let mut eg = [0 as Score; 2];
    let mut phase = 0i32;

    for c in Color::ALL {
        for pt in PieceType::ALL {
            let pi = pt.index();
            for sq in b.colored(c, pt) {
                // White reads the table mirrored, since the tables are written
                // a8-first from White's point of view.
                let idx = if c == Color::White {
                    sq.flip_rank().index()
                } else {
                    sq.index()
                };
                mg[c.index()] += MG_VALUE[pi] + MG_PST[pi][idx];
                eg[c.index()] += EG_VALUE[pi] + EG_PST[pi][idx];
                phase += PHASE_INC[pi];
            }
        }
    }

    let us = b.stm().index();
    let them = us ^ 1;
    let mg_score = mg[us] - mg[them];
    let eg_score = eg[us] - eg[them];
    // Phase can exceed 24 after multiple promotions; clamp so the taper stays
    // an interpolation rather than an extrapolation.
    let p = phase.min(MAX_PHASE);
    (mg_score * p + eg_score * (MAX_PHASE - p)) / MAX_PHASE
}

/// Static exchange evaluation: the material outcome of the capture sequence on
/// `mv`'s target square, assuming both sides continue capturing with their
/// least valuable attacker. Used to skip losing captures in quiescence and to
/// order captures — the main reason `attackers_to` takes an explicit occupancy.
pub fn see(b: &Board, mv: crate::chess_move::Move, threshold: Score) -> bool {
    use crate::chess_move::Move;
    let _ = Move::NONE;
    let from = mv.from();
    let to = mv.to();

    let mut next_victim = if mv.is_promotion() {
        mv.promo_piece()
    } else {
        b.piece_at(from).piece_type()
    };

    // Value swing of the move itself.
    let captured = b.piece_at(to);
    let mut balance = if captured.is_some() {
        SEE_VALUE[captured.piece_type().index()]
    } else if mv.is_ep() {
        SEE_VALUE[PieceType::Pawn.index()]
    } else {
        0
    };
    if mv.is_promotion() {
        balance += SEE_VALUE[mv.promo_piece().index()] - SEE_VALUE[PieceType::Pawn.index()];
    }
    balance -= threshold;
    if balance < 0 {
        return false;
    }
    // Even losing the moving piece outright still clears the threshold.
    balance -= SEE_VALUE[next_victim.index()];
    if balance >= 0 {
        return true;
    }

    let mut occ = b.occupied() - crate::bitboard::Bitboard::from_square(from)
        | crate::bitboard::Bitboard::from_square(to);
    if mv.is_ep() {
        occ.clear(Square::new(to.file(), from.rank()));
    }
    let mut attackers = b.attackers_to(to, occ) & occ;
    let mut stm = b.stm().flip();

    loop {
        let our_attackers = attackers & b.colors(stm);
        if our_attackers.is_empty() {
            break;
        }
        // Always recapture with the least valuable attacker.
        let mut found = None;
        for pt in PieceType::ALL {
            let set = our_attackers & b.pieces(pt);
            if set.any() {
                found = Some((pt, set.lsb()));
                break;
            }
        }
        let (pt, sq) = match found {
            Some(x) => x,
            None => break,
        };
        next_victim = pt;
        occ.clear(sq);

        // Removing an attacker can reveal a slider behind it.
        if matches!(pt, PieceType::Pawn | PieceType::Bishop | PieceType::Queen) {
            attackers |= crate::attacks::bishop_attacks(to, occ)
                & (b.pieces(PieceType::Bishop) | b.pieces(PieceType::Queen));
        }
        if matches!(pt, PieceType::Rook | PieceType::Queen) {
            attackers |= crate::attacks::rook_attacks(to, occ)
                & (b.pieces(PieceType::Rook) | b.pieces(PieceType::Queen));
        }
        attackers &= occ;

        stm = stm.flip();
        balance = -balance - 1 - SEE_VALUE[next_victim.index()];
        if balance >= 0 {
            // Cannot capture with the king into a still-defended square.
            if next_victim == PieceType::King && (attackers & b.colors(stm)).any() {
                stm = stm.flip();
            }
            break;
        }
    }
    b.stm() != stm
}

/// A first-order estimate, in centipawns, of what a move does to the static
/// evaluation — from the moving side's point of view, before the opponent
/// replies.
///
/// This is the search's **price signal**. `search::Pricing` charges a move by
/// how far behind the running best score it is predicted to land, and this is
/// the prediction. It is deliberately crude: one piece-square delta plus the
/// victim's value, with no exchange sequence and no phase taper (the mid- and
/// endgame tables are averaged). That is the point — a feature costing a full
/// eval call per move would cost more than the search it buys, and the whole
/// design rests on the price of a move being roughly five table lookups.
pub fn move_gain(b: &Board, mv: crate::chess_move::Move) -> Score {
    let c = b.stm();
    let mover = b.piece_at(mv.from()).piece_type().index();
    // White reads the tables mirrored, exactly as `evaluate_pst` does.
    let idx = |sq: Square| {
        if c == Color::White {
            sq.flip_rank().index()
        } else {
            sq.index()
        }
    };
    let pst = |p: usize, sq: Square| (MG_PST[p][idx(sq)] + EG_PST[p][idx(sq)]) / 2;

    let mut g = pst(mover, mv.to()) - pst(mover, mv.from());
    if mv.is_ep() {
        g += SEE_VALUE[PieceType::Pawn.index()];
    } else {
        let victim = b.piece_at(mv.to());
        if victim.is_some() {
            g += SEE_VALUE[victim.piece_type().index()];
        }
    }
    if mv.is_promotion() {
        let p = mv.promo_piece().index();
        g += SEE_VALUE[p] - SEE_VALUE[PieceType::Pawn.index()];
        g += pst(p, mv.to()) - pst(mover, mv.to());
    }
    g
}

#[cfg(test)]
mod draw_value_tests {
    use super::*;

    #[test]
    fn symmetric_knob_roundtrips() {
        // One test owns the globals so parallel tests cannot race it.
        let o = draw_value();
        set_draw_value(0.0);
        assert_eq!(draw_value(), 0.0);
        assert_eq!(draw_value_for(Color::White), 0.0);
        set_draw_value(o);
    }

    #[test]
    fn contempt_bonus_shape() {
        // Off is exactly off, whatever the position.
        assert_eq!(contempt_bonus(0.9, 2.0, 1.0, 0.0), 0.0);
        // Parity and decisive positions get nothing, exactly.
        assert_eq!(contempt_bonus(0.9, 0.0, 1.0, 100.0), 0.0);
        assert_eq!(contempt_bonus(0.0, 2.0, 1.0, 100.0), 0.0);
        // Antisymmetric in the gap (negates across a ply), bounded by b.
        let (hi, lo) = (contempt_bonus(0.9, 2.0, 1.0, 100.0), contempt_bonus(0.9, -2.0, 1.0, 100.0));
        assert!((hi + lo).abs() < 1e-9);
        assert!(hi > 0.0 && hi <= 100.0);
        assert!(lo < 0.0 && lo >= -100.0);
        // Behind seeks, ahead fights.
        assert!(contempt_bonus(0.5, 0.5, 1.0, 100.0) > 0.0);
        assert!(contempt_bonus(0.5, -0.5, 1.0, 100.0) < 0.0);
    }

    #[test]
    fn contempt_knobs_default_off() {
        // Owns the contempt globals so parallel tests cannot race it.
        let (oa, ob, of) = (contempt_a(), contempt_b(), contempt_flat());
        assert_eq!((oa, ob, of), (1.0, 0.0, false));
        set_contempt_a(2.0);
        set_contempt_b(100.0);
        set_contempt_flat(true);
        assert_eq!((contempt_a(), contempt_b(), contempt_flat()), (2.0, 100.0, true));
        set_contempt_a(oa);
        set_contempt_b(ob);
        set_contempt_flat(of);
    }
}
