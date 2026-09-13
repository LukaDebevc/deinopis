//! Game-level rules: when a game is over and why.
//!
//! The search has its own, deliberately different, draw detection (a *single*
//! repetition inside the tree counts as a draw — see `search::is_draw`). That
//! is a pruning heuristic and is wrong as a rule of chess. This module is the
//! rule: three occurrences, fifty moves, no legal move. Anything that
//! adjudicates a real game — the match runner and the web board — must use
//! this one, or engines will claim draws that never happened.

use crate::board::Board;
use crate::chess_move::MoveList;
use crate::movegen::{generate, GenType};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Ongoing,
    Checkmate,
    Stalemate,
    FiftyMove,
    Repetition,
    InsufficientMaterial,
}

impl Status {
    pub fn is_over(self) -> bool {
        self != Status::Ongoing
    }

    /// Lowercase tag, as sent to the web board and written into PGN.
    pub fn tag(self) -> &'static str {
        match self {
            Status::Ongoing => "ok",
            Status::Checkmate => "checkmate",
            Status::Stalemate => "stalemate",
            Status::FiftyMove => "fifty-move",
            Status::Repetition => "repetition",
            Status::InsufficientMaterial => "insufficient-material",
        }
    }
}

/// `history` holds the zobrist key of every position *before* the current one,
/// so the current position is a threefold repetition when its key already
/// appears there twice.
pub fn status(b: &Board, history: &[u64]) -> Status {
    let mut list = MoveList::new();
    generate(b, GenType::All, &mut list);
    if list.is_empty() {
        return if b.in_check() {
            Status::Checkmate
        } else {
            Status::Stalemate
        };
    }
    if insufficient_material(b) {
        return Status::InsufficientMaterial;
    }
    if b.halfmove() >= 100 {
        return Status::FiftyMove;
    }
    // Only positions with the same side to move can repeat, and a repetition
    // cannot cross an irreversible move, so scanning the last `halfmove` keys
    // is both sufficient and cheaper than scanning the whole game.
    //
    // The `skip(1)` is load-bearing: the last entry in `history` is the
    // position one ply ago, which has the *other* side to move and can never
    // equal the current key. Stepping by two from there would sample exactly
    // the wrong half of the game.
    let key = b.key();
    let span = (b.halfmove() as usize).min(history.len());
    let reps = history[history.len() - span..]
        .iter()
        .rev()
        .skip(1)
        .step_by(2)
        .filter(|&&k| k == key)
        .count();
    if reps >= 2 {
        return Status::Repetition;
    }
    Status::Ongoing
}

/// FIDE dead positions, restricted to the cases every arbiter and every engine
/// agrees on: lone kings, and king plus one minor. K+B vs K+B on the same
/// colour is also dead but needs the square colours, and K+N+N vs K is *not*
/// dead (mate is possible, just not forcible), so neither is claimed here.
pub fn insufficient_material(b: &Board) -> bool {
    use crate::types::PieceType::*;
    if b.pieces(Pawn).any() || b.pieces(Rook).any() || b.pieces(Queen).any() {
        return false;
    }
    let minors = b.pieces(Knight).count() + b.pieces(Bishop).count();
    minors <= 1
}
