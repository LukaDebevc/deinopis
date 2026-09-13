//! Legal move generation.
//!
//! Moves are generated *legal*, not pseudo-legal-then-filtered. The search
//! never has to make a move to discover it was illegal, and `MoveList::len()`
//! is directly the number of legal moves — which makes mate and stalemate
//! detection free.
//!
//! Legality is enforced by three masks computed once per node:
//!   * `check_mask` — when in check, the squares a non-king move must land on
//!     (capture the checker, or block the ray). Double check leaves king moves only.
//!   * `pinned` — our pieces with exactly one blocker between our king and an
//!     enemy slider. A pinned piece may only move along `line(king, itself)`.
//!   * king danger squares — enemy attacks computed with OUR KING REMOVED from
//!     the occupancy, so a king cannot walk backwards along a checking ray.

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::board::{castling, Board};
use crate::chess_move::{flags, Move, MoveList};
use crate::types::{squares::*, Color, PieceType, Square};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenType {
    /// Every legal move.
    All,
    /// Captures and queen promotions only — the quiescence-search set.
    Captures,
}

pub fn generate(b: &Board, gen: GenType, list: &mut MoveList) {
    let us = b.stm();
    let them = us.flip();
    let occ = b.occupied();
    let own = b.colors(us);
    let enemy = b.colors(them);
    let ksq = b.king_square(us);
    let checkers = b.checkers();
    let n_checkers = checkers.count();

    // ---- king moves. Always legal to try; always the only option in double check.
    let danger = king_danger(b, us, occ - Bitboard::from_square(ksq));
    let mut king_targets = attacks::king_attacks(ksq) - own - danger;
    if gen == GenType::Captures {
        king_targets &= enemy;
    }
    for to in king_targets {
        let flag = if enemy.contains(to) {
            flags::CAPTURE
        } else {
            flags::QUIET
        };
        list.push(Move::new(ksq, to, flag));
    }

    if n_checkers >= 2 {
        return; // only the king can escape a double check
    }

    // ---- what a non-king move must do
    let check_mask = if n_checkers == 1 {
        let checker = checkers.lsb();
        attacks::between(ksq, checker) | checkers
    } else {
        Bitboard::ALL
    };

    let mut targets = check_mask - own;
    if gen == GenType::Captures {
        targets &= enemy;
    }

    let pinned = compute_pinned(b, us, ksq, occ);

    // A pinned piece may only move along the ray through the king. `pinned` is
    // empty in most positions, so the whole check is branched around.
    let legal = |from: Square, to: Square| -> bool {
        pinned.is_empty() || !pinned.contains(from) || attacks::line(ksq, from).contains(to)
    };

    // ---- knights (a pinned knight can never move at all)
    for from in b.colored(us, PieceType::Knight) - pinned {
        for to in attacks::knight_attacks(from) & targets {
            list.push(Move::new(from, to, cap_flag(enemy, to)));
        }
    }

    // ---- sliders
    for (pt, f) in [
        (PieceType::Bishop, attacks::bishop_attacks as fn(Square, Bitboard) -> Bitboard),
        (PieceType::Rook, attacks::rook_attacks),
        (PieceType::Queen, attacks::queen_attacks),
    ] {
        for from in b.colored(us, pt) {
            for to in f(from, occ) & targets {
                if legal(from, to) {
                    list.push(Move::new(from, to, cap_flag(enemy, to)));
                }
            }
        }
    }

    gen_pawns(b, gen, us, occ, enemy, ksq, check_mask, pinned, list);

    // ---- castling: never legal out of check, and the king may not pass
    // through or land on an attacked square.
    if gen == GenType::All && n_checkers == 0 {
        let (kside, qside, e, g, c, d, f_sq, b_sq) = if us == Color::White {
            (castling::WK, castling::WQ, E1, G1, C1, D1, F1, B1)
        } else {
            (castling::BK, castling::BQ, E8, G8, C8, D8, F8, B8)
        };
        let rights = b.castling_rights();
        if rights & kside != 0
            && (occ & (Bitboard::from_square(f_sq) | Bitboard::from_square(g))).is_empty()
            && !danger.contains(f_sq)
            && !danger.contains(g)
        {
            list.push(Move::new(e, g, flags::KING_CASTLE));
        }
        if rights & qside != 0
            && (occ
                & (Bitboard::from_square(d)
                    | Bitboard::from_square(c)
                    | Bitboard::from_square(b_sq)))
            .is_empty()
            && !danger.contains(d)
            && !danger.contains(c)
        {
            list.push(Move::new(e, c, flags::QUEEN_CASTLE));
        }
    }
}

#[inline(always)]
fn cap_flag(enemy: Bitboard, to: Square) -> u16 {
    if enemy.contains(to) {
        flags::CAPTURE
    } else {
        flags::QUIET
    }
}

/// Squares the enemy attacks, computed with our king removed from occupancy so
/// that a slider's ray extends *through* the king's current square. Without
/// this the king would be allowed to step backwards along a checking ray.
fn king_danger(b: &Board, us: Color, occ_no_king: Bitboard) -> Bitboard {
    let them = us.flip();
    let mut d = Bitboard::EMPTY;
    for from in b.colored(them, PieceType::Pawn) {
        d |= attacks::pawn_attacks(them, from);
    }
    for from in b.colored(them, PieceType::Knight) {
        d |= attacks::knight_attacks(from);
    }
    for from in b.diagonal_sliders(them) {
        d |= attacks::bishop_attacks(from, occ_no_king);
    }
    for from in b.orthogonal_sliders(them) {
        d |= attacks::rook_attacks(from, occ_no_king);
    }
    d |= attacks::king_attacks(b.king_square(them));
    d
}

/// Our pieces standing alone between our king and an enemy slider.
fn compute_pinned(b: &Board, us: Color, ksq: Square, occ: Bitboard) -> Bitboard {
    let them = us.flip();
    // Sliders that would hit the king on an empty board are the only candidates.
    let snipers = (attacks::rook_attacks(ksq, Bitboard::EMPTY) & b.orthogonal_sliders(them))
        | (attacks::bishop_attacks(ksq, Bitboard::EMPTY) & b.diagonal_sliders(them));
    let mut pinned = Bitboard::EMPTY;
    for sniper in snipers {
        let blockers = attacks::between(ksq, sniper) & occ;
        if blockers.count() == 1 {
            pinned |= blockers & b.colors(us);
        }
    }
    pinned
}

#[allow(clippy::too_many_arguments)]
fn gen_pawns(
    b: &Board,
    gen: GenType,
    us: Color,
    occ: Bitboard,
    enemy: Bitboard,
    ksq: Square,
    check_mask: Bitboard,
    pinned: Bitboard,
    list: &mut MoveList,
) {
    let pawns = b.colored(us, PieceType::Pawn);
    if pawns.is_empty() {
        return;
    }
    let promo_rank = if us == Color::White {
        Bitboard::rank(7)
    } else {
        Bitboard::rank(0)
    };
    let double_rank = if us == Color::White {
        Bitboard::rank(2)
    } else {
        Bitboard::rank(5)
    };
    let legal = |from: Square, to: Square| -> bool {
        pinned.is_empty() || !pinned.contains(from) || attacks::line(ksq, from).contains(to)
    };

    let emit = |from: Square, to: Square, capture: bool, list: &mut MoveList| {
        if !legal(from, to) {
            return;
        }
        if promo_rank.contains(to) {
            let base = if capture {
                flags::PROMO_CAP_N
            } else {
                flags::PROMO_N
            };
            // Queen first: it is the only promotion the search usually needs,
            // and emitting it first helps move ordering before scoring runs.
            list.push(Move::new(from, to, base + 3));
            if gen == GenType::All {
                list.push(Move::new(from, to, base));
                list.push(Move::new(from, to, base + 1));
                list.push(Move::new(from, to, base + 2));
            }
        } else {
            list.push(Move::new(
                from,
                to,
                if capture { flags::CAPTURE } else { flags::QUIET },
            ));
        }
    };

    // Captures (and capture-promotions).
    for (shifted, df) in [
        (pawns.forward_left(us), -1i8),
        (pawns.forward_right(us), 1i8),
    ] {
        for to in shifted & enemy & check_mask {
            let from = to.offset(-df, -us.forward()).unwrap();
            emit(from, to, true, list);
        }
    }

    // Quiet pushes. Promotions are generated even in Captures mode: a promotion
    // swings the material balance and must be seen by quiescence.
    let single = pawns.forward(us) - occ;
    if gen == GenType::All {
        for to in single & check_mask {
            let from = to.offset(0, -us.forward()).unwrap();
            emit(from, to, false, list);
        }
        for to in (single & double_rank).forward(us) - occ & check_mask {
            let from = to.offset(0, -2 * us.forward()).unwrap();
            if legal(from, to) {
                list.push(Move::new(from, to, flags::DOUBLE_PUSH));
            }
        }
    } else {
        for to in single & promo_rank & check_mask {
            let from = to.offset(0, -us.forward()).unwrap();
            emit(from, to, false, list);
        }
    }

    // En passant. Rare enough that full legality is checked explicitly rather
    // than folded into the masks: capturing e.p. removes TWO pawns from one
    // rank, which can expose a horizontal discovered check that `pinned` (which
    // only ever sees a single blocker) cannot detect.
    if let Some(ep) = b.ep_square() {
        let cap_sq = Square::new(ep.file(), if us == Color::White { 4 } else { 3 });
        // Resolving check by capturing the checking pawn is legal even though
        // `ep` itself is not in check_mask.
        if check_mask.contains(ep) || b.checkers().contains(cap_sq) {
            for from in attacks::pawn_attacks(us.flip(), ep) & b.colored(us, PieceType::Pawn) {
                if ep_is_legal(b, us, ksq, occ, from, ep, cap_sq) {
                    list.push(Move::new(from, ep, flags::EP_CAPTURE));
                }
            }
        }
    }
}

fn ep_is_legal(
    b: &Board,
    us: Color,
    ksq: Square,
    occ: Bitboard,
    from: Square,
    to: Square,
    cap_sq: Square,
) -> bool {
    let them = us.flip();
    let occ2 = (occ - Bitboard::from_square(from) - Bitboard::from_square(cap_sq))
        | Bitboard::from_square(to);
    // Only sliders can be newly revealed; a pawn/knight/king attack on the king
    // cannot appear from moving these two squares.
    let diag = b.diagonal_sliders(them) - Bitboard::from_square(cap_sq);
    let orth = b.orthogonal_sliders(them) - Bitboard::from_square(cap_sq);
    (attacks::bishop_attacks(ksq, occ2) & diag).is_empty()
        && (attacks::rook_attacks(ksq, occ2) & orth).is_empty()
}

/// Convenience wrapper.
pub fn legal_moves(b: &Board) -> MoveList {
    let mut l = MoveList::new();
    generate(b, GenType::All, &mut l);
    l
}
