//! Standard Algebraic Notation.
//!
//! Not needed by the engine — UCI is coordinate notation all the way down —
//! but needed by everything that a *human* reads: the PGN files the match
//! runner writes, and the principal variation the web board displays. "Nf3 Nc6
//! Bb5" is legible at a glance in a way that "g1f3 b8c6 f1b5" is not, and the
//! whole point of the GUI work is being able to see what the engine is doing.
//!
//! Generation only. Parsing SAN would need the full disambiguation grammar and
//! nothing here has an input in SAN.

use crate::board::Board;
use crate::chess_move::{Move, MoveList};
use crate::movegen::{generate, GenType};
use crate::types::PieceType;

/// SAN for one legal move in `b`.
///
/// Disambiguation follows the PGN spec: file if that is unique, else rank, else
/// both. `+`/`#` are appended by actually making the move, which is cheap
/// (copy-make) and cannot disagree with the search's own notion of check.
pub fn san(b: &Board, mv: Move) -> String {
    if mv.is_none() {
        return "--".to_string();
    }
    let mut s = String::with_capacity(8);
    let pt = b.piece_at(mv.from()).piece_type();

    if mv.is_castle() {
        s.push_str(if mv.flag() == crate::chess_move::flags::KING_CASTLE {
            "O-O"
        } else {
            "O-O-O"
        });
    } else if pt == PieceType::Pawn {
        if mv.is_capture() {
            s.push((b'a' + mv.from().file()) as char);
            s.push('x');
        }
        s.push_str(&mv.to().to_string());
        if mv.is_promotion() {
            s.push('=');
            s.push(mv.promo_piece().char().to_ascii_uppercase());
        }
    } else {
        s.push(pt.char().to_ascii_uppercase());

        // Any other move by the same piece type landing on the same square has
        // to be distinguished from this one.
        let mut list = MoveList::new();
        generate(b, GenType::All, &mut list);
        let mut same_file = false;
        let mut same_rank = false;
        let mut ambiguous = false;
        for other in list.iter() {
            if other.0 == mv.0 || other.to() != mv.to() {
                continue;
            }
            if b.piece_at(other.from()).piece_type() != pt {
                continue;
            }
            ambiguous = true;
            if other.from().file() == mv.from().file() {
                same_file = true;
            }
            if other.from().rank() == mv.from().rank() {
                same_rank = true;
            }
        }
        if ambiguous {
            if !same_file {
                s.push((b'a' + mv.from().file()) as char);
            } else if !same_rank {
                s.push((b'1' + mv.from().rank()) as char);
            } else {
                s.push((b'a' + mv.from().file()) as char);
                s.push((b'1' + mv.from().rank()) as char);
            }
        }
        if mv.is_capture() {
            s.push('x');
        }
        s.push_str(&mv.to().to_string());
    }

    let after = b.make_move(mv);
    if after.in_check() {
        let mut replies = MoveList::new();
        generate(&after, GenType::All, &mut replies);
        s.push(if replies.is_empty() { '#' } else { '+' });
    }
    s
}

/// SAN for a whole line, e.g. a principal variation. Moves are applied as they
/// are converted, so the notation is correct for each successive position;
/// conversion stops at the first move that is not legal.
pub fn san_line(start: &Board, line: &[Move]) -> Vec<String> {
    let mut b = *start;
    let mut out = Vec::with_capacity(line.len());
    for &mv in line {
        let mut list = MoveList::new();
        generate(&b, GenType::All, &mut list);
        if !list.contains(mv) {
            break;
        }
        out.push(san(&b, mv));
        b = b.make_move(mv);
    }
    out
}

/// A movetext line with move numbers: `1. e4 e5 2. Nf3`. Used for PGN bodies
/// and for the GUI move list.
pub fn movetext(start: &Board, line: &[Move], wrap: usize) -> String {
    let sans = san_line(start, line);
    let mut out = String::new();
    let mut col = 0;
    let mut fullmove = start.fullmove() as usize;
    let mut white_to_move = start.stm() == crate::types::Color::White;
    for (i, s) in sans.iter().enumerate() {
        let mut tok = String::new();
        if white_to_move {
            tok.push_str(&format!("{fullmove}. "));
        } else if i == 0 {
            tok.push_str(&format!("{fullmove}... "));
        }
        tok.push_str(s);
        if wrap > 0 && col + tok.len() + 1 > wrap {
            out.push('\n');
            col = 0;
        } else if col > 0 {
            out.push(' ');
            col += 1;
        }
        col += tok.len();
        out.push_str(&tok);
        if !white_to_move {
            fullmove += 1;
        }
        white_to_move = !white_to_move;
    }
    out
}
