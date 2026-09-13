//! Move generation correctness harness.
//!
//! Perft counts leaf nodes of the legal move tree to a fixed depth. It is the
//! only way to know movegen is right: the counts are exact, published, and a
//! single missing edge case (an en-passant pin, a castling-through-check) shows
//! up as a mismatch at depth 4 or 5. Nothing above movegen is worth writing
//! until every one of these passes.

use crate::board::Board;
use crate::movegen::{generate, GenType};
use crate::chess_move::MoveList;

pub fn perft(b: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    generate(b, GenType::All, &mut list);
    // Bulk counting: at depth 1 the move count IS the answer, since generation
    // is legal. Worth ~5x and is standard, but it means depth-1 perft never
    // exercises make_move — the deeper tests do.
    if depth == 1 {
        return list.len() as u64;
    }
    let mut n = 0;
    for m in list.iter() {
        n += perft(&b.make_move(m), depth - 1);
    }
    n
}

/// Per-root-move breakdown, for bisecting a mismatch against another engine.
pub fn perft_divide(b: &Board, depth: u32) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let mut list = MoveList::new();
    generate(b, GenType::All, &mut list);
    for m in list.iter() {
        let n = if depth <= 1 {
            1
        } else {
            perft(&b.make_move(m), depth - 1)
        };
        out.push((m.to_uci(), n));
    }
    out.sort();
    out
}

/// The standard positions from the Chess Programming Wiki. Between them they
/// cover every awkward rule: en-passant pins, castling rights lost by rook
/// capture, promotion into check, and discovered checks.
pub const SUITE: &[(&str, &str, &[u64])] = &[
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &[1, 20, 400, 8902, 197_281, 4_865_609, 119_060_324],
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        &[1, 48, 2039, 97_862, 4_085_603, 193_690_690],
    ),
    (
        "position 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        &[1, 14, 191, 2812, 43_238, 674_624, 11_030_083],
    ),
    (
        "position 4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        &[1, 6, 264, 9467, 422_333, 15_833_292],
    ),
    (
        "position 5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        &[1, 44, 1486, 62_379, 2_103_487, 89_941_194],
    ),
    (
        "position 6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        &[1, 46, 2079, 89_890, 3_894_594, 164_075_551],
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Depths kept modest so `cargo test` stays fast; the `perft` binary runs
    /// the full suite including the 100M+ nodes at the deepest levels.
    #[test]
    fn suite_shallow() {
        crate::init();
        for (name, fen, counts) in SUITE {
            let b = Board::from_fen(fen).unwrap_or_else(|e| panic!("{name}: {e}"));
            for (depth, &expect) in counts.iter().enumerate().take(5) {
                let got = perft(&b, depth as u32);
                assert_eq!(got, expect, "{name} depth {depth}: got {got}, want {expect}");
            }
        }
    }

    #[test]
    fn fen_roundtrip() {
        crate::init();
        for (name, fen, _) in SUITE {
            let b = Board::from_fen(fen).unwrap();
            assert_eq!(&b.to_fen(), fen, "{name}");
        }
    }

    /// The incremental zobrist updates in `make_move` must agree with a
    /// from-scratch recomputation after every move, or the TT will alias
    /// unrelated positions.
    #[test]
    fn zobrist_incremental_matches_scratch() {
        crate::init();
        for (_, fen, _) in SUITE {
            let b = Board::from_fen(fen).unwrap();
            assert_eq!(b.pawn_key(), b.recompute_pawn_key(), "pawn key mismatch at {fen}");
            let mut list = MoveList::new();
            generate(&b, GenType::All, &mut list);
            for m in list.iter() {
                let nb = b.make_move(m);
                assert_eq!(
                    nb.key(),
                    nb.recompute_key(),
                    "key mismatch after {m} from {fen}"
                );
                assert_eq!(
                    nb.pawn_key(),
                    nb.recompute_pawn_key(),
                    "pawn key mismatch after {m} from {fen}"
                );
                let mut l2 = MoveList::new();
                generate(&nb, GenType::All, &mut l2);
                for m2 in l2.iter() {
                    let nb2 = nb.make_move(m2);
                    assert_eq!(nb2.key(), nb2.recompute_key(), "key mismatch after {m} {m2}");
                    assert_eq!(
                        nb2.pawn_key(),
                        nb2.recompute_pawn_key(),
                        "pawn key mismatch after {m} {m2}"
                    );
                }
            }
        }
    }
}
