//! Board representation, FEN, and copy-make.
//!
//! `Board` is `Copy` (~180 bytes) and `make_move` returns a new one rather than
//! mutating in place with an undo log. Copying ~180 bytes costs well under 2% of
//! a node, and in exchange the search recursion is `search(&b.make_move(m), ..)`
//! with no state to unwind — which removes an entire class of bug and makes the
//! search trivially restructurable. That matters more than the 2% for an engine
//! we intend to keep rewriting.
//!
//! Note the NNUE accumulator will NOT live here: at 4 KB it is far too big to
//! copy per node, so it gets its own ply-indexed stack with incremental update.

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::chess_move::{flags, Move};
use crate::types::{squares::*, Color, Piece, PieceType, Square};
use crate::zobrist;

/// Castling rights as a 4-bit mask.
pub mod castling {
    pub const WK: u8 = 1;
    pub const WQ: u8 = 2;
    pub const BK: u8 = 4;
    pub const BQ: u8 = 8;
    pub const ALL: u8 = 15;
}

/// Per-square mask of the rights that survive a piece moving to or from it.
/// Applied as `rights &= CASTLE_MASK[from] & CASTLE_MASK[to]`, which handles
/// king moves, rook moves, and rooks being captured on their home square in one
/// branchless step.
static CASTLE_MASK: [u8; 64] = {
    let mut m = [castling::ALL; 64];
    m[4] = !(castling::WK | castling::WQ) & castling::ALL; // e1
    m[0] = !castling::WQ & castling::ALL; // a1
    m[7] = !castling::WK & castling::ALL; // h1
    m[60] = !(castling::BK | castling::BQ) & castling::ALL; // e8
    m[56] = !castling::BQ & castling::ALL; // a8
    m[63] = !castling::BK & castling::ALL; // h8
    m
};

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Board {
    by_color: [Bitboard; 2],
    by_type: [Bitboard; 6],
    mailbox: [Piece; 64],
    stm: Color,
    castling: u8,
    /// 64 means "none". Kept as a raw u8 so `Board` stays `Copy` and small.
    ep: u8,
    halfmove: u8,
    fullmove: u16,
    key: u64,
    /// Zobrist hash of the pawns alone. Maintained incrementally beside
    /// `key` (two gated XORs in add/remove_piece) for correction history
    /// (`library/015`): pawn structure persists across nodes, so the same
    /// entry serves many positions. Ignored by perft and the TT.
    pawn_key: u64,
    /// Enemy pieces giving check. Cached because nearly every node asks.
    checkers: Bitboard,
}

impl Board {
    pub fn empty() -> Board {
        Board {
            by_color: [Bitboard::EMPTY; 2],
            by_type: [Bitboard::EMPTY; 6],
            mailbox: [Piece::NONE; 64],
            stm: Color::White,
            castling: 0,
            ep: 64,
            halfmove: 0,
            fullmove: 1,
            key: 0,
            pawn_key: 0,
            checkers: Bitboard::EMPTY,
        }
    }

    pub fn startpos() -> Board {
        Board::from_fen(START_FEN).expect("start FEN is valid")
    }

    // ---------------------------------------------------------- accessors

    #[inline(always)]
    pub fn stm(&self) -> Color {
        self.stm
    }
    #[inline(always)]
    pub fn key(&self) -> u64 {
        self.key
    }
    /// Zobrist hash of the pawns alone (both colors, no side-to-move).
    /// Correction history indexes `[stm][pawn_key & 16383]` off this.
    #[inline(always)]
    pub fn pawn_key(&self) -> u64 {
        self.pawn_key
    }
    #[inline(always)]
    pub fn halfmove(&self) -> u8 {
        self.halfmove
    }
    #[inline(always)]
    pub fn fullmove(&self) -> u16 {
        self.fullmove
    }
    #[inline(always)]
    pub fn castling_rights(&self) -> u8 {
        self.castling
    }
    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        self.checkers
    }
    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers.any()
    }

    #[inline(always)]
    pub fn ep_square(&self) -> Option<Square> {
        if self.ep < 64 {
            Some(Square(self.ep))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.by_color[0] | self.by_color[1]
    }

    #[inline(always)]
    pub fn colors(&self, c: Color) -> Bitboard {
        unsafe { *self.by_color.get_unchecked(c.index()) }
    }

    #[inline(always)]
    pub fn pieces(&self, pt: PieceType) -> Bitboard {
        unsafe { *self.by_type.get_unchecked(pt.index()) }
    }

    #[inline(always)]
    pub fn colored(&self, c: Color, pt: PieceType) -> Bitboard {
        self.colors(c) & self.pieces(pt)
    }

    #[inline(always)]
    pub fn piece_at(&self, sq: Square) -> Piece {
        unsafe { *self.mailbox.get_unchecked(sq.index()) }
    }

    #[inline(always)]
    pub fn king_square(&self, c: Color) -> Square {
        self.colored(c, PieceType::King).lsb()
    }

    /// Sliders of colour `c` that move diagonally / orthogonally. Queens are in
    /// both, which is what every x-ray and SEE query wants.
    #[inline(always)]
    pub fn diagonal_sliders(&self, c: Color) -> Bitboard {
        self.colors(c) & (self.pieces(PieceType::Bishop) | self.pieces(PieceType::Queen))
    }

    #[inline(always)]
    pub fn orthogonal_sliders(&self, c: Color) -> Bitboard {
        self.colors(c) & (self.pieces(PieceType::Rook) | self.pieces(PieceType::Queen))
    }

    /// Every piece of either colour attacking `sq`, given occupancy `occ`.
    ///
    /// This is the engine's central primitive — SEE, check detection, recapture
    /// detection and "is this square defended" are all one call. Passing `occ`
    /// explicitly is what lets SEE peel attackers off one at a time and let
    /// x-rays behind them come into view.
    #[inline]
    pub fn attackers_to(&self, sq: Square, occ: Bitboard) -> Bitboard {
        (attacks::pawn_attacks(Color::White, sq) & self.colored(Color::Black, PieceType::Pawn))
            | (attacks::pawn_attacks(Color::Black, sq)
                & self.colored(Color::White, PieceType::Pawn))
            | (attacks::knight_attacks(sq) & self.pieces(PieceType::Knight))
            | (attacks::king_attacks(sq) & self.pieces(PieceType::King))
            | (attacks::bishop_attacks(sq, occ)
                & (self.pieces(PieceType::Bishop) | self.pieces(PieceType::Queen)))
            | (attacks::rook_attacks(sq, occ)
                & (self.pieces(PieceType::Rook) | self.pieces(PieceType::Queen)))
    }

    #[inline]
    pub fn is_attacked(&self, sq: Square, by: Color, occ: Bitboard) -> bool {
        (self.attackers_to(sq, occ) & self.colors(by)).any()
    }

    /// True if the side to move has any piece other than pawns and the king.
    /// Null-move pruning must be disabled when this is false — that is exactly
    /// the zugzwang-prone material configuration.
    #[inline]
    pub fn has_non_pawn_material(&self, c: Color) -> bool {
        (self.colors(c)
            - self.pieces(PieceType::Pawn)
            - self.pieces(PieceType::King))
        .any()
    }

    // ---------------------------------------------------------- mutation

    #[inline(always)]
    fn add_piece(&mut self, sq: Square, p: Piece) {
        debug_assert!(self.mailbox[sq.index()].is_none());
        let bb = Bitboard::from_square(sq);
        self.by_color[p.color().index()] |= bb;
        self.by_type[p.piece_type().index()] |= bb;
        self.mailbox[sq.index()] = p;
        self.key ^= zobrist::piece_key(p, sq);
        if p.piece_type() == PieceType::Pawn {
            self.pawn_key ^= zobrist::pawn_key_component(p.color(), sq);
        }
    }

    #[inline(always)]
    fn remove_piece(&mut self, sq: Square) -> Piece {
        let p = self.mailbox[sq.index()];
        debug_assert!(p.is_some());
        let bb = Bitboard::from_square(sq);
        self.by_color[p.color().index()] ^= bb;
        self.by_type[p.piece_type().index()] ^= bb;
        self.mailbox[sq.index()] = Piece::NONE;
        self.key ^= zobrist::piece_key(p, sq);
        if p.piece_type() == PieceType::Pawn {
            self.pawn_key ^= zobrist::pawn_key_component(p.color(), sq);
        }
        p
    }

    #[inline(always)]
    fn move_piece(&mut self, from: Square, to: Square) {
        let p = self.remove_piece(from);
        self.add_piece(to, p);
    }

    fn recompute_checkers(&mut self) {
        let ksq = self.king_square(self.stm);
        self.checkers = self.attackers_to(ksq, self.occupied()) & self.colors(self.stm.flip());
    }

    /// Apply `mv`, returning the resulting position. `self` is untouched.
    #[must_use]
    pub fn make_move(&self, mv: Move) -> Board {
        let mut b = *self;
        let us = b.stm;
        let them = us.flip();
        let from = mv.from();
        let to = mv.to();
        let moving = b.piece_at(from);
        debug_assert!(moving.is_some() && moving.color() == us);

        // Clear the old ep key before anything else can change the file.
        if b.ep < 64 {
            b.key ^= zobrist::keys().ep_file[Square(b.ep).file() as usize];
        }
        b.ep = 64;

        b.halfmove = b.halfmove.saturating_add(1);
        if moving.piece_type() == PieceType::Pawn {
            b.halfmove = 0;
        }

        match mv.flag() {
            flags::EP_CAPTURE => {
                // The captured pawn is on the *mover's* origin rank, not `to`.
                let cap_sq = Square::new(to.file(), from.rank());
                b.remove_piece(cap_sq);
                b.move_piece(from, to);
                b.halfmove = 0;
            }
            flags::KING_CASTLE => {
                let (rf, rt) = if us == Color::White { (H1, F1) } else { (H8, F8) };
                b.move_piece(from, to);
                b.move_piece(rf, rt);
            }
            flags::QUEEN_CASTLE => {
                let (rf, rt) = if us == Color::White { (A1, D1) } else { (A8, D8) };
                b.move_piece(from, to);
                b.move_piece(rf, rt);
            }
            flags::DOUBLE_PUSH => {
                b.move_piece(from, to);
                let epsq = Square::new(from.file(), (from.rank() + to.rank()) / 2);
                b.ep = epsq.0;
                b.key ^= zobrist::keys().ep_file[epsq.file() as usize];
            }
            _ => {
                if mv.is_capture() {
                    b.remove_piece(to);
                    b.halfmove = 0;
                }
                if mv.is_promotion() {
                    b.remove_piece(from);
                    b.add_piece(to, Piece::new(us, mv.promo_piece()));
                } else {
                    b.move_piece(from, to);
                }
            }
        }

        let new_rights = b.castling & CASTLE_MASK[from.index()] & CASTLE_MASK[to.index()];
        if new_rights != b.castling {
            b.key ^= zobrist::keys().castling[b.castling as usize];
            b.key ^= zobrist::keys().castling[new_rights as usize];
            b.castling = new_rights;
        }

        if us == Color::Black {
            b.fullmove += 1;
        }
        b.stm = them;
        b.key ^= zobrist::keys().side;
        b.recompute_checkers();
        b
    }

    /// Pass the turn without moving. Used by null-move pruning; illegal when in
    /// check, which the caller must ensure.
    #[must_use]
    pub fn make_null(&self) -> Board {
        debug_assert!(!self.in_check());
        let mut b = *self;
        if b.ep < 64 {
            b.key ^= zobrist::keys().ep_file[Square(b.ep).file() as usize];
            b.ep = 64;
        }
        b.stm = b.stm.flip();
        b.key ^= zobrist::keys().side;
        b.halfmove = b.halfmove.saturating_add(1);
        b.checkers = Bitboard::EMPTY; // side to move cannot be in check after a null
        b
    }

    // ---------------------------------------------------------- FEN

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let mut b = Board::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("FEN needs at least 4 fields, got {}", parts.len()));
        }

        let mut rank: i32 = 7;
        let mut file: i32 = 0;
        for c in parts[0].chars() {
            match c {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => file += c as i32 - '0' as i32,
                _ => {
                    let p = Piece::from_char(c).ok_or(format!("bad piece '{c}'"))?;
                    if !(0..8).contains(&file) || !(0..8).contains(&rank) {
                        return Err("piece placement out of range".into());
                    }
                    b.add_piece(Square::new(file as u8, rank as u8), p);
                    file += 1;
                }
            }
        }

        b.stm = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            s => return Err(format!("bad side to move '{s}'")),
        };

        b.castling = 0;
        for c in parts[2].chars() {
            match c {
                'K' => b.castling |= castling::WK,
                'Q' => b.castling |= castling::WQ,
                'k' => b.castling |= castling::BK,
                'q' => b.castling |= castling::BQ,
                '-' => {}
                _ => return Err(format!("bad castling char '{c}'")),
            }
        }

        // Drop any right the board cannot actually back with a king on its
        // home square AND a rook in the matching corner.
        //
        // Without this, movegen happily generates the castle, `make_move`
        // reads the rook off an empty corner, and `Piece::NONE.piece_type()`
        // panics — with `panic = "abort"` that takes the whole process down,
        // GUI server included. It is not a hypothetical: hand-editing a FEN to
        // remove a rook (setting up a material-odds game, say) leaves "KQkq"
        // sitting there, and every engine sanitises this on input for exactly
        // that reason. A FEN is untrusted input; the board invariants are not
        // the typist's job to maintain.
        let mut keep = 0u8;
        for (bit, king, rook, col) in [
            (castling::WK, 4usize, 7usize, Color::White),
            (castling::WQ, 4, 0, Color::White),
            (castling::BK, 60, 63, Color::Black),
            (castling::BQ, 60, 56, Color::Black),
        ] {
            let ok = b.piece_at(Square::from_index(king)) == Piece::new(col, PieceType::King)
                && b.piece_at(Square::from_index(rook)) == Piece::new(col, PieceType::Rook);
            if ok {
                keep |= bit;
            }
        }
        b.castling &= keep;

        b.ep = match parts[3] {
            "-" => 64,
            s => Square::from_uci(s).ok_or(format!("bad ep square '{s}'"))?.0,
        };

        b.halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        b.fullmove = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

        if b.colored(Color::White, PieceType::King).count() != 1
            || b.colored(Color::Black, PieceType::King).count() != 1
        {
            return Err("each side needs exactly one king".into());
        }

        // The piece keys were folded in by add_piece; add the rest.
        b.key ^= zobrist::keys().castling[b.castling as usize];
        if b.ep < 64 {
            b.key ^= zobrist::keys().ep_file[Square(b.ep).file() as usize];
        }
        if b.stm == Color::Black {
            b.key ^= zobrist::keys().side;
        }
        b.recompute_checkers();
        Ok(b)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut gap = 0;
            for file in 0..8 {
                let p = self.piece_at(Square::new(file, rank));
                if p.is_none() {
                    gap += 1;
                } else {
                    if gap > 0 {
                        s.push_str(&gap.to_string());
                        gap = 0;
                    }
                    s.push(p.char());
                }
            }
            if gap > 0 {
                s.push_str(&gap.to_string());
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.stm == Color::White { 'w' } else { 'b' });
        s.push(' ');
        if self.castling == 0 {
            s.push('-');
        } else {
            for (bit, c) in [
                (castling::WK, 'K'),
                (castling::WQ, 'Q'),
                (castling::BK, 'k'),
                (castling::BQ, 'q'),
            ] {
                if self.castling & bit != 0 {
                    s.push(c);
                }
            }
        }
        s.push(' ');
        match self.ep_square() {
            Some(sq) => s.push_str(&sq.to_string()),
            None => s.push('-'),
        }
        format!("{s} {} {}", self.halfmove, self.fullmove)
    }

    /// Recompute the key from scratch. Only used by tests and debug assertions
    /// to prove the incremental updates in `make_move` stay in sync.
    pub fn recompute_key(&self) -> u64 {
        let mut k = 0u64;
        for i in 0..64 {
            let sq = Square::from_index(i);
            let p = self.piece_at(sq);
            if p.is_some() {
                k ^= zobrist::piece_key(p, sq);
            }
        }
        k ^= zobrist::keys().castling[self.castling as usize];
        if self.ep < 64 {
            k ^= zobrist::keys().ep_file[Square(self.ep).file() as usize];
        }
        if self.stm == Color::Black {
            k ^= zobrist::keys().side;
        }
        k
    }

    /// Recompute the pawn key from scratch. Test-only, beside `recompute_key`.
    pub fn recompute_pawn_key(&self) -> u64 {
        let mut k = 0u64;
        for i in 0..64 {
            let sq = Square::from_index(i);
            let p = self.piece_at(sq);
            if p.is_some() && p.piece_type() == PieceType::Pawn {
                k ^= zobrist::pawn_key_component(p.color(), sq);
            }
        }
        k
    }
}

impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  +------------------------+")?;
        for rank in (0..8).rev() {
            write!(f, "{} |", rank + 1)?;
            for file in 0..8 {
                write!(f, " {} ", self.piece_at(Square::new(file, rank)).char())?;
            }
            writeln!(f, "|")?;
        }
        writeln!(f, "  +------------------------+")?;
        writeln!(f, "    a  b  c  d  e  f  g  h")?;
        writeln!(f, "  {}", self.to_fen())?;
        write!(f, "  key: {:016X}", self.key)
    }
}
