//! Packed move representation.
//!
//! 16 bits: `from` (0-5), `to` (6-11), `flags` (12-15).
//! Flag layout follows the standard scheme so the two high bits are directly
//! meaningful: bit 15 = promotion, bit 14 = capture.
//!
//! 16 bits matters: the transposition table stores a move per entry, and the
//! killer/counter-move tables are indexed by the million. Anything wider costs
//! cache lines in the hottest structures in the engine.

use crate::types::{PieceType, Square};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(pub u16);

pub mod flags {
    pub const QUIET: u16 = 0;
    pub const DOUBLE_PUSH: u16 = 1;
    pub const KING_CASTLE: u16 = 2;
    pub const QUEEN_CASTLE: u16 = 3;
    pub const CAPTURE: u16 = 4;
    pub const EP_CAPTURE: u16 = 5;
    pub const PROMO_N: u16 = 8;
    pub const PROMO_B: u16 = 9;
    pub const PROMO_R: u16 = 10;
    pub const PROMO_Q: u16 = 11;
    pub const PROMO_CAP_N: u16 = 12;
    pub const PROMO_CAP_B: u16 = 13;
    pub const PROMO_CAP_R: u16 = 14;
    pub const PROMO_CAP_Q: u16 = 15;
}

impl Move {
    /// Not a legal move (from == to is impossible), so it doubles as the empty
    /// slot in killer tables and TT entries.
    pub const NONE: Move = Move(0);

    #[inline(always)]
    pub const fn new(from: Square, to: Square, flag: u16) -> Move {
        Move((from.0 as u16) | ((to.0 as u16) << 6) | (flag << 12))
    }

    #[inline(always)]
    pub const fn from(self) -> Square {
        Square((self.0 & 63) as u8)
    }

    #[inline(always)]
    pub const fn to(self) -> Square {
        Square(((self.0 >> 6) & 63) as u8)
    }

    #[inline(always)]
    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn is_some(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        self.0 & 0x4000 != 0
    }

    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        self.0 & 0x8000 != 0
    }

    #[inline(always)]
    pub const fn is_ep(self) -> bool {
        self.flag() == flags::EP_CAPTURE
    }

    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        matches!(self.flag(), flags::KING_CASTLE | flags::QUEEN_CASTLE)
    }

    #[inline(always)]
    pub const fn is_double_push(self) -> bool {
        self.flag() == flags::DOUBLE_PUSH
    }

    /// Only meaningful when `is_promotion()`.
    #[inline(always)]
    pub const fn promo_piece(self) -> PieceType {
        PieceType::from_index(1 + (self.flag() & 3) as usize)
    }

    /// A move is "quiet" if it neither captures nor promotes — the class that
    /// gets history-ordered and late-move-reduced.
    #[inline(always)]
    pub const fn is_quiet(self) -> bool {
        self.0 & 0xC000 == 0
    }

    pub fn to_uci(self) -> String {
        if self.is_none() {
            return "0000".to_string();
        }
        let mut s = format!("{}{}", self.from(), self.to());
        if self.is_promotion() {
            s.push(self.promo_piece().char());
        }
        s
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

/// Fixed-capacity, allocation-free move list.
///
/// 256 is comfortably above the maximum number of legal moves in any reachable
/// position (the known maximum is 218).
pub const MAX_MOVES: usize = 256;

#[derive(Clone)]
pub struct MoveList {
    moves: [Move; MAX_MOVES],
    /// Parallel array so ordering can score moves without touching `Move`'s
    /// packed layout.
    scores: [i32; MAX_MOVES],
    len: usize,
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl MoveList {
    #[inline]
    pub fn new() -> MoveList {
        MoveList {
            moves: [Move::NONE; MAX_MOVES],
            scores: [0; MAX_MOVES],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES);
        unsafe {
            *self.moves.get_unchecked_mut(self.len) = m;
        }
        self.len += 1;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> Move {
        debug_assert!(i < self.len);
        unsafe { *self.moves.get_unchecked(i) }
    }

    #[inline(always)]
    pub fn set_score(&mut self, i: usize, s: i32) {
        debug_assert!(i < self.len);
        unsafe {
            *self.scores.get_unchecked_mut(i) = s;
        }
    }

    #[inline(always)]
    pub fn score(&self, i: usize) -> i32 {
        unsafe { *self.scores.get_unchecked(i) }
    }

    /// Selection sort one element into place.
    ///
    /// Deliberately not a full sort: a beta cutoff usually happens within the
    /// first few moves, so sorting the tail is wasted work. This pulls the best
    /// remaining move to index `i` in O(n) and is called lazily by the search.
    #[inline]
    pub fn pick_best(&mut self, i: usize) -> Move {
        let mut best = i;
        for j in (i + 1)..self.len {
            if self.scores[j] > self.scores[best] {
                best = j;
            }
        }
        self.moves.swap(i, best);
        self.scores.swap(i, best);
        self.moves[i]
    }

    pub fn contains(&self, m: Move) -> bool {
        self.iter().any(|x| x == m)
    }

    /// Look a move up by its UCI string. This is how the front-ends turn
    /// "e1g1" into a properly flagged castling move: rather than parsing the
    /// notation, match it against what generation already produced.
    ///
    /// The comparison ignores ASCII case. UCI says the promotion piece is
    /// lowercase ("a7a8q") but real engines emit "a7a8Q" -- Simbelmyne 1.10.0
    /// does, and rejecting it cost 28 games by forfeit in one gauntlet before
    /// anyone noticed. Nothing else in a UCI move is case-bearing, so
    /// accepting either case cannot make an illegal move look legal.
    pub fn find_uci(&self, s: &str) -> Option<Move> {
        for i in 0..self.len {
            if self.moves[i].to_uci().eq_ignore_ascii_case(s) {
                return Some(self.moves[i]);
            }
        }
        None
    }

    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        (0..self.len).map(move |i| self.moves[i])
    }
}
