//! 64-bit board sets. Bit `i` is square `i` (A1 = LSB, H8 = MSB).
//!
//! `Bitboard` is its own `Iterator`, yielding set squares low-to-high and
//! consuming the set as it goes. `for sq in board.knights(Color::White)` is
//! therefore the idiomatic loop, and compiles to `tzcnt` + `blsr`.

use crate::types::{Color, Square};
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, Not, Shl, Shr, Sub,
};

#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const ALL: Bitboard = Bitboard(!0);

    pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
    pub const FILE_H: Bitboard = Bitboard(0x8080_8080_8080_8080);
    pub const RANK_1: Bitboard = Bitboard(0x0000_0000_0000_00FF);
    pub const RANK_8: Bitboard = Bitboard(0xFF00_0000_0000_0000);
    /// The frame of the board — the squares a slider's relevant-occupancy mask
    /// excludes, since a blocker on the edge cannot block anything beyond it.
    pub const EDGES: Bitboard = Bitboard(0xFF81_8181_8181_81FF);

    #[inline(always)]
    pub const fn from_square(sq: Square) -> Bitboard {
        Bitboard(1u64 << sq.0)
    }

    #[inline(always)]
    pub const fn file(f: u8) -> Bitboard {
        Bitboard(Bitboard::FILE_A.0 << f)
    }

    #[inline(always)]
    pub const fn rank(r: u8) -> Bitboard {
        Bitboard(Bitboard::RANK_1.0 << (8 * r))
    }

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub const fn contains(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.0) != 0
    }

    /// Lowest set square. Undefined (returns A1) on an empty set — callers in
    /// the search always guard with `any()` or know the set is non-empty.
    #[inline(always)]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square(self.0.trailing_zeros() as u8)
    }

    #[inline(always)]
    pub const fn msb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square(63 - self.0.leading_zeros() as u8)
    }

    #[inline(always)]
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0 - 1;
        sq
    }

    #[inline(always)]
    pub const fn with(self, sq: Square) -> Bitboard {
        Bitboard(self.0 | (1u64 << sq.0))
    }

    #[inline(always)]
    pub const fn without(self, sq: Square) -> Bitboard {
        Bitboard(self.0 & !(1u64 << sq.0))
    }

    #[inline(always)]
    pub fn set(&mut self, sq: Square) {
        self.0 |= 1u64 << sq.0;
    }

    #[inline(always)]
    pub fn clear(&mut self, sq: Square) {
        self.0 &= !(1u64 << sq.0);
    }

    #[inline(always)]
    pub fn toggle(&mut self, sq: Square) {
        self.0 ^= 1u64 << sq.0;
    }

    /// Shift by one rank in the direction `c`'s pawns move.
    #[inline(always)]
    pub const fn forward(self, c: Color) -> Bitboard {
        match c {
            Color::White => Bitboard(self.0 << 8),
            Color::Black => Bitboard(self.0 >> 8),
        }
    }

    #[inline(always)]
    pub const fn backward(self, c: Color) -> Bitboard {
        self.forward(c.flip())
    }

    /// Diagonal pawn-push shifts. The file mask strips the wraparound that a
    /// raw `<< 9` would produce off the H file.
    #[inline(always)]
    pub const fn forward_left(self, c: Color) -> Bitboard {
        let b = Bitboard(self.0 & !Bitboard::FILE_A.0);
        match c {
            Color::White => Bitboard(b.0 << 7),
            Color::Black => Bitboard(b.0 >> 9),
        }
    }

    #[inline(always)]
    pub const fn forward_right(self, c: Color) -> Bitboard {
        let b = Bitboard(self.0 & !Bitboard::FILE_H.0);
        match c {
            Color::White => Bitboard(b.0 << 9),
            Color::Black => Bitboard(b.0 >> 7),
        }
    }

    /// Iterate the subsets of `self` in Carry-Rippler order, hitting each of
    /// the `2^popcount` subsets exactly once. Used to enumerate every blocker
    /// configuration when building the slider tables.
    pub fn subsets(self) -> impl Iterator<Item = Bitboard> {
        let mask = self.0;
        let mut sub: u64 = 0;
        let mut done = false;
        std::iter::from_fn(move || {
            if done {
                return None;
            }
            let cur = sub;
            sub = sub.wrapping_sub(mask) & mask;
            if sub == 0 {
                done = true;
            }
            Some(Bitboard(cur))
        })
    }
}

impl Iterator for Bitboard {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pop_lsb())
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.count() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for Bitboard {}

macro_rules! binop {
    ($trait:ident, $fn:ident, $assign_trait:ident, $assign_fn:ident, $op:tt) => {
        impl $trait for Bitboard {
            type Output = Bitboard;
            #[inline(always)]
            fn $fn(self, rhs: Bitboard) -> Bitboard { Bitboard(self.0 $op rhs.0) }
        }
        impl $assign_trait for Bitboard {
            #[inline(always)]
            fn $assign_fn(&mut self, rhs: Bitboard) { self.0 = self.0 $op rhs.0; }
        }
    };
}

binop!(BitAnd, bitand, BitAndAssign, bitand_assign, &);
binop!(BitOr, bitor, BitOrAssign, bitor_assign, |);
binop!(BitXor, bitxor, BitXorAssign, bitxor_assign, ^);

impl Not for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

/// Set subtraction: `a - b` is `a & !b`.
impl Sub for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn sub(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & !rhs.0)
    }
}

impl Mul<u64> for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn mul(self, rhs: u64) -> Bitboard {
        Bitboard(self.0.wrapping_mul(rhs))
    }
}

impl Shl<u32> for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Bitboard {
        Bitboard(self.0 << rhs)
    }
}

impl Shr<u32> for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Bitboard {
        Bitboard(self.0 >> rhs)
    }
}

impl From<Square> for Bitboard {
    #[inline(always)]
    fn from(sq: Square) -> Bitboard {
        Bitboard::from_square(sq)
    }
}

impl std::fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..8 {
                let c = if self.contains(Square::new(file, rank)) {
                    'X'
                } else {
                    '.'
                };
                write!(f, "{c} ")?;
            }
            writeln!(f)?;
        }
        write!(f, "  a b c d e f g h    0x{:016X}", self.0)
    }
}
