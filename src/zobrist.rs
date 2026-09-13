//! Zobrist hashing. Keys are generated at startup from a fixed seed, so a key
//! means the same thing across runs and across threads — required for the
//! transposition table to be shareable and for debugging to be reproducible.

use crate::types::{Color, Piece, PieceType, Square};
use std::cell::UnsafeCell;
use std::sync::Once;

pub struct Keys {
    /// Indexed `[color][piece_type][square]`.
    pub piece: [[[u64; 64]; 6]; 2],
    pub castling: [u64; 16],
    /// Only the file matters — the rank is implied by side to move.
    pub ep_file: [u64; 8],
    pub side: u64,
}

impl Keys {
    const EMPTY: Keys = Keys {
        piece: [[[0; 64]; 6]; 2],
        castling: [0; 16],
        ep_file: [0; 8],
        side: 0,
    };
}

struct SyncCell<T>(UnsafeCell<T>);
unsafe impl<T> Sync for SyncCell<T> {}

static KEYS: SyncCell<Keys> = SyncCell(UnsafeCell::new(Keys::EMPTY));
static INIT: Once = Once::new();

#[inline(always)]
pub fn keys() -> &'static Keys {
    debug_assert!(INIT.is_completed(), "chess::init() was not called");
    unsafe { &*KEYS.0.get() }
}

#[inline(always)]
pub fn piece_key(p: Piece, sq: Square) -> u64 {
    let k = keys();
    unsafe {
        *k.piece
            .get_unchecked(p.color().index())
            .get_unchecked(p.piece_type().index())
            .get_unchecked(sq.index())
    }
}

pub fn init() {
    INIT.call_once(|| {
        let k: &mut Keys = unsafe { &mut *KEYS.0.get() };
        let mut s: u64 = 0x0F56_1234_9ABC_DEF1;
        let mut next = || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            s.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for c in 0..2 {
            for pt in 0..6 {
                for sq in 0..64 {
                    k.piece[c][pt][sq] = next();
                }
            }
        }
        for i in 0..16 {
            k.castling[i] = next();
        }
        for i in 0..8 {
            k.ep_file[i] = next();
        }
        k.side = next();
    });
}

/// Convenience for eval/correction-history buckets later on.
pub fn pawn_key_component(c: Color, sq: Square) -> u64 {
    piece_key(Piece::new(c, PieceType::Pawn), sq)
}
