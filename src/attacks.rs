//! Attack generation.
//!
//! Leaper tables (pawn/knight/king) are `const fn`-generated and live in
//! `.rodata`. Slider attacks use fancy magic bitboards whose magic constants are
//! *searched for at startup* rather than hardcoded — it costs a few ms and
//! removes a 128-entry table of unexplainable numbers from the source.
//!
//! `attackers_to()` is the primitive the rest of the engine is built on: SEE,
//! check detection, recapture detection and defended-square move ordering are
//! all one call to it. That is why we don't maintain incremental attack tables —
//! this answers the same question lazily, only where the search actually asks.
//!
//! Call [`init`] once before anything else. [`crate::init`] does that for you.

use crate::bitboard::Bitboard;
use crate::types::{Color, PieceType, Square};
use std::cell::UnsafeCell;
use std::sync::Once;

// ---------------------------------------------------------------- leapers

const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const KING_DELTAS: [(i8, i8); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];
const ROOK_DELTAS: [(i8, i8); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
const BISHOP_DELTAS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, -1), (-1, 1)];

const fn leaper_table(deltas: &[(i8, i8); 8]) -> [Bitboard; 64] {
    let mut table = [Bitboard::EMPTY; 64];
    let mut i = 0;
    while i < 64 {
        let sq = Square(i as u8);
        let mut bb = 0u64;
        let mut d = 0;
        while d < 8 {
            match sq.offset(deltas[d].0, deltas[d].1) {
                Some(to) => bb |= 1u64 << to.0,
                None => {}
            }
            d += 1;
        }
        table[i] = Bitboard(bb);
        i += 1;
    }
    table
}

const fn pawn_table() -> [[Bitboard; 64]; 2] {
    let mut table = [[Bitboard::EMPTY; 64]; 2];
    let mut c = 0;
    while c < 2 {
        let dr: i8 = if c == 0 { 1 } else { -1 };
        let mut i = 0;
        while i < 64 {
            let sq = Square(i as u8);
            let mut bb = 0u64;
            match sq.offset(-1, dr) {
                Some(to) => bb |= 1u64 << to.0,
                None => {}
            }
            match sq.offset(1, dr) {
                Some(to) => bb |= 1u64 << to.0,
                None => {}
            }
            table[c][i] = Bitboard(bb);
            i += 1;
        }
        c += 1;
    }
    table
}

static KNIGHT: [Bitboard; 64] = leaper_table(&KNIGHT_DELTAS);
static KING: [Bitboard; 64] = leaper_table(&KING_DELTAS);
static PAWN: [[Bitboard; 64]; 2] = pawn_table();

#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    unsafe { *KNIGHT.get_unchecked(sq.index()) }
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    unsafe { *KING.get_unchecked(sq.index()) }
}

#[inline(always)]
pub fn pawn_attacks(c: Color, sq: Square) -> Bitboard {
    unsafe { *PAWN.get_unchecked(c.index()).get_unchecked(sq.index()) }
}

// ---------------------------------------------------------------- sliders

/// Total entries across all 64 sub-tables: 102400 for rooks, 5248 for bishops.
/// 861 KB of `.bss` — sized exactly, so a wrong magic search trips the assert
/// in `init` rather than silently overlapping two squares' tables.
const ROOK_TABLE_SIZE: usize = 102_400;
const BISHOP_TABLE_SIZE: usize = 5_248;
const SLIDER_TABLE_SIZE: usize = ROOK_TABLE_SIZE + BISHOP_TABLE_SIZE;

#[derive(Clone, Copy)]
struct Magic {
    /// Relevant-occupancy mask: the ray squares excluding the board edge, since
    /// a blocker on the last square of a ray blocks nothing beyond it.
    mask: u64,
    magic: u64,
    /// Base index of this square's sub-table within the flat slider table.
    offset: u32,
    /// `64 - popcount(mask)`.
    shift: u32,
}

impl Magic {
    const EMPTY: Magic = Magic {
        mask: 0,
        magic: 0,
        offset: 0,
        shift: 0,
    };

    #[inline(always)]
    fn index(&self, occ: Bitboard) -> usize {
        let relevant = occ.0 & self.mask;
        self.offset as usize + ((relevant.wrapping_mul(self.magic)) >> self.shift) as usize
    }
}

struct Tables {
    rook: [Magic; 64],
    bishop: [Magic; 64],
    slider: [Bitboard; SLIDER_TABLE_SIZE],
    /// Squares strictly between two aligned squares; empty if not aligned.
    between: [[Bitboard; 64]; 64],
    /// The full rank/file/diagonal through two aligned squares, both endpoints
    /// included; empty if not aligned. Used to test whether a pinned piece is
    /// staying on its pin ray.
    line: [[Bitboard; 64]; 64],
}

impl Tables {
    const EMPTY: Tables = Tables {
        rook: [Magic::EMPTY; 64],
        bishop: [Magic::EMPTY; 64],
        slider: [Bitboard::EMPTY; SLIDER_TABLE_SIZE],
        between: [[Bitboard::EMPTY; 64]; 64],
        line: [[Bitboard::EMPTY; 64]; 64],
    };
}

struct SyncCell<T>(UnsafeCell<T>);
// SAFETY: written exactly once inside `Once::call_once` in `init()`, read-only
// thereafter. Every public accessor debug-asserts that `init()` has run.
unsafe impl<T> Sync for SyncCell<T> {}

static TABLES: SyncCell<Tables> = SyncCell(UnsafeCell::new(Tables::EMPTY));
static INIT: Once = Once::new();

#[inline(always)]
fn tables() -> &'static Tables {
    debug_assert!(INIT.is_completed(), "chess::init() was not called");
    // SAFETY: see the `unsafe impl Sync` above.
    unsafe { &*TABLES.0.get() }
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let t = tables();
    let m = unsafe { t.rook.get_unchecked(sq.index()) };
    unsafe { *t.slider.get_unchecked(m.index(occ)) }
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let t = tables();
    let m = unsafe { t.bishop.get_unchecked(sq.index()) };
    unsafe { *t.slider.get_unchecked(m.index(occ)) }
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

/// Attacks of a non-pawn piece. Pawns need a colour, so they have their own fn.
#[inline(always)]
pub fn piece_attacks(pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard {
    match pt {
        PieceType::Knight => knight_attacks(sq),
        PieceType::Bishop => bishop_attacks(sq, occ),
        PieceType::Rook => rook_attacks(sq, occ),
        PieceType::Queen => queen_attacks(sq, occ),
        PieceType::King => king_attacks(sq),
        PieceType::Pawn => panic!("piece_attacks: pawns need a colour"),
    }
}

#[inline(always)]
pub fn between(a: Square, b: Square) -> Bitboard {
    unsafe {
        *tables()
            .between
            .get_unchecked(a.index())
            .get_unchecked(b.index())
    }
}

#[inline(always)]
pub fn line(a: Square, b: Square) -> Bitboard {
    unsafe {
        *tables()
            .line
            .get_unchecked(a.index())
            .get_unchecked(b.index())
    }
}

/// True if all three squares share a rank, file or diagonal.
#[inline(always)]
pub fn aligned(a: Square, b: Square, c: Square) -> bool {
    line(a, b).contains(c)
}

// ---------------------------------------------------------------- init

/// Walk each ray from `sq`, stopping on (and including) the first blocker.
/// Only used at startup to fill the magic tables.
fn slow_slider(sq: Square, occ: Bitboard, deltas: &[(i8, i8); 4]) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    for &(df, dr) in deltas {
        let mut cur = sq;
        while let Some(next) = cur.offset(df, dr) {
            bb.set(next);
            if occ.contains(next) {
                break;
            }
            cur = next;
        }
    }
    bb
}

/// Ray squares excluding the final square of each ray — a blocker there cannot
/// affect anything, so it need not be part of the index.
fn relevant_mask(sq: Square, deltas: &[(i8, i8); 4]) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    for &(df, dr) in deltas {
        let mut cur = sq;
        while let Some(next) = cur.offset(df, dr) {
            if next.offset(df, dr).is_none() {
                break;
            }
            bb.set(next);
            cur = next;
        }
    }
    bb
}

/// xorshift64* — fixed seed, so the magics found are identical on every run and
/// every machine. Reproducible builds matter when a perft mismatch has to be
/// chased down.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Magics need few set bits — a dense multiplier smears the index bits and
    /// essentially never yields a collision-free mapping.
    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

/// Search for a magic for one square and fill its sub-table in place.
/// Returns the number of entries consumed.
fn build_square(
    sq: Square,
    deltas: &[(i8, i8); 4],
    offset: u32,
    table: &mut [Bitboard],
    rng: &mut Rng,
) -> (Magic, usize) {
    let mask = relevant_mask(sq, deltas);
    let bits = mask.count();
    let size = 1usize << bits;
    let shift = 64 - bits;

    // Reference: the true attack set for every blocker configuration.
    let occs: Vec<Bitboard> = mask.subsets().collect();
    debug_assert_eq!(occs.len(), size);
    let refs: Vec<Bitboard> = occs
        .iter()
        .map(|&occ| slow_slider(sq, occ, deltas))
        .collect();

    let mut scratch = vec![Bitboard::EMPTY; size];
    let mut epoch = vec![0u32; size];
    let mut cur_epoch = 0u32;

    loop {
        let magic = rng.sparse();
        // Cheap reject: a magic must at least spread the mask's high bits.
        if (mask.0.wrapping_mul(magic) >> 56).count_ones() < 6 {
            continue;
        }
        cur_epoch += 1;
        let mut ok = true;
        for i in 0..size {
            let idx = ((occs[i].0.wrapping_mul(magic)) >> shift) as usize;
            if epoch[idx] != cur_epoch {
                epoch[idx] = cur_epoch;
                scratch[idx] = refs[i];
            } else if scratch[idx] != refs[i] {
                // Constructive collisions (same attack set) are fine; this is a
                // real one.
                ok = false;
                break;
            }
        }
        if ok {
            table[..size].copy_from_slice(&scratch[..size]);
            return (
                Magic {
                    mask: mask.0,
                    magic,
                    offset,
                    shift,
                },
                size,
            );
        }
    }
}

/// Build all runtime tables. Idempotent and thread-safe; costs a few ms.
pub fn init() {
    INIT.call_once(|| {
        // SAFETY: `call_once` guarantees we are the only accessor, and no
        // reader can observe the tables before this returns.
        let t: &mut Tables = unsafe { &mut *TABLES.0.get() };

        let mut rng = Rng(0x246C_CB2D_3B40_2853);
        let mut offset: u32 = 0;

        for i in 0..64 {
            let sq = Square::from_index(i);
            let (m, used) = build_square(
                sq,
                &ROOK_DELTAS,
                offset,
                &mut t.slider[offset as usize..],
                &mut rng,
            );
            t.rook[i] = m;
            offset += used as u32;
        }
        assert_eq!(offset as usize, ROOK_TABLE_SIZE, "rook table size mismatch");

        for i in 0..64 {
            let sq = Square::from_index(i);
            let (m, used) = build_square(
                sq,
                &BISHOP_DELTAS,
                offset,
                &mut t.slider[offset as usize..],
                &mut rng,
            );
            t.bishop[i] = m;
            offset += used as u32;
        }
        assert_eq!(offset as usize, SLIDER_TABLE_SIZE, "slider table size mismatch");

        // between/line depend on the magics above, so they are built second.
        for a in 0..64 {
            let sa = Square::from_index(a);
            for b in 0..64 {
                let sb = Square::from_index(b);
                if a == b {
                    continue;
                }
                let bb_a = Bitboard::from_square(sa);
                let bb_b = Bitboard::from_square(sb);

                for deltas in [&ROOK_DELTAS, &BISHOP_DELTAS] {
                    if !slow_slider(sa, Bitboard::EMPTY, deltas).contains(sb) {
                        continue;
                    }
                    // Squares each sees toward the other, intersected, is
                    // exactly the open segment between them.
                    t.between[a][b] =
                        slow_slider(sa, bb_b, deltas) & slow_slider(sb, bb_a, deltas);
                    t.line[a][b] = (slow_slider(sa, Bitboard::EMPTY, deltas)
                        & slow_slider(sb, Bitboard::EMPTY, deltas))
                        | bb_a
                        | bb_b;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::squares::*;

    fn setup() {
        init();
    }

    #[test]
    fn leapers() {
        setup();
        assert_eq!(knight_attacks(A1).count(), 2);
        assert_eq!(knight_attacks(D4).count(), 8);
        assert_eq!(knight_attacks(H8).count(), 2);
        assert_eq!(king_attacks(A1).count(), 3);
        assert_eq!(king_attacks(D4).count(), 8);
        assert_eq!(pawn_attacks(Color::White, A2).count(), 1);
        assert_eq!(pawn_attacks(Color::White, D2).count(), 2);
        assert!(pawn_attacks(Color::White, D2).contains(C3));
        assert!(pawn_attacks(Color::White, D2).contains(E3));
        assert!(pawn_attacks(Color::Black, D7).contains(C6));
        // No wraparound off the H file.
        assert!(!pawn_attacks(Color::White, H2).contains(A3));
        assert!(!knight_attacks(H1).contains(A2));
    }

    #[test]
    fn sliders_match_reference() {
        setup();
        let mut rng = Rng(0xDEAD_BEEF_CAFE_1234);
        for i in 0..64 {
            let sq = Square::from_index(i);
            for _ in 0..2000 {
                // Random occupancies, biased sparse so rays actually run.
                let occ = Bitboard(rng.next() & rng.next());
                assert_eq!(
                    rook_attacks(sq, occ),
                    slow_slider(sq, occ, &ROOK_DELTAS),
                    "rook mismatch at {sq} occ 0x{:016X}",
                    occ.0
                );
                assert_eq!(
                    bishop_attacks(sq, occ),
                    slow_slider(sq, occ, &BISHOP_DELTAS),
                    "bishop mismatch at {sq} occ 0x{:016X}",
                    occ.0
                );
            }
        }
    }

    #[test]
    fn empty_board_counts() {
        setup();
        // Known totals: a rook always sees 14 squares on an empty board; a
        // bishop sees 7 (corner) to 13 (centre).
        for i in 0..64 {
            let sq = Square::from_index(i);
            assert_eq!(rook_attacks(sq, Bitboard::EMPTY).count(), 14);
        }
        assert_eq!(bishop_attacks(A1, Bitboard::EMPTY).count(), 7);
        assert_eq!(bishop_attacks(D4, Bitboard::EMPTY).count(), 13);
    }

    #[test]
    fn between_and_line() {
        setup();
        assert_eq!(between(A1, A4), Bitboard::from_square(A2) | Bitboard::from_square(A3));
        assert_eq!(between(A1, B2), Bitboard::EMPTY); // adjacent: nothing between
        assert_eq!(between(A1, C3), Bitboard::from_square(B2));
        assert_eq!(between(A1, B3), Bitboard::EMPTY); // not aligned
        assert!(line(A1, A4).contains(A8));
        assert!(line(A1, A4).contains(A1));
        assert_eq!(line(A1, B3), Bitboard::EMPTY);
        assert!(aligned(A1, D4, H8));
        assert!(!aligned(A1, D4, H7));
    }
}
