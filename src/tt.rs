//! Shared transposition table.
//!
//! Designed for Lazy SMP from the start: every thread reads and writes the same
//! array with no locking. Races are handled by Hyatt's XOR trick — each slot
//! stores `key ^ data` alongside `data`, so a torn write (one field from writer
//! A, one from writer B) fails the checksum and is discarded rather than being
//! interpreted as a valid entry for the wrong position.
//!
//! The atomics use `Ordering::Relaxed`, which on x86 compiles to a plain `mov`:
//! the concurrency here is free at runtime. This is the one place where C++
//! engines commit deliberate UB (unsynchronised access to shared memory) and
//! Rust does not have to.

use crate::chess_move::Move;
use crate::eval::{Score, MATE_BOUND};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    /// Score is exact — the node was a PV node.
    Exact = 1,
    /// Score is a lower bound — the node failed high (beta cutoff).
    Lower = 2,
    /// Score is an upper bound — the node failed low.
    Upper = 3,
}

impl Bound {
    fn from_u8(v: u8) -> Bound {
        match v & 3 {
            1 => Bound::Exact,
            2 => Bound::Lower,
            3 => Bound::Upper,
            _ => Bound::None,
        }
    }
}

pub struct Hit {
    pub score: Score,
    pub eval: Score,
    pub mv: Move,
    pub depth: u8,
    pub bound: Bound,
}

/// Marker depth for quiescence stores. A quiescence score is a different
/// function from a main-search score (captures only, stand pat allowed), so
/// its bounds must never satisfy a main-search probe: the main search only
/// cuts off on `depth >= tt_depth` with `depth != Q_DEPTH`, and quiescence
/// only on `depth == Q_DEPTH`. Move ordering and the stored static eval are
/// position properties and cross freely. No real budget reaches 255 plies,
/// so the marker cannot collide with a genuine depth.
pub const Q_DEPTH: u8 = 255;

#[repr(align(16))]
struct Slot {
    /// `key ^ data`
    check: AtomicU64,
    data: AtomicU64,
}

impl Default for Slot {
    fn default() -> Slot {
        Slot {
            check: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

/// data layout: move 0..16 | score 16..32 | eval 32..48 | depth 48..56 | bound 56..58 | age 58..64
#[inline(always)]
fn pack(mv: Move, score: i16, eval: i16, depth: u8, bound: Bound, age: u8) -> u64 {
    (mv.0 as u64)
        | ((score as u16 as u64) << 16)
        | ((eval as u16 as u64) << 32)
        | ((depth as u64) << 48)
        | ((bound as u64) << 56)
        | (((age & 0x3F) as u64) << 58)
}

pub struct TranspositionTable {
    slots: Vec<Slot>,
    mask: usize,
    /// Bumped each `ucinewgame`/`go`, so stale entries are preferentially
    /// replaced without clearing the table.
    age: std::sync::atomic::AtomicU8,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> TranspositionTable {
        let mut tt = TranspositionTable {
            slots: Vec::new(),
            mask: 0,
            age: std::sync::atomic::AtomicU8::new(0),
        };
        tt.resize(mb);
        tt
    }

    pub fn resize(&mut self, mb: usize) {
        let bytes = mb.max(1) * 1024 * 1024;
        let n = (bytes / std::mem::size_of::<Slot>()).next_power_of_two() / 2;
        let n = n.max(1024);
        self.slots = (0..n).map(|_| Slot::default()).collect();
        self.mask = n - 1;
    }

    pub fn clear(&self) {
        for s in &self.slots {
            s.check.store(0, Ordering::Relaxed);
            s.data.store(0, Ordering::Relaxed);
        }
    }

    pub fn new_search(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    fn slot(&self, key: u64) -> &Slot {
        unsafe { self.slots.get_unchecked((key as usize) & self.mask) }
    }

    /// Hint the cache before the search does its (expensive) move generation,
    /// so the TT line is resident by the time we actually probe.
    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let p = self.slot(key) as *const Slot as *const i8;
            std::arch::x86_64::_mm_prefetch(p, std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = key;
    }

    /// `ply` is needed because mate scores are stored relative to the node they
    /// were found at, not the root — otherwise a "mate in 3" cached at ply 8
    /// would be read back as "mate in 3" at ply 2.
    pub fn probe(&self, key: u64, ply: usize) -> Option<Hit> {
        let s = self.slot(key);
        let data = s.data.load(Ordering::Relaxed);
        let check = s.check.load(Ordering::Relaxed);
        if check ^ data != key || data == 0 {
            return None;
        }
        let mut score = ((data >> 16) as u16) as i16 as Score;
        if score >= MATE_BOUND {
            score -= ply as Score;
        } else if score <= -MATE_BOUND {
            score += ply as Score;
        }
        Some(Hit {
            score,
            eval: ((data >> 32) as u16) as i16 as Score,
            mv: Move((data & 0xFFFF) as u16),
            depth: ((data >> 48) & 0xFF) as u8,
            bound: Bound::from_u8((data >> 56) as u8),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        key: u64,
        mv: Move,
        score: Score,
        eval: Score,
        depth: u8,
        bound: Bound,
        ply: usize,
    ) {
        let s = self.slot(key);
        let age = self.age.load(Ordering::Relaxed);

        // Replacement: keep the existing entry only if it is from this search,
        // deeper, and not an exact bound we are replacing with a better one.
        // Depths compare semantically: a quiescence store (Q_DEPTH) proves
        // less than any main-search store, so a deep main entry survives it
        // and a main store always displaces a quiescence one. (Q_DEPTH as u8
        // arithmetic would wrap here — 255 + 3 overflows — so widen first.)
        let old_data = s.data.load(Ordering::Relaxed);
        if old_data != 0 && s.check.load(Ordering::Relaxed) ^ old_data == key {
            let old_raw = ((old_data >> 48) & 0xFF) as u8;
            let old_age = ((old_data >> 58) & 0x3F) as u8;
            let sem = |d: u8| if d == Q_DEPTH { -1i32 } else { d as i32 };
            if bound != Bound::Exact && old_age == (age & 0x3F) && sem(old_raw) > sem(depth) + 3 {
                return;
            }
        }

        // Store mate scores relative to this node.
        let adj = if score >= MATE_BOUND {
            score + ply as Score
        } else if score <= -MATE_BOUND {
            score - ply as Score
        } else {
            score
        };

        // Never overwrite a good move with a null one: on a fail-low we have no
        // best move, but the previously stored one is still worth trying first.
        let keep_move = if mv.is_none() && old_data != 0 {
            Move((old_data & 0xFFFF) as u16)
        } else {
            mv
        };

        let data = pack(
            keep_move,
            adj.clamp(i16::MIN as Score, i16::MAX as Score) as i16,
            eval.clamp(i16::MIN as Score, i16::MAX as Score) as i16,
            depth,
            bound,
            age,
        );
        s.check.store(key ^ data, Ordering::Relaxed);
        s.data.store(data, Ordering::Relaxed);
    }

    /// How much memory the table actually occupies. `resize` rounds to a power
    /// of two, so this is usually *not* what was asked for, and the GUI should
    /// report what was allocated rather than what was requested.
    pub fn size_bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<Slot>()
    }

    /// Permille of slots filled, sampled over the first 1000 — this is what UCI
    /// `hashfull` expects.
    pub fn hashfull(&self) -> usize {
        self.slots
            .iter()
            .take(1000)
            .filter(|s| s.data.load(Ordering::Relaxed) != 0)
            .count()
    }
}
