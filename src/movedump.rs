//! What the tree actually plays, written out so a routing rule can be priced
//! on it instead of on a guess.
//!
//! `nnue/refresh.py` prices a rule by taking training positions, enumerating
//! their legal moves, and reweighting the per-move rates by the tree's
//! *move-class* mix (69% quiet / 30% capture / 1% promo, from `chess bench`).
//! That corrects one bias and leaves three:
//!
//! * **Which piece moves.** Move ordering searches captures, killers and
//!   history-rich moves first, and most interior nodes make one move before a
//!   beta cutoff. King moves are history-poor and ordered late, so the tree
//!   almost certainly plays fewer of them than a class-reweighted legal-move
//!   mix implies — which would mean every king-reading rule is priced HIGH.
//! * **Which captures.** MVV-LVA order and the SEE filter mean the tree's
//!   captures take valuable pieces more often than a uniform one does, which
//!   pushes material-reading rules the other way.
//! * **Which positions.** The tree's nodes are not the training positions.
//!   Quiescence lives mid-exchange and the deep plies skew towards thin
//!   material, and both visit mass and transition rates depend on that.
//!
//! Guessing at the size of any of those is exactly what this project does not
//! do, so: dump the real (position, move) pairs and price on them. One line
//! per sampled made move, `parent_fen<TAB>child_fen<TAB>uci<TAB>site`, site
//! 0 = main search, 1 = quiescence. Both FENs, because the pricing then never
//! has to reimplement en passant, castling or promotion in feature space --
//! which is where the last pricing bug lived. Compiled out unless
//! `--features movedump`.

use crate::board::Board;
use crate::chess_move::{Move, MoveList};
use crate::movegen::{generate, GenType};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

static ON: AtomicBool = AtomicBool::new(false);
static SEEN: AtomicU64 = AtomicU64::new(0);
static WROTE: AtomicU64 = AtomicU64::new(0);
static EVERY: AtomicU64 = AtomicU64::new(1);
static OUT: Mutex<Option<BufWriter<File>>> = Mutex::new(None);
static RAND: AtomicBool = AtomicBool::new(false);
static RNG: AtomicU64 = AtomicU64::new(0x2545_F491_4F6C_DD1D);

/// The control stream: same sampled position, but a move drawn uniformly from
/// the legal list instead of the one the tree chose. The difference between
/// the two streams is move ordering and pruning, with the positions held
/// fixed -- which is the only way to attribute a price gap to either.
pub fn with_random(on: bool) {
    RAND.store(on, Relaxed);
}

fn next_rand() -> u64 {
    let mut x = RNG.load(Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    RNG.store(x, Relaxed);
    x
}

/// Sample one made move in `every`. Deterministic, so two runs of the same
/// search dump the same stream.
pub fn open(path: &str, every: u64) -> Result<(), String> {
    let f = File::create(path).map_err(|e| format!("{path}: {e}"))?;
    *OUT.lock().unwrap() = Some(BufWriter::new(f));
    EVERY.store(every.max(1), Relaxed);
    SEEN.store(0, Relaxed);
    WROTE.store(0, Relaxed);
    ON.store(true, Relaxed);
    Ok(())
}

/// `b` is the position BEFORE `mv`; the pair is what an accumulator update
/// would see. Called from both make-move sites in the search.
pub fn record(b: &Board, nb: &Board, mv: Move, site: usize) {
    if !ON.load(Relaxed) {
        return;
    }
    if SEEN.fetch_add(1, Relaxed) % EVERY.load(Relaxed) != 0 {
        return;
    }
    let rnd = if RAND.load(Relaxed) {
        let mut l = MoveList::new();
        generate(b, GenType::All, &mut l);
        if l.is_empty() {
            None
        } else {
            let m = l.get((next_rand() % l.len() as u64) as usize);
            Some((b.make_move(m), m))
        }
    } else {
        None
    };
    if let Some(w) = OUT.lock().unwrap().as_mut() {
        let _ = writeln!(w, "{}\t{}\t{}\t{}", b.to_fen(), nb.to_fen(), mv.to_uci(), site);
        if let Some((rb, rm)) = rnd {
            let _ = writeln!(w, "{}\t{}\t{}\t2", b.to_fen(), rb.to_fen(), rm.to_uci());
        }
        WROTE.fetch_add(1, Relaxed);
    }
}

/// (made moves seen, lines written).
pub fn close() -> (u64, u64) {
    ON.store(false, Relaxed);
    if let Some(mut w) = OUT.lock().unwrap().take() {
        let _ = w.flush();
    }
    (SEEN.load(Relaxed), WROTE.load(Relaxed))
}
