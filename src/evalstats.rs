//! What the eval is actually called on, versus what it was trained on.
//!
//! The training filter (`nnue/extract`, `keep()`) keeps only positions whose
//! *best move* is a quiet non-capture, on the grounds that a search score
//! taken mid-exchange describes the tactic rather than the position. But the
//! engine calls `evaluate()` as the qsearch stand-pat, *before* captures are
//! generated, and again at every non-check interior node for pruning. So the
//! distribution at inference is not the distribution we trained on, and this
//! module measures the gap instead of guessing at it.
//!
//! Compiled out unless `--features evalstats`: the classification generates
//! moves and runs SEE, which would be absurd in the hot path.

use crate::board::Board;
use crate::eval::see;
use crate::movegen::{generate, GenType};
use crate::chess_move::MoveList;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// 0 = qsearch stand-pat, 1 = main-search static eval.
pub const SITES: usize = 2;
pub static CALLS: [AtomicU64; SITES] = [AtomicU64::new(0), AtomicU64::new(0)];
pub static IN_CHECK: [AtomicU64; SITES] = [AtomicU64::new(0), AtomicU64::new(0)];
pub static ANY_CAPTURE: [AtomicU64; SITES] = [AtomicU64::new(0), AtomicU64::new(0)];
pub static GOOD_CAPTURE: [AtomicU64; SITES] = [AtomicU64::new(0), AtomicU64::new(0)];
pub static WINNING_CAPTURE: [AtomicU64; SITES] = [AtomicU64::new(0), AtomicU64::new(0)];

pub fn record(b: &Board, site: usize) {
    CALLS[site].fetch_add(1, Relaxed);
    if b.in_check() {
        IN_CHECK[site].fetch_add(1, Relaxed);
    }
    let mut moves = MoveList::new();
    generate(b, GenType::Captures, &mut moves);
    let mut any = false;
    let mut good = false;
    let mut winning = false;
    for i in 0..moves.len() {
        let mv = moves.get(i);
        // GenType::Captures also yields queen promotions, which are not
        // captures; the training filter rejects those on a separate clause.
        if b.piece_at(mv.to()).is_none() {
            continue;
        }
        any = true;
        // SEE >= 0: qsearch would search it. SEE >= 100: it wins material
        // outright, which is the case a static eval provably cannot express.
        if see(b, mv, 0) {
            good = true;
            if see(b, mv, 100) {
                winning = true;
            }
        }
    }
    if any {
        ANY_CAPTURE[site].fetch_add(1, Relaxed);
    }
    if good {
        GOOD_CAPTURE[site].fetch_add(1, Relaxed);
    }
    if winning {
        WINNING_CAPTURE[site].fetch_add(1, Relaxed);
    }
}

pub fn report() {
    let names = ["qsearch stand-pat", "main-search static"];
    let total: u64 = CALLS.iter().map(|c| c.load(Relaxed)).sum();
    println!("\neval call sites ({total} calls)");
    println!(
        "{:<20}{:>14}{:>9}{:>12}{:>12}{:>12}",
        "site", "calls", "share", "any capt", "SEE>=0", "SEE>=100"
    );
    for s in 0..SITES {
        let n = CALLS[s].load(Relaxed);
        if n == 0 {
            continue;
        }
        let pct = |x: u64| 100.0 * x as f64 / n as f64;
        println!(
            "{:<20}{:>14}{:>8.1}%{:>11.1}%{:>11.1}%{:>11.1}%",
            names[s],
            n,
            100.0 * n as f64 / total as f64,
            pct(ANY_CAPTURE[s].load(Relaxed)),
            pct(GOOD_CAPTURE[s].load(Relaxed)),
            pct(WINNING_CAPTURE[s].load(Relaxed)),
        );
    }
    let cap: u64 = ANY_CAPTURE.iter().map(|c| c.load(Relaxed)).sum();
    let good: u64 = GOOD_CAPTURE.iter().map(|c| c.load(Relaxed)).sum();
    let win: u64 = WINNING_CAPTURE.iter().map(|c| c.load(Relaxed)).sum();
    println!(
        "\noverall: {:.1}% of eval calls have a capture available, {:.1}% a \
         non-losing one, {:.1}% one that wins material outright",
        100.0 * cap as f64 / total as f64,
        100.0 * good as f64 / total as f64,
        100.0 * win as f64 / total as f64,
    );
}
