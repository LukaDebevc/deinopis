//! Phase-2 probe P1: what does quiescence stand pat on?
//!
//! `evalstats` measured captures at eval call sites. This measures checks:
//! how often a qsearch stand-pat happens while in check (evasions already
//! cover those) versus with a legal *quiet* check available (the blind spot
//! ORDERING.md Phase 2 is about). Read-only: it generates moves and tests
//! check-giving, which is far too expensive for the hot path.
//!
//! Compiled out unless `--features checkstats`. Like `evalstats`, the record
//! sites sit inside `search.rs` behind the same flag, so the default build is
//! bit-identical (bench fingerprint must read 219127 with m1-a1).

use crate::board::Board;
use crate::movegen::{generate, GenType};
use crate::chess_move::MoveList;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub static CALLS: AtomicU64 = AtomicU64::new(0);
pub static IN_CHECK: AtomicU64 = AtomicU64::new(0);
pub static QUIET_CHECK_AVAIL: AtomicU64 = AtomicU64::new(0);
pub static QUIET_CHECKS: AtomicU64 = AtomicU64::new(0);
// P2: the refutation probe below. PROBED counts q-nodes that searched
// captures without failing high (stand-pat fail-highs return before the
// hook, so they are out of scope); AVAIL/REFUTED split qd==0 (what Q2's
// first-ply cap would reach) from deeper.
pub static PROBED: AtomicU64 = AtomicU64::new(0);
pub static PROBED_D0: AtomicU64 = AtomicU64::new(0);
pub static AVAIL: AtomicU64 = AtomicU64::new(0);
pub static AVAIL_D0: AtomicU64 = AtomicU64::new(0);
pub static REFUTED: AtomicU64 = AtomicU64::new(0);
pub static REFUTED_D0: AtomicU64 = AtomicU64::new(0);
pub static REPLIES: AtomicU64 = AtomicU64::new(0);
pub static REPLIES_SEE_OK: AtomicU64 = AtomicU64::new(0);

/// Call at each qsearch stand-pat, before captures are generated.
pub fn record(b: &Board) {
    CALLS.fetch_add(1, Relaxed);
    if b.in_check() {
        IN_CHECK.fetch_add(1, Relaxed);
        return;
    }
    let mut moves = MoveList::new();
    generate(b, GenType::All, &mut moves);
    let mut n = 0u64;
    for i in 0..moves.len() {
        let mv = moves.get(i);
        if !mv.is_quiet() {
            continue;
        }
        if b.make_move(mv).in_check() {
            n += 1;
        }
    }
    if n > 0 {
        QUIET_CHECK_AVAIL.fetch_add(1, Relaxed);
        QUIET_CHECKS.fetch_add(n, Relaxed);
    }
}

/// P2 hook: one probed node. `has_check` = a legal quiet check exists.
pub fn record_probe(qd: i32, has_check: bool) {
    PROBED.fetch_add(1, Relaxed);
    if qd == 0 {
        PROBED_D0.fetch_add(1, Relaxed);
    }
    if has_check {
        AVAIL.fetch_add(1, Relaxed);
        if qd == 0 {
            AVAIL_D0.fetch_add(1, Relaxed);
        }
    }
}

/// P2 hook: one refutation reply searched. `see_ok` = the check passes SEE>=0
/// (what Q2's prune would keep).
pub fn record_reply(see_ok: bool) {
    REPLIES.fetch_add(1, Relaxed);
    if see_ok {
        REPLIES_SEE_OK.fetch_add(1, Relaxed);
    }
}

/// P2 hook: the node's return stands or falls.
pub fn record_refuted(qd: i32, refuted: bool) {
    if refuted {
        REFUTED.fetch_add(1, Relaxed);
        if qd == 0 {
            REFUTED_D0.fetch_add(1, Relaxed);
        }
    }
}

pub fn report() {
    let n = CALLS.load(Relaxed);
    if n == 0 {
        return;
    }
    let pct = |x: u64| 100.0 * x as f64 / n as f64;
    let in_check = IN_CHECK.load(Relaxed);
    let avail = QUIET_CHECK_AVAIL.load(Relaxed);
    // `avail` excludes in-check positions by construction (early return
    // above), so it IS the blind spot: stand-pats with a quiet check the
    // search never generates.
    println!(
        "\nqsearch nodes checked for quiet checks ({n} q-nodes)\n\
         {:<30}{:>10}\n\
         {:<30}{:>9.1}%\n\
         {:<30}{:>9.1}%  ({:.2} available per such node)",
        "condition", "share",
        "in check (evasions cover, no stand-pat)",
        pct(in_check),
        "stand-pat with quiet check available (blind spot)",
        pct(avail),
        QUIET_CHECKS.load(Relaxed) as f64 / avail.max(1) as f64,
    );
    let probed = PROBED.load(Relaxed);
    if probed > 0 {
        let pctp = |x: u64| 100.0 * x as f64 / probed as f64;
        let d0 = PROBED_D0.load(Relaxed);
        let av = AVAIL.load(Relaxed);
        let av0 = AVAIL_D0.load(Relaxed);
        let rf = REFUTED.load(Relaxed);
        let rf0 = REFUTED_D0.load(Relaxed);
        let rep = REPLIES.load(Relaxed);
        println!(
            "\nP2 refutation probe ({probed} q-nodes past stand-pat, {d0} at qd==0)\n\
             {:<46}{:>9.1}%  ({:.2} replies/node, {:.1}% pass SEE)\n\
             {:<46}{:>9.1}%\n\
             qd==0 split: {:.1}% avail, {:.1}% refuted",
            "nodes with a quiet check available",
            pctp(av),
            rep as f64 / probed as f64,
            100.0 * REPLIES_SEE_OK.load(Relaxed) as f64 / rep.max(1) as f64,
             "nodes whose return a quiet check refutes (Q2 upper bound)",
            pctp(rf),
            100.0 * av0 as f64 / d0.max(1) as f64,
            100.0 * rf0 as f64 / d0.max(1) as f64,
        );
    }
}
