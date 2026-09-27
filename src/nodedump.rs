//! The nodes where the search decides to prune, written out so an uncertainty
//! head can be trained on the positions it would actually be read at.
//!
//! RFP, null move and razoring all compare the static eval with a bound, at
//! non-PV nodes that are not in check. A head that says "this eval is
//! unreliable" is worth something only there, and those nodes are not game
//! positions (the tree is thinner in material and lives mid-exchange), so
//! they are sampled from the tree itself: one line per sampled node,
//! `fen<TAB>depth<TAB>static<TAB>alpha<TAB>beta<TAB>improving<TAB>ply`,
//! scores from the side to move. `chess nodelabel` reads these back.
//!
//! Armed by environment rather than by a flag, because the positions come
//! from real games and `chess match` launches the engines: `CHESS_NODEDUMP`
//! is a path prefix (each process appends its pid), `CHESS_NODEDUMP_EVERY`
//! the sampling period (default 5000). Compiled out unless
//! `--features nodedump`; each line is written straight through, since a
//! match ends by killing its engines and nothing buffered would survive.

use crate::board::Board;
use crate::eval::Score;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

static OUT: OnceLock<Option<Mutex<File>>> = OnceLock::new();
static EVERY: AtomicU64 = AtomicU64::new(5000);
static RNG: AtomicU64 = AtomicU64::new(0);

fn out() -> &'static Option<Mutex<File>> {
    OUT.get_or_init(|| {
        let prefix = std::env::var("CHESS_NODEDUMP").ok()?;
        if let Some(e) = std::env::var("CHESS_NODEDUMP_EVERY").ok().and_then(|s| s.parse().ok()) {
            EVERY.store(std::cmp::max(e, 1), Relaxed);
        }
        let pid = std::process::id() as u64;
        // Seeded by pid so the two engines of one game do not sample in step.
        RNG.store(pid.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1, Relaxed);
        let path = format!("{prefix}.{pid}.tsv");
        File::create(&path).ok().map(Mutex::new)
    })
}

fn next_rand() -> u64 {
    let mut x = RNG.load(Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    RNG.store(x, Relaxed);
    x
}

/// Called at the head of the whole-node pruning block. Random rather than
/// one-in-N, so a periodic tree cannot alias with the sampler.
pub fn record(b: &Board, depth: i32, static_eval: Score, alpha: Score, beta: Score, improving: bool, ply: usize) {
    let Some(f) = out() else { return };
    if next_rand() % EVERY.load(Relaxed) != 0 {
        return;
    }
    let line = format!(
        "{}\t{depth}\t{static_eval}\t{alpha}\t{beta}\t{}\t{ply}\n",
        b.to_fen(),
        improving as u8
    );
    let _ = f.lock().unwrap().write_all(line.as_bytes());
}
