//! UCI protocol front-end.
//!
//! UCI is the reason this is the primary interface rather than a nicety: it is
//! what every chess GUI speaks, and what cutechess-cli and OpenBench drive to
//! run SPRT matches. Any change we want to claim is worth Elo has to be
//! measured through this interface.

use crate::board::{Board, START_FEN};
use crate::chess_move::Move;
use crate::eval::Evaluator;
use crate::qeval::DefaultEval;
use crate::movegen::{generate, GenType};
use crate::chess_move::MoveList;
use crate::search::{go_parallel, persist_mode, prepare_td, score_to_uci, Limits, Params, SearchResult, Shared, ThreadData};
use std::io::{BufRead, Write};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::JoinHandle;

pub const NAME: &str = "Deinopis";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const AUTHOR: &str = "Luka Debevc";

/// TU-2: `--set k=v,...` (or `$CHESS_SET`) at UCI startup, so a playing
/// build can run one arm of an SPSA pair without recompiling. Parsed once;
/// `Engine::new` applies it to the default `Params`. A bad spec fails fast
/// (exit 2) rather than silently running defaults — a 30k-pair campaign
/// must not tune nothing because of a typo.
static STARTUP_PARAMS: std::sync::OnceLock<Params> = std::sync::OnceLock::new();

fn startup_params() -> Params {
    *STARTUP_PARAMS.get_or_init(|| {
        let mut spec = std::env::var("CHESS_SET").ok();
        let args: Vec<String> = std::env::args().collect();
        let mut i = 0;
        while i < args.len() {
            if args[i] == "--set" {
                spec = args.get(i + 1).cloned();
                i += 1;
            } else if let Some(v) = args[i].strip_prefix("--set=") {
                spec = Some(v.to_string());
            }
            i += 1;
        }
        let mut p = Params::default();
        if let Some(s) = spec {
            if let Err(e) = crate::tune::apply_overrides(&mut p, &s) {
                eprintln!("bad --set '{s}': {e}");
                std::process::exit(2);
            }
        }
        p
    })
}

pub struct Engine {
    pub shared: Arc<Shared>,
    pub board: Board,
    /// Zobrist keys of every position in the game so far, for repetition
    /// detection across the root.
    pub history: Vec<u64>,
    pub params: Params,
    pub threads: usize,
    hash_mb: usize,
    /// One learned table per thread, reused across moves (FX-1). Empty means
    /// "not yet sized"; `take_td` sizes and applies the persist mode before
    /// every search, and the worker thread hands the tables back when it
    /// finishes, so a table is lost only if a search panics. Cleared on
    /// `ucinewgame`, like the TT.
    td: Vec<ThreadData>,
    worker: Option<JoinHandle<Vec<ThreadData>>>,
    /// The running search has no limit of its own (`go infinite`, or a bare
    /// `go`), so only `stop` ends it.
    open_ended: bool,
    adapt: std::sync::Arc<std::sync::Mutex<crate::adaptive::AdaptiveState>>,
}

impl Engine {
    pub fn new() -> Engine {
        // adaptive::init_from_env already ran via crate::init(), but ensure params captured
        let params = crate::adaptive::global_handle().lock().map(|g| g.params.clone()).unwrap_or_default();
        Engine {
            shared: Arc::new(Shared::new(64)),
            board: Board::startpos(),
            history: Vec::new(),
            params: startup_params(),
            threads: 1,
            hash_mb: 64,
            td: Vec::new(),
            worker: None,
            open_ended: false,
            adapt: std::sync::Arc::new(std::sync::Mutex::new(crate::adaptive::AdaptiveState::new(params))),
        }
    }

    /// Apply a UCI move string to the current board, returning false if it is
    /// not legal here. Matching against generated moves is what turns "e1g1"
    /// into a castling move with the right flags.
    pub fn apply_uci_move(&mut self, s: &str) -> bool {
        let mut list = MoveList::new();
        generate(&self.board, GenType::All, &mut list);
        if let Some(m) = list.find_uci(s) {
            self.history.push(self.board.key());
            self.board = self.board.make_move(m);
            return true;
        }
        false
    }

    pub fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), String> {
        self.board = Board::from_fen(fen)?;
        self.history.clear();
        for m in moves {
            if !self.apply_uci_move(m) {
                return Err(format!("illegal move '{m}' in position command"));
            }
        }
        Ok(())
    }

    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.worker.take() {
            // `go_parallel` clears the flag when it starts, so a `stop` that
            // lands before the worker gets there would be erased and the join
            // would never return. Keep raising it until the worker is gone.
            while !h.is_finished() {
                self.shared.stop.store(true, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            // The finished search hands its learned tables back; only a
            // panicked search loses them, and the next `take_td` rebuilds.
            if let Ok(td) = h.join() {
                self.td = td;
            }
        }
    }

    /// End of input. It is not `quit`: `printf 'go depth 8\n' | chess` still
    /// owes its `bestmove`, so a search with a limit runs to it. One without a
    /// limit would never finish, so it is stopped, which also prints one.
    fn finish(&mut self) {
        if self.open_ended {
            self.stop();
        } else if let Some(h) = self.worker.take() {
            if let Ok(td) = h.join() {
                self.td = td;
            }
        }
    }

    /// Take the engine's learned tables, sized for `threads` and prepared
    /// for one more move. Node-limited searches always get fresh tables, so
    /// `go nodes` (and everything built on it: bench, the tuner, every
    /// reproducibility check) never sees another search's history. Timed
    /// searches follow `--persist-hist` (default 1 = persist-all).
    fn take_td(&mut self, limits: &Limits, threads: usize) -> Vec<ThreadData> {
        let mut td = std::mem::take(&mut self.td);
        let mode = if limits.nodes.is_some() { 0 } else { persist_mode() };
        prepare_td(&mut td, threads, mode);
        td
    }

    /// Run a search on the calling thread and return the result. Used by the
    /// web GUI and terminal play, where there is nothing to do concurrently.
    pub fn search_blocking(&mut self, limits: Limits) -> SearchResult {
        self.shared.stop.store(false, Ordering::Relaxed);
        let adapt = self.adapt.clone();
        let threads = self.search_threads(&limits);
        let mut td = self.take_td(&limits, threads);
        let r = go_parallel(
            &self.shared,
            &self.board,
            &self.history,
            &limits,
            self.params,
            threads,
            move || crate::qeval::DefaultEval::with_adaptive(adapt.clone()),
            None,
            &mut td,
        );
        self.td = td;
        r
    }

    /// Start a search on a worker thread, printing UCI `info` lines as it goes
    /// and `bestmove` at the end. Returns immediately so `stop` can be handled.
    fn go_async(&mut self, limits: Limits) {
        self.stop();
        self.shared.stop.store(false, Ordering::Relaxed);
        self.open_ended = limits.infinite
            || (limits.depth.is_none()
                && limits.nodes.is_none()
                && limits.work.is_none()
                && limits.movetime.is_none()
                && limits.time[self.board.stm().index()].is_none());
        let shared = Arc::clone(&self.shared);
        let board = self.board;
        let history = self.history.clone();
        let params = self.params;

        let threads = self.search_threads(&limits);
        let adapt = self.adapt.clone();
        // `take_td` after `stop`: the previous search (if any) handed its
        // tables back in `stop`'s join, so these are last move's tables with
        // this move's start rule applied — or fresh ones at mode 0.
        let mut td = self.take_td(&limits, threads);

        self.worker = Some(std::thread::spawn(move || {
            let mut emit = |r: &SearchResult, el: std::time::Duration, seldepth: usize| {
                let ms = el.as_millis().max(1) as u64;
                // Under SMP the played line is thread 0's, but the work done is
                // every thread's — reporting only thread 0's nodes would
                // understate nps by a factor of `threads` and make a
                // `nodes`-based comparison between builds meaningless.
                let nodes = if threads > 1 {
                    shared.nodes.load(Ordering::Relaxed).max(r.nodes)
                } else {
                    r.nodes
                };
                let nps = nodes * 1000 / ms;
                let pv: Vec<String> = r.pv.iter().map(|m| m.to_uci()).collect();
                println!(
                    "info depth {} seldepth {} score {} nodes {} nps {} hashfull {} time {} pv {}",
                    r.depth,
                    seldepth,
                    score_to_uci(r.score),
                    nodes,
                    nps,
                    shared.tt.hashfull(),
                    ms,
                    pv.join(" ")
                );
                let _ = std::io::stdout().flush();
            };
            let res = go_parallel(
                &shared,
                &board,
                &history,
                &limits,
                params,
                threads,
                move || crate::qeval::DefaultEval::with_adaptive(adapt.clone()),
                Some(&mut emit),
                &mut td,
            );
            println!("bestmove {}", res.best_move.to_uci());
            let _ = std::io::stdout().flush();
            // Hand the learned tables back; `stop`/`finish` reclaims them.
            td
        }));
    }

    /// How many threads this search may use.
    ///
    /// A node-limited search is forced to one thread. `go nodes N` is how the
    /// tuner and every reproducibility check ask for a fixed unit of work, and
    /// under Lazy SMP "N nodes" would be N nodes on *some* thread with the
    /// others racing it — a different search every run. Time-limited searches
    /// are already nondeterministic, so they lose nothing.
    fn search_threads(&self, limits: &Limits) -> usize {
        if limits.nodes.is_some() {
            1
        } else {
            self.threads
        }
    }

    fn set_option(&mut self, name: &str, value: &str) {
        // UCI only allows `setoption` while idle, and every option below
        // assumes it: "hash" swaps `self.shared`, and a worker still holding
        // the old `Arc` would never see the new stop flag, so `stop` hung.
        self.stop();
        match name.to_ascii_lowercase().as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    self.hash_mb = mb.clamp(1, 65536);
                    // Rebuilding requires exclusive access, so a fresh Shared is
                    // simpler and safer than resizing under other threads.
                    self.shared = Arc::new(Shared::new(self.hash_mb));
                }
            }
            "threads" => {
                if let Ok(t) = value.parse::<usize>() {
                    self.threads = t.clamp(1, 256);
                }
            }
            // Percent, because UCI spins are integers. 50 is a normal engine;
            // 0 makes a draw worth exactly as much as a loss.
            "drawvalue" => {
                if let Ok(v) = value.parse::<i64>() {
                    crate::eval::set_draw_value(v.clamp(0, 100) as f32 / 100.0);
                }
            }
            // Asymmetric contempt as an additive bonus: steepness as a percent
            // (100 = 1.0), scale in centipawns (0 = off), flat weighting
            // on/off. See `eval.rs`.
            "contempta" => {
                if let Ok(v) = value.parse::<i64>() {
                    crate::eval::set_contempt_a(v.max(0) as f32 / 100.0);
                }
            }
            "contemptb" => {
                if let Ok(v) = value.parse::<i64>() {
                    crate::eval::set_contempt_b(v.max(0) as f32);
                }
            }
            "contemptflat" => {
                if let Ok(v) = value.parse::<i64>() {
                    crate::eval::set_contempt_flat(v != 0);
                }
            }
            // Every search constant, by its own name, when the build asks
            // for it. Behind a feature because a shipped engine has no
            // business letting an operator move `skip_below`: a tuner is the
            // only caller that should ever see these, and a parameter set
            // between games is a version nobody measured.
            #[cfg(feature = "tune")]
            other => {
                if let Ok(v) = value.parse::<i32>() {
                    // `Params::set` clamps to the declared box, so an
                    // out-of-range value is corrected rather than obeyed.
                    let hit = crate::search::Params::all_names()
                        .iter()
                        .find(|n| n.eq_ignore_ascii_case(other))
                        .map(|n| self.params.set(n, v));
                    if hit != Some(true) {
                        println!("info string unknown option {other}");
                    }
                }
            }
            #[cfg(not(feature = "tune"))]
            _ => {}
        }
    }
}

impl Default for Engine {
    fn default() -> Engine {
        Engine::new()
    }
}

fn parse_limits(tokens: &[&str]) -> Limits {
    let mut l = Limits::default();
    let mut i = 0;
    let num = |t: Option<&&str>| t.and_then(|s| s.parse::<u64>().ok());
    while i < tokens.len() {
        match tokens[i] {
            "depth" => l.depth = num(tokens.get(i + 1)).map(|v| v as u32),
            "nodes" => l.nodes = num(tokens.get(i + 1)),
            "movetime" => l.movetime = num(tokens.get(i + 1)),
            "wtime" => l.time[0] = num(tokens.get(i + 1)),
            "btime" => l.time[1] = num(tokens.get(i + 1)),
            "winc" => l.inc[0] = num(tokens.get(i + 1)).unwrap_or(0),
            "binc" => l.inc[1] = num(tokens.get(i + 1)).unwrap_or(0),
            "movestogo" => l.movestogo = num(tokens.get(i + 1)).map(|v| v as u32),
            "infinite" => l.infinite = true,
            _ => {}
        }
        i += 1;
    }
    l
}

pub fn run() {
    crate::qeval::warn_if_fallback();
    let mut engine = Engine::new();
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        match t[0] {
            "uci" => {
                println!("id name {NAME} {VERSION}");
                println!("id author {AUTHOR}");
                println!("option name Hash type spin default 64 min 1 max 65536");
                println!("option name Threads type spin default 1 min 1 max 256");
                println!(
                    "option name DrawValue type spin default {} min 0 max 100",
                    (crate::eval::draw_value() * 100.0).round() as i64
                );
                println!(
                    "option name ContemptA type spin default {} min 0 max 500",
                    (crate::eval::contempt_a() * 100.0).round() as i64
                );
                println!(
                    "option name ContemptB type spin default {} min 0 max 500",
                    crate::eval::contempt_b().round() as i64
                );
                println!(
                    "option name ContemptFlat type spin default {} min 0 max 1",
                    if crate::eval::contempt_flat() { 1 } else { 0 }
                );
                // The whole search-control surface, generated from the same
                // registry `chess params` reads, so the two cannot disagree.
                #[cfg(feature = "tune")]
                {
                    let d = crate::search::Params::default();
                    for n in crate::search::Params::all_names() {
                        let (lo, hi) = crate::search::Params::bounds(n).unwrap();
                        println!(
                            "option name {n} type spin default {} min {lo} max {hi}",
                            d.get(n).unwrap()
                        );
                    }
                }
                // TU-2: non-default params on the record, so a match log
                // identifies which arm this engine is.
                {
                    let d = Params::default();
                    let diffs: Vec<String> = Params::all_names()
                        .iter()
                        .filter_map(|n| {
                            let (a, b) = (engine.params.get(n), d.get(n));
                            if a != b {
                                a.map(|v| format!("{n}={v}"))
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !diffs.is_empty() {
                        println!("info string params: {}", diffs.join(","));
                    }
                }
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => {
                engine.stop();
                engine.shared.tt.clear();
                engine.board = Board::startpos();
                engine.history.clear();
                for t in engine.td.iter_mut() {
                    t.clear();
                }
                if let Ok(mut s) = engine.adapt.lock() { s.clear(); }
            }
            "setoption" => {
                // setoption name <name> value <value>
                let name_i = t.iter().position(|&x| x == "name");
                let val_i = t.iter().position(|&x| x == "value");
                if let Some(ni) = name_i {
                    let end = val_i.unwrap_or(t.len());
                    let name = t[ni + 1..end].join(" ");
                    let value = val_i.map(|vi| t[vi + 1..].join(" ")).unwrap_or_default();
                    engine.set_option(&name, &value);
                }
            }
            "position" => {
                let (fen, rest) = if t.get(1) == Some(&"startpos") {
                    (START_FEN.to_string(), 2)
                } else if t.get(1) == Some(&"fen") {
                    let end = t.iter().position(|&x| x == "moves").unwrap_or(t.len());
                    (t[2..end].join(" "), end)
                } else {
                    continue;
                };
                let moves: Vec<String> = if t.get(rest) == Some(&"moves") {
                    t[rest + 1..].iter().map(|s| s.to_string()).collect()
                } else {
                    Vec::new()
                };
                if let Err(e) = engine.set_position(&fen, &moves) {
                    eprintln!("info string {e}");
                }
            }
            "go" => {
                let limits = parse_limits(&t[1..]);
                engine.go_async(limits);
            }
            "stop" => engine.stop(),
            "quit" => {
                engine.stop();
                break;
            }
            // Non-standard conveniences.
            "d" | "board" => println!("{}", engine.board),
            // What the search uses, not the PeSTO table, and which eval it was.
            "eval" => {
                let cp = DefaultEval::without_adaptive().evaluate(&engine.board);
                println!("{cp} cp (side to move, {})", crate::qeval::eval_name());
            }
            "perft" => {
                let depth: u32 = t.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
                let start = std::time::Instant::now();
                let mut total = 0;
                for (m, n) in crate::perft::perft_divide(&engine.board, depth) {
                    println!("{m}: {n}");
                    total += n;
                }
                let el = start.elapsed().as_secs_f64();
                println!("\nnodes {total}  time {el:.3}s  {:.1} Mnps", total as f64 / el / 1e6);
            }
            _ => {}
        }
        let _ = std::io::stdout().flush();
    }
    engine.finish();
}

/// Parse a UCI move against a board without applying it.
pub fn parse_move(b: &Board, s: &str) -> Option<Move> {
    let mut list = MoveList::new();
    generate(b, GenType::All, &mut list);
    list.find_uci(s)
}
