//! Engine-vs-engine match runner.
//!
//! This is the measurement instrument. Everything the project claims —
//! "this patch is worth Elo", "the engine is rated about X" — is a number that
//! comes out of here, so it is deliberately built to be boring and hard to
//! fool:
//!
//! * **Games are played in colour-reversed pairs from the same opening.** The
//!   pair is the unit of statistics (see `sprt`), which removes colour and
//!   opening from the variance instead of letting them inflate it.
//! * **Both a built-in and an external opponent.** Internal play is a process
//!   per nothing — same binary, different `Params` — which is the cheap way to
//!   test a search change. External play drives any UCI engine over a pipe,
//!   which is how a *rated* opponent gets used and therefore the only way an
//!   absolute Elo number is ever produced.
//! * **Adjudication is two-sided.** A resign or draw claim requires both
//!   engines to agree. One engine's opinion is exactly the thing under test;
//!   trusting it to adjudicate its own games would let an eval bug manufacture
//!   its own wins.
//!
//! It is not cutechess-cli and does not try to be — no Chess960, no tournament
//! formats, no PGN book parsing. It is ~600 lines of `std` that keeps the
//! measurement in-repo and reproducible.

use crate::board::{Board, START_FEN};
use crate::chess_move::{Move, MoveList};
use crate::eval::Score;
use crate::game::{self, Status};
use crate::movegen::{generate, GenType};
use crate::san;
use crate::search::{Limits, Params, Searcher, Shared, ThreadData};
use crate::sprt::{MatchStats, PairResult, Sprt, SprtVerdict};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------- config

#[derive(Clone, Copy, Debug)]
pub struct TimeControl {
    pub base_ms: u64,
    pub inc_ms: u64,
    /// Fixed time per move, overriding the clock.
    pub movetime: Option<u64>,
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    /// Grace before a clock overrun is called as a loss. Non-zero because a
    /// shared machine can stall a process for tens of milliseconds through no
    /// fault of the engine; time losses are reported separately so a real time
    /// management bug is still visible.
    pub margin_ms: u64,
}

impl Default for TimeControl {
    fn default() -> TimeControl {
        TimeControl {
            base_ms: 8000,
            inc_ms: 80,
            movetime: None,
            depth: None,
            nodes: None,
            margin_ms: 100,
        }
    }
}

impl TimeControl {
    /// Parse `8+0.08`, `10+0.1`, `60` (seconds), or `mt=1000` / `depth=8` /
    /// `nodes=100000` for fixed-work controls.
    pub fn parse(s: &str) -> Result<TimeControl, String> {
        let mut tc = TimeControl::default();
        if let Some(v) = s.strip_prefix("mt=") {
            tc.movetime = Some(v.parse().map_err(|_| format!("bad movetime {v}"))?);
            return Ok(tc);
        }
        if let Some(v) = s.strip_prefix("depth=") {
            tc.depth = Some(v.parse().map_err(|_| format!("bad depth {v}"))?);
            return Ok(tc);
        }
        if let Some(v) = s.strip_prefix("nodes=") {
            tc.nodes = Some(v.parse().map_err(|_| format!("bad nodes {v}"))?);
            return Ok(tc);
        }
        let (base, inc) = match s.split_once('+') {
            Some((b, i)) => (b, i),
            None => (s, "0"),
        };
        let base: f64 = base.parse().map_err(|_| format!("bad time control {s}"))?;
        let inc: f64 = inc.parse().map_err(|_| format!("bad time control {s}"))?;
        tc.base_ms = (base * 1000.0) as u64;
        tc.inc_ms = (inc * 1000.0) as u64;
        Ok(tc)
    }

    pub fn describe(&self) -> String {
        if let Some(mt) = self.movetime {
            format!("{mt}ms/move")
        } else if let Some(d) = self.depth {
            format!("depth {d}")
        } else if let Some(n) = self.nodes {
            format!("{n} nodes/move")
        } else {
            format!(
                "{}+{}",
                self.base_ms as f64 / 1000.0,
                self.inc_ms as f64 / 1000.0
            )
        }
    }
}

/// When to stop a game early. Both thresholds require agreement from both
/// engines, so a single engine's misevaluation cannot decide a game.
#[derive(Clone, Copy, Debug)]
pub struct Adjudication {
    /// Both sides agree the margin exceeds this (cp) for `resign_plies`.
    pub resign_score: Score,
    pub resign_plies: u32,
    /// Both sides report |score| under this for `draw_plies`, after move
    /// `draw_after_move`.
    pub draw_score: Score,
    pub draw_plies: u32,
    pub draw_after_move: u32,
    pub max_plies: usize,
}

impl Default for Adjudication {
    fn default() -> Adjudication {
        Adjudication {
            resign_score: 900,
            resign_plies: 8,
            draw_score: 8,
            draw_plies: 16,
            draw_after_move: 40,
            max_plies: 600,
        }
    }
}

#[derive(Clone, Debug)]
pub enum PlayerSpec {
    /// The engine in this binary, with these search parameters.
    Internal { name: String, params: Params },
    /// Any UCI engine, driven over a pipe.
    External { path: String, args: Vec<String> },
}

impl PlayerSpec {
    pub fn label(&self) -> String {
        match self {
            PlayerSpec::Internal { name, .. } => name.clone(),
            PlayerSpec::External { path, .. } => std::path::Path::new(path)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.clone()),
        }
    }
}

pub struct MatchConfig {
    pub a: PlayerSpec,
    pub b: PlayerSpec,
    /// Names written to the PGN White/Black tags. When both arms report the
    /// same `id name` (two builds of this engine) the tags would be identical
    /// and `chess elo` on the file would silently score White instead of an
    /// arm — so identical tags are qualified with ` [A]` / ` [B]` unless the
    /// caller names the arms explicitly.
    pub name_a: Option<String>,
    pub name_b: Option<String>,
    pub games: u32,
    pub tc: TimeControl,
    pub concurrency: usize,
    pub hash_mb: usize,
    /// Search threads per engine (UCI `Threads`). 1 is the default: more
    /// threads is Elo per game, not games per hour.
    /// Per-arm: the SMP number is Threads 6 vs Threads 1, same binary, so
    /// one value for both arms cannot express it.
    pub threads_a: usize,
    pub threads_b: usize,
    pub sprt: Option<Sprt>,
    pub pgn: Option<String>,
    pub book: Vec<Vec<String>>,
    pub adj: Adjudication,
    pub quiet: bool,
}

// ---------------------------------------------------------------- external engines

/// A UCI engine in another process.
///
/// The reader runs on its own thread feeding a channel, which is the only way
/// to put a timeout on a pipe read in `std`. Without it a hung opponent hangs
/// the whole match, which is exactly the failure that eats an overnight run.
struct UciProcess {
    child: Child,
    stdin: ChildStdin,
    rx: mpsc::Receiver<String>,
    name: String,
}

impl UciProcess {
    fn spawn(path: &str, args: &[String], hash_mb: usize, threads: usize) -> Result<UciProcess, String> {
        let mut child = Command::new(path)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("cannot start {path}: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            return;
                        }
                    }
                    Err(_) => return,
                }
            }
        });

        let mut p = UciProcess {
            child,
            stdin,
            rx,
            name: path.to_string(),
        };
        p.send("uci")?;
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let line = p.line(deadline)?;
            if let Some(rest) = line.strip_prefix("id name ") {
                p.name = rest.trim().to_string();
            }
            if line.trim() == "uciok" {
                break;
            }
        }
        p.send(&format!("setoption name Hash value {hash_mb}"))?;
        // Search threads per engine. Lazy SMP shares one TT across threads;
        // more threads is Elo per game (ROADMAP 2), not games per hour, so
        // SPRT throughput still wants Threads 1 with high --concurrency.
        p.send(&format!("setoption name Threads value {}", threads.max(1)))?;
        p.isready(Duration::from_secs(20))?;
        Ok(p)
    }

    fn send(&mut self, s: &str) -> Result<(), String> {
        writeln!(self.stdin, "{s}").map_err(|e| format!("{}: write failed: {e}", self.name))?;
        self.stdin
            .flush()
            .map_err(|e| format!("{}: flush failed: {e}", self.name))
    }

    fn line(&mut self, deadline: Instant) -> Result<String, String> {
        let now = Instant::now();
        if now >= deadline {
            return Err(format!("{}: timed out", self.name));
        }
        self.rx
            .recv_timeout(deadline - now)
            .map_err(|_| format!("{}: timed out waiting for output", self.name))
    }

    fn isready(&mut self, within: Duration) -> Result<(), String> {
        self.send("isready")?;
        let deadline = Instant::now() + within;
        loop {
            if self.line(deadline)?.trim() == "readyok" {
                return Ok(());
            }
        }
    }
}

impl Drop for UciProcess {
    fn drop(&mut self) {
        let _ = self.send("quit");
        // Give it a moment to exit cleanly; a UCI engine that ignores `quit`
        // gets killed rather than left behind on a shared machine.
        for _ in 0..20 {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------- players

struct Thought {
    mv: Move,
    /// Side-to-move point of view, as UCI defines it. `None` if the engine
    /// reported no score.
    score: Option<Score>,
    depth: u32,
    nodes: u64,
    elapsed: Duration,
}

enum Player {
    Internal {
        shared: Arc<Shared>,
        params: Params,
        name: String,
        adapt: std::sync::Arc<std::sync::Mutex<crate::adaptive::AdaptiveState>>,
    },
    External(Box<UciProcess>),
}

impl Player {
    fn new(spec: &PlayerSpec, hash_mb: usize, threads: usize) -> Result<Player, String> {
        Ok(match spec {
            PlayerSpec::Internal { name, params } => {
                // Each Internal player gets its own adaptive handle so games don't cross-contaminate
                // even with concurrency>1. Params encode adaptive spec via name suffix "[adapt:...]"
                // or via CHESS_ADAPTIVE env captured at thread start.
                if threads > 1 {
                    return Err(format!(
                        "{name}: --threads {threads} needs a subprocess engine (Internal play is single-threaded) — pass a binary path instead of 'self'"
                    ));
                }
                let adapt_params = if name.contains("[adapt:") {
                    let s = name.split("[adapt:").nth(1).and_then(|x| x.split(']').next()).unwrap_or("");
                    crate::adaptive::AdaptiveParams::from_spec(s).unwrap_or_default()
                } else {
                    crate::adaptive::global_handle().lock().map(|g| g.params.clone()).unwrap_or_default()
                };
                Player::Internal {
                    shared: Arc::new(Shared::new(hash_mb)),
                    params: *params,
                    name: name.clone(),
                    adapt: std::sync::Arc::new(std::sync::Mutex::new(crate::adaptive::AdaptiveState::new(adapt_params))),
                }
            },
            PlayerSpec::External { path, args } => {
                Player::External(Box::new(UciProcess::spawn(path, args, hash_mb, threads)?))
            }
        })
    }

    fn name(&self) -> String {
        match self {
            Player::Internal { name, .. } => name.clone(),
            Player::External(p) => p.name.clone(),
        }
    }

    fn new_game(&mut self) -> Result<(), String> {
        match self {
            Player::Internal { shared, adapt, .. } => {
                shared.tt.clear();
                if let Ok(mut s) = adapt.lock() { s.clear(); }
                Ok(())
            }
            Player::External(p) => {
                p.send("ucinewgame")?;
                p.isready(Duration::from_secs(20))
            }
        }
    }

    fn think(
        &mut self,
        board: &Board,
        start_fen: &str,
        moves: &[String],
        keys: &[u64],
        clock: [u64; 2],
        tc: &TimeControl,
    ) -> Result<Thought, String> {
        let limits = build_limits(tc, clock, board.stm().index());
        let t0 = Instant::now();
        match self {
            Player::Internal { shared, params, adapt, .. } => {
                shared.stop.store(false, Ordering::Relaxed);
                let adapt = adapt.clone();
                let mut s = Searcher::new(shared, ThreadData::new(0), crate::qeval::DefaultEval::with_adaptive(adapt), *params);
                let r = s.go(board, keys, &limits, None);
                Ok(Thought {
                    mv: r.best_move,
                    score: Some(r.score),
                    depth: r.depth,
                    nodes: r.nodes,
                    elapsed: t0.elapsed(),
                })
            }
            Player::External(p) => {
                let mut pos = format!("position fen {start_fen}");
                if !moves.is_empty() {
                    pos.push_str(" moves ");
                    pos.push_str(&moves.join(" "));
                }
                p.send(&pos)?;
                p.send(&go_command(tc, clock))?;

                // A wall-clock ceiling well above any legitimate think time:
                // past this the engine is hung, not slow.
                let budget = tc
                    .movetime
                    .map(|mt| mt + 10_000)
                    .unwrap_or_else(|| clock[board.stm().index()] + 15_000);
                let deadline = t0 + Duration::from_millis(budget.max(30_000));

                let mut score = None;
                let mut depth = 0;
                let mut nodes = 0;
                loop {
                    let line = p.line(deadline)?;
                    let t: Vec<&str> = line.split_whitespace().collect();
                    match t.first() {
                        Some(&"info") => {
                            if let Some(v) = field(&t, "depth") {
                                depth = v.parse().unwrap_or(depth);
                            }
                            if let Some(v) = field(&t, "nodes") {
                                nodes = v.parse().unwrap_or(nodes);
                            }
                            if let Some(i) = t.iter().position(|&x| x == "score") {
                                match (t.get(i + 1), t.get(i + 2)) {
                                    (Some(&"cp"), Some(v)) => {
                                        score = v.parse::<Score>().ok();
                                    }
                                    (Some(&"mate"), Some(v)) => {
                                        score = v.parse::<i32>().ok().map(mate_score);
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Some(&"bestmove") => {
                            let uci = t.get(1).ok_or("bestmove with no move")?;
                            let mut list = MoveList::new();
                            generate(board, GenType::All, &mut list);
                            let mv = list
                                .find_uci(uci)
                                .ok_or_else(|| format!("{}: illegal move {uci}", p.name))?;
                            return Ok(Thought {
                                mv,
                                score,
                                depth,
                                nodes,
                                elapsed: t0.elapsed(),
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

fn field<'a>(t: &'a [&'a str], key: &str) -> Option<&'a str> {
    t.iter().position(|&x| x == key).and_then(|i| t.get(i + 1)).copied()
}

/// Mate-in-N as a centipawn-like score, for adjudication only.
fn mate_score(n: i32) -> Score {
    if n >= 0 {
        crate::eval::MATE - n.abs() as Score
    } else {
        -crate::eval::MATE + n.abs() as Score
    }
}

fn build_limits(tc: &TimeControl, clock: [u64; 2], _stm: usize) -> Limits {
    let mut l = Limits::default();
    if let Some(mt) = tc.movetime {
        l.movetime = Some(mt);
    } else if let Some(d) = tc.depth {
        l.depth = Some(d);
    } else if let Some(n) = tc.nodes {
        l.nodes = Some(n);
    } else {
        l.time = [Some(clock[0]), Some(clock[1])];
        l.inc = [tc.inc_ms, tc.inc_ms];
    }
    l
}

fn go_command(tc: &TimeControl, clock: [u64; 2]) -> String {
    if let Some(mt) = tc.movetime {
        format!("go movetime {mt}")
    } else if let Some(d) = tc.depth {
        format!("go depth {d}")
    } else if let Some(n) = tc.nodes {
        format!("go nodes {n}")
    } else {
        format!(
            "go wtime {} btime {} winc {} binc {}",
            clock[0], clock[1], tc.inc_ms, tc.inc_ms
        )
    }
}

// ---------------------------------------------------------------- one game

pub struct GameRecord {
    /// Points scored by white: 1.0, 0.5 or 0.0.
    pub white_points: f64,
    pub reason: String,
    pub plies: usize,
    pub pgn: String,
    /// Set when a game ended because an engine broke, not because chess ended.
    pub error: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn play_game(
    white: &mut Player,
    black: &mut Player,
    white_tag: &str,
    black_tag: &str,
    opening: &[String],
    tc: &TimeControl,
    adj: &Adjudication,
    round: u32,
) -> GameRecord {
    let mut board = Board::startpos();
    let mut keys: Vec<u64> = Vec::new();
    let mut moves: Vec<String> = Vec::new();
    let mut comments: Vec<String> = Vec::new();
    let mut sans: Vec<String> = Vec::new();

    // Replay the opening. A book line that is not legal is a bug in the book,
    // not a game result, so it is loud.
    for m in opening {
        let mut list = MoveList::new();
        generate(&board, GenType::All, &mut list);
        match list.find_uci(m) {
            Some(mv) => {
                sans.push(san::san(&board, mv));
                comments.push(String::new());
                keys.push(board.key());
                board = board.make_move(mv);
                moves.push(m.clone());
            }
            None => {
                return GameRecord {
                    white_points: 0.5,
                    reason: "bad book line".into(),
                    plies: 0,
                    pgn: String::new(),
                    error: Some(format!("illegal book move {m}")),
                }
            }
        }
    }

    let mut clock = [tc.base_ms, tc.base_ms];
    // White-point-of-view scores, tagged with the ply that produced them, so
    // adjudication can verify it really has both engines' opinions and not a
    // run of one engine's (an engine that reports no score leaves gaps).
    let mut white_scores: Vec<(usize, Score)> = Vec::new();

    let _ = white.new_game();
    let _ = black.new_game();

    // The loop yields the game result, so every exit path has to state one —
    // there is no "fell out of the loop with no result" case to forget.
    let (white_points, reason, error): (f64, String, Option<String>) = loop {
        let st = game::status(&board, &keys);
        if st.is_over() {
            let pts = match st {
                Status::Checkmate => {
                    if board.stm() == crate::types::Color::White {
                        0.0
                    } else {
                        1.0
                    }
                }
                _ => 0.5,
            };
            break (pts, st.tag().to_string(), None);
        }
        if moves.len() >= adj.max_plies {
            break (0.5, "adjudication: too long".to_string(), None);
        }

        let stm = board.stm().index();
        let player: &mut Player = if stm == 0 { white } else { black };
        let thought = match player.think(&board, START_FEN, &moves, &keys, clock, tc) {
            Ok(t) => t,
            Err(e) => {
                // A broken engine loses the game, and the error is recorded so
                // it cannot be silently absorbed into the Elo estimate.
                let pts = if stm == 0 { 0.0 } else { 1.0 };
                break (pts, format!("engine error: {e}"), Some(e));
            }
        };

        if tc.movetime.is_none() && tc.depth.is_none() && tc.nodes.is_none() {
            let used = thought.elapsed.as_millis() as u64;
            if used > clock[stm] + tc.margin_ms {
                let pts = if stm == 0 { 0.0 } else { 1.0 };
                break (pts, "time forfeit".to_string(), None);
            }
            clock[stm] = clock[stm].saturating_sub(used) + tc.inc_ms;
        }

        let mut legal = MoveList::new();
        generate(&board, GenType::All, &mut legal);
        if !legal.contains(thought.mv) {
            let e = format!("{} played illegal move {}", player.name(), thought.mv);
            let pts = if stm == 0 { 0.0 } else { 1.0 };
            break (pts, e.clone(), Some(e));
        }

        let wpov = thought.score.map(|s| if stm == 0 { s } else { -s });
        if let Some(s) = wpov {
            white_scores.push((moves.len(), s));
        }

        sans.push(san::san(&board, thought.mv));
        // The same shape cutechess writes: eval/depth, time, nodes. Enough to
        // reconstruct what the engine thought at every move of a lost game.
        comments.push(format!(
            "{}/{} {:.2}s {}n",
            thought
                .score
                .map(|s| format!("{:+.2}", s as f64 / 100.0))
                .unwrap_or_else(|| "?".into()),
            thought.depth,
            thought.elapsed.as_secs_f64(),
            thought.nodes
        ));
        keys.push(board.key());
        board = board.make_move(thought.mv);
        moves.push(thought.mv.to_uci());

        if let Some((pts, why)) = adjudicate(&white_scores, moves.len(), adj) {
            break (pts, why, None);
        }
    };

    let pgn = write_pgn(
        white_tag,
        black_tag,
        white_points,
        &reason,
        round,
        tc,
        &sans,
        &comments,
    );
    GameRecord {
        white_points,
        reason,
        plies: moves.len(),
        pgn,
        error,
    }
}

/// Two-sided adjudication: both engines must have agreed, for long enough.
///
/// `scores` is white's point of view, tagged with the ply that produced it.
/// The tag matters: "two-sided" is only true if the window actually contains
/// moves by both players. An engine that reports no score for some moves would
/// otherwise leave a window of one engine's opinions, and one engine deciding
/// its own games is exactly what this is here to prevent.
fn adjudicate(
    scores: &[(usize, Score)],
    plies: usize,
    adj: &Adjudication,
) -> Option<(f64, String)> {
    let both_sides = |w: &[(usize, Score)]| {
        w.iter().any(|(p, _)| p % 2 == 1) && w.iter().any(|(p, _)| p % 2 == 0)
    };

    let n = adj.resign_plies as usize;
    if n > 0 && scores.len() >= n {
        let tail = &scores[scores.len() - n..];
        if both_sides(tail) {
            if tail.iter().all(|&(_, s)| s >= adj.resign_score) {
                return Some((1.0, "adjudication: white winning".into()));
            }
            if tail.iter().all(|&(_, s)| s <= -adj.resign_score) {
                return Some((0.0, "adjudication: black winning".into()));
            }
        }
    }
    let d = adj.draw_plies as usize;
    if d > 0 && plies >= adj.draw_after_move as usize * 2 && scores.len() >= d {
        let tail = &scores[scores.len() - d..];
        if both_sides(tail) && tail.iter().all(|&(_, s)| s.abs() <= adj.draw_score) {
            return Some((0.5, "adjudication: drawn".into()));
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn write_pgn(
    white: &str,
    black: &str,
    white_points: f64,
    reason: &str,
    round: u32,
    tc: &TimeControl,
    sans: &[String],
    comments: &[String],
) -> String {
    let res = if white_points > 0.75 {
        "1-0"
    } else if white_points < 0.25 {
        "0-1"
    } else {
        "1/2-1/2"
    };
    let mut s = String::new();
    s.push_str("[Event \"engine match\"]\n[Site \"local\"]\n");
    s.push_str(&format!("[Round \"{round}\"]\n"));
    s.push_str(&format!("[White \"{white}\"]\n[Black \"{black}\"]\n"));
    s.push_str(&format!("[Result \"{res}\"]\n"));
    s.push_str(&format!("[TimeControl \"{}\"]\n", tc.describe()));
    s.push_str(&format!("[Termination \"{reason}\"]\n\n"));

    let mut line = String::new();
    for (i, mv) in sans.iter().enumerate() {
        let mut tok = String::new();
        if i % 2 == 0 {
            tok.push_str(&format!("{}. ", i / 2 + 1));
        }
        tok.push_str(mv);
        if let Some(c) = comments.get(i) {
            if !c.is_empty() {
                tok.push_str(&format!(" {{{c}}}"));
            }
        }
        if line.len() + tok.len() > 78 {
            s.push_str(line.trim_end());
            s.push('\n');
            line.clear();
        }
        line.push_str(&tok);
        line.push(' ');
    }
    line.push_str(res);
    s.push_str(line.trim_end());
    s.push_str("\n\n");
    s
}

// ---------------------------------------------------------------- the match

struct PairJob {
    round: u32,
    opening: Vec<String>,
}

/// PGN White/Black tags for the two arms: explicit names win, otherwise the
/// players' own `id name`s. Two builds of this engine report the same id
/// name, and a PGN whose White and Black tags are identical cannot be scored
/// per-arm (`points_for` would match White first and report White's score as
/// the arm's) — so a collision is qualified with the arm, loudly, once.
fn pgn_tags(a_name: &str, b_name: &str, want_a: Option<&str>, want_b: Option<&str>) -> (String, String) {
    let mut ta = want_a.unwrap_or(a_name).to_string();
    let mut tb = want_b.unwrap_or(b_name).to_string();
    if ta == tb {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!("note: both arms report '{ta}' — qualifying PGN tags as '{ta} [A]' / '{tb} [B]'. Pass --name-a/--name-b for stable names.");
        });
        ta.push_str(" [A]");
        tb.push_str(" [B]");
    }
    (ta, tb)
}

/// Run the match. Returns the final statistics and the SPRT verdict, if one
/// was requested.
pub fn run(cfg: MatchConfig) -> (MatchStats, Option<SprtVerdict>) {
    let pairs = (cfg.games as usize + 1) / 2;
    let jobs: Vec<PairJob> = (0..pairs)
        .map(|i| PairJob {
            round: i as u32 + 1,
            opening: cfg.book[i % cfg.book.len()].clone(),
        })
        .collect();

    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let jobs = Arc::new(jobs);
    let pgn_file = cfg.pgn.as_ref().map(|p| {
        Arc::new(Mutex::new(
            std::fs::File::create(p).expect("cannot create pgn file"),
        ))
    });
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<(PairResult, [f64; 2], Vec<String>, Vec<String>)>();

    let threads = cfg.concurrency.max(1);
    let mut handles = Vec::new();
    // PGN arm tags, resolved late: external players only learn their `id
    // name` at handshake, so each worker computes the same tags from the
    // names it sees. Explicit --name-a/--name-b always win; otherwise the
    // players' own names are used, qualified on collision (below).
    let tag_a_cfg = cfg.name_a.clone();
    let tag_b_cfg = cfg.name_b.clone();
    for _ in 0..threads {
        let jobs = Arc::clone(&jobs);
        let next = Arc::clone(&next);
        let stop = Arc::clone(&stop);
        let tx = tx.clone();
        let pgn_file = pgn_file.clone();
        let a_spec = cfg.a.clone();
        let b_spec = cfg.b.clone();
        let tag_a_cfg = tag_a_cfg.clone();
        let tag_b_cfg = tag_b_cfg.clone();
        let tc = cfg.tc;
        let adj = cfg.adj;
        let hash = cfg.hash_mb;
        let threads_a = cfg.threads_a.max(1);
        let threads_b = cfg.threads_b.max(1);
        handles.push(std::thread::spawn(move || {
            let mut a = match Player::new(&a_spec, hash, threads_a) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("cannot start A: {e}");
                    stop.store(true, Ordering::Relaxed);
                    return;
                }
            };
            let mut b = match Player::new(&b_spec, hash, threads_b) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("cannot start B: {e}");
                    stop.store(true, Ordering::Relaxed);
                    return;
                }
            };
            let (tag_a, tag_b) = pgn_tags(&a.name(), &b.name(), tag_a_cfg.as_deref(), tag_b_cfg.as_deref());
            loop {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= jobs.len() {
                    return;
                }
                let job = &jobs[i];

                // Same opening, both colours. This is the pair.
                // Tags track arms, not colours: g1 is A-white, g2 is B-white.
                let g1 = play_game(&mut a, &mut b, &tag_a, &tag_b, &job.opening, &tc, &adj, job.round * 2 - 1);
                let g2 = play_game(&mut b, &mut a, &tag_b, &tag_a, &job.opening, &tc, &adj, job.round * 2);
                let a_points = [g1.white_points, 1.0 - g2.white_points];

                if let Some(f) = &pgn_file {
                    if let Ok(mut f) = f.lock() {
                        let _ = f.write_all(g1.pgn.as_bytes());
                        let _ = f.write_all(g2.pgn.as_bytes());
                    }
                }
                let errs: Vec<String> =
                    [g1.error, g2.error].into_iter().flatten().collect();
                let reasons = vec![g1.reason, g2.reason];
                let pair = PairResult::from_points(a_points[0] + a_points[1]);
                if tx.send((pair, a_points, reasons, errs)).is_err() {
                    return;
                }
            }
        }));
    }
    drop(tx);

    let mut stats = MatchStats::default();
    let mut verdict = None;
    let start = Instant::now();
    let mut error_count = 0usize;
    for (pair, points, _reasons, errs) in rx {
        stats.add_game(points[0]);
        stats.add_game(points[1]);
        stats.add_pair(pair);
        for e in errs {
            error_count += 1;
            eprintln!("  ! {e}");
        }
        if !cfg.quiet {
            println!("{}", crate::sprt::summary(&stats, cfg.sprt.as_ref()));
        }
        if let Some(t) = &cfg.sprt {
            let v = t.verdict(&stats);
            if v != SprtVerdict::Continue {
                verdict = Some(v);
                stop.store(true, Ordering::Relaxed);
                break;
            }
        }
    }
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }

    if error_count > 0 {
        eprintln!("\n{error_count} game(s) ended in an engine error — treat the result with suspicion.");
    }
    println!(
        "\n{} vs {}   {}   {:.0}s",
        cfg.a.label(),
        cfg.b.label(),
        cfg.tc.describe(),
        start.elapsed().as_secs_f64()
    );
    println!("{}", crate::sprt::summary(&stats, cfg.sprt.as_ref()));
    // One machine-readable line, for scripts that need the numbers rather than
    // the prose. Anything that parses match output should parse this.
    if let (Some((elo, lo, hi)), Some(los)) = (stats.elo(), stats.los()) {
        println!(
            "RESULT games={} w={} l={} d={} elo={:.2} lo={:.2} hi={:.2} los={:.4} pairs={}",
            stats.games(),
            stats.wins,
            stats.losses,
            stats.draws,
            elo,
            lo,
            hi,
            los,
            stats.pair_count()
        );
    }
    if let Some(v) = verdict {
        println!(
            "SPRT: {}",
            match v {
                SprtVerdict::AcceptH1 => "H1 accepted — the change is an improvement",
                SprtVerdict::AcceptH0 => "H0 accepted — the change is not an improvement",
                SprtVerdict::Continue => "inconclusive",
            }
        );
    } else if cfg.sprt.is_some() {
        println!("SPRT: inconclusive within {} games", cfg.games);
    }
    (stats, verdict)
}

// ---------------------------------------------------------------- reading games back

/// One finished game, as recovered from a PGN file.
#[derive(Clone)]
pub struct PgnGame {
    pub round: u32,
    pub white: String,
    pub black: String,
    /// Points scored by White: 1.0, 0.5 or 0.0.
    pub white_points: f64,
}

impl PgnGame {
    /// Points scored by `who`. `None` if that engine did not play this game,
    /// which is the caller's cue that it has the wrong name — not a zero.
    pub fn points_for(&self, who: &str) -> Option<f64> {
        if self.white == who {
            Some(self.white_points)
        } else if self.black == who {
            Some(1.0 - self.white_points)
        } else {
            None
        }
    }
}

/// Recover results from a PGN written by this runner.
///
/// The point is to be able to compute the statistics of a match that is still
/// running — or one that finished last week — with *the same* estimator that
/// produced them live, rather than a second implementation in a shell script
/// that quietly disagrees. Only headers are read; the moves are skipped.
///
/// Games whose `Round` is 2k-1 and 2k form the colour-reversed pair k, which is
/// how `run` assigns them.
pub fn read_pgn(path: &str) -> Result<Vec<PgnGame>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut games = Vec::new();
    let (mut round, mut white, mut black, mut result) = (None, None, None, None);
    for line in text.lines() {
        let line = line.trim();
        let tag = |name: &str| -> Option<String> {
            let p = format!("[{name} \"");
            line.strip_prefix(&p)
                .and_then(|r| r.strip_suffix("\"]"))
                .map(|s| s.to_string())
        };
        if let Some(v) = tag("Round") {
            // A new header block: whatever was half-collected is abandoned.
            round = v.parse::<u32>().ok();
            white = None;
            black = None;
            result = None;
        } else if let Some(v) = tag("White") {
            white = Some(v);
        } else if let Some(v) = tag("Black") {
            black = Some(v);
        } else if let Some(v) = tag("Result") {
            result = Some(v);
        }
        if let (Some(r), Some(w), Some(b), Some(res)) =
            (round, white.as_ref(), black.as_ref(), result.as_ref())
        {
            let white_points = match res.as_str() {
                "1-0" => 1.0,
                "0-1" => 0.0,
                "1/2-1/2" => 0.5,
                _ => continue, // "*" — game not finished
            };
            games.push(PgnGame {
                round: r,
                white: w.clone(),
                black: b.clone(),
                white_points,
            });
            round = None;
            white = None;
            black = None;
            result = None;
        }
    }
    Ok(games)
}

/// The engine that played every game in the set — i.e. the one under test in a
/// gauntlet, as opposed to the opponents it faced.
///
/// Guessing this wrongly does not fail: it silently attributes the opponent's
/// results to us and reports a number that looks perfectly reasonable and is
/// backwards. So it is derived from the games rather than defaulted, and an
/// ambiguous set is an error the caller has to resolve with `--name`.
pub fn engine_under_test(games: &[PgnGame]) -> Result<String, String> {
    if games.is_empty() {
        return Err("no finished games".into());
    }
    let mut names: Vec<String> = games
        .iter()
        .flat_map(|g| [g.white.clone(), g.black.clone()])
        .collect();
    names.sort();
    names.dedup();
    let constant: Vec<String> = names
        .into_iter()
        .filter(|n| games.iter().all(|g| g.white == *n || g.black == *n))
        .collect();
    if constant.len() == 1 {
        return Ok(constant[0].clone());
    }
    // A single head-to-head has two engines in every game. If one of them is
    // this binary, that is obviously the one meant.
    let ours = format!("{} {}", crate::uci::NAME, crate::uci::VERSION);
    if constant.iter().any(|n| *n == ours) {
        return Ok(ours);
    }
    if constant.is_empty() {
        return Err("no single engine played every game — pass --name".into());
    }
    Err(format!(
        "ambiguous: {} each played every game — pass --name",
        constant.join(", ")
    ))
}

/// Fold recovered games into match statistics, from `who`'s point of view.
/// Games are counted individually; only *complete* pairs contribute to the
/// pentanomial statistics, so a match read mid-pair is not misreported.
///
/// Each call scopes pairs to the games passed: merging files from different
/// matches (or opponents) must merge the resulting `MatchStats`, not the game
/// lists, or round numbers from unrelated matches form bogus pairs.
pub fn stats_from_games(games: &[PgnGame], who: &str) -> Result<MatchStats, String> {
    for g in games {
        if g.white == g.black {
            return Err(format!(
                "round {} has identical White/Black tags ('{}') — this PGN cannot be scored per-arm (it would report White's score). Re-run the match with --name-a/--name-b so the arms differ",
                g.round, g.white
            ));
        }
    }
    let mut by_pair: std::collections::BTreeMap<u32, Vec<f64>> = std::collections::BTreeMap::new();
    let mut stats = MatchStats::default();
    for g in games {
        let pts = g
            .points_for(who)
            .ok_or_else(|| format!("'{who}' did not play round {} ({} vs {})", g.round, g.white, g.black))?;
        stats.add_game(pts);
        by_pair.entry((g.round - 1) / 2).or_default().push(pts);
    }
    for (_, pts) in by_pair {
        if pts.len() == 2 {
            stats.add_pair(PairResult::from_points(pts[0] + pts[1]));
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(vals: &[(usize, Score)]) -> Vec<(usize, Score)> {
        vals.to_vec()
    }

    #[test]
    fn resign_needs_both_engines_to_agree() {
        let adj = Adjudication { resign_plies: 4, ..Default::default() };

        // Alternating plies, both engines see white winning by a lot.
        let agreed = scores(&[(41, 1200), (42, 1300), (43, 1250), (44, 1400)]);
        assert_eq!(
            adjudicate(&agreed, 44, &adj).map(|(p, _)| p),
            Some(1.0),
            "both engines agreed and it was not adjudicated"
        );

        // Same margins, but every entry came from the same side — the other
        // engine reported nothing. This must NOT be adjudicated: it is one
        // engine deciding its own game.
        let one_sided = scores(&[(41, 1200), (43, 1300), (45, 1250), (47, 1400)]);
        assert_eq!(
            adjudicate(&one_sided, 47, &adj),
            None,
            "adjudicated on one engine's opinion alone"
        );
    }

    #[test]
    fn draw_adjudication_waits_for_the_move_number() {
        let adj = Adjudication {
            draw_plies: 4,
            draw_after_move: 40,
            draw_score: 8,
            ..Default::default()
        };
        let quiet = scores(&[(77, 2), (78, -3), (79, 0), (80, 4)]);
        // Before move 40 a quiet evaluation means nothing — the game has not
        // started yet as far as adjudication is concerned.
        assert_eq!(adjudicate(&quiet, 20, &adj), None);
        assert_eq!(adjudicate(&quiet, 80, &adj).map(|(p, _)| p), Some(0.5));

        // One entry outside the band is enough to keep playing.
        let unsettled = scores(&[(77, 2), (78, -3), (79, 40), (80, 4)]);
        assert_eq!(adjudicate(&unsettled, 80, &adj), None);
    }

    #[test]
    fn time_control_parses_the_forms_we_use() {
        let tc = TimeControl::parse("8+0.08").unwrap();
        assert_eq!((tc.base_ms, tc.inc_ms), (8000, 80));
        assert_eq!(TimeControl::parse("60").unwrap().base_ms, 60_000);
        assert_eq!(TimeControl::parse("mt=250").unwrap().movetime, Some(250));
        assert_eq!(TimeControl::parse("depth=7").unwrap().depth, Some(7));
        assert_eq!(TimeControl::parse("nodes=5000").unwrap().nodes, Some(5000));
        assert!(TimeControl::parse("banana").is_err());
    }

    #[test]
    fn pgn_tags_pass_through_distinct_names() {
        assert_eq!(
            pgn_tags("chess 0.1.0", "4ku 5.1", None, None),
            ("chess 0.1.0".to_string(), "4ku 5.1".to_string())
        );
    }

    #[test]
    fn pgn_tags_qualify_on_collision() {
        // Two builds of this engine: identical id names must not reach the PGN.
        let (a, b) = pgn_tags("chess 0.1.0", "chess 0.1.0", None, None);
        assert_ne!(a, b);
        assert!(a.contains("[A]") && b.contains("[B]"), "{a} / {b}");
    }

    #[test]
    fn pgn_tags_explicit_names_win() {
        assert_eq!(
            pgn_tags("chess 0.1.0", "chess 0.1.0", Some("o1"), Some("base")),
            ("o1".to_string(), "base".to_string())
        );
    }

    #[test]
    fn identical_pgn_names_are_an_error_not_white() {
        let games = vec![PgnGame { round: 1, white: "same".into(), black: "same".into(), white_points: 1.0 }];
        // Before the guard this scored White's point as the arm's — the exact
        // failure that made `chess elo` on a self-match measure White.
        assert!(stats_from_games(&games, "same").is_err());
    }
}
