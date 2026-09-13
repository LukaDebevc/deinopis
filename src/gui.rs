//! A local web board, so the engine can actually be watched.
//!
//! Deliberately dependency-free: a small HTTP/1.1 server over `std::net` and
//! one self-contained HTML page. Adding `axum`/`serde` for this would drag a
//! hundred crates into an engine whose entire point is control over what the
//! CPU does.
//!
//! Two properties are load-bearing:
//!
//! **The server is stateless.** Every request carries the whole game (base FEN
//! + moves). The engine therefore sees the real repetition history and
//! evaluates threefold draws correctly, and "step back through the game" costs
//! the client nothing but a shorter move list — no server-side undo stack to
//! get out of sync with what is on screen.
//!
//! **Search is streamed, not awaited.** `/api/think` emits one JSON object per
//! completed iteration, on the same connection, as the search produces them.
//! This is the whole point of the GUI: watching depth, score and PV move is
//! how you *see* a search behaving badly — a score oscillating between
//! iterations, a PV that changes every depth, a node count that explodes at
//! one depth — none of which is visible from the final move alone.

use crate::board::Board;
use crate::chess_move::{Move, MoveList};
use crate::qeval::DefaultEval;
use crate::game;
use crate::movegen::{generate, GenType};
use crate::san;
use crate::search::{score_to_uci, Limits, Params, SearchResult, Searcher, Shared, ThreadData};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Server-side engine resources. The `Arc<Shared>` is swapped wholesale when
/// the hash size changes, because rebuilding the table needs exclusive access
/// and a fresh one is simpler than resizing under a running search.
struct Engine {
    shared: Mutex<Arc<Shared>>,
    hash_mb: AtomicUsize,
    /// One search at a time. Held for the duration of a `/api/think`, so a
    /// second request queues instead of two searches fighting over one TT.
    searching: Mutex<()>,
}

impl Engine {
    fn snapshot(&self) -> Arc<Shared> {
        Arc::clone(&self.shared.lock().unwrap())
    }

    fn set_hash(&self, mb: usize) {
        let mb = mb.clamp(1, 8192);
        if mb == self.hash_mb.load(Ordering::Relaxed) {
            return;
        }
        let mut guard = self.shared.lock().unwrap();
        *guard = Arc::new(Shared::new(mb));
        self.hash_mb.store(mb, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------- request body

/// Request bodies are `key value` lines. Not JSON, because parsing JSON
/// without a dependency is more code than this whole module needs, and the
/// only structured field is a list of moves separated by spaces.
struct Req(Vec<(String, String)>);

impl Req {
    fn parse(body: &str) -> Req {
        Req(body
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                if l.is_empty() {
                    return None;
                }
                let (k, v) = l.split_once(' ').unwrap_or((l, ""));
                Some((k.to_string(), v.trim().to_string()))
            })
            .collect())
    }

    fn get(&self, key: &str) -> &str {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or("")
    }

    fn num<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.get(key).parse().unwrap_or(default)
    }

    fn flag(&self, key: &str) -> bool {
        matches!(self.get(key), "1" | "true")
    }
}

/// Rebuild a position from base FEN + moves, keeping the keys of every
/// position along the way so repetition is judged on the real game.
fn replay(fen: &str, moves: &str) -> Result<(Board, Vec<u64>, Vec<Move>), String> {
    let fen = if fen.trim().is_empty() {
        crate::board::START_FEN
    } else {
        fen.trim()
    };
    let mut b = Board::from_fen(fen)?;
    let mut keys = Vec::new();
    let mut played = Vec::new();
    for tok in moves.split_whitespace() {
        let mut list = MoveList::new();
        generate(&b, GenType::All, &mut list);
        match list.find_uci(tok) {
            Some(m) => {
                keys.push(b.key());
                played.push(m);
                b = b.make_move(m);
            }
            None => return Err(format!("illegal move {tok}")),
        }
    }
    Ok((b, keys, played))
}

/// Apply the request's "play for win" settings.
///
/// Two fields, because contempt needs both halves and the GUI is stateless:
/// `winonly 1` sets what a draw is worth to the engine (0 instead of 1/2), and
/// `engine w|b` says which side that engine is. Without the second one the
/// eval bar would be read from whoever happens to be on move — so a position
/// would score +200 for you and −200 for the engine on alternate plies, which
/// is not contempt, it is a sign error with a UI.
///
/// The search sets the root itself from the side to move, which is the same
/// answer whenever the engine is the one thinking. This call is what makes the
/// EVAL BAR agree on the human's turn too.
fn apply_contempt(req: &Req) {
    crate::eval::set_draw_value(if req.flag("winonly") { 0.0 } else { 0.5 });
    let engine_is_white = match req.get("engine") {
        "w" => true,
        "b" => false,
        // "both sides are me" (analysis): nobody is the contempt holder, so
        // fall back to the side to move and let it mean "play for a win from
        // here", which is the only reading that makes sense with no engine.
        _ => return,
    };
    crate::eval::set_root(if engine_is_white {
        crate::types::Color::White
    } else {
        crate::types::Color::Black
    });
}

fn json_strings(v: impl IntoIterator<Item = String>) -> String {
    v.into_iter()
        .map(|s| format!("\"{s}\""))
        .collect::<Vec<_>>()
        .join(",")
}

/// Resident set size, for the "what is this costing" panel. Linux-only and
/// best-effort: the point is an order of magnitude, not an audit.
fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()))
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}

fn state_json(start_fen: &str, b: &Board, keys: &[u64], played: &[Move], eng: &Engine) -> String {
    let mut list = MoveList::new();
    generate(b, GenType::All, &mut list);
    let moves: Vec<String> = list.iter().map(|m| m.to_uci()).collect();
    // SAN for every legal move, so the client can show a real move list and
    // real move names without reimplementing disambiguation in JavaScript.
    let sans: Vec<String> = list.iter().map(|m| san::san(b, m)).collect();

    // SAN is only meaningful relative to where the game started, so the move
    // list is rebuilt from the base position the client sent, not from startpos.
    let start = Board::from_fen(start_fen).unwrap_or_else(|_| Board::startpos());
    let history: Vec<String> = san::san_line(&start, played);
    let shared = eng.snapshot();

    // The eval the ENGINE actually uses, in the same precedence the search
    // applies: WDL net, else the quadratic net, else the PeSTO control arm.
    //
    // This used to be `evaluate_pst` unconditionally, so the bar showed PeSTO
    // while the engine searched with a net — two different evaluations of the
    // same position, one of them on screen. That is worse than no bar, and it
    // gets worse still with `DrawValue`, whose entire visible effect is that
    // the eval changes. Rebuilt from scratch rather than incrementally: this
    // is one call per human move, not per node.
    let cp = if let Some(n) = crate::wdleval::net() {
        n.evaluate(b)
    } else if let Some(n) = crate::qeval::net() {
        n.evaluate(b)
    } else {
        crate::eval::evaluate_pst(b)
    };
    // Side-to-move relative, as UCI requires. The board is a white-perspective
    // display — an eval bar that flips meaning every ply is worse than no eval
    // bar — so it is converted here, once, at the boundary.
    let eval_white = if b.stm() == crate::types::Color::White { cp } else { -cp };

    format!(
        "{{\"fen\":\"{}\",\"stm\":\"{}\",\"check\":{},\"status\":\"{}\",\"eval\":{},\
         \"fullmove\":{},\"halfmove\":{},\"moves\":[{}],\"san\":[{}],\"history\":[{}],\
         \"hashMb\":{},\"hashBytes\":{},\"hashfull\":{},\"threads\":1,\"rss\":{},\
         \"drawValue\":{}}}",
        b.to_fen(),
        if b.stm() == crate::types::Color::White { "w" } else { "b" },
        b.in_check(),
        game::status(b, keys).tag(),
        eval_white,
        b.fullmove(),
        b.halfmove(),
        json_strings(moves),
        json_strings(sans),
        json_strings(history),
        eng.hash_mb.load(Ordering::Relaxed),
        shared.tt.size_bytes(),
        shared.tt.hashfull(),
        rss_bytes(),
        crate::eval::draw_value(),
    )
}

fn info_json(
    root: &Board,
    r: &SearchResult,
    elapsed: Duration,
    seldepth: usize,
    shared: &Shared,
) -> String {
    let ms = elapsed.as_millis().max(1) as u64;
    let pv_uci: Vec<String> = r.pv.iter().map(|m| m.to_uci()).collect();
    let pv_san = san::san_line(root, &r.pv);
    format!(
        "{{\"type\":\"info\",\"depth\":{},\"seldepth\":{},\"score\":\"{}\",\"cp\":{},\
         \"nodes\":{},\"nps\":{},\"ms\":{},\"hashfull\":{},\"rss\":{},\
         \"pv\":[{}],\"pvSan\":[{}]}}",
        r.depth,
        seldepth,
        score_to_uci(r.score),
        r.score,
        r.nodes,
        r.nodes * 1000 / ms,
        ms,
        shared.tt.hashfull(),
        rss_bytes(),
        json_strings(pv_uci),
        json_strings(pv_san),
    )
}

// ---------------------------------------------------------------- handlers

/// Stream one search, one JSON object per completed iteration.
///
/// The response has no `Content-Length`; it ends when the connection closes.
/// If the client goes away mid-search the write fails, and that sets the stop
/// flag — closing the tab must not leave a search burning a core.
fn handle_think(mut stream: TcpStream, req: &Req, eng: &Engine) -> std::io::Result<()> {
    apply_contempt(req);
    let (board, keys, _played) = match replay(req.get("fen"), req.get("moves")) {
        Ok(v) => v,
        Err(e) => return respond(stream, "400 Bad Request", "application/json", &format!("{{\"error\":\"{e}\"}}")),
    };

    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n\
          Cache-Control: no-store\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n",
    )?;
    stream.flush()?;

    let _guard = eng.searching.lock().unwrap();
    let shared = eng.snapshot();
    shared.stop.store(false, Ordering::Relaxed);

    let limits = if req.flag("infinite") {
        Limits { infinite: true, ..Default::default() }
    } else if req.num::<u32>("depth", 0) > 0 {
        Limits { depth: Some(req.num("depth", 12)), ..Default::default() }
    } else {
        Limits { movetime: Some(req.num("movetime", 1000)), ..Default::default() }
    };

    let mut out = stream.try_clone()?;
    let stop_on_write_failure = |e: std::io::Error, shared: &Shared| {
        shared.stop.store(true, Ordering::Relaxed);
        e
    };
    let root = board;
    let sh = Arc::clone(&shared);
    let mut emit = |r: &SearchResult, el: Duration, seldepth: usize| {
        let line = info_json(&root, r, el, seldepth, &sh);
        if let Err(e) = out
            .write_all(line.as_bytes())
            .and_then(|_| out.write_all(b"\n"))
            .and_then(|_| out.flush())
        {
            stop_on_write_failure(e, &sh);
        }
    };

    let mut s = Searcher::new(&shared, ThreadData::new(0), DefaultEval::default(), Params::default());
    let r = s.go(&board, &keys, &limits, Some(&mut emit));

    let after = if r.best_move.is_some() {
        let mut l = MoveList::new();
        generate(&board, GenType::All, &mut l);
        if l.contains(r.best_move) {
            san::san(&board, r.best_move)
        } else {
            String::new()
        }
    } else {
        String::new()
    };
    let tail = format!(
        "{{\"type\":\"bestmove\",\"move\":\"{}\",\"san\":\"{}\",\"score\":\"{}\",\"cp\":{},\
          \"depth\":{},\"nodes\":{}}}\n",
        r.best_move.to_uci(),
        after,
        score_to_uci(r.score),
        r.score,
        r.depth,
        r.nodes
    );
    stream.write_all(tail.as_bytes())?;
    stream.flush()
}

/// The position the game started from, defaulting to the standard array.
fn base_fen(req: &Req) -> &str {
    let f = req.get("fen").trim();
    if f.is_empty() {
        crate::board::START_FEN
    } else {
        f
    }
}

fn respond(mut stream: TcpStream, status: &str, ctype: &str, payload: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\
         Connection: close\r\nCache-Control: no-store\r\n\r\n",
        payload.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.write_all(payload.as_bytes())?;
    stream.flush()
}

fn handle(stream: TcpStream, eng: Arc<Engine>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length.min(1 << 20)];
    if !body.is_empty() {
        reader.read_exact(&mut body)?;
    }
    let req = Req::parse(&String::from_utf8_lossy(&body));

    match (method.as_str(), path.as_str()) {
        ("GET", "/") => respond(stream, "200 OK", "text/html; charset=utf-8", PAGE),
        (_, "/api/state") => match {
            apply_contempt(&req);
            replay(req.get("fen"), req.get("moves"))
        } {
            Ok((b, keys, played)) => respond(
                stream,
                "200 OK",
                "application/json",
                &state_json(base_fen(&req), &b, &keys, &played, &eng),
            ),
            Err(e) => respond(
                stream,
                "400 Bad Request",
                "application/json",
                &format!("{{\"error\":\"{e}\"}}"),
            ),
        },
        (_, "/api/think") => handle_think(stream, &req, &eng),
        (_, "/api/stop") => {
            eng.snapshot().stop.store(true, Ordering::Relaxed);
            respond(stream, "200 OK", "application/json", "{\"ok\":true}")
        }
        (_, "/api/config") => {
            let mb = req.num::<usize>("hash", 0);
            if mb > 0 {
                // Wait for any running search to finish before swapping the
                // table out from under it.
                eng.snapshot().stop.store(true, Ordering::Relaxed);
                let _g = eng.searching.lock().unwrap();
                eng.set_hash(mb);
            }
            let shared = eng.snapshot();
            respond(
                stream,
                "200 OK",
                "application/json",
                &format!(
                    "{{\"hashMb\":{},\"hashBytes\":{},\"threads\":1,\"rss\":{}}}",
                    eng.hash_mb.load(Ordering::Relaxed),
                    shared.tt.size_bytes(),
                    rss_bytes()
                ),
            )
        }
        _ => respond(stream, "404 Not Found", "text/plain", "not found"),
    }
}

pub fn serve(port: u16, hash_mb: usize) -> std::io::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let eng = Arc::new(Engine {
        shared: Mutex::new(Arc::new(Shared::new(hash_mb))),
        hash_mb: AtomicUsize::new(hash_mb),
        searching: Mutex::new(()),
    });
    println!("board at http://127.0.0.1:{port}   (ctrl-c to stop)");
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let eng = Arc::clone(&eng);
        std::thread::spawn(move || {
            let _ = handle(stream, eng);
        });
    }
    Ok(())
}

const PAGE: &str = include_str!("gui.html");
