//! Tuning the price list without playing games.
//!
//! # Why this exists
//!
//! `Pricing` has twelve parameters and games are the only honest measure of
//! strength — but a 5-Elo effect needs thousands of games, and a gradient step
//! needs tens of evaluations. Tuning search control against Elo directly is a
//! throughput problem we lose by two orders of magnitude.
//!
//! So the objective here is a **proxy with a specific, checkable relationship
//! to the thing we want**: a cheap search should choose what an expensive
//! search would have chosen.
//!
//! 1. Take positions from real games.
//! 2. Label each one *once*, expensively: search the top few root moves at a
//!    large node limit and record what each is worth. That is the reference
//!    opinion — "depth n+k".
//! 3. Score a candidate price list by running the search at a *small* budget —
//!    "depth n" — and asking how much the reference thinks its choice cost.
//!    That is the **regret**, in centipawns.
//!
//! Minimising regret at a fixed budget is exactly "make the shallow search
//! agree with the deep one", which is what search control is for. Labels are
//! computed once and reused for every candidate, so a full evaluation costs
//! `positions * budget` and nothing more.
//!
//! The budget is a `Budget`: nodes, or **work units** where one unit is one
//! main-search node at the deploy configuration. Fixed nodes is the historical
//! setting and the one every number on record used; fixed work is the honest
//! one, because a quiescence node costs 0.275 of a main-search node and the
//! price list is free to move effort between the two. Either can be given per
//! position or as a total for the pass (`per_position`), which is the knob that
//! trades coverage against resolution at fixed cost.
//!
//! # What this is not
//!
//! It is a proxy, and the project rule is that a proxy winning is not a result.
//! Three ways it can lie, all worth remembering before believing a number:
//!
//! * **The labels come from the same engine.** A price list that steers the
//!   shallow search toward the deep search's *mistakes* scores perfectly here.
//!   The labels are a strong opinion, not ground truth.
//! * **The positions may come from our own games**, in which case the
//!   distribution is the one this engine already reaches, not the one a better
//!   engine would reach. `positions_from_fens` is the fix — a corpus drawn from
//!   a binpack of strong-engine games — and it is the cheap half of the
//!   problem. `positions_from_pgn` is still there and still carries this bias.
//! * **Regret at fixed nodes is not Elo.** It ignores time management, the
//!   value of a stable PV, and everything that happens across moves.
//!
//! Its job is to *propose*. `tools/checkpoint.sh` still decides.
//!
//! # The awkward part: the objective is a step function
//!
//! Search decisions are discrete, so for one position the regret is piecewise
//! constant in the parameters and its true gradient is zero almost everywhere.
//! Averaging over a thousand positions is what rescues it: the sum is still a
//! step function, but with a thousand small steps instead of one big one, and a
//! finite difference over a large enough perturbation sees a real slope. This
//! is the same reason SPSA works on Elo — except that here an evaluation costs
//! seconds instead of hours, so we can afford a full two-sided gradient in
//! every coordinate rather than one noisy random projection.

use crate::board::Board;
use crate::chess_move::{Move, MoveList};
use crate::eval::Score;
use crate::qeval::DefaultEval;
use crate::movegen::{generate, GenType};
use crate::san::san;
use crate::search::{Limits, Params, SearchResult, Searcher, Shared, ThreadData};

/// One position, with the reference search's opinion of its best few moves.
#[derive(Clone)]
pub struct Labelled {
    pub fen: String,
    /// `(move, score)` from the side to move's point of view, best first.
    pub moves: Vec<(Move, Score)>,
}

impl Labelled {
    fn best(&self) -> Score {
        self.moves.first().map(|m| m.1).unwrap_or(0)
    }
    fn worst(&self) -> Score {
        self.moves.last().map(|m| m.1).unwrap_or(0)
    }
    fn score_of(&self, mv: Move) -> Option<Score> {
        self.moves.iter().find(|m| m.0 == mv).map(|m| m.1)
    }
}

// ---------------------------------------------------------------- positions

/// Replay a PGN's movetext into positions.
///
/// SAN is parsed by generating the legal moves and rendering each one with
/// `san::san` until the strings match. That is quadratic and completely
/// indifferent to disambiguation, check suffixes and castling notation — the
/// writer is the oracle, so the parser cannot disagree with it.
fn replay(text: &str, out: &mut Vec<Board>, skip_plies: usize, stride: usize) {
    let mut b = Board::from_fen(crate::board::START_FEN).unwrap();
    let mut ply = 0usize;
    for tok in movetext_tokens(text) {
        let mut moves = MoveList::new();
        generate(&b, GenType::All, &mut moves);
        let mut found = None;
        for i in 0..moves.len() {
            let mv = moves.get(i);
            if san(&b, mv) == tok {
                found = Some(mv);
                break;
            }
        }
        let Some(mv) = found else { return }; // unparseable: abandon the game
        b = b.make_move(mv);
        ply += 1;
        if ply >= skip_plies && ply % stride == 0 && !b.in_check() {
            out.push(b);
        }
    }
}

/// Strip everything from PGN movetext that is not a move: headers, comments,
/// variations, move numbers, NAGs and the result.
fn movetext_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            continue;
        }
        let mut s = String::with_capacity(line.len());
        let (mut depth_brace, mut depth_paren) = (0i32, 0i32);
        for c in line.chars() {
            match c {
                '{' => depth_brace += 1,
                '}' => depth_brace -= 1,
                '(' => depth_paren += 1,
                ')' => depth_paren -= 1,
                _ if depth_brace == 0 && depth_paren == 0 => s.push(c),
                _ => {}
            }
        }
        for tok in s.split_whitespace() {
            let tok = tok.trim_end_matches(['!', '?']);
            if tok.is_empty()
                || tok.starts_with('$')
                || matches!(tok, "1-0" | "0-1" | "1/2-1/2" | "*")
                || tok.chars().next().is_some_and(|c| c.is_ascii_digit())
            {
                continue;
            }
            out.push(tok.to_string());
        }
    }
    out
}

/// Positions from a set of PGN files, deduplicated by zobrist key.
///
/// `skip_plies` drops the book, `stride` decorrelates neighbouring positions
/// within a game — consecutive plies share almost all of their tree, so
/// sampling every one buys sample size that is not there.
pub fn positions_from_pgn(
    paths: &[String],
    skip_plies: usize,
    stride: usize,
    max: usize,
) -> Result<Vec<Board>, String> {
    let mut raw = Vec::new();
    for p in paths {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
        // Games are separated by the next header block. PGN puts a blank line
        // between the headers and the movetext, which is the only reliable
        // place to cut — the tag lines themselves are not all bracketed once
        // the chunk has been split on `[Event `.
        for chunk in text.split("[Event ") {
            if chunk.trim().is_empty() {
                continue;
            }
            let body = chunk.split_once("\n\n").map(|(_, b)| b).unwrap_or(chunk);
            replay(body, &mut raw, skip_plies, stride);
        }
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for b in raw {
        if seen.insert(b.key()) {
            out.push(b);
            if out.len() >= max {
                break;
            }
        }
    }
    Ok(out)
}

/// Draw a corpus from a flat file of FENs, one per line.
///
/// # Why not our own games
///
/// `positions_from_pgn` samples the distribution *this engine already reaches*,
/// which is one of the three ways the module header says this objective can lie.
/// A binpack of strong-engine games does not fix the other two — the labels are
/// still our own search — but it does fix this one, and it is the cheap half.
/// Produce the file with the extractor that already exists:
///
/// ```text
/// nnue/extract/target/release/nnue-extract --input <x.binpack> --output /dev/null \
///     --mark --game-stride 401 --fen-dump corpus.fens
/// ```
///
/// # The sample is nested in `n`
///
/// Positions are ranked by a hash of `(seed, fen)` and the lowest `n` are kept.
/// That ordering does not depend on `n`, so the 1000-position corpus is a strict
/// **subset** of the 14000-position one at the same seed. Lifting the size later
/// therefore costs only the labels for the positions that are new — the old ones
/// stay valid — which is what makes "start at 1k and raise it when the noise
/// floor starts to bind" a cheap plan rather than a re-labelling every time.
/// Changing `seed` draws a fresh, disjoint-in-expectation corpus.
///
/// Two filters, both matching what `replay` already does for PGNs: positions in
/// check are dropped (the root has almost no choice to make, so they carry
/// little signal about move ordering) and anything before `min_fullmove` is
/// dropped as book. Duplicate positions are dropped by board key.
pub fn positions_from_fens(
    path: &str,
    n: usize,
    seed: u64,
    min_fullmove: u32,
) -> Result<Vec<Board>, String> {
    use std::collections::{BinaryHeap, HashSet};
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    // A max-heap of the best `n` so far: the largest hash is on top and is the
    // one that gets displaced. O(N log n) with only n FENs ever held.
    let mut heap: BinaryHeap<(u64, String)> = BinaryHeap::new();
    let mut seen: HashSet<u64> = HashSet::new();
    let (mut lines, mut parsed) = (0usize, 0usize);
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        lines += 1;
        // The cheap text filter first, so the parse only runs on survivors.
        let fullmove: u32 = line.split_whitespace().nth(5).and_then(|f| f.parse().ok()).unwrap_or(1);
        if fullmove < min_fullmove {
            continue;
        }
        let Ok(b) = Board::from_fen(line) else { continue };
        if b.in_check() {
            continue;
        }
        if !seen.insert(b.key()) {
            continue;
        }
        parsed += 1;
        let h = fen_hash(seed, line);
        if heap.len() < n {
            heap.push((h, line.to_string()));
        } else if let Some(top) = heap.peek() {
            if h < top.0 {
                heap.pop();
                heap.push((h, line.to_string()));
            }
        }
    }
    if heap.len() < n {
        eprintln!(
            "  note: asked for {n} positions, {path} yielded only {} ({lines} lines, {parsed} usable)",
            heap.len()
        );
    }
    let mut kept: Vec<(u64, String)> = heap.into_vec();
    kept.sort_unstable();
    Ok(kept.iter().map(|(_, f)| Board::from_fen(f).unwrap()).collect())
}

/// FNV-1a over the seed then the FEN. Same construction as `split`, so the two
/// samplers cannot accidentally correlate through a shared hash of the FEN
/// alone: the corpus draw is seeded and the train/test split is not.
fn fen_hash(seed: u64, fen: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in seed.to_le_bytes().iter().chain(fen.as_bytes()) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------- labelling

/// What one scoring search is allowed to spend.
///
/// `Nodes` is what every number on record was measured with. `Work` is the same
/// idea in the unit the machine actually pays: one unit is one main-search node
/// at the deploy configuration (`src/work.rs`).
///
/// The difference is not pedantry. A quiescence node costs **0.275** of a
/// main-search node (`library/007`), so a fixed-*node* budget quietly subsidises
/// any candidate that shifts effort into quiescence — and quiescence is 44.9% of
/// search time (LEDGER 059), controlled by exactly the parameters being tuned. A
/// price list can therefore "win" at fixed nodes by buying cheap nodes, which is
/// not a win at fixed time. At fixed work it cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Budget {
    Nodes(u64),
    Work(u64),
}

impl Budget {
    fn limits(self) -> Limits {
        match self {
            Budget::Nodes(n) => Limits { nodes: Some(n), ..Default::default() },
            Budget::Work(w) => Limits { work: Some(w), ..Default::default() },
        }
    }

    /// Scale a budget, with a floor. Used for the cheap ranking pass in
    /// `label_one`.
    fn scaled(self, num: u64, den: u64, floor: u64) -> Budget {
        match self {
            Budget::Nodes(n) => Budget::Nodes((n * num / den).max(floor)),
            Budget::Work(w) => Budget::Work((w * num / den).max(floor)),
        }
    }

    /// A work ceiling is enforced by a counter that only exists under
    /// `--features work`. In a plain build `Limits::work` is ignored, so the
    /// search would run to `MAX_PLY` and return a number that looks fine — a
    /// wrong answer rather than an error. Refuse up front instead.
    pub fn available(self) -> Result<(), String> {
        match self {
            Budget::Work(_) if !cfg!(feature = "work") => Err(
                "a work budget needs a binary built with --features work \
                 (cargo build --release --features work); or score with --nodes"
                    .to_string(),
            ),
            _ => Ok(()),
        }
    }
}

/// Split a total budget for a whole scoring pass across `n` positions.
///
/// This is the knob that trades **coverage against resolution** at fixed cost.
/// The corpus and the per-position budget both cost linearly, so a run is
/// `n * per_position` either way; asking for the total instead of the
/// per-position budget makes a pass take the same wall time whatever `n` is,
/// which is what makes "score 1000 positions in 30 s" a specification rather
/// than a hope.
///
/// It has a floor, and the floor is measured, not aesthetic. LEDGER 014 swept
/// the same parameter at five budgets over 4000 positions: contrast-to-noise
/// (worst arm minus best, in standard errors) is **1.9 at 1k nodes, 2.0 at 4k,
/// 4.6 at 15k** and flat above that. At 1k the student reaches depth 3.0, there
/// is no tree left to over-prune, and the known-bad over-pruning arm scores
/// *better* than the default — the proxy has the wrong sign. So buying coverage
/// by shrinking the student below ~15k nodes buys a number that points the wrong
/// way; buy it by raising the total instead. `warn_if_cheap` says so out loud.
pub fn per_position(total: Budget, n: usize) -> Budget {
    let n = n.max(1) as u64;
    match total {
        Budget::Nodes(t) => Budget::Nodes((t / n).max(1)),
        Budget::Work(t) => Budget::Work((t / n).max(1)),
    }
}

/// Print the LEDGER 014 warning if a per-position budget is below the measured
/// operating point. Work units are compared on the same scale: one unit is one
/// main-search node, and 15000 nodes of a real search is ~15000 units.
pub fn warn_if_cheap(b: Budget) {
    let v = match b {
        Budget::Nodes(n) => n,
        Budget::Work(w) => w,
    };
    if v < 15_000 {
        eprintln!(
            "  warning: {b} per position is below the 15k operating point. LEDGER 014: \n\
             \x20 contrast-to-noise falls 4.6 -> 2.0 -> 1.9 at 15k -> 4k -> 1k, and at 1k the \n\
             \x20 known-bad over-pruning arm scores BETTER than the default. More positions at \n\
             \x20 a budget this small is more coverage of a metric with the wrong sign."
        );
    }
}

impl std::fmt::Display for Budget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Budget::Nodes(n) => write!(f, "{n} nodes"),
            Budget::Work(w) => write!(f, "{w} work units"),
        }
    }
}

/// Run one search to a hard budget, from a clean table.
///
/// The table is cleared rather than aged: two evaluations of the same candidate
/// must produce the same number, and a table carrying another position's
/// entries would make the objective depend on the order positions were visited.
fn search_once(shared: &Shared, b: &Board, budget: Budget, params: Params) -> (Move, Score, u32, u64, u64, u64) {
    shared.tt.clear();
    shared.stop.store(false, std::sync::atomic::Ordering::Relaxed);
    let mut s = Searcher::new(shared, ThreadData::new(0), DefaultEval::default(), params);
    let r = s.go(b, &[], &budget.limits(), None);
    (r.best_move, r.score, r.depth, r.nodes, s.td.util_searched, s.td.util_useful)
}

/// Label one position: what the reference search thinks the best `top_k` moves
/// are worth.
///
/// Two passes. A cheap one ranks every root move so the expensive one is not
/// spent on moves nothing would ever play, then the survivors are re-searched
/// at full strength. Each child is searched from a cleared table, so a move's
/// label does not depend on which of its siblings was labelled first.
fn label_one(shared: &Shared, b: &Board, teacher: Budget, top_k: usize) -> Labelled {
    let params = Params::default();
    let mut moves = MoveList::new();
    generate(b, GenType::All, &mut moves);

    let rank = teacher.scaled(1, 16, 2000);
    let mut scored: Vec<(Move, Score)> = (0..moves.len())
        .map(|i| {
            let mv = moves.get(i);
            let child = b.make_move(mv);
            let (_, s, _, _, _, _) = search_once(shared, &child, rank, params);
            (mv, -s)
        })
        .collect();
    scored.sort_by_key(|m| -m.1);
    scored.truncate(top_k);

    let mut out: Vec<(Move, Score)> = scored
        .iter()
        .map(|&(mv, _)| {
            let child = b.make_move(mv);
            let (_, s, _, _, _, _) = search_once(shared, &child, teacher, params);
            (mv, -s)
        })
        .collect();
    out.sort_by_key(|m| -m.1);
    Labelled {
        fen: b.to_fen(),
        moves: out,
    }
}

/// Label a set of positions, in parallel.
///
/// The teacher is budgeted in **nodes**, deliberately, even when candidates are
/// scored at fixed work. A label set is reused for months; a node count is the
/// same search forever, while a work budget shifts the moment anyone re-freezes
/// the price table in `work.rs`. The reason to score candidates at fixed work —
/// that the price list can move effort between main and quiescence nodes — does
/// not apply to the teacher, which runs one fixed price list.
pub fn label(positions: &[Board], teacher: Budget, top_k: usize, threads: usize) -> Vec<Labelled> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<Labelled>>> =
        (0..positions.len()).map(|_| std::sync::Mutex::new(None)).collect();

    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let shared = Shared::new(32);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= positions.len() {
                        break;
                    }
                    let l = label_one(&shared, &positions[i], teacher, top_k);
                    *out[i].lock().unwrap() = Some(l);
                    let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if d % 25 == 0 {
                        eprint!("\r  labelled {d}/{}", positions.len());
                    }
                }
            });
        }
    });
    eprintln!("\r  labelled {}/{}   ", positions.len(), positions.len());
    out.into_iter().filter_map(|m| m.into_inner().unwrap()).collect()
}

/// `header` lines record where the corpus came from and what labelled it.
/// A label file outlives the session that made it, and a set of labels whose
/// teacher budget nobody wrote down cannot be extended or compared later.
pub fn write_labels(path: &str, labels: &[Labelled], header: &[String]) -> Result<(), String> {
    let mut s = String::new();
    for h in header {
        s.push_str("# ");
        s.push_str(h);
        s.push('\n');
    }
    s.push_str("# fen\tmove:score ...\n");
    for l in labels {
        s.push_str(&l.fen);
        for (mv, sc) in &l.moves {
            s.push('\t');
            s.push_str(&format!("{}:{}", mv.to_uci(), sc));
        }
        s.push('\n');
    }
    std::fs::write(path, s).map_err(|e| format!("{path}: {e}"))
}

pub fn read_labels(path: &str) -> Result<Vec<Labelled>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut it = line.split('\t');
        let fen = it.next().ok_or("empty line")?.to_string();
        let b = Board::from_fen(&fen)?;
        // Moves are stored as UCI; resolve them against the legal list so a
        // corrupt label file fails here rather than silently scoring nothing.
        let mut legal = MoveList::new();
        generate(&b, GenType::All, &mut legal);
        let mut moves = Vec::new();
        for tok in it {
            let (m, s) = tok.split_once(':').ok_or_else(|| format!("bad label {tok}"))?;
            let sc: Score = s.parse().map_err(|_| format!("bad score {s}"))?;
            let mv = (0..legal.len())
                .map(|i| legal.get(i))
                .find(|mv| mv.to_uci() == m)
                .ok_or_else(|| format!("{fen}: illegal labelled move {m}"))?;
            moves.push((mv, sc));
        }
        out.push(Labelled { fen, moves });
    }
    Ok(out)
}

// ---------------------------------------------------------------- objective

#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    pub positions: usize,
    /// Mean centipawns given up against the reference opinion. The objective.
    pub regret: f64,
    /// Standard error of that mean over the position sample. The objective is
    /// deterministic, so this is *not* run-to-run noise — it is how well this
    /// many positions pins down the number, and therefore the smallest
    /// difference between two price lists that means anything on a fresh
    /// sample. Differences smaller than this are fitting the sample.
    pub stderr: f64,
    /// Fraction of positions where the cheap search picked the reference's move.
    pub agree: f64,
    /// Fraction where it picked a move the reference never looked at. If this
    /// is large the label set's `top_k` is too small and `regret` is a floor,
    /// not a measurement. Measured at 4.6-10.5% (LEDGER 010), so it is not.
    pub off_list: f64,
    /// Mean depth of the last completed iteration, at the same node budget.
    ///
    /// **This is a constraint, not a second objective.** LEDGER 009/010:
    /// deleting the whole price list moves `regret` by +1.37 cp (t = 3.4) and
    /// `depth` by -2.91 plies (t = -231). Depth is where under-pruning is
    /// visible, and it costs nothing — it needs no labels at all.
    ///
    /// It must never be optimised on its own or summed into `regret`: pricing
    /// harder buys depth *and* loses Elo, so the known-bad `over2.4` arm scores
    /// +3.1 plies against +4.9 cp and any linear combination rewards it. Read
    /// it as a floor a candidate may not fall through.
    pub depth: f64,
    /// Mean nodes actually searched per position. Constant by construction
    /// under a node budget; under a **work** budget it is the free readout of
    /// how the candidate spent its money — a price list that shifted effort
    /// into quiescence buys more nodes for the same work, and that shows up
    /// here rather than being silently rewarded.
    pub nodes: f64,
    /// Node utilisation: fraction of searched priced children that raised
    /// alpha or cut off (library/005 item 2). The precision readout regret is
    /// blind to — deleting the whole price list barely moves regret and tanks
    /// this. A constraint like depth, never an objective: searching one move
    /// everywhere maximises it and loses every game.
    pub util: f64,
}

/// Mean regret of `params` at one `budget` per position, each position's regret
/// capped at `cap` centipawns.
///
/// The cap is not cosmetic. 5% of a labelled set drawn from real games contains
/// a forced mate, where the spread between the best and worst labelled move is
/// ~32000 rather than ~100. Uncapped, those positions contribute 94% of the
/// mean and the objective is a measurement of them alone — the gradient would
/// be entirely about mate-finding and blind to everything else. Beyond a couple
/// of pawns the move is already a blunder, and "more blunder" carries no extra
/// information about how to allocate a search budget.
pub fn evaluate(
    labels: &[Labelled],
    budget: Budget,
    params: Params,
    threads: usize,
    cap: Score,
) -> Report {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let totals = std::sync::Mutex::new((0f64, 0f64, 0usize, 0usize, 0f64, 0f64, 0u64, 0u64));

    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let shared = Shared::new(16);
                let (mut regret, mut sq, mut agree, mut off) = (0f64, 0f64, 0usize, 0usize);
                let (mut depth, mut nodes) = (0f64, 0f64);
                let (mut searched, mut useful) = (0u64, 0u64);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= labels.len() {
                        break;
                    }
                    let l = &labels[i];
                    let b = Board::from_fen(&l.fen).unwrap();
                    let (mv, _, d, n, se, us) = search_once(&shared, &b, budget, params);
                    depth += d as f64;
                    nodes += n as f64;
                    searched += se;
                    useful += us;
                    let r;
                    match l.score_of(mv) {
                        Some(s) => {
                            r = (l.best() - s).clamp(0, cap) as f64;
                            if mv == l.moves[0].0 {
                                agree += 1;
                            }
                        }
                        None => {
                            // Outside the labelled set, so it is worse than
                            // every move we priced — but we do not know by how
                            // much. Charge the worst label: a floor, and the
                            // reason `off_list` is reported.
                            r = (l.best() - l.worst()).clamp(0, cap) as f64;
                            off += 1;
                        }
                    }
                    regret += r;
                    sq += r * r;
                }
                let mut t = totals.lock().unwrap();
                t.0 += regret;
                t.1 += sq;
                t.2 += agree;
                t.3 += off;
                t.4 += depth;
                t.5 += nodes;
                t.6 += searched;
                t.7 += useful;
            });
        }
    });

    let (regret, sq, agree, off, depth, nodes, searched, useful) = *totals.lock().unwrap();
    let n = labels.len().max(1) as f64;
    let mean = regret / n;
    let var = (sq / n - mean * mean).max(0.0);
    Report {
        positions: labels.len(),
        regret: mean,
        stderr: (var / n).sqrt(),
        agree: agree as f64 / n,
        off_list: off as f64 / n,
        depth: depth / n,
        nodes: nodes / n,
        util: if searched == 0 { 0.0 } else { useful as f64 / searched as f64 },
    }
}

/// Split a label set into a fitting half and a held-out half, by hash of the
/// FEN so the split is stable across runs and independent of file order.
///
/// This is not optional. The objective's dynamic range across sane price lists
/// is a couple of centipawns and its standard error over 1500 positions is of
/// the same order, so a descent run on the whole set will happily buy
/// improvements that exist only in this sample. The held-out half is the only
/// thing that says whether a fitted price list found policy or found noise.
pub fn split(labels: &[Labelled], test_fraction: f64) -> (Vec<Labelled>, Vec<Labelled>) {
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for l in labels {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in l.fen.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        if ((h >> 32) as f64 / u32::MAX as f64) < test_fraction {
            test.push(l.clone());
        } else {
            train.push(l.clone());
        }
    }
    (train, test)
}

/// A paired comparison of two price lists on the same positions.
///
/// This, not `Report`, is the instrument that matters. The standard error of a
/// single price list's mean regret is ~1 cp on 1500 positions, because
/// positions differ enormously from each other — but two price lists choose the
/// *same move* in most positions, and those contribute exactly zero to the
/// difference. Pairing removes the between-position variance that dominates the
/// mean and leaves only the variance that is actually about the price lists.
///
/// It is the same reason the match runner scores colour-reversed pairs instead
/// of independent games, applied to a cheap objective instead of to Elo.
#[derive(Clone, Copy, Debug, Default)]
pub struct Comparison {
    pub positions: usize,
    /// Mean of `regret(b) - regret(a)`. Negative means `b` is better.
    pub mean_diff: f64,
    /// Standard error of that mean, over the paired differences.
    pub stderr: f64,
    /// Positions where the two chose different moves. Everything else is a tie
    /// contributing nothing, and if this is near zero the two price lists are
    /// the same policy however different their parameters look.
    pub differ: usize,
    /// Mean depth reached at the same node count. LEDGER 010: depth is the
    /// **constraint** and regret the objective, never a term in a sum — so a
    /// comparison that does not report it can be walked straight into the
    /// under-pruning direction the objective is blind to.
    pub depth_a: f64,
    pub depth_b: f64,
    /// Mean nodes each side searched. Under a work budget these differ, and the
    /// difference is the point: it says where the candidate moved the money.
    pub nodes_a: f64,
    pub nodes_b: f64,
    /// Node utilisation per side (see `Report::util`): the precision readout.
    /// A candidate that wins regret by searching less must show it here, not
    /// hide it — and one that loses utilisation while holding regret is
    /// buying depth with waste.
    pub util_a: f64,
    pub util_b: f64,
}

pub fn compare(
    labels: &[Labelled],
    budget: Budget,
    a: Params,
    b: Params,
    threads: usize,
    cap: Score,
) -> Comparison {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let totals = std::sync::Mutex::new((0f64, 0f64, 0usize, 0f64, 0f64, 0f64, 0f64, 0u64, 0u64, 0u64, 0u64));

    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let shared = Shared::new(16);
                let (mut sum, mut sq, mut differ) = (0f64, 0f64, 0usize);
                let (mut da, mut db) = (0f64, 0f64);
                let (mut na, mut nb) = (0f64, 0f64);
                let (mut sea, mut usa, mut seb, mut usb) = (0u64, 0u64, 0u64, 0u64);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= labels.len() {
                        break;
                    }
                    let l = &labels[i];
                    let board = Board::from_fen(&l.fen).unwrap();
                    let (ma, _, dep_a, nod_a, se_a, us_a) = search_once(&shared, &board, budget, a);
                    let (mb, _, dep_b, nod_b, se_b, us_b) = search_once(&shared, &board, budget, b);
                    da += dep_a as f64;
                    db += dep_b as f64;
                    na += nod_a as f64;
                    nb += nod_b as f64;
                    sea += se_a;
                    usa += us_a;
                    seb += se_b;
                    usb += us_b;
                    if ma != mb {
                        differ += 1;
                    }
                    let regret = |mv| match l.score_of(mv) {
                        Some(s) => (l.best() - s).clamp(0, cap) as f64,
                        None => (l.best() - l.worst()).clamp(0, cap) as f64,
                    };
                    let d = regret(mb) - regret(ma);
                    sum += d;
                    sq += d * d;
                }
                let mut t = totals.lock().unwrap();
                t.0 += sum;
                t.1 += sq;
                t.2 += differ;
                t.3 += da;
                t.4 += db;
                t.5 += na;
                t.6 += nb;
                t.7 += sea;
                t.8 += usa;
                t.9 += seb;
                t.10 += usb;
            });
        }
    });

    let (sum, sq, differ, da, db, na, nb, sea, usa, seb, usb) = *totals.lock().unwrap();
    let n = labels.len().max(1) as f64;
    let mean = sum / n;
    let var = (sq / n - mean * mean).max(0.0);
    let ratio = |u: u64, s: u64| if s == 0 { 0.0 } else { u as f64 / s as f64 };
    Comparison {
        positions: labels.len(),
        mean_diff: mean,
        stderr: (var / n).sqrt(),
        differ,
        depth_a: da / n,
        depth_b: db / n,
        nodes_a: na / n,
        nodes_b: nb / n,
        util_a: ratio(usa, sea),
        util_b: ratio(usb, seb),
    }
}

// ---------------------------------------------------------------- optimiser

/// Two-sided finite-difference gradient descent with a backtracking line
/// search, over the named parameters.
///
/// Why not SPSA: SPSA exists because each objective evaluation is expensive
/// enough that you can only afford two, so it settles for one noisy projection
/// of the gradient. Here an evaluation is seconds, so a full gradient costs
/// `2n+1` evaluations and is worth having — it gives a descent direction rather
/// than a direction that descends on average.
///
/// The perturbation is deliberately coarse. The objective is piecewise constant
/// in each parameter, so a small step returns exactly zero and reports "no
/// gradient" for a parameter that matters; `h` has to be large enough to flip
/// real decisions in a real fraction of the position set.
///
/// That coarseness is why `free` defaults to the price list alone even though
/// `apply_overrides` can now reach all 20 registered parameters. `h` is
/// `max(|v|/8, 25)` in milli-plies, which is right for a price and absurd for
/// `nmp_min_depth` (whose whole range is 1..12): one step saturates it against
/// its bound and the "gradient" is the difference between two clamps. Name such
/// a parameter in `--free` deliberately if you want it, and read the trajectory
/// rather than trusting the step.
///
/// # The depth floor
///
/// `depth_floor` is the number of plies of mean search depth a candidate may
/// give up against the starting price list before it is rejected outright,
/// whatever it does to regret.
///
/// This exists because of LEDGER 009/010. Regret is *attenuated* in the
/// under-pruning direction, not blind — deleting the entire price list costs
/// 2.91 plies but moves regret by only +1.37 cp, roughly a third of its true
/// 80-150 Elo. Over a descent run those thirds accumulate in the tuner's
/// favour: it can trade real depth for sample noise and the objective will
/// thank it. Depth at fixed nodes is the direction regret is weak in, it is
/// free, and it needs no labels — so it is the natural guard.
///
/// It is a *floor*, never a term. Pricing harder buys depth and loses Elo
/// (`over2.4`: +3.1 plies, +4.9 cp), so anything that adds depth to the
/// objective is maximised by the known-bad arm.
pub fn descend(
    labels: &[Labelled],
    held_out: &[Labelled],
    budget: Budget,
    start: Params,
    free: &[String],
    iters: usize,
    threads: usize,
    cap: Score,
    depth_floor: f64,
) -> Params {
    let mut params = start;
    let mut best = evaluate(labels, budget, params, threads, cap);
    let held0 = evaluate(held_out, budget, params, threads, cap);
    let depth0 = best.depth;
    let mut blocked = 0usize;
    println!(
        "iter  0   train {:6.2} +/- {:.2}   held-out {:6.2} +/- {:.2}   agree {:4.1}%   depth {:5.2}   {}",
        best.regret,
        best.stderr,
        held0.regret,
        held0.stderr,
        best.agree * 100.0,
        best.depth,
        fmt_free(&params, free)
    );
    println!("           depth floor {:.2} plies (reject below {:.2})", depth_floor, depth0 - depth_floor);

    for it in 1..=iters {
        // ---- gradient
        let mut grad = Vec::with_capacity(free.len());
        for name in free {
            let v = params.get(name).unwrap();
            let h = (v.abs() / 8).max(25);
            let mut plus = params;
            plus.set(name, v + h);
            let mut minus = params;
            minus.set(name, v - h);
            let rp = evaluate(labels, budget, plus, threads, cap).regret;
            let rm = evaluate(labels, budget, minus, threads, cap).regret;
            grad.push((rp - rm) / (2.0 * h as f64));
        }
        let norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm < 1e-9 {
            println!("iter {it:2}   gradient is flat in every coordinate — stop");
            break;
        }

        // ---- backtracking line search. Steps are integers, so a step small
        // enough to round to zero everywhere is the real termination condition.
        let mut step = 400.0 / norm;
        let mut improved = false;
        for _ in 0..5 {
            let mut cand = params;
            let mut moved = false;
            for (name, g) in free.iter().zip(&grad) {
                let v = params.get(name).unwrap();
                let d = (-g * step).round() as i32;
                if d != 0 {
                    moved = true;
                }
                cand.set(name, v + d);
            }
            if !moved {
                break;
            }
            let r = evaluate(labels, budget, cand, threads, cap);
            // The depth floor is checked *before* the objective, so a
            // candidate that bought its regret by under-pruning is rejected
            // rather than compared. See the note on `depth_floor` above.
            if r.depth < depth0 - depth_floor {
                blocked += 1;
                step /= 2.0;
                continue;
            }
            if r.regret < best.regret {
                params = cand;
                best = r;
                improved = true;
                break;
            }
            step /= 2.0;
        }

        // The held-out set is scored but never steered on: it is the report,
        // not the objective. Reading it and then choosing the iteration where
        // it happened to dip would make it a second training set.
        let held = evaluate(held_out, budget, params, threads, cap);
        println!(
            "iter {it:2}   train {:6.2} +/- {:.2}   held-out {:6.2} +/- {:.2}   agree {:4.1}%   depth {:5.2}{}   {}",
            best.regret,
            best.stderr,
            held.regret,
            held.stderr,
            best.agree * 100.0,
            best.depth,
            if blocked > 0 { format!(" [{blocked} blocked]") } else { String::new() },
            fmt_free(&params, free)
        );
        if !improved {
            println!("           no step improved the objective — stop");
            break;
        }
    }
    params
}

fn fmt_free(p: &Params, free: &[String]) -> String {
    free.iter()
        .map(|n| format!("{n}={}", p.get(n).unwrap()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Apply a `name=value,name=value` override string to the search parameters.
///
/// Takes the whole `Params`, not just its price list: `params!` generates a
/// `set` that sees through to the nested struct, so every one of the 20
/// registered names is reachable here. It used to take `Pricing` alone, which
/// meant `--a delta_margin=400` was rejected as an unknown parameter -- the
/// tuner could not A/B any of the eight search constants that are not prices.
pub fn apply_overrides(p: &mut Params, spec: &str) -> Result<(), String> {
    for kv in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad override {kv}"))?;
        let val: i32 = v.trim().parse().map_err(|_| format!("bad value {v}"))?;
        if !p.set(k.trim(), val) {
            return Err(format!("unknown search parameter {k}"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- trajectory

/// One completed root iteration: the nodes spent when it finished, and the
/// move the engine would have played had it been stopped at that instant.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub nodes: u64,
    pub mv: Move,
    pub score: Score,
    pub depth: u32,
}

/// Run one search, recording what the root believed after every iteration.
///
/// Iterations are spaced by roughly the effective branching factor (1.88
/// measured in `library/004`), so these samples are already close to uniform in
/// `log(nodes)` — which is the measure every metric below integrates against.
/// The reason is that Elo is linear in log nodes (LEDGER 007: 4x nodes is
/// 100-140 Elo), so every doubling is worth the same and deserves equal weight.
/// Sampling per iteration is therefore not a coarse approximation to sampling
/// per node; it is nearly the right grid.
///
/// Uses the existing `InfoFn` hook, so the search itself is untouched and the
/// bench fingerprint cannot move.
pub fn trajectory(shared: &Shared, b: &Board, budget: Budget, params: Params) -> (Vec<Step>, u64) {
    shared.tt.clear();
    shared.stop.store(false, std::sync::atomic::Ordering::Relaxed);
    let mut s = Searcher::new(shared, ThreadData::new(0), DefaultEval::default(), params);
    let limits = budget.limits();
    let mut steps: Vec<Step> = Vec::new();
    let r = {
        let mut f = |r: &SearchResult, _: std::time::Duration, _: usize| {
            steps.push(Step {
                nodes: r.nodes,
                mv: r.best_move,
                score: r.score,
                depth: r.depth,
            });
        };
        s.go(b, &[], &limits, Some(&mut f))
    };
    (steps, r.nodes.max(1))
}

/// The trajectory reduced to every candidate objective at once.
///
/// They are all reductions of the same recorded data, so choosing between them
/// costs no extra searching — which is the point. The question "which of these
/// predicts Elo" is then empirical rather than an argument.
#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    /// Doublings spent on a move other than the teacher's, over `[c, N]`.
    /// This is the log-time occupation measure: `sum 1/i` in continuous form.
    /// Units are doublings, so at the measured 50-70 Elo per doubling it is an
    /// Elo number rather than a proxy needing conversion. Lower is better.
    pub d_log: f64,
    /// Fraction of *linear* nodes agreeing — the unweighted variant, kept so
    /// the weighting choice can be tested rather than assumed.
    pub f_lin: f64,
    /// Nodes after which the move never differs again, censored at `N`. The
    /// rigorous reading of "appears and stays", and deliberately fragile: one
    /// late flicker sends it to the ceiling.
    pub last_exit: f64,
    /// Times the declared move changed. Instability diagnostic, not an
    /// objective — "never change your mind" maximises it.
    pub flips: f64,
    /// 1.0 if the move finally played is the teacher's.
    pub final_agree: f64,
    /// Capped centipawn regret of the final move — the existing objective,
    /// computed on the same searches so the comparison is exactly paired.
    pub regret: f64,
    /// Centipawn-weighted occupation: `integral loss(m(t)) d log2 t` over
    /// `[c, N]`, in cp-doublings. `d_log` counts a 1 cp near-tie exactly as
    /// heavily as a blunder, and with final agreement near 50% that noise is
    /// half the mass; this weights each doubling by what holding that move
    /// actually costs. At `t = N` it degenerates to `regret`, so it is a
    /// strict generalisation of the existing objective along the time axis.
    pub d_cp: f64,
    /// 1.0 if the final move is outside the teacher's labelled top-k, where
    /// `regret` saturates to the constant `best - worst` and therefore
    /// *cancels in the paired difference*. A high rate means the objective is
    /// clipped, not insensitive — a different disease with a different cure
    /// (label more moves, not more positions).
    pub off_list: f64,
    /// Fraction of the log-window already spent holding the move the search
    /// will *finally* declare, irrespective of whether that move is the
    /// teacher's. This is self-occupancy: high means the arm settled early and
    /// stopped revising. It is the direct test of whether an occupation
    /// measure is rewarding stasis rather than accuracy.
    pub settle: f64,
    /// Mean depth of the last completed iteration. Not an objective — it is
    /// the confound check. Under-pruning *is* "less depth for the same
    /// nodes", so if an arm's depth does not move, that arm is not doing what
    /// its name claims and no other number in the row means anything.
    pub depth: f64,
}

/// Reduce one trajectory against the teacher's move.
///
/// `c` discards the head of the search. It is not a cosmetic trim: in log
/// measure the first 1000 nodes carry ~10 doublings while 1k->200k carries
/// ~7.6, so the discarded head is larger than the kept tail. Set it to where
/// the search first has an opinion worth reading.
pub fn metrics(steps: &[Step], total: u64, l: &Labelled, c: u64, cap: Score) -> Metrics {
    let m_star = match l.moves.first() {
        Some(&(m, _)) => m,
        None => return Metrics::default(),
    };
    let n = total.max(c + 1);
    let (lo, hi) = (c as f64, n as f64);
    let span = (hi / lo).log2();

    let mut s_log = 0.0;
    let mut lin_ok = 0.0;
    let mut last_exit = lo;
    let mut flips = 0.0;
    let mut d_cp = 0.0;
    let mut settle = 0.0;
    let m_final = steps.last().map(|s| s.mv);

    // Teacher's cost of holding `mv`, capped the same way `regret` is. A move
    // outside the labelled top-k is charged the worst labelled loss rather
    // than the cap, so the two agree wherever the label set is complete.
    let loss = |mv: Move| -> f64 {
        match l.score_of(mv) {
            Some(sc) => (l.best() - sc).clamp(0, cap) as f64,
            None => (l.best() - l.worst()).clamp(0, cap) as f64,
        }
    };

    for (i, st) in steps.iter().enumerate() {
        let start = (st.nodes.max(c) as f64).min(hi);
        let end = (steps.get(i + 1).map(|x| x.nodes).unwrap_or(n).max(c) as f64).min(hi);
        if i > 0 && steps[i - 1].mv != st.mv {
            flips += 1.0;
        }
        if end <= start {
            continue;
        }
        let w = (end / start).log2();
        d_cp += w * loss(st.mv);
        if Some(st.mv) == m_final {
            settle += w;
        }
        if st.mv == m_star {
            s_log += w;
            lin_ok += end - start;
        } else {
            last_exit = end;
        }
    }

    // Anything not positively credited counts as disagreement, including the
    // head before the first iteration lands: having no opinion yet is not the
    // same as holding the teacher's move.
    let regret = match steps.last().map(|s| s.mv) {
        Some(mv) => match l.score_of(mv) {
            Some(s) => (l.best() - s).clamp(0, cap) as f64,
            None => (l.best() - l.worst()).clamp(0, cap) as f64,
        },
        None => 0.0,
    };
    Metrics {
        off_list: match steps.last().map(|s| s.mv) {
            Some(mv) => (l.score_of(mv).is_none()) as u8 as f64,
            None => 0.0,
        },
        d_log: (span - s_log).max(0.0),
        d_cp: d_cp / span.max(1e-9),
        settle: settle / span.max(1e-9),
        depth: steps.last().map_or(0.0, |s| s.depth as f64),
        f_lin: lin_ok / (hi - lo).max(1.0),
        last_exit: last_exit.log2(),
        flips,
        final_agree: steps.last().map_or(0.0, |s| (s.mv == m_star) as u8 as f64),
        regret,
    }
}

/// Mean of every metric for one price list, plus its paired difference against
/// the reference arm.
#[derive(Clone, Debug, Default)]
pub struct ArmReport {
    pub name: String,
    pub mean: Metrics,
    /// Paired mean difference `this - reference`, and its standard error, for
    /// the two objectives worth testing against each other.
    pub d_log_diff: (f64, f64),
    pub regret_diff: (f64, f64),
    pub d_cp_diff: (f64, f64),
    pub depth_diff: (f64, f64),
}

/// Score several price lists on the same positions and the same trajectories.
///
/// Every arm is searched on every position inside one thread, so the pairing is
/// exact: positions where two arms behave identically contribute exactly zero
/// to the difference, and the between-position variance that dominates any
/// single arm's mean cancels. Same argument as `compare`, extended to a family
/// of metrics measured on one pass.
pub fn metric_sweep(
    labels: &[Labelled],
    budget: Budget,
    c: u64,
    cap: Score,
    arms: &[(String, Params)],
    threads: usize,
) -> Vec<ArmReport> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let rows = std::sync::Mutex::new(Vec::<Vec<Metrics>>::new());

    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let shared = Shared::new(16);
                let mut local: Vec<Vec<Metrics>> = Vec::new();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= labels.len() {
                        break;
                    }
                    let l = &labels[i];
                    let board = match Board::from_fen(&l.fen) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let row = arms
                        .iter()
                        .map(|(_, p)| {
                            let (steps, total) = trajectory(&shared, &board, budget, *p);
                            metrics(&steps, total, l, c, cap)
                        })
                        .collect();
                    local.push(row);
                }
                rows.lock().unwrap().extend(local);
            });
        }
    });

    let rows = rows.into_inner().unwrap();
    let n = rows.len().max(1) as f64;
    let paired = |f: &dyn Fn(&Metrics) -> f64, k: usize| -> (f64, f64) {
        let (mut s, mut sq) = (0.0, 0.0);
        for r in &rows {
            let d = f(&r[k]) - f(&r[0]);
            s += d;
            sq += d * d;
        }
        let m = s / n;
        (m, ((sq / n - m * m).max(0.0) / n).sqrt())
    };

    (0..arms.len())
        .map(|k| {
            let avg = |f: &dyn Fn(&Metrics) -> f64| rows.iter().map(|r| f(&r[k])).sum::<f64>() / n;
            ArmReport {
                name: arms[k].0.clone(),
                mean: Metrics {
                    d_log: avg(&|m| m.d_log),
                    f_lin: avg(&|m| m.f_lin),
                    last_exit: avg(&|m| m.last_exit),
                    flips: avg(&|m| m.flips),
                    final_agree: avg(&|m| m.final_agree),
                    regret: avg(&|m| m.regret),
                    d_cp: avg(&|m| m.d_cp),
                    depth: avg(&|m| m.depth),
                    settle: avg(&|m| m.settle),
                    off_list: avg(&|m| m.off_list),
                },
                d_log_diff: paired(&|m| m.d_log, k),
                regret_diff: paired(&|m| m.regret, k),
                d_cp_diff: paired(&|m| m.d_cp, k),
                depth_diff: paired(&|m| m.depth, k),
            }
        })
        .collect()
}

/// Parse `name=overrides;name=overrides` into price lists, `base` for an empty
/// override list.
pub fn parse_arms(spec: &str, base: Params) -> Result<Vec<(String, Params)>, String> {
    let mut out = Vec::new();
    for arm in spec.split(';').filter(|s| !s.trim().is_empty()) {
        let (name, ov) = arm.split_once(':').unwrap_or((arm, ""));
        let mut p = base;
        apply_overrides(&mut p, ov)?;
        out.push((name.trim().to_string(), p));
    }
    Ok(out)
}
