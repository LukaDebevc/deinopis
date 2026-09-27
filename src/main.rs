//! Entry point. Defaults to UCI so the binary can be handed straight to a GUI
//! or to cutechess-cli; subcommands cover the development workflow.

use chess::board::Board;
use chess::chess_move::MoveList;
use chess::qeval::DefaultEval;
use chess::movegen::{generate, GenType};
use chess::search::{score_to_uci, Limits, Params, Searcher, Shared, ThreadData};

const HELP: &str = "\
Deinopis — a chess engine (binary: chess)

USAGE
  chess                      speak UCI on stdin/stdout (default)
  chess serve [port]         local web board at http://127.0.0.1:8080
  chess play [movetime_ms]   play in the terminal
  chess perft <depth> [fen]  per-move node counts for one position
  chess perft-suite          full published perft suite (slow, minutes)
  chess bench [depth]        fixed-depth node count — the regression check
  chess evalprof [options]   where an evaluation's time goes, stage by stage
  chess nodeprof [depth]     where a NODE's time goes  (--features nodeprof)
  chess work [depth]         cost in DETERMINISTIC units  (--features work)
  chess params [--spsa]      every tunable search constant, its default and box
  chess features             the declared feature surface: group, kind, cost, state
  chess match [options]      play a match, report Elo / run an SPRT
  chess elo <pgn>...         statistics for games already played (or in flight)
                              several files pool per-file pairs (pairs never
                              span files); --combine <results.tsv> anchors a
                              gauntlet against CCRL ratings instead
  chess tune <mode> [opts]   tune the search price list without playing games
  chess trace [options]      what the price list did, one position at a time
  chess help

MATCH OPTIONS
  --opponent <path|self>     UCI engine to play against  (default: self)
  --engine <path|self>       engine under test           (default: this build)
                             a quoted '<path> <arg>...' passes arguments, so
                             one binary can play two different net files
  --games N                  games to play, rounded up to a pair  (default 200)
  --tc <spec>                8+0.08 | mt=1000 | depth=8 | nodes=200000
  --sprt <elo0,elo1>         sequential test, e.g. 0,5   (alpha=beta=0.05)
  --concurrency K            games in parallel           (default cores/2)
  --threads T                search threads per engine   (default 1; Elo per
                              game, not games per hour — needs subprocess arms)
                              sets both arms; --threads-a/--threads-b override
                              per arm (e.g. SMP 6v1: --threads-a 6)
  --hash MB                  per engine                  (default 32)
  --book <file>              opening book: UCI move lines, or EPD/FEN
                             positions, one per line (detected per line)
  --pgn <file>               write the games
  --name-a/--name-b <name>    PGN arm tags (default: each engine's id name;
                              identical tags are qualified [A]/[B]. Pass these
                              whenever both arms are builds of this engine)
  --quiet                    only the final summary

  Exit status with --sprt: 0 = H1 accepted, 1 = H0 accepted, 2 = inconclusive.

EVALPROF OPTIONS
  --positions N              tree positions in the corpus     (default 512)
  --reps N                   timing passes over it            (default 200)
  --depth D                  search depth the corpus is drawn from (default 8)
  Needs a WDL net: `--wdl <file>`. Each stage is priced as the difference
  between two prefixes of the same code the search runs, so the parts sum to
  the whole. Compare ns/MAC across stages: a stage far off the others is a
  code-generation problem, not an arithmetic one.

TRACE OPTIONS
  --fen <fen>                position (default: the bench corpus, all 12)
  --depth D                  fixed depth              (default 8)
  --nodes N                  node limit instead of a depth
  --set k=v,...              price-list overrides for arm A
  --vs k=v,...               second arm; prints A|B side by side and says
                             whether the answer changed
  --ply P                    print the per-child table down to ply P (default 1)
  --full                     print the aggregate report too

TUNE MODES
  label   --out <f>                label positions with a reference search
          --fens <f>               corpus: a FEN per line, from a binpack (see
                                   below). --n N positions, --seed S, and
                                   --min-move M to drop book plies (default 8)
          --pgn <f>...             corpus: our own games instead — the biased
                                   source tune.rs warns about
                                   with --stride S --skip P
          --ref-nodes N            nodes per labelled move    (default 400000)
          --top K                  root moves labelled        (default 5)
          --reuse                  keep labels whose teacher line differs
          An existing --out file is a CACHE. The FEN sample is nested in --n at
          a fixed seed, so raising --n relabels only what is new. The cache key
          is the whole teacher line — node budget, --top, engine revision and
          net — so promoting a stronger engine to labeller relabels everything,
          which is the point: its opinion is a different opinion.
  eval    --labels <f>             mean regret of one price list
  compare --labels <f> --a <spec> --b <spec>
                                   paired difference between two price lists —
                                   the instrument to use, see tune.rs
  search  --labels <f>             finite-difference descent on the price list
          --free a,b,c             parameters to move  (default: all but fare)
          --iters N                (default 15)
          --test-frac F            held-out share  (default 0.4)

  eval, compare, metrics and search take a budget, four ways to say it:
    --nodes N        per position  (default 60000)
    --work N         per position, in work units — needs --features work.
                     A quiescence node costs 0.275 of a main-search node, so
                     at fixed NODES a price list can win by buying cheap ones.
    --total-nodes N / --total-work N
                     for the whole pass, divided by the corpus size. A pass
                     then costs the same whatever --n is, and raising --n
                     trades resolution for coverage. Below ~15000 per position
                     the proxy has the wrong sign (LEDGER 014) and it says so.
  ... plus --cap C (per-position regret cap, cp, default 200 — 5% of real
  positions contain a mate and would otherwise be the entire objective),
  --set k=v,... (price-list overrides) and --threads K.

  Making a FEN corpus from a binpack (positions from strong engines rather
  than from our own games):
    nnue/extract/target/release/nnue-extract --input <x.binpack> \\
        --output /dev/null --mark --game-stride 401 --fen-dump corpus.fens
";

fn main() {
    chess::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        None | Some("uci") => chess::uci::run(),
        // Before the flag arm below, or `--help` would start UCI and wait.
        Some("-h" | "--help") => print!("{HELP}"),
        // A leading flag is not a subcommand: `chess --quad net.nnue` is the
        // UCI engine with a net chosen on the command line, which is how two
        // arms of an eval SPRT are the same binary with different weights.
        Some(a) if a.starts_with('-') => chess::uci::run(),
        Some("serve") => {
            let port = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8080);
            chess::qeval::warn_if_fallback();
            if let Err(e) = chess::gui::serve(port, 64) {
                eprintln!("serve failed: {e}");
                std::process::exit(1);
            }
        }
        Some("play") => play(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000)),
        // Dump the legal moves of each FEN on stdin, one line of
        // `from:to:flag` per position. Exists so the eval-side rule-set study
        // can measure how often a routing rule changes per *move* rather than
        // per synthetic relocation -- the refresh rate is what decides whether
        // a rule can sit in front of the accumulator, and guessing the legal
        // move mix instead of generating it was the weak part of that study.
        Some("moves") => {
            use std::io::{BufRead, Write};
            let stdin = std::io::stdin();
            let out = std::io::stdout();
            let mut out = std::io::BufWriter::new(out.lock());
            for line in stdin.lock().lines() {
                let line = match line { Ok(l) => l, Err(_) => break };
                let fen = line.trim();
                if fen.is_empty() { continue; }
                let b = match Board::from_fen(fen) {
                    Ok(b) => b,
                    Err(e) => { println!("ERR {e}"); continue; }
                };
                let mut first = true;
                for m in chess::movegen::legal_moves(&b).iter() {
                    if !first { let _ = write!(out, ","); }
                    let _ = write!(out, "{}:{}:{}", m.from().0, m.to().0, m.flag());
                    first = false;
                }
                let _ = writeln!(out);
            }
            let _ = out.flush();
        }
        Some("perft") => {
            let depth: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
            let fen = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                chess::board::START_FEN.to_string()
            };
            let b = match Board::from_fen(&fen) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("bad FEN: {e}");
                    std::process::exit(1);
                }
            };
            let t = std::time::Instant::now();
            let mut total = 0u64;
            for (m, n) in chess::perft::perft_divide(&b, depth) {
                println!("{m}: {n}");
                total += n;
            }
            let el = t.elapsed().as_secs_f64();
            println!("\nnodes {total}  {el:.3}s  {:.1} Mnps", total as f64 / el / 1e6);
        }
        Some("perft-suite") => perft_suite(),
        Some("match") => run_match(&args[1..]),
        Some("elo") => elo_of_pgns(&args[1..]),
        Some("tune") => tune(&args[1..]),
        Some("trace") => trace_cmd(&args[1..]),
        Some("bench") => bench(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(9)),
        Some("smp") => smp_bench(&args[1..]),
        #[cfg(feature = "movedump")]
        Some("movedump") => movedump_cmd(&args[1..]),
        #[cfg(feature = "movedump")]
        Some("expand") => expand_cmd(&args[1..]),
        // Debug/verification: FENs on stdin, one per line; prints the static
        // eval and the active feature indices. Used to cross-check the engine's
        // feature convention and quantised arithmetic against the trainer,
        // which is the one place a silent mismatch would cost a whole match.
        Some("evalfen") => eval_fens(),
        Some("evalprof") => eval_prof(&args[1..]),
        Some("nodeprof") => node_prof(&args[1..]),
        Some("work") => work_cmd(&args[1..]),
        Some("params") => params_cmd(&args[1..]),
        Some("features") => features_cmd(),
        Some("fendump") => fen_dump(&args[1..]),
        Some("nodelabel") => node_label(&args[1..]),
        _ => print!("{HELP}"),
    }
}

fn perft_suite() {
    let mut ok = true;
    for (name, fen, counts) in chess::perft::SUITE {
        let b = Board::from_fen(fen).unwrap();
        for (depth, &want) in counts.iter().enumerate() {
            let t = std::time::Instant::now();
            let got = chess::perft::perft(&b, depth as u32);
            let el = t.elapsed().as_secs_f64();
            if got != want {
                ok = false;
                println!("{name} d{depth}: FAIL got {got} want {want}");
            } else if depth >= 4 {
                println!(
                    "{name:<12} d{depth} {got:>12} ok  {el:>6.2}s  {:>6.1} Mnps",
                    got as f64 / el.max(1e-9) / 1e6
                );
            }
        }
    }
    println!("\n{}", if ok { "ALL PASS" } else { "FAILURES" });
    std::process::exit(if ok { 0 } else { 1 });
}

/// Positions spanning openings, tactics and endgames. The node count at a fixed
/// depth is a deterministic fingerprint of the search: any change to move
/// ordering or pruning moves it, so `bench` is the first thing to run after a
/// change that was supposed to be behaviour-neutral. It is also what OpenBench
/// parses to normalise time control across machines.
const BENCH_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",
    "8/p3k3/8/8/8/8/5PP1/4K3 w - - 0 1",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
];

/// `chess fendump [n] [seed]` -- n FENs reached by random legal play from the
/// start position, one per line.
///
/// A verifier needs positions that are legal, varied and reproducible, and
/// nothing else in the tree produced them: `verify_deep.py` was run against a
/// hand-kept file that is not in the repo, so its measurement cannot be
/// re-run. This is 30 lines and fixes that for every verifier.
///
/// The walk restarts from the initial position whenever the game ends or the
/// ply cap is hit, so the sample spans openings, middlegames and endings rather
/// than one very long game.
fn fen_dump(rest: &[String]) {
    let n: usize = rest.first().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let mut rng: u64 = rest.get(1).and_then(|s| s.parse().ok()).unwrap_or(0x2545_F491_4F6C_DD1D);
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut b = Board::startpos();
    let mut ply = 0usize;
    let mut out = 0usize;
    while out < n {
        let moves = chess::movegen::legal_moves(&b);
        if moves.len() == 0 || ply >= 120 {
            b = Board::startpos();
            ply = 0;
            continue;
        }
        let mv = moves.get((next() % moves.len() as u64) as usize);
        b = b.make_move(mv);
        ply += 1;
        // Skip the first few plies of each restart: they are the same handful
        // of positions every time and would dominate a small sample.
        if ply > 4 {
            println!("{}", b.to_fen());
            out += 1;
        }
    }
}

/// `chess nodelabel IN OUT [--nodes N] [--hash MB]` — label the nodes
/// `--features nodedump` sampled from the tree, for the uncertainty-head probe.
///
/// Per input line (the dump's TSV), one output line in `OUT.tsv`:
/// the seven dump columns, then `static q vfinal dfinal nodes best best_cap
/// mate v1 .. v12` -- every score from the node's side to move, with that side
/// as the contempt side, `v_d` the score of completed iteration `d` of one
/// `--nodes`-bounded search (`nan` past the last), `best_cap` whether that
/// search's best move captures. `OUT.f32` gets the net's inner layers
/// (`WdlNet::inner`) as little-endian f32, one fixed-width row per line.
/// Each position gets a fresh search and a cleared hash, so a label does not
/// depend on which positions came before it.
fn node_label(rest: &[String]) {
    use std::io::{BufRead, Write};
    let (Some(inp), Some(outp)) = (rest.first(), rest.get(1)) else {
        eprintln!("usage: chess nodelabel IN OUT [--nodes N] [--hash MB] (needs --wdl or $CHESS_WDL)");
        std::process::exit(2);
    };
    let flag = |k: &str, d: u64| {
        rest.iter().position(|a| a == k).and_then(|i| rest.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(d)
    };
    let nodes = flag("--nodes", 65536);
    let Some(net) = chess::wdleval::net() else {
        eprintln!("nodelabel needs a WDL net: --wdl <file> (or $CHESS_WDL)");
        std::process::exit(2);
    };
    let shared = Shared::new(flag("--hash", 16) as usize);
    let limits = Limits { nodes: Some(nodes), ..Default::default() };
    let f = std::fs::File::open(inp).unwrap_or_else(|e| panic!("{inp}: {e}"));
    let mut tsv = std::io::BufWriter::new(std::fs::File::create(format!("{outp}.tsv")).unwrap());
    let mut bin = std::io::BufWriter::new(std::fs::File::create(format!("{outp}.f32")).unwrap());
    const MAXD: usize = 12;
    for line in std::io::BufReader::new(f).lines() {
        let line = line.unwrap();
        let Some(fen) = line.split('\t').next() else { continue };
        let Ok(b) = Board::from_fen(fen) else { continue };
        shared.tt.clear();
        let mut s = Searcher::new(&shared, ThreadData::new(0), DefaultEval::default(), Params::default());
        let q = s.qsearch_value(&b);
        let st = net.cp_from_logits_asym(net.raw(&b), true);
        let mut v = [f64::NAN; MAXD];
        let mut cb = |r: &chess::search::SearchResult, _: std::time::Duration, _: usize| {
            if (1..=MAXD as u32).contains(&r.depth) {
                v[r.depth as usize - 1] = r.score as f64;
            }
        };
        shared.tt.clear();
        let r = s.go(&b, &[], &limits, Some(&mut cb));
        let vs: Vec<String> = v.iter().map(|x| if x.is_nan() { "nan".into() } else { format!("{x}") }).collect();
        writeln!(
            tsv,
            "{line}\t{st}\t{q}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.score,
            r.depth,
            r.nodes,
            r.best_move,
            r.best_move.is_capture() as u8,
            chess::eval::is_mate_score(r.score) as u8,
            vs.join("\t")
        )
        .unwrap();
        for x in net.inner(&b) {
            bin.write_all(&x.to_le_bytes()).unwrap();
        }
    }
    tsv.flush().unwrap();
    bin.flush().unwrap();
}

fn eval_fens() {
    use std::io::BufRead;
    let net = chess::qeval::net();
    let deep = chess::deepeval::net();
    let wdl = chess::wdleval::net();
    // `evalfen --wdl FILE --small`: read the dual file's quiescence tail, so
    // `verify_wdl.py` can check the small tail against its own checkpoint the
    // same way it checks the big one.
    let small = std::env::args().any(|a| a == "--small") && wdl.is_some_and(|n| n.has_small());
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap_or_default();
        let fen = line.trim();
        if fen.is_empty() {
            continue;
        }
        match Board::from_fen(fen) {
            Ok(b) => {
                let cp = match (wdl, deep, net) {
                    (Some(x), _, _) if small => x.cp_from_logits_asym(
                        x.raw_small(&b),
                        chess::eval::stm_is_root(b.stm()),
                    ),
                    (Some(x), _, _) => x.evaluate(&b),
                    (None, Some(d), _) => d.evaluate(&b),
                    (None, None, Some(n)) => n.evaluate(&b),
                    (None, None, None) => chess::eval::evaluate_pst(&b),
                };
                let mut f = Vec::new();
                let stm = b.stm();
                let flip = if stm == chess::types::Color::White { 0 } else { 56 };
                for c in chess::types::Color::ALL {
                    let side = if c == stm { 0 } else { 384 };
                    for pt in chess::types::PieceType::ALL {
                        let base = side + 64 * pt.index();
                        for sq in b.colored(c, pt) {
                            f.push(base + (sq.index() ^ flip));
                        }
                    }
                }
                f.sort_unstable();
                let feats: Vec<String> = f.iter().map(|x| x.to_string()).collect();
                // The bucket indices too: they are the one part of the eval
                // that is reimplemented rather than read from the file, so
                // `nnue/verify.py` has to be able to check them.
                let bm = chess::qeval::QuadNet::bucket_of(&b, chess::qeval::FAM_MATERIAL);
                let bc = chess::qeval::QuadNet::bucket_of(&b, chess::qeval::FAM_COUNT);
                // With a WDL net loaded, two more fields: its five bucket
                // indices and the raw (L, D, W) logits. Both are reimplemented
                // rather than read from the file -- a wrong bucket or a swapped
                // W/L still evaluates plausibly -- so `nnue/verify_wdl.py` has
                // to be able to check them separately from the cp they collapse
                // to.
                match wdl {
                    Some(x) => {
                        let mut ff = [0u16; chess::qeval::MAX_FEAT];
                        let nf = chess::qeval::QuadNet::features(&b, &mut ff);
                        let k = chess::wdleval::Bks::of(&ff, nf);
                        let lg = if small { x.raw_small(&b) } else { x.raw(&b) };
                        println!(
                            "{cp}|{}|{bm},{bc}|{},{},{},{},{}|{:.6},{:.6},{:.6}",
                            feats.join(","),
                            k.king.0, k.king.1, k.mat.0, k.mat.1, k.sym,
                            lg[0], lg[1], lg[2]
                        );
                    }
                    None => println!("{cp}|{}|{bm},{bc}", feats.join(",")),
                }
            }
            Err(e) => println!("ERR {e}"),
        }
    }
}

/// `chess expand` — every legal move of every given position, in the same
/// format `chess movedump` writes. The control for it: the SAME pricing code
/// on GAME positions instead of tree nodes, so the position effect and the
/// move-ordering effect can be told apart.
#[cfg(feature = "movedump")]
fn expand_cmd(args: &[String]) {
    use std::io::Write;
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let labels = flag("--labels").unwrap_or_else(|| "labels-4k.bin".into());
    let out = flag("--out").unwrap_or_else(|| "expand.txt".into());
    let max: usize = flag("--max").and_then(|v| v.parse().ok()).unwrap_or(4000);

    let roots: Vec<Board> = match chess::tune::read_labels(&labels) {
        Ok(l) => l.iter().take(max).filter_map(|x| Board::from_fen(&x.fen).ok()).collect(),
        Err(e) => { eprintln!("{e}"); std::process::exit(3); }
    };
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    let mut n = 0u64;
    for b in &roots {
        let mut l = MoveList::new();
        generate(b, GenType::All, &mut l);
        for i in 0..l.len() {
            let mv = l.get(i);
            let nb = b.make_move(mv);
            let _ = writeln!(w, "{}\t{}\t{}\t2", b.to_fen(), nb.to_fen(), mv.to_uci());
            n += 1;
        }
    }
    let _ = w.flush();
    println!("{} positions, {n} legal moves -> {out}", roots.len());
}

/// `chess movedump` — search a set of real positions and write out the moves
/// the tree actually makes, so a routing rule can be priced on the real
/// distribution rather than on a class-reweighted legal-move mix. See
/// `src/movedump.rs` for why the reweighting is not enough.
#[cfg(feature = "movedump")]
fn movedump_cmd(args: &[String]) {
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let num = |n: &str, d: u64| flag(n).and_then(|v| v.parse().ok()).unwrap_or(d);

    let labels = flag("--labels").unwrap_or_else(|| "labels-4k.bin".into());
    let out = flag("--out").unwrap_or_else(|| "movedump.txt".into());
    let nodes = num("--nodes", 15_000);
    let every = num("--every", 25);
    let max = num("--max", 400) as usize;

    let roots: Vec<Board> = match chess::tune::read_labels(&labels) {
        Ok(l) => l.iter().take(max).filter_map(|x| Board::from_fen(&x.fen).ok()).collect(),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(3);
        }
    };
    println!("{} roots from {labels}, {nodes} nodes each, sampling 1 in {every}", roots.len());

    chess::movedump::with_random(args.iter().any(|a| a == "--rand"));
    if let Err(e) = chess::movedump::open(&out, every) {
        eprintln!("{e}");
        std::process::exit(3);
    }
    let shared = Shared::new(64);
    let limits = Limits { nodes: Some(nodes), ..Default::default() };
    let t = std::time::Instant::now();
    let mut total = 0u64;
    let mut mix = [0u64; 6];
    for (i, b) in roots.iter().enumerate() {
        shared.tt.clear();
        let mut s = Searcher::new(&shared, ThreadData::new(0), DefaultEval::default(), Params::default());
        let r = s.go(b, &[], &limits, None);
        total += r.nodes;
        for (m, v) in mix.iter_mut().zip(s.td.movemix.iter()) {
            *m += v;
        }
        if i % 25 == 0 {
            eprint!("\r  {i}/{}", roots.len());
        }
    }
    eprintln!("\r  {}/{}   ", roots.len(), roots.len());
    let (seen, wrote) = chess::movedump::close();
    let made: u64 = mix.iter().sum();
    let names = ["main quiet", "main capture", "main promo", "q quiet", "q capture", "q promo"];
    println!("\n{total} nodes, {made} made moves in {:.0}s", t.elapsed().as_secs_f64());
    for (n, v) in names.iter().zip(mix.iter()) {
        println!("  {n:<14}{v:>12}  {:>6.2}%", *v as f64 * 100.0 / made.max(1) as f64);
    }
    println!("\nsampled {wrote} of {seen} -> {out}");
}

/// Wraps the real evaluator and keeps every `stride`-th board it is asked
/// about, so the profiling corpus is drawn from the search tree rather than
/// from a random walk. Tree positions average 16.6 pieces against 19.4 for
/// game positions, and the feature count is what prices `psqt`.
struct Rec {
    inner: DefaultEval,
    seen: u64,
    stride: u64,
    cap: usize,
    out: std::sync::Arc<std::sync::Mutex<Vec<Board>>>,
}

impl chess::eval::Evaluator for Rec {
    fn evaluate(&mut self, b: &Board) -> chess::eval::Score {
        if self.seen % self.stride == 0 {
            if let Ok(mut v) = self.out.lock() {
                if v.len() < self.cap {
                    v.push(*b);
                }
            }
        }
        self.seen += 1;
        self.inner.evaluate(b)
    }
    fn push(&mut self, b: &Board) {
        self.inner.push(b);
    }
    fn pop(&mut self) {
        self.inner.pop();
    }
    fn set_small(&mut self, small: bool) {
        self.inner.set_small(small);
    }
}

/// Where an evaluation's time goes, stage by stage.
///
/// Each stage is priced as the difference between two prefixes of the very
/// same code the search runs (`wdleval::logits_upto`), so the parts sum to the
/// whole rather than to something near it. The MAC column is what the stage
/// has to compute; ns/MAC is the number to attack.
fn eval_prof(rest: &[String]) {
    let opt = |k: &str, d: usize| -> usize {
        rest.iter()
            .position(|a| a == k)
            .and_then(|i| rest.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let npos = opt("--positions", 512);
    let reps = opt("--reps", 200);
    let depth = opt("--depth", 8) as u32;

    let Some(net) = chess::wdleval::net() else {
        eprintln!("evalprof needs a WDL net: --wdl <file> (or $CHESS_WDL)");
        std::process::exit(2);
    };

    // Collect the corpus from a real search on the bench positions.
    let out = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let shared = Shared::new(64);
    let limits = Limits { depth: Some(depth), ..Default::default() };
    for fen in BENCH_FENS {
        if out.lock().map(|v| v.len() >= npos).unwrap_or(true) {
            break;
        }
        let b = Board::from_fen(fen).unwrap();
        shared.tt.clear();
        let ev = Rec {
            inner: DefaultEval::without_adaptive(),
            seen: 0,
            // Spread the sample across the tree instead of taking the first
            // `npos` nodes, which are all one opening.
            stride: 97,
            cap: npos,
            out: out.clone(),
        };
        let mut s = Searcher::new(&shared, ThreadData::new(0), ev, Params::default());
        s.go(&b, &[], &limits, None);
    }
    let boards = std::mem::take(&mut *out.lock().unwrap());
    if boards.is_empty() {
        eprintln!("no positions collected");
        std::process::exit(3);
    }

    let nf = chess::wdleval::WdlNet::mean_feats(&boards);
    let t = net.profile(&boards, reps);
    let macs = net.stage_macs(nf.round() as usize);
    let (w, d, h) = net.dims();

    println!(
        "net width {w} bneck {d} hidden {h} | {} tree positions, {} feats mean, {reps} reps",
        boards.len(),
        format_args!("{nf:.1}")
    );
    println!("\n{:<8}{:>10}{:>9}{:>12}{:>10}", "stage", "ns", "%", "MACs", "ns/MAC");
    let total = t[t.len() - 1];
    let mut prev = 0.0;
    let mut tm = 0u64;
    for (i, name) in chess::wdleval::PROF_STAGES.iter().enumerate() {
        let dt = (t[i] - prev).max(0.0) * 1e9;
        prev = t[i];
        tm += macs[i];
        let per = if macs[i] > 0 { format!("{:.3}", dt / macs[i] as f64) } else { "-".into() };
        println!(
            "{name:<8}{dt:>10.1}{:>8.1}%{:>12}{per:>10}",
            100.0 * dt / (total * 1e9),
            macs[i]
        );
    }
    println!("{:-<49}", "");
    println!(
        "{:<8}{:>10.1}{:>8.1}%{tm:>12}{:>10.3}",
        "eval",
        total * 1e9,
        100.0,
        total * 1e9 / tm as f64
    );
    // What one evaluation costs the search, for scale: the same corpus size
    // and the same net, but measured through `bench`, is the honest check.
    println!(
        "\n{:.0} evals/s single-thread; {:.2} GMAC/s",
        1.0 / total,
        tm as f64 / total / 1e9
    );
}

/// The whole tunable surface of the search, in one place.
///
/// `Params` and `Pricing` generate their own registry (`search::params!`), so
/// this listing cannot go stale: a constant that is not here is not a
/// parameter, and a parameter cannot be added without appearing here.
///
/// `--spsa` prints the same thing in the format OpenBench's tuner reads, so
/// handing an optimiser the search is a copy-paste rather than a translation.
/// The step is a sixteenth of the box, which is a starting guess and not a
/// measurement — the objective's noise is what actually sets it.
/// The declared feature surface: what the search is allowed to look at when it
/// prices a child, and which of them have earned a coefficient.
///
/// Deliberately its own listing rather than a column in `chess params`. A
/// parameter is a number to fit; a feature is a claim about what predicts a
/// good allocation, and the claim is the `group` column — sigma, gap, or the
/// cost of an error at the root. See `library/013`.
fn features_cmd() {
    use chess::search::Feat;
    let d = Params::default();
    let names = Feat::NAMES;
    println!(
        "{} declared features. group = which term of\n  \
         b ~ (PLY/gamma) * [ log2(sigma) - log2(gap) + log2(C) ]\n  \
         the feature estimates, which is what fixes its sign before it is measured.\n",
        names.len()
    );
    println!("{:<16}{:>8}  {:<7}{:<5}{:<7}{}", "name", "coeff", "group", "kind", "cost", "state");
    for (i, n) in names.iter().enumerate() {
        let v = d.get(n).unwrap();
        println!(
            "{:<16}{:>8}  {:<7}{:<5}{:<7}{}",
            n, v, Feat::GROUPS[i], Feat::KINDS[i], Feat::COSTS[i],
            if v != 0 { "live" } else { "dormant" }
        );
    }
    println!(
        "\n`Gated` features are computed only while their coefficient is non-zero, so a\n\
         dormant one costs a predictable not-taken branch and nothing else. `Free` ones\n\
         are register arithmetic and always run. Screen a candidate with\n\
         `chess tune compare --a <name>=0 --b <name>=V`; see library/013 for the protocol."
    );
}

fn params_cmd(rest: &[String]) {
    let spsa = rest.iter().any(|a| a == "--spsa");
    let d = Params::default();
    let names = Params::all_names();
    if spsa {
        for n in &names {
            let (lo, hi) = Params::bounds(n).unwrap();
            let v = d.get(n).unwrap();
            println!(
                "{n}, int, {v}, {lo}, {hi}, {:.3}, 0.002",
                ((hi - lo) as f64 / 16.0).max(0.5)
            );
        }
        return;
    }
    println!("{} tunable search parameters\n", names.len());
    for n in &names {
        let (lo, hi) = Params::bounds(n).unwrap();
        println!("{:<18}{:>8}   [{lo}, {hi}]", n, d.get(n).unwrap());
        let doc = Params::doc(n);
        for line in doc.split_whitespace().collect::<Vec<_>>().chunks(11) {
            println!("{:18}{}", "", line.join(" "));
        }
        if !doc.is_empty() {
            println!();
        }
    }
    println!(
        "`set` clamps to the box, so a tuner cannot step a parameter into a value the\n\
         search does not survive. `chess params --spsa` prints this as an SPSA config."
    );
}

/// Where a node's time goes.
///
/// Deliberately the *same* workload as `bench` — the same FENs, the same
/// depth, one thread — so the two instruments can be checked against each
/// other: `nodeprof`'s ns/node times `bench`'s node count must land on
/// `bench`'s wall clock, or one of them is lying.
#[cfg(not(feature = "nodeprof"))]
fn node_prof(_rest: &[String]) {
    eprintln!(
        "nodeprof is compiled out of this build.\n\
         Rebuild with:  cargo build --release --features nodeprof"
    );
    std::process::exit(2);
}

/// `chess work [depth] [--prices <file>]` -- the same workload `nodeprof`
/// runs, priced from a frozen table instead of a clock.
///
/// It prints the measured wall time next to the modelled total on purpose.
/// The model is a claim about how long the search took; a claim that cannot be
/// checked against the thing it models is not an instrument. Reconciliation
/// within a few percent is the pass condition, and a drift means the table is
/// stale for this machine or this net -- re-emit it with
/// `chess nodeprof --emit-prices`.
#[cfg(not(feature = "work"))]
fn work_cmd(_rest: &[String]) {
    eprintln!(
        "the work meter is compiled out of this build.\n\
         Rebuild with:  cargo build --release --features work"
    );
    std::process::exit(2);
}

#[cfg(feature = "work")]
fn work_cmd(rest: &[String]) {
    let depth: u32 = rest.first().and_then(|s| s.parse().ok()).unwrap_or(9);
    let flag = |name: &str| {
        rest.iter().position(|a| a == name).and_then(|i| rest.get(i + 1)).cloned()
    };
    let prices = match flag("--prices") {
        Some(path) => match std::fs::read_to_string(&path) {
            Ok(t) => match chess::work::Prices::parse(&t, &path) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{path}: {e}");
                    std::process::exit(3);
                }
            },
            Err(e) => {
                eprintln!("{path}: {e}");
                std::process::exit(3);
            }
        },
        None => chess::work::Prices::deploy(),
    };

    let shared = Shared::new(64);
    let limits = Limits { depth: Some(depth), ..Default::default() };
    let mut tally = chess::nodeprof::Tally::default();
    let mut nodes = 0u64;
    // Wall clock over the searches only, so the 64 MB table clear per position
    // -- setup, not search -- stays out of the comparison, exactly as
    // `nodeprof` keeps it out.
    let mut wall = std::time::Duration::ZERO;
    for fen in BENCH_FENS {
        let b = Board::from_fen(fen).unwrap();
        shared.tt.clear();
        let mut td = ThreadData::new(0);
        // Install the prices so the INCREMENTAL counter runs too. It is the
        // one a work-limited search tests against, and it takes a different
        // route to the same number than the report below does -- integer adds
        // per call, against a float sum over the totals at the end. Printing
        // both is how a disagreement between them gets noticed.
        td.prof.set_prices(&prices.ns, prices.control_per_node);
        let mut s = Searcher::new(&shared, td, DefaultEval::default(), Params::default());
        let t0 = std::time::Instant::now();
        let r = s.go(&b, &[], &limits, None);
        wall += t0.elapsed();
        nodes += r.nodes;
        tally.merge(&s.td.prof);
    }

    let m = chess::work::Model::of(&tally, &prices);
    m.report(&prices);

    let measured = wall.as_secs_f64() * 1e9;
    println!(
        "\nmodelled {:.1} ms, measured {:.1} ms -- {:+.1}%. \
         The measured side carries this build's own counter increments, so a \
         small positive drift is expected.",
        m.total_ns / 1e6,
        measured / 1e6,
        100.0 * (m.total_ns - measured) / measured,
    );
    println!(
        "work {:.0} units ({} nodes, {:.3} units/node). A unit is one \
         main-search node at the deploy config.",
        chess::work::units(m.total_ns),
        nodes,
        chess::work::units(m.total_ns) / nodes.max(1) as f64,
    );
    // Two independent routes to the same total. They should differ only by the
    // picosecond rounding in `set_prices`.
    let incr_ns = tally.work_ps as f64 / 1000.0;
    println!(
        "cross-check: incremental counter {:.1} ms vs batch model {:.1} ms -- {:+.4}%",
        incr_ns / 1e6,
        m.total_ns / 1e6,
        100.0 * (incr_ns - m.total_ns) / m.total_ns,
    );
}

#[cfg(feature = "nodeprof")]
fn node_prof(rest: &[String]) {
    let depth: u32 = rest.first().and_then(|s| s.parse().ok()).unwrap_or(9);
    let cal = chess::nodeprof::calibrate();
    let shared = Shared::new(64);
    let limits = Limits { depth: Some(depth), ..Default::default() };
    let mut tally = chess::nodeprof::Tally::default();
    let mut total = 0u64;
    let t = std::time::Instant::now();
    for fen in BENCH_FENS {
        let b = Board::from_fen(fen).unwrap();
        shared.tt.clear();
        let mut s =
            Searcher::new(&shared, ThreadData::new(0), DefaultEval::default(), Params::default());
        // The whole-search cycle count is taken here rather than inside
        // `go`, so the denominator is the same interval the zones live in.
        let c0 = chess::nodeprof::tsc();
        let r = s.go(&b, &[], &limits, None);
        s.td.prof.total = chess::nodeprof::tsc().wrapping_sub(c0);
        total += r.nodes;
        tally.merge(&s.td.prof);
    }
    let el = t.elapsed().as_secs_f64();

    if rest.iter().any(|a| a == "--emit-prices") {
        chess::nodeprof::emit_prices(&tally, &cal);
        return;
    }
    chess::nodeprof::report(&tally, &cal);
    chess::nodeprof::legend();
}

fn bench(depth: u32) {
    let shared = Shared::new(64);
    let limits = Limits { depth: Some(depth), ..Default::default() };
    let mut total = 0u64;
    #[cfg(feature = "hyst")]
    let mut hyst = chess::hyst::Tally::default();
    let mut mix = [0u64; 6];
    // `--set k=v,...` applies here too, so a dormant flag can be checked to
    // reach the tree before any game is played on it.
    let args: Vec<String> = std::env::args().collect();
    let mut params = Params::default();
    if let Some(spec) = args.iter().position(|a| a == "--set").and_then(|i| args.get(i + 1)) {
        if let Err(e) = chess::tune::apply_overrides(&mut params, spec) {
            eprintln!("{e}");
            std::process::exit(3);
        }
    }
    let t = std::time::Instant::now();
    for fen in BENCH_FENS {
        let b = Board::from_fen(fen).unwrap();
        shared.tt.clear();
        let mut s = Searcher::new(&shared, ThreadData::new(0), DefaultEval::default(), params);
        let r = s.go(&b, &[], &limits, None);
        println!(
            "{:>10} nodes  d{:<2} {:>10}  {}",
            r.nodes,
            r.depth,
            score_to_uci(r.score),
            r.best_move
        );
        total += r.nodes;
        for (m, v) in mix.iter_mut().zip(s.td.movemix.iter()) {
            *m += v;
        }
        #[cfg(feature = "hyst")]
        hyst.merge(&s.td.hyst);
    }
    let el = t.elapsed().as_secs_f64();
    // OpenBench parses exactly this line.
    println!("\n{total} nodes {:.0} nps", total as f64 / el.max(1e-9));
    // A correct wiring never rebuilds: every eval finds the accumulator stack
    // already describing the board it was handed.
    let (rs, ev, mism, worst) = chess::deepeval::resync_stats();
    if ev > 0 {
        println!(
            "acc: {ev} evals, {rs} rebuilds ({:.4}%){}",
            100.0 * rs as f64 / ev as f64,
            if mism > 0 { format!(", {mism} MISMATCHES worst {worst} cp") } else { String::new() }
        );
        let bm = chess::deepeval::bucket_misses();
        println!("bucket weights: {bm} rebuilds ({:.1}% hit)", 100.0 * (1.0 - bm as f64 / ev as f64));
    }
    let (rs, ev, mism, worst) = chess::wdleval::resync_stats();
    if ev > 0 {
        println!(
            "wdl acc: {ev} evals, {rs} rebuilds ({:.4}%){}",
            100.0 * rs as f64 / ev as f64,
            if mism > 0 { format!(", {mism} MISMATCHES worst {worst} cp") } else { String::new() }
        );
    }
    // Which moves the tree actually makes. The legal-move mix is 94% quiet;
    // the made-move mix is not, because captures are searched first and
    // quiescence searches nothing else. A routing rule that reads material is
    // stable per legal move and much less stable per made move.
    let made: u64 = mix.iter().sum();
    let names = ["main quiet", "main capture", "main promo",
                 "q quiet", "q capture", "q promo"];
    #[cfg(feature = "hyst")]
    hyst.report();
    println!("\nmade moves {made}");
    for (n, v) in names.iter().zip(mix.iter()) {
        println!("  {n:<14}{v:>12}  {:>6.2}%", *v as f64 * 100.0 / made.max(1) as f64);
    }
    #[cfg(feature = "evalstats")]
    chess::evalstats::report();
    #[cfg(feature = "checkstats")]
    chess::checkstats::report();
}


/// `chess trace` — run one position and print every pricing decision.
///
/// The tuner answers "is A better than B" with one number over 14000
/// positions. This answers "what did A actually *do*" on one position, which
/// is the question you need when the number stops making sense.
fn trace_cmd(args: &[String]) {
    use chess::trace::Trace;

    let flag = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    let depth = flag("--depth").and_then(|v| v.parse::<u32>().ok()).unwrap_or(8);
    let nodes = flag("--nodes").and_then(|v| v.parse::<u64>().ok());
    let event_ply = flag("--ply").and_then(|v| v.parse::<u8>().ok()).unwrap_or(1);

    let mut pa = Params::default();
    if let Some(spec) = flag("--set") {
        if let Err(e) = chess::tune::apply_overrides(&mut pa, &spec) {
            eprintln!("{e}");
            std::process::exit(3);
        }
    }
    let pb = flag("--vs").map(|spec| {
        let mut p = Params::default();
        if let Err(e) = chess::tune::apply_overrides(&mut p, &spec) {
            eprintln!("{e}");
            std::process::exit(3);
        }
        p
    });

    let fens: Vec<String> = match flag("--fen") {
        Some(f) => vec![f],
        None => BENCH_FENS.iter().map(|s| s.to_string()).collect(),
    };

    let limits = Limits {
        depth: if nodes.is_some() { None } else { Some(depth) },
        nodes,
        ..Default::default()
    };

    let run = |b: &Board, p: Params| -> (chess::search::SearchResult, Trace) {
        let shared = Shared::new(64);
        shared.tt.clear();
        let mut td = ThreadData::new(0);
        td.trace = Some(Box::new(Trace::new(event_ply)));
        let mut s = Searcher::new(&shared, td, DefaultEval::default(), p);
        let r = s.go(b, &[], &limits, None);
        let mut t = *s.td.trace.take().unwrap();
        t.agg.nodes = r.nodes;
        (r, t)
    };

    let pv_str = |r: &chess::search::SearchResult| -> String {
        r.pv.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(" ")
    };

    // ---- single arm, one or more positions
    if pb.is_none() {
        for fen in &fens {
            let b = match Board::from_fen(fen) {
                Ok(b) => b,
                Err(e) => { eprintln!("bad FEN: {e}"); std::process::exit(3); }
            };
            let (r, t) = run(&b, pa);
            println!("\n{}", "=".repeat(78));
            println!("{fen}");
            println!("{}", "=".repeat(78));
            println!(
                "  depth {}  score {}  nodes {}  best {}\n  pv {}",
                r.depth, score_to_uci(r.score), r.nodes, r.best_move, pv_str(&r)
            );
            for ply in 0..event_ply {
                let any = t.events.iter().any(|e| e.ply == ply);
                if any {
                    println!("\n  --- ply {ply} ---");
                    print!("{}", chess::trace::report_events(&t.events, ply));
                }
            }
            if has("--full") || fens.len() == 1 {
                println!();
                print!("{}", chess::trace::report_agg(&t.agg));
            }
        }
        return;
    }

    // ---- two arms, side by side. The question is not "which is faster" but
    // "did the cheaper one still find the same move and the same score".
    let pb = pb.unwrap();
    println!(
        "{:<44} {:>9} {:>9} {:>8} {:>7} {:>7}  {}",
        "position", "nodes A", "nodes B", "saved", "cp A", "cp B", "answer"
    );
    println!("{}", "-".repeat(104));
    let (mut na, mut nb) = (0u64, 0u64);
    let (mut changed, mut pv_div, mut agg_a, mut agg_b) =
        (0usize, 0usize, chess::trace::Agg::default(), chess::trace::Agg::default());
    let mut cp_delta = Vec::new();

    for fen in &fens {
        let b = match Board::from_fen(fen) {
            Ok(b) => b,
            Err(e) => { eprintln!("bad FEN: {e}"); std::process::exit(3); }
        };
        let (ra, ta) = run(&b, pa);
        let (rb, tb) = run(&b, pb);
        na += ra.nodes;
        nb += rb.nodes;
        merge(&mut agg_a, &ta.agg);
        merge(&mut agg_b, &tb.agg);
        let same = ra.best_move == rb.best_move;
        if !same { changed += 1; }
        let pa_s = pv_str(&ra);
        let pb_s = pv_str(&rb);
        let diverge = pa_s
            .split(' ')
            .zip(pb_s.split(' '))
            .position(|(x, y)| x != y)
            .unwrap_or(pa_s.split(' ').count().min(pb_s.split(' ').count()));
        if same && diverge < 3 { pv_div += 1; }
        cp_delta.push((rb.score - ra.score).abs());
        let short: String = fen.split(' ').next().unwrap_or(fen).chars().take(42).collect();
        println!(
            "{:<44} {:>9} {:>9} {:>7.1}% {:>7} {:>7}  {}",
            short,
            ra.nodes,
            rb.nodes,
            100.0 * (rb.nodes as f64 / ra.nodes.max(1) as f64 - 1.0),
            score_to_uci(ra.score),
            score_to_uci(rb.score),
            if !same {
                format!("MOVE CHANGED {} -> {}", ra.best_move, rb.best_move)
            } else if diverge < 3 {
                format!("pv diverges at {diverge}")
            } else {
                "same".to_string()
            }
        );
    }

    let n = fens.len();
    cp_delta.sort_unstable();
    println!("{}", "-".repeat(104));
    println!(
        "{:<44} {:>9} {:>9} {:>7.1}%",
        format!("TOTAL ({n} positions)"), na, nb,
        100.0 * (nb as f64 / na.max(1) as f64 - 1.0)
    );
    println!(
        "\n  best move changed in {changed}/{n};  pv diverges early in {pv_div}/{n}"
    );
    println!(
        "  |score delta|: median {} cp, max {} cp",
        cp_delta[n / 2], cp_delta[n - 1]
    );
    println!("\nARM A  --set {}", flag("--set").unwrap_or_else(|| "(default)".into()));
    print!("{}", chess::trace::report_agg(&agg_a));
    println!("\nARM B  --vs {}", flag("--vs").unwrap());
    print!("{}", chess::trace::report_agg(&agg_b));
}

fn merge(a: &mut chess::trace::Agg, b: &chess::trace::Agg) {
    a.nodes += b.nodes;
    a.interior += b.interior;
    a.priced += b.priced;
    a.priced_out += b.priced_out;
    a.researched += b.researched;
    a.research_nodes += b.research_nodes;
    a.raised_alpha += b.raised_alpha;
    a.cutoffs += b.cutoffs;
    a.cutoff_at_first += b.cutoff_at_first;
    a.dropped_to_q += b.dropped_to_q;
    a.gap_positive += b.gap_positive;
    for i in 0..a.price_hist.len() { a.price_hist[i] += b.price_hist[i]; }
    for i in 0..a.researched_by_rank.len() { a.researched_by_rank[i] += b.researched_by_rank[i]; }
    for i in 0..a.nodes_by_rank.len() {
        a.nodes_by_rank[i] += b.nodes_by_rank[i];
        a.children_by_rank[i] += b.children_by_rank[i];
    }
}

fn play(movetime: u64) {
    use std::io::Write;
    let mut engine = chess::uci::Engine::new();
    println!("Enter moves as UCI (e2e4). Commands: quit, undo(not supported), fen, help.\n");
    loop {
        println!("{}", engine.board);
        let mut list = MoveList::new();
        generate(&engine.board, GenType::All, &mut list);
        if list.is_empty() {
            println!(
                "\n{}",
                if engine.board.in_check() { "Checkmate." } else { "Stalemate." }
            );
            return;
        }
        print!("\nyour move> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let cmd = line.trim();
        match cmd {
            "quit" | "q" => return,
            "fen" => {
                println!("{}", engine.board.to_fen());
                continue;
            }
            "help" => {
                println!("moves: {}", list.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" "));
                continue;
            }
            _ => {}
        }
        if !engine.apply_uci_move(cmd) {
            println!("illegal move '{cmd}' (type `help` for the legal list)");
            continue;
        }

        let mut l2 = MoveList::new();
        generate(&engine.board, GenType::All, &mut l2);
        if l2.is_empty() {
            println!("{}", engine.board);
            println!(
                "\n{}",
                if engine.board.in_check() { "Checkmate — you win." } else { "Stalemate." }
            );
            return;
        }

        let r = engine.search_blocking(Limits { movetime: Some(movetime), ..Default::default() });
        println!(
            "\nengine: {}   (depth {} {} {} nodes)",
            r.best_move,
            r.depth,
            score_to_uci(r.score),
            r.nodes
        );
        engine.apply_uci_move(&r.best_move.to_uci());
    }
}

/// `chess match` — the measurement front-end. See `matchplay` for why the
/// match is shaped the way it is, and `sprt` for what the numbers mean.
fn run_match(args: &[String]) {
    use chess::matchplay::{Adjudication, MatchConfig, PlayerSpec, TimeControl};
    use chess::sprt::{Sprt, SprtVerdict};

    // `--engine "<path> <arg>..."`: the arguments matter because two arms of
    // an eval SPRT are the SAME binary handed different net files, which is
    // the only way to be sure nothing but the eval differs.
    let spec = |s: &str| -> PlayerSpec {
        if s == "self" || s == "internal" {
            PlayerSpec::Internal { name: format!("{}-dev", chess::uci::NAME), params: Params::default() }
        } else {
            let mut it = s.split_whitespace();
            let path = it.next().unwrap_or(s).to_string();
            PlayerSpec::External { path, args: it.map(|x| x.to_string()).collect() }
        }
    };

    let mut a = spec("self");
    let mut b = spec("self");
    let mut games = 200u32;
    let mut tc = TimeControl::default();
    let mut sprt = None;
    let mut concurrency = (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2) / 2).max(1);
    let mut threads_a = 1usize;
    let mut threads_b = 1usize;
    let mut hash_mb = 32usize;
    let mut book = chess::book::default_book();
    let mut pgn = None;
    let mut quiet = false;
    let mut name_a: Option<String> = None;
    let mut name_b: Option<String> = None;

    let mut i = 0;
    let fail = |m: String| -> ! {
        eprintln!("{m}");
        std::process::exit(3);
    };
    while i < args.len() {
        let next = |i: usize| -> String {
            args.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("{} needs a value", args[i]);
                std::process::exit(3);
            })
        };
        match args[i].as_str() {
            "--engine" | "-a" => a = spec(&next(i)),
            "--opponent" | "-b" => b = spec(&next(i)),
            "--games" | "-n" => games = next(i).parse().unwrap_or(games),
            "--tc" => match TimeControl::parse(&next(i)) {
                Ok(t) => tc = t,
                Err(e) => fail(e),
            },
            "--sprt" => {
                let v = next(i);
                let (lo, hi) = v.split_once(',').unwrap_or(("0", "5"));
                match (lo.trim().parse::<f64>(), hi.trim().parse::<f64>()) {
                    (Ok(l), Ok(h)) => sprt = Some(Sprt::new(l, h)),
                    _ => fail(format!("bad --sprt bounds '{v}', expected e.g. 0,5")),
                }
            }
            "--concurrency" | "-c" => concurrency = next(i).parse().unwrap_or(concurrency).max(1),
            "--threads" => {
                let t = next(i).parse().unwrap_or(1usize).max(1);
                threads_a = t;
                threads_b = t;
            }
            "--threads-a" => threads_a = next(i).parse().unwrap_or(threads_a).max(1),
            "--threads-b" => threads_b = next(i).parse().unwrap_or(threads_b).max(1),
            "--hash" => hash_mb = next(i).parse().unwrap_or(hash_mb).max(1),
            "--book" => match next(i).as_str() {
                // The 43 balanced theory lines, kept as an escape hatch: they
                // are what every number before LEDGER 112 was measured on.
                "builtin" => book = chess::book::builtin(),
                path => match chess::book::load(path) {
                    Ok(bk) => book = bk,
                    Err(e) => fail(e),
                },
            },
            "--pgn" => pgn = Some(next(i)),
            "--name-a" => name_a = Some(next(i)),
            "--name-b" => name_b = Some(next(i)),
            "--quiet" | "-q" => {
                quiet = true;
                i -= 1; // no value to skip
            }
            other => fail(format!("unknown option '{other}' (see `chess help`)")),
        }
        i += 2;
    }

    // An SPRT with no game cap runs until it decides; a large cap keeps a
    // pathological run from going forever.
    if sprt.is_some() && games == 200 {
        games = 20_000;
    }

    println!(
        "{} vs {}   {}   {} games max, {concurrency} concurrent, {hash_mb} MB hash, {} openings",
        a.label(),
        b.label(),
        tc.describe(),
        games,
        book.len()
    );
    if threads_a > 1 || threads_b > 1 {
        println!("search threads per engine: A={threads_a} B={threads_b}");
    }
    // `--threads T` sets both; `--threads-a/b` override per arm. Internal
    // (in-process) play stays single-threaded — the SMP number needs
    // subprocess arms, which is enforced in `Player::new`.
    let (_stats, verdict) = chess::matchplay::run(MatchConfig {
        a,
        b,
        name_a,
        name_b,
        games,
        tc,
        concurrency,
        threads_a,
        threads_b,
        hash_mb,
        sprt,
        pgn,
        book,
        adj: Adjudication::default(),
        quiet,
    });
    std::process::exit(match verdict {
        Some(SprtVerdict::AcceptH1) => 0,
        Some(SprtVerdict::AcceptH0) => 1,
        Some(SprtVerdict::Continue) => 2,
        None => 0,
    });
}

/// `chess elo <pgn>...` — the statistics of games that have already been
/// played, including a match that is still running.
///
/// It exists so that watching a match in progress uses the *same* estimator as
/// the match itself. A second implementation in a shell script would eventually
/// disagree with the first, and there would be no way to tell which number was
/// the real one.
fn elo_of_pgns(args: &[String]) {
    let mut forced_name: Option<String> = None;
    let mut combine: Option<String> = None;
    let mut tc_disp = String::new();
    let mut games_disp = String::new();
    let mut files = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                forced_name = args.get(i + 1).cloned();
                i += 2;
            }
            "--combine" => {
                combine = args.get(i + 1).cloned();
                i += 2;
            }
            "--tc" => {
                tc_disp = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--games" => {
                games_disp = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            other => {
                files.push(other.to_string());
                i += 1;
            }
        }
    }
    if let Some(tsv) = combine {
        combine_anchors_tsv(&tsv, &tc_disp, &games_disp);
        return;
    }
    if files.is_empty() {
        eprintln!("usage: chess elo [--name <engine>] [--combine <results.tsv>] <pgn>...");
        std::process::exit(3);
    }

    // Per-file pairing: round numbers restart in every match file, so pairs
    // are scoped to one file and the resulting statistics merged. Pooling the
    // game lists first would pair round 2k-1 of one match with round 2k of
    // another.
    let mut per_file: Vec<Vec<chess::matchplay::PgnGame>> = Vec::new();
    for f in &files {
        match chess::matchplay::read_pgn(f) {
            Ok(g) => per_file.push(g),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(3);
            }
        }
    }
    let all: Vec<&chess::matchplay::PgnGame> = per_file.iter().flatten().collect();
    if all.is_empty() {
        println!("no finished games yet");
        return;
    }
    let games: Vec<chess::matchplay::PgnGame> = per_file.iter().flatten().cloned().collect();

    // Whose point of view. Derived from the games unless forced, because
    // getting this wrong reports the opponent's result as ours, and the number
    // looks entirely plausible.
    let who = match forced_name {
        Some(n) => n,
        None => match chess::matchplay::engine_under_test(&games) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(3);
            }
        },
    };
    let mut stats = chess::sprt::MatchStats::default();
    for (f, g) in files.iter().zip(per_file.iter()) {
        match chess::matchplay::stats_from_games(g, &who) {
            Ok(s) => stats.merge(&s),
            Err(e) => {
                eprintln!("{f}: {e}");
                std::process::exit(3);
            }
        }
    }

    let opponents: Vec<String> = {
        let mut v: Vec<String> = games
            .iter()
            .map(|g| if g.white == who { g.black.clone() } else { g.white.clone() })
            .collect();
        v.sort();
        v.dedup();
        v
    };
    println!("{who}  vs  {}", opponents.join(", "));
    println!("{}", chess::sprt::summary(&stats, None));
    if let (Some((elo, lo, hi)), Some(los)) = (stats.elo(), stats.los()) {
        println!(
            "RESULT games={} w={} l={} d={} elo={elo:.2} lo={lo:.2} hi={hi:.2} los={los:.4} pairs={}",
            stats.games(),
            stats.wins,
            stats.losses,
            stats.draws,
            stats.pair_count()
        );
    }
    if opponents.len() > 1 {
        eprintln!("note: {} opponents pooled into one difference-vs-pool number. That is not an anchored rating — pass each leg through `tools/ladder.sh run` and combine with `chess elo --combine <results.tsv>`.", opponents.len());
    }
}

/// `chess elo --combine <results.tsv>` — the anchored gauntlet combination.
///
/// Each row of the TSV is one leg (opponent key, its CCRL Elo and 95% error,
/// engine-error count, and the match runner's RESULT line). The math is
/// `sprt::combine_anchors`, the same estimator `tools/ladder.sh` used to carry
/// as embedded python — one implementation, covered by `cargo test`.
fn combine_anchors_tsv(path: &str, tc_disp: &str, games_disp: &str) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(3);
    });
    let mut anchors = Vec::new();
    let mut errs = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 5 {
            eprintln!("{path}:{}: expected 5 tab columns, found {}", ln + 1, cols.len());
            std::process::exit(3);
        }
        let (key, ccrl, err, nerr, result) = (cols[0], cols[1], cols[2], cols[3], cols[4]);
        // A leg whose match contained protocol failures is not a weak
        // measurement, it is a void one: every forfeit scores our way, so
        // including it biases the combination upward. Drop it loudly.
        let nerr: u32 = nerr.trim().parse().unwrap_or_else(|_| {
            eprintln!("{path}:{}: bad engine-error count '{nerr}'", ln + 1);
            std::process::exit(3);
        });
        if nerr > 0 {
            println!("  EXCLUDED {key}: {nerr} game(s) ended in an engine error — fix the protocol bug and re-run this opponent");
            continue;
        }
        let ccrl: f64 = ccrl.trim().parse().unwrap_or_else(|_| {
            eprintln!("{path}:{}: bad CCRL elo '{ccrl}'", ln + 1);
            std::process::exit(3);
        });
        let err: f64 = err.trim().parse().unwrap_or_else(|_| {
            eprintln!("{path}:{}: bad CCRL error '{err}'", ln + 1);
            std::process::exit(3);
        });
        let kv: std::collections::BTreeMap<&str, &str> = result
            .split_whitespace()
            .skip(1) // RESULT
            .filter_map(|t| t.split_once('='))
            .collect();
        let num = |k: &str| -> f64 {
            kv.get(k)
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| {
                    eprintln!("{path}:{}: RESULT line is missing '{k}': {result}", ln + 1);
                    std::process::exit(3);
                })
        };
        anchors.push(chess::sprt::Anchor {
            key: key.to_string(),
            ccrl,
            diff: num("elo"),
            lo: num("lo"),
            hi: num("hi"),
            games: num("games") as u32,
            wins: num("w") as u32,
            draws: num("d") as u32,
        });
        errs.push(err);
    }
    if anchors.is_empty() {
        eprintln!("{path}: no usable anchors");
        std::process::exit(3);
    }
    println!();
    println!("{:<12}{:>7}{:>9}{:>9}{:>10}   95% interval", "opponent", "CCRL", "score", "diff", "implied");
    for (a, e) in anchors.iter().zip(errs.iter()) {
        let (r, s) = a.implied(*e);
        let (lo, hi) = (r - 1.96 * s, r + 1.96 * s);
        println!(
            "{:<12}{:>7.0}{:>8.1}%{:+9.0}{:>10.0}   [{:.0}, {:.0}]  ({} games)",
            a.key,
            a.ccrl,
            a.score_pct(),
            a.diff,
            r,
            lo,
            hi,
            a.games,
        );
    }
    match chess::sprt::combine_anchors(&anchors, &errs) {
        Some(c) => {
            println!();
            println!("  rating  {:.0}  ±{:.0}   (95%, scaled for anchor spread)", c.mean, c.half_width);
            println!("          ±{:.0} unscaled — quote the scaled one", c.half_width_unscaled);
            println!("  scale   CCRL Blitz, tc {tc_disp}, {games_disp} games per opponent");
            println!(
                "  spread  chi2/dof = {:.2} over {} opponents, error x{:.2}{}",
                c.chi2_dof,
                c.n,
                c.scale,
                if c.chi2_dof > 2.5 {
                    "  — estimates disagree, treat the combination with suspicion"
                } else {
                    ""
                }
            );
            println!();
            println!("  On top of this sits a systematic offset (different book, no tablebases,");
            println!("  different time control and a different opponent pool from CCRL's own).");
            println!("  Quote it as a scale, not a certificate: 'about X on the CCRL Blitz scale'.");
        }
        None => {
            eprintln!("cannot combine: no anchors");
            std::process::exit(3);
        }
    }
}

// ---------------------------------------------------------------- tune

/// A hash of the running binary, for stamping on a label file.
///
/// The commit is not enough and `-dirty` is worse than nothing: every working
/// tree with any uncommitted change produces the same string, so two different
/// searches would share a cache key during exactly the phase when the search is
/// being changed. The binary is the search — it also covers the feature flags
/// and the compiler, which the commit does not.
///
/// FNV-1a, not a cryptographic hash. This is a cache key and a provenance note,
/// not a signature; nobody is trying to forge a label file.
fn exe_fingerprint() -> String {
    let Ok(path) = std::env::current_exe() else { return "?".into() };
    let Ok(bytes) = std::fs::read(path) else { return "?".into() };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in &bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// `chess tune label|eval|search`. See `tune.rs` for what the objective is and
/// the three specific ways it can lie to us.
fn tune(args: &[String]) {
    let mode = args.first().map(|s| s.as_str()).unwrap_or("");
    let flag = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let num = |name: &str, dflt: u64| -> u64 {
        flag(name).and_then(|v| v.parse().ok()).unwrap_or(dflt)
    };
    // Leave a core free: this box is also what plays the matches.
    let threads = num("--threads", (std::thread::available_parallelism().map_or(4, |n| n.get()) as u64 - 1).max(1)) as usize;

    let mut params = Params::default();
    if let Some(spec) = flag("--set") {
        if let Err(e) = chess::tune::apply_overrides(&mut params, &spec) {
            eprintln!("{e}");
            std::process::exit(3);
        }
    }

    // One budget for a scoring pass, said four ways. `--nodes` and `--work`
    // are per position. `--total-nodes` and `--total-work` are for the whole
    // pass and are divided by the corpus size, so a pass costs the same wall
    // time whatever `n` is and raising `n` trades resolution for coverage at
    // fixed cost. `tune::warn_if_cheap` prints the LEDGER 014 floor if that
    // trade goes too far.
    let budget_for = |n: usize| -> chess::tune::Budget {
        use chess::tune::Budget;
        let parse = |name: &str| -> u64 {
            match flag(name).unwrap().parse() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("{name}: expected a number");
                    std::process::exit(3);
                }
            }
        };
        let b = if flag("--total-work").is_some() {
            chess::tune::per_position(Budget::Work(parse("--total-work")), n)
        } else if flag("--total-nodes").is_some() {
            chess::tune::per_position(Budget::Nodes(parse("--total-nodes")), n)
        } else if flag("--work").is_some() {
            Budget::Work(parse("--work"))
        } else {
            Budget::Nodes(num("--nodes", 60_000))
        };
        if let Err(e) = b.available() {
            eprintln!("{e}");
            std::process::exit(3);
        }
        chess::tune::warn_if_cheap(b);
        b
    };

    match mode {
        "label" => {
            let Some(out) = flag("--out") else {
                eprintln!("tune label: --out is required");
                std::process::exit(3);
            };
            // Everything after --pgn up to the next flag.
            let pgns: Vec<String> = match args.iter().position(|a| a == "--pgn") {
                Some(i) => args[i + 1..].iter().take_while(|a| !a.starts_with("--")).cloned().collect(),
                None => Vec::new(),
            };
            let n = num("--n", num("--max", 1000)) as usize;
            let seed = num("--seed", 0);
            let min_move = num("--min-move", 8) as u32;
            let source;
            let positions = if let Some(fens) = flag("--fens") {
                source = format!("fens {fens} seed {seed} min-move {min_move}");
                chess::tune::positions_from_fens(&fens, n, seed, min_move)
            } else if !pgns.is_empty() {
                source = format!(
                    "pgn {} skip {} stride {}",
                    pgns.join(" "),
                    num("--skip", 16),
                    num("--stride", 7)
                );
                chess::tune::positions_from_pgn(
                    &pgns,
                    num("--skip", 16) as usize,
                    num("--stride", 7) as usize,
                    n,
                )
            } else {
                eprintln!(
                    "tune label: --fens <file> (a binpack FEN dump) or --pgn <file>... is required.\n\
                     Make a FEN dump with:\n  \
                     nnue/extract/target/release/nnue-extract --input <x.binpack> \\\n    \
                     --output /dev/null --mark --game-stride 401 --fen-dump corpus.fens"
                );
                std::process::exit(3);
            };
            let positions = match positions {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(3);
                }
            };
            let ref_nodes = num("--ref-nodes", 400_000);
            let top = num("--top", 5) as usize;

            let git = |a: &[&str]| {
                std::process::Command::new("git")
                    .args(a)
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
            };
            let rev = format!(
                "{} bin:{}",
                git(&["rev-parse", "--short", "HEAD"]).map(|s| s.trim().to_string()).unwrap_or_else(|| "?".into()),
                exe_fingerprint()
            );
            let net = args
                .iter()
                .position(|a| a == "--wdl")
                .and_then(|i| args.get(i + 1).cloned())
                .or_else(|| std::env::var("CHESS_WDL").ok())
                .unwrap_or_else(|| "(none: quad or PeSTO)".to_string());
            // The teacher's whole identity, on one line, compared verbatim
            // against the file's own header to decide whether its labels can
            // still be reused.
            let teacher_id = format!("teacher {ref_nodes} nodes, top {top}, engine {rev}, net {net}");

            // Labels are per position and expensive, so an existing file is a
            // cache, not an obstacle. The FEN sample is nested in `n`
            // (`positions_from_fens`), so raising `n` at the same seed relabels
            // only the positions that are new -- which is what makes "start at
            // 1k and lift it later" cost the difference rather than the whole
            // corpus.
            //
            // But the cache key is the ENTIRE teacher, not just its node
            // budget. The intended workflow is that a candidate which wins its
            // SPRT becomes the new labeller, and a stronger engine's opinion of
            // a position is a different opinion at the same node count -- so
            // keying on `--ref-nodes` alone would silently mix two teachers'
            // labels and make the objective depend on which positions happened
            // to be labelled before the promotion. A dirty tree counts as a
            // different engine, because it is one. `--reuse` overrides, for
            // when the uncommitted diff provably cannot reach the search.
            let reuse_stale = args.iter().any(|a| a == "--reuse");
            let cached: std::collections::HashMap<String, chess::tune::Labelled> =
                match std::fs::read_to_string(&out) {
                    Ok(text) => {
                        let old_id = text
                            .lines()
                            .find(|l| l.starts_with("# teacher "))
                            .map(|l| l.trim_start_matches("# ").to_string());
                        let same = old_id.as_deref() == Some(teacher_id.as_str());
                        match (same || reuse_stale, chess::tune::read_labels(&out)) {
                            (true, Ok(l)) => {
                                if !same {
                                    println!("  --reuse: keeping labels from a different teacher");
                                    println!("    file: {}", old_id.as_deref().unwrap_or("(no header)"));
                                    println!("    now:  {teacher_id}");
                                }
                                l.into_iter().map(|l| (l.fen.clone(), l)).collect()
                            }
                            (false, Ok(l)) => {
                                println!("  {out} holds {} labels from a different teacher — relabelling all", l.len());
                                println!("    file: {}", old_id.as_deref().unwrap_or("(no header)"));
                                println!("    now:  {teacher_id}");
                                Default::default()
                            }
                            (_, Err(e)) => {
                                eprintln!("{out}: {e}");
                                std::process::exit(3);
                            }
                        }
                    }
                    Err(_) => Default::default(),
                };
            let todo: Vec<chess::board::Board> = positions
                .iter()
                .filter(|b| !cached.contains_key(&b.to_fen()))
                .cloned()
                .collect();
            println!(
                "{} positions ({} reused from {out}), top {top} moves at {ref_nodes} nodes each, {threads} threads",
                positions.len(),
                positions.len() - todo.len()
            );
            let t = std::time::Instant::now();
            let fresh = chess::tune::label(&todo, chess::tune::Budget::Nodes(ref_nodes), top, threads);
            let mut by_fen: std::collections::HashMap<String, chess::tune::Labelled> = cached;
            for l in fresh {
                by_fen.insert(l.fen.clone(), l);
            }
            // Written in corpus order, which is the hash order the sampler
            // produced -- so the file itself is nested too and a diff against
            // the smaller corpus is an append.
            let labels: Vec<chess::tune::Labelled> = positions
                .iter()
                .filter_map(|b| by_fen.get(&b.to_fen()).cloned())
                .collect();
            let header = vec![
                format!("corpus {source} n {}", labels.len()),
                teacher_id.clone(),
            ];
            if let Err(e) = chess::tune::write_labels(&out, &labels, &header) {
                eprintln!("{e}");
                std::process::exit(3);
            }
            println!(
                "wrote {} labels to {out} ({} new) in {:.0}s",
                labels.len(),
                todo.len(),
                t.elapsed().as_secs_f64()
            );
        }

        "metrics" => {
            let Some(lf) = flag("--labels") else {
                eprintln!("tune metrics: --labels <file> is required");
                std::process::exit(3);
            };
            let labels = match chess::tune::read_labels(&lf) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(3);
                }
            };
            let budget = budget_for(labels.len());
            let c = num("--c", 1000);
            let cap = num("--cap", 200) as chess::eval::Score;
            let spec = flag("--arms").unwrap_or_else(|| "base:".to_string());
            let arms = match chess::tune::parse_arms(&spec, params) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(3);
                }
            };
            println!(
                "{} positions, {budget}, head discarded below {c}, {} arms, {threads} threads",
                labels.len(),
                arms.len()
            );
            let t = std::time::Instant::now();
            let reps = chess::tune::metric_sweep(&labels, budget, c, cap, &arms, threads);
            println!();
            println!(
                "{:<11} {:>6} {:>7} {:>6} | {:>7} {:>7} {:>6} | {:>7} {:>7} {:>6} | {:>7} {:>7} {:>6} | {:>6} {:>6} {:>6}",
                "arm", "depth", "d_dep", "t",
                "D_log", "dD", "t", "D_cp", "dCp", "t", "regret", "dR", "t", "agree", "settle", "offlst"
            );
            let t_of = |(d, se): (f64, f64)| if se > 0.0 { d / se } else { 0.0 };
            for r in &reps {
                println!(
                    "{:<11} {:>6.2} {:>+7.2} {:>+6.1} | {:>7.4} {:>+7.4} {:>+6.1} | {:>7.2} {:>+7.2} {:>+6.1} \
| {:>7.2} {:>+7.2} {:>+6.1} | {:>6.3} {:>6.3} {:>6.3}",
                    r.name,
                    r.mean.depth,
                    r.depth_diff.0,
                    t_of(r.depth_diff),
                    r.mean.d_log,
                    r.d_log_diff.0,
                    t_of(r.d_log_diff),
                    r.mean.d_cp,
                    r.d_cp_diff.0,
                    t_of(r.d_cp_diff),
                    r.mean.regret,
                    r.regret_diff.0,
                    t_of(r.regret_diff),
                    r.mean.final_agree,
                    r.mean.settle,
                    r.mean.off_list,
                );
            }
            println!();
            println!("d_* are paired differences against the first arm; t = diff/se, so |t| > 2");
            println!("is resolved. Lower is better for D_log, D_cp and regret.");
            println!("d_dep is the confound check: an arm whose depth does not move is not");
            println!("actually pruning differently, and its other columns mean nothing.");
            println!("D_log is in doublings: at the LEDGER 007 rate of ~50-70 Elo per doubling,");
            println!("a paired dD of -0.010 is worth roughly 0.6 Elo. Lower D_log is better.");
            println!("elapsed {:.0}s", t.elapsed().as_secs_f64());
        }

        "compare" => {
            let Some(lf) = flag("--labels") else {
                eprintln!("tune compare: --labels <file> is required");
                std::process::exit(3);
            };
            let labels = match chess::tune::read_labels(&lf) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(3);
                }
            };
            let build = |spec: Option<String>| {
                let mut p = Params::default();
                if let Some(s) = spec {
                    if let Err(e) = chess::tune::apply_overrides(&mut p, &s) {
                        eprintln!("{e}");
                        std::process::exit(3);
                    }
                }
                p
            };
            let a = build(flag("--a"));
            let b = build(flag("--b"));
            let budget = budget_for(labels.len());
            let c = chess::tune::compare(
                &labels,
                budget,
                a,
                b,
                threads,
                num("--cap", 200) as chess::eval::Score,
            );
            println!(
                "{} positions @ {budget}: b - a = {:+.3} +/- {:.3} cp   depth {:.2} -> {:.2} ({:+.2})   ({} of {} positions differ)",
                c.positions, c.mean_diff, c.stderr, c.depth_a, c.depth_b,
                c.depth_b - c.depth_a, c.differ, c.positions
            );
            // Under a work budget the two sides do not search the same number
            // of nodes, and the gap says where the candidate moved the money.
            // Utilisation is the precision readout (005 item 2): searched
            // priced children that raised alpha or cut off. A constraint like
            // depth — never an objective.
            println!(
                "           nodes {:.0} -> {:.0} ({:+.1}%)   util {:.1}% -> {:.1}% ({:+.1}pp)",
                c.nodes_a,
                c.nodes_b,
                100.0 * (c.nodes_b - c.nodes_a) / c.nodes_a.max(1.0),
                100.0 * c.util_a,
                100.0 * c.util_b,
                100.0 * (c.util_b - c.util_a),
            );
        }

        "eval" | "search" => {
            let Some(lf) = flag("--labels") else {
                eprintln!("tune {mode}: --labels <file> is required");
                std::process::exit(3);
            };
            let labels = match chess::tune::read_labels(&lf) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(3);
                }
            };
            let budget = budget_for(labels.len());
            let cap = num("--cap", 200) as chess::eval::Score;
            if mode == "eval" {
                let t = std::time::Instant::now();
                let r = chess::tune::evaluate(&labels, budget, params, threads, cap);
                println!(
                    "{} positions @ {budget}: regret {:.2} +/- {:.2} cp   depth {:.2}   nodes {:.0}   util {:.1}%   agree {:.1}%   off-list {:.1}%   ({:.1}s)",
                    r.positions,
                    r.regret,
                    r.stderr,
                    r.depth,
                    r.nodes,
                    r.util * 100.0,
                    r.agree * 100.0,
                    r.off_list * 100.0,
                    t.elapsed().as_secs_f64()
                );
            } else {
                // `fare` is excluded by default: it is the unit the other
                // parameters are denominated in, so moving it just rescales
                // them and the descent wanders along a flat direction.
                let free: Vec<String> = flag("--free")
                    .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
                    .unwrap_or_else(|| {
                        chess::search::Pricing::NAMES
                            .iter()
                            .filter(|n| **n != "fare")
                            .map(|n| n.to_string())
                            .collect()
                    });
                for n in &free {
                    if params.get(n).is_none() {
                        eprintln!("unknown pricing parameter {n}");
                        std::process::exit(3);
                    }
                }
                let test_frac = flag("--test-frac")
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(0.4);
                let (train, test) = chess::tune::split(&labels, test_frac);
                println!(
                    "{} fit / {} held out @ {budget}, {threads} threads, {} free parameters",
                    train.len(),
                    test.len(),
                    free.len()
                );
                let best = chess::tune::descend(
                    &train,
                    &test,
                    budget,
                    params,
                    &free,
                    num("--iters", 15) as usize,
                    threads,
                    cap,
                    flag("--depth-floor")
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(0.25),
                );
                println!("\n--set {}", chess::search::Pricing::NAMES
                    .iter()
                    .map(|n| format!("{n}={}", best.get(n).unwrap()))
                    .collect::<Vec<_>>()
                    .join(","));
            }
        }

        _ => {
            eprintln!("tune: expected label | eval | compare | search");
            std::process::exit(3);
        }
    }
}

/// Lazy SMP scaling, measured two ways, because they answer different
/// questions and only one of them is about strength.
///
/// **nps scaling** is how much raw throughput the threads add. It is nearly
/// free to get right and nearly useless on its own: N threads searching the
/// same tree N times over would report a perfect N-fold nps and be worth zero
/// Elo.
///
/// **Depth at a fixed time** is the one that matters, and it is why this
/// measures at a fixed *movetime* rather than a fixed depth. The number to
/// watch is the effective speed-up: how many times faster a single thread
/// would have to be to reach the same depth in the same time. Under the usual
/// branching factor an extra ply costs about 2x, so `+1.0 ply` is `2x
/// effective` no matter what nps says.
fn smp_bench(args: &[String]) {
    let flag = |k: &str, d: u64| -> u64 {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let movetime = flag("--movetime", 3000);
    let hash_mb = flag("--hash", 256) as usize;
    let max_threads =
        flag("--threads", std::thread::available_parallelism().map_or(4, |n| n.get()) as u64) as usize;
    let skip = flag("--skip", 1) as i32;
    let reps = flag("--reps", 1).max(1) as usize;
    // Time-to-depth is the honest scaling metric: it is continuous, where
    // depth-at-fixed-time is one integer per position and needs a lot of
    // positions before a fraction of a ply means anything.
    let fixed_depth = args
        .iter()
        .position(|a| a == "--depth")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<u32>().ok());
    let list: Vec<usize> = args
        .iter()
        .position(|a| a == "--list")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.split(',').filter_map(|t| t.parse().ok()).collect())
        .unwrap_or_else(|| {
            let mut v = vec![1];
            let mut t = 2;
            while t <= max_threads {
                v.push(t);
                t *= 2;
            }
            if *v.last().unwrap() != max_threads {
                v.push(max_threads);
            }
            v
        });

    let params = Params { smp_skip: skip, ..Default::default() };
    let npos = BENCH_FENS.len() * reps;
    match fixed_depth {
        Some(d) => println!(
            "{npos} searches, fixed depth {d}, {hash_mb} MB hash, smp_skip={skip}\n"
        ),
        None => println!(
            "{npos} searches, {movetime} ms each, {hash_mb} MB hash, smp_skip={skip}\n"
        ),
    }

    // [arm][search] -> (seconds, depth reached, nodes)
    let mut grid: Vec<Vec<(f64, u32, u64)>> = Vec::new();
    for &t in &list {
        // A fresh table per arm: a warm TT from the previous arm would hand
        // this one free depth, and the comparison would be measuring
        // carry-over rather than threads.
        let shared = Shared::new(hash_mb);
        let limits = match fixed_depth {
            Some(d) => Limits { depth: Some(d), ..Default::default() },
            None => Limits { movetime: Some(movetime), ..Default::default() },
        };
        let mut row = Vec::with_capacity(npos);
        for _ in 0..reps {
            for fen in BENCH_FENS {
                let b = Board::from_fen(fen).unwrap();
                shared.tt.clear();
                shared.nodes.store(0, std::sync::atomic::Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                // Fresh tables every search, like the TT clear above: this
                // measures threads, not history carry-over.
                let mut td = Vec::new();
                let r = chess::search::go_parallel(
                    &shared, &b, &[], &limits, params, t, DefaultEval::default, None,
                    &mut td,
                );
                let el = t0.elapsed().as_secs_f64();
                let n = shared
                    .nodes
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .max(r.nodes);
                row.push((el, r.depth, n));
            }
        }
        grid.push(row);
    }

    // A search that ended early because it proved a mate did less work than
    // the depth suggests, and different arms prove it at different depths. So
    // time-to-depth is scored only on the searches where EVERY arm came back
    // at the requested depth; anything else compares two different jobs.
    let usable: Vec<usize> = (0..npos)
        .filter(|&i| match fixed_depth {
            Some(d) => grid.iter().all(|r| r[i].1 == d),
            None => true,
        })
        .collect();

    println!(
        "{:>3}  {:>13}  {:>10}  {:>7}  {:>7}  {:>8}  {:>7}",
        "thr", "nodes", "nps", "nps x", "depth", "sec", "eff x"
    );
    let (mut base_nps, mut base_depth, mut base_time) = (0f64, 0f64, 0f64);
    for (a, &t) in list.iter().enumerate() {
        let row = &grid[a];
        let nodes: u64 = row.iter().map(|r| r.2).sum();
        let secs: f64 = row.iter().map(|r| r.0).sum();
        let nps = nodes as f64 / secs;
        let depth = row.iter().map(|r| r.1 as f64).sum::<f64>() / npos as f64;
        // Scored time: the subset all arms completed identically.
        let scored: f64 = usable.iter().map(|&i| row[i].0).sum();
        if a == 0 {
            base_nps = nps;
            base_depth = depth;
            base_time = scored;
        }
        // At a fixed depth the speed-up is measured directly, in wall time. At
        // a fixed time it has to be inferred from the extra depth: one ply is
        // worth about a factor of `EBF` in time, taken as 2.0. That is
        // conservative for this search — the price list gives it an effective
        // branching factor nearer 1.8, and a lower EBF would make the same
        // depth gain look LARGER, so 2.0 does not flatter the result.
        let eff = if fixed_depth.is_some() {
            base_time / scored
        } else {
            2f64.powf(depth - base_depth)
        };
        println!(
            "{t:>3}  {nodes:>13}  {nps:>10.0}  {:>6.2}x  {depth:>7.2}  {secs:>8.2}  {:>6.2}x",
            nps / base_nps,
            eff
        );
    }
    if fixed_depth.is_some() {
        println!(
            "\ntime-to-depth scored on {} of {npos} searches (the rest ended early on a \
             proven mate, at a different depth in different arms)",
            usable.len()
        );
    }
}
