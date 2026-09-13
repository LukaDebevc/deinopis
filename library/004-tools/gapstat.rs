//! Two measurements the price-list theory needs, from one pass over positions.
//!
//! 1. GAMMA: how fast search error decays with budget. sd(V_n(c) - V_ref(c))
//!    against n. The allocation law predicts price ~ (PLY/gamma) * log2(gap).
//! 2. DILUTION: how good `static_eval + move_gain` is as a predictor of the
//!    true child value. `gap` in Pricing::price is built from that predictor,
//!    so the usable coefficient is the theoretical one times the regression
//!    slope of true-gap on predicted-gap, in the same log space the price uses.

use chess::board::Board;
use chess::chess_move::MoveList;
use chess::eval::{evaluate_pst, move_gain, PstEval, Score, MATE_BOUND};
use chess::movegen::{generate, GenType};
use chess::search::{Limits, Params, Searcher, Shared, ThreadData};

const LADDER: [u64; 5] = [2_000, 8_000, 32_000, 128_000, 512_000];
const CLAMP: Score = 1500;

fn search_once(shared: &Shared, b: &Board, nodes: u64) -> (Score, u32) {
    shared.tt.clear();
    shared.stop.store(false, std::sync::atomic::Ordering::Relaxed);
    let mut s = Searcher::new(shared, ThreadData::new(0), PstEval, Params::default());
    let r = s.go(b, &[], &Limits { nodes: Some(nodes), ..Default::default() }, None);
    (r.score, r.depth)
}

/// log2(1 + max(g,0)/unit), the exact shape Pricing::price uses, in f64.
fn lgap(g: f64, unit: f64) -> f64 {
    (1.0 + g.max(0.0) / unit).log2()
}

struct Row {
    /// value of each child at each ladder rung, stm-relative to the parent
    v: Vec<[Score; LADDER.len()]>,
    pred: Vec<Score>,
    depth: Vec<[u32; LADDER.len()]>,
}

fn main() {
    chess::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |flag: &str, d: usize| -> usize {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok()).unwrap_or(d)
    };
    let max = num("--max", 120);
    let threads = num("--threads", 6);
    let top = num("--top", 8);

    let pgns: Vec<String> = std::fs::read_dir(".ladder").map(|d| {
        let mut v: Vec<String> = d.filter_map(|e| e.ok())
            .map(|e| e.path().to_string_lossy().to_string())
            .filter(|p| p.ends_with(".pgn")).collect();
        v.sort();
        v
    }).unwrap_or_default();
    if pgns.is_empty() { eprintln!("no PGNs in .ladder/"); return; }

    let positions = chess::tune::positions_from_pgn(&pgns, 16, 7, max).unwrap();
    eprintln!("{} positions, top {top} moves each, ladder {:?}", positions.len(), LADDER);

    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<Row>>> =
        (0..positions.len()).map(|_| std::sync::Mutex::new(None)).collect();

    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let shared = Shared::new(32);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= positions.len() { break }
                    let b = &positions[i];
                    let se = evaluate_pst(b);
                    let mut moves = MoveList::new();
                    generate(b, GenType::All, &mut moves);
                    // rank by the engine's own cheap predictor, keep the top few:
                    // the tail is where every policy agrees to spend nothing.
                    let mut cand: Vec<(chess::chess_move::Move, Score)> = (0..moves.len())
                        .map(|k| { let mv = moves.get(k); (mv, se + move_gain(b, mv)) })
                        .collect();
                    cand.sort_by_key(|c| -c.1);
                    cand.truncate(top);

                    let mut row = Row { v: vec![], pred: vec![], depth: vec![] };
                    for (mv, p) in &cand {
                        let child = b.make_move(*mv);
                        let mut vs = [0; LADDER.len()];
                        let mut ds = [0; LADDER.len()];
                        for (j, n) in LADDER.iter().enumerate() {
                            let (s, d) = search_once(&shared, &child, *n);
                            vs[j] = -s;
                            ds[j] = d;
                        }
                        row.v.push(vs);
                        row.pred.push(*p);
                        row.depth.push(ds);
                    }
                    *out[i].lock().unwrap() = Some(row);
                    let d = i + 1;
                    if d % 10 == 0 { eprint!("\r  {d}/{}", positions.len()) }
                }
            });
        }
    });
    eprintln!("\r  done{}", " ".repeat(20));

    let rows: Vec<Row> = out.into_iter().filter_map(|m| m.into_inner().unwrap()).collect();
    let refi = LADDER.len() - 1;

    // ---------------- gamma: sd of (V_n - V_ref) vs n
    println!("\n== search error decay ==");
    println!("{:>9}  {:>8}  {:>7}  {:>6}", "nodes", "sd(cp)", "mean|d|", "depth");
    let mut pts: Vec<(f64, f64)> = vec![];
    for j in 0..refi {
        let (mut s, mut ss, mut n, mut ad, mut dsum) = (0f64, 0f64, 0usize, 0f64, 0f64);
        for r in &rows {
            for (k, vs) in r.v.iter().enumerate() {
                if vs[refi].abs() >= MATE_BOUND { continue }
                let d = (vs[j] - vs[refi]).clamp(-CLAMP, CLAMP) as f64;
                s += d; ss += d * d; n += 1; ad += d.abs();
                dsum += r.depth[k][j] as f64;
            }
        }
        if n == 0 { continue }
        let (m, nf) = (s / n as f64, n as f64);
        let sd = (ss / nf - m * m).max(0.0).sqrt();
        println!("{:>9}  {:>8.1}  {:>7.1}  {:>6.2}", LADDER[j], sd, ad / nf, dsum / nf);
        pts.push(((LADDER[j] as f64).log2(), sd.log2()));
    }
    // slope of log2(sd) vs log2(nodes)
    let n = pts.len() as f64;
    let (sx, sy) = (pts.iter().map(|p| p.0).sum::<f64>(), pts.iter().map(|p| p.1).sum::<f64>());
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let gam_nodes = -(n * sxy - sx * sy) / (n * sxx - sx * sx);

    // empirical branching factor from depth vs nodes at the ladder ends
    let davg = |j: usize| -> f64 {
        let (mut s, mut c) = (0f64, 0f64);
        for r in &rows { for d in &r.depth { s += d[j] as f64; c += 1.0 } }
        s / c
    };
    let (d0, d1) = (davg(0), davg(refi - 1));
    let ebf = 2f64.powf(((LADDER[refi - 1] as f64 / LADDER[0] as f64).log2()) / (d1 - d0));
    let gam_ply = gam_nodes * ebf.log2();
    println!("\n  sd ~ nodes^-{gam_nodes:.4}     empirical EBF {ebf:.2}");
    println!("  => gamma (per ply) = {gam_ply:.3}   [sd halves every {:.1} plies]", 1.0 / gam_ply);
    println!("  => theoretical c_gap = PLY/gamma = {:.0} milli-plies", 1000.0 / gam_ply);

    // ---------------- dilution: regress true gap on predicted gap, in log space
    println!("\n== gap predictor quality ==");
    for unit in [32.0, 64.0, 128.0] {
        let (mut sx, mut sy, mut sxy, mut sxx, mut syy, mut n) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for r in &rows {
            if r.v.is_empty() { continue }
            if r.v.iter().any(|v| v[refi].abs() >= MATE_BOUND) { continue }
            let bp = *r.pred.iter().max().unwrap() as f64;
            let bt = r.v.iter().map(|v| v[refi]).max().unwrap() as f64;
            for (k, vs) in r.v.iter().enumerate() {
                let x = lgap(bp - r.pred[k] as f64, unit);
                let y = lgap(bt - vs[refi] as f64, unit);
                sx += x; sy += y; sxy += x * y; sxx += x * x; syy += y * y; n += 1.0;
            }
        }
        let cov = sxy / n - (sx / n) * (sy / n);
        let vx = sxx / n - (sx / n).powi(2);
        let vy = syy / n - (sy / n).powi(2);
        let slope = cov / vx;
        let r2 = cov * cov / (vx * vy);
        println!("  gap_unit {unit:>5.0}:  slope(true~pred) = {slope:.3}   r = {:.3}   r2 = {r2:.3}   n = {n:.0}",
                 r2.sqrt());
        println!("                  => usable c_gap = {:.0} * {:.3} = {:.0} milli-plies",
                 1000.0 / gam_ply, slope, 1000.0 / gam_ply * slope);
    }
}
