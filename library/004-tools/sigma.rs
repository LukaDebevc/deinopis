//! Is there a cheap feature that predicts how *uncertain* a shallow search is?
//!
//! The allocation law says budget carries +log(sigma) with the same magnitude
//! as -log(gap). `features.rs` found nothing, but it regressed on single
//! |deviations|, where even a perfect predictor is capped near r2 = 0.3
//! because |d|/sigma is itself random. This bins by the candidate instead and
//! reports the sd *within* each bin, which is the quantity sigma actually is.
//!
//! Nothing here may use V_2k: a candidate built from the same search as the
//! target correlates with it through its own outliers.

use chess::board::Board;
use chess::chess_move::{Move, MoveList};
use chess::eval::{evaluate_pst, move_gain, PstEval, Score, MATE, MATE_BOUND};
use chess::movegen::{generate, GenType};
use chess::search::{Limits, Params, Searcher, Shared, ThreadData};
use chess::types::PieceType;

fn search_once(shared: &Shared, b: &Board, nodes: u64) -> Score {
    shared.tt.clear();
    shared.stop.store(false, std::sync::atomic::Ordering::Relaxed);
    let mut s = Searcher::new(shared, ThreadData::new(0), PstEval, Params::default());
    s.go(b, &[], &Limits { nodes: Some(nodes), ..Default::default() }, None).score
}
fn qsearch(shared: &Shared, b: &Board) -> Score {
    shared.stop.store(false, std::sync::atomic::Ordering::Relaxed);
    let mut s = Searcher::new(shared, ThreadData::new(0), PstEval, Params::default());
    s.quiescence(b, -MATE, MATE, 0)
}

const CAND: [&str; 5] = [
    "|qsearch - 1k search|",
    "|static - qsearch|",
    "sibling spread of move_gain",
    "legal move count",
    "non-pawn material",
];

fn main() {
    chess::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |f: &str, d: usize| args.iter().position(|a| a == f)
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (max, threads, top) = (num("--max", 300), num("--threads", 6), num("--top", 8));

    let mut pgns: Vec<String> = std::fs::read_dir(".ladder").unwrap().filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().to_string())
        .filter(|p| p.ends_with(".pgn")).collect();
    pgns.sort();
    let positions = chess::tune::positions_from_pgn(&pgns, 16, 7, max).unwrap();
    eprintln!("{} positions", positions.len());

    let next = std::sync::atomic::AtomicUsize::new(0);
    // (candidate values, deviation V_2k - V_ref)
    let acc: std::sync::Mutex<Vec<([f64; CAND.len()], f64)>> = std::sync::Mutex::new(vec![]);

    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let shared = Shared::new(32);
                let mut local = vec![];
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= positions.len() { break }
                    let b = &positions[i];
                    let se = evaluate_pst(b);
                    let mut ml = MoveList::new();
                    generate(b, GenType::All, &mut ml);
                    let nmoves = ml.len() as f64;
                    let mut cand: Vec<(Move, Score)> = (0..ml.len())
                        .map(|k| { let mv = ml.get(k); (mv, se + move_gain(b, mv)) }).collect();
                    cand.sort_by_key(|c| -c.1);
                    let spread = {
                        let v: Vec<f64> = cand.iter().map(|c| c.1 as f64).collect();
                        let m = v.iter().sum::<f64>() / v.len() as f64;
                        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
                    };
                    let npm = [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen]
                        .iter().map(|p| b.pieces(*p).count() as f64).sum::<f64>();
                    cand.truncate(top);

                    for (mv, _) in &cand {
                        let child = b.make_move(*mv);
                        let vref = -search_once(&shared, &child, 512_000);
                        if vref.abs() >= MATE_BOUND { continue }
                        let v2k = -search_once(&shared, &child, 2_000);
                        let v1k = -search_once(&shared, &child, 1_000);
                        let vq = -qsearch(&shared, &child);
                        let cst = -evaluate_pst(&child);
                        local.push(([
                            (vq - v1k).abs() as f64,
                            (cst - vq).abs() as f64,
                            spread, nmoves, npm,
                        ], (v2k - vref).clamp(-1500, 1500) as f64));
                    }
                }
                acc.lock().unwrap().extend(local);
            });
        }
    });

    let data = acc.into_inner().unwrap();
    let n = data.len();
    let all_sd = {
        let m: f64 = data.iter().map(|d| d.1).sum::<f64>() / n as f64;
        (data.iter().map(|d| (d.1 - m).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    println!("\n{n} child samples.  overall sd(V_2k - V_ref) = {all_sd:.1} cp\n");
    println!("Candidate sigma estimators, quintiles of the candidate:");
    println!("{:<30} {:>8} {:>8} {:>8} {:>8} {:>8}   {:>6}",
             "candidate", "Q1 sd", "Q2 sd", "Q3 sd", "Q4 sd", "Q5 sd", "Q5/Q1");
    for j in 0..CAND.len() {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| data[a].0[j].partial_cmp(&data[b].0[j]).unwrap());
        let mut sds = vec![];
        for q in 0..5 {
            let (lo, hi) = (q * n / 5, (q + 1) * n / 5);
            let sl = &idx[lo..hi];
            let m: f64 = sl.iter().map(|&i| data[i].1).sum::<f64>() / sl.len() as f64;
            sds.push((sl.iter().map(|&i| (data[i].1 - m).powi(2)).sum::<f64>() / sl.len() as f64).sqrt());
        }
        println!("{:<30} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1}   {:>6.2}",
                 CAND[j], sds[0], sds[1], sds[2], sds[3], sds[4], sds[4] / sds[0].max(1e-9));
    }
    {
        let mut t = String::from("qs_1k\tst_qs\tspread\tnmoves\tnpm\tdev\n");
        for (c, d) in &data {
            t.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\n", c[0], c[1], c[2], c[3], c[4], d));
        }
        std::fs::write("sigma.tsv", t).unwrap();
        eprintln!("wrote sigma.tsv");
    }
    println!("\n  A usable sigma estimator needs Q5/Q1 well above 1. Monotone matters");
    println!("  more than the ratio: a non-monotone column is not a scale, it is noise.");
}
