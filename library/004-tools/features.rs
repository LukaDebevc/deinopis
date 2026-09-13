//! Which cheap features actually predict what a deep search will say?
//!
//! `Pricing` prices a child on `gap = best - static_eval - move_gain`. gapstat
//! showed that predictor carries r2 = 0.03 against the true gap. This asks
//! whether the OCBA story is wrong or the *feature* is: it scores four
//! candidate predictors of the same target, and two candidate estimators of
//! the local noise scale sigma that the allocation law says must appear.

use chess::board::Board;
use chess::chess_move::{Move, MoveList};
use chess::eval::{evaluate_pst, move_gain, see, PstEval, Score, MATE, MATE_BOUND};
use chess::movegen::{generate, GenType};
use chess::search::{Limits, Params, Searcher, Shared, ThreadData};

const REF: u64 = 512_000;
const SHALLOW: u64 = 1_000;

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

fn lgap(g: f64) -> f64 { (1.0 + g.max(0.0) / 32.0).log2() }

/// One (predictor, target) accumulator.
#[derive(Default, Clone)]
struct Fit { sx: f64, sy: f64, sxy: f64, sxx: f64, syy: f64, n: f64 }
impl Fit {
    fn add(&mut self, x: f64, y: f64) {
        self.sx += x; self.sy += y; self.sxy += x * y;
        self.sxx += x * x; self.syy += y * y; self.n += 1.0;
    }
    fn merge(&mut self, o: &Fit) {
        self.sx += o.sx; self.sy += o.sy; self.sxy += o.sxy;
        self.sxx += o.sxx; self.syy += o.syy; self.n += o.n;
    }
    /// (slope of y on x, r, r2)
    fn stats(&self) -> (f64, f64, f64) {
        let n = self.n;
        let cov = self.sxy / n - (self.sx / n) * (self.sy / n);
        let vx = self.sxx / n - (self.sx / n).powi(2);
        let vy = self.syy / n - (self.sy / n).powi(2);
        let r2 = if vx * vy > 0.0 { cov * cov / (vx * vy) } else { 0.0 };
        (cov / vx.max(1e-12), r2.sqrt() * cov.signum(), r2)
    }
}

const PRED: [&str; 4] = ["move_gain (current)", "move_gain + SEE veto", "qsearch of child", "1k-node search"];
const SIG: [&str; 3] = ["|static - qsearch| (child)", "sibling spread of move_gain", "|static - qsearch| (parent)"];

fn main() {
    chess::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |f: &str, d: usize| args.iter().position(|a| a == f)
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(d);
    let max = num("--max", 300);
    let threads = num("--threads", 6);
    let top = num("--top", 8);

    let mut pgns: Vec<String> = std::fs::read_dir(".ladder").unwrap().filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().to_string())
        .filter(|p| p.ends_with(".pgn")).collect();
    pgns.sort();
    let positions = chess::tune::positions_from_pgn(&pgns, 16, 7, max).unwrap();
    eprintln!("{} positions, top {top} moves", positions.len());

    let next = std::sync::atomic::AtomicUsize::new(0);
    // [predictor] -> Fit of log-gap(true) on log-gap(pred)
    let gfits = std::sync::Mutex::new(vec![Fit::default(); PRED.len()]);
    // [sigma estimator] -> Fit of log|V_2k - V_ref| on log(sigma-hat)
    let sfits = std::sync::Mutex::new(vec![Fit::default(); SIG.len()]);
    // raw-space fit of true child value on each predictor, for scale
    let rawfits = std::sync::Mutex::new(vec![Fit::default(); PRED.len()]);

    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let shared = Shared::new(32);
                let mut g = vec![Fit::default(); PRED.len()];
                let mut s = vec![Fit::default(); SIG.len()];
                let mut raw = vec![Fit::default(); PRED.len()];
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= positions.len() { break }
                    let b = &positions[i];
                    let se = evaluate_pst(b);
                    let p_unstable = (se - qsearch(&shared, b)).abs() as f64;

                    let mut ml = MoveList::new();
                    generate(b, GenType::All, &mut ml);
                    let mut cand: Vec<(Move, Score)> = (0..ml.len())
                        .map(|k| { let mv = ml.get(k); (mv, se + move_gain(b, mv)) }).collect();
                    cand.sort_by_key(|c| -c.1);
                    let spread = {
                        let v: Vec<f64> = cand.iter().map(|c| c.1 as f64).collect();
                        let m = v.iter().sum::<f64>() / v.len() as f64;
                        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
                    };
                    cand.truncate(top);

                    // predictors and truth for each candidate move
                    let mut pv: Vec<[f64; PRED.len()]> = vec![];
                    let mut tv: Vec<f64> = vec![];
                    let mut v2k: Vec<f64> = vec![];
                    let mut cinst: Vec<f64> = vec![];
                    let mut bad = false;
                    for (mv, gain) in &cand {
                        let child = b.make_move(*mv);
                        let vref = -search_once(&shared, &child, REF);
                        if vref.abs() >= MATE_BOUND { bad = true; break }
                        let cq = qsearch(&shared, &child);
                        let cst = evaluate_pst(&child);
                        // SEE veto: a capture that loses material keeps no
                        // material credit at all, only the static eval.
                        let seev = if mv.is_capture() && !see(b, *mv, 0) { se } else { *gain };
                        pv.push([
                            *gain as f64,
                            seev as f64,
                            -cq as f64,
                            -search_once(&shared, &child, SHALLOW) as f64,
                        ]);
                        tv.push(vref as f64);
                        v2k.push(-search_once(&shared, &child, 2_000) as f64);
                        cinst.push((cst - cq).abs() as f64);
                    }
                    if bad || tv.len() < 2 { continue }

                    let bt = tv.iter().cloned().fold(f64::MIN, f64::max);
                    for j in 0..PRED.len() {
                        let bp = pv.iter().map(|p| p[j]).fold(f64::MIN, f64::max);
                        for k in 0..tv.len() {
                            g[j].add(lgap(bp - pv[k][j]), lgap(bt - tv[k]));
                            raw[j].add(pv[k][j], tv[k]);
                        }
                    }
                    for k in 0..tv.len() {
                        let err = ((v2k[k] - tv[k]).abs() + 1.0).ln();
                        s[0].add((cinst[k] + 1.0).ln(), err);
                        s[1].add((spread + 1.0).ln(), err);
                        s[2].add((p_unstable + 1.0).ln(), err);
                    }
                }
                let mut lg = gfits.lock().unwrap();
                for j in 0..PRED.len() { lg[j].merge(&g[j]) }
                drop(lg);
                let mut ls = sfits.lock().unwrap();
                for j in 0..SIG.len() { ls[j].merge(&s[j]) }
                drop(ls);
                let mut lr = rawfits.lock().unwrap();
                for j in 0..PRED.len() { lr[j].merge(&raw[j]) }
            });
        }
    });

    // gamma from gapstat: sd halves every ~7.9 plies -> PLY/gamma ~ 7900
    let ply_over_gamma = 7908.0;
    println!("\n== predictors of the true gap (log space, the space price() uses) ==");
    println!("{:<24} {:>8} {:>8} {:>8}   {:>16}", "predictor", "slope", "r", "r2", "usable c_gap");
    let g = gfits.lock().unwrap();
    let r = rawfits.lock().unwrap();
    for j in 0..PRED.len() {
        let (sl, rr, r2) = g[j].stats();
        println!("{:<24} {sl:>8.3} {rr:>8.3} {r2:>8.3}   {:>16.0}", PRED[j], ply_over_gamma * sl);
    }
    println!("\n  raw-space (centipawns) fit of true child value on predictor:");
    for j in 0..PRED.len() {
        let (sl, rr, r2) = r[j].stats();
        println!("    {:<24} slope {sl:>6.3}  r {rr:>6.3}  r2 {r2:>6.3}", PRED[j]);
    }

    println!("\n== estimators of sigma (does anything predict how wrong a shallow search is?) ==");
    println!("{:<30} {:>8} {:>8} {:>8}", "sigma-hat", "slope", "r", "r2");
    let s = sfits.lock().unwrap();
    for j in 0..SIG.len() {
        let (sl, rr, r2) = s[j].stats();
        println!("{:<30} {sl:>8.3} {rr:>8.3} {r2:>8.3}", SIG[j]);
    }
    println!("\n  (target = ln|V_2k - V_ref|; slope is the exponent of sigma-hat in a power law)");
}
