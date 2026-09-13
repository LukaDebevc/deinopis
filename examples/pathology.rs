//! Is minimax pathology real, and do alternative backup operators fix it?
//!
//! Setup, following Nau/Beal properly this time:
//!   * A depth-D tree. Leaf values are generated hierarchically so that sibling
//!     correlation is tunable: a child's value is its parent's plus N(0, w_lvl).
//!     `leaf_frac` = share of total variance injected at the last level.
//!         1.0  -> leaves iid given nothing (pathological regime)
//!         1/D  -> uniform per level, leaves sharing k of D ancestors correlate k/D
//!   * True node values = exact minimax backup from the leaves.
//!   * A depth-K search backs up K levels, applying a NOISY static eval at the
//!     frontier:  eval(n) = true(n) + sigma * N(0,1).
//!
//! Pathology = regret at the root increasing with K.
//!
//! Three backup operators are compared on the SAME trees and SAME noise draws
//! (paired design, so differences are not sampling noise):
//!   max     plain minimax
//!   soft    softmax-weighted average of children, temperature tau.
//!           tau -> 0 recovers max; tau -> inf recovers the mean (MCTS-like).
//!   double  two independent eval streams; one picks the argmax child, the
//!           OTHER supplies its value. This is van Hasselt's Double Q-learning
//!           trick: it removes the optimism bias of E[max] > max E.

const B: usize = 3;
const D: usize = 10;

fn n_nodes() -> usize {
    let mut n = 1usize;
    let mut p = 1usize;
    for _ in 0..D {
        p *= B;
        n += p;
    }
    n
}

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

struct Tree {
    depth: Vec<u8>,
    tv: Vec<f64>, // true minimax value
    e1: Vec<f64>, // eval noise, stream A
    e2: Vec<f64>, // eval noise, stream B
}

#[inline(always)]
fn kids(i: usize) -> std::ops::Range<usize> {
    (B * i + 1)..(B * i + 1 + B)
}

impl Tree {
    fn new(n: usize) -> Tree {
        let mut depth = vec![0u8; n];
        for i in 1..n {
            depth[i] = depth[(i - 1) / B] + 1;
        }
        Tree { depth, tv: vec![0.0; n], e1: vec![0.0; n], e2: vec![0.0; n] }
    }

    fn generate(&mut self, sd: &[f64], rng: &mut Rng) {
        let n = self.tv.len();
        // Top-down: hierarchical values. Only the leaves' values are "truth".
        self.tv[0] = 0.0;
        for i in 1..n {
            let d = self.depth[i] as usize;
            self.tv[i] = self.tv[(i - 1) / B] + sd[d] * rng.normal();
        }
        // Bottom-up: exact minimax over the leaves overwrites internal nodes.
        for i in (0..n).rev() {
            if self.depth[i] as usize == D {
                continue;
            }
            let is_max = self.depth[i] % 2 == 0;
            let mut acc = self.tv[B * i + 1];
            for c in kids(i).skip(1) {
                let v = self.tv[c];
                if (is_max && v > acc) || (!is_max && v < acc) {
                    acc = v;
                }
            }
            self.tv[i] = acc;
        }
        for i in 0..n {
            self.e1[i] = rng.normal();
            self.e2[i] = rng.normal();
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Max,
    Soft(f64),
    Double,
}

/// Backed-up value of node `i` searching `r` more plies. Returns a pair of
/// value streams; only `Double` keeps them distinct.
fn search(t: &Tree, i: usize, r: usize, sigma: f64, op: Op) -> (f64, f64) {
    if r == 0 {
        return (t.tv[i] + sigma * t.e1[i], t.tv[i] + sigma * t.e2[i]);
    }
    let is_max = t.depth[i] % 2 == 0;
    let mut a = [0.0f64; B];
    let mut b = [0.0f64; B];
    for (j, c) in kids(i).enumerate() {
        let (x, y) = search(t, c, r - 1, sigma, op);
        a[j] = x;
        b[j] = y;
    }
    match op {
        Op::Max => {
            let mut v = a[0];
            for &x in &a[1..] {
                if (is_max && x > v) || (!is_max && x < v) {
                    v = x;
                }
            }
            (v, v)
        }
        Op::Soft(tau) => {
            // Softmax-weighted mean. Sign flips at min nodes so the same
            // temperature means the same thing on both.
            let s = if is_max { 1.0 } else { -1.0 };
            let m = a.iter().cloned().fold(f64::NEG_INFINITY, |p, q| p.max(s * q));
            let mut wsum = 0.0;
            let mut vsum = 0.0;
            for &x in &a {
                let w = ((s * x - m) / tau).exp();
                wsum += w;
                vsum += w * x;
            }
            let v = vsum / wsum;
            (v, v)
        }
        Op::Double => {
            // Select with A, evaluate with B (and symmetrically), so the value
            // returned is never the same sample that won the argmax.
            let mut ja = 0;
            let mut jb = 0;
            for j in 1..B {
                if (is_max && a[j] > a[ja]) || (!is_max && a[j] < a[ja]) {
                    ja = j;
                }
                if (is_max && b[j] > b[jb]) || (!is_max && b[j] < b[jb]) {
                    jb = j;
                }
            }
            (b[ja], a[jb])
        }
    }
}

struct Acc {
    regret: Vec<f64>,
    flip: Vec<f64>,
    rootval: Vec<f64>,
}
impl Acc {
    fn new() -> Acc {
        Acc { regret: vec![0.0; D + 1], flip: vec![0.0; D + 1], rootval: vec![0.0; D + 1] }
    }
}

fn run(leaf_frac: f64, sigma: f64, trials: usize, ops: &[(&str, Op)], seed: u64) {
    let mut sd = vec![0.0f64; D + 1];
    let spread = (1.0 - leaf_frac) / (D - 1) as f64;
    for (l, s) in sd.iter_mut().enumerate().take(D + 1).skip(1) {
        *s = (if l == D { leaf_frac } else { spread }).sqrt();
    }

    let n = n_nodes();
    let mut tree = Tree::new(n);
    let mut rng = Rng(seed);
    let mut accs: Vec<Acc> = ops.iter().map(|_| Acc::new()).collect();
    // Mean spread of true values among the root's grandchildren-at-depth-K,
    // the quantity sigma has to compete with.
    let mut spread_at = vec![0.0f64; D + 1];

    for _ in 0..trials {
        tree.generate(&sd, &mut rng);
        let truth: Vec<f64> = kids(0).map(|c| tree.tv[c]).collect();
        let best = truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        for k in 1..=D {
            let lo = {
                let mut a = 0usize;
                let mut p = 1usize;
                for _ in 0..k {
                    p *= B;
                    a += p / B;
                }
                a
            };
            let hi = lo + {
                let mut p = 1usize;
                for _ in 0..k {
                    p *= B;
                }
                p
            };
            let vals = &tree.tv[lo..hi.min(n)];
            let mu = vals.iter().sum::<f64>() / vals.len() as f64;
            spread_at[k] += (vals.iter().map(|v| (v - mu).powi(2)).sum::<f64>()
                / vals.len() as f64)
                .sqrt();
        }

        for (oi, (_, op)) in ops.iter().enumerate() {
            let mut prev = usize::MAX;
            for k in 1..=D {
                let mut choice = 0usize;
                let mut bestv = f64::NEG_INFINITY;
                for (j, c) in kids(0).enumerate() {
                    let (x, y) = search(&tree, c, k - 1, sigma, *op);
                    let v = 0.5 * (x + y);
                    if v > bestv {
                        bestv = v;
                        choice = j;
                    }
                }
                accs[oi].rootval[k] += bestv;
                accs[oi].regret[k] += best - truth[choice];
                if prev != usize::MAX && choice != prev {
                    accs[oi].flip[k] += 1.0;
                }
                prev = choice;
            }
        }
    }

    let t = trials as f64;
    println!("leaf_frac={leaf_frac:.2}  sigma={sigma:.2}  ({trials} trials)");
    print!("  depth K            ");
    for k in 1..=D {
        print!("{k:>7}");
    }
    println!();
    print!("  sibling spread     ");
    for k in 1..=D {
        print!("{:>7.3}", spread_at[k] / t);
    }
    println!();
    for (oi, (name, _)) in ops.iter().enumerate() {
        print!("  regret  {name:<10} ");
        for k in 1..=D {
            print!("{:>7.4}", accs[oi].regret[k] / t);
        }
        println!();
    }
    for (oi, (name, _)) in ops.iter().enumerate() {
        print!("  flip    {name:<10} ");
        for k in 1..=D {
            print!("{:>7.3}", accs[oi].flip[k] / t);
        }
        println!();
    }
    print!("  drift   max        ");
    for k in 1..=D {
        print!("{:>7.3}", accs[0].rootval[k] / t);
    }
    println!("\n");
}

fn main() {
    let trials: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    let ops: [(&str, Op); 4] = [
        ("max", Op::Max),
        ("soft t=.3", Op::Soft(0.3)),
        ("soft t=1", Op::Soft(1.0)),
        ("double", Op::Double),
    ];
    println!("b={B} d={D}\n");
    for (lf, sg) in [(1.0, 0.5), (1.0, 1.5), (0.10, 0.5), (0.10, 1.5), (0.10, 3.0)] {
        run(lf, sg, trials, &ops, 0x1234_5678_9ABC_DEF0 ^ lf.to_bits() ^ (sg.to_bits() << 1));
    }
}
