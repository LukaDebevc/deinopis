//! Price a hysteresis rule on total material, on the real tree.
//!
//! The rule under test (Luka's): 16 buckets whose CENTRES are 14, 18, ... 74,
//! covering the 0..78 range of `2 * (8*1 + 2*3 + 2*3 + 2*5 + 9)`. A node keeps
//! whatever bucket it inherited from its parent for as long as the material is
//! within `slack` of that bucket's centre; when it drifts out, the centre moves
//! by one step towards the material and the accumulator is rebuilt.
//!
//! Two things make this different from every rule priced so far. It reads only
//! the material SUM, which a capture moves by 1, 3, 5 or 9 and a quiet move not
//! at all -- so quiet moves, 69% of the tree's made moves, are free by
//! construction. And the overlap means a capture near a boundary does not
//! rebuild the way a hard partition's would; it costs a rebuild only once the
//! drift has accumulated.
//!
//! That second property is PATH-DEPENDENT, which is why this cannot be priced
//! from the `movedump` edge list the way LEDGER 034's rules were: the bucket in
//! force at a node depends on the whole line from the root, not on the node.
//! So it is counted here instead, on the search stack, where the path is.
//!
//! `hard` is the same 16 buckets with no hysteresis -- nearest centre, rebuild
//! whenever the index differs. It is the comparison that says what the overlap
//! is worth. Compiled out unless `--features hyst`.

use crate::board::Board;
use crate::search::MAX_PLY;
use crate::types::{Color, PieceType};

/// Centre of bucket 0, and the gap between centres.
pub const LO: i32 = 14;
pub const STEP: i32 = 4;
pub const N: usize = 16;
/// Rebuild when `|material - centre| >= slack`. 8 is Luka's proposal; the
/// others bracket it so the price comes out as a curve rather than a point.
pub const SLACKS: [i32; 4] = [6, 8, 10, 12];
pub const NV: usize = SLACKS.len();

/// Both sides, pawns through queens. Kings are not counted: they never leave.
#[inline]
pub fn material(b: &Board) -> i32 {
    const V: [i32; 5] = [1, 3, 3, 5, 9];
    let mut m = 0;
    for (i, pt) in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
                    PieceType::Rook, PieceType::Queen].iter().enumerate() {
        m += V[i] * b.pieces(*pt).count() as i32;
    }
    m
}

#[inline]
pub const fn centre(k: i32) -> i32 {
    LO + STEP * k
}

/// Nearest centre, clamped. The no-hysteresis control.
#[inline]
pub fn hard(mat: i32) -> u8 {
    let k = (mat - LO + STEP / 2).div_euclid(STEP);
    k.clamp(0, N as i32 - 1) as u8
}

/// The bucket in force after a move, given the parent's and the child's
/// material. Loops because one capture can be worth more than one step: a
/// queen taken at the far edge of the window moves the centre twice.
#[inline]
pub fn step(k: u8, mat: i32, slack: i32) -> u8 {
    let mut k = k as i32;
    loop {
        let c = centre(k);
        if (mat - c).abs() < slack {
            break;
        }
        let next = if mat > c { k + 1 } else { k - 1 };
        if next < 0 || next >= N as i32 {
            break;
        }
        k = next;
    }
    k as u8
}

/// The `material 24^2` rule from LEDGER 034, counted the same way, purely as a
/// unit check: its price is on record as 22.55% and if this instrument says
/// 11.3% then that record is per-ply-across-BOTH-accumulators and every number
/// here has to be doubled before the two can be compared. The light/dark
/// bishop bits are assigned without the `sq ^ 56` flip the trainer uses, which
/// swaps the two bits globally -- a relabelling, so the CHANGE RATE is
/// unaffected, which is all this is measuring.
#[inline]
pub fn material24(b: &Board) -> u16 {
    let mut code = [0u16; 2];
    const LIGHT: u64 = 0x55AA_55AA_55AA_55AA;
    for c in [crate::types::Color::White, crate::types::Color::Black] {
        let r = b.colored(c, PieceType::Rook).count().min(2) as u16;
        let bb = b.colored(c, PieceType::Bishop).0;
        let lb = ((bb & LIGHT) != 0) as u16;
        let db = ((bb & !LIGHT) != 0) as u16;
        let q = (b.colored(c, PieceType::Queen).count() > 0) as u16;
        code[c.index()] = ((r * 2 + lb) * 2 + db) * 2 + q;
    }
    code[0] * 24 + code[1]
}

/// Per-thread tallies. Boxed in `ThreadData` so the normal build's struct is
/// untouched and this one's is not 128 plies bigger.
pub struct Tally {
    /// Bucket in force at each ply, per slack. Indexed by ply, so the DFS
    /// gives each node its own root-to-here path for free.
    pub k: [[u8; MAX_PLY]; NV],
    /// Made moves, `[main, q]`. The denominator.
    pub edges: [u64; 2],
    /// Rebuilds under each slack, `[main, q]`.
    pub rebuild: [[u64; 2]; NV],
    /// Rebuilds under the hard partition, `[main, q]`.
    pub hard: [u64; 2],
    /// `material 24^2` changes, `[main, q]`. The unit check.
    pub m24: [u64; 2],
    /// Edges where either king square changed (only the mover's can). That is
    /// the refresh rate a HalfKA-style input table would pay: one perspective
    /// rebuilds per king move. Same denominator as every row above, so this is
    /// the apples-to-apples comparison the material rows needed.
    pub king: [u64; 2],
    /// Finny-table simulation: last-seen 12-bitboard placement per (king
    /// square, colour), the exact Stockfish `49ef4c9` scheme. On a king move by
    /// side `c` to square `k`, a Finny refresh applies only the placement diff
    /// against the cached entry instead of a full rebuild, so the distribution
    /// of `diff` is the price of the scheme. `cold` is first visits (full
    /// rebuild unavoidable); `pieces` is the mean piece count, i.e. what a
    /// full rebuild sums over. Arrays are simulation state, not tallies --
    /// `merge` takes the scalars only, so each bench position starts cold.
    pub fin_last: [[[u64; 12]; 2]; 64],
    pub fin_valid: [[bool; 2]; 64],
    pub fin_n: u64,
    pub fin_cold: u64,
    pub fin_sum: u64,
    pub fin_pieces: u64,
    pub fin_hist: [u64; 6],
    /// Set once per root so ply 0 starts from a real build.
    pub seeded: bool,
}

impl Default for Tally {
    fn default() -> Tally {
        Tally {
            k: [[0; MAX_PLY]; NV],
            edges: [0; 2],
            rebuild: [[0; 2]; NV],
            hard: [0; 2],
            m24: [0; 2],
            king: [0; 2],
            fin_last: [[[0; 12]; 2]; 64],
            fin_valid: [[false; 2]; 64],
            fin_n: 0,
            fin_cold: 0,
            fin_sum: 0,
            fin_pieces: 0,
            fin_hist: [0; 6],
            seeded: false,
        }
    }
}

impl Tally {
    /// One made move: `b` is the parent, `nb` the child, `site` 0 main / 1 q.
    pub fn record(&mut self, b: &Board, nb: &Board, ply: usize, site: usize) {
        let mp = material(b);
        let mc = material(nb);
        if !self.seeded {
            let k0 = hard(mp);
            for v in 0..NV {
                self.k[v][0] = k0;
            }
            self.seeded = true;
        }
        self.edges[site] += 1;
        if hard(mc) != hard(mp) {
            self.hard[site] += 1;
        }
        if material24(nb) != material24(b) {
            self.m24[site] += 1;
        }
        let kmoved = b.colored(Color::White, PieceType::King).lsb()
                != nb.colored(Color::White, PieceType::King).lsb()
            || b.colored(Color::Black, PieceType::King).lsb()
                != nb.colored(Color::Black, PieceType::King).lsb();
        if kmoved {
            self.king[site] += 1;
        }
        // Finny simulation for the side whose king moved (at most one can).
        // The toggle diff is exact for a HalfKA input: each changed
        // (colour, piece, square) membership is one feature add or remove.
        const PTS: [PieceType; 6] = [PieceType::Pawn, PieceType::Knight,
            PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King];
        for c in [Color::White, Color::Black] {
            if b.colored(c, PieceType::King).lsb()
                == nb.colored(c, PieceType::King).lsb()
            {
                continue;
            }
            let k = nb.colored(c, PieceType::King).lsb().index();
            let ci = c.index();
            let mut cur = [0u64; 12];
            for (pi, pp) in PTS.iter().enumerate() {
                cur[pi] = nb.colored(Color::White, *pp).0;
                cur[6 + pi] = nb.colored(Color::Black, *pp).0;
            }
            self.fin_n += 1;
            self.fin_pieces += cur.iter().map(|x| x.count_ones() as u64).sum::<u64>();
            if !self.fin_valid[k][ci] {
                self.fin_cold += 1;
            } else {
                let mut d = 0u32;
                for i in 0..12 {
                    d += (cur[i] ^ self.fin_last[k][ci][i]).count_ones();
                }
                self.fin_sum += d as u64;
                self.fin_hist[if d <= 2 { 0 } else if d <= 4 { 1 }
                    else if d <= 8 { 2 } else if d <= 16 { 3 }
                    else if d <= 32 { 4 } else { 5 }] += 1;
            }
            self.fin_last[k][ci] = cur;
            self.fin_valid[k][ci] = true;
        }
        for v in 0..NV {
            let pk = self.k[v][ply];
            let ck = step(pk, mc, SLACKS[v]);
            if ck != pk {
                self.rebuild[v][site] += 1;
            }
            if ply + 1 < MAX_PLY {
                self.k[v][ply + 1] = ck;
            }
        }
    }

    pub fn merge(&mut self, o: &Tally) {
        for s in 0..2 {
            self.edges[s] += o.edges[s];
            self.hard[s] += o.hard[s];
            self.m24[s] += o.m24[s];
            self.king[s] += o.king[s];
        }
        self.fin_n += o.fin_n;
        self.fin_cold += o.fin_cold;
        self.fin_sum += o.fin_sum;
        self.fin_pieces += o.fin_pieces;
        for i in 0..6 {
            self.fin_hist[i] += o.fin_hist[i];
        }
        for s in 0..2 {
            for v in 0..NV {
                self.rebuild[v][s] += o.rebuild[v][s];
            }
        }
    }

    pub fn report(&self) {
        let (em, eq) = (self.edges[0], self.edges[1]);
        let tot = (em + eq) as f64;
        if tot == 0.0 {
            return;
        }
        println!("\nhysteresis on total material  ({N} buckets, centres {LO}..{}, step {STEP})",
                 centre(N as i32 - 1));
        println!("  made moves: {em} main + {eq} q = {}", em + eq);
        println!("  {:<12} {:>10} {:>10} {:>10}", "rule", "main", "q", "all");
        let row = |name: String, m: u64, q: u64| {
            println!("  {:<12} {:>9.2}% {:>9.2}% {:>9.2}%", name,
                     100.0 * m as f64 / em.max(1) as f64,
                     100.0 * q as f64 / eq.max(1) as f64,
                     100.0 * (m + q) as f64 / tot);
        };
        row("hard (none)".to_string(), self.hard[0], self.hard[1]);
        row("[material24]".to_string(), self.m24[0], self.m24[1]);
        row("[king move]".to_string(), self.king[0], self.king[1]);
        for v in 0..NV {
            row(format!("slack {}", SLACKS[v]), self.rebuild[v][0], self.rebuild[v][1]);
        }
        if self.fin_n > 0 {
            let warm = self.fin_n - self.fin_cold;
            println!("  finny diff on {} king-move refreshes: cold {} ({:.1}%), \
                      warm mean {:.1} toggles vs {:.1} pieces (full rebuild), \
                      bins <=2/4/8/16/32/>32: {}/{}/{}/{}/{}/{}",
                     self.fin_n, self.fin_cold,
                     100.0 * self.fin_cold as f64 / self.fin_n as f64,
                     self.fin_sum as f64 / warm.max(1) as f64,
                     self.fin_pieces as f64 / self.fin_n as f64,
                     self.fin_hist[0], self.fin_hist[1], self.fin_hist[2],
                     self.fin_hist[3], self.fin_hist[4], self.fin_hist[5]);
        }
    }
}
