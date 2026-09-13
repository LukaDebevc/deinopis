//! The WDL net: a bucketed trunk with a three-way outcome head.
//!
//! This is the shape `nnue/wdlnet.py` trains as the `combo` arm — the one the
//! whole `runs/wdlnet` sweep converged on — evaluated in f32:
//!
//! ```text
//!   768 psq --ft--> width  (one table, two perspectives)  -> crelu, OR
//!                    the `gate` fold, which halves the width (FLAG_GATE)
//!   width   --down[king]--> d          each side, one shared table
//!   [d|d]   --mid[sym]--> 2d           no activation: a factored linear map
//!   2d      --up[mat]--> hidden        each side -> crelu
//!   [h|h]   --l2[mat]--> hidden        each side, same input -> crelu
//!   [h|h]   --head[mat]--> 3 logits    (L, D, W)
//!   [h|h]   --skip--> 3 logits         added
//!   768     --psqt--> 3 logits         ours minus theirs, W<->L swapped
//! ```
//!
//! Two things are different from `deepeval` and both are the point.
//!
//! **The output is a distribution, not a score.** LEDGER 048 measured that the
//! outcome is two-dimensional and a scalar throws away the larger axis: at
//! |score| < 20 the expected score moves 0.013 across material while the draw
//! rate moves 0.99 -> 0.39. The search still wants one number, so for now this
//! module collapses the distribution with the formula in `nnue/wdl.py`:
//!
//! ```text
//!   E = W + D/2,        cp = K * logit(E)
//! ```
//!
//! with the same `K` the scalar arms fit (`fit_k`, 288.5 cp), so the cp this
//! emits is on the scale every search constant was tuned against. That
//! deliberately discards the draw-rate axis. Reading `D` in the search — a
//! contempt or risk term — is the next question and belongs at the root with a
//! fixed sign, not in a leaf score both sides read; see `nnue/wdl.py`'s note on
//! why `(1-D)^q` is a search parameter scored by an SPRT and not a training
//! target.
//!
//! **Every layer is bucketed, and by a different family.** `down` is chosen by
//! each king, `mid` by a side-invariant material/pawn bucket, `up`, `l2` and
//! `head` by each side's material. The tables are folded at export, so the file
//! stores absolute per-bucket weights and this module never adds a base to a
//! delta. The bucket arithmetic here is a reimplementation of
//! `wdlnet.king_sqs` / `wdlnet.mat_buckets` / `wdl.b_sym`, which is exactly the
//! kind of thing that fails silently, so `chess evalfen --wdl` prints the five
//! indices and `nnue/verify_wdl.py` checks every one of them.
//!
//! Everything is f32, for the reason `deepeval` says: in floats a verifier
//! mismatch is a bug rather than a rounding budget.

use crate::board::Board;
use crate::eval::{Evaluator, Score};
use crate::qeval::{QuadNet, MAX_FEAT};
use crate::types::{Color, PieceType};

const MAGIC: u32 = 0x5155_4144; // "QUAD", shared with the other net files
/// 8 is all-f32; 9 additionally stores the five bucketed layers as i8 with a
/// per-(bucket, output column) f32 scale. Both are read here, because the f32
/// file is the reference an i8 file is checked against and losing the ability
/// to load it would lose the control arm.
const VERSION_F32: u32 = 8;
const VERSION_Q8: u32 = 9;
/// Dual tails: one shared accumulator, two int8 tails (big, then small).
/// Written by `nnue/export_dual.py`; without `--dual` the file behaves
/// exactly like its big tail alone.
const VERSION_DUAL_Q8: u32 = 10;
const NFEAT: usize = 768;
const NROWS: usize = NFEAT + 1; // + the frozen padding row the trainer keeps

/// Bucket families, as written in the file. `NONE` is a single bucket and is
/// how an unbucketed layer is spelled, so there is no second code path for it.
const FAM_NONE: u32 = 0;
const FAM_KING: u32 = 1;
const FAM_MAT: u32 = 2;
const FAM_SYM: u32 = 3;

const FLAG_SKIP: u32 = 1;
const FLAG_PSQT: u32 = 2;
/// `bn_act`: a crelu at the waist, between `down` and `mid`. Off in `combo`.
const FLAG_BN_ACT: u32 = 4;
/// The accumulator activation is the gated fold rather than a crelu: split the
/// `width` accumulator in half and return `clamp(a, -1, 1) * clamp(b, 0, 1)`,
/// so `down` reads `width / 2`. This is `wdlnet.WdlNet._act`'s `gate`, and it
/// is a flag rather than an implied thing because the same `width` in the
/// header then means two different `down` shapes -- a net exported one way and
/// read the other loads cleanly and evaluates nonsense. LEDGER 051 measured
/// the fold at 0.0060 of val loss, the largest architecture effect in the
/// sweep.
const FLAG_GATE: u32 = 8;
/// The bucketed layers are i8. Set by a version-9 file and never on its own —
/// it is in the flags so that `apply` can branch on one word rather than
/// threading the version through, and so a mixed file is impossible to spell.
const FLAG_Q8: u32 = 16;

// --------------------------------------------------------------- linear algebra
//
// Same lesson as `deepeval`: the inner loop has to have a trip count LLVM can
// see, or it spills the destination to memory on every iteration of the outer
// loop. Every weight matrix here is stored **input-major** `[din][dout]`, so
// the inner loop writes `dout` independent accumulators and vectorises, where
// the row-major dot-product order is a serial f32 reduction.

/// The `n` rows of `w`, `dout` wide, as an iterator with no per-row bounds
/// check.
///
/// This is not a tidiness change. Written as `&w[j * dout..j * dout + dout]`
/// the slice index is a *conditional panic inside the loop*, and LLVM has to
/// keep the destination coherent at every possible exit — so `z[k] += ...`
/// compiled to `vaddps (%rbx),%ymm0,%ymm0` / `vmovups %ymm0,(%rbx)` on every
/// single iteration, a store-to-load forward per multiply-add instead of an
/// accumulator living in a register. `chunks_exact` does the one bounds check
/// up front, the loop body cannot exit, and the accumulators stay in ymm.
/// Measured at 3.7x on `down` (`chess evalprof`).
#[inline(always)]
fn rows<T>(w: &[T], n: usize, dout: usize) -> std::slice::ChunksExact<'_, T> {
    w[..n * dout].chunks_exact(dout)
}

macro_rules! matvec {
    ($n:expr, $skip:literal, $($a:expr),* $(,)?) => {
        match $n {
            3 => matvec_n::<3, $skip>($($a),*),
            8 => matvec_n::<8, $skip>($($a),*),
            // 12: the b12 distill tail's waist (rung 3). Without this arm it
            // falls into `matvec_dyn` — the bounds-check trap LEDGER 055
            // removed from the 8/16/32 paths — and the small tail evals
            // slower than the big one it is meant to undercut.
            12 => matvec_n::<12, $skip>($($a),*),
            16 => matvec_n::<16, $skip>($($a),*),
            32 => matvec_n::<32, $skip>($($a),*),
            64 => matvec_n::<64, $skip>($($a),*),
            // 128 and 256 would need `U * DO` floats of partials, which is
            // stack rather than registers; the plain loop is better there.
            n @ (128 | 256) => matvec_dyn::<$skip>(n, $($a),*),
            n => matvec_dyn::<$skip>(n, $($a),*),
        }
    };
}

/// `z[k] += sum_j input[j] * w[j*DO + k]`.
///
/// `SKIP` drops zero inputs. It pays where the input is a narrow crelu'd vector
/// with real zeros in it and does not where the input is the accumulator —
/// the same split `deepeval` measured, so the call sites here choose the same
/// way rather than guessing.
/// How many partial accumulators the matvecs carry.
///
/// One accumulator makes `z[k] += x * w[j][k]` a chain of `din` dependent
/// adds, and a narrow layer has too few output lanes to fill the pipeline with
/// anything else: `down` is 256 inputs into 16 outputs, so two ymm registers
/// carry 256 serial adds each and the cost tracks the *latency* of the add,
/// not the number of them. The proof is that switching that loop to FMA — half
/// the instructions, one cycle more latency — made it 28% slower, against 33%
/// predicted from 4 cycles over 3. Four partials cut the chain to `din / 4`
/// and let FMA be the win it should be everywhere.
const U: usize = 4;

/// `a * b + c`, fused where the target has FMA.
///
/// `f32::mul_add` promises a single rounding, so on a target without FMA it
/// cannot be a multiply and an add: it becomes a call into libm per element,
/// and the net ran 16x slower on an x86-64-v2 build (49k vs 528k nps, same
/// nodes; library/016 A8). Native and v3 builds have FMA and compile exactly
/// as before.
#[inline(always)]
pub(crate) fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_feature = "fma")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_feature = "fma"))]
    {
        a * b + c
    }
}

/// One step of the unrolled matvec: `p[k] += x * row[k]`, `row` already sliced
/// to exactly `DO` so nothing in here can branch out of the loop.
#[inline(always)]
fn step<const DO: usize, const SKIP: bool>(x: f32, row: &[f32], p: &mut [f32; DO]) {
    if SKIP && x == 0.0 {
        return;
    }
    for k in 0..DO {
        p[k] = fmadd(x, row[k], p[k]);
    }
}

#[inline(always)]
fn stepq<const DO: usize, const SKIP: bool>(x: f32, row: &[i8], p: &mut [f32; DO]) {
    if SKIP && x == 0.0 {
        return;
    }
    for k in 0..DO {
        p[k] = fmadd(x, row[k] as f32, p[k]);
    }
}

/// `[[f32; DO]; U]` indexed by the unroll counter is a *stack* array — LLVM
/// will not promote it to registers, and doing it that way measured 7x slower
/// than one accumulator. Four separately-named ones is the whole trick.
macro_rules! unrolled {
    ($step:ident, $ty:ty, $input:expr, $w:expr, $z:expr, $DO:expr, $SKIP:expr) => {{
        let (input, w, z): (&[f32], &[$ty], &mut [f32]) = ($input, $w, &mut $z[..$DO]);
        let n = input.len();
        let full = n - n % U;
        let (mut p0, mut p1) = ([0f32; $DO], [0f32; $DO]);
        let (mut p2, mut p3) = ([0f32; $DO], [0f32; $DO]);
        for (xs, g) in input[..full]
            .chunks_exact(U)
            .zip(w[..full * $DO].chunks_exact(U * $DO))
        {
            let (g0, r) = g.split_at($DO);
            let (g1, r) = r.split_at($DO);
            let (g2, g3) = r.split_at($DO);
            $step::<$DO, $SKIP>(xs[0], g0, &mut p0);
            $step::<$DO, $SKIP>(xs[1], g1, &mut p1);
            $step::<$DO, $SKIP>(xs[2], g2, &mut p2);
            $step::<$DO, $SKIP>(xs[3], g3, &mut p3);
        }
        for (&x, row) in input[full..].iter().zip(rows(&w[full * $DO..], n - full, $DO)) {
            $step::<$DO, $SKIP>(x, row, &mut p0);
        }
        for k in 0..$DO {
            z[k] += (p0[k] + p1[k]) + (p2[k] + p3[k]);
        }
    }};
}

#[inline(always)]
fn matvec_n<const DO: usize, const SKIP: bool>(input: &[f32], w: &[f32], z: &mut [f32]) {
    unrolled!(step, f32, input, w, z, DO, SKIP)
}

#[inline(always)]
fn matvec_dyn<const SKIP: bool>(n: usize, input: &[f32], w: &[f32], z: &mut [f32]) {
    let z = &mut z[..n];
    for (&x, row) in input.iter().zip(rows(w, input.len(), n)) {
        if SKIP && x == 0.0 {
            continue;
        }
        for (zk, &wk) in z.iter_mut().zip(row) {
            *zk = fmadd(x, wk, *zk);
        }
    }
}

macro_rules! matvecq {
    ($n:expr, $skip:literal, $($a:expr),* $(,)?) => {
        match $n {
            3 => matvecq_n::<3, $skip>($($a),*),
            8 => matvecq_n::<8, $skip>($($a),*),
            // 12: see the same arm in `matvec!` — the b12 tail's waist.
            12 => matvecq_n::<12, $skip>($($a),*),
            16 => matvecq_n::<16, $skip>($($a),*),
            32 => matvecq_n::<32, $skip>($($a),*),
            64 => matvecq_n::<64, $skip>($($a),*),
            // 128 and 256 would need `U * DO` floats of partials, which is
            // stack rather than registers; the plain loop is better there.
            n @ (128 | 256) => matvecq_dyn::<$skip>(n, $($a),*),
            n => matvecq_dyn::<$skip>(n, $($a),*),
        }
    };
}

/// `z[k] += sum_j input[j] * w[j*DO + k]`, with `w` int8 and the scale applied
/// afterwards by the caller.
///
/// The activations stay f32 and so does the accumulation, which is the whole
/// design: what is being bought is the 4x cut in bytes read (LEDGER 040
/// measured the arithmetic speed-up to be invisible under real memory
/// pressure and the footprint cut not to be), and keeping f32 here means
/// there is no overflow to reason about and no activation scale to calibrate.
/// The widening `as f32` is one `vpmovsxbd`/`vcvtdq2ps` pair per vector on a
/// loop whose trip count LLVM can see — the same reason the f32 kernel is
/// monomorphised on `DO`.
#[inline(always)]
fn matvecq_n<const DO: usize, const SKIP: bool>(input: &[f32], w: &[i8], z: &mut [f32]) {
    unrolled!(stepq, i8, input, w, z, DO, SKIP)
}

#[inline(always)]
fn matvecq_dyn<const SKIP: bool>(n: usize, input: &[f32], w: &[i8], z: &mut [f32]) {
    let z = &mut z[..n];
    for (&x, row) in input.iter().zip(rows(w, input.len(), n)) {
        if SKIP && x == 0.0 {
            continue;
        }
        for (zk, &wk) in z.iter_mut().zip(row) {
            *zk = fmadd(x, wk as f32, *zk);
        }
    }
}

macro_rules! addsub {
    ($n:expr, $call:ident, $dynamic:ident, $($a:expr),* $(,)?) => {
        match $n {
            64 => $call::<64>($($a),*),
            128 => $call::<128>($($a),*),
            256 => $call::<256>($($a),*),
            512 => $call::<512>($($a),*),
            1024 => $call::<1024>($($a),*),
            n => $dynamic(n, $($a),*),
        }
    };
}

/// The accumulator scale: `ft` is quantised to i16 at load as
/// `round(ft * ACC_SCALE)`, and the gate dequantises by `ACC_INV_SCALE`.
/// Measured on the deploy net over 200k tune FENs: max|acc| = 13.05, so 1024
/// carries 2.5x headroom (512 would be 4.9x; the error halves at 1024 and
/// nothing else changes, so take the accuracy).
const ACC_SCALE: f32 = 1024.0;
const ACC_INV_SCALE: f32 = 1.0 / 1024.0;

/// Wrapping integer row updates. Wrapping (not saturating) keeps the
/// incremental push and the from-scratch rebuild bit-exact in ALL cases:
/// modular addition does not depend on grouping, where f32 rounding does.
/// Values are only meaningful inside the measured envelope above, which
/// `CHESS_ACC_CHECK` still guards along with the stack bookkeeping.
#[inline(always)]
fn addq_n<const N: usize>(dst: &mut [i16], src: &[i16]) {
    let (d, s) = (&mut dst[..N], &src[..N]);
    for k in 0..N {
        d[k] = d[k].wrapping_add(s[k]);
    }
}

#[inline(always)]
fn addq_dyn(n: usize, dst: &mut [i16], src: &[i16]) {
    for k in 0..n {
        dst[k] = dst[k].wrapping_add(src[k]);
    }
}

#[inline(always)]
fn subq_n<const N: usize>(dst: &mut [i16], src: &[i16]) {
    let (d, s) = (&mut dst[..N], &src[..N]);
    for k in 0..N {
        d[k] = d[k].wrapping_sub(s[k]);
    }
}

#[inline(always)]
fn subq_dyn(n: usize, dst: &mut [i16], src: &[i16]) {
    for k in 0..n {
        dst[k] = dst[k].wrapping_sub(src[k]);
    }
}

// ------------------------------------------------------------------- buckets

/// The five bucket indices one position needs, from one pass over the
/// canonicalised feature list.
///
/// Read off the features rather than the board for the same reason
/// `QuadNet::bucket` is: it is then the same arithmetic the trainer does on the
/// same numbers, and `verify_wdl.py` can compare them directly.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Bks {
    /// `wdlnet.king_sqs`: our king's square, and theirs mirrored into our frame.
    pub king: (usize, usize),
    /// `wdlnet.mat_buckets`: pawns x minors x rooks x queen, 0..53, per side.
    pub mat: (usize, usize),
    /// `wdl.b_sym`: total material x total pawns, 0..39. Side-swap invariant,
    /// so both perspectives read the same row — that is the whole point of it.
    pub sym: usize,
}

impl Bks {
    pub fn of(f: &[u16; MAX_FEAT], n: usize) -> Bks {
        let mut king = [0usize; 2];
        let mut cnt = [[0usize; 6]; 2];
        for &x in f.iter().take(n) {
            let x = x as usize;
            let (side, pt, sq) = (x / 384, (x % 384) / 64, x % 64);
            cnt[side][pt] += 1;
            if pt == 5 {
                king[side] = sq;
            }
        }
        Self::from_counts(cnt, king)
    }

    /// Bucket indices from piece counts and king squares directly, without
    /// decoding a feature list. `logits_upto` counts these in its board pass;
    /// `of` decodes them from features for the `evalfen`/verifier path. The
    /// bucket arithmetic lives here exactly once.
    fn from_counts(cnt: [[usize; 6]; 2], king: [usize; 2]) -> Bks {
        let matb = |s: usize| {
            let (p, mi) = (cnt[s][0], cnt[s][1] + cnt[s][2]);
            let (r, q) = (cnt[s][3], cnt[s][4]);
            let pb = (p / 3).min(2);
            let mb = if mi == 0 { 0 } else if mi <= 2 { 1 } else { 2 };
            let rb = r.min(2);
            let qb = usize::from(q > 0);
            ((pb * 3 + mb) * 3 + rb) * 2 + qb
        };
        // 1/3/3/5/9/0, summed over BOTH sides, so a side swap leaves it alone.
        const VAL: [usize; 6] = [1, 3, 3, 5, 9, 0];
        let mut mat = 0;
        let mut pawns = 0;
        for s in 0..2 {
            for pt in 0..6 {
                mat += VAL[pt] * cnt[s][pt];
            }
            pawns += cnt[s][0];
        }
        // `torch.bucketize(v, edges)` with the default `right=False` returns
        // the count of edges strictly below v, i.e. `edges[i-1] < v <= edges[i]`.
        const MAT_EDGES: [usize; 7] = [16, 24, 32, 42, 52, 62, 72];
        const PAWN_EDGES: [usize; 4] = [3, 6, 9, 12];
        let mb = MAT_EDGES.iter().take_while(|&&t| mat > t).count();
        let pb = PAWN_EDGES.iter().take_while(|&&t| pawns > t).count();
        Bks {
            king: (king[0], king[1] ^ 56),
            mat: (matb(0), matb(1)),
            sym: mb * 5 + pb,
        }
    }

    /// `(our index, their index)` for one family. `sym` is invariant under a
    /// side swap by construction, so both sides read the same row.
    #[inline]
    fn pick(&self, fam: u32) -> (usize, usize) {
        match fam {
            FAM_KING => self.king,
            FAM_MAT => self.mat,
            FAM_SYM => (self.sym, self.sym),
            FAM_NONE => (0, 0),
            _ => (0, 0),
        }
    }
}

// ------------------------------------------------------------------ the file

/// One bucketed `din -> dout` layer, weights folded to absolute at export.
struct Layer {
    fam: u32,
    nb: usize,
    din: usize,
    dout: usize,
    /// `[nb][din][dout]` — input-major within a bucket. Empty in an i8 layer.
    w: Vec<f32>,
    /// `[nb][din][dout]`, the i8 form. Empty in an f32 layer; exactly one of
    /// `w` and `qw` is populated.
    qw: Vec<i8>,
    /// `[nb][dout]`, the scale each i8 column dequantises by.
    sc: Vec<f32>,
    /// `[nb][dout]`.
    b: Vec<f32>,
}

impl Layer {
    #[inline]
    fn apply(&self, bk: usize, input: &[f32], out: &mut [f32], skip: bool) {
        let bk = bk.min(self.nb - 1);
        let (dout, din) = (self.dout, self.din);
        if !self.qw.is_empty() {
            // Accumulate from zero, then scale, then add the bias — the bias
            // is in the layer's own units and is NOT quantised, so it cannot
            // ride along inside the integer sum.
            out[..dout].fill(0.0);
            let w = &self.qw[bk * din * dout..(bk + 1) * din * dout];
            if skip {
                matvecq!(dout, true, &input[..din], w, out);
            } else {
                matvecq!(dout, false, &input[..din], w, out);
            }
            let sc = &self.sc[bk * dout..bk * dout + dout];
            let b = &self.b[bk * dout..bk * dout + dout];
            for k in 0..dout {
                out[k] = out[k] * sc[k] + b[k];
            }
            return;
        }
        out[..dout].copy_from_slice(&self.b[bk * dout..bk * dout + dout]);
        let w = &self.w[bk * din * dout..(bk + 1) * din * dout];
        if skip {
            matvec!(dout, true, &input[..din], w, out);
        } else {
            matvec!(dout, false, &input[..din], w, out);
        }
    }
}

pub struct WdlNet {
    width: usize,
    hidden: usize,
    /// `cp = k_cp * logit(W + D/2)`. In the file so that a net trained against
    /// a different fitted K cannot load with the wrong scale.
    k_cp: f32,
    /// `[NROWS][width]`, quantised at load as `round(ft * ACC_SCALE)`.
    /// The file stays f32; the quantisation error (<= 1/2048 per entry) is a
    /// deliberate approximation priced by games (needs an SPRT), not a second
    /// file format. The push moves half the bytes it did as f32.
    ftq: Vec<i16>,
    ft_bias: Vec<f32>,
    /// The main-search tail. In a single-tail file this is the whole net.
    big: Tail,
    /// The quiescence tail behind `--dual`, sharing the accumulator above.
    /// Same width, hidden, K, families and shape flags — checked at load —
    /// so the only thing that differs per tier is the waist and the weights.
    small: Option<Tail>,
}

/// One tail: everything downstream of the shared accumulator.
struct Tail {
    d: usize,
    flags: u32,
    down: Layer,
    /// `None` when the arm has no middle stage (`bn_mid = ""`).
    mid: Option<Layer>,
    up: Layer,
    l2: Layer,
    head: Layer,
    /// `[2*hidden][3]`, input-major, no bias.
    skiph: Vec<f32>,
    /// `[NROWS][3]`.
    psqt: Vec<f32>,
}

struct Rdr<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rdr<'a> {
    fn u32(&mut self) -> Result<u32, String> {
        let e = self.p + 4;
        if e > self.b.len() {
            return Err("truncated".into());
        }
        let v = u32::from_le_bytes(self.b[self.p..e].try_into().unwrap());
        self.p = e;
        Ok(v)
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn i8vec(&mut self, n: usize) -> Result<Vec<i8>, String> {
        let e = self.p + n;
        if e > self.b.len() {
            return Err(format!("truncated: wanted {n} int8s"));
        }
        let v = self.b[self.p..e].iter().map(|&c| c as i8).collect();
        self.p = e;
        Ok(v)
    }
    fn vec(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let e = self.p + 4 * n;
        if e > self.b.len() {
            return Err(format!("truncated: wanted {n} floats"));
        }
        let v = self.b[self.p..e]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.p = e;
        Ok(v)
    }
    /// A bucketed layer: `[nb][din][dout]` weights then `[nb][dout]` biases.
    /// The exporter has already transposed each bucket to input-major and
    /// folded `base + delta`, so nothing here rearranges anything.
    fn layer(&mut self, fam: u32, nb: usize, din: usize, dout: usize) -> Result<Layer, String> {
        if nb == 0 {
            return Err("bucketed layer with 0 buckets".into());
        }
        let w = self.vec(nb * din * dout)?;
        let b = self.vec(nb * dout)?;
        Ok(Layer { fam, nb, din, dout, w, qw: Vec::new(), sc: Vec::new(), b })
    }

    /// The i8 form: `[nb][din][dout]` int8, then `[nb][dout]` scales, then
    /// `[nb][dout]` biases. Scales come before the biases because that is the
    /// order they are consumed in, and because a truncated file then fails on
    /// the scales rather than loading with plausible-looking garbage.
    fn layer_q(&mut self, fam: u32, nb: usize, din: usize, dout: usize) -> Result<Layer, String> {
        if nb == 0 {
            return Err("bucketed layer with 0 buckets".into());
        }
        let qw = self.i8vec(nb * din * dout)?;
        let sc = self.vec(nb * dout)?;
        let b = self.vec(nb * dout)?;
        if sc.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err("i8 layer has a non-positive or non-finite scale".into());
        }
        Ok(Layer { fam, nb, din, dout, w: Vec::new(), qw, sc, b })
    }

    /// One bucketed layer in whichever precision the file uses. The family
    /// id is checked here rather than trusted: an unknown id would
    /// otherwise read bucket 0 forever and evaluate plausibly.
    fn tail_layer(
        &mut self,
        q8: bool,
        fam: u32,
        nb: usize,
        din: usize,
        dout: usize,
    ) -> Result<Layer, String> {
        if fam > FAM_SYM {
            return Err(format!("unknown bucket family {fam}"));
        }
        if q8 {
            self.layer_q(fam, nb, din, dout)
        } else {
            self.layer(fam, nb, din, dout)
        }
    }
}

/// Bits that change shapes or the code path, so two tails sharing one
/// accumulator must agree on all of them. SKIP/PSQT only add a term and
/// may differ per tail.
const SHAPE_FLAGS: u32 = FLAG_BN_ACT | FLAG_GATE;

/// One tail header of a version-10 file: waist, flags, families.
fn tail_head(r: &mut Rdr) -> Result<(usize, u32, [(u32, usize); 5]), String> {
    let d = r.u32()? as usize;
    let flags = r.u32()?;
    if d == 0 {
        return Err("dual tail with d = 0".into());
    }
    let mut fam = [(0u32, 0usize); 5];
    for slot in fam.iter_mut() {
        slot.0 = r.u32()?;
        slot.1 = r.u32()? as usize;
    }
    Ok((d, flags, fam))
}

/// One full tail: the layers, skip and psqt behind a header read by
/// `tail_head`. Both tails of a dual file parse through here, so a skew
/// between them is a shape error, not a silent offset.
fn tail_body(
    r: &mut Rdr,
    wpost: usize,
    hidden: usize,
    d: usize,
    flags: u32,
    fam: [(u32, usize); 5],
) -> Result<Tail, String> {
    let down = r.tail_layer(true, fam[0].0, fam[0].1, wpost, d)?;
    let mid = if fam[1].1 > 0 {
        Some(r.tail_layer(true, fam[1].0, fam[1].1, 2 * d, 2 * d)?)
    } else {
        None
    };
    let up = r.tail_layer(true, fam[2].0, fam[2].1, 2 * d, hidden)?;
    let l2 = r.tail_layer(true, fam[3].0, fam[3].1, 2 * hidden, hidden)?;
    let head = r.tail_layer(true, fam[4].0, fam[4].1, 2 * hidden, 3)?;
    let skiph = if flags & FLAG_SKIP != 0 {
        r.vec(2 * hidden * 3)?
    } else {
        Vec::new()
    };
    let psqt = if flags & FLAG_PSQT != 0 {
        r.vec(NROWS * 3)?
    } else {
        Vec::new()
    };
    Ok(Tail { d, flags, down, mid, up, l2, head, skiph, psqt })
}

impl WdlNet {
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Self::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Rdr { b: bytes, p: 0 };
        if r.u32()? != MAGIC {
            return Err("not a chess net".into());
        }
        let ver = r.u32()?;
        if ver != VERSION_F32 && ver != VERSION_Q8 && ver != VERSION_DUAL_Q8 {
            return Err(format!(
                "version {ver}, expected {VERSION_F32}, {VERSION_Q8} or {VERSION_DUAL_Q8}"
            ));
        }
        if ver == VERSION_DUAL_Q8 {
            return Self::parse_dual(&mut r, bytes);
        }
        let q8 = ver == VERSION_Q8;
        let width = r.u32()? as usize;
        let d = r.u32()? as usize;
        let hidden = r.u32()? as usize;
        let flags = r.u32()?;
        let k_cp = r.f32()?;
        let mut fam = [(0u32, 0usize); 5];
        for slot in fam.iter_mut() {
            let f = r.u32()?;
            slot.0 = f;
            slot.1 = r.u32()? as usize;
        }
        if width == 0 || d == 0 || hidden == 0 {
            return Err(format!("degenerate dims {width}/{d}/{hidden}"));
        }
        if !(k_cp.is_finite() && k_cp > 0.0) {
            return Err(format!("k_cp {k_cp} is not a positive finite number"));
        }

        // What survives the accumulator activation, and therefore what
        // `down` reads. `wdlnet.WdlNet` calls this `wpost`.
        let gated = flags & FLAG_GATE != 0;
        if gated && width % 2 != 0 {
            return Err(format!("gated net with odd width {width}"));
        }
        let wpost = if gated { width / 2 } else { width };

        // `ft` and `psqt` stay f32 in a version-9 file. LEDGER 053 measured
        // int8 on the ft table at 4x the val-loss cost of every other layer
        // combined (its column ranges are set by a heavy tail, so the bulk of
        // it quantises at about five effective bits) — and its per-node read
        // is zero anyway, because the accumulator is incremental. Its win
        // would be L2 residency, which is a different question.
        let ft = r.vec(NROWS * width)?;
        let ft_bias = r.vec(width)?;
        let ftq = ft.iter().map(|&v| (v * ACC_SCALE).round().clamp(-32768.0, 32767.0) as i16).collect();
        let down = r.tail_layer(q8, fam[0].0, fam[0].1, wpost, d)?;
        let mid = if fam[1].1 > 0 {
            Some(r.tail_layer(q8, fam[1].0, fam[1].1, 2 * d, 2 * d)?)
        } else {
            None
        };
        let up = r.tail_layer(q8, fam[2].0, fam[2].1, 2 * d, hidden)?;
        let l2 = r.tail_layer(q8, fam[3].0, fam[3].1, 2 * hidden, hidden)?;
        let head = r.tail_layer(q8, fam[4].0, fam[4].1, 2 * hidden, 3)?;
        let skiph = if flags & FLAG_SKIP != 0 {
            r.vec(2 * hidden * 3)?
        } else {
            Vec::new()
        };
        let psqt = if flags & FLAG_PSQT != 0 {
            r.vec(NROWS * 3)?
        } else {
            Vec::new()
        };
        if r.p != bytes.len() {
            return Err(format!("{} trailing bytes", bytes.len() - r.p));
        }
        let flags = if q8 { flags | FLAG_Q8 } else { flags & !FLAG_Q8 };
        let big = Tail { d, flags, down, mid, up, l2, head, skiph, psqt };
        Ok(WdlNet { width, hidden, k_cp, ftq, ft_bias, big, small: None })
    }

    /// Version 10: shared accumulator width / hidden / K, shared tables,
    /// then the big tail and the small tail through the same reader.
    fn parse_dual(r: &mut Rdr, bytes: &[u8]) -> Result<Self, String> {
        let width = r.u32()? as usize;
        let hidden = r.u32()? as usize;
        let k_cp = r.f32()?;
        if width == 0 || hidden == 0 {
            return Err(format!("degenerate dims {width}/{hidden}"));
        }
        if !(k_cp.is_finite() && k_cp > 0.0) {
            return Err(format!("k_cp {k_cp} is not a positive finite number"));
        }
        // `ft` and `psqt` stay f32, for the reason the v8/v9 path says:
        // the accumulator is incremental, so the table's per-node read is
        // zero and quantising it buys L2 residency at 4x the val-loss cost
        // of every other layer combined.
        let ft = r.vec(NROWS * width)?;
        let ft_bias = r.vec(width)?;
        let ftq = ft.iter().map(|&v| (v * ACC_SCALE).round().clamp(-32768.0, 32767.0) as i16).collect();
        // The gate bit fixes what `down` reads, so it is read off the big
        // tail's header before either tail's layers parse — and then checked
        // against the small tail's, since they share the one accumulator.
        let (bd, bf, bfam) = tail_head(r)?;
        let gated = bf & FLAG_GATE != 0;
        if gated && width % 2 != 0 {
            return Err(format!("gated net with odd width {width}"));
        }
        let wpost = if gated { width / 2 } else { width };
        let big = tail_body(r, wpost, hidden, bd, bf | FLAG_Q8, bfam)?;
        let (sd, sf, sfam) = tail_head(r)?;
        let small = tail_body(r, wpost, hidden, sd, sf | FLAG_Q8, sfam)?;
        if sfam != bfam {
            return Err("dual tails read different bucket families".into());
        }
        if (sf ^ bf) & SHAPE_FLAGS != 0 {
            return Err("dual tails disagree on shape flags (gate/waist act)".into());
        }
        if r.p != bytes.len() {
            return Err(format!("{} trailing bytes", bytes.len() - r.p));
        }
        Ok(WdlNet { width, hidden, k_cp, ftq, ft_bias, big, small: Some(small) })
    }

    /// The two accumulator rows one piece contributes: what White's chair sees
    /// and what Black's. Identical to `DeepNet::rows`; kept local so the two
    /// nets stay independently readable.
    #[inline]
    fn rows(c: usize, pt: usize, sq: usize) -> (usize, usize) {
        (
            (if c == 0 { 0 } else { 384 }) + pt * 64 + sq,
            (if c == 1 { 0 } else { 384 }) + pt * 64 + (sq ^ 56),
        )
    }

    /// Fill a `2 * width` accumulator, keyed by **perspective colour**: first
    /// half is White's chair, second is Black's. Keying by colour rather than
    /// by side to move is what makes a null move free.
    fn fill_colour(&self, b: &Board, acc: &mut [i16]) {
        let w = self.width;
        acc[..2 * w].fill(0);
        for c in Color::ALL {
            for pt in PieceType::ALL {
                for sq in b.colored(c, pt) {
                    let (wi, bi) = Self::rows(c.index(), pt.index(), sq.index());
                    let (lo, hi) = acc.split_at_mut(w);
                    addsub!(w, addq_n, addq_dyn, lo, &self.ftq[wi * w..]);
                    addsub!(w, addq_n, addq_dyn, hi, &self.ftq[bi * w..]);
                }
            }
        }
    }

    /// The three logits `(L, D, W)`, given the two accumulator halves already
    /// ordered for the side to move.
    fn logits(&self, tail: &Tail, b: &Board, own_pieces: &[i16], opp_pieces: &[i16], s: &mut Scratch) -> [f32; 3] {
        self.logits_upto::<NSTAGE>(tail, b, own_pieces, opp_pieces, s)
    }

    /// `logits`, stopping after `stop` stages. `stop == NSTAGE` is the real
    /// thing and is what the search calls; a smaller `stop` runs a prefix of
    /// the same code and returns a meaningless score.
    ///
    /// This exists so `chess evalprof` can price the stages by *difference* of
    /// two prefixes rather than by timing them in isolation: a prefix runs the
    /// production code path, in the production order, with the production
    /// cache state, and the last prefix is the whole evaluation, so the stage
    /// costs sum to the total by construction instead of by hope.
    ///
    /// `STOP` is a const parameter, not an argument, so every `if STOP <= k`
    /// below folds at compile time and a prefix costs exactly the work in it —
    /// a runtime `stop` would put a branch in each stage and the profile would
    /// be measuring the instrument.
    #[inline(always)]
    fn logits_upto<const STOP: u8>(
        &self,
        tail: &Tail,
        b: &Board,
        own_pieces: &[i16],
        opp_pieces: &[i16],
        s: &mut Scratch,
    ) -> [f32; 3] {
        let (w, d, h) = (self.width, tail.d, self.hidden);
        let Scratch { acc, cu, ct, ta, tb, hv, zv } = s;
        let mut out = [0f32; 3];

        // ---------------------------------------------------------- 0: feat
        // The piece list, its flipped twins (for the psqt stage), and the
        // counts the bucket indices read — one pass over the board instead of
        // `QuadNet::features` plus two decode loops over the list. The list
        // itself is bit-identical to `QuadNet::features` (same loop order),
        // so the verifier path through `Bks::of` is unaffected.
        let mut f = [0u16; MAX_FEAT];
        let mut g = [0u16; MAX_FEAT];
        let mut cnt = [[0usize; 6]; 2];
        let mut king = [0usize; 2];
        let stm = b.stm();
        let flip = if stm == Color::White { 0 } else { 56 };
        let mut n = 0;
        for c in Color::ALL {
            let s = if c == stm { 0 } else { 1 };
            for pt in PieceType::ALL {
                let (pi, base) = (pt.index(), s * 384 + pt.index() * 64);
                for sq in b.colored(c, pt) {
                    let q = sq.index() ^ flip;
                    f[n] = (base + q) as u16;
                    g[n] = ((1 - s) * 384 + pt.index() * 64 + (q ^ 56)) as u16;
                    n += 1;
                    cnt[s][pi] += 1;
                    if pi == 5 {
                        king[s] = q;
                    }
                }
            }
        }
        let bk = Bks::from_counts(cnt, king);
        if STOP <= 1 {
            return sink(acc, hv, zv, (n + bk.king.0 + bk.mat.0 + bk.sym) as u64, out);
        }

        // ---------------------------------------------------------- 1: gate
        // The accumulator activation, both perspectives, in place.
        //
        // `crelu` writes `w` values per half. `gate` writes `w / 2`: it reads
        // entries `k` and `k + w/2` and folds them into `k`, which is safe in
        // place because every read is at or above the write. Either way each
        // half is left with its live values at the FRONT, so `down.apply` --
        // which slices `input[..din]` -- needs no further arrangement.
        //
        // Written with zipped iterators rather than indices for the reason in
        // `rows`: an indexed read of `ft_bias` is a conditional panic in the
        // loop body, and the whole thing then refuses to stay in registers.
        {
            let bias = &self.ft_bias[..w];
            if tail.flags & FLAG_GATE != 0 {
                // Fold straight from the two sources into the accumulator
                // front: the gate reads entries `k` and `k + w/2` and writes
                // `k`, so it never needs its input staged in `acc` first.
                // Skips two `w`-float copies per evaluation.
                let wp = w / 2;
                let (b0, b1) = bias.split_at(wp);
                for half in 0..2 {
                    let src = if half == 0 { own_pieces } else { opp_pieces };
                    let (s0, s1) = src.split_at(wp);
                    let lo = &mut acc[half * w..half * w + wp];
                    for ((((v, &a), &g), &c0), &c1) in
                        lo.iter_mut().zip(s0).zip(s1).zip(b0).zip(b1)
                    {
                        *v = (a as f32 * ACC_INV_SCALE + c0).clamp(-1.0, 1.0)
                            * (g as f32 * ACC_INV_SCALE + c1).clamp(0.0, 1.0);
                    }
                }
            } else {
                for half in 0..2 {
                    let src = if half == 0 { own_pieces } else { opp_pieces };
                    for (v, (&q, &c)) in acc[half * w..half * w + w]
                        .iter_mut()
                        .zip(src.iter().zip(bias))
                    {
                        *v = (q as f32 * ACC_INV_SCALE + c).clamp(0.0, 1.0);
                    }
                }
            }
        }
        if STOP <= 2 {
            return sink(acc, hv, zv, 0, out);
        }

        // ---------------------------------------------------------- 2: down
        // down: each side's king picks the projection. The input is the wide
        // crelu'd accumulator, where `deepeval` measured the zero-skip branch
        // does not pay.
        //
        // The two halves are then concatenated BEFORE the up projection, and
        // the their-side concat is reversed -- that is what keeps the us/them
        // symmetry exact rather than fitted, and it is why our hidden vector
        // sees their king even though the projection was chosen by one king.
        {
            let (au, at) = acc[..2 * w].split_at(w);
            let (di_u, di_t) = tail.down.fam_idx(&bk);
            tail.down.apply(di_u, au, &mut ta[..d], false);
            tail.down.apply(di_t, at, &mut tb[..d], false);
            if tail.flags & FLAG_BN_ACT != 0 {
                for k in 0..d {
                    ta[k] = ta[k].clamp(0.0, 1.0);
                    tb[k] = tb[k].clamp(0.0, 1.0);
                }
            }
            cu[..d].copy_from_slice(&ta[..d]);
            cu[d..2 * d].copy_from_slice(&tb[..d]);
            ct[..d].copy_from_slice(&tb[..d]);
            ct[d..2 * d].copy_from_slice(&ta[..d]);
        }
        if STOP <= 3 {
            return sink(cu, ct, zv, 0, out);
        }

        // ----------------------------------------------------------- 3: mid
        // mid: one further bucketed matrix with NO activation around it, so the
        // whole down-mid-up stack is still a single linear map per
        // (king, sym, material) cell -- a low-rank factorisation of a table far
        // too big to store, not a deeper network.
        if let Some(m) = &tail.mid {
            let (mu, mt) = m.fam_idx(&bk);
            // `skip = false`: there is no activation between `down` and
            // `mid`, so `cu`/`ct` have no structural zeros to skip and the
            // test is pure overhead. Same for `up`, which reads the same pair.
            m.apply(mu, &cu[..2 * d], &mut ta[..2 * d], false);
            m.apply(mt, &ct[..2 * d], &mut tb[..2 * d], false);
            cu[..2 * d].copy_from_slice(&ta[..2 * d]);
            ct[..2 * d].copy_from_slice(&tb[..2 * d]);
        }
        if STOP <= 4 {
            return sink(cu, ct, zv, 0, out);
        }

        // ------------------------------------------------------------ 4: up
        {
            let (uu, ut) = tail.up.fam_idx(&bk);
            let (lo, hi) = hv[..2 * h].split_at_mut(h);
            tail.up.apply(uu, &cu[..2 * d], lo, false);
            tail.up.apply(ut, &ct[..2 * d], hi, false);
        }
        for v in hv[..2 * h].iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
        if STOP <= 5 {
            return sink(acc, hv, zv, 0, out);
        }

        // ------------------------------------------------------------ 5: l2
        // l2: both sides read the SAME h and differ only in the bucket.
        {
            let (l2u, l2t) = tail.l2.fam_idx(&bk);
            let (lo, hi) = zv[..2 * h].split_at_mut(h);
            tail.l2.apply(l2u, &hv[..2 * h], lo, false);
            tail.l2.apply(l2t, &hv[..2 * h], hi, false);
        }
        for v in zv[..2 * h].iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
        if STOP <= 6 {
            return sink(acc, hv, zv, 0, out);
        }

        // ---------------------------------------------------------- 6: head
        tail.head.apply(tail.head.fam_idx(&bk).0, &zv[..2 * h], &mut out[..], false);
        if tail.flags & FLAG_SKIP != 0 {
            matvec!(3usize, false, &hv[..2 * h], &tail.skiph, &mut out[..]);
        }
        if STOP <= 7 {
            return sink(acc, hv, zv, 0, out);
        }

        // ---------------------------------------------------------- 7: psqt
        // The psqt passthrough is antisymmetric by construction: a side swap
        // exchanges W and L and leaves D alone, so their contribution enters
        // reversed. `own`/`opp` are the same feature set from the two chairs.
        if tail.flags & FLAG_PSQT != 0 {
            let mut pu = [0f32; 3];
            let mut pt_ = [0f32; 3];
            for (&x, &y) in f.iter().zip(g.iter()).take(n) {
                let (i, j) = (x as usize, y as usize);
                for k in 0..3 {
                    pu[k] += tail.psqt[i * 3 + k];
                    pt_[k] += tail.psqt[j * 3 + k];
                }
            }
            for k in 0..3 {
                out[k] += pu[k] - pt_[2 - k];
            }
        }
        out
    }

    /// Time every prefix of `logits_upto` over `boards`, `reps` passes each.
    /// Returns seconds per evaluation for prefix `1..=NSTAGE`; stage `i` costs
    /// `t[i] - t[i-1]`, and `t[NSTAGE-1]` is the whole evaluation, so the parts
    /// sum to the whole by construction.
    ///
    /// The accumulators are built once, up front, and read from a flat buffer.
    /// In the engine they come off a per-ply stack that is hot in L1; here a
    /// large corpus pushes them to L2. That biases `gate` (which reads the
    /// accumulator) and `down` (which reads it again) upward and nothing else,
    /// which is why `--positions` defaults small enough to stay in L2 and why
    /// `CHESS_WDL_ABL` exists to cross-check those two in a real search.
    pub fn profile(&self, boards: &[Board], reps: usize) -> [f64; NSTAGE as usize] {
        let w = self.width;
        let mut accs = vec![0i16; boards.len() * 2 * w];
        for (i, b) in boards.iter().enumerate() {
            self.fill_colour(b, &mut accs[i * 2 * w..(i + 1) * 2 * w]);
        }
        let mut s = Scratch::new(w, self.big.d, self.hidden);
        let mut t = [0f64; NSTAGE as usize];
        macro_rules! pass {
            ($k:expr) => {{
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    for (i, b) in boards.iter().enumerate() {
                        let (wh, bl) = accs[i * 2 * w..(i + 1) * 2 * w].split_at(w);
                        let (own, opp) =
                            if b.stm() == Color::White { (wh, bl) } else { (bl, wh) };
                        std::hint::black_box(self.logits_upto::<$k>(&self.big, b, own, opp, &mut s));
                    }
                }
                t0.elapsed().as_secs_f64() / (reps * boards.len()) as f64
            }};
        }
        // One untimed pass so the first prefix does not pay for cold weights.
        let _ = pass!(8);
        t[0] = pass!(1);
        t[1] = pass!(2);
        t[2] = pass!(3);
        t[3] = pass!(4);
        t[4] = pass!(5);
        t[5] = pass!(6);
        t[6] = pass!(7);
        t[7] = pass!(8);
        t
    }

    /// Multiply-adds each stage performs on this net, at `nfeat` features.
    /// `gate` does no multiply-adds; its cost is `width` elementwise ops and it
    /// is reported as zero here on purpose, so a nonzero time against zero MACs
    /// reads as "this stage is not arithmetic".
    pub fn stage_macs(&self, nfeat: usize) -> [u64; NSTAGE as usize] {
        let lm = |l: &Layer| 2 * (l.din * l.dout) as u64;
        let mut m = [0u64; NSTAGE as usize];
        m[2] = lm(&self.big.down);
        m[3] = self.big.mid.as_ref().map_or(0, lm);
        m[4] = lm(&self.big.up);
        m[5] = lm(&self.big.l2);
        m[6] = (self.big.head.din * self.big.head.dout) as u64
            + if self.big.flags & FLAG_SKIP != 0 { 2 * self.hidden as u64 * 3 } else { 0 };
        m[7] = if self.big.flags & FLAG_PSQT != 0 { nfeat as u64 * 3 * 2 } else { 0 };
        m
    }

    /// How many features the corpus averages, which is what `stage_macs` needs
    /// to price `psqt`.
    pub fn mean_feats(boards: &[Board]) -> f64 {
        let mut f = [0u16; MAX_FEAT];
        let n: usize = boards.iter().map(|b| QuadNet::features(b, &mut f)).sum();
        n as f64 / boards.len().max(1) as f64
    }

    /// `cp = K * logit(W + v*D)`, the readout in `nnue/wdl.py`, with `v` what
    /// a draw is worth to the side to move (`eval::draw_value_for`).
    ///
    /// At the default `v = 0.5` this is `W + D/2` and is bit-identical to what
    /// it always computed — `0.5 * dr` and `dr / 2.0` are the same IEEE
    /// operation — so the torch verifier still matches to 5.5e-6.
    ///
    /// Splitting the draw mass unevenly between the two wings is the whole
    /// mechanism: a net that says "40% win, 55% draw, 5% loss" scores +112 cp
    /// at `v = 0.5` and **-27 cp** at `v = 0`, because the draw has stopped
    /// counting for us and started counting for them. That is available here
    /// only because the eval is a distribution. A scalar eval cannot do it:
    /// there is nothing in one number that says which positions are drawish.
    ///
    /// Written as a ratio of two positive sums rather than as `logit` of a
    /// normalised probability: the softmax denominator cancels, so there is no
    /// catastrophic cancellation near a decided position and no clamp needed to
    /// keep `E` off 0 and 1.
    pub fn cp_from_logits(&self, lg: [f32; 3], v: f32) -> Score {
        let m = lg[0].max(lg[1]).max(lg[2]) as f64;
        let e = |x: f32| ((x as f64) - m).exp();
        let (l, dr, w) = (e(lg[0]), e(lg[1]), e(lg[2]));
        let v = v as f64;
        let cp = self.k_cp as f64 * ((w + v * dr) / (l + (1.0 - v) * dr)).ln();
        cp.round().clamp(-30_000.0, 30_000.0) as Score
    }

    /// `cp_from_logits` with asymmetric contempt: the additive bonus
    /// `b * weight * tanh(a * gap)` (`eval::contempt_bonus`), stm-relative
    /// so it negates across a ply with no root tracking. `weight` is the
    /// draw mass, or identically 1 when flat is on.
    ///
    /// The base collapse stays at the fixed symmetric `v = 0.5` and the bonus
    /// is bounded by `b`: cross-parity comparisons keep the coherent base's
    /// ordering, the nudge only tips close calls. A state-dependent `v` was
    /// tried instead and is globally incoherent — in drawish territory a 2%
    /// position reads +418cp while its 18% neighbour reads −418, because
    /// min-loss and max-win are different objectives that do not compare
    /// across the boundary.
    ///
    /// With `b = 0` this is the exact old path, float for float — bench and
    /// `verify_wdl.py` cannot tell them apart. `cp_from_logits` above keeps
    /// its signature for the torch verifier, which always collapses at an
    /// explicit `v`.
    pub fn cp_from_logits_asym(&self, lg: [f32; 3], root_stm: bool) -> Score {
        let (ca, cb) = (crate::eval::contempt_a(), crate::eval::contempt_b());
        // Side-to-move-relative symmetric collapse, exactly what
        // `draw_value_for` yields — so a pure `--draw-value` arm behaves as
        // it always did, with or without the bonus on top.
        let sym = crate::eval::draw_value();
        let v = if root_stm { sym } else { 1.0 - sym };
        if cb == 0.0 {
            return self.cp_from_logits(lg, v);
        }
        let m = lg[0].max(lg[1]).max(lg[2]) as f64;
        let e = |x: f32| ((x as f64) - m).exp();
        let (l, dr, w) = (e(lg[0]), e(lg[1]), e(lg[2]));
        let v = v as f64;
        let mut cp = self.k_cp as f64 * ((w + v * dr) / (l + (1.0 - v) * dr)).ln();
        // The gap straight from the raw logits: the shared additive shift
        // cancels in the difference, so no exp/ln is needed for it. `D`
        // does need the normalised mass.
        let gap = (lg[0] - lg[2]) as f64;
        let weight = if crate::eval::contempt_flat() { 1.0 } else { dr / (l + dr + w) };
        cp += crate::eval::contempt_bonus(weight, gap, ca, cb);
        cp.round().clamp(-30_000.0, 30_000.0) as Score
    }

    /// `eval` in centipawns from the side to move's point of view, rebuilt from
    /// the board. This is what `evalfen` and `nnue/verify_wdl.py` call; the
    /// search uses `WdlEval`, which reaches the same `logits` with an
    /// incrementally maintained accumulator.
    pub fn evaluate(&self, b: &Board) -> Score {
        self.cp_from_logits_asym(self.raw(b), crate::eval::stm_is_root(b.stm()))
    }

    /// The three logits, rebuilt from scratch. Exposed so `evalfen` can print
    /// them and the verifier can compare the distribution, not only the cp it
    /// collapses to — a sign error in `_swap` moves W and L and can leave the
    /// rounded cp untouched.
    pub fn raw(&self, b: &Board) -> [f32; 3] {
        self.raw_tail(&self.big, b)
    }

    /// The small tail's logits, for `evalfen --small` and the verifier. The
    /// accumulator is shared, so this is the same code reaching the same
    /// rows the search's quiescence tier reads.
    pub fn raw_small(&self, b: &Board) -> [f32; 3] {
        self.raw_tail(self.small.as_ref().expect("no small tail in this file"), b)
    }

    fn raw_tail(&self, tail: &Tail, b: &Board) -> [f32; 3] {
        let w = self.width;
        let mut s = Scratch::new(w, tail.d, self.hidden);
        let mut acc = vec![0i16; 2 * w];
        self.fill_colour(b, &mut acc);
        let (wh, bl) = acc.split_at(w);
        let (own, opp) = if b.stm() == Color::White { (wh, bl) } else { (bl, wh) };
        self.logits(tail, b, own, opp, &mut s)
    }

    /// Whether this file carries a quiescence tail for `--dual`. Without one
    /// `--dual` keeps its old meaning (the quad net in quiescence).
    pub fn has_small(&self) -> bool {
        self.small.is_some()
    }

    pub fn small_dims(&self) -> Option<(usize, usize, usize)> {
        self.small.as_ref().map(|t| (self.width, t.d, self.hidden))
    }

    pub fn dims(&self) -> (usize, usize, usize) {
        (self.width, self.big.d, self.hidden)
    }
    pub fn k_cp(&self) -> f32 {
        self.k_cp
    }
}

impl Layer {
    #[inline]
    fn fam_idx(&self, bk: &Bks) -> (usize, usize) {
        bk.pick(self.fam)
    }
}

/// Per-thread working memory. Every buffer here would otherwise be a `vec!` in
/// the hottest loop in the program.
///
/// `cu`/`ct`/`ta`/`tb` used to be `[f32; 256]` stack arrays sized for the
/// widest bottleneck the sweep ever tried, which meant `combo` (d = 16) paid
/// to zero six kilobytes of stack per evaluation to use 128 bytes of it.
struct Scratch {
    acc: Vec<f32>,
    /// The us-side and them-side concatenations, `2 * d` each.
    cu: Vec<f32>,
    ct: Vec<f32>,
    /// Somewhere for a bucketed layer to write before it is copied back, since
    /// `mid` reads and writes the same pair.
    ta: Vec<f32>,
    tb: Vec<f32>,
    hv: Vec<f32>,
    zv: Vec<f32>,
}

impl Scratch {
    fn new(w: usize, d: usize, h: usize) -> Self {
        Scratch {
            acc: vec![0.0; 2 * w],
            cu: vec![0.0; 2 * d],
            ct: vec![0.0; 2 * d],
            ta: vec![0.0; 2 * d],
            tb: vec![0.0; 2 * d],
            hv: vec![0.0; 2 * h],
            zv: vec![0.0; 2 * h],
        }
    }
}

/// The stages `logits_upto` will stop after, in order. `evalprof` prices stage
/// `i` as the time of prefix `i + 1` minus the time of prefix `i`.
pub const PROF_STAGES: [&str; NSTAGE as usize] =
    ["feat", "gate", "down", "mid", "up", "l2", "head", "psqt"];
const NSTAGE: u8 = 8;

/// Stop a prefix without letting the optimiser delete the work behind it.
///
/// `black_box` on the three buffer *addresses* is enough: it is opaque to LLVM
/// and may read through them, so every store into them has to be real. Without
/// this a short prefix compiles to almost nothing and the profile is a lie.
#[inline(never)]
fn sink(a: &[f32], b: &[f32], c: &[f32], keep: u64, out: [f32; 3]) -> [f32; 3] {
    std::hint::black_box((a.as_ptr(), b.as_ptr(), c.as_ptr(), keep));
    std::hint::black_box(out)
}

/// The 12 piece bitboards, `colour * 6 + piece type`.
#[inline]
fn bbs(b: &Board) -> [u64; 12] {
    let mut o = [0u64; 12];
    for c in Color::ALL {
        for pt in PieceType::ALL {
            o[c.index() * 6 + pt.index()] = b.colored(c, pt).0;
        }
    }
    o
}

const STACK: usize = crate::search::MAX_PLY + 8;

/// The WDL net with an incremental accumulator.
///
/// Same mechanism as `DeepEval`, and deliberately so: `push` is told only the
/// new board, XORs the 12 piece bitboards against the parent's and walks the
/// set bits, so quiet moves, captures, castling, en passant and promotion are
/// one code path with no move-type switch to get wrong. `evaluate` checks the
/// stack against the board it was handed and rebuilds on a disagreement, so a
/// missing `push` costs speed rather than correctness.
pub struct WdlEval {
    net: &'static WdlNet,
    acc: Vec<i16>,
    bb: Vec<[u64; 12]>,
    ok: Vec<bool>,
    sp: usize,
    s: Scratch,
    resyncs: u64,
    evals: u64,
    /// `CHESS_ACC_CHECK=1`: recompute every eval from scratch and compare. Off,
    /// one predictable branch per node; on, the only thing that exercises the
    /// incremental path, since the verifier goes through `evaluate`.
    check: bool,
    /// `CHESS_ACC_OFF=1`: rebuild every eval, so both paths can be timed in the
    /// same binary with everything else held fixed.
    off: bool,
    mism: u64,
    worst: i32,
}

impl WdlEval {
    pub     fn new(net: &'static WdlNet) -> Self {
        let w = net.width;
        // One scratch for both tiers, sized by the wider waist — the small
        // tail only ever slices prefixes of it.
        let d = net.small.as_ref().map_or(net.big.d, |t| net.big.d.max(t.d));
        WdlEval {
            net,
            acc: vec![0; (STACK + 1) * 2 * w],
            bb: vec![[0u64; 12]; STACK + 1],
            ok: vec![false; STACK + 1],
            sp: 0,
            s: Scratch::new(w, d, net.hidden),
            resyncs: 0,
            evals: 0,
            check: std::env::var("CHESS_ACC_CHECK").is_ok(),
            off: std::env::var("CHESS_ACC_OFF").is_ok(),
            mism: 0,
            worst: 0,
        }
    }

    /// Whether the loaded file carries a quiescence tail, for the `--dual`
    /// tier switch in `qeval`.
    pub fn net_has_small(&self) -> bool {
        self.net.has_small()
    }

    /// The accumulator slot for this board: the pushed row when it matches,
    /// a from-scratch rebuild when it does not. Shared by both tiers — the
    /// small tail reads the same accumulator the big one maintains, which is
    /// the whole point of the dual file.
    fn slot(&mut self, b: &Board) -> usize {
        let (net, w) = (self.net, self.net.width);
        let nbb = bbs(b);
        if self.sp < STACK && self.ok[self.sp] && self.bb[self.sp] == nbb {
            return self.sp;
        }
        self.resyncs += 1;
        let slot = self.sp.min(STACK);
        let base = slot * 2 * w;
        net.fill_colour(b, &mut self.acc[base..base + 2 * w]);
        self.bb[slot] = nbb;
        self.ok[slot] = true;
        slot
    }

    /// The quiescence tier: the small tail on the live accumulator. The
    /// caller (`DefaultEval` behind `--dual`) guarantees the file has one.
    pub fn evaluate_small(&mut self, b: &Board) -> Score {
        let (net, w) = (self.net, self.net.width);
        self.evals += 1;
        let small = net.small.as_ref().expect("dual tier on a single-tail file");
        if self.off {
            return net.cp_from_logits_asym(net.raw_small(b), crate::eval::stm_is_root(b.stm()));
        }
        let slot = self.slot(b);
        let base = slot * 2 * w;
        let (wh, bl) = self.acc[base..base + 2 * w].split_at(w);
        let (own, opp) = if b.stm() == Color::White { (wh, bl) } else { (bl, wh) };
        let got = net.cp_from_logits_asym(net.logits(small, b, own, opp, &mut self.s),
                                             crate::eval::stm_is_root(b.stm()));
        if self.check {
            let want = net.cp_from_logits_asym(net.raw_small(b), crate::eval::stm_is_root(b.stm()));
            if got != want {
                self.mism += 1;
                self.worst = self.worst.max((got as i32 - want as i32).abs());
            }
        }
        got
    }
}

impl Clone for WdlEval {
    fn clone(&self) -> Self {
        WdlEval::new(self.net)
    }
}

impl Drop for WdlEval {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        RESYNCS.fetch_add(self.resyncs, Relaxed);
        EVALS.fetch_add(self.evals, Relaxed);
        MISM.fetch_add(self.mism, Relaxed);
        WORST.fetch_max(self.worst as u64, Relaxed);
    }
}

static RESYNCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static EVALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MISM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WORST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(rebuilds, evals, mismatches, worst cp)` summed over every `WdlEval`
/// dropped so far. The last two are zero unless `CHESS_ACC_CHECK` was set.
pub fn resync_stats() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (RESYNCS.load(Relaxed), EVALS.load(Relaxed), MISM.load(Relaxed), WORST.load(Relaxed))
}

impl Evaluator for WdlEval {
    fn evaluate(&mut self, b: &Board) -> Score {
        let (net, w) = (self.net, self.net.width);
        self.evals += 1;
        if self.off {
            return net.evaluate(b);
        }
        let slot = self.slot(b);
        let base = slot * 2 * w;
        let (wh, bl) = self.acc[base..base + 2 * w].split_at(w);
        let (own, opp) = if b.stm() == Color::White { (wh, bl) } else { (bl, wh) };
        let got = net.cp_from_logits_asym(net.logits(&net.big, b, own, opp, &mut self.s),
                                             crate::eval::stm_is_root(b.stm()));
        if self.check {
            let want = net.evaluate(b);
            if got != want {
                self.mism += 1;
                self.worst = self.worst.max((got as i32 - want as i32).abs());
            }
        }
        got
    }

    fn push(&mut self, nb: &Board) {
        let (net, w) = (self.net, self.net.width);
        let (from, to) = (self.sp, self.sp + 1);
        self.sp = to;
        if to >= STACK {
            return; // past the stack; `evaluate` rebuilds until we come back
        }
        let nbb = bbs(nb);
        self.bb[to] = nbb;
        self.ok[to] = true;
        if !self.ok[from] {
            self.resyncs += 1;
            let base = to * 2 * w;
            net.fill_colour(nb, &mut self.acc[base..base + 2 * w]);
            return;
        }
        let old = self.bb[from];
        let (lo, hi) = self.acc.split_at_mut(to * 2 * w);
        let next = &mut hi[..2 * w];
        next.copy_from_slice(&lo[from * 2 * w..from * 2 * w + 2 * w]);
        for i in 0..12 {
            let mut df = old[i] ^ nbb[i];
            if df == 0 {
                continue;
            }
            let (c, pt) = (i / 6, i % 6);
            while df != 0 {
                let sq = df.trailing_zeros() as usize;
                df &= df - 1;
                let (wi, bi) = WdlNet::rows(c, pt, sq);
                let (lo, hi) = next.split_at_mut(w);
                if nbb[i] >> sq & 1 == 1 {
                    addsub!(w, addq_n, addq_dyn, lo, &net.ftq[wi * w..]);
                    addsub!(w, addq_n, addq_dyn, hi, &net.ftq[bi * w..]);
                } else {
                    addsub!(w, subq_n, subq_dyn, lo, &net.ftq[wi * w..]);
                    addsub!(w, subq_n, subq_dyn, hi, &net.ftq[bi * w..]);
                }
            }
        }
    }

    fn pop(&mut self) {
        debug_assert!(self.sp > 0);
        self.sp -= 1;
    }
}

static NET: std::sync::OnceLock<Option<WdlNet>> = std::sync::OnceLock::new();

/// `--wdl <path>` or `$CHESS_WDL`. Same shape as `qeval::net` and
/// `deepeval::net` so all three are selected the same way; when more than one
/// is given this wins, since it is the strictly newer model.
pub fn net() -> Option<&'static WdlNet> {
    NET.get_or_init(|| {
        let args: Vec<String> = std::env::args().collect();
        let path = args
            .iter()
            .position(|a| a == "--wdl")
            .and_then(|i| args.get(i + 1))
            .cloned()
            .or_else(|| std::env::var("CHESS_WDL").ok())?;
        match WdlNet::load(&path) {
            Ok(n) => {
                let (w, d, h) = n.dims();
                let small = n
                    .small_dims()
                    .map_or(String::new(), |(_, ds, _)| format!(", small bneck {ds}"));
                eprintln!(
                    "info string wdl eval loaded from {path} (width {w}, bneck {d}, hidden {h}, K {:.1}{small})",
                    n.k_cp()
                );
                Some(n)
            }
            Err(e) => {
                eprintln!("info string wdl eval NOT loaded: {e}");
                None
            }
        }
    })
    .as_ref()
}
