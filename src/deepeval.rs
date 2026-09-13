//! The NNUE head, evaluated from scratch.
//!
//! `acc -> crelu -> Linear(2w, h) -> crelu -> bucketed h x h -> crelu ->
//! bucketed h -> 1`, plus a PSQT skip. This is the shape `nnue/train.py`'s
//! `Deep` trains and it beats the deployed quadratic form by 11.6% at a third
//! of the parameters (library/009 section 6).
//!
//! The piece half of the accumulator is incremental (`DeepEval`); the derived
//! pawn features are not. `pawnfile` is the furthest-advanced pawn per file and
//! has no add/sub delta, so an incremental version would be a second mechanism
//! with its own bugs for 15 of the 94 rows. From scratch they are eight
//! bit-scans and cannot be wrong in a way the verifier misses.
//!
//! `DeepNet::head` is shared between the from-scratch and incremental paths and
//! sums each perspective in the same order, so the two agree bit for bit and
//! `nnue/verify_deep.py` covers both.
//!
//! Everything is f32. Quantising a crelu net is a separate problem: the
//! accumulator scale, l1's input range and the clip bounds all have to agree,
//! and when they do not the eval stays plausible and every game is played
//! slightly wrong. In floats, a `verify.py` mismatch is a bug rather than
//! rounding.

use crate::board::Board;
use crate::eval::{Evaluator, Score};
use crate::qeval::{QuadNet, MAX_FEAT};
use crate::types::{Color, PieceType};
use crate::wdleval::fmadd;

// ------------------------------------------------------- constant-width loops
//
// `width` and `hidden` are read from the net file, so every inner loop in this
// module used to be `for k in 0..w` with a trip count LLVM could not see. That
// is expensive in a specific way: it cannot keep the destination in registers
// across the outer loop, so `l1` spilled 32 floats to memory on each of its 256
// iterations, and it emits a vector body guarded by a runtime length check that
// often loses to the scalar tail.
//
// Handing it the constant instead is worth **3.6x on l1** with bit-identical
// output -- the same operations in the same order, so every eval, every node
// count and every match result is unchanged. Explicit AVX2 intrinsics were
// measured too and were *slower* than this (235 ns against 219): LLVM already
// vectorises well once it knows the shape. See `library/010-eval-speed.md`.
//
// The dispatch covers the shapes the trainer actually emits. Anything else
// falls back to the old dynamic loop, which is correct and merely slow, so an
// experimental net with an odd width still runs.

macro_rules! dims {
    ($n:expr, $call:ident, $dynamic:ident, $($a:expr),* $(,)?) => {
        match $n {
            16 => $call::<16>($($a),*),
            32 => $call::<32>($($a),*),
            64 => $call::<64>($($a),*),
            128 => $call::<128>($($a),*),
            256 => $call::<256>($($a),*),
            512 => $call::<512>($($a),*),
            768 => $call::<768>($($a),*),
            1024 => $call::<1024>($($a),*),
            n => $dynamic(n, $($a),*),
        }
    };
}

macro_rules! matvec {
    ($n:expr, $skip:literal, $($a:expr),* $(,)?) => {
        match $n {
            16 => matvec_n::<16, $skip>($($a),*),
            32 => matvec_n::<32, $skip>($($a),*),
            64 => matvec_n::<64, $skip>($($a),*),
            128 => matvec_n::<128, $skip>($($a),*),
            256 => matvec_n::<256, $skip>($($a),*),
            512 => matvec_n::<512, $skip>($($a),*),
            768 => matvec_n::<768, $skip>($($a),*),
            1024 => matvec_n::<1024, $skip>($($a),*),
            n => matvec_dyn::<$skip>(n, $($a),*),
        }
    };
}

#[inline(always)]
fn add_n<const N: usize>(dst: &mut [f32], src: &[f32]) {
    let (d, s) = (&mut dst[..N], &src[..N]);
    for k in 0..N {
        d[k] += s[k];
    }
}

#[inline(always)]
fn add_dyn(n: usize, dst: &mut [f32], src: &[f32]) {
    for k in 0..n {
        dst[k] += src[k];
    }
}

#[inline(always)]
fn sub_n<const N: usize>(dst: &mut [f32], src: &[f32]) {
    let (d, s) = (&mut dst[..N], &src[..N]);
    for k in 0..N {
        d[k] -= s[k];
    }
}

#[inline(always)]
fn sub_dyn(n: usize, dst: &mut [f32], src: &[f32]) {
    for k in 0..n {
        dst[k] -= src[k];
    }
}

/// `z[k] += sum_j in[j] * w[j*H + k]`, input-major.
///
/// A constant inner trip count is necessary but was not sufficient: written as
/// `&w[j * H..j * H + H]` the slice index is a conditional panic *inside* the
/// loop, so `z` had to be coherent at every iteration and the generated code
/// was `vaddps (%rbx),%ymm0,%ymm0` / `vmovups %ymm0,(%rbx)` — a store-to-load
/// forward per multiply-add. `chunks_exact` does the check once, up front. The
/// same fix on `wdleval` was worth 4.3x on its widest layer, and every price
/// LEDGER 038/040 put on this net was measured before it.
///
/// `SKIP` drops zero inputs. Both call sites now pass `false`: once the
/// accumulators live in registers the branch costs a mispredict and saves a
/// couple of vector ops, which is a loss at every width in this net
/// (re-measured, `wdleval` first, LEDGER 055).
#[inline(always)]
fn matvec_n<const H: usize, const SKIP: bool>(input: &[f32], w: &[f32], z: &mut [f32]) {
    let z = &mut z[..H];
    for (&a, col) in input.iter().zip(w[..input.len() * H].chunks_exact(H)) {
        if SKIP && a == 0.0 {
            continue;
        }
        for k in 0..H {
            z[k] = fmadd(a, col[k], z[k]);
        }
    }
}

#[inline(always)]
fn matvec_dyn<const SKIP: bool>(h: usize, input: &[f32], w: &[f32], z: &mut [f32]) {
    let z = &mut z[..h];
    for (&a, col) in input.iter().zip(w[..input.len() * h].chunks_exact(h)) {
        if SKIP && a == 0.0 {
            continue;
        }
        for (zk, &wk) in z.iter_mut().zip(col) {
            *zk = fmadd(a, wk, *zk);
        }
    }
}

const MAGIC: u32 = 0x5155_4144;
/// Version 5 is the `Deep` head (l1 -> bucketed h x h -> bucketed h -> 1).
/// Version 6 is `CoreLora`: one shared core matrix on the FIRST half of each
/// perspective, a per-bucket adapter on the second half, both summing into the
/// same `hidden` units, plus a per-bucket direct read of the whole
/// accumulator. Both are read here because both are trained by `train.py` and
/// a file that silently loads as the wrong head evaluates plausibly and plays
/// every game slightly wrong.
const VERSION: u32 = 5;
const VERSION_CORELORA: u32 = 6;
const PAD: u16 = 768;
/// 32 pieces + 8 `pawnfile` + 7 `pawnpair`.
const MAX_ALL: usize = 48;

const EX_PAWNFILE: u32 = 0;
const EX_PAWNPAIR: u32 = 1;

struct Extra {
    kind: u32,
    off: usize,
}

struct Family {
    fam: u32,
    nbuck: usize,
    /// `[nbuck][h*h + h]`, w **input-major** `[in][out]` then the bias -- see
    /// `DeepNet::l1_w` for why.
    l2: Vec<f32>,
    /// `[nbuck][h + 1]`.
    l3: Vec<f32>,
}

/// `CoreLora`'s per-bucket tables, sized so that one evaluation touches the
/// shared `core_w` plus exactly one bucket's `lora` and `read` row.
///
/// That is the point of the split, and it is a cache property rather than an
/// arithmetic one: at width 128 / hidden 16 this is 8 KB shared + 9 KB per
/// bucket against version 5's 32.8 KB l1 + 4.2 KB per bucket, and Zen 3's L1D
/// is 32 KB. LEDGER 038 measured l1's footprint as the binding cost once the
/// weight layout was fixed.
struct CoreLoraHead {
    /// Dims per perspective the shared core reads. Stored in the file rather
    /// than assumed to be `width / 2`, so a net trained with a different
    /// `--core-frac` cannot load with the wrong split.
    wc: usize,
    fam: u32,
    nbuck: usize,
    /// `[2*wc][hidden]` -- input-major, transposed at load. Same reason as
    /// `DeepNet::l1_w`.
    core_w: Vec<f32>,
    core_b: Vec<f32>,
    /// `[nbuck][2*wl][hidden]`, each bucket input-major. `wl = width - wc`.
    lora: Vec<f32>,
    /// `[nbuck][2*width + 1]` -- weights over the whole activated accumulator,
    /// then the bias.
    read: Vec<f32>,
    /// `[nbuck][hidden + 1]` -- weights then bias.
    out: Vec<f32>,
}

pub struct DeepNet {
    width: usize,
    hidden: usize,
    scale: f32,
    extras: Vec<Extra>,
    fams: Vec<Family>,
    /// `[nrows][width]`.
    v: Vec<f32>,
    psqt: Vec<f32>,
    /// `[2*width][hidden]` -- **input-major**, transposed at load. The file
    /// stores `[hidden][2*width]`, which is the natural way to write a matrix
    /// and the wrong way to evaluate one: a row-major dot product is a
    /// reduction into a single scalar, and f32 addition is not associative, so
    /// LLVM must emit it as a serial chain of scalar FMAs. Input-major turns
    /// the inner loop into `hidden` independent accumulators, which vectorises.
    l1_w: Vec<f32>,
    l1_b: Vec<f32>,
    bias: f32,
    /// `Some` for a version-6 file. `fams`, `l1_w` and `l1_b` are then empty
    /// and `head` takes the CoreLora branch.
    cl: Option<CoreLoraHead>,
}

/// `[rows][cols]` row-major to `[cols][rows]` row-major.
fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut o = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            o[c * rows + r] = m[r * cols + c];
        }
    }
    o
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
}

impl DeepNet {
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
        if ver != VERSION && ver != VERSION_CORELORA {
            return Err(format!(
                "version {ver}, expected {VERSION} or {VERSION_CORELORA}"));
        }
        let nrows = r.u32()? as usize;
        let width = r.u32()? as usize;
        let hidden = r.u32()? as usize;
        let n_extra = r.u32()? as usize;
        let n_fam = r.u32()? as usize;
        let scale = r.f32()?;
        let wc = if ver == VERSION_CORELORA { r.u32()? as usize } else { 0 };

        let mut extras = Vec::with_capacity(n_extra);
        for _ in 0..n_extra {
            let kind = r.u32()?;
            let off = r.u32()? as usize;
            let _size = r.u32()?;
            extras.push(Extra { kind, off });
        }
        let mut dir = Vec::with_capacity(n_fam);
        for _ in 0..n_fam {
            let fam = r.u32()?;
            dir.push((fam, r.u32()? as usize));
        }

        let v = r.vec(nrows * width)?;
        let psqt = r.vec(nrows)?;

        if ver == VERSION_CORELORA {
            if n_fam != 1 {
                return Err(format!("corelora: {n_fam} families, expected 1"));
            }
            if wc == 0 || wc >= width {
                return Err(format!("corelora: wc {wc} out of range for width {width}"));
            }
            let (fam, nbuck) = dir[0];
            let wl = width - wc;
            let core_w = transpose(&r.vec(hidden * 2 * wc)?, hidden, 2 * wc);
            let core_b = r.vec(hidden)?;
            // One bucket's adapter is `[hidden][2*wl]` in the file and
            // `[2*wl][hidden]` here, for the same vectorisation reason l1 is.
            let mut lora = r.vec(nbuck * hidden * 2 * wl)?;
            let stride = hidden * 2 * wl;
            for b in 0..nbuck {
                let t = transpose(&lora[b * stride..(b + 1) * stride], hidden, 2 * wl);
                lora[b * stride..(b + 1) * stride].copy_from_slice(&t);
            }
            let read = r.vec(nbuck * (2 * width + 1))?;
            let out = r.vec(nbuck * (hidden + 1))?;
            let bias = r.vec(1)?[0];
            return Ok(DeepNet {
                width, hidden, scale, extras, fams: Vec::new(), v, psqt,
                l1_w: Vec::new(), l1_b: Vec::new(), bias,
                cl: Some(CoreLoraHead { wc, fam, nbuck, core_w, core_b, lora, read, out }),
            });
        }

        let l1_w = transpose(&r.vec(hidden * 2 * width)?, hidden, 2 * width);
        let l1_b = r.vec(hidden)?;
        let mut l2s = Vec::with_capacity(n_fam);
        for &(_, nb) in &dir {
            let mut t = r.vec(nb * (hidden * hidden + hidden))?;
            let stride = hidden * hidden + hidden;
            for b in 0..nb {
                let w = transpose(&t[b * stride..b * stride + hidden * hidden], hidden, hidden);
                t[b * stride..b * stride + hidden * hidden].copy_from_slice(&w);
            }
            l2s.push(t);
        }
        let mut fams = Vec::with_capacity(n_fam);
        for (i, &(fam, nbuck)) in dir.iter().enumerate() {
            fams.push(Family {
                fam,
                nbuck,
                l2: std::mem::take(&mut l2s[i]),
                l3: r.vec(nbuck * (hidden + 1))?,
            });
        }
        let bias = r.vec(1)?[0];

        Ok(DeepNet { width, hidden, scale, extras, fams, v, psqt, l1_w, l1_b, bias, cl: None })
    }

    /// The same board from the other chair: colours swap, ranks mirror. Matches
    /// `train.py::flip_feat`, which is what builds the second perspective.
    #[inline]
    fn flip(x: u16) -> u16 {
        let (side, pt, sq) = (x / 384, (x % 384) / 64, x % 64);
        (1 - side) * 384 + pt * 64 + (sq ^ 56)
    }

    /// Append the extra-input indices for one already-canonicalised feature set.
    ///
    /// Both families are pure functions of the two pawn sets, which is why they
    /// are cheap from scratch. `up = 0` and `dn = 7` are the "no pawn on this
    /// file" codes and are unambiguous: a pawn is never on rank 0 or rank 7.
    fn extras_of(&self, f: &[u16; MAX_FEAT], n: usize, out: &mut [u16], m: &mut usize) {
        let mut up = [0u16; 8];
        let mut dn = [7u16; 8];
        let mut has = [[0u16; 8]; 2];
        for &x in f.iter().take(n) {
            if (x % 384) / 64 != 0 {
                continue; // pawns only
            }
            let side = (x / 384) as usize;
            let (rk, fl) = ((x % 64) / 8, ((x % 64) % 8) as usize);
            has[side][fl] = 1;
            if side == 0 {
                up[fl] = up[fl].max(rk);
            } else {
                dn[fl] = dn[fl].min(rk);
            }
        }
        for e in &self.extras {
            match e.kind {
                EX_PAWNFILE => {
                    for fl in 0..8 {
                        out[*m] = (e.off + fl * 64 + (up[fl] * 8 + dn[fl]) as usize) as u16;
                        *m += 1;
                    }
                }
                EX_PAWNPAIR => {
                    for fl in 0..7 {
                        let code = ((has[0][fl] * 2 + has[0][fl + 1]) * 2 + has[1][fl]) * 2
                            + has[1][fl + 1];
                        out[*m] = (e.off + fl * 16 + code as usize) as u16;
                        *m += 1;
                    }
                }
                _ => {}
            }
        }
    }

    /// The two accumulator rows one piece contributes: what White's chair sees,
    /// and what Black's. `sq` is A1-relative, so Black's view mirrors it.
    /// Matches `qeval::QuadNet::features`, which canonicalises the same way.
    #[inline]
    fn rows(c: usize, pt: usize, sq: usize) -> (usize, usize) {
        (
            (if c == 0 { 0 } else { 384 }) + pt * 64 + sq,
            (if c == 1 { 0 } else { 384 }) + pt * 64 + (sq ^ 56),
        )
    }

    /// Fill a `2 * width` accumulator with the piece rows only, keyed by
    /// **perspective colour**: first half is White's chair, second is Black's.
    ///
    /// Keying by colour rather than by side to move is what makes a null move
    /// free -- no piece moves, so neither half changes -- and lets `evaluate`
    /// pick its halves with an index instead of a rebuild.
    fn fill_colour(&self, b: &Board, acc: &mut [f32]) {
        let w = self.width;
        acc[..2 * w].fill(0.0);
        for c in Color::ALL {
            for pt in PieceType::ALL {
                for sq in b.colored(c, pt) {
                    let (wi, bi) = Self::rows(c.index(), pt.index(), sq.index());
                    let (rw, rb) = (wi * w, bi * w);
                    let (lo, hi) = acc.split_at_mut(w);
                    dims!(w, add_n, add_dyn, lo, &self.v[rw..]);
                    dims!(w, add_n, add_dyn, hi, &self.v[rb..]);
                }
            }
        }
    }

    /// Everything downstream of the piece accumulator: add the pawn-structure
    /// rows, then crelu, l1, the bucketed pair, and the PSQT skip.
    ///
    /// `own_pieces`/`opp_pieces` are the two halves `fill_colour` produced,
    /// already ordered for the side to move.
    fn head(&self, b: &Board, own_pieces: &[f32], opp_pieces: &[f32], s: &mut Scratch) -> Score {
        let (w, h) = (self.width, self.hidden);
        let mut f = [0u16; MAX_FEAT];
        let n = QuadNet::features(b, &mut f);

        // Perspective 0 is the side to move, perspective 1 the same board seen
        // from the other chair. The extras are recomputed on the FLIPPED set,
        // not shared -- `Deep.forward` calls `self.ex(flip_feat(feat))`, so a
        // shared copy would be a different model.
        let mut fw = [0u16; MAX_FEAT];
        let mut own = [0u16; MAX_ALL];
        let mut opp = [0u16; MAX_ALL];
        for i in 0..n {
            fw[i] = Self::flip(f[i]);
            own[i] = f[i];
            opp[i] = fw[i];
        }
        let (mut nown, mut nopp) = (n, n);
        self.extras_of(&f, n, &mut own, &mut nown);
        self.extras_of(&fw, n, &mut opp, &mut nopp);

        // Accumulator: 2 x width, own half then opponent half, exactly the
        // concatenation order `Deep.forward` builds. The piece rows arrive
        // summed; only the pawn-structure rows are added here.
        let acc = &mut s.acc[..2 * w];
        acc[..w].copy_from_slice(own_pieces);
        acc[w..].copy_from_slice(opp_pieces);
        for (half, (idx, cnt)) in [(&own, nown), (&opp, nopp)].iter().enumerate() {
            let base = half * w;
            for &x in idx[n..*cnt].iter() {
                if x == PAD {
                    continue;
                }
                let row = x as usize * w;
                dims!(w, add_n, add_dyn, &mut acc[base..], &self.v[row..]);
            }
        }
        if s.abl & 1 != 0 {
            let d = &mut s.dummy[..2 * w];
            for (half, (idx, cnt)) in [(&own, nown), (&opp, nopp)].iter().enumerate() {
                for &x in idx[n..*cnt].iter() {
                    let row = x as usize * w;
                    dims!(w, add_n, add_dyn, &mut d[half * w..], &self.v[row..]);
                }
            }
        }
        let acc = &mut s.acc[..2 * w];
        for a in acc.iter_mut() {
            *a = a.clamp(0.0, 1.0);
        }

        // ------------------------------------------------- CoreLora (v6)
        // `z = crelu(core @ [a_own[..wc], a_opp[..wc]] + core_b
        //             + lora[b] @ [a_own[wc..], a_opp[wc..]])`, then the answer
        // is a direct per-bucket read of the whole accumulator plus a
        // per-bucket read of `z`. Exactly `train.py::CoreLora.forward`.
        //
        // There is no bucket-weight cache here and there should not be: with a
        // single family the summed copy version 5 needs collapses to indexing,
        // so a miss costs nothing but the loads it was going to do anyway.
        if let Some(cl) = &self.cl {
            let (acc, z) = (&s.acc[..2 * w], &mut s.z[..h]);
            let (wc, wl) = (cl.wc, w - cl.wc);
            let bk = QuadNet::bucket(cl.fam, &f, n).min(cl.nbuck - 1);

            z.copy_from_slice(&cl.core_b[..h]);
            matvec!(h, false, &acc[..wc], &cl.core_w, z);
            matvec!(h, false, &acc[w..w + wc], &cl.core_w[wc * h..], z);
            let lw = &cl.lora[bk * 2 * wl * h..(bk + 1) * 2 * wl * h];
            matvec!(h, false, &acc[wc..w], lw, z);
            matvec!(h, false, &acc[w + wc..2 * w], &lw[wl * h..], z);

            let rd = &cl.read[bk * (2 * w + 1)..(bk + 1) * (2 * w + 1)];
            let ou = &cl.out[bk * (h + 1)..(bk + 1) * (h + 1)];
            let mut out = rd[2 * w] + ou[h];
            for i in 0..2 * w {
                out += acc[i] * rd[i];
            }
            for k in 0..h {
                out += z[k].clamp(0.0, 1.0) * ou[k];
            }
            // The PSQT skip reads the side-to-move perspective only, matching
            // `self.psqt(ext)` -- ext, not the flipped set.
            for &x in own.iter().take(nown) {
                if x != PAD {
                    out += self.psqt[x as usize];
                }
            }
            let cp = (out + self.bias) * self.scale;
            return cp.round().clamp(-30_000.0, 30_000.0) as Score;
        }

        // l1, shared across buckets. Loop order is input-major on purpose: the
        // inner loop writes `h` independent accumulators, so it vectorises,
        // where the dot-product order is a serial f32 reduction and cost 2.5x
        // as much. Skipping `a == 0.0` was tried and measured neutral -- crelu
        // does not zero enough of the accumulator to pay for the branch.
        let (acc, z) = (&s.acc[..2 * w], &mut s.z[..h]);
        for _ in 0..if s.abl & 2 != 0 { 2 } else { 1 } {
            z.copy_from_slice(&self.l1_b[..h]);
            matvec!(h, false, acc, &self.l1_w, z);
            for t in z.iter_mut() {
                *t = t.clamp(0.0, 1.0);
            }
        }

        // Bucketed layers. Families are SUMMED, not crossed, so the weights add
        // before they are applied -- one 576-row table plus one 8-row table,
        // not a 4608-row product.
        // `QuadNet::bucket` walks the feature list, so it is computed once per
        // family and then reused for both the cache key and the sum.
        let mut bks = [0usize; 4];
        let mut key = 1u64;
        for (i, fam) in self.fams.iter().enumerate() {
            debug_assert!(fam.nbuck < 1 << 20 && i < bks.len());
            bks[i] = QuadNet::bucket(fam.fam, &f, n).min(fam.nbuck - 1);
            key = (key << 20) | bks[i] as u64;
        }
        if s.abl & 4 != 0 {
            s.bkey = 0; // cache off, for the ablation
        }
        let (w2, b2, w3) = (&mut s.w2[..h * h], &mut s.b2[..h], &mut s.w3[..h]);
        if key != s.bkey {
            s.bkey = key;
            s.b3 = 0.0;
            w2.fill(0.0);
            b2.fill(0.0);
            w3.fill(0.0);
            for (i, fam) in self.fams.iter().enumerate() {
                let bk = bks[i];
                let p = &fam.l2[bk * (h * h + h)..(bk + 1) * (h * h + h)];
                for i in 0..h * h {
                    w2[i] += p[i];
                }
                for i in 0..h {
                    b2[i] += p[h * h + i];
                }
                let q = &fam.l3[bk * (h + 1)..(bk + 1) * (h + 1)];
                for i in 0..h {
                    w3[i] += q[i];
                }
                s.b3 += q[h];
            }
            BUCKMISS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let b3 = s.b3;

        let mut out = b3;
        let y = &mut s.y[..h];
        for _ in 0..if s.abl & 8 != 0 { 2 } else { 1 } {
            y.copy_from_slice(b2);
            matvec!(h, false, z, w2, y);
            out = b3;
            for k in 0..h {
                out += y[k].clamp(0.0, 1.0) * w3[k];
            }
        }

        // The PSQT skip reads the side-to-move perspective only, matching
        // `self.psqt(ext)` -- ext, not the flipped set.
        for &x in own.iter().take(nown) {
            if x != PAD {
                out += self.psqt[x as usize];
            }
        }
        let cp = (out + self.bias) * self.scale;
        cp.round().clamp(-30_000.0, 30_000.0) as Score
    }

    /// `eval` in centipawns, from the side to move's point of view, rebuilt
    /// from the board. This is what `evalfen` and `nnue/verify_deep.py` call;
    /// the search uses `DeepEval`, which reaches the same `head` with an
    /// incrementally maintained accumulator.
    pub fn evaluate(&self, b: &Board) -> Score {
        let w = self.width;
        let mut s = Scratch::new(w, self.hidden);
        let mut acc = vec![0f32; 2 * w];
        self.fill_colour(b, &mut acc);
        let (wh, bl) = acc.split_at(w);
        let (own, opp) = if b.stm() == Color::White { (wh, bl) } else { (bl, wh) };
        self.head(b, own, opp, &mut s)
    }
}

/// Per-thread working memory. Every buffer here used to be a `vec!` inside
/// `evaluate` -- six allocations per node, in the hottest loop in the program.
struct Scratch {
    /// `CHESS_ABL=<bits>` runs one stage of `head` a second time, writing the
    /// same answer. Node counts and evals are untouched, so the drop in nps is
    /// that stage's marginal share of eval time. Needed because
    /// `perf_event_paranoid` is 4 on this machine and profiling wants sudo.
    /// 1 = extras gather twice, 2 = l1 twice, 4 = bucket-weight cache OFF,
    /// 8 = l2 twice.
    abl: u8,
    dummy: Vec<f32>,
    acc: Vec<f32>,
    z: Vec<f32>,
    y: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: f32,
    /// The bucket indices `w2`/`b2`/`w3`/`b3` were summed for, packed 20 bits
    /// per family; 0 means "nothing cached". The pair only moves when material
    /// does, so between captures this is the same read every node -- and it is
    /// a random 4.2 KB row out of a 2.4 MB table, which the ablation says was
    /// the most expensive thing in the evaluator.
    bkey: u64,
}

impl Scratch {
    fn new(w: usize, h: usize) -> Self {
        Scratch {
            abl: std::env::var("CHESS_ABL").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            dummy: vec![0.0; 2 * w],
            acc: vec![0.0; 2 * w],
            z: vec![0.0; h],
            y: vec![0.0; h],
            w2: vec![0.0; h * h],
            b2: vec![0.0; h],
            w3: vec![0.0; h],
            b3: 0.0,
            bkey: 0,
        }
    }
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

/// One more than the deepest ply the search can reach, plus a slot at the end
/// for rebuilding when quiescence runs past it.
const STACK: usize = crate::search::MAX_PLY + 8;

/// The deep net with an incremental accumulator.
///
/// `push` is told only the new board, not the move: it XORs the 12 piece
/// bitboards against the parent's and walks the set bits. That is one mechanism
/// for quiet moves, captures, castling, en passant and promotion alike -- there
/// is no move-type switch to get wrong -- and it costs at most five row updates
/// where a rebuild costs 64.
///
/// `evaluate` checks the stack against the board it was handed and rebuilds if
/// they disagree, so a missing `push` anywhere in the search costs speed rather
/// than correctness. `bench` prints the count; a correct wiring reports zero.
pub struct DeepEval {
    net: &'static DeepNet,
    /// `[STACK + 1][2 * width]`, keyed by perspective colour. The last slot is
    /// where a too-deep ply rebuilds into.
    acc: Vec<f32>,
    bb: Vec<[u64; 12]>,
    /// Whether `acc[i]`/`bb[i]` describe a real position. A level is seeded by
    /// `evaluate` rebuilding into it, or by a `push` from a seeded parent --
    /// and the root is seeded by neither until something evaluates it. The
    /// static eval is skipped in check and behind a TT hit, so an unguarded
    /// `push` from the root diffs against nothing and every descendant of that
    /// node inherits the garbage.
    ok: Vec<bool>,
    sp: usize,
    s: Scratch,
    resyncs: u64,
    evals: u64,
    /// `CHESS_ACC_CHECK=1`: recompute every eval from scratch and compare.
    /// Off, this is one predictable branch per node; on, it is the only test
    /// that actually exercises the incremental path, since `verify_deep.py`
    /// goes through `evaluate` and never touches `push`.
    check: bool,
    /// `CHESS_ACC_OFF=1`: rebuild every eval, so the two paths can be timed and
    /// counted in the same binary with everything else held fixed.
    off: bool,
    mism: u64,
    worst: i32,
}

impl DeepEval {
    pub fn new(net: &'static DeepNet) -> Self {
        let w = net.width;
        DeepEval {
            net,
            acc: vec![0.0; (STACK + 1) * 2 * w],
            bb: vec![[0u64; 12]; STACK + 1],
            ok: vec![false; STACK + 1],
            sp: 0,
            s: Scratch::new(w, net.hidden),
            resyncs: 0,
            evals: 0,
            check: std::env::var("CHESS_ACC_CHECK").is_ok(),
            off: std::env::var("CHESS_ACC_OFF").is_ok(),
            mism: 0,
            worst: 0,
        }
    }
}

impl Clone for DeepEval {
    fn clone(&self) -> Self {
        DeepEval::new(self.net)
    }
}

impl Drop for DeepEval {
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
/// Counted only to report the bucket cache's hit rate in `bench`; one relaxed
/// add on the miss path, none on the hit path.
static BUCKMISS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MISM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WORST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(rebuilds, evals, mismatches, worst cp)` summed over every `DeepEval`
/// dropped so far. Counted locally and flushed on drop, so the hot path pays
/// one increment. The last two are zero unless `CHESS_ACC_CHECK` was set.
pub fn bucket_misses() -> u64 {
    BUCKMISS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn resync_stats() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (RESYNCS.load(Relaxed), EVALS.load(Relaxed), MISM.load(Relaxed), WORST.load(Relaxed))
}

impl Evaluator for DeepEval {
    fn evaluate(&mut self, b: &Board) -> Score {
        let (net, w) = (self.net, self.net.width);
        self.evals += 1;
        if self.off {
            return net.evaluate(b);
        }
        let nbb = bbs(b);
        let slot = if self.sp < STACK && self.ok[self.sp] && self.bb[self.sp] == nbb {
            self.sp
        } else {
            self.resyncs += 1;
            let slot = self.sp.min(STACK);
            let base = slot * 2 * w;
            net.fill_colour(b, &mut self.acc[base..base + 2 * w]);
            self.bb[slot] = nbb;
            self.ok[slot] = true;
            slot
        };
        let base = slot * 2 * w;
        let (wh, bl) = self.acc[base..base + 2 * w].split_at(w);
        let (own, opp) = if b.stm() == Color::White { (wh, bl) } else { (bl, wh) };
        let got = net.head(b, own, opp, &mut self.s);
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
            // Nothing to diff against. Rebuilding is correct and rare.
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
            let mut d = old[i] ^ nbb[i];
            if d == 0 {
                continue;
            }
            let (c, pt) = (i / 6, i % 6);
            while d != 0 {
                let sq = d.trailing_zeros() as usize;
                d &= d - 1;
                let (wi, bi) = DeepNet::rows(c, pt, sq);
                let (rw, rb) = (wi * w, bi * w);
                let (lo, hi) = next.split_at_mut(w);
                if nbb[i] >> sq & 1 == 1 {
                    dims!(w, add_n, add_dyn, lo, &net.v[rw..]);
                    dims!(w, add_n, add_dyn, hi, &net.v[rb..]);
                } else {
                    dims!(w, sub_n, sub_dyn, lo, &net.v[rw..]);
                    dims!(w, sub_n, sub_dyn, hi, &net.v[rb..]);
                }
            }
        }
    }

    fn pop(&mut self) {
        debug_assert!(self.sp > 0);
        self.sp -= 1;
    }
}

static NET: std::sync::OnceLock<Option<DeepNet>> = std::sync::OnceLock::new();

/// `--deep <path>` or `$CHESS_DEEP`. Same shape as `qeval::net` so the two can
/// be selected the same way; when both are given the deep net wins, since it is
/// the strictly newer model.
pub fn net() -> Option<&'static DeepNet> {
    NET.get_or_init(|| {
        let args: Vec<String> = std::env::args().collect();
        let path = args
            .iter()
            .position(|a| a == "--deep")
            .and_then(|i| args.get(i + 1))
            .cloned()
            .or_else(|| std::env::var("CHESS_DEEP").ok())?;
        match DeepNet::load(&path) {
            Ok(n) => {
                eprintln!("info string deep eval loaded from {path}");
                Some(n)
            }
            Err(e) => {
                eprintln!("info string deep eval NOT loaded: {e}");
                None
            }
        }
    })
    .as_ref()
}
