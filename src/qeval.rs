//! The trained quadratic evaluator: `eval = x^T M x + c`.
//!
//! `x` is the 768-bit piece-square occupancy vector — 12 piece types by 64
//! squares — canonicalised so the side to move is always "White at the bottom".
//! `M` is a symmetric 768x768 matrix trained by `nnue/train.py` and collapsed by
//! `nnue/export.py`. Because `x` is binary, `x_i^2 = x_i`, so the piece-square
//! table folds into the diagonal of `M` and there is no separate linear term:
//! the whole eval is one matrix and one scalar.
//!
//! **There is no accumulator.** `eval` is a sum over *pairs of pieces on the
//! board*, so lifting a piece off square `p` costs
//!
//! ```text
//! delta = -M[p][p] - 2 * sum_{i in S, i != p} M[p][i]
//! ```
//!
//! which is ~32 gathers, and adding one costs the same. An NNUE keeps `a = Wx`
//! only because a nonlinearity downstream has to see every hidden unit before
//! it can be applied; a quadratic form has no such layer, so an accumulator
//! would cost 768 adds per toggle to save work that was never needed. See
//! LEDGER 019 and the note in ARCHITECTURE.md.
//!
//! The diagonal is stored separately and at higher precision. It is the PSQT —
//! "opponent queen on e4" is -1635 cp — while the off-diagonal peaks at 452, so
//! one i16 scale for both would be set by the diagonal and throw away five bits
//! on every pairwise term.
//!
//! All three terms share a single rounding at the end. Rounding each one
//! separately cost up to 0.5 cp apiece and put a systematic -0.46 cp into every
//! eval, which is real strength thrown away for nothing.
//!
//! # The bucketed PSQT (file version 4)
//!
//! A version-4 file carries extra diagonals, one set per bucket of a routing
//! rule, added to `diag` for whichever bucket the position falls in. Because
//! the linear term has no accumulator, this is the one conditioning in the
//! eval study that costs the engine nothing at all: still 32 gathers, from a
//! 584 x 768 table instead of a 768 one. Everything else the study found
//! (read buckets, pawn features) needs `a = Vx` kept incrementally first.
//!
//! Families are SUMMED, not crossed -- a 576-entry table and an 8-entry one,
//! not 4608 -- exactly as `Bucketed` in `nnue/train.py` trains them.
//!
//! The bucket index is computed from the canonicalised feature list rather
//! than from the board, because that is what `nnue/rulestats.py` indexes and
//! the two must agree bit for bit. In particular a bishop's square colour is
//! read AFTER the `sq ^ 56` flip, so "light-squared" means light in the
//! network's frame, not on the real board. `chess evalfen` plus
//! `nnue/verify.py` is what checks this.

use std::sync::OnceLock;

use crate::board::Board;
use crate::eval::{Evaluator, Score};
use crate::types::{Color, PieceType};

pub const NFEAT: usize = 768;
const MAGIC: u32 = 0x5155_4144; // "QUAD"
/// Highest file version understood. Older ones still load: v3 is v4 with no
/// bucket families, so the deployed net keeps working untouched.
const VERSION: u32 = 4;
/// Family ids in the file. Must match `FAMILY_ID` in `nnue/export.py`.
pub const FAM_MATERIAL: u32 = 0;
pub const FAM_COUNT: u32 = 1;
/// A position has at most 32 pieces, so at most 32 active features.
pub const MAX_FEAT: usize = 32;

/// One bucketed-PSQT family: `table[bucket * NFEAT + feature]`, at `q_diag`.
struct PsqtBucket {
    fam: u32,
    table: Vec<i32>,
}

pub struct QuadNet {
    q_off: i64,
    q_diag: i64,
    bias: i64,
    diag: Vec<i32>,
    off: Vec<i16>,
    pq: Vec<PsqtBucket>,
}

fn rd(v: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([v[at], v[at + 1], v[at + 2], v[at + 3]])
}

impl QuadNet {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let head = 6 * 4;
        if bytes.len() < head {
            return Err(format!("short file: {} bytes, want at least {head}", bytes.len()));
        }
        if rd(bytes, 0) != MAGIC {
            return Err("bad magic, not a QUAD file".into());
        }
        let version = rd(bytes, 4);
        if version != 3 && version != VERSION {
            return Err(format!("version {version}, want 3 or {VERSION}"));
        }
        let q_off = rd(bytes, 8) as i32 as i64;
        let q_diag = rd(bytes, 12) as i32 as i64;
        let bias = rd(bytes, 16) as i32 as i64;
        // v3 wrote a zero here, so reading it as a family count is correct for
        // both versions and there is no branch on the version below.
        let nfam = rd(bytes, 20) as usize;

        // The family directory sits between the header and the diagonal, so
        // every later offset shifts by it.
        let dir = head + nfam * 8;
        let mut fams = Vec::with_capacity(nfam);
        for k in 0..nfam {
            let fam = rd(bytes, head + k * 8);
            let nb = rd(bytes, head + k * 8 + 4) as usize;
            let want = match fam {
                FAM_MATERIAL => 576,
                FAM_COUNT => 8,
                _ => return Err(format!("unknown psqt bucket family {fam}")),
            };
            if nb != want {
                return Err(format!("family {fam}: {nb} buckets, want {want}"));
            }
            fams.push((fam, nb));
        }
        let want = dir + NFEAT * 4 + NFEAT * NFEAT * 2
            + fams.iter().map(|(_, nb)| nb * NFEAT * 4).sum::<usize>();
        if bytes.len() < want {
            return Err(format!("short file: {} bytes, want {want}", bytes.len()));
        }

        let mut diag = vec![0i32; NFEAT];
        for (i, d) in diag.iter_mut().enumerate() {
            *d = rd(bytes, dir + i * 4) as i32;
        }
        let obase = dir + NFEAT * 4;
        let mut off = vec![0i16; NFEAT * NFEAT];
        for (i, o) in off.iter_mut().enumerate() {
            let at = obase + i * 2;
            *o = i16::from_le_bytes([bytes[at], bytes[at + 1]]);
        }
        let mut at = obase + NFEAT * NFEAT * 2;
        let mut pq = Vec::with_capacity(nfam);
        for (fam, nb) in fams {
            let mut table = vec![0i32; nb * NFEAT];
            for (i, t) in table.iter_mut().enumerate() {
                *t = rd(bytes, at + i * 4) as i32;
            }
            at += nb * NFEAT * 4;
            pq.push(PsqtBucket { fam, table });
        }
        Ok(Self { q_off, q_diag, bias, diag, off, pq })
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Self::parse(&bytes)
    }

    /// Active feature indices, side-to-move relative.
    ///
    /// When Black is to move the board is mirrored vertically (`sq ^ 56`) and
    /// the colours exchange roles, so the network always sees itself as White
    /// at the bottom. This must match `nnue/extract` exactly: the training
    /// records were canonicalised the same way, and getting it wrong mislabels
    /// every Black position without any test failing.
    #[inline]
    pub fn features(b: &Board, out: &mut [u16; MAX_FEAT]) -> usize {
        let stm = b.stm();
        let flip = if stm == Color::White { 0 } else { 56 };
        let mut n = 0;
        for c in Color::ALL {
            let side = if c == stm { 0 } else { 384 };
            for pt in PieceType::ALL {
                let base = side + 64 * pt.index();
                for sq in b.colored(c, pt) {
                    out[n] = (base + (sq.index() ^ flip)) as u16;
                    n += 1;
                }
            }
        }
        n
    }

    /// The bucket a position falls in, for one family.
    ///
    /// Read off the canonicalised feature list, not the board, so that this is
    /// the same arithmetic `nnue/rulestats.py` does on the same numbers. A
    /// feature is `side*384 + type*64 + square` with side 0 = the side to move
    /// and the square already flipped for Black.
    pub fn bucket_of(b: &Board, fam: u32) -> usize {
        let mut f = [0u16; MAX_FEAT];
        let n = Self::features(b, &mut f);
        Self::bucket(fam, &f, n)
    }

    pub fn bucket(fam: u32, f: &[u16; MAX_FEAT], n: usize) -> usize {
        match fam {
            // Standard NNUE output bucketing: total pieces in 8 bins.
            FAM_COUNT => (n.saturating_sub(2) / 4).min(7),
            // [rooks 0/1/2+] x [light bishop] x [dark bishop] x [queen],
            // squared: our code * 24 + theirs.
            FAM_MATERIAL => {
                let mut rook = [0usize; 2];
                let mut queen = [0usize; 2];
                let mut bishop = [[0usize; 2]; 2];
                for &x in f.iter().take(n) {
                    let x = x as usize;
                    let side = x / 384;
                    let pt = (x % 384) / 64;
                    let sq = x % 64;
                    match pt {
                        2 => bishop[side][(sq / 8 + sq % 8) & 1] = 1,
                        3 => rook[side] += 1,
                        4 => queen[side] = 1,
                        _ => {}
                    }
                }
                let code = |s: usize| {
                    ((rook[s].min(2) * 2 + bishop[s][0]) * 2 + bishop[s][1]) * 2 + queen[s]
                };
                code(0) * 24 + code(1)
            }
            _ => 0,
        }
    }

    /// Evaluate from scratch: the upper triangle of the active submatrix.
    pub fn evaluate(&self, b: &Board) -> Score {
        let mut f = [0u16; MAX_FEAT];
        let n = Self::features(b, &mut f);

        // One extra diagonal per bucketed family, resolved once per eval. The
        // rows are then summed into `lin` alongside `diag`, so the bucketed
        // PSQT costs the same 32 gathers per family and no extra rounding.
        let mut rows: [&[i32]; 4] = [&[]; 4];
        let mut nrows = 0;
        for p in &self.pq {
            let b = Self::bucket(p.fam, &f, n);
            rows[nrows] = &p.table[b * NFEAT..b * NFEAT + NFEAT];
            nrows += 1;
        }

        let mut pair: i64 = 0;
        let mut lin: i64 = 0;
        for i in 0..n {
            let fi = f[i] as usize;
            lin += self.diag[fi] as i64;
            for r in rows.iter().take(nrows) {
                // SAFETY: fi < 768 as argued below, and each row is a
                // NFEAT-long slice taken from the table above.
                lin += unsafe { *r.get_unchecked(fi) } as i64;
            }
            // SAFETY: fi < 768 because `features` builds every index as
            // side(0|384) + 64*pt(0..5) + sq(0..63) <= 767, so the row
            // [fi*768, fi*768+768) is inside `off`, which has 768*768 entries.
            let row = unsafe { self.off.get_unchecked(fi * NFEAT..fi * NFEAT + NFEAT) };
            for j in (i + 1)..n {
                pair += unsafe { *row.get_unchecked(f[j] as usize) } as i64;
            }
        }
        // `off` is symmetric with a zero diagonal, so the full double sum is
        // twice the strict upper triangle.
        // One rounding, at the very end. Rounding each term separately costs
        // up to 0.5 cp apiece, and `Score` only carries whole centipawns to
        // begin with. `bias` is stored at q_diag precision for the same reason.
        let num = 2 * pair * self.q_diag + (lin + self.bias) * self.q_off;
        let cp = div_round(num, self.q_off * self.q_diag);
        cp.clamp(-30_000, 30_000) as Score
    }
}

#[inline]
fn div_round(a: i64, b: i64) -> i64 {
    if a >= 0 { (a + b / 2) / b } else { (a - b / 2) / b }
}

// ---------------------------------------------------------------- selection

static NET: OnceLock<Option<QuadNet>> = OnceLock::new();

/// Load the net once, from `--quad <path>`, then `$CHESS_QUAD`, then
/// `quad.nnue` beside the binary. Absent or unreadable means the engine runs
/// the PeSTO control arm, which is what makes one binary able to play both
/// sides of the SPRT.
pub fn net() -> Option<&'static QuadNet> {
    NET.get_or_init(|| {
        let args: Vec<String> = std::env::args().collect();
        let mut path = args
            .iter()
            .position(|a| a == "--quad")
            .and_then(|i| args.get(i + 1))
            .cloned()
            .or_else(|| std::env::var("CHESS_QUAD").ok());
        if path.is_none() {
            let beside = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("quad.nnue")));
            if let Some(p) = beside {
                if p.exists() {
                    path = Some(p.to_string_lossy().into_owned());
                }
            }
        }
        let path = path?;
        match QuadNet::load(&path) {
            Ok(n) => {
                eprintln!("info string quad eval loaded from {path}");
                Some(n)
            }
            Err(e) => {
                eprintln!("info string quad eval NOT loaded ({e}); using PeSTO");
                None
            }
        }
    })
    .as_ref()
}

/// The evaluator the engine actually runs: the deep net if one is loaded, else
/// the trained quadratic form, else the PeSTO control arm. One type so
/// `Searcher` stays monomorphic and there is a single code path.
///
/// `--wdl` wins over `--deep`, which wins over `--quad`: newest model first,
/// and having one dispatch point means bench, match, the GUI and UCI all pick
/// it up without a second wiring. Only the winner is constructed, so a loser's
/// accumulator is never built and never pushed. The deep arm owns an accumulator stack,
/// which is why this is no longer a unit struct -- it must be built once per
/// search thread and then pushed and popped, not copied per call.
/// Only build the deep net's accumulator if the WDL net did not claim the slot;
/// both would otherwise be pushed on every node for one of them to be read.
fn deep_unless_wdl() -> Option<Box<crate::deepeval::DeepEval>> {
    if crate::wdleval::net().is_some() {
        return None;
    }
    crate::deepeval::net().map(|n| Box::new(crate::deepeval::DeepEval::new(n)))
}

#[derive(Clone)]
pub struct DefaultEval {
    wdl: Option<Box<crate::wdleval::WdlEval>>,
    deep: Option<Box<crate::deepeval::DeepEval>>,
    adapt: Option<std::sync::Arc<std::sync::Mutex<crate::adaptive::AdaptiveState>>>,
    /// Dual-eval tier. When set (quiescence, behind `--dual`), `evaluate`
    /// skips the WDL/deep net and runs the quad/PeSTO path — no accumulator,
    /// no incremental state, so there is nothing to push and nothing to keep
    /// in sync. A plain store, set once per search node.
    small: bool,
}

impl Default for DefaultEval {
    fn default() -> Self {
        DefaultEval {
            wdl: crate::wdleval::net().map(|n| Box::new(crate::wdleval::WdlEval::new(n))),
            deep: deep_unless_wdl(),
            adapt: Some(crate::adaptive::global_handle()),
            small: false,
        }
    }
}

impl DefaultEval {
    pub fn without_adaptive() -> Self {
        Self {
            wdl: crate::wdleval::net().map(|n| Box::new(crate::wdleval::WdlEval::new(n))),
            deep: deep_unless_wdl(),
            adapt: None,
            small: false,
        }
    }
    pub fn with_adaptive(handle: std::sync::Arc<std::sync::Mutex<crate::adaptive::AdaptiveState>>) -> Self {
        Self {
            wdl: crate::wdleval::net().map(|n| Box::new(crate::wdleval::WdlEval::new(n))),
            deep: deep_unless_wdl(),
            adapt: Some(handle),
            small: false,
        }
    }

    /// The quad/PeSTO path with the adaptive correction: what the engine
    /// evaluates when no deep net is loaded, and what `--dual` runs in
    /// quiescence. Lives here, in the inherent block — a helper with this
    /// signature inside `impl Evaluator` would not be a trait member.
    fn quad_eval(&mut self, b: &Board) -> Score {
        let base = match net() {
            Some(n) => n.evaluate(b),
            None => crate::eval::evaluate_pst(b),
        };
        if let Some(h) = &self.adapt {
            if let Ok(st) = h.lock() {
                if !st.params.disabled() {
                    let mut f = [0u16; MAX_FEAT];
                    let n = QuadNet::features(b, &mut f);
                    let corr = st.eval_correction(&f, n);
                    return (base as f32 + corr).round().clamp(-30_000.0, 30_000.0) as Score;
                }
            }
        }
        base
    }
}

/// `--dual` or `$CHESS_DUAL`: quiescence evaluates with the cheap quad net
/// while the main search keeps the WDL net. Parsed once at startup (see
/// `crate::init`); a match arm enables it by appending `--dual` to its
/// `--engine "..."` spec, so both sides run the same binary and only the
/// flag differs.
static DUAL: OnceLock<bool> = OnceLock::new();

pub fn init_dual() {
    let args: Vec<String> = std::env::args().collect();
    let on = args.iter().any(|a| a == "--dual") || std::env::var("CHESS_DUAL").is_ok();
    let _ = DUAL.set(on);
    let sig = args
        .iter()
        .position(|a| a == "--qnoise")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<f32>().ok())
        .or_else(|| std::env::var("CHESS_QNOISE").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(0.0);
    let _ = QNOISE.set(sig.max(0.0));
}

/// Whether this process evaluates quiescence on the small net. One `OnceLock`
/// load per search node; false in every existing build, test and fingerprint.
#[inline]
pub fn dual_enabled() -> bool {
    *DUAL.get_or_init(|| false)
}

/// `--qnoise <cp>` or `$CHESS_QNOISE`: the noise-tolerance probe. Quiescence
/// keeps the FULL net but every q-eval gets a deterministic pseudo-Gaussian
/// perturbation with this σ, seeded by the position key — so a search is
/// exactly reproducible and the TT stores one value per key. It prices how
/// much tier disagreement the search tolerates at zero speed benefit: the
/// number a distilled small net has to beat. It also stacks on the dual
/// small tail when both flags are given (that combination has no ledger
/// entry yet — the probe it was built for ran on the full net).
static QNOISE: OnceLock<f32> = OnceLock::new();

/// The probe σ in cp. Zero when off.
#[inline]
fn qnoise_sigma() -> f32 {
    *QNOISE.get_or_init(|| 0.0)
}

/// Whether the probe is armed. The search reads this once per node alongside
/// `dual_enabled` to decide whether the q-tier flag is worth setting.
#[inline]
pub fn qnoise_armed() -> bool {
    qnoise_sigma() > 0.0
}

/// Deterministic N(0, σ) perturbation in cp, seeded by the position key.
/// splitmix64 for the bits, Box-Muller for the shape; `+0.5` keeps `u1` off
/// zero so the log is finite. Clamped back into eval range so a 6σ draw can
/// never be mistaken for a mate score.
fn perturb(s: Score, key: u64, sig: f32) -> Score {
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    const TWO53: f64 = 9007199254740992.0;
    let u1 = ((mix(key) >> 11) as f64 + 0.5) / TWO53;
    let u2 = ((mix(key.wrapping_add(1)) >> 11) as f64 + 0.5) / TWO53;
    let g = (-2.0 * u1.ln()).sqrt() * (6.283185307179586 * u2).cos();
    (s as f64 + sig as f64 * g).round().clamp(-30_000.0, 30_000.0) as Score
}

impl Evaluator for DefaultEval {
    fn evaluate(&mut self, b: &Board) -> Score {
        // Q-tier behind `--dual`. A dual file serves the distilled small tail
        // off the live accumulator; a single-tail file keeps the rung-1
        // policy (the cheap quad net), which is what priced the 1.29x.
        if self.small && dual_enabled() {
            if let Some(d) = self.wdl.as_mut() {
                if d.net_has_small() {
                    let s = d.evaluate_small(b);
                    return if qnoise_armed() { perturb(s, b.key(), qnoise_sigma()) } else { s };
                }
            }
            return self.quad_eval(b);
        }
        // Everywhere else the full net runs. In quiescence with the probe
        // armed (--qnoise) its score carries the deterministic perturbation;
        // the TT stores it like any other eval, which is exactly the tier
        // disagreement being priced.
        if let Some(d) = self.wdl.as_mut() {
            let s = d.evaluate(b);
            return if self.small && qnoise_armed() { perturb(s, b.key(), qnoise_sigma()) } else { s };
        }
        if let Some(d) = self.deep.as_mut() {
            let s = d.evaluate(b);
            return if self.small && qnoise_armed() { perturb(s, b.key(), qnoise_sigma()) } else { s };
        }
        self.quad_eval(b)
    }

    fn observe(&mut self, b: &Board, predicted: Score, actual: Score, is_q: bool) {
        let h = match &self.adapt { Some(h) => h.clone(), None => return };
        let Ok(mut st) = h.try_lock() else { return; };
        if st.params.disabled() { return; }
        if is_q && !matches!(st.params.scope, crate::adaptive::Scope::All) { return; }
        if crate::eval::is_mate_score(actual) || crate::eval::is_mate_score(predicted) { return; }
        let clip = st.params.clip;
        let err = crate::adaptive::clipped_err(predicted, actual, clip);
        if err == 0.0 { return; }
        let mut f = [0u16; MAX_FEAT];
        let n = QuadNet::features(b, &mut f);
        st.observe(&f, n, err, is_q);
    }

    fn push(&mut self, b: &Board) {
        if let Some(d) = self.wdl.as_mut() {
            d.push(b);
        }
        if let Some(d) = self.deep.as_mut() {
            d.push(b);
        }
    }

    fn pop(&mut self) {
        if let Some(d) = self.wdl.as_mut() {
            d.pop();
        }
        if let Some(d) = self.deep.as_mut() {
            d.pop();
        }
    }

    /// The tier switch the search sets once per node. `push`/`pop` deliberately
    /// ignore it: the big net's accumulator stays live through quiescence, so
    /// the main search above and below never rebuilds.
    fn set_small(&mut self, small: bool) {
        self.small = small;
    }
}
