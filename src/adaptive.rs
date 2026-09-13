//! Runtime adaptation of the quadratic eval.
//!
//! Idea: `eval = x^T W x` with learned `W`. After a search returns a
//! backed-up value `v` for a position previously predicted as `p`, we step
//! `W` a little along the gradient of `(v-p)^2`.
//!
//! `x` is binary occupancy (PST) with at most 32 active features out of 768.
//! Gradient is `err * x x^T` (diagonal = 1, off-diag = 2*). Steps are applied
//! to float delta tables overlaid on the quantised base, so small `eta` is
//! observable. The delta persists *across moves in the same game* (fit to the
//! game), but is cleared on `ucinewgame` / `Player::new_game`.

use crate::eval::Score;
use crate::qeval::NFEAT;
use std::sync::{Arc, Mutex, OnceLock};

// ---------------------------------------------------------------- config

#[derive(Clone, Copy, Debug)]
pub enum Target {
    Diag,
    Off,
    Both,
}
#[derive(Clone, Copy, Debug)]
pub enum Scope {
    Interior,       // only negamax interior nodes
    All,            // interior + qsearch stand-pat
}

#[derive(Clone, Debug)]
pub struct AdaptiveParams {
    pub eta: f32,          // base lr in cp units, e.g. 0.002
    pub normalized: bool,   // divide by n^2
    pub target: Target,
    pub scope: Scope,
    pub clip: i32,         // max |err| used
    pub decay: f32,        // 1.0 = no decay, 0.999 = leak
    pub q_eta_scale: f32,  // multiply eta in qsearch
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self { eta: 0.0, normalized: false, target: Target::Both, scope: Scope::Interior, clip: 300, decay: 1.0, q_eta_scale: 1.0 }
    }
}

impl AdaptiveParams {
    pub fn disabled(&self) -> bool { self.eta == 0.0 }

    pub fn from_spec(s: &str) -> Result<Self, String> {
        // spec like "eta=0.002,target=both,scope=interior,clip=300,decay=1,norm=0,qscale=0.5"
        let mut p = Self { eta: 0.002, ..Default::default() };
        if s.trim().is_empty() || s.trim() == "0" || s.trim() == "off" { p.eta = 0.0; return Ok(p); }
        for kv in s.split(',') {
            let kv = kv.trim();
            if kv.is_empty() { continue; }
            if kv.starts_with("eta=") { p.eta = kv[4..].parse::<f32>().map_err(|e| format!("eta: {e}"))?; }
            else if kv.starts_with("clip=") { p.clip = kv[5..].parse().map_err(|e| format!("clip: {e}"))?; }
            else if kv.starts_with("decay=") { p.decay = kv[6..].parse::<f32>().map_err(|e| format!("decay: {e}"))?; }
            else if kv.starts_with("norm=") { p.normalized = kv[5..].parse::<u8>().map(|v| v!=0).unwrap_or(&kv[5..]=="1"||&kv[5..]=="true"); }
            else if kv.starts_with("target=") {
                p.target = match &kv[7..] { "diag"=>Target::Diag, "off"=>Target::Off, "both"=>Target::Both, x=> return Err(format!("target {x}")) };
            } else if kv.starts_with("scope=") {
                p.scope = match &kv[6..] { "interior"=>Scope::Interior, "all"=>Scope::All, x=> return Err(format!("scope {x}")) };
            } else if kv.starts_with("qscale=") { p.q_eta_scale = kv[7..].parse::<f32>().map_err(|e| format!("qscale: {e}"))?; }
            else { return Err(format!("unknown adaptive key '{kv}'")); }
        }
        Ok(p)
    }
}

// ---------------------------------------------------------------- shared state per game

pub struct AdaptiveState {
    pub params: AdaptiveParams,
    // delta_diag in cp-equivalent float (added directly to linear term before quant scale)
    // For simplicity we keep everything in "cp" space and add as float to final score.
    diag: Vec<f32>,      // NFEAT
    off: Vec<f32>,       // NFEAT*NFEAT, only upper used but keep full for cache locality
    updates: u64,
}

impl AdaptiveState {
    pub fn new(params: AdaptiveParams) -> Self {
        Self { params, diag: vec![0.0; NFEAT], off: vec![0.0; NFEAT * NFEAT], updates: 0 }
    }
    pub fn clear(&mut self) { self.diag.fill(0.0); self.off.fill(0.0); self.updates = 0; }

    #[inline]
    pub fn eval_correction(&self, f: &[u16], n: usize) -> f32 {
        if self.params.disabled() { return 0.0; }
        let mut s = 0.0f32;
        let target = self.params.target;
        if matches!(target, Target::Diag | Target::Both) {
            for i in 0..n { s += self.diag[f[i] as usize]; }
        }
        if matches!(target, Target::Off | Target::Both) {
            // sum_{i<j} 2* off[fi][fj]  — off is symmetric, we store both triangles equal
            // so we can just sum i<j once*2
            for i in 0..n {
                let row = f[i] as usize * NFEAT;
                for j in (i+1)..n {
                    let fj = f[j] as usize;
                    s += 2.0 * self.off[row + fj];
                }
            }
        }
        s
    }

    #[inline]
    pub fn observe(&mut self, f: &[u16], n: usize, err_clipped: f32, is_q: bool) {
        if self.params.disabled() { return; }
        let mut eta = self.params.eta * if is_q { self.params.q_eta_scale } else { 1.0 };
        if self.params.normalized {
            let nn = (n * n).max(1) as f32;
            eta /= nn;
        }
        if eta == 0.0 { return; }
        // decay
        if self.params.decay != 1.0 {
            let d = self.params.decay;
            for v in &mut self.diag { *v *= d; }
            for v in &mut self.off { *v *= d; }
        }
        let step = eta * err_clipped;
        match self.params.target {
            Target::Diag => {
                for i in 0..n { let idx = f[i] as usize; self.diag[idx] += step; }
            }
            Target::Off => {
                for i in 0..n {
                    let fi = f[i] as usize;
                    for j in (i+1)..n {
                        let fj = f[j] as usize;
                        self.off[fi * NFEAT + fj] += step;
                        self.off[fj * NFEAT + fi] += step; // keep symmetric
                    }
                }
            }
            Target::Both => {
                for i in 0..n { let idx = f[i] as usize; self.diag[idx] += step; }
                for i in 0..n {
                    let fi = f[i] as usize;
                    for j in (i+1)..n {
                        let fj = f[j] as usize;
                        self.off[fi * NFEAT + fj] += step;
                        self.off[fj * NFEAT + fi] += step;
                    }
                }
            }
        }
        self.updates += 1;
    }
    pub fn stats(&self) -> (u64, f32) {
        let max_abs = self.diag.iter().chain(self.off.iter()).map(|v| v.abs()).fold(0.0f32, f32::max);
        (self.updates, max_abs)
    }
}

// Global per-process adaptive state shared by UCI engine threads.
// For internal matchplay we use per-Player state (see matchplay.rs);
// this global is for the subprocess/UCI path where the engine lives
// in a single process and games are sequential (ucinewgame clears).
static GLOBAL: OnceLock<Arc<Mutex<AdaptiveState>>> = OnceLock::new();

fn global() -> Arc<Mutex<AdaptiveState>> {
    GLOBAL.get_or_init(|| Arc::new(Mutex::new(AdaptiveState::new(AdaptiveParams::default())))).clone()
}

pub fn set_global_params(p: AdaptiveParams) {
    if let Ok(mut g) = global().lock() {
        g.params = p;
        g.clear();
    }
}
pub fn clear_global() { if let Ok(mut g) = global().lock() { g.clear(); } }
pub fn global_handle() -> Arc<Mutex<AdaptiveState>> { global() }

// Init from env / CLI at startup. Call from main/lib init.
pub fn init_from_env() {
    // 1) CHESS_ADAPTIVE env var, e.g. "eta=0.002"
    if let Ok(spec) = std::env::var("CHESS_ADAPTIVE") {
        if let Ok(p) = AdaptiveParams::from_spec(&spec) { set_global_params(p); }
    }
    // 2) --adaptive <spec> CLI args (takes precedence)
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--adaptive" { if let Some(spec) = args.get(i+1) { if let Ok(p) = AdaptiveParams::from_spec(spec) { set_global_params(p); break; } } }
        if args[i].starts_with("--adaptive=") { if let Ok(p) = AdaptiveParams::from_spec(&args[i]["--adaptive=".len()..]) { set_global_params(p); break; } }
    }
}

// Helper to compute clipped err as f32
#[inline]
pub fn clipped_err(predicted: Score, actual: Score, clip: i32) -> f32 {
    let e = (actual - predicted) as f32;
    e.clamp(-(clip as f32), clip as f32)
}

// Expose for matchplay
pub fn parse_spec_or_default(s: Option<&str>) -> AdaptiveParams {
    match s {
        Some(spec) => AdaptiveParams::from_spec(spec).unwrap_or_default(),
        None => AdaptiveParams::default(),
    }
}
