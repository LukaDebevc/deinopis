//! Match statistics: Elo with error bars, LOS, and a sequential test.
//!
//! Why this exists rather than "it won more games": at the effect sizes an
//! engine change produces, a fixed-length match is either far too long or
//! completely uninformative, and the difference is not visible by eye. A 100-
//! game match that finishes +10 has a 95% interval of roughly ±70 Elo — it
//! cannot distinguish a good patch from a bad one. The point of this project
//! is structural changes big enough to resolve quickly; this module is what
//! decides when "quickly" has happened.
//!
//! **Pairs, not games.** Games are played two at a time from the same opening
//! with colours reversed, and the *pair* is the sampling unit. Colour and
//! opening are then differenced out of the variance rather than contributing
//! to it, which is worth roughly a 25-30% reduction in games to a decision for
//! free. This is the pentanomial model fishtest uses; a trinomial (per-game)
//! model on paired games understates the correlation and gives error bars that
//! are too narrow.

/// Expected score for a player `elo` points stronger, under the logistic model
/// (the one Elo differences are *defined* by, and the one every engine testing
/// framework uses; FIDE's tables are a piecewise fit to something else).
pub fn elo_to_score(elo: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-elo / 400.0))
}

/// Inverse of `elo_to_score`. Saturates rather than returning infinity so a
/// 100% score in a short match prints a large number instead of `inf`.
pub fn score_to_elo(score: f64) -> f64 {
    let s = score.clamp(1e-6, 1.0 - 1e-6);
    -400.0 * (1.0 / s - 1.0).log10()
}

/// Standard normal CDF, via `erf` from a rational approximation
/// (Abramowitz & Stegun 7.1.26; |error| < 1.5e-7, far below what a few hundred
/// games can resolve).
pub fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Outcome of one colour-reversed game pair, as the number of half-points the
/// engine under test scored out of two: 0, 0.5, 1, 1.5 or 2.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PairResult {
    LossLoss,
    LossDraw,
    Balanced,
    WinDraw,
    WinWin,
}

impl PairResult {
    pub fn from_points(points: f64) -> PairResult {
        match (points * 2.0).round() as i32 {
            0 => PairResult::LossLoss,
            1 => PairResult::LossDraw,
            2 => PairResult::Balanced,
            3 => PairResult::WinDraw,
            _ => PairResult::WinWin,
        }
    }

    /// Score rate for the pair, in [0, 1].
    pub fn rate(self) -> f64 {
        match self {
            PairResult::LossLoss => 0.0,
            PairResult::LossDraw => 0.25,
            PairResult::Balanced => 0.5,
            PairResult::WinDraw => 0.75,
            PairResult::WinWin => 1.0,
        }
    }
}

/// Running tally of a match. Games are counted trinomially for reporting
/// (people read W-L-D), pairs pentanomially for the statistics.
#[derive(Clone, Default, Debug)]
pub struct MatchStats {
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    /// Counts of each pair outcome, indexed by `PairResult` order.
    pub pairs: [u32; 5],
}

impl MatchStats {
    pub fn add_game(&mut self, points: f64) {
        if points > 0.75 {
            self.wins += 1;
        } else if points < 0.25 {
            self.losses += 1;
        } else {
            self.draws += 1;
        }
    }

    pub fn add_pair(&mut self, p: PairResult) {
        self.pairs[p as usize] += 1;
    }

    /// Merge per-file statistics. Pairs are scoped to one match by
    /// construction (`stats_from_games`), so cross-file totals add — but the
    /// game lists themselves must never be pooled before pairing.
    pub fn merge(&mut self, other: &MatchStats) {
        self.wins += other.wins;
        self.losses += other.losses;
        self.draws += other.draws;
        for (a, b) in self.pairs.iter_mut().zip(other.pairs.iter()) {
            *a += *b;
        }
    }

    pub fn games(&self) -> u32 {
        self.wins + self.losses + self.draws
    }

    pub fn pair_count(&self) -> u32 {
        self.pairs.iter().sum()
    }

    /// Rates the five pair outcomes correspond to.
    const RATES: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

    /// Mean pair score rate and its variance. `None` until there are at least
    /// two pairs, since one pair has no sample variance.
    ///
    /// The mean is the raw sample mean — an anchored rating must not be shrunk
    /// towards the opponent's. The **variance** is regularised: see
    /// `smoothed_variance`.
    pub fn moments(&self) -> Option<(f64, f64)> {
        let n = self.pair_count();
        if n < 2 {
            return None;
        }
        let n = n as f64;
        let mean: f64 = self
            .pairs
            .iter()
            .zip(Self::RATES)
            .map(|(&c, r)| c as f64 * r)
            .sum::<f64>()
            / n;
        Some((mean, self.smoothed_variance()))
    }

    /// Per-pair variance with one pseudo-pair added to each of the five
    /// outcomes — a Dirichlet(1) prior on the pair distribution.
    ///
    /// The raw sample variance is zero for a match that is all draws, and the
    /// normal approximation then claims infinite precision: an SPRT would
    /// decide instantly and the Elo interval would collapse to a point. A hard
    /// floor fixes that but makes the strength of the conclusion depend on an
    /// arbitrary constant. Five pseudo-pairs is the standard correction, costs
    /// about 2% of the variance estimate at 200 real pairs, and degrades
    /// smoothly instead of switching on at a threshold.
    fn smoothed_variance(&self) -> f64 {
        let n = self.pair_count() as f64 + 5.0;
        let mean: f64 = self
            .pairs
            .iter()
            .zip(Self::RATES)
            .map(|(&c, r)| (c as f64 + 1.0) * r)
            .sum::<f64>()
            / n;
        self.pairs
            .iter()
            .zip(Self::RATES)
            .map(|(&c, r)| (c as f64 + 1.0) * (r - mean).powi(2))
            .sum::<f64>()
            / n
    }

    /// Standard error of the mean pair score. Divided by the number of *real*
    /// pairs, not the smoothed count, which is the conservative choice.
    pub fn stderr(&self) -> Option<f64> {
        let (_, var) = self.moments()?;
        Some((var / self.pair_count() as f64).sqrt())
    }

    /// Elo point estimate and the half-width of its 95% interval, in Elo.
    /// The interval is computed on the score scale and mapped through
    /// `score_to_elo`, so it is asymmetric near the extremes — as it should be.
    pub fn elo(&self) -> Option<(f64, f64, f64)> {
        let (mean, _) = self.moments()?;
        let se = self.stderr()?;
        let lo = score_to_elo(mean - 1.96 * se);
        let hi = score_to_elo(mean + 1.96 * se);
        Some((score_to_elo(mean), lo, hi))
    }

    /// Likelihood of superiority: P(true Elo > 0) under the normal
    /// approximation. This is the number that answers "is this actually
    /// better", and it is not the same as "it scored above 50%".
    pub fn los(&self) -> Option<f64> {
        let (mean, _) = self.moments()?;
        let se = self.stderr()?;
        Some(phi((mean - 0.5) / se))
    }
}

/// A sequential probability ratio test on Elo bounds.
///
/// H0: the change is worth `elo0`; H1: it is worth `elo1`. The test stops as
/// soon as the evidence is decisive at the given error rates, which for a real
/// effect is typically 2-5x fewer games than a fixed-length match of the same
/// power. `alpha` is the false-accept rate for H1, `beta` for H0.
#[derive(Clone, Copy, Debug)]
pub struct Sprt {
    pub elo0: f64,
    pub elo1: f64,
    pub alpha: f64,
    pub beta: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SprtVerdict {
    /// LLR crossed the upper bound: the change is at least `elo1`.
    AcceptH1,
    /// LLR crossed the lower bound: the change is not worth `elo0`.
    AcceptH0,
    Continue,
}

impl Sprt {
    pub fn new(elo0: f64, elo1: f64) -> Sprt {
        Sprt { elo0, elo1, alpha: 0.05, beta: 0.05 }
    }

    pub fn bounds(&self) -> (f64, f64) {
        (
            (self.beta / (1.0 - self.alpha)).ln(),
            ((1.0 - self.beta) / self.alpha).ln(),
        )
    }

    /// Log-likelihood ratio under a normal approximation to the pair-score
    /// distribution. For a normal with known variance this is exact:
    /// `n (mu1 - mu0)(2 xbar - mu0 - mu1) / 2 sigma^2`. The variance is
    /// estimated from the sample, which is the usual (and mild) approximation
    /// — it is the same one fishtest makes.
    pub fn llr(&self, stats: &MatchStats) -> Option<f64> {
        let (mean, var) = stats.moments()?;
        let n = stats.pair_count() as f64;
        let s0 = elo_to_score(self.elo0);
        let s1 = elo_to_score(self.elo1);
        Some(n * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * var))
    }

    pub fn verdict(&self, stats: &MatchStats) -> SprtVerdict {
        let (lower, upper) = self.bounds();
        match self.llr(stats) {
            Some(l) if l >= upper => SprtVerdict::AcceptH1,
            Some(l) if l <= lower => SprtVerdict::AcceptH0,
            _ => SprtVerdict::Continue,
        }
    }
}

/// One-line summary, the format the match runner prints as it goes.
pub fn summary(stats: &MatchStats, sprt: Option<&Sprt>) -> String {
    let mut s = format!(
        "{} games: +{} ={} -{}",
        stats.games(),
        stats.wins,
        stats.draws,
        stats.losses
    );
    if let Some((elo, lo, hi)) = stats.elo() {
        s.push_str(&format!("   {elo:+.1} Elo  [{lo:+.0}, {hi:+.0}]"));
    }
    if let Some(los) = stats.los() {
        s.push_str(&format!("   LOS {:.1}%", los * 100.0));
    }
    if let (Some(t), Some(l)) = (sprt, sprt.and_then(|t| t.llr(stats))) {
        let (lo, hi) = t.bounds();
        s.push_str(&format!("   LLR {l:.2} [{lo:.2}, {hi:.2}]"));
    }
    s
}

// ---------------------------------------------------------------- anchored gauntlet combination
//
// A match measures a *difference*. Turning per-opponent differences into one
// rating needs the opponents' known ratings (CCRL Blitz here): each anchor
// implies our rating as `ccrl + diff`, and the anchors combine by
// inverse-variance weight. This is the same estimator `tools/ladder.sh` used
// to carry inline — it lives here now so the number `chess elo` prints and
// the number the ladder prints cannot disagree, and so `cargo test` covers it.
//
// Errors on both halves count: our own interval on the difference and CCRL's
// published interval on the anchor, added in quadrature. CCRL publishes 95%
// intervals, as do we, so both are divided by 1.96 to standard errors first.
//
// When the anchors disagree by more than their error bars (chi2/dof > 1) the
// excess is real inconsistency in the anchoring — usually a style mismatch
// against one opponent — so the error is scaled by sqrt(chi2/dof), the way
// the PDG scales disagreeing measurements. Reported always, not past a
// threshold, because the unscaled number is the one that gets copied into a
// document and believed. It does NOT cover the systematic offset between our
// conditions and CCRL's (book, tablebases, time control, pool).

/// One gauntlet leg: our difference against a rated opponent.
#[derive(Clone, Debug)]
pub struct Anchor {
    pub key: String,
    pub ccrl: f64,
    /// Our Elo difference vs this opponent, with its 95% interval.
    pub diff: f64,
    pub lo: f64,
    pub hi: f64,
    pub games: u32,
    pub wins: u32,
    pub draws: u32,
}

impl Anchor {
    /// Standard error of our difference, symmetrising the interval the way
    /// the ladder always has (intervals are near-symmetric except at extremes).
    pub fn diff_se(&self) -> f64 {
        ((self.hi - self.diff).max(self.diff - self.lo)) / 1.96
    }

    /// The rating this anchor implies, and its standard error: CCRL's 95%
    /// interval converted to a standard error, added in quadrature.
    pub fn implied(&self, ccrl_err95: f64) -> (f64, f64) {
        let sd = (self.diff_se().powi(2) + (ccrl_err95 / 1.96).powi(2)).sqrt();
        (self.ccrl + self.diff, sd)
    }

    pub fn score_pct(&self) -> f64 {
        (self.wins as f64 + self.draws as f64 / 2.0) / self.games.max(1) as f64 * 100.0
    }
}

#[derive(Clone, Debug)]
pub struct Combined {
    pub mean: f64,
    /// 95% half-width, scaled for anchor spread — the number to quote.
    pub half_width: f64,
    /// 95% half-width without the spread scaling, for the record.
    pub half_width_unscaled: f64,
    pub chi2_dof: f64,
    pub scale: f64,
    pub n: usize,
}

/// Inverse-variance weighted mean of the implied ratings. `ccrl_errs` carries
/// each anchor's published 95% interval, positionally. `None` on empty input.
pub fn combine_anchors(anchors: &[Anchor], ccrl_errs: &[f64]) -> Option<Combined> {
    if anchors.is_empty() || anchors.len() != ccrl_errs.len() {
        return None;
    }
    let implied: Vec<(f64, f64)> =
        anchors.iter().zip(ccrl_errs.iter()).map(|(a, e)| a.implied(*e)).collect();
    let wsum: f64 = implied.iter().map(|&(_, s)| 1.0 / (s * s)).sum();
    let mean: f64 = implied.iter().map(|&(r, s)| r / (s * s)).sum::<f64>() / wsum;
    let se = (1.0 / wsum).sqrt();
    let chi2: f64 = implied.iter().map(|&(r, s)| ((r - mean) / s).powi(2)).sum();
    let dof = (implied.len() as f64 - 1.0).max(1.0);
    let scale = (chi2 / dof).max(1.0).sqrt();
    Some(Combined {
        mean,
        half_width: 1.96 * se * scale,
        half_width_unscaled: 1.96 * se,
        chi2_dof: chi2 / dof,
        scale,
        n: implied.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_all_drawn_match_does_not_claim_certainty() {
        let mut all_draws = MatchStats::default();
        for _ in 0..200 {
            all_draws.add_pair(PairResult::Balanced);
            all_draws.add_game(0.5);
            all_draws.add_game(0.5);
        }
        let (elo, lo, hi) = all_draws.elo().unwrap();
        assert!(elo.abs() < 1e-9);
        // Zero observed variance must not become zero uncertainty...
        assert!(hi - lo > 2.0, "interval collapsed to {:.3} Elo", hi - lo);
        assert!(hi - lo < 30.0, "interval implausibly wide: {:.1}", hi - lo);
        // ...but 400 drawn games *are* evidence against "+10 Elo", and the
        // test should say so rather than run forever.
        assert_eq!(Sprt::new(0.0, 10.0).verdict(&all_draws), SprtVerdict::AcceptH0);
    }

    #[test]
    fn smoothing_is_negligible_on_a_real_sized_match() {        let mut noisy = MatchStats::default();
        for i in 0..200 {
            noisy.add_pair(if i % 2 == 0 { PairResult::WinWin } else { PairResult::LossLoss });
        }
        let (mean, var) = noisy.moments().unwrap();
        // Raw variance of an even split of 0 and 1 is exactly 0.25.
        assert!((mean - 0.5).abs() < 1e-12);
        assert!(
            (var - 0.25).abs() < 0.01,
            "smoothing moved the variance to {var}, expected ~0.25"
        );
    }

    #[test]
    fn merged_stats_add_games_and_pairs() {
        let mut a = MatchStats::default();
        a.add_game(1.0);
        a.add_game(0.0);
        a.add_pair(PairResult::Balanced);
        let mut b = MatchStats::default();
        b.add_game(0.5);
        b.add_pair(PairResult::WinWin);
        a.merge(&b);
        assert_eq!(a.games(), 3);
        assert_eq!(a.pair_count(), 2);
        assert_eq!(a.wins, 1);
        assert_eq!(a.draws, 1);
        assert_eq!(a.losses, 1);
    }

    #[test]
    fn single_anchor_combines_to_itself_unscaled() {
        let a = Anchor {
            key: "x".into(),
            ccrl: 3000.0,
            diff: 100.0,
            lo: 80.0,
            hi: 120.0,
            games: 200,
            wins: 100,
            draws: 50,
        };
        let c = combine_anchors(&[a], &[14.0]).unwrap();
        assert!((c.mean - 3100.0).abs() < 1e-9);
        assert!((c.scale - 1.0).abs() < 1e-12, "one anchor cannot disagree with itself");
        assert!((c.half_width - c.half_width_unscaled).abs() < 1e-12);
        assert_eq!(c.n, 1);
    }

    #[test]
    fn disagreeing_anchors_scale_the_error() {
        let mk = |diff: f64| Anchor {
            key: "x".into(),
            ccrl: 3000.0,
            diff,
            lo: diff - 10.0,
            hi: diff + 10.0,
            games: 400,
            wins: 150,
            draws: 100,
        };
        let agree = combine_anchors(&[mk(50.0), mk(52.0)], &[10.0, 10.0]).unwrap();
        let clash = combine_anchors(&[mk(0.0), mk(100.0)], &[10.0, 10.0]).unwrap();
        assert!(clash.scale > agree.scale, "clash {} vs agree {}", clash.scale, agree.scale);
        assert!(clash.half_width > clash.half_width_unscaled);
    }

    #[test]
    fn gauntlet_071_reproduces_the_hand_combination() {
        // `.ladder/gauntlet-20260908-191341.tsv`, quoted in LEDGER 071 as
        // 3176 ±42 at chi2/dof 3.43. If this drifts, the estimator changed.
        let legs = [
            ("byteknight", 2859.0, 17.0, 235.45, 188.82, 290.91, 200, 133, 52),
            ("4ku", 3057.0, 14.0, 130.94, 89.00, 176.91, 200, 105, 62),
            ("inanis", 3087.0, 13.0, 85.04, 46.96, 125.25, 200, 83, 82),
            ("simbelmyne", 3238.0, 12.0, -31.35, -68.05, 4.66, 200, 46, 90),
        ];
        let anchors: Vec<Anchor> = legs
            .iter()
            .map(|&(key, ccrl, _, diff, lo, hi, games, wins, draws)| Anchor {
                key: key.into(),
                ccrl,
                diff,
                lo,
                hi,
                games,
                wins,
                draws,
            })
            .collect();
        let errs: Vec<f64> = legs.iter().map(|l| l.2).collect();
        let c = combine_anchors(&anchors, &errs).unwrap();
        assert!((c.mean - 3176.0).abs() < 1.5, "mean {}", c.mean);
        assert!((c.half_width - 42.0).abs() < 1.5, "half-width {}", c.half_width);
        assert!((c.chi2_dof - 3.43).abs() < 0.05, "chi2/dof {}", c.chi2_dof);
    }
}
