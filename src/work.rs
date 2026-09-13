//! Cost, in a unit that does not depend on what else the machine is doing.
//!
//! # Why this exists
//!
//! Two instruments already price the search and neither can answer the
//! question a tuning loop actually asks.
//!
//! * **The wall clock** is the truth, and on this box it swings 2-6% under
//!   desktop load and is *biased* -19% under contention (LEDGER 039). It needs
//!   interleaved pinned A/B pairs to resolve 1.5%, which is minutes per
//!   candidate. A search that scores 14000 positions cannot pay that.
//! * **The node count** is deterministic and free, and is not a unit of work:
//!   a quiescence node costs 0.275 main-search nodes (`library/007`). Anything
//!   scored at fixed nodes hands a candidate that shifts effort into
//!   quiescence a discount it did not earn -- and quiescence is already 44.6%
//!   of search time (LEDGER 059).
//!
//! So: **count the operations, and price them from a frozen table.**
//! `nodeprof` already brackets every non-recursive leaf operation inside
//! `negamax` and `quiescence`, and its `calls[]` array is exact rather than
//! sampled. The counts are deterministic; only the timing is noisy. This
//! module separates them. A `work` build increments a counter per bracketed
//! call and reads no clock at all, and the cost is
//!
//! ```text
//! work_ns = sum_z  calls[z] * PRICE_NS[z]  +  nodes * CONTROL_NS_PER_NODE
//! ```
//!
//! Same binary, same position, same answer, every time, under any load. That
//! is the property the wall clock cannot have and the node count buys by
//! lying.
//!
//! # What this cannot see, stated before it bites
//!
//! **It prices calls, not cycles.** Every operation of a given kind costs the
//! table's number whatever the machine really did, so the model is blind to
//! cache behaviour, memory bandwidth, branch prediction and instruction-level
//! parallelism. The consequence is concrete: the i16 accumulator (the largest
//! measured lever in the engine, ~10% of search) performs *exactly the same
//! number of pushes* and would read as **zero change** here, because its whole
//! win is moving 2048 bytes instead of 4096.
//!
//! So there are two axes and they need two instruments:
//!
//! | change | instrument |
//! |---|---|
//! | **behaviour** -- pruning, ordering, extensions, the price list | this model |
//! | **implementation** -- kernels, layout, i16, quantisation | interleaved pinned wall-clock A/B, and re-calibrate the table afterwards |
//!
//! Crossing them is the way to get a confident wrong answer.
//!
//! # How good is it
//!
//! `control` -- the search's own control flow, plus the five zones that
//! measure at or below the instrument's 10 ns floor -- is 10.5% of the WDL
//! search and is modelled as a flat per-node constant. Across the three nets
//! on record it varies 112.5-128.7 ns/node, about 14%, so the *absolute*
//! total carries roughly +/-1.5% of systematic error.
//!
//! That error is a **bias, not noise**: it is the same every run, so it very
//! largely cancels in a difference between two candidates with a similar
//! operation mix, which is the only way this number is meant to be read. Use
//! it for A-minus-B. Do not quote an absolute work figure as a speed.
//!
//! The one thing that breaks the cancellation is a candidate that moves the
//! main:q node ratio a long way, because the control term is calibrated at the
//! mix the three reference workloads share (1:1.61, and they agree closely
//! enough that a separate main and q term is not identifiable from them).
//! `Model::mix_warning` reports the ratio for exactly this reason.

#[cfg(feature = "prof")]
use crate::nodeprof::{Tally, N, NAMES, SITE};

/// A price list: what each zone costs, in nanoseconds per call.
///
/// Net-dependent by construction -- `eval` is 908 ns for the width-512 deploy
/// net and less for a narrower one -- so a table belongs to a configuration,
/// not to the engine. Changing the net means re-emitting it.
#[derive(Clone)]
pub struct Prices {
    /// Nanoseconds per call, indexed by zone. **0.0 means "not priced"**, not
    /// "free": those zones measured at or below the profiler's 10 ns probe
    /// floor and their real cost is inside `control`. Pricing them at a guess
    /// would move nanoseconds from a measured bucket to an invented one.
    pub ns: Vec<f64>,
    /// The unattributed remainder, per node. See the module docs.
    pub control_per_node: f64,
    /// Where this table came from. Printed with every report, because a price
    /// list without its conditions is the same kind of rumour as an Elo
    /// without its time control.
    pub provenance: String,
}

/// The table for the deploy configuration.
///
/// Measured on `chess nodeprof --emit-prices --wdl nnue/nets/gate-3ep-q8.nnue`,
/// mean of 5 runs, `taskset -c 5`, Ryzen 5 5500 (Zen 3, no VNNI), TSC 3.593
/// GHz, 2026-09-07, unified-`search` tree with `c_see=4000` (bench 227236).
/// Re-emitted because `c_see` put a `see()` call inside the main `price` zone:
/// 4.29 -> 37.44 ns, so every work number priced off the old table was ~3%
/// low on main-search-heavy mixes.
///
/// The counts underlying it were **bit-identical across all 5 runs**; the
/// spread quoted per zone is in the price alone. Zones contributing under 0.1%
/// of the total are noisy (`history` 26%, `observe` 117%) and it does not
/// matter -- the two that carry the search, `eval` and `acc push`, are stable
/// to 1-2.5% and ~3.5%.
pub const DEPLOY_NS: [f64; 21] = [
    0.00,    // 0  draw       m   <probe floor
    1.05,    // 1  tt probe   m   +/-18%
    916.41,  // 2  eval       m   +/-1.1%
    129.05,  // 3  movegen    m   +/-2.9%
    54.48,   // 4  order      m   +/-2.8%
    37.44,   // 5  price      m   +/-4.0%   (was 4.29 before c_see's see() call)
    14.48,   // 6  make       m   +/-7.7%
    263.12,  // 7  acc push   m   +/-3.5%
    0.00,    // 8  acc pop    m   <probe floor
    0.00,    // 9  prefetch   m   <probe floor
    3.03,    // 10 tt store   m   +/-14.7%
    2.73,    // 11 history    m   +/-26%
    0.47,    // 12 observe    m   +/-117%
    929.53,  // 13 eval       q   +/-2.4%
    89.31,   // 14 movegen    q   +/-5.1%
    4.06,    // 15 order      q   +/-12.9%
    6.07,    // 16 delta+SEE  q   +/-6.4%
    12.65,   // 17 make       q   +/-12.2%
    288.68,  // 18 acc push   q   +/-3.9%
    0.00,    // 19 acc pop    q   <probe floor
    0.00,    // 20 observe    q   <probe floor
];

/// `control` for the deploy configuration: 136.84 ns/node, spread 1.2% over
/// the same 5 runs, ~10% of the search.
pub const DEPLOY_CONTROL_NS: f64 = 136.84;

pub const DEPLOY_PROVENANCE: &str =
    "gate-3ep-q8 (width 512), Ryzen 5 5500, nodeprof x5 pinned, 2026-09-07, c_see=4000 tree";

impl Prices {
    pub fn deploy() -> Prices {
        Prices {
            ns: DEPLOY_NS.to_vec(),
            control_per_node: DEPLOY_CONTROL_NS,
            provenance: DEPLOY_PROVENANCE.to_string(),
        }
    }

    /// Parse a table emitted by `chess nodeprof --emit-prices`. The generator
    /// and the consumer share a format so a table is never transcribed by
    /// hand, which is how a price list acquires a typo nobody can see.
    pub fn parse(text: &str, provenance: &str) -> Result<Prices, String> {
        let mut ns = vec![0.0; 21];
        let mut control = None;
        let mut seen = 0usize;
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let t: Vec<&str> = line.split_whitespace().collect();
            match t.first() {
                Some(&"price") => {
                    // price <index> <name...> <site> <calls> <ns>
                    if t.len() < 6 {
                        return Err(format!("short price line: {line}"));
                    }
                    let i: usize =
                        t[1].parse().map_err(|_| format!("bad zone index: {}", t[1]))?;
                    if i >= ns.len() {
                        return Err(format!("zone index {i} out of range"));
                    }
                    ns[i] = t[t.len() - 1]
                        .parse()
                        .map_err(|_| format!("bad price: {}", t[t.len() - 1]))?;
                    seen += 1;
                }
                Some(&"control") => {
                    control = Some(
                        t.get(1)
                            .ok_or("control line has no value")?
                            .parse::<f64>()
                            .map_err(|_| "bad control value".to_string())?,
                    );
                }
                _ => {}
            }
        }
        if seen != 21 {
            return Err(format!("expected 21 price lines, got {seen}"));
        }
        Ok(Prices {
            ns,
            control_per_node: control.ok_or("no control line")?,
            provenance: provenance.to_string(),
        })
    }
}

/// What a search cost, in modelled nanoseconds.
pub struct Model {
    /// Per-zone modelled cost, ns.
    pub zone_ns: Vec<f64>,
    pub control_ns: f64,
    pub total_ns: f64,
    pub calls: Vec<u64>,
    /// `[main, q]`.
    pub nodes: [u64; 2],
}

impl Model {
    /// Price a tally. This is the whole model.
    #[cfg(feature = "prof")]
    pub fn of(t: &Tally, p: &Prices) -> Model {
        let mut zone_ns = vec![0.0; N];
        for i in 0..N {
            zone_ns[i] = t.calls[i] as f64 * p.ns[i];
        }
        let nodes = t.nodes[0] + t.nodes[1];
        let control_ns = nodes as f64 * p.control_per_node;
        Model {
            total_ns: zone_ns.iter().sum::<f64>() + control_ns,
            zone_ns,
            control_ns,
            calls: t.calls.to_vec(),
            nodes: t.nodes,
        }
    }

    /// The main:q node ratio, which the control term is calibrated at. The
    /// three reference workloads all sit at 1:1.61; a candidate far from that
    /// is outside the calibration and its absolute total should not be
    /// compared with one that is not.
    pub fn mix(&self) -> f64 {
        self.nodes[1] as f64 / self.nodes[0].max(1) as f64
    }

    /// Non-empty when the mix has moved far enough to matter.
    pub fn mix_warning(&self) -> Option<String> {
        let m = self.mix();
        (!(1.35..=1.90).contains(&m)).then(|| {
            format!(
                "main:q mix is 1:{m:.2}, outside the 1:1.35-1.90 the control term was \
                 calibrated over -- the absolute total is off-model; the difference \
                 against a candidate at a similar mix is still fine"
            )
        })
    }

    #[cfg(feature = "prof")]
    pub fn report(&self, p: &Prices) {
        let nodes = self.nodes[0] + self.nodes[1];
        println!("prices: {}", p.provenance);
        println!(
            "{} nodes ({} main, {} q, 1:{:.2})",
            nodes,
            self.nodes[0],
            self.nodes[1],
            self.mix()
        );
        println!(
            "\n{:<10}{:>6}{:>12}{:>11}{:>11}{:>9}",
            "zone", "site", "calls", "ns/call", "total ms", "share"
        );
        let mut order: Vec<usize> = (0..N).filter(|&i| self.calls[i] > 0).collect();
        order.sort_by(|&a, &b| self.zone_ns[b].total_cmp(&self.zone_ns[a]));
        for i in order {
            let site = if SITE[i] == 0 { "m" } else { "q" };
            if p.ns[i] == 0.0 {
                println!(
                    "{:<10}{:>6}{:>12}{:>11}{:>11}{:>9}",
                    NAMES[i], site, self.calls[i], "<floor", "-", "-"
                );
                continue;
            }
            println!(
                "{:<10}{:>6}{:>12}{:>11.2}{:>11.1}{:>8.1}%",
                NAMES[i],
                site,
                self.calls[i],
                p.ns[i],
                self.zone_ns[i] / 1e6,
                100.0 * self.zone_ns[i] / self.total_ns,
            );
        }
        println!("{:-<59}", "");
        println!(
            "{:<10}{:>6}{:>12}{:>11.2}{:>11.1}{:>8.1}%",
            "control",
            "",
            "",
            p.control_per_node,
            self.control_ns / 1e6,
            100.0 * self.control_ns / self.total_ns,
        );
        println!(
            "{:<10}{:>6}{:>12}{:>11.1}{:>11.1}{:>8.1}%",
            "work",
            "",
            nodes,
            self.total_ns / nodes.max(1) as f64,
            self.total_ns / 1e6,
            100.0,
        );
        if let Some(w) = self.mix_warning() {
            println!("\nWARNING: {w}");
        }
    }
}

/// A work unit. One **main-search node at the deploy configuration**, so the
/// number reads on the same scale the node count does and the two can be
/// compared directly -- which is the point, since the gap between them is
/// exactly the discount a fixed-node budget was handing out.
///
/// Deliberately a round *measured* number and not a definition: it is
/// `nodeprof`'s ns/node for the deploy net, so "15000 work units" and "15000
/// nodes" cost the same wall time at the operating point LEDGER 014 selected,
/// and a ladder measured in one can be read against a ladder measured in the
/// other.
pub const NS_PER_UNIT: f64 = 1232.3;

pub fn units(total_ns: f64) -> f64 {
    total_ns / NS_PER_UNIT
}

/// The frozen deploy table as integer picoseconds, for `Tally::default`.
///
/// Rounded the same way `Prices::set_prices` rounds, so an explicitly
/// installed copy of the deploy table is bit-identical to the default one and
/// a work budget does not move by a picosecond depending on which path set it.
#[cfg(feature = "prof")]
pub fn deploy_price_ps() -> [u64; crate::nodeprof::N] {
    let mut out = [0u64; crate::nodeprof::N];
    for (o, ns) in out.iter_mut().zip(DEPLOY_NS.iter()) {
        *o = (ns * 1000.0).round() as u64;
    }
    out
}

#[cfg(feature = "prof")]
pub fn deploy_control_ps() -> u64 {
    (DEPLOY_CONTROL_NS * 1000.0).round() as u64
}
