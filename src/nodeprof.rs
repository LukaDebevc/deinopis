//! Where a *node* spends its time.
//!
//! `chess evalprof` prices one evaluation by taking the difference of two
//! prefixes of the same code the search runs. A node cannot be sliced that
//! way — it is a tree of calls, not a pipeline — so this instrument does the
//! other thing: it reads the timestamp counter around each **non-recursive
//! leaf operation** inside `negamax` and `quiescence` and accumulates cycles
//! per zone. No zone contains another, so the self-time is the inclusive time
//! and the parts add up. Whatever is left over is the search's own control
//! flow, and it is reported as such rather than hidden.
//!
//! Three things this has to get right, all of them already paid for once:
//!
//! * **A node is not a unit of work.** A qsearch node costs 0.275 main-search
//!   nodes (`library/007`), so every zone is tallied separately for the two
//!   sites and the per-node column is divided by that site's own node count.
//! * **Wall clock is biased −19% under contention** (LEDGER 039), not merely
//!   noisy. So `nodeprof` runs single-threaded and says so, and the counter it
//!   reads is the invariant TSC, which is a clock rather than a core-cycle
//!   count.
//! * **The instrument costs something.** Two `rdtsc`s bracket every zone, and
//!   the shortest zones here are tens of cycles. The pair cost is calibrated
//!   at startup and subtracted per call, and both the raw and the corrected
//!   number are printed so the correction can be checked rather than trusted.
//!
//! Compiled out entirely unless `--features nodeprof`. Verify that with
//! `chess bench`: the node count must be identical to the plain build, and
//! the nps must not be.

use std::arch::x86_64::_rdtsc;

/// Read the invariant TSC. Not serialising: the surrounding zone is tens to
/// thousands of cycles and out-of-order slop is a constant offset, which the
/// calibration below removes. `rdtscp`/`lfence` would triple the probe cost to
/// buy a precision no zone here needs.
#[inline(always)]
pub fn tsc() -> u64 {
    // SAFETY: `rdtsc` is unprivileged on every x86-64 that runs this engine,
    // and `constant_tsc` is required for the number to mean anything — both
    // are already assumed by `.cargo/config`'s `target-cpu=native`.
    unsafe { _rdtsc() }
}
#[cfg(feature = "prof")]

macro_rules! zones {
    ($($v:ident => $site:expr, $name:expr, $what:expr;)*) => {
        #[derive(Clone, Copy, PartialEq, Eq)]
        #[repr(usize)]
        pub enum Z { $($v),* }
        pub const NAMES: &[&str] = &[$($name),*];
        /// 0 = main search, 1 = quiescence.
        pub const SITE: &[u8] = &[$($site),*];
        /// What the zone actually brackets, for the report's legend.
        pub const WHAT: &[&str] = &[$($what),*];
    };
}
#[cfg(feature = "prof")]

zones! {
    Draw      => 0, "draw",      "repetition + 50-move scan over the key stack";
    TtProbe   => 0, "tt probe",  "hash read; the one guaranteed cache miss";
    Eval      => 0, "eval",      "static eval, only when the TT did not carry one";
    Gen       => 0, "movegen",   "generate(All)";
    Order     => 0, "order",     "score_moves: MVV-LVA, killers, history";
    Price     => 0, "price",     "move_gain + history read + Pricing::price, per child";
    Make      => 0, "make",      "Board::make_move";
    Push      => 0, "acc push",  "incremental accumulator update for the child";
    Pop       => 0, "acc pop",   "accumulator stack unwind";
    Prefetch  => 0, "prefetch",  "tt.prefetch on the child key";
    TtStore   => 0, "tt store",  "hash write";
    Hist      => 0, "history",   "update_quiet_heuristics on a cutoff";
    Observe   => 0, "observe",   "runtime eval adaptation";
    QEval     => 1, "eval",      "stand-pat static eval";
    QGen      => 1, "movegen",   "generate(Captures), or All in check";
    QOrder    => 1, "order",     "score_moves";
    QSee      => 1, "delta+SEE", "delta-pruning gain lookup and see()";
    QMake     => 1, "make",      "Board::make_move";
    QPush     => 1, "acc push",  "incremental accumulator update";
    QPop      => 1, "acc pop",   "accumulator stack unwind";
    QObserve  => 1, "observe",   "runtime eval adaptation";
}
#[cfg(feature = "prof")]

pub const N: usize = NAMES.len();

#[cfg(feature = "prof")]
/// Per-thread tallies. Lives in `ThreadData` for the same reason `hyst` does:
/// the search already has one of those per thread, and `bench` already knows
/// how to merge them.
pub struct Tally {
    /// Cycles per zone. Only a timing build has these; a `work` build carries
    /// the counts alone, which is the whole point of it -- see `work.rs`.
    #[cfg(feature = "nodeprof")]
    pub cyc: [u64; N],
    pub calls: [u64; N],
    /// Nodes at each site, so the per-node column has the right denominator.
    pub nodes: [u64; 2],
    /// Total cycles inside the root search, the denominator everything else
    /// is a share of.
    pub total: u64,
    /// Counters that answer one question each, and cost a branch. `[probes,
    /// hits with a usable static eval, hits with a usable bound]` for a TT
    /// read in quiescence, which the search does not currently do. Quiescence
    /// is 45% of search time and effectively all of it is static eval, so
    /// whether the hash already knows the answer is the single largest open
    /// question in the profile — and it is answerable without changing the
    /// search, by probing and throwing the result away.
    pub qtt: [u64; 3],
    /// `[q-nodes, q-keys already seen in quiescence this search]`. The 1.2%
    /// above says the main search's stores do not cover quiescence; this says
    /// whether quiescence would cover *itself* if it stored. A direct-mapped
    /// witness table, so collisions make this an over-count and the number is
    /// an upper bound — which is the right direction for deciding not to
    /// bother.
    pub qseen: [u64; 2],
    pub witness: Vec<u64>,
    /// Running cost in **picoseconds**, maintained incrementally: one integer
    /// add per bracketed call. Integer and not `f64` so that a work-limited
    /// search is bit-reproducible, for exactly the reason the node ceiling is
    /// tested every node rather than every 2048 -- an objective that depends
    /// on where a rounding error landed is not an objective. Zero until
    /// `set_prices` is called; a run that never sets them still gets exact
    /// counts, which is all `chess work` needs.
    pub work_ps: u64,
    price_ps: [u64; N],
    control_ps: u64,
}
#[cfg(feature = "prof")]

impl Default for Tally {
    fn default() -> Tally {
        Tally {
            #[cfg(feature = "nodeprof")]
            cyc: [0; N],
            calls: [0; N],
            nodes: [0; 2],
            total: 0,
            qtt: [0; 3],
            qseen: [0; 2],
            work_ps: 0,
            // Priced from the frozen deploy table by default, NOT zero. A zero
            // table makes `work_ps` stay at 0, so a `Limits::work` ceiling
            // never binds and the search runs to `MAX_PLY` instead of erroring
            // -- which is exactly what happened the first time `tune` asked for
            // a work budget. Any caller that wants a different table (a fresh
            // `nodeprof` emit, another machine) overwrites it with
            // `set_prices`; nobody has to remember to install one to make the
            // meter work.
            price_ps: crate::work::deploy_price_ps(),
            control_ps: crate::work::deploy_control_ps(),
            // 32 MB, and only the `qprobe` build ever reads it. Allocating it
            // unconditionally would evict the eval weights on every build.
            #[cfg(feature = "qprobe")]
            witness: vec![0; 1 << 22],
            #[cfg(not(feature = "qprobe"))]
            witness: Vec::new(),
        }
    }
}
#[cfg(feature = "prof")]

impl Tally {
    /// Has this quiescence key been seen before in this search? Records it
    /// either way. `qprobe` only -- the table is empty otherwise.
    #[cfg(feature = "qprobe")]
    #[inline(always)]
    pub fn q_witness(&mut self, key: u64) {
        self.qseen[0] += 1;
        let i = (key as usize) & (self.witness.len() - 1);
        if self.witness[i] == key {
            self.qseen[1] += 1;
        }
        self.witness[i] = key;
    }

    #[cfg(feature = "nodeprof")]
    #[inline(always)]
    pub fn add(&mut self, z: Z, dt: u64) {
        let i = z as usize;
        // SAFETY: `Z` is `repr(usize)` with exactly `N` variants, so `i < N`.
        unsafe {
            *self.cyc.get_unchecked_mut(i) += dt;
            *self.calls.get_unchecked_mut(i) += 1;
            self.work_ps += *self.price_ps.get_unchecked(i);
        }
    }

    /// Count a call without timing it. One increment, no `rdtsc`, no
    /// serialisation -- so a `work` build runs at very nearly playing speed
    /// and its counts are exact rather than sampled.
    #[inline(always)]
    pub fn hit(&mut self, z: Z) {
        let i = z as usize;
        // SAFETY: `Z` is `repr(usize)` with exactly `N` variants, so `i < N`.
        unsafe {
            *self.calls.get_unchecked_mut(i) += 1;
            self.work_ps += *self.price_ps.get_unchecked(i);
        }
    }

    /// Charge a node its share of `control` -- the search's own control flow
    /// and the zones below the profiler's floor. Called once per node at each
    /// site, so the two together are the whole model.
    #[inline(always)]
    pub fn hit_node(&mut self, site: usize) {
        self.nodes[site] += 1;
        self.work_ps += self.control_ps;
    }

    /// Install a price list, in picoseconds per call. Rounded once, here, so
    /// every later add is exact.
    pub fn set_prices(&mut self, ns: &[f64], control_per_node: f64) {
        for i in 0..N.min(ns.len()) {
            self.price_ps[i] = (ns[i] * 1000.0).round() as u64;
        }
        self.control_ps = (control_per_node * 1000.0).round() as u64;
    }

    pub fn merge(&mut self, o: &Tally) {
        for i in 0..N {
            #[cfg(feature = "nodeprof")]
            {
                self.cyc[i] += o.cyc[i];
            }
            self.calls[i] += o.calls[i];
        }
        self.nodes[0] += o.nodes[0];
        self.nodes[1] += o.nodes[1];
        self.total += o.total;
        for i in 0..3 {
            self.qtt[i] += o.qtt[i];
        }
        self.qseen[0] += o.qseen[0];
        self.qseen[1] += o.qseen[1];
        self.work_ps += o.work_ps;
    }
}

/// Bracket a leaf operation. Expands to the bare expression without the
/// feature, so a non-profiling build cannot pay for this even in principle.
#[cfg(feature = "nodeprof")]
macro_rules! zone {
    ($s:expr, $z:expr, $e:expr) => {{
        let __t0 = $crate::nodeprof::tsc();
        // Evaluated before `$s.td.prof` is borrowed, so `$e` may use `$s`.
        let __v = $e;
        $s.td.prof.add($z, $crate::nodeprof::tsc().wrapping_sub(__t0));
        __v
    }};
}
#[cfg(all(feature = "prof", not(feature = "nodeprof")))]
macro_rules! zone {
    ($s:expr, $z:expr, $e:expr) => {{
        // Evaluated before `$s.td.prof` is borrowed, so `$e` may use `$s`.
        let __v = $e;
        $s.td.prof.hit($z);
        __v
    }};
}
#[cfg(not(feature = "prof"))]
macro_rules! zone {
    ($s:expr, $z:expr, $e:expr) => {
        $e
    };
}
pub(crate) use zone;

#[cfg(feature = "nodeprof")]
/// What one `tsc()` pair costs, and what a TSC cycle is worth in nanoseconds.
///
/// The pair cost is the median of many samples rather than the mean: an
/// interrupt during calibration inflates the mean and would then be subtracted
/// from every zone in the report.
pub struct Calib {
    pub pair_cyc: f64,
    pub ns_per_cyc: f64,
}
#[cfg(feature = "nodeprof")]

pub fn calibrate() -> Calib {
    // TSC frequency, against a clock that is defined in seconds.
    let t0 = tsc();
    let w0 = std::time::Instant::now();
    while w0.elapsed() < std::time::Duration::from_millis(200) {
        std::hint::spin_loop();
    }
    let ns_per_cyc = w0.elapsed().as_nanos() as f64 / tsc().wrapping_sub(t0) as f64;

    let mut s: Vec<u64> = Vec::with_capacity(4096);
    let mut sink = 0u64;
    for _ in 0..4096 {
        let a = tsc();
        std::hint::black_box(&mut sink);
        s.push(tsc().wrapping_sub(a));
    }
    s.sort_unstable();
    Calib { pair_cyc: s[s.len() / 2] as f64, ns_per_cyc }
}

#[cfg(feature = "nodeprof")]
/// The table. `total` is the measured cycle count of the whole search, so the
/// share column is a share of the real thing and the remainder is honest.
pub fn report(t: &Tally, c: &Calib) {
    let ns = |cyc: f64| cyc * c.ns_per_cyc;
    let nodes = t.nodes[0] + t.nodes[1];
    let probes: u64 = t.calls.iter().sum();
    // The denominator is the search **without** the instrument: the measured
    // interval minus the probe pairs that only exist because we are measuring.
    // Using the raw total instead would charge every zone a share of the cost
    // of watching it, and would dump the whole correction into `control`.
    let base = (t.total as f64 - c.pair_cyc * probes as f64).max(1.0);
    println!(
        "\n{} nodes ({} main, {} q) | TSC {:.3} GHz, probe pair {:.0} cyc ({:.1} ns)",
        nodes, t.nodes[0], t.nodes[1], 1.0 / c.ns_per_cyc, c.pair_cyc, ns(c.pair_cyc),
    );
    println!(
        "measured {:.1} ms, of which {:.1} ms is the instrument -> {:.1} ms of search, \
         {:.1} ns/node",
        ns(t.total as f64) / 1e6,
        ns(c.pair_cyc * probes as f64) / 1e6,
        ns(base) / 1e6,
        ns(base) / nodes.max(1) as f64,
    );
    println!(
        "\n{:<10}{:>12}{:>10}{:>11}{:>10}{:>9}",
        "zone", "calls", "/node", "ns/call", "total ms", "share"
    );

    let mut attributed = 0.0;
    let mut floored = 0usize;
    for site in 0..2u8 {
        println!("{}", if site == 0 { "-- main search" } else { "-- quiescence" });
        let denom = t.nodes[site as usize].max(1) as f64;
        let mut order: Vec<usize> = (0..N).filter(|&i| SITE[i] == site && t.calls[i] > 0).collect();
        order.sort_by(|&a, &b| t.cyc[b].cmp(&t.cyc[a]));
        for i in order {
            let calls = t.calls[i] as f64;
            let net = t.cyc[i] as f64 - c.pair_cyc * calls;
            // A zone whose measured cost is at or below the probe pair is not
            // "free" — it is below the resolution of this instrument, and
            // saying so is the difference between a measurement and a claim.
            if net <= 0.0 {
                floored += 1;
                println!(
                    "{:<10}{:>12}{:>10.2}{:>11}{:>10}{:>9}",
                    NAMES[i], t.calls[i], calls / denom, "<probe", "-", "-"
                );
                continue;
            }
            attributed += net;
            println!(
                "{:<10}{:>12}{:>10.2}{:>11.1}{:>10.1}{:>8.1}%",
                NAMES[i],
                t.calls[i],
                calls / denom,
                ns(net) / calls,
                ns(net) / 1e6,
                100.0 * net / base,
            );
        }
    }

    // Everything the zones do not cover: the pruning tests, the move loop, the
    // PV copy, the recursion itself. A large number here is not an error, it
    // is the search's own control flow — but it is also where an unattributed
    // cost would hide, so it is printed, not omitted.
    let rest = base - attributed;
    println!("{:-<62}", "");
    println!(
        "{:<10}{:>12}{:>10}{:>11}{:>10.1}{:>8.1}%",
        "control", "", "", "", ns(rest) / 1e6, 100.0 * rest / base
    );
    println!(
        "{:<10}{:>12}{:>10}{:>11.1}{:>10.1}{:>8.1}%",
        "node", nodes, "", ns(base) / nodes.max(1) as f64, ns(base) / 1e6, 100.0
    );
    if floored > 0 {
        println!(
            "\n{floored} zone(s) measured at or below the {:.1} ns probe pair; their real \
             cost is inside `control`.",
            ns(c.pair_cyc),
        );
    }
    if t.qtt[0] > 0 {
        let p = t.qtt[0] as f64;
        println!(
            "\nquiescence TT (probed and discarded, the search does not read it): \
             {:.1}% of {} q-nodes already have a stored static eval, {:.1}% a bound \
             deep enough to return on.",
            100.0 * t.qtt[1] as f64 / p,
            t.qtt[0],
            100.0 * t.qtt[2] as f64 / p,
        );
    }
    if t.qseen[0] > 0 {
        println!(
            "of {} q-nodes, {:.1}% repeat a key quiescence already visited (upper bound: \
             a direct-mapped witness, so collisions inflate it).",
            t.qseen[0],
            100.0 * t.qseen[1] as f64 / t.qseen[0] as f64,
        );
    }
    println!(
        "search-only speed {:.0} nps. `chess bench` reports LESS than this and is not \n\
         wrong to: its wall clock also contains one 64 MB TT clear per position, which \n\
         is setup, not search. `nodeprof` prints both above so the gap is visible.",
        nodes as f64 / (ns(base) / 1e9),
    );
}

/// Emit the price table in a form `work.rs` can parse, so a frozen table is
/// generated from a measurement rather than copied out of a report by hand.
///
/// One line per zone: `index name site calls ns_per_call`. Zones at or below
/// the probe floor emit **0.0** rather than a guess — their real cost stays
/// inside `control`, which is emitted as its own per-node line. That keeps the
/// modelled total equal to the measured total by construction; it does not
/// pretend to know where those nanoseconds went.
#[cfg(feature = "nodeprof")]
pub fn emit_prices(t: &Tally, c: &Calib) {
    let ns = |cyc: f64| cyc * c.ns_per_cyc;
    let probes: u64 = t.calls.iter().sum();
    let base = (t.total as f64 - c.pair_cyc * probes as f64).max(1.0);
    let nodes = t.nodes[0] + t.nodes[1];

    let mut attributed = 0.0;
    println!("# index name site calls ns_per_call");
    for i in 0..N {
        let calls = t.calls[i] as f64;
        let net = t.cyc[i] as f64 - c.pair_cyc * calls;
        let per = if calls > 0.0 && net > 0.0 {
            attributed += net;
            ns(net) / calls
        } else {
            0.0
        };
        println!("price {i} {} {} {} {per:.4}", NAMES[i], SITE[i], t.calls[i]);
    }
    println!(
        "control {:.4}  # ns/node over {} nodes ({} main, {} q)",
        ns(base - attributed) / nodes.max(1) as f64,
        nodes,
        t.nodes[0],
        t.nodes[1],
    );
    println!("total_ns {:.1}  # measured search, instrument removed", ns(base));
    println!("nodes {} {} {}", nodes, t.nodes[0], t.nodes[1]);
}

/// The legend, so a column can be read without opening this file.
#[cfg(feature = "nodeprof")]
pub fn legend() {
    println!("\nwhat each zone brackets");
    for i in 0..N {
        println!("  {:<10}{:<3}{}", NAMES[i], if SITE[i] == 0 { "m" } else { "q" }, WHAT[i]);
    }
}
