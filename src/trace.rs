//! What the price list actually did, on one position, by hand.
//!
//! The tuner scores a price list with one number over 14000 positions. That is
//! the right instrument for deciding, and the wrong one for *understanding* —
//! LEDGER 007/009/010 are a sequence of discoveries that the number was
//! measuring something other than what we thought. This module is the other
//! half: run one position, record every pricing decision, and print it.
//!
//! The metrics here are chosen to be **free and oracle-free**. Everything in
//! `Agg` is computed from information alpha-beta already has, which is what
//! `library/005` argues is the missing half — the existing objective is a
//! recall metric and nearly every precision metric costs nothing.
//!
//! The one that matters most is `researched`. When a child is priced down and
//! its reduced search fails high, the search must redo it at full budget: the
//! price was **wrong**, the search says so itself, and we paid twice to find
//! out. No teacher, no labels, no oracle. That is a per-decision precision
//! signal, and it is sitting in the move loop already.

use crate::chess_move::Move;
use crate::search::PLY;
use crate::eval::Score;

/// One priced child, recorded at shallow plies only.
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub ply: u8,
    pub rank: u16,
    pub mv: Move,
    /// Budget the parent had.
    pub parent_budget: i32,
    /// Charged on top of the fare, in budget units.
    pub cost: i32,
    /// What the child was left with.
    pub child_budget: i32,
    pub gap: Score,
    pub hist: i32,
    pub is_pv: bool,
    pub priced_out: bool,
    /// The reduced search failed high and had to be redone at full budget.
    pub researched: bool,
    pub raised_alpha: bool,
    pub cutoff: bool,
    pub dropped_to_q: bool,
    /// Nodes spent in this child's subtree, re-search included.
    pub nodes: u64,
    pub score: Score,
}

impl Event {
    /// Price in plies, for reading.
    pub fn price_plies(&self) -> f64 {
        self.cost as f64 / PLY as f64
    }
}

/// Whole-tree counters. Every one of these is free.
#[derive(Clone, Default, Debug)]
pub struct Agg {
    pub nodes: u64,
    /// Interior nodes that ran a move loop.
    pub interior: u64,
    /// Children that reached the pricing code (rank > 0).
    pub priced: u64,
    /// Children `skip_below` refused outright.
    pub priced_out: u64,
    /// Children whose reduced search failed high — the price was wrong.
    pub researched: u64,
    /// Nodes burned re-searching those. Pure waste: the same subtree twice.
    pub research_nodes: u64,
    /// Children that improved alpha.
    pub raised_alpha: u64,
    /// Children that caused a beta cutoff.
    pub cutoffs: u64,
    /// Interior nodes where the *first* move cut off. Ordering quality —
    /// `library/005` calls this out as the confound in every allocation
    /// experiment, because prices change ordering through the history table.
    pub cutoff_at_first: u64,
    /// Children left with a non-positive budget, i.e. straight to quiescence.
    pub dropped_to_q: u64,
    /// Charged price in whole plies, bucketed. Index 0 is "free".
    pub price_hist: [u64; 12],
    /// Subtree nodes by rank bucket: 0, 1, 2-3, 4-7, 8-15, 16-31, 32+.
    pub nodes_by_rank: [u64; 7],
    pub children_by_rank: [u64; 7],
    /// Children whose `gap` was strictly positive. `price()` computes
    /// `log2(max(gap,0)/gap_unit + 1)`, so a non-positive gap makes the whole
    /// `c_gap` term **exactly zero** no matter what `c_gap` is set to. If this
    /// is near zero the gap feature is not weak, it is dead code.
    pub gap_positive: u64,
    /// Re-searches by rank bucket — where the price list is getting it wrong.
    pub researched_by_rank: [u64; 7],
}

pub fn rank_bucket(rank: usize) -> usize {
    match rank {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        16..=31 => 5,
        _ => 6,
    }
}

pub const RANK_LABELS: [&str; 7] = ["0", "1", "2-3", "4-7", "8-15", "16-31", "32+"];

/// Collector. Lives in `ThreadData` behind an `Option`, so the hot path pays
/// one predictable not-taken branch and the bench fingerprint cannot move.
#[derive(Clone, Debug)]
pub struct Trace {
    /// Record individual events only for `ply < event_ply`. The aggregate is
    /// always whole-tree. Depth 10 is millions of nodes; the point of the
    /// per-event list is to be read by a person.
    pub event_ply: u8,
    pub events: Vec<Event>,
    pub agg: Agg,
}

impl Trace {
    pub fn new(event_ply: u8) -> Trace {
        Trace { event_ply, events: Vec::new(), agg: Agg::default() }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn child(&mut self, e: Event) {
        let a = &mut self.agg;
        a.priced += 1;
        if e.priced_out {
            a.priced_out += 1;
        }
        if e.researched {
            a.researched += 1;
            a.researched_by_rank[rank_bucket(e.rank as usize)] += 1;
        }
        if e.gap > 0 {
            a.gap_positive += 1;
        }
        if e.raised_alpha {
            a.raised_alpha += 1;
        }
        if e.cutoff {
            a.cutoffs += 1;
        }
        if e.dropped_to_q {
            a.dropped_to_q += 1;
        }
        let p = ((e.cost + PLY / 2) / PLY).clamp(0, 11) as usize;
        a.price_hist[p] += 1;
        let rb = rank_bucket(e.rank as usize);
        a.nodes_by_rank[rb] += e.nodes;
        a.children_by_rank[rb] += 1;
        if (e.ply as u8) < self.event_ply {
            self.events.push(e);
        }
    }

    /// The main line's child, which pays only the fare. Recorded so the rank-0
    /// column of the node histogram is populated — without it the "where did
    /// the budget go" table is missing the row that dominates it.
    #[inline]
    pub fn main_line(&mut self, e: Event) {
        self.agg.nodes_by_rank[0] += e.nodes;
        self.agg.children_by_rank[0] += 1;
        if e.cutoff {
            self.agg.cutoff_at_first += 1;
        }
        if (e.ply as u8) < self.event_ply {
            self.events.push(e);
        }
    }
}

// ------------------------------------------------------------------ reporting

/// The free precision/recall numbers, as a block of text.
///
/// Read it as: `re-search rate` is over-pruning the search caught itself doing,
/// `utilisation` is how much of what we did search was worth searching, and
/// `waste` is what the mistakes cost in nodes.
pub fn report_agg(a: &Agg) -> String {
    let mut s = String::new();
    let pct = |x: u64, y: u64| if y == 0 { 0.0 } else { 100.0 * x as f64 / y as f64 };
    s.push_str(&format!("  nodes                {:>12}\n", a.nodes));
    s.push_str(&format!("  interior nodes       {:>12}\n", a.interior));
    s.push_str(&format!("  priced children      {:>12}\n", a.priced));
    s.push_str("\n  -- precision: was the money we spent worth spending? --\n");
    s.push_str(&format!(
        "  utilisation          {:>11.2}%   (searched children that raised alpha or cut off)\n",
        pct(a.raised_alpha + a.cutoffs, a.priced - a.priced_out)
    ));
    s.push_str(&format!(
        "  re-search rate       {:>11.2}%   (price was wrong; search said so itself)\n",
        pct(a.researched, a.priced - a.priced_out)
    ));
    s.push_str(&format!(
        "  re-search waste      {:>11.2}%   ({} nodes searched twice)\n",
        pct(a.research_nodes, a.nodes),
        a.research_nodes
    ));
    s.push_str("\n  -- recall: did we refuse to spend where it mattered? --\n");
    s.push_str(&format!(
        "  priced out           {:>11.2}%   ({} children never searched at all)\n",
        pct(a.priced_out, a.priced),
        a.priced_out
    ));
    s.push_str(&format!(
        "  dropped to qsearch   {:>11.2}%   ({} children left with no budget)\n",
        pct(a.dropped_to_q, a.priced),
        a.dropped_to_q
    ));
    s.push_str(&format!(
        "\n  gap > 0              {:>11.2}%   ({} of {} priced children — the c_gap term is\n           {:>35}exactly zero on the rest, whatever c_gap is set to)\n",
        pct(a.gap_positive, a.priced), a.gap_positive, a.priced, ""
    ));
    s.push_str("\n  -- confound: ordering moves when prices move --\n");
    s.push_str(&format!(
        "  cutoff at first move {:>11.2}%   ({} of {} interior nodes)\n",
        pct(a.cutoff_at_first, a.interior),
        a.cutoff_at_first,
        a.interior
    ));

    s.push_str("\n  -- where the budget went --\n");
    s.push_str("  rank      children        nodes     share\n");
    let tot: u64 = a.nodes_by_rank.iter().sum();
    for i in 0..7 {
        s.push_str(&format!(
            "  {:<8} {:>9} {:>12} {:>8.2}%\n",
            RANK_LABELS[i],
            a.children_by_rank[i],
            a.nodes_by_rank[i],
            pct(a.nodes_by_rank[i], tot)
        ));
    }

    s.push_str("\n  -- re-searches by rank (where the price is wrong) --\n");
    s.push_str("  rank     children  re-searched     rate\n");
    for i in 1..7 {
        s.push_str(&format!(
            "  {:<8} {:>9} {:>12} {:>8.2}%\n",
            RANK_LABELS[i], a.children_by_rank[i], a.researched_by_rank[i],
            pct(a.researched_by_rank[i], a.children_by_rank[i])
        ));
    }

    s.push_str("\n  -- price charged, in plies --\n");
    let ptot: u64 = a.price_hist.iter().sum();
    for (i, &c) in a.price_hist.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let bar = "#".repeat(((60.0 * c as f64 / ptot.max(1) as f64) as usize).max(1));
        let lbl = if i == 11 { "11+".to_string() } else { i.to_string() };
        s.push_str(&format!("  {lbl:>3}  {c:>9} {:>6.2}%  {bar}\n", pct(c, ptot)));
    }
    s
}

/// Per-child table for one node. `events` should already be filtered to a ply.
pub fn report_events(events: &[Event], ply: u8) -> String {
    let mut s = String::new();
    s.push_str(
        "  rank move      price  budget    gap   hist  nodes        score  flags\n",
    );
    for e in events.iter().filter(|e| e.ply == ply) {
        let mut flags = String::new();
        if e.is_pv {
            flags.push_str("pv ");
        }
        if e.priced_out {
            flags.push_str("SKIPPED ");
        }
        if e.dropped_to_q {
            flags.push_str("qsearch ");
        }
        if e.researched {
            flags.push_str("RESEARCH ");
        }
        if e.cutoff {
            flags.push_str("cutoff ");
        }
        if e.raised_alpha {
            flags.push_str("alpha ");
        }
        s.push_str(&format!(
            "  {:>4} {:<8} {:>5.2}p {:>6.2}p {:>6} {:>6} {:>10} {:>8} {}\n",
            e.rank,
            e.mv.to_string(),
            e.price_plies(),
            e.child_budget as f64 / PLY as f64,
            e.gap,
            e.hist,
            e.nodes,
            if e.priced_out { "-".to_string() } else { e.score.to_string() },
            flags.trim_end()
        ));
    }
    s
}
