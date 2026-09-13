//! Does raw `ply` identify game boundaries in a binpack?
//!
//! The game-stride extractor needs a reliable "this entry starts a new game"
//! signal. The cheap candidate is `ply <= prev_ply`. This measures whether that
//! is actually true: within a game ply must step by exactly +1, and a reset must
//! land on a small starting ply.

use std::fs::File;
use std::io::BufReader;

use sfbinpack::{read_chunk_into, ChunkReader};

fn main() {
    let input = std::env::args().nth(1).expect("usage: plyprobe <file.binpack>");
    let mut reader = BufReader::with_capacity(1 << 22, File::open(&input).expect("open"));

    let mut chunk = Vec::new();
    let mut seen: u64 = 0;
    let mut resets: u64 = 0;
    // Histogram of ply deltas within a run, and of the ply a run restarts at.
    let mut delta_hist = [0u64; 8]; // index 0 => delta != 1..=6 ("other")
    let mut start_hist = [0u64; 32]; // starting ply, clamped
    let mut len_hist = [0u64; 16]; // game length in entries, log2 bucket
    let mut prev_ply: i64 = -1;
    let mut cur_len: u64 = 0;

    while read_chunk_into(&mut reader, &mut chunk).unwrap_or(false) {
        let mut cr = ChunkReader::default();
        while cr.has_next(&chunk) {
            let e = cr.next(&chunk);
            let ply = i64::from(e.ply);
            seen += 1;

            if prev_ply < 0 || ply <= prev_ply {
                if cur_len > 0 {
                    let b = (64 - (cur_len.leading_zeros() as usize)).min(15);
                    len_hist[b] += 1;
                }
                resets += 1;
                start_hist[(ply as usize).min(31)] += 1;
                cur_len = 0;
            } else {
                let d = ply - prev_ply;
                delta_hist[if (1..=6).contains(&d) { d as usize } else { 0 }] += 1;
            }
            cur_len += 1;
            prev_ply = ply;
        }
    }

    println!("entries seen        : {seen}");
    println!("game starts (resets): {resets}");
    println!("mean entries/game   : {:.1}", seen as f64 / resets.max(1) as f64);
    println!();
    println!("ply delta within a game (must be +1 if ply is contiguous):");
    for d in 1..=6 {
        if delta_hist[d] > 0 {
            println!("  +{d:<3}: {:>12}  {:6.3}%", delta_hist[d], 100.0 * delta_hist[d] as f64 / (seen - resets) as f64);
        }
    }
    println!("  other: {:>11}  {:6.3}%", delta_hist[0], 100.0 * delta_hist[0] as f64 / (seen - resets) as f64);
    println!();
    println!("starting ply of each run:");
    for (p, &c) in start_hist.iter().enumerate() {
        if c > 0 {
            println!("  ply {p:<3}: {c:>10}  {:6.2}%", 100.0 * c as f64 / resets as f64);
        }
    }
    println!();
    println!("game length (entries), log2 buckets:");
    for (b, &c) in len_hist.iter().enumerate() {
        if c > 0 {
            println!("  [{:>5}..{:>5}): {c:>10}", 1u64 << (b - 1), 1u64 << b);
        }
    }
}
