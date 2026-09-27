# 017/06 · Revisit (RV) and measurement (MS)

## RV · Results worth doubting

Each names what was weak about the original test and the smallest re-test.
Re-running a killed idea is justified only when the re-test fixes the named
weakness.

**RV-1 · correction history (085, 0.0 over 586).** Weaknesses: tables reset
every move (FX-1); per-thread 16k entries learned within one search, which
085 itself named as suspect (i); used only in the RFP and NMP gates, not in
futility, razoring or the q stand-pat. Re-test after FX-1: first the
convergence diagnostic 085 asked for (visits per entry and residual variance
at the end of a 10+0.1 move); then SPRT with the corrected eval used in every
pruning decision. Variants: a. pawn key; b. pawn + non-pawn-material key
(two tables summed); c. b. plus continuation correction on `path.mv`.

**RV-2 · continuation history (089, +3.2 over 2424).** Weaknesses: reset
every move; one ply of context only; fed ordering but not the `c_hist` price
(check what `H` the price reads). Re-test after FX-1: a. 1-ply conth,
persistent; b. 1- and 2-ply summed; c. a. with conth included in `H`.

**RV-3 · O3 capture history, O5 countermove, O6 piece/to (ORDERING.md).**
Screened at 100–150 games, intervals ±35–40: these were never resolved,
only screened, and under per-move reset. One bundle SPRT after FX-1 (all
three on vs off), then split only if the bundle wins.

**RV-4 · 068's mechanism does not hold.** It says m2-a2 lost on time
because its tree is 17% bigger at equal depth — but an equal-nodes match
already charges for a bigger tree. The two results differ by ~2.2σ
(+23.5 [+0.7,+46] vs −12.2 [−33,+8]); the boring reading is noise, the
other is an nps difference between the two nets. The ship-on-the-clock
decision stands. Consequence: equal-nodes (or equal-work) matches are not
shown to be biased for same-net search-parameter comparisons, which is what
EC-6 needs. Test: one known pair at equal work and at equal time.

**RV-5 · the 065 screens ("measure ≤ 0").** At a 4.1 cp floor, "≤ 0" means
"unresolved". Re-screen `c_offpv`, `c_qdrop`, `c_ply`, `c_depth`,
`c_incheck`, `c_gives_check` on fresh 4k (or 16k) labels (MS-6) with the
relaxed teacher (LC-6). Or skip straight to TU-13.

**RV-6 · `c_gap` (008).** Killed on a feature with r² 0.03 (004). The idea
lives on as LC-2; the old `c_gap` stays 0.

**RV-7 · the σ phase law (004).** Corrected by 092 (material Q5/Q1 0.75 on
NNUE vs 0.13 on PeSTO). Anything that cites "σ ∝ npm^−0.89" is stale.

**RV-8 · learned ordering (O11/O12, −24/−21).** Part of the loss was speed
(−10% and −16% nps). Low prior; stays closed until LC-1 exists (014's own
reopen condition), then re-try with a refutation-aware teacher and int8 φ.

**RV-9 · IID (074).** Fine as killed. The only cheap check left is 074's
own: first-move fail-high rate at PV nodes with no TT move. If < 80%, IID
has something to fix.

**RV-10 · agree, do not reopen:** dual-net fidelity tiering (077/081/082),
symmetric and asymmetric contempt as shaped, risk-sensitive readouts (066),
whole-node futility return-static (084 — razoring is the fix), c0=200
(086), c1 more passes (083), depth-skip SMP (039).

## MS · Measurement and instruments

**MS-1 · PROBE · cluster CPU capacity.** If the GPU cluster has idle CPU cores,
SPSA and SPRT throughput multiply and CLAUDE.md's "6 cores" premise
changes. Use the `cluster` skill; count cores and typical load over a day;
never assume free, never start games on shared nodes without asking Luka.

**MS-2 · SPRT · a long-TC confirmation.** Everything so far is 8+0.08 or
10+0.1. Replay cp-0004 vs cp-0003 at 40+0.4, conc 5, 400 games (fixed, not
SPRT). If the gap shrinks a lot, depth-sensitive items (SR-2, SR-6, LC-*)
need LTC SPRTs.

**MS-3 · policy · bundles and bounds.** `[0,5]` cannot ship +3–4 effects
within 3000 games (089/090). Two responses, both allowed: bundle
small-positive items (FX-2, RV-3), and let SPSA harvest parameter-level
gains in one run. Do not lower the bound to `[0,2]`; that is the patch grind
CLAUDE.md rules out.

**MS-4 · PROBE · SPRT statistics self-check.** Re-score the last three
SPRT PGNs with `chess elo` pentanomial and check the log's LLR matches.
Already done per run (PGN re-score); make it a script (`tools/rescore.sh`).

**MS-5 · the opening book** (FX-12) [done → LEDGER 096]. The small book
does not replay games (598 of 600 distinct), and an unbalanced book cuts
draws 72%→58% without cutting games to a decision. Support kept on branch
`uho-book` (`--book <file|uho>`); default unchanged. Ceiling on any book at our 72%
draws is 1.14x, so re-run only if the draw rate goes above ~80%.
**Left open:** the gain came from the positions being sharp, not from the
skew — a *balanced* sharp book would collect it without changing the Elo
scale. Unmeasured, capped at 1.12x, ranks below the time control (096).

**MS-6 · PROBE · relabel the proxy corpus from cp-0004 + m1-b1.** All label
files name `gate-3ep-q8` / `b986f5b` (HK-5). 4k positions at 400k nodes
~36 min (9 min/1k), 16k ~2.5 h; EC-4 would cut that. Then re-measure the SE
at n = 4k and 16k (expect ~2.0 → ~1.0 → ~0.5 cp).

**MS-7 · PROBE · re-emit the work price table** (`nodeprof --emit-prices`,
mean of 5, pinned, quiet box). The table in `work.rs` predates the Q-store.

**MS-8 · policy · frozen binaries live in `~/chess-runs/`** (HK-4, P3).

**MS-9 · gauntlet cadence.** Every ~3 checkpoints, `tools/ladder.sh` with
the same four anchors (071/076/088); drop byteknight first if the set
changes (088). Add one stronger anchor above 3246 so the top is bracketed.

**MS-10 · PROBE · proxy vs games, the running tally.** Keep a table in
`library/012`: every proxy-screened change that reached games, with proxy
Δ and Elo. Today 2–1 in direction, size over-predicted ~2x. SPSA (TU-12)
and LC items add rows.

**MS-11 · NEUTRAL · a bench-exact guard in `tools/verify.sh`** (HK-8), so
no agent can commit a behaviour change labelled neutral.

**MS-12 · PROBE · first-move fail-high rate and re-search share over
time.** 30% of nodes are re-searches (008). Log both in `bench` output so
every change shows whether it moved ordering quality or pricing precision —
two numbers that explain many results for free.
