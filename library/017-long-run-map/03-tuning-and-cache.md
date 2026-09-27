# 017/03 · Tuning (TU) and the eval cache (EC)

## Why tuning is the biggest untouched lever

`chess params` lists 50 tunable parameters and none has been fitted in games.
The regret proxy is at its floor at n=1000 (4.1 cp, 065) and its one
near-floor promotion lost in games (086, −23). SPSA in games is the standard
tool for exactly this regime: many parameters, small individual effects,
only games see them. The box plays ~780 games/h at 8+0.08 conc 5 (STATE,
contempt run: 400 games in 1836 s), so ~18k/day; a 30k-pair campaign at a
shorter TC is 1–3 days. And library/013 rule 2 says the learned controller
must beat a *tuned* baseline, so this is also a prerequisite for the
research claim.

## TU · Tuning

**TU-1 · NEUTRAL · move the remaining hard-coded constants into `Params`**,
defaults = today's values, bench exact. List (search for the literal):
history bonus `min(depth², 1200)` (cap and exponent shape), history gravity
divisor `16384`, quiescence delta margin `100`, SEE threshold in quiescence
`0`, singular `tt_stored >= depth − 3`, aspiration growth `delta += delta/2`,
aspiration full-window depth `4`, time manager `moves_to_go = 30`, increment
share `3/4`, cap `time_left/3`, next-iteration gate `0.5`, movetime margin
`20 ms`, eval readout scale `K = 288.5` (as a search-side multiplier on cp,
not in the net file). Each gets a doc comment and a box.

**TU-2 · NEUTRAL · `--set k=v,...` at UCI startup.** `apply_overrides`
exists (`tune.rs`); call it from `main` when the engine starts in UCI mode,
so `chess match --engine "$BIN --wdl $NET --set c_rank=281"` works with a
playing build. Print one `info string params: ...` line listing every
non-default value (so a PGN's engine is identifiable). Bench unchanged
without the flag.

**TU-3 · NEUTRAL · `tools/spsa.py`** (stdlib only). Inputs: a parameter
file (`name default min max c_end`), `--pairs N`, `--tc`, `--concurrency`,
binary, net. Loop: keep `concurrency` games-pairs in flight; each pair runs
`chess match --engine "... --set θ+" --opponent "... --set θ−" --games 2
--tc T --quiet --pgn <pair.pgn>`; parse the score; apply the P6 update;
append `iter, θ` to `spsa.csv` every 100 pairs; resume from `spsa.csv` on
restart (the PC gets shut down). Test on a toy: tune `aspiration_window`
alone from 100 for 2000 pairs and check it moves toward 25–50.
Faster variant (later): a `chess spsa` subcommand reusing `matchplay`
in-process with one engine pool, avoiding process start per pair (needs
016 C4 first: eval settings are process globals).

**TU-4 · PROBE · pick `c_end` per parameter from the proxy.** For each
parameter, `tune compare` default vs default ± step at 2–3 step sizes on 4k
labels; `c_end` = the step where regret moves ~1 SE. Keeps SPSA from
wasting pairs on steps too small to matter or too large to be local.

**TU-5 · SPRT · campaign A: the price list + whole-node pruning** (~16
params): `c_rank`, `c_rank_depth`, `c_hist`, `pv_discount`, `max_base`,
`max_slope`, `skip_below`, `fare`, `c_see`, `rfp_margin`, `rfp_max_depth`,
`nmp_base_reduction`, `nmp_depth_divisor`, `aspiration_window`,
`check_extension`, delta margin. Exclude small-integer boxes
(`nmp_min_depth`) and dormant features. 20k pairs at 5+0.05. Final: P4
SPRT at 8+0.08. Variants as separate campaigns:
- a. all 16 at once (standard);
- b. price list only (8), then whole-node only (8) sequentially — tests
  whether joint fitting matters;
- c. same as a. but `fare` and `c0` tied (004: they are degenerate where the
  clamp does not bind — tune one, fix the other).

**TU-6 · SPRT · campaign B: the time manager** (TU-1's TM constants +
easy-move params + a new "score dropped by > X cp since last iteration →
extend by Y%" pair). TM is invisible to node-limited tools, so games only.
Variants: a. allocation only; b. allocation + extension-on-instability;
c. b + node-share of best move (spend less when the best move took > 90%
of root nodes).

**TU-7 · SPRT · campaign C: history and ordering constants** (bonus cap,
bonus shape `depth²` vs `depth·(depth+1)` vs linear, gravity divisor,
killer/history weights in `score_moves`). Only after FX-1, since persistence
changes what the right constants are.

**TU-8 · SPRT · re-run campaign A at a longer TC** (20+0.2, fewer pairs
seeded from A's result). Search constants scale with depth (004). If the LTC
fit differs from STC, ship the LTC one and record both.

**TU-9 · PROBE · the eval scale K.** SPSA over `K` alone (TU-1 readout
multiplier), all margins fixed. Every cp margin reads K's units, so an
off-K is a uniform mis-scaling of every margin. Then include K in campaign A's
next round.

**TU-10 · SPRT · re-tune after every net change.** A new net moves the cp
scale of static-vs-search error (092: σ is 107 cp on m1-b1 vs 206 on PeSTO),
so the margins are net-specific. Standing rule, not a one-off.

**TU-11 · PROBE · SPSA on the proxy** as a pre-conditioner: same driver, but
the "match" is `tune compare` on 4k fresh labels at 15k nodes with θ± — a
deterministic, 10-second iteration. Use it only to pick SPSA's starting
point for games, never to ship. Needs EC-1 to be cheap at 16k labels.

**TU-12 · PROBE · does the proxy agree with SPSA?** After TU-5, score
default vs SPSA-θ on the proxy at 4k labels. Agreement in sign = one more
data point for the exchange rate (2–1 so far, 086); disagreement = the
proxy's blind spot, recorded.

**TU-13 · SPRT · tune the dormant features once, jointly.** After LC-1
exists, the corpus fits them; before that, a small SPSA over `c_recap`,
`c_gives_check`, `c_incheck`, `c_ply`, `c_depth` starting at 0 tells whether
any has a non-zero optimum in games (065 could only say "≤ 0 at 4.1 cp").

## EC · The eval cache (LEDGER 091)

Measured: hit rate vs one previous pass is 38–96% depending on the knob;
40% for an SPSA-shaped 4-knob step, **58% once 5 passes are cached**.
Speed-up ceiling **3.7x** (~23% of a pass is search code). No re-pricing:
eval is a pure function of the position and the budgets count calls.

**EC-1 · NEUTRAL · a cross-pass eval cache in `tune`.** Lockless table like
`tt.rs` (key 64 bit, value i16 cp; ~16 B/entry; 8M entries = 128 MB),
shared by the tuner's threads, living for the process (so `tune search`,
`compare` and TU-11 benefit across passes). Keyed by zobrist; the table
records the net's sha and the draw value and refuses to serve if either
differs. Engine builds never allocate it. Verification: `tune eval` output
identical to the digit with cache on vs off, at `--nodes` and `--work`.
With `--work`, the meter must charge a full eval on a hit.

**EC-2 · NEUTRAL · lazy accumulator** (needed to reach the ceiling; also
SP-1). `push` records the move delta and marks the level dirty; `evaluate`
on a miss walks back to the last computed level and applies the pending
deltas. Verification: `CHESS_ACC_CHECK` 0 mismatches; bench exact.

**EC-3 · NEUTRAL · persist the cache to disk** per (net sha, corpus file):
`chess-data/tune/<corpus>.<netsha>.evalcache`. A new session's first pass
starts warm. Measure the hit rate on the first pass of a new session.

**EC-4 · PROBE · the cache in labelling.** `label_one` searches each top-5
child at 400k nodes from a cleared TT; siblings share much of their
subtree. An eval cache keeps the "each child from a cleared table"
independence (eval is order-independent, the TT is not). Measure the hit
rate across siblings; labelling 1k takes ~9 min today, MS-6 needs 4–16k.

**EC-5 · PROBE · the cache in the per-child oracle (LC-1).** Full-width
searches at d = 6, 7, 8 of the same positions revisit leaves heavily.
Measure before building LC-1's pipeline around it.

**EC-6 · PROBE · work-budgeted games with a shared cache.** For tuning
*search* params with the net fixed, games at a fixed work budget per move
(deterministic, load-immune) could share one eval cache across all games of
an SPSA run. Two probes first: (a) hit rate across game pairs (same opening,
θ±); (b) does a work-budget match agree with a time match on one known pair
(e.g. FX-3 arms)? 068's retirement of equal-nodes does not settle (b) — its
mechanism is doubtful (RV-4) and it compared nets, not search params.
