# 018 · What is verified, and the standing measurement facts

_Split out of `STATE.md` on 2026-09-19, which had reached 473 lines against
a 150-line budget. This file is the durable half: things that were measured
once, hold until something overturns them, and are cited rather than re-read.
Correct them here at the source; do not append._

## What is verified

| | |
|---|---|
| perft | all 6 standard positions, full published depth, exact |
| FEN input | castling rights not backed by a king and rook on their home squares are dropped on parse. Taking "KQkq" at its word let movegen generate a castle, `make_move` read a rook off an empty corner, and `Piece::NONE.piece_type()` abort the process — hand-editing a FEN to remove a rook was enough. Regression test in `tests/search.rs`. |
| zobrist | incremental keys match from-scratch recomputation after every move |
| search | mate distances correct, self-play game legal throughout, deterministic |
| harness | SAN, book legality, game-end rules, SPRT statistics — 14 tests |
| bench | **234598 nodes @ depth 9** with the quadratic net, **227236** with the WDL net (both moved when `c_see` shipped, LEDGER 064; the pre-`c_see` pair was 334110 / 396299). **567321 is PeSTO** — `qeval::net()` falls back to it whenever `quad.nnue` is not beside the executable, which is what a frozen binary in a run directory looks like. Name the eval explicitly in any match script; `checkpoint.sh` now does. |
| wdl net path | engine matches torch to **5.5e-6** in logits (i8; f32 3.2e-6), 0/1500 feature and 0/1500 bucket mismatches, accumulator rebuilds 0.03% (LEDGER 050/055) |
| evalprof stage column | **noisy — rank with it, never price with it.** Each stage is a difference of two prefix timings, so the total's 4.6% run-to-run spread becomes **24% on `head` and 50% on `psqt`** (8 runs, same binary, `taskset -c 5`). The *total* is trustworthy. Price a change with a paired nps run holding the node count identical instead (LEDGER 106). |
| eval cost | **949 ns/eval**, 700,250 nps; stage split from `chess evalprof` (LEDGER 055) |
| node cost | **1272 ns/node** WDL, of which eval 64.5% and accumulator push 20.2%; the whole search apart from those two is 5.4%. `chess nodeprof`, agreeing with `bench` to 0.4% and `evalprof` to 1.2% (LEDGER 059) |
| lc0 plane encoder | two code paths sharing no code produce **bit-identical** planes on 4000 real records; K fitted at 260 against LEDGER 016's independent 258.7 (`nnue/lc0/README.md`) |
| tuner | bit-reproducible. Paired SE **±2.06 cp at n=1000** on the binpack corpus, so it resolves **4.1 cp** at \|t\|=2 (LEDGER 061); ±0.26 cp on the old 14000-position PGN set. The two corpora are not comparable. |
| work meter | `--features work`: counts bit-identical across 5 runs, and **unmoved to the last digit** while 6 spinners moved the wall clock 2.06x. Prices calls, not cycles — blind to the i16 accumulator (LEDGER 061). |
| absolute rating | **2554 ± 56** CCRL Blitz; anchor-limited, not game-limited (LEDGER 015) |
| tuning budget | contrast/noise peaks at **15k nodes**; 1k has the *wrong sign* on over-pruning (LEDGER 014) |
| exchange rate | 4x nodes ~ 4.5 cp regret, so **1 cp ~ 25-30 Elo** (LEDGER 007). **Biased high**: it over-predicted `c_see` by 2.3x (−4.6 cp read as ~115 Elo, games said +51) and LEDGER 008 by ~1.3x. Right in direction twice, wrong in size. |
| budget unit | a qsearch node costs **0.275** main-search nodes (`library/007`) |
| price list | 30.45% of nodes go to re-searches; ranks 2-7 hold all the headroom; **57-60% of priced children are refused outright** by the `max_base` ceiling, which is late move pruning back by accident (`library/008`) |


## What is NOT done

- **The budget arm has caught up but not overtaken:** **+10.4 Elo [−8, +29]**
  over 600 games vs 006, against a pre-registered +12 to +14. The interval
  contains zero, so the exchange rate is **not falsified rather than
  confirmed**. LEDGER 008.
- **The root proxy saturates at ~1.7 cp at any depth.** Score at 60k nodes;
  deeper costs 12x and buys nothing. LEDGER 009/010.
- **Lazy SMP is built and measured, but has never played a game.** 6 threads is
  5.5x the nodes and **2.3x the effective search speed** (LEDGER 039). No Elo.
- `chess elo` cannot combine a multi-opponent gauntlet into one anchored
  rating; LEDGER 004's number was combined by hand. And two builds of this
  engine report the same UCI name, so `chess elo` on a self-match measures
  White — the match runner's own summary is correct, the PGN path is not.


## Standing design facts

## The WDL net is the design, and the engine runs it

`cp = 288.5 * logit(W + D/2)` over one 768->512 table, then `down[king] ->
mid[sym] -> up[mat] -> l2[mat] -> head[mat]` plus a skip and a psqt
passthrough. Selected with `--wdl`; verified end to end against the checkpoint
(LEDGER 050). All three 10B arms passed an SPRT [0,5] against the quadratic
eval at 10+0.1, int8, same binary both sides:

| arm | shape | Elo vs quad | games | val ce_out | nps |
|---|---|---|---|---|---|
| `gate-A` | 512, bneck 16 | **+84.6 [+61, +109]** | 402 | 0.72582 | 688,748 |
| `gateh-C` | 256 wide | **+96.5 [+71, +123]** | 384 | 0.73270 | 870,471 |
| `gateb8-D` | bneck 8 | **+53.9 [+35, +73]** | 624 | 0.72847 | 790,550 |

**`gateh` — half the accumulator — leads on Elo while scoring 0.0069 WORSE on
val**, the third time val loss has ranked these backwards. The intervals
overlap and the ranking is only transitive through quad; `tools/netcmp.sh` with
`gate-A` as reference settles it directly and has not been run.

The earlier −127.7 Elo (LEDGER 052) was this design on a kernel 2.2x slower
than it needed to be. Removing a bounds check from the matvec inner loop
(LEDGER 055) is the single thing that changed: one eval costs **949 ns instead
of 3059**, and i8 is free accuracy (−0.00002 val CE, LEDGER 053) worth +4%.
`chess evalprof` is the instrument. Another ~2x is available in f32; the int8
case is **memory, not instructions** — this CPU has no VNNI, so the `vpdpbusd`
argument that used to sit here does not apply (ROADMAP 4, LEDGER 059).

## A Leela teacher wins, and it is not smoothing

Training the same arm on lc0's WDL scores **0.76551 vs 0.76937 ce_out**, three
seeds each, no overlap — even though lc0's own ce_out is 0.029 *worse*. An
entropy-matched sigmoid does not reproduce it and softening the sigmoid makes
the student monotonically worse, so this is shape, not label smoothing.
LEDGER 058, which corrects 057 at the source.

**Next step is the full relabel: 893.5M positions, 15.5 GPU-hours** at 16,060
pos/s; tooling is `nnue/lc0/` plus `wdlarms.py --labels`. `ce_out` is a proxy.

## Contempt is a readout, not a retrain

`DrawValue` (UCI, or `--draw-value 0..100`) is what a draw is worth to the root
side: `E = W + v*D`, so **0 makes a draw count as a loss**. Root-relative, so
negamax stays consistent across a ply. Only possible because the eval is a
distribution — at `v=0` a dead-drawn K vs K moves −1365 cp, a balanced
middlegame only −116. Default 50 is bit-identical to the old readout.
**Symmetric 0 priced 2026-09-10: −188.5 [−229,−152]**, fixed 400, 14 draws —
it fights draws it should take. Asymmetric form since (uncommitted tree):
`--contempt-b` adds `b*weight*tanh(a*gap)` in cp, stm-relative, no root
tracking; `weight` is the normalised draw mass, or 1 with `--contempt-flat`.
A `v`-blend was tried first and DELETED 2026-09-10: numerically probed
globally incoherent (2%-drawish reads +418, its 18% neighbour −418).
Bench-exact at defaults (199091); 12/12 bench FENs move the predicted sign
under both weightings. **RUNNING tanh(100,1.0) vs plain, fixed 400 @8+0.08
conc 5**, same frozen binary both arms (`/tmp/qcontempt/chess-ct`
— predates the v-blend deletion, whose code path its flags never touch),
log `/tmp/contempt.log`, ~−50 interim at 204. Flat arm frozen
(`/tmp/qctflat/chess-ctf`), queued behind it.
**RESOLVED: tanh(100,1.0) loses −64.1 [−87,−42]**, fixed 400 (+43 =241 −116,
1836s; PGN re-score identical, pairs=200). Draw rate normal (60% vs 63% in
the Q-store SPRT) — unlike draw0's 3.5%, it doesn't destroy drawing. A third
of draw0's price for the same fighting idea.
**RUNNING flat(100,1.0) vs plain, fixed 400 @8+0.08 conc 5**, same frozen
binary both arms, log `/tmp/contemptflat.log`.
**RESOLVED: flat loses −46.3 [−70,−23]**, fixed 400 (+65 =217 −118; PGN
re-score identical). Same price family as tanh's −64 — the form doesn't
matter much, the cost is inherent to fighting.
**DIRECTION KILLED 2026-09-10.** Three arms priced (symmetric −189, tanh
−64, flat −46); fighting draws costs ~50+ Elo in self-play however it is
shaped. Code stays dormant (default-off, bench-exact, tested) — it costs
nothing to carry. Revisit only with an actual sharpness/complexity measure
to gate the weight on: the failure is indiscriminate fighting, and `D`
alone doesn't separate "dead draw" from "live position". Sibling-spread
(`library/001`) is the obvious candidate — the search already prices the
children, their disagreement is a sharpness readout waiting to be used.

## The sweep is over; the wins were the optimiser and the activation

`combo` 0.75661 -> `gate` 0.74033: **0.0026 was the learning rate**, **0.0060
the accumulator activation**. Architecture contributed almost nothing; removing
every crelu costs 0.063. Defaults are batch 16,384, gated fold, king buckets
only. **lr 2e-3 was a 250M result and does not transfer** — at 20B it saturates
the gate and destroys the run; use 6.7e-4 with wd 1e-2 for long runs and tune
the LR at the length you will actually run. LEDGER 051/052.

Not settled: HalfKA lost at 250M with 13.5M parameters — a budget result, not a
verdict. **All 893M positions are ONE month at stride 1**; LEDGER 021 said to
buy distinct games and it was never done (~106 GB unused).


## The profile is done, and the search's own code is not the problem

`chess nodeprof` prices every non-recursive leaf operation inside a node. In
the deploy config **eval is 64.5% of search time and the accumulator push
another 20.2%; movegen, make/unmake, the TT, ordering and the price list are
5.4% together.** Two things nobody had priced: the accumulator push **copies
4096 bytes of f32 per made move**, and **quiescence never reads the hash** —
the main search's stores reach 1.2% of q-nodes (dead) but **26.9% repeat a key
quiescence itself visited**, which only a store in quiescence unlocks.
LEDGER 059.

**The two diagnostic probes that produced those q-numbers now live behind
`--features qprobe`.** On `nodeprof` they sat outside every zone and their cost
fell into `control`, reporting it as 33.5% of the search when it is 10.4% —
059's table was right, its instrument had stopped reproducing it. LEDGER 060.

## Every search constant is tunable, and none of it ships

A `params!` macro generates the field, its default, its search box and the
`NAMES`/`get`/`set` registry from one declaration, so a constant cannot be
added without becoming tunable. **12 reachable names became 20.** `chess
params` lists them, `--spsa` prints an optimiser config, `--features tune`
exposes all twenty as UCI spins; a playing build has three options and cannot
move any of them. `tune`'s `--set/--a/--b/--free` reach all 20 as well.

## Tuning against positions instead of games

The loop is built and documented in **`library/012-tuning-without-games.md`**.
Corpus is 6.38M FENs from `fishpack32.binpack` (not our own games — that was
one of the three documented biases); 1000 of them labelled at 400k nodes live
in `data/tune/`. **A full scoring pass is 3.0
seconds.** Budgets can be nodes or deterministic **work units**
(`--features work`), per position or as a total for the pass.

**1000 positions resolve 4.1 cp at |t|=2** — a ~100 Elo floor at LEDGER 007's
exchange rate. That is deliberate: we are shopping for structural wins, not +1
Elo patches, and the corpus sample is nested so lifting `--n` later costs only
the new positions. Fixed work vs fixed nodes agree to 0.25 cp against a 2.06 cp
SE, so the work budget is a correctness fix that changes no answer yet.
LEDGER 061.

## Bringing information to the search: the platform

**`features!` in `src/search.rs` is how new information reaches the search**,
and `library/013` is the manual. One declaration gives a coefficient, its box,
its docs, its registry entry, its lazy extraction and its price term. Each
feature declares a **group** — `Sigma` / `Gap` / `Cost`, the three terms of
`library/004`'s allocation law — which fixes its sign before it is measured,
and a **cost class**: a `Gated` feature does not execute while its coefficient
is zero. `chess features` lists the surface; **30 tunable names**, 7 declared
features, 1 live.

Carrying a dormant feature is **unmeasurable** — +0.25% mean over 5 interleaved
pinned pairs, spread −2.6% to +3.7%. So declare freely; the bar to switch one
on is an SPRT.

Screened (LEDGER 062/065): **`c_see` won and shipped, +51 Elo.** Everything
else measures at or below zero — `c_offpv`, `c_qdrop`, `c_ply`, `c_depth`,
`c_incheck`, `c_gives_check`; `c_recap` fires in 22-68 of 1000 and is
unmeasured rather than zero.

**And so does every knob that was already there.** On the relabelled corpus at
the current default, `c_rank`, `c_rank_depth`, `skip_below`, `pv_discount`,
`c0` and `check_extension` are all worse in *both* directions, and `max_base`
is inert above 1000 (documented in its own doc comment; re-checked, not a
bug). Only `c_hist` reads slightly negative, on a handful of positions.

**Single-coordinate screening on root regret is exhausted at this operating
point.** The self-referential teacher was tested as the explanation — one
manipulation, teacher with vs without `c_see` — and shows no systematic
direction, so it is not the cause. This is the 4.1-cp floor, and it is the
strongest evidence yet for the per-child corpus below. LEDGER 065.

Two protocol corrections worth keeping: **do not match depth** (it is an output
of the search, not an input — matching it deletes the mechanism a feature works
through), and **compare against a tuned baseline** — `c0` alone is worth −2.2
cp, because the price list has never been fitted.

**A budget that binds on quiescence *depth* has nothing to allocate.**
Quiescence deeper than 8 plies does not exist: capping there is bit-identical
at 15k *and* at 60k, while the main horizon grew 2.6 plies between them. That
kills one design and nothing else — the boundary as an *indicator* is `c_qdrop`
and 27.9% of priced children land on the far side of it. LEDGER 063.


## Open research questions


- **Why does the worse teacher train the better student?** LEDGER 058 rules
  out label smoothing — an entropy-matched sigmoid does not reproduce the gain,
  and softening it monotonically hurts — so the value is in the shape of the
  distribution. What the net knows that (score, material) cannot express is
  still open, and the honest answer would be a feature-level probe rather than
  another arm.
- `c_gap` did **not** survive: it measures better at zero. That falsifies this
  gap feature, not the OCBA argument — `move_gain` has no exchange sequence, so
  a move that hangs a piece two plies later looks fine to it. An SEE-based gap
  is a one-line change and a rerun.
- Does regret at fixed nodes correlate with Elo *at all*? 008 is the first real
  test: the proxy committed to a number in advance.
- Does the sibling-spread prediction from `library/001` hold in real positions?
  Still open, and now directly relevant: OCBA says the price should carry
  `sigma_eval`, and 001 says that is the quantity that governs everything.

## Instrument traps (moved from STATE.md, 2026-09-28)

Each of these has already produced a wrong number once. Read before trusting a
measurement.

- **Any measurement on the 43-line book understates its SE by up to 2.2x**
  (LEDGER 111). Multiply by ~2.1 at R1 and 1.5 at R3 before quoting anything
  from LEDGER ≤112. `lich.epd` (4000 lines) is the default since 112. A/A nulls
  and θ are both blind to this, which is why no control ever caught it.
- **`chess bench` silently reports whatever eval it can find**: `$CHESS_WDL`,
  else `quad.nnue` beside the binary, else PeSTO with only an `info string`.
  It also ignores `--set`. Always pass `--wdl`/`$CHESS_WDL` and check the node
  count against a known value.
- **A build outside the repo needs `.cargo/config.toml` copied with it.**
  Without `target-cpu=native` the binary loses popcnt/BMI1/AVX and runs at
  **half nps** with unchanged node counts, so only the timing column lies
  (LEDGER 113). On the cluster use `-C target-cpu=x86-64-v3` instead (the desk
  is Zen 3, the cluster Zen 2).
- **`evalprof`'s per-stage column swings 24–50% run to run** on the same
  binary, while its total holds to 4.6% (LEDGER 106).
- **The gate once played the same stale net on both sides** (cp-0006 read
  +37.4, true +61.1). Fixed: it resolves its net from `publish/net.sha256` and
  hash-checks it (LEDGER 114).
- **`chess match` never respawns a hung opponent**, so one hang poisons a
  worker, which then burns through the job queue at zero cost and corrupts ~90%
  of the remaining games (3 hangs in 10,210 games; `matchplay.rs:866`).
  Deliberately unfixed; fix when it next bites.
