# 017 · The long-run map

_Written 2026-09-15 at cp-0004 (+ uncommitted 089/090 work). A backlog of
113 concrete items (most with variants a/b/c) to try, grouped by track, each detailed enough that an
agent can pick one up cold. It will be wrong in places; re-rank as results
come in, and correct items at the source rather than appending._

## How to use this

1. Pick an item. Read **its part only** (table below) and `01-protocol.md`,
   which holds the recipes every item refers to (freeze, bench check, SPRT,
   proxy screen, SPSA, GPU run, how to record).
2. Every item says its **class**: `NEUTRAL` (must leave bench 199091 exact),
   `SPRT` (changes play; build behind a default-off flag first), `PROBE`
   (measurement only, no engine change), `GPU` (training run), `DOCS`, `ASK`
   (Luka decides first).
3. Items marked **variants a/b/c** are meant to be tried as separate arms,
   not as one mixed change. One manipulation per arm.
4. Result goes in the ledger (`library/ledger/NNN-slug.md` + one line in
   `LEDGER.md`) whether it wins, loses or is inconclusive. Then mark the item
   here `[done → LEDGER NNN]`.

Priors marked [GUESS] are from engine-development folklore. This engine has
come in **below** such priors every time (continuation history +3 vs a
literature +10–30; correction history 0 vs +20–40), so read them as upper
bounds.

## The parts

| file | items | what |
|---|---|---|
| `017-long-run-map/00-progress.md` | — | item status: what is done, running, blocked, next |
| `017-long-run-map/01-protocol.md` | — | recipes: freeze, SPRT, screen, SPSA, GPU, record |
| `017-long-run-map/02-housekeeping-and-fixes.md` | HK-*, FX-* | tree, docs, 016 list; engine bugs and unfair test conditions |
| `017-long-run-map/03-tuning-and-cache.md` | TU-*, EC-* | SPSA in games, the proxy, the eval cache |
| `017-long-run-map/04-search.md` | SR-*, LC-* | standard search pieces; the learned-control bet |
| `017-long-run-map/05-wdl-net-speed.md` | WD-*, NN-*, SP-* | what WDL is worth; the net; engine speed |
| `017-long-run-map/06-revisit-and-measure.md` | RV-*, MS-* | killed results worth doubting; instruments |
| `017-long-run-map/probes/` | — | scripts behind LEDGER 091/092 |

## Measured while writing this (2026-09-15)

- **History tables reset on every move.** `go_parallel` builds a new
  `ThreadData` per `go` (`src/search.rs`, `ThreadData::new`), so history,
  killers, continuation history and the correction table start empty on each
  move. Every history-type idea (085, 089, O3, O5, O6) was tested that way.
  → FX-1, RV-1..RV-3. [FACT]
- **The match runner's default hash is 32 MB, which the A3 bug makes 16 MB.**
  Every SPRT on record ran on 16 MB. → FX-3. [FACT from code]
- **Both label files name the old teacher** (`gate-3ep-q8.nnue`, engine
  `b986f5b`); a 4k label set exists beside the 1k one. → MS-6. [FACT]
- **Eval cache for the tuner: 38–96% hits, 1.4–3.4x, ceiling 3.7x.**
  LEDGER 091. 34% of accumulator pushes are never read → SP-1.
- **The net's D is ~96% a function of E and material**; its residual carries
  a modest σ signal; σ no longer follows material on NNUE. LEDGER 092.
- `/tmp/sprt/` is gone (the S1/S2 frozen binaries and PGNs). → HK-4.

## The map

Three resources run in parallel: **the box** (games, ~780/h at 8+0.08 conc
5), **the GPU** (training), **the desk** (code, probes). Keep each busy.

```
Phase 0  (days)     HK-1 commit · FX-1 history persistence · FX-3/4 TT fixes
                    HK-4 durable run dir · MS-6 relabel from cp-0004
                         │
Phase 1  (1–2 wk)   TU-1..3 SPSA plumbing ──► TU-5 first SPSA campaign (box)
                    LC-1 per-child corpus (desk, while SPSA plays)
                    NN-1 unique data extraction + lc0 relabel (GPU)
                         │  gate: SPSA result SPRT'd; corpus reproduces
                         ▼
Phase 2  (2–4 wk)   SR queue (futility, SE, razoring, improving, ProbCut…)
                    one SPRT each; re-run SPSA every ~3 shipped changes
                    WD-1 scalar-vs-3-way (GPU), WD probes (desk)
                         │  gate: search pieces exhausted or all small
                         ▼
Phase 3  (month 2)  LC-2 two-stage gap · LC-3 σ term · LC-4 fitted price
                    LC-5 non-linear price — the project's actual bet
                         │
Phase 4  (month 2–3) NN-2..5 net v2 on unique data → re-tune (SPSA) because
                    every margin is in the new net's cp units
```

Standing rules for the map: after every checkpoint, relabel the proxy
corpus (MS-6) and re-run the last SPSA campaign's final SPRT if the change
touched pruning. Every ~3 checkpoints, a gauntlet (MS-9).

## First ten, in order

1. **HK-1** commit the tree (dormant, bench-exact).
2. **FX-1** persist history across moves → then **RV-1/RV-2** re-test
   correction and continuation history on top.
3. **FX-3 + FX-4** the TT hash-size and stale-move fixes, one gate.
4. **MS-6** relabel from cp-0004 + m1-b1; lift to 4k.
5. **TU-1..TU-3** params into `Params`, `--set` at UCI startup, `tools/spsa.py`.
6. **TU-5** SPSA campaign A (price list + whole-node), then its SPRT.
7. **SR-1** futility skip-quiets (already queued).
8. **SR-2** singular extensions (already built).
9. **LC-1** per-child corpus, built on the desk while 6–8 play.
10. **NN-1** unique games on the GPU.
