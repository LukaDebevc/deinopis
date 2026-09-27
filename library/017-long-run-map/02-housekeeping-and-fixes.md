# 017/02 · Housekeeping (HK) and fixes (FX)

## HK · Housekeeping

**HK-1 · NEUTRAL · commit the working tree.** Uncommitted: LEDGER 089/090,
091/092, this map, the dormant `--conth`, easy-move, singular-extension,
`qevade_see` and futility-skip-quiets code. All bench-exact. Check bench
199091 and `cargo test --release`, then one commit per logical piece
(`089/090 entries`, `dormant search flags`, `017 map + 091/092`).

**HK-2 · DOCS · STATE.md to ≤150 lines** (now 422). Keep: where we are
(cp-0004, 3246 gauntlet), what is running, the next 5 from 017's "first
ten", pointers. Move every dated session block (2026-09-07 … 09-13) into
`library/018-session-diary.md`, newest first, unchanged text. Delete from
"Next" the items that already shipped (c_see, i16 — both in cp-0002;
Q-store — cp-0004). The "What is verified" table moves to `ENGINE.md` or
the diary; STATE keeps one line pointing at it.

**HK-3 · DOCS · ROADMAP.md to ≤100 lines** (now 184). ROADMAP becomes the
ordered list of *tracks* with one line each pointing into 017. Move 4a–4d
(quantiser, profiler, profile-says, budget-ratio) into `library/010`.

**HK-4 · NEUTRAL · a durable run directory.** Create `~/chess-runs/`, write
its README (one screen: layout `YYYYMMDD-item/`, what a run dir must hold —
binary, net, SHA256 with bench fingerprint, log, pgn). Update
`tools/checkpoint.sh`, `netcmp.sh`, `sprt3.sh`, `wdlmatch.sh` to default
there instead of `/tmp`. Point STATE at it.

**HK-5 · DOCS · a label inventory.** `chess-data/tune/README`: every
`.labels` file, its teacher header, n, date, and whether it is current.
Today: `fishpack-1k.labels`, `-1k-see`, `-4k` all name `gate-3ep-q8` and
engine `b986f5b` — stale (MS-6).

**HK-6 · DOCS · fold the stray top-level docs** (016 I2): `ORDERING.md` and
`OPTIMIZATIONS.md` into `library/` (ordering is retired; optimizations is a
reference), `START.md` into `CLAUDE.md` or its file table. Private
`README.md` becomes a 5-line pointer (016 I3).

**HK-7 · NEUTRAL · work the 016 cleanup list in its order**, one item per
commit, with the classes it gives. Specifically:
- a. A10 (ASK: net auto-discovery or `include_bytes!`), A11 (stop race fix).
- b. **B1–B5 now** (dual tier, qnoise, adaptive.rs, IID ramp, whole-node
  futility return-static). **Hold B6/B8** (correction, continuation
  history) until RV-1/RV-2 run — they are the code those re-tests use.
  B7 (`--pot`, `off_us/off_them`) ASK. B9 (quad + old deep net) ASK.
- c. C1 split `search()` — **after** the SR items that edit the move loop
  land, or the split will conflict with every one of them. Check nps (E0).
- d. C2 split `main.rs`; C3 one `Config`; C4 eval settings out of globals
  (needed before in-process SPSA); C5–C13.
- e. F1–F6 tests. F1 (TT size) goes in with FX-3.
- f. G1–G7 trainer hygiene; H1–H5 tools.

**HK-8 · NEUTRAL · a CI-style check script** `tools/verify.sh`: build,
`cargo test --release`, bench == fingerprint in `ENGINE.md`, clippy
`-D warnings` once C12 is done. Every agent runs it before committing.

**HK-9 · DOCS · `ENGINE.md` "deliberately NOT in here"** — update after
each item that ships or dies (016 I4).

**HK-10 · NEUTRAL · the bench fingerprint table** in `ENGINE.md` and
`OPTIMIZATIONS.md` disagree (OPTIMIZATIONS still lists 234598/227236/248619
from before cp-0002). Keep one table, in `ENGINE.md`; the other points to it.

**HK-11 · DOCS · memory hygiene.** `MEMORY.md` entries "Chess tuning plan"
and "Price-list feature quality" predate 065/092; update them when TU/LC
items land (σ phase law is corrected by 092).

## FX · Fixes: bugs, and tests that were run under unfair conditions

**FX-1 · SPRT · persist history across moves.** Today `go_parallel` builds
`ThreadData::new(id)` for every `go`, so history, killers, continuation
history and the correction table are empty at the start of every move.
Do: keep one `ThreadData` per thread inside the engine (`Engine` in
`uci.rs` owns a `Vec<ThreadData>`; `go_parallel` takes `&mut` to them),
clear it on `ucinewgame` only. Node-limited searches (`tune`, `bench`) keep
constructing fresh ones, so bench stays exact and the tuner stays
reproducible. Variants, as separate arms:
- a. persist everything unchanged;
- b. persist, but halve history (and conth) at the start of each move;
- c. persist, killers cleared (killers are per-ply and refer to the
  previous root, so they may be noise).
[GUESS] +5–15. After it ships, run RV-1, RV-2, RV-3 on top.

**FX-2 · SPRT · bundle the two small positives.** Continuation history
(+3.2 [−4,+10], 089) and easy move (+4.1 [−2,+10], 090), on together vs
off, same binary. If effects add, ~+7 resolves in ~2000 games. Run after
FX-1 (conth changes meaning once it persists).

**FX-3 · SPRT · TT allocates half the requested hash** (016 A3):
`(bytes/16).next_power_of_two()/2` gives 32 MB for 64 and 16 MB for the
match runner's default 32. Fix: largest power of two ≤ n. Add F1's test
(`size_bytes() == 64 << 20`). Gate together with FX-4. Bench moves (tables
double), so it is an SPRT, not neutral.

**FX-4 · SPRT · fail-low store keeps another position's move** (016 A4):
keep the old move only when the old entry's key matches.

**FX-5 · SPRT · TT buckets** (016 D5). Variants: a. 2 entries/bucket, b. 4
entries (one cache line), replacement = lowest `depth − 4·age_delta`; c. 4
entries with quiescence entries always the first victims. Since cp-0004
quiescence stores twice per node, replacement pressure is new. Measure
first: `hashfull` after a 10+0.1 game move, and fraction of main-search
probes that miss because a q-entry displaced them (probe counter under
`--features qprobe`).

**FX-6 · SPRT · play the unfinished last iteration's move** (016 D2) when
the first root move has completed and a later one beat it.

**FX-7 · NEUTRAL-ish · one legal move → play at once** (016 D4). Changes
games only, not bench.

**FX-8 · PROBE then SPRT · repetition before the root** (016 D6): a single
earlier occurrence is scored as a draw. Check `ARCHITECTURE.md` for intent.
Probe: count games in the last SPRT PGN where the engine avoided or took a
repetition at ≥ +100 cp. Variants: a. two-fold in-tree only, three-fold for
pre-root history (standard); b. two-fold everywhere (today).

**FX-9 · NEUTRAL · move overhead option** (016 D3). Local matches do not
need it; a lichess bot or CCRL submission does.

**FX-10 · PROBE · the systematic engine-vs-torch offset** (072 caveat: f32
mean 5.9e-3 logits, max 12.7 cp, reproduces on cp-0002 and m1-a1, "needs its
own entry", never written). A 12.7 cp eval disagreement could be a feature
or bucket edge case. Do: `verify_wdl.py` on 2000 fens, sort by |Δ|, read the
20 worst positions, find what they share (in check? castling? en passant?
promotion piece? a king-bucket boundary?).

**FX-11 · SPRT · the stored-eval-0 trap** (016 C9) is neutral today; fix it
with a flag bit when TT buckets (FX-5) change the entry layout anyway.

**FX-12 · PROBE · the builtin book** [done → LEDGER 096]. The repeated
openings cost nothing: 600 games off the 43 lines are 598 *distinct* games,
clock jitter decorrelating the repeats. An unbalanced book was measured at
the same time and does not resolve faster either.
