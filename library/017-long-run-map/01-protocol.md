# 017/01 · Protocol: the recipes every item uses

Follow these literally. They encode mistakes that each cost a night once
(`CLAUDE.md`, memory "match harness gotchas", LEDGER 024/087).

## P1 · Before touching anything

```bash
cd "$(dirname "$0")/.."
git status --short          # changes you did not make? stop and ask
pgrep -fa 'chess match'     # a match running? never build into ./target
```
If a match is running, build with `CARGO_TARGET_DIR=/tmp/chess-target`.

## P2 · Build a change behind a flag (NEUTRAL commit)

1. New constant → a `params!` line in `search::Params` (default = today's
   behaviour, or 0 = off). New *information* for the price → a `features!`
   line with a group (`Sigma`/`Gap`/`Cost`), default coefficient 0.
2. `cargo test --release` passes.
3. `./target/release/chess bench --wdl nnue/runs/b1-20260909/m1-b1.nnue`
   prints **exactly 199091 nodes**. If not, the "off" state is not off: fix
   it. Never edit the expected number.
4. Movegen/`make_move`/board touched → `./target/release/chess perft-suite`.
5. Commit: `<ITEM-ID>: <what>` (e.g. `FX-1: persist history across moves`).

## P3 · Freeze binaries (durable, not /tmp)

```bash
RUN=$HOME/chess-runs/$(date +%Y%m%d)-<item-id>; mkdir -p $RUN
cp target/release/chess $RUN/chess
cp nnue/runs/b1-20260909/m1-b1.nnue $RUN/
sha256sum $RUN/chess $RUN/m1-b1.nnue > $RUN/SHA256
$RUN/chess bench --wdl $RUN/m1-b1.nnue | tail -1 >> $RUN/SHA256   # fingerprint
```
`/tmp` is wiped on reboot (it took the S1/S2 binaries). Both arms use the
**same binary** when the change is a flag or a `--set`; two binaries only
when a default had to change in code.

## P4 · SPRT (the ship decision)

```bash
cd $RUN
setsid nohup ./chess match \
  --engine   "$RUN/chess --wdl $RUN/m1-b1.nnue <FLAG_OR_SET_A>" \
  --opponent "$RUN/chess --wdl $RUN/m1-b1.nnue" \
  --tc 8+0.08 --concurrency 5 --sprt 0,5 --games 3000 --hash 64 \
  --pgn $RUN/games.pgn > $RUN/sprt.log 2>&1 &
```
- Name the eval on **both** sides. A missing net silently plays PeSTO
  (bench 233103).
- `setsid`: a match tied to the launching shell dies with it (087).
- Poll with short `tail -3 $RUN/sprt.log`; never `sleep` in a tool call.
- Afterwards: `./chess elo $RUN/games.pgn` re-score must agree with the log.
- Default bounds `[0,5]`, cap 3000. For a bundle or an SPSA result, `[0,5]`
  still. For a pure speed change with no behaviour change, `[0,3]`.
- Hand-kill only with a written reason (see 085/089: fade shape, interval).

## P5 · Proxy screen (cheap, 3 s/pass, promotes nothing)

```bash
L=data/tune/<current>.labels
./target/release/chess tune compare --labels $L --nodes 15000 \
   --wdl nnue/runs/b1-20260909/m1-b1.nnue --a <k>=<default> --b <k>=<v>
```
Rules (013, 086): sweep ≥4 values, not one; only **vs-default** comparisons
promote, and only with |t| > 2 **and** a smooth neighbourhood; neighbour
comparisons diagnose shape only. Regret may reject; games promote.

## P6 · SPSA in games

See `03-tuning-and-cache.md` TU-3 for the driver. Hyper-parameters, fixed
unless the item says otherwise (OpenBench conventions):
- iterations `N` = 15k–30k game pairs; `α = 0.602`, `γ = 0.101`,
  `A = 0.1·N`.
- per parameter: `c_end` = the step that should cost ~2–3 Elo (start at
  1/20 of its box width); `R_end = a_end / c_end² = 0.002`.
- each iteration: draw Δ ∈ {±1}ᵖ, play one game **pair** (same opening,
  colours swapped) θ+cΔ vs θ−cΔ, `r` = θ+'s score − θ−'s score ∈ [−2, 2];
  `θᵢ += a_k · r / (c_k · Δᵢ)`; clamp to the box; integers rounded only when
  sent to the engine.
- log θ every 500 pairs; plot each parameter's trajectory; a parameter still
  drifting at the end is not converged — say so.
- **The tuner's own numbers are not a result.** Finish with P4: final θ vs
  default, `[0,5]`, same binary, `--set` on one side.

## P7 · GPU run

Cluster work goes through the `cluster` skill. Every run records: data file
+ `--pool-end`, frozen val file, steps × batch (must be ≤ unique positions,
memory "one epoch"), lr/wd, seed, parent checkpoint. Val differences under
~0.05% are not rankings (044); val has ranked nets backwards against Elo
three times (052/056) — **the clock decides** (`tools/netcmp.sh`, equal time).

## P8 · Probes

A probe changes no engine behaviour and must show its instrument works
before its number is read: a self-replay or known-answer case (e.g. 091's
100% self-hit, 092's mate-in-1 parse check). Read ~10 raw rows before any
summary. Scripts go in the repo next to the item, not in `/tmp`.

## P9 · Recording

- Ledger entry: setup line (date, commit/binary sha, net sha, TC, games,
  concurrency), result table, reading (label [FACT]/[INFERENCE]/[GUESS]),
  decision. One line in `LEDGER.md`.
- Finished study code → `nnue/attic/` or deleted with a
  `Code removed in <sha>` line in its entry (016 B rules).
- `STATE.md` gets a **pointer**, never the prose. Budget ~150 lines.
