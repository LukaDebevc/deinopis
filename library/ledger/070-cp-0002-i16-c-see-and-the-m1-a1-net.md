# 070 · cp-0002: i16 accumulator, c_see pricing, m1-a1 net

_2026-09-08. `tools/checkpoint.sh -m "i16 accumulator, c_see pricing, m1-a1
net" --net nnue/runs/rr-20260908-103255/m1-a1.nnue`. Full gate, pushed._

Pre-gate hygiene: `git add -A --dry-run` showed the gate would have swept in
~100 regenerable `nnue/gametype32/results/*.json` screens and a stray
`nohup.out`. Removed the latter; added `nnue/gametype32/results/` to
`.gitignore` (source is committed, artefacts are ignored). Run
`commit.txt`/`speed.txt`/`results.tsv`/`ratings.txt` under `nnue/runs/`
ride along deliberately — the file's own comment keeps the conditions the
numbers were measured under.

## Result

Build ok (warnings only), `cargo test --release` pass, `perft-suite`
ALL PASS, bench **219127 nodes @ 670515 nps** (new fingerprint).

```
SPRT vs baseline [0,10]   8+0.08, conc 5, same m1-a1 net both sides
380 games: +95 =229 -56   +35.8 Elo  [+16, +56]   LOS 100.0%   LLR 2.96
SPRT: H1 accepted
```

Committed, tagged `cp-0002`, pushed; baseline for the next gate is this
build. CHECKPOINTS.md row written by the script.

## Decision

- The gate measures the **code** delta with the net held equal: **+35.8
  [+16, +56]** over cp-0001. The net delta was measured separately, same
  code both sides: **+40.1 [+19, +62]** (LEDGER 069). The combined jump from
  cp-0001-as-shipped was not measured directly; do not add the intervals.
- Riders: c_see and the whole dirty tree (tuner, feature platform, nodeprof,
  docs) ship inside this one checkpoint. The "two changes" framing understates
  it; the SPRT re-validates the pile as one build.
- Tree is dirty on top of cp-0002 by one line (`src/search.rs`: history read
  replaced with `0 // NOHIST`, 19:03, origin unknown) — not in the tag.
