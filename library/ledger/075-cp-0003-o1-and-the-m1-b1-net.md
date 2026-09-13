# 075 · cp-0003: o1 SEE-split ordering, m1-b1 net

_2026-09-09. `tools/checkpoint.sh -m "o1 SEE-split ordering, m1-b1 net"
--net nnue/runs/b1-20260909/m1-b1.nnue`. Full gate, pushed._

Pre-gate: A-vs-B SPRT (LEDGER 073) picked O1+m1-b1 over base+m1-b1,
+21.8 [+11,+33], H1 over 1164 games — the net-only candidate was dropped
before the gate, not in it.

## Result

Build ok (warnings only), `cargo test --release` pass, `perft-suite`
ALL PASS, bench **243605 nodes @ 690292 nps** (new fingerprint).

```
SPRT vs baseline [0,10]   8+0.08, conc 5, same m1-b1 net both sides
684 games: +135 =455 -94   +20.9 Elo  [+7, +35]   LOS 99.8%   LLR 2.99
SPRT: H1 accepted
```

Committed, tagged `cp-0003`, pushed; baseline for the next gate is this
build. CHECKPOINTS.md row written by the script. Net sha256
`ce1f265647c060a2` in the tag message (nets are gitignored).

## Decision

- The gate measures the **code** delta with the net held equal: **+20.9
  [+7, +35]** over cp-0002 (agrees with 073's +21.8 on frozen binaries).
  The net delta was measured separately, same O1 code both sides: **+22.6
  [+4, +42]** (LEDGER 072). The combined jump from cp-0002-as-shipped was
  not measured directly; do not add the intervals.
- Riders: the `checkstats` probe wiring (cfg-gated, absent from the playing
  build), ORDERING.md + `library/014` + ledger 067-074, run provenance
  (`setup.txt`/`speed.txt`/`results.tsv`/`ratings.txt`/`commit.txt`).
  LEDGER 074 (IID budget ramp) came from the working tree and is unrelated
  to this gate.
- Tree is clean on top of the tag.
