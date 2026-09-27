# 097 · cp-0005: bundle persist-all + conth + easy-move gate, +7.5 over 3776, H1 at [0,5]

_2026-09-16. `tools/checkpoint.sh -m "bundle persist-all + conth +
easy-move (LEDGER 094) [0,5]" --net nnue/runs/b1-20260909/m1-b1.nnue`.
Full gate, pushed as `cp-0005` (`b4f6008`). Net sha `ce1f265647c060a2`
(same m1-b1 both sides — the gate measures the code delta with the net
held equal). Candidate flips the three 094 defaults on the FX-3/4-less
tree (revert `9ad3a64` landed first): `persist-hist` default 0→1,
conth off→on (`--no-conth` / `CHESS_CONTH=0` is the off-ramp),
`easy_stable` 0→2. Baseline is the cp-0004 binary rebuilt from the tag.
TC 8+0.08, conc 5, 32 MB hash, builtin 43-line book, both arms as
subprocesses. Run artefacts: `.checkpoints/last-match.txt` (log),
`.checkpoints/last.pgn`, tag message carries bench + net sha._

## Result

Build ok, `cargo test --release` pass, `perft-suite` ALL PASS (gate
order; no failures to record), bench **199091 → 219718 nodes**
(@ ~844k nps in the tag; 820k on re-run — box load, not a measurement).
The bench move is the conth default: persist and easy-move never touch a
node-limited search.

```
SPRT vs baseline [0,5]   8+0.08, conc 5, same m1-b1 net both sides
3776 games: +543 =2772 -461   +7.5 Elo  [+2, +13]   LOS 99.5%   LLR 2.95
SPRT: H1 accepted — 13646 s wall
```

PGN re-score (`chess elo --name candidate .checkpoints/last.pgn`):
3786 games +546 =2778 −462, +7.7 [+2, +13] — the 10 extra games are the 5
in-flight pairs finished after the decision (same pattern as 094's 64 at
conc 32). Elo Δ=0.2, agrees.

Committed, tagged `cp-0005`, pushed; baseline for the next gate is this
build. CHECKPOINTS.md row written by the script.

## Reading

- [FACT] This is the 094 bundle (+11.2 [+4, +19], 2318g, cluster, scaled
  13+0.13) re-measured as a default-flip on the desk at 8+0.08: +7.5
  [+2, +13], 3776g. Intervals overlap; both accept H1 at [0,5]. The gate
  is the ship decision, 094 was the candidate selection — do not quote
  them as two independent gains.
- [INFERENCE] +7.5 is biased high, same as every gate number in
  CHECKPOINTS.md: an SPRT that stops on H1 stops on an upward run (096's
  side effect measured this directly on cp-0004: +61.4 at the gate vs
  +32.5 over 1200 fixed games). The smaller the game count the larger
  the bias; at 3776 games this is the least-biased gate yet, but it is
  still a crossing, not a fixed-sample estimate.
- [FACT] Attribution is muddy by design (MS-3, carried over from 094):
  three things ship as a unit. Component sum was +13.0 (089 +3.2, 090
  +4.1, 093 +5.7); the bundle readings (+11.2, +7.5) are consistent with
  plain additivity, which says "no large negative interaction" and
  nothing finer.
- [FACT] Ship tree is the FX-3/4-less tree: the 095 H0 (−4.6, 2622g)
  stayed out, revert verified by the 199091 bench before the flip.

## Decision

- **cp-0005 is the baseline.** Next Elo work starts from its binary.
- Unblocks the queue behind the gate (00-progress "Blocked on the box"):
  TU-4 (`c_end` from proxy) → TU-5 campaign A, RV-1/RV-2 (runnable with
  `--persist-hist=1` on both arms per 093 — now the default, so run them
  with `--persist-hist=0` on both arms to isolate the rule from the
  default), SR-1/SR-2. MS-6 relabel still waits for a quiet box.
