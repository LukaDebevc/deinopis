# 094 · Bundle persist-all + conth + easy-move: +11.2 over 2318, H1 at [0,5]

_2026-09-15. The three small positives run together (MS-3 bundle, extends
FX-2): A `--persist-hist=1 --conth --set easy_stable=2` vs B base, same
frozen binary both arms, m1-b1 net (`ce1f2656`) both sides, builtin book,
--hash 64 (32 MB effective — FX-3 unfixed in this tree, equal both sides).
Binary: HEAD `e2b662f` with FX-3/4 (`0c28087`) reverted, built with
`target-cpu=x86-64-v3`, bench **199091** — the pre-FX-3 tree S1/S2/FX-1a ran
on, plus TU-2's `--set` at UCI startup. Played on the GPU cluster (2x EPYC 7742),
conc 32, `nice 19`, node load ~150/256 threads. **TC 13+0.13** there, scaled
from 8+0.08 by bench nps (510k cluster vs 808k desk) so nodes per move match
the component runs. Run dir `~/chess-runs/20260915-stack/`._

Flag checks on the frozen binary before launch: `info string params:
easy_stable=2` (a misspelt name exits 2); `--conth` moves bench position 1
26358 → 29479 nodes; `--persist-hist=1` leaves search 1 of a game identical
and moves searches 2-3 (18052 → 18532, 16702 → 17718).
**The pre-TU-2 binary (`3e907cd`, the gate's `chess-old`) silently ignores
`--set`** — the bundle as first queued would have run without easy-move.

## Result

| games | score | Elo | LLR |
|---|---|---|---|
| 2318 | +379 =1635 −304 | **+11.2 [+4, +19]** | 3.01 (bounds ±2.94) |

H1 accepted, LOS 99.8%, 2256 s wall. PGN re-score (`tools/rescore.sh`)
+11.09 [+3.7, +18.5] over 2382 games — the 64 extra are the 32 in-flight
pairs finished after the decision (rescore's 2-game tolerance assumes
conc 5). Colours balanced, 1191 each. Zero engine errors, zero time
forfeits; 500-game interim was +6.9 [−10, +24].

## Reading

- Sum of the components is +13.0 (089 +3.2, 090 +4.1, 093 +5.7); the
  bundle's +11.2 is consistent with plain additivity. The sum's own 95%
  interval is about ±11, so this says nothing finer about interactions than
  "no large negative one".
- [INFERENCE] +11.2 is biased high: an SPRT that stops on H1 stops on an
  upward run. The components ran to their cap, so they carry less of this.
- Conditions differ from the components: other hardware, conc 32 under
  shared load, scaled TC. That matters for comparing magnitudes, not for
  the A-vs-B verdict — both arms share every one of them.
- Attribution is muddy by design (MS-3): this ships three things as a unit.

## Decision

- **Ship candidate.** Next step is flipping the three defaults in code
  (bench moves: conth changes node counts; persist and easy-move do not
  touch node-limited search) and running `tools/checkpoint.sh` against
  cp-0004. This SPRT is not the gate — the checkpoint is.
- Gate for FX-3/4 resolved the same night: H0, −4.6 [−11, +2], 2622 games
  (`~/chess-runs/20260915-fx34gate/`), so the tree to ship on is the one
  this ran on.
