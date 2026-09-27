# 092 · The net's D is 96% a function of E and material; σ no longer follows material

_2026-09-15. Two exploratory probes (not pre-registered), scripts in
`library/017-long-run-map/probes/`. 20,000 FENs sampled (`shuf`, fixed
source) from `fishpack-gs401.fens`; m1-b1 net via `chess evalfen`, which
prints the raw (L, D, W) logits. Second probe: the first 6,000 of those
searched at 2k and 64k nodes through UCI (`go nodes`, `ucinewgame` between
positions), cp-0004-behaviour binary. The sample contains no |score| > 3000
and no mate scores (checked: the parser reads `score mate 1` correctly on a
mate-in-1), so this covers playable positions only._

## Probe 1: what D knows beyond E and material

Boosted trees (sklearn HGB), fit on half, R² on the held-out half:

| predict D (draw probability) from | R² | residual sd |
|---|---|---|
| E = W + D/2 only | 0.649 | 0.142 |
| E + non-pawn material + pawn count | **0.958** | 0.050 |
| E + 10 piece counts | 0.962 | 0.047 |
| E + counts + 768 occupancy bits | 0.971 | 0.041 |

**At inference the net's D is ~96% predictable from (E, material)** — the same
structure LEDGER 048 found in Stockfish's WDL. At most ~4% of D's variance is
position-specific. Anything the search does with D is therefore mostly a
material rule, unless that residual predicts something.

Readouts diverge exactly where D is high: the slope of `A/2 = log(W/L)/2`
against `logit(E)` is 0.89 for D < 0.3 and **7.6 for D ≥ 0.8**. (066 already
priced an A-like readout: −62.)

## Probe 2: what predicts the error of a shallow search (σ)

sd(V2k − V64k) overall **107.5 cp** (library/004 on PeSTO: 205.6 at 2k vs a
512k reference — the references differ, so compare loosely). By quintile:

| predictor | Q1 … Q5 sd (cp) | Q5/Q1 | top/bottom half within npm quartiles |
|---|---|---|---|
| non-pawn material | 118 105 118 108 88 | **0.75** | — |
| D | 132 101 96 108 95 | 0.72 | 0.95 0.74 0.80 0.75 |
| D residual after E + counts | 121 126 112 86 85 | 0.70 | 0.72 0.67 0.95 0.76 |
| \|E − 0.5\| | 99 96 93 108 135 | 1.36 | 1.07 1.37 1.16 1.21 |
| \|static − V2k\| | 76 89 88 104 157 | **2.06** | 1.38 1.50 1.54 1.74 |

r² of ln|dev| on held-out half: log npm 0.020; + D 0.087; + D residual 0.042;
+ log|static − V2k| 0.127; everything 0.145 (ceiling for this target is
~0.29, library/004).

## Reading

1. **library/004's phase law is gone on the NNUE eval**: material spans
   Q5/Q1 0.75 here against 0.13 on PeSTO. 004 predicted this ("expect this
   coefficient to move substantially after NNUE"). Corrected at the source.
2. **Disagreement between two shallow views is still the best σ signal**,
   and it survives the material control — the same finding as 004's
   |qsearch − 1k|. This is what a two-stage price would read.
3. **D's position-specific residual carries a modest σ signal**: positions
   the net calls drawish, beyond what E and material say, have ~20–30% less
   shallow-search error. Sign fits the `Sigma` group: drawish → price higher.
4. |E − 0.5| is partly a unit effect — cp is a logit, so the same error in E
   reads larger in cp far from 0.5. Do not read it as uncertainty.

Caveats: one sample, V64k shares the engine's own errors (understates σ),
fishpack positions are not tree positions, no decisive positions.

## Decision

- WDL's value is most likely in **training** (058), not at inference. The
  clean test of "does the 3-way head do more than a scalar" is map WD-1.
- D-in-search ideas must beat a material-only control (map WD-2..WD-6).
