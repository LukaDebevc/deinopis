# 099 · Speedups: TC ladder + adjudication sim; TU-5 relaunched at 2+0.02

_2026-09-17. Two speedup questions (box was locked for ~2.5 d by TU-5 at
5+0.05): do faster TCs measure the same thing, and can games end sooner?
Campaign parked at 23 pairs first (spsa.csv kept, resumed later).
Run dir `~/chess-runs/20260917-tc/` (frozen cp3/cp4 verified 243605 /
199091, m1-b1 `ce1f2656`)._

## Adjudication was already on — the prize is small

The runner adjudicates two-sided by default (resign ±900cp/8 plies, draw
±8cp/16 plies past move 40, max 600). In the cp-0005 gate PGN, 2405/3786
games ended adjudicated and only 2 checkmates played out.

`tools/adjsim.py` (committed) replays PGN `{score/depth time nodes}`
comments through an exact mirror of `adjudicate()`: the CONTROL ruleset
reproduces all 2405 endings with **0 flips**, so the mirror is exact
(PGN rounding ±1cp caused zero deviation).

| tightening | think-time saved | flips (simulated ≠ actual) |
|---|---|---|
| resign 500cp/6 | ~6% | 138/3786 — 137 are draws called as wins |
| draw 12cp/12 plies | ~3% | **1** |

- [INFERENCE] The 137 draws-called-wins inflate the Elo scale ~4 (same
  mechanism as 096's UHO scale effect — manufactured decisive games). ~6%
  speedup for measurable bias: rejected. The resign rule stays 900/8.
- [FACT] Draw 12/12@40 is nearly free (+3.2% plies, 1 flip). **Shipped**
  (match-only default change, bench 219718 exact, all tests pass, 100-game
  A/A smoke at 0.5+0.01: −28 ± 45, zero errors, all paths firing).
- [FACT] Late moves are fast: resign-500 saves 7.2% of plies but 5.9% of
  think time. Plies overstate adjudication's prize.

## TC ladder: throughput multiplies, the ruler stretches, the gap grows

cp4 vs cp3 (reference gap +32.5 @8+0.08, LEDGER 096), fixed games, conc 5,
hash 32, builtin book, sequential rungs. Zero forfeits/errors at every rung.

| TC | score (cp4) | Elo | draws | wall | games/h |
|---|---|---|---|---|---|
| 8+0.08 × 300 | +74 =193 −33 | +47.8 [+25, +72] | 64% | 1103 s | 980 |
| 2+0.02 × 300 | +90 =170 −40 | +58.5 [+33, +85] | 57% | 285 s | 3790 |
| 0.5+0.01 × 400 | +158 =181 −61 | +86.0 [+61, +112] | 45% | 125 s | 11520 |
| 0.2+0.005 × 400 | +188 =148 −64 | +111.4 [+83, +141] | 37% | ~100 s | ~14400 |

Latent-gap fit (096's `N(b+Δ,1)` cut at ±θ, per rung W/D/L):

| TC | θ | Δ | I/game | info/hour |
|---|---|---|---|---|
| 8+0.08 | 0.96 | 0.27σ | 0.749 | 730 |
| 2+0.02 | 0.82 | 0.29σ | 0.781 | 2960 (**4.0x**) |
| 0.5+0.01 | 0.65 | 0.38σ | 0.789 | 9090 (**12.4x**) |
| 0.2+0.005 | 0.54 | 0.46σ | 0.772 | ~11100 (**~15x**) |

- [FACT] I/game is flat (~0.78 — the 096 ceiling again: sharper positions
  rescale, they don't add information). All the gain is games/hour.
- [FACT] Anchor reproduces 096 within noise (+47.8 vs +32.5, ~1σ combined).
- [INFERENCE] At 2+0.02 the latent gap agrees with the anchor (0.29 vs
  0.27σ) — the Elo rise 48→58 is pure ruler-stretch as draws fall. At 0.5
  Δ reads 1.3σ high, at 0.2 2.2σ high: either the gap is really bigger at
  bullet (tactics/conversion rewarded) or the probit model misreads
  bullet's heavier tails. Unseparated — so the campaign does NOT go there.
- [DECISION] ~~TU-5 runs at **2+0.02**: transfer evidence clean.~~
  **OVERTURNED by LEDGER 108 (2026-09-19).** The shape evidence this entry
  asked for was run — 5 arms x 4 rungs, 147,000 games — and Δ is **not**
  TC-invariant. A single stretch factor across arms is rejected at χ² = 10.76
  on 2 dof (p ≈ 0.005). The "2+0.02 agrees with the anchor" reading above
  rests on Δ = 0.29 vs 0.27 at n=300 (SE ≈ 0.09), which could not have seen
  the effect: at n=3000 the same comparison reads **+0.368 ± 0.021 vs
  +0.269 ± 0.031, a 0.099 ± 0.038 gap (2.6σ)**. **TU-5 campaign A is blocked**
  — its SPSA objective is match results at 2+0.02, and FUT is worth 1.9x more
  at 8+0.08 than at 2+0.02. The ruler-stretch model in the table above is
  *confirmed* (stretch 1.48–1.52 across arms); it is the residual latent gap
  that fails to transfer. See `library/ledger/108-tc-transfer-ladder.md`.

## TU-5 relaunched

`spsa.py ... --tc 2+0.02` resumed in `~/chess-runs/20260916-tu5/campaign-a`
from pair 24 (first 23 at 5+0.05 — negligible at N=20000, schedule k
continuous). New binary carries the draw-12 rule on both arms (symmetric).
Tripwire from 098 stands: wall-to-wall blowouts at pair ~500 → halve
`skip_below`/`fare` c_end and restart while k is small.

Side note: `exe_fingerprint` hashes the whole binary, so this match-only
rebuild changed the label-cache key though the search is identical. Next
`label` invocation on old files should pass `--reuse` (its designed case:
diff provably cannot reach the search) — or MS-6 re-runs in 36 min on a
quiet box.
