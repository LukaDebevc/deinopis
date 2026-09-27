# 098 · TU-4 c_end screen on the MS-6 labels: 16 params, step sizes set

_2026-09-16. `tools/tu4_screen.sh` (committed `4c42284`, parse fixed
after): for each TU-5 campaign-A parameter, `tune compare` default vs
default±step at 2–4 step sizes on `fishpack-4k-cp5.labels` (MS-6: 4000
positions, teacher = cp-0005 tree + m1-b1, LEDGER 097's tree) at 15k
nodes, `--threads 5`. ~110 compares, all vs-default. Full table:
`~/chess-runs/20260916-tu5/` holds the campaign; raw screen
`~/chess-runs/20260916-tu4/screen.tsv`. SE is ~0.85 cp across params
(MS-6 expected ~1.0)._


## Instrument checks (P8 — before reading)

- [FACT] `max_base` +steps (up to 5000) read +0.000 ± 0.000, 0 positions
  differ — inert above 1000, reproducing 065 on a new teacher. `max_slope`
  +side likewise. The screen sees what 065 saw.
- [FACT] Clamp-flagged rows read identical to their same-value twin
  (`aspiration` −40/−80 both clamp to 4: +0.771 twice; `delta_margin`
  −120/−240 both clamp to 0: +0.978 twice). `Params::set` clamps; the
  flag column is honest.
- [FACT] `fare` → 28 reads +78.9 cp, `skip_below` +200 (→72) reads +8.4 —
  cliffs with smooth bad neighbourhoods below them, not noise spikes.

## c_end readings (step where |regret| ~= 1 SE; sign: + means worse for the step)

| param | default | screen | c_end |
|---|---|---|---|
| `c_rank` | 250 | −200: +2.12 (t≈2.5); +200: −0.31 | 200 |
| `c_rank_depth` | 204 | ±200: +1.25/+1.27 (t≈1.4 both) | 200 |
| `c_hist` | 100 | never reaches 1 SE (max −0.62±0.47 at +800) | 400* |
| `pv_discount` | 1000 | flat to ±3200 (max +1.65±0.90) | 800* |
| `max_base` | 1000 | +side inert to 5000; −1000: +0.82±0.81 | 1000 |
| `max_slope` | 1000 | +side inert; −500: +0.46, −1000: +4.07 | 650 |
| `skip_below` | −128 | ±50 ≈ 1 SE; +100: +1.49; +200: +8.35 cliff | 50 |
| `fare` | 128 | ±25 ≈ 0.7–1 SE; −100 (→28): +78.9 cliff | 25 |
| `c_see` | 4000 | −1200: −0.96 (t≈1.6); −2400: −0.02 (non-monotone) | 1000 |
| `rfp_margin` | 75 | ±30 quiet; ±60: +1.78/+0.92 | 45 |
| `rfp_max_depth` | 7 | ±1: 2–8 positions differ; −4 (→3): −0.17±0.38 | 1* |
| `nmp_base_reduction` | 3 | ±1: −0.55/−0.71 (t≈1.5/1.3, both better) | 1 |
| `nmp_depth_divisor` | 4 | −1 (→3): −0.54 (t≈1.5); rest flat-ish | 1 |
| `aspiration_window` | 25 | +80: +1.45 (t≈1.8); −side quiet | 40 |
| `check_extension` | 1000 | ±400: −0.74/−0.84 (≈1 SE, both better); +800: +2.43 | 300 |
| `delta_margin` | 100 | −60: +1.58 (t≈2.0); +240: −1.95 (t≈2.4) | 60 |

Starred = proxy-blind, P6 1/20-box fallback; games decide.

## Reading

- [INFERENCE] Three spots read "both directions better" (`nmp_base` ±1,
  `check_extension` ±400, `delta_margin` +side): that shape is a minimum
  off-default or asymmetric noise, and P5 promotes only vs-default with
  |t| > 2 *and* a smooth neighbourhood — the strongest here is |t|≈2.4
  on a single arm. This screen calibrates step sizes; it ships nothing.
  Games promote.
- [FACT] `c_hist`, `pv_discount`, `rfp_max_depth` move nothing at any
  screened step (consistent with 065's "≤ 0 at 4.1 cp" class). They stay
  in the campaign at fallback steps — SPSA in games is exactly the
  instrument for effects the proxy cannot see.
- [GUESS] SPSA's opening `c0 ≈ 2.7 × c_end` (N=20000, γ=0.101) lands
  `skip_below`'s first perturbations near the +100/+200 cliff edge and
  `fare`'s near the sub-78 slope. Early pairs will be noisy; the schedule
  decays out of it. If the first 500 pairs show repeated blowouts, halve
  those two `c_end`s and restart — cheap while k is small.

## Decision

- c_end column written to `~/chess-runs/20260916-tu5/campaign-a.params`
  (table above). Launch TU-5 campaign A: 20k pairs, 5+0.05, `--seed 16`.
- Next proxy screen that needs labels uses `fishpack-4k-cp5.labels`;
  the three stale files stay for provenance, never as instruments.
