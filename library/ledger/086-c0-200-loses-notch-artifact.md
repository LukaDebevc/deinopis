# 086 · c0=200 loses −23 in games; the proxy notch was an artifact

_2026-09-12. `c0` (price-list intercept) screened a sharp notch on the
relabelled 1k-see corpus @15k nodes, m1-b1: 100 → +3.90 ± 1.61, **200 →
−1.36 ± 1.63**, 300 → +2.58 ± 1.61, 400 → +2.30 ± 1.68, with head-to-head
200-vs-100 at −5.26 ± 1.70 (|t| 3.1) and 200-vs-300 at +3.94 ± 1.67. SPRT
[0,5] @8+0.08 conc 5, two frozen binaries from the same dirty tree differing
in one default line, A c0=200 (bench 192320) vs B c0=0 (bench 199091),
m1-b1 both sides. Killed by hand at LLR −1.39, log + PGN `/tmp/c0/`._

## Result

| games | score | Elo | LLR |
|---|---|---|---|
| 422 | +52 =290 −80 | **−23.1 [−42, −4]** | −1.39 |

Whole interval negative. PGN re-score identical (pairs=211), zero engine errors.

## What this teaches about the proxy

This is the first proxy-picked winner to lose in games, and the shape
matters: vs *default* it was only −1.36 ± 1.63 (|t| 0.83) — the strong
|t| > 2 numbers were vs *other off-default values*, which never was the
shipping question. A sharp notch (|t| 3 better than both neighbors, weak vs
default) is a corpus artifact, not an optimum; a real intercept effect
should move smoothly. Rule going forward: **only the vs-default comparison
promotes; neighbor comparisons diagnose shape but promote nothing.** The
exchange rate's direction record stands at 2-1 (c_see won, sweep pick
confirmed-ish per 008, this lost) — and this loss came from promoting on
|t| 0.83, which the 4.1-cp floor already said was below resolution.

## Decision

- c0 stays 0. Tree default untouched (benchmark 199091 restored, binary ==
  tree rebuilt after the freeze).
- No further price-list screening without |t| > 2 vs default AND a smooth
  neighborhood. The c_rank / max_base<1000 / rfp_max_depth / delta_margin=50
  queue from 084 inherits this bar.
