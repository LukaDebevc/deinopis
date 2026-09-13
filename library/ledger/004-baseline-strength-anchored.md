**004 · Baseline strength, anchored** · 2026-08-24 · `.ladder/gauntlet-20260824-223916*`
Five CCRL-rated opponents built at their exact rated tags, 200 games each,
10+0.1, colour-reversed pairs, both sides as subprocesses.
**Result:**

| opponent | CCRL Blitz | our score | delta | implied |
|---|---|---|---|---|
| BBC 1.1 | 2019 | .933 | +456 | 2475 |
| Goldfish 2.1.1 | 2252 | .835 | +282 | 2534 |
| Cinnamon 2.4 | 2326 | .710 | +156 | 2482 |
| Tantabus 2.0.0 | 2555 | .493 | −5 | 2550 |
| Blunder 8.5.5 | 2664 | .418 | −58 | 2606 |

The ladder does **not** agree with itself: 131 Elo of spread, monotone in
opponent strength, against per-match intervals of ±25–50. That is Elo-model
misfit across a wide rating gap, not noise. A logistic MLE over all five gives
**2543**; over the two nearest anchors only, **2577**.
**Decision:** quote **~2550 ± 60 on the CCRL Blitz scale**, dominated by the
anchoring disagreement rather than the game count, and on top of that the
systematic offset from CCRL's conditions (different book, no tablebases,
10+0.1 vs 2'+1"). Never "we are 2550".
**Gap:** `chess elo` over all five PGNs prints the raw W/D/L and no Elo —
pentanomial pairing does not survive the merge, so `moments()` returns `None` —
and there is no code that combines per-opponent deltas into one anchored
rating. The number above was combined by hand, which is exactly what this
project's rules say not to leave standing. Both need fixing.

---
