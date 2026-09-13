**010 · Regret saturates at the root, and 60k is enough** · 2026-08-25
Ladder over student budgets, `nopolicy` vs `base`, paired. Two label sets, so
the teacher's own strength is a controlled variable rather than an assumption.

**A. 8000 positions, teacher 400k:**

| student | base depth | Δdepth (t) | Δregret (t) | `D_log` (t) |
|---|---|---|---|---|
| 15k | 7.05 | −1.95 (−195) | +0.29 (+0.7) | −0.111 (**−10.4**) |
| 60k | 9.73 | −2.91 (−231) | **+1.37 (+3.4)** | −0.090 (**−6.2**) |

**B. 2000 positions, teacher 4M** — chosen so the teacher still leads by 5.7x at
the deepest student, which A could not do:

| student | teacher edge | base depth | Δdepth (t) | Δregret (t) | `D_log` (t) | agree |
|---|---|---|---|---|---|---|
| 60k | 66.7x | 9.68 | −2.89 (−116) | +1.65 (+2.0) | −0.119 (−4.5) | 0.481 |
| 240k | 16.7x | 12.64 | −4.07 (−117) | +1.85 (+2.2) | −0.081 (−2.4) | 0.542 |
| 700k | 5.7x | 15.03 | **−5.03** (−116) | +1.71 (+2.2) | −0.007 (−0.2) | 0.583 |

**Result 1 — 007 measured at the shallowest rung.** At 15k regret sees nothing
(+0.29, `t = 0.7`), reproducing 007. By 60k it resolves the arm (`t = 3.4`).
Below about depth 7 the root move is simply not a function of the pruning
policy.
**Result 2, and it is the one that matters — regret saturates.** From 60k to
700k the **depth gap widens monotonically from 2.89 to 5.03 plies while regret
stays pinned at ~1.7 cp**. The earlier 240k plateau was not the 400k teacher
running out of authority: it survives a teacher 5.7-66.7x stronger. The
attenuation therefore gets *worse* with depth, not better.
**Mechanism:** the root move is a bounded-information channel. Agreement climbs
(0.481 -> 0.583) and the regret gap does not, because one move out of ~35 cannot
express more of a policy's value however much depth separates the arms. This is
007's finding and `library/005`'s "the root is the wrong place to measure",
now with a number on it: **the root channel caps out near 1.7 cp**, about 45 Elo
at the 007 exchange rate, against a policy worth 80-150.
**Decision — and it saves compute rather than costing it: score at 60k.** Going
to 700k costs 12x and buys **zero** additional signal. The honest target was
assumed to be the operating point; it is not. What deeper scoring *does* buy is
a wider depth gap, which is why the guard below is worth more than the deeper
labels would have been.
**Implemented:** `Report` now carries `depth`, and `descend` takes a
`--depth-floor` (default 0.25 plies) that rejects a candidate falling that far
below the starting depth **before** its regret is compared. Regret is attenuated
by ~3x in this direction, so across a descent run the tuner can trade real depth
for sample noise and the objective will thank it. Depth is free, needs no
labels, and separates the arms at 116-247 sigma. It is a **floor, never a
term**: `over2.4` is +3.1 to +4.7 plies deeper and worse everywhere, so any
objective containing depth is maximised by the known-bad arm.
**The guard binds on the first real descent run**, which is better validation
than the known-answer arms. Two free parameters, 1167 fit / 833 held out, 60k:

| | iter 1 | iter 2, floor off | iter 2, floor on |
|---|---|---|---|
| `c_rank` | 300 | **200** | 300 (3 blocked) |
| train regret | 21.80 ± 1.39 | **21.45** | 21.80 |
| held-out regret | 22.47 ± 1.65 | 22.37 | 22.47 |
| depth | 10.03 | **9.31** | 10.03 |

Unguarded, the tuner lowers `c_rank` — *less* pricing, the direction regret is
weak in — buying **0.35 cp of training regret, itself a quarter of its own
stderr, for 0.72 plies of real depth.** Held-out moves 0.10 cp against ±1.65,
i.e. the regret gain does not exist. This is precisely the failure the
saturation result predicts, and it appeared on the first two-parameter run
rather than needing to be constructed.
**Alternative explanation killed:** *regret clips because the chosen move falls
outside the labelled top-6, where it saturates to `best - worst` and cancels in
the pairing.* Off-list rate is **4.6-10.5%** across every arm and budget. Not
clipping — most disagreements are near-ties inside the top six.
**Consequence for the roadmap:** no root-based objective can be fixed by
spending more. The remaining headroom is per-child supervision against a
full-width oracle (`library/005` category C), which is already track 1 of the
tuning plan. Write-up: `library/006-trajectory-metrics.md`.

---
