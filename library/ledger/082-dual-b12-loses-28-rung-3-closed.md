# 082 · Dual b12 with fat-tail loss loses −28, rung 3 closed

_2026-09-10. Last rung-3 shot per 081's ranked list: bneck-12 student +
`--mse-lambda 1e-5` (loss = KL + λ·MSE(cp), 081 item 1), warm-started from
m1-b1, full 800M on the frozen accumulator. Six 50M feel-runs picked the
setup: capacity monotone (b4 68.9 → b8 58.8 → b12 51.4 MAE), MSE cuts p99
~60cp at every λ with a flat ladder (372/377/375 — 1e-5 dominates).
`tail-b12-mse.pt`: MAE **28.8**, rmse 50, p99 **198**, converged (flat last
~100M). Exported `dual-b12-mse.nnue` (v10, big bneck 16 + small 12)._

## Result

SPRT [0,5] @8+0.08 conc 5, same frozen binary both arms
(`/tmp/dual3/`), A `--dual` vs B base:

| games | score | Elo | LLR |
|---|---|---|---|
| 744 | +95 =494 −155 | **−28.1 [−42, −14]** | −2.95, H0 |

Log `/tmp/dual3/sprt.log`. Better than b8's −43, same verdict.
Slope across three tails (520→−245, 39→−43, 29→−28) says breakeven needs
MAE ~10–15 — the distill slope (66→39→29, converged twice) never gets there.

## Incident: the b12 tail ran the slow kernel

`matvec!`/`matvecq!` in `wdleval.rs` specialize widths 3/8/16/32/64 — no
12 — so every b12 projection fell into bound-checked `matvec_dyn` (the
LEDGER 055 trap). Dual benched 192965 nodes @519k nps, 1.55x slower on time
than base; b8 had its fast path all along, flattering the old comparison.
Two-line fix (a `12 =>` arm in each macro): tests 45 pass, nodes
bit-identical both modes, dual nps 519k → 746–774k. Behavior-neutral,
stays in the tree. Bench-time math is still mildly negative (tree −3.1%,
nps −9% ≈ −6% on time) — games confirmed it matters less than the residual.

## Close-call probe (081 item 2, measured then reverted)

Temporary `$CHESS_QMARGIN` probe: at every fresh q stand-pat, big-tier
re-eval beside the small, recording `|s_small − beta|` in 10cp buckets with
cutoff flips. Control (no `--dual`): 82,589 stand-pats, zero flips — wiring
valid. Dual (`dual-b12-mse.nnue`, bench): **86,481 stand-pats, 3,948 flips
(4.6%)**:

| gate ≤ | share re-evaluated | flips recovered |
|---|---|---|
| 29cp | 7.3% | 1795/3948 (46%) |
| 49cp | 12.4% | 2253/3948 (57%) |
| 99cp | 25.0% | 2740/3948 (69%) |

Kill reasoning: `mean|err|` sits at **~30cp flat across every bucket** —
far-from-window positions carry just as much returned-value noise, shifting
every `alpha.max(s)` and fail-soft return, and no window gate touches that.
Recovering 57% of flips at ~+4–5% node cost while the noise floor stands
cannot turn −28 positive. (2) killed without implementing; (3) TT-tagging
attacks only the small main-search-reuse slice while this q-noise stands —
killed with it. Probe reverted (nodes re-verified exact); the `12` dispatch
arms and `--mse-lambda` stay as dormant instruments.

## Decision

- **Rung 3 closed.** Fidelity tiering loses at every tried capacity and
  loss; the eval's node share caps supply near +10 while MAE ~29 costs ~−28.
- The dual-file seam stays dormant and bench-exact; it serves a future
  better net as top tier, not a cheaper tail.
