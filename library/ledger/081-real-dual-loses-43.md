# 081 · Real dual loses −43 over 242 games, killed by hand

_2026-09-10. First SPRT of the distilled tier: v10 dual file
(`dual-b8.nnue`: m1-b1 + LEDGER 080's bneck-8 tail, one accumulator),
same frozen binary both arms (`/tmp/dual2/`, bench re-verified
199091), A `--dual` vs B base, SPRT [0,5] @8+0.08 conc 5._

## Result

| games | score | Elo | LLR |
|---|---|---|---|
| 242 | +28 =156 −58 | **−43.3 [−69, −18]** | −1.36, heading H0 |

Killed by hand before the bound (like rung 1's 56-game kill, but a
different animal: −43, not −245). Log `/tmp/dual2/sprt.log`,
PGN alongside. Bench foretold it: dual grows the tree +14.9% (228812 vs
199091) at +10% nps — about −5% on time at depth 9.

## Mechanism facts (all measured for this entry)

- **Checks are never served.** `search.rs`: `static_eval` is −INFINITY
  when in check — check-evasion nodes never evaluate. The tail's worst
  bucket is therefore undeployed.
- **Disagreement by position type** (15k SPRT-game positions, torch cps,
  `/tmp/disagree_by_type.py`):

  | bucket | share | mae | rmse | p99 |
  |---|---|---|---|---|
  | check | 7% | 65.5 | 107.9 | 426.7 |
  | capture | 70% | 38.8 | 68.2 | 281.1 |
  | quiet | 23% | 42.4 | 82.5 | 341.0 |

  Served mix is capture + quiet (checks skipped): effective MAE ≈ 39–42,
  the headline number. No distribution-shift rescue here.
- **Bias is zero** (+0.27 cp mean on 200k val): no free recalibration.
- The tail is **converged** (080: flat since ~700M) — capacity, not steps.

## Reading

Supply ≈ +8–10 Elo against demand ≈ −50 at MAE 39 / std 73 / p99 279.
The probe (078) priced σ25-white at −12; our residual is ~3x wider with
fat tails, and the tree pays for the tails (stand-pat cutoff errors both
ways grow trees). Breakeven needs MAE ≲ 15–20 at ≥1.05x supply — a 2.5x
agreement improvement the current slope (66 → 39 over 16x positions,
converged) will not reach at bneck 8.

## Decision

- Rung 3 as-shipped is **closed**: distilled-fidelity tiering at bneck 8
  loses −43. The v10 seam stays (dormant, bench-exact); nothing reverts.
- Next attempts, ranked (all keep the seam; the SPRT stays the judge):
  1. **Fat-tail loss**: KL + λ·MSE(cp) retrain — same pipeline, targets
     p99 (279) directly instead of the mean. λ swept at 50M feel-runs
     (~3 min each), winner takes one 800M. Cheap, attacks the tree
     mechanism rather than the headline MAE.
  2. **Confidence-gated tiering**: small tail first; re-evaluate big iff
     |s − beta| < margin. Attacks cutoff errors (the known mechanism),
     needs engine work + a close-call-fraction measurement first
     (unknown — decides whether the margin buys anything).
  3. **Tier-tagged TT-evals**: main search re-evaluates big on small-tier
     TT hits instead of reusing them (077 dropped this for quad because
     it rebought half the win; at MAE 39 the trade differs). TT format
     change, moderate work.
  4. **Bneck-12 student**: fallback. Slope looks bad (agreement gains
     eat the supply), run only if 1–3 need a better tail underneath.
- Kill criterion, written in advance: if (1) still loses, and the
  close-call fraction kills (2), close rung 3 — the eval's 42% node
  share caps dual supply near +10 Elo and no tail on this slope pays
  for itself. Faster-big-net kernels (no disagreement at all) would
  then be the better project.
