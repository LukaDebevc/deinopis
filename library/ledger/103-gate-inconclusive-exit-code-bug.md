# 103 · Gate inconclusive (+0.9); exit-code bug shipped bogus cp-0006, repaired

_2026-09-18. `tools/checkpoint.sh -m "futility skip-quiets (LEDGER 102)"
--sprt 0,5 --net m1-b1` vs cp-0005. Full gate (build, tests, perft, bench
178660 det. ×2), then 4000-game SPRT, log `~/chess-runs/20260918-cp06/`._

## Result

```
4000 games: +573 =2864 -563   +0.9 Elo  [-5, +7]   LOS 61.9%   LLR -0.99
SPRT: inconclusive within 4000 games
```

Trajectory: flat from the start (0.0 at 200 games, +1..+7 throughout) —
never at the SPRT's +24.6. Game-shape audit (median plies 261 vs 271,
termination mix scaled) rules out different-distributions: the two
matches played the same kind of chess. Remaining difference between the
matches: hash 64 (SPRT) vs 32 (gate). Discrepancy 3.6σ.

## The exit-code bug (found because of this gate)

`matchplay::run()` returned `None` when an SPRT hit the cap undecided;
`main` maps `None` to exit 0 (its no-SPRT meaning); `checkpoint.sh` reads
0 as H1. So the script printed "SPRT accepted H1", committed, tagged and
pushed **cp-0006** for a +0.9 inconclusive gate. Latent since introduction
— every prior gate crossed a bound, so this path never fired.

Repair (all verified): fix in `run()` (cap + sprt.is_some() →
`Some(Continue)`) + regression test `inconclusive_at_cap_returns_a_verdict`
(zero games exercises the branch; the gate itself is the negative
control); `git revert` of the gate commit (no force-push — history shows
the bogus commit and its revert); local + remote `cp-0006` tags deleted
(remote empty); baseline restored from the cp-0005 tag (bench 219718);
CHECKPOINTS.md clean. 29 lib tests pass.

## Decision

- **Not shipped.** `fut_max_depth` is back to 0 (revert); the split and
  the dormant flags stand.
- Next: 2×2 hash experiment (same frozen split binary, flag on/off × hash
  32/64, ~1000 fixed games each) — replicates gate vs SPRT exactly. If
  H64 >> H32, the interaction is real (and every mixed-hash comparison on
  record is suspect); if both ~0, SR-1 was fluctuation + H1 overshoot and
  futility is killed as shipped.
