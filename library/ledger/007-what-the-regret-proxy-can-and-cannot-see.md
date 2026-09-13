**007 · What the regret proxy can and cannot see** · 2026-08-25 · `src/tune.rs`
Built the tuning platform (label / eval / compare / search) and, before
descending on it, asked what it resolves. 14000 positions from the gauntlet and
self-match PGNs, labelled top-5 root moves at 400k nodes each; candidates
scored at 60k nodes; per-position regret capped at 200 cp because 5% of real
positions contain a mate and uncapped they are 94% of the mean.
**Result — calibration:** regret falls monotonically with effort (27.4 / 23.0 /
17.8 / 13.2 cp at 5k / 15k / 60k / 240k nodes), so 4x nodes ~ 4.5 cp. At
~100-140 Elo per 4x nodes that is **1 cp ~ 25-30 Elo**.
**Result — resolution:** the mean's standard error is the wrong statistic;
candidates must be scored on the same positions and differenced. Paired SE is
±0.80 cp at 1500 positions and **±0.26 cp at 14000**, scaling as 1/sqrt(n) as
it should.
**Result — the important one, and it is negative:** the objective is
**asymmetric**. Over-pricing is detected loudly and monotonically — scaling all
coefficients by 1.0 / 1.6 / 2.4 gives 0 / +1.40 / **+4.38 ± 0.33** cp, the last
at 13 sigma. Under-pricing is nearly invisible: deleting the *entire* pricing
policy measures **+0.006 ± 0.283** cp at 15k nodes and +0.445 ± 0.278 at 60k,
for a policy worth 80-150 Elo in real play. Mechanism: over-pruning makes the
search miss the right move, which is what regret measures; under-pruning only
makes it shallower, and at fixed nodes a shallower search still agrees about
the root move, because that choice is dominated by the first few plies, the TT,
move ordering and quiescence — all shared by both arms.
**Decision:** treat the objective as a **safety rail, not a fitness function**.
It may reject a candidate that prunes too hard; it must never be used to argue
that a conservative price list is good. Do not run `tune search` as a
general-purpose optimiser until the blindness is fixed — the proposed fix is to
label at two node budgets and keep only positions where the reference's best
move changed, which is where depth actually decides.
**Also found:** `skip_below` fires in **0 of 14000 positions** — the ceiling was
set to reproduce the old `clamp(0, depth-2)`, which by construction leaves every
child at least one ply, so late move pruning was removed and *not* replaced.
**This holds only at `max_base = -2000`, the value in force when it was
measured.** 008 raised the ceiling to +1000 and thereby turned LMP back on: at
the current default 57-60% of priced children are refused outright. See
`library/008`; the consequence for 008's own reading is recorded there.
Raising the ceiling from `depth+1` to `depth+2` is byte-identical, proving the
price formula saturates below it. ARCHITECTURE and `library/003` claimed LMP was
subsumed; both are corrected at the source.

---
