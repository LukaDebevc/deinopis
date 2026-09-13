**009 · Trajectory metrics fail a known-answer test** · 2026-08-25 · `src/tune.rs`
`chess tune metrics` records the root's declared move after every completed
iteration (via the existing `InfoFn` hook, so `search.rs` and the bench
fingerprint are untouched) and reduces the trajectory to several candidate
objectives at once. The question was whether reading *when* the search arrives
at the right move sees the under-pruning direction that 007 showed regret is
blind to.
**Setup:** 800 positions, teacher 400k nodes top-6, student 60k, `c = 1000`,
paired within position so every arm sees the same searches. Arms: `base`
(008's price list), `nopolicy` (all coefficients zero — worth 80-150 Elo by
007), `over2.4` (007's known over-pruning case).
**Confound check first, and it passes emphatically:** `nopolicy` reaches depth
**7.09 vs base's 10.00** at the same node count, `t = -70`. `over2.4` reaches
13.20, `t = +40`. The arms are not subtle.
**Result — negative.** Paired differences vs `base`, `t = Δ/se`:

| arm | `D_log` (t) | `D_cp` (t) | regret (t) | agree | settle |
|---|---|---|---|---|---|
| nopolicy | **−0.072 (−1.5)** | −0.20 (−0.3) | +0.69 (+0.6) | 0.496 | 0.623 |
| over2.4 | +0.163 (+3.0) | +2.11 (+3.2) | +4.86 (+3.7) | 0.450 | 0.647 |

Neither trajectory metric beats plain regret on over-pruning, and on
under-pruning both have the **wrong sign**. Narrowing the window to the tail
makes it worse: at `c = 30000` (one doubling) `D_log` prefers `nopolicy` at
**t = −3.8**. Two candidate explanations were tested and both killed — it is
not an early-search artifact (the `c` sweep goes the wrong way) and it is not
"occupancy rewards stasis" (`settle` shows `nopolicy` revising *more*).
**Mechanism:** `nopolicy` holds the teacher's move for more log-time, ends on
it less often, and revises more — it reaches the right move and talks itself
out of it. Only the final declaration is played, so an occupation measure
credits *visiting* the right answer while the game pays for *committing* to it.
The family is recall-over-time and inherits exactly the blindness it was built
to remove. No repair exists inside it: weighting late time degenerates to
regret, weighting early time is what fails here.
**Decision:** drop the trajectory family as an objective. `trajectory()` and
`metrics()` stay as diagnostics — `settle` and `flips` explain *why* a price
list behaves as it does — but nothing is fitted against them.
**What replaces it: `depth` at fixed nodes**, which separates `nopolicy` at
70 sigma, is free, and needs no teacher. It is degenerate alone and this run
measures that rather than assuming it — `over2.4` is +3.2 plies deeper and
worse everywhere else. So depth is the **constraint** and regret the
**objective**, and they must not be summed in Elo: over2.4 scores +3.2 plies
(~+176 Elo) against +4.86 cp (~−131 Elo), so a linear combination rewards it.
Write-up: `library/006-trajectory-metrics.md`.

---
