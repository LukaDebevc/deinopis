# 006 · Trajectory metrics: a known-answer test, and a negative result

_2026-08-25. Answers the question `005-what-to-measure.md` left open: can a
metric that reads the search's **trajectory** rather than its final answer see
the under-pruning direction that LEDGER 007 showed regret is blind to?_

**No. It sees it less well, and at low node counts it sees it backwards.**

The thing that does work was already in `SearchResult` and needs no teacher.

## The design

A metric proposal is worth nothing until it is shown a case whose answer is
already known. LEDGER 007 supplies one: **deleting the entire pricing policy is
worth 80–150 Elo**, and regret measures it at `+0.006 ± 0.283 cp`. Any metric
that claims to fix that blindness must see `nopolicy` loudly, and must still see
the over-pruning case (`over2.4`) that regret already catches at 13σ.

The candidates, all reductions of one recorded trajectory so they cost nothing
extra to compare:

| | definition |
|---|---|
| `D_log` | `∫ 1[m(t) ≠ m*] d log₂ t` over `[c, N]` — doublings spent wrong |
| `D_cp`  | `∫ loss(m(t)) d log₂ t / span` — the same, weighted by what the held move costs |
| `regret` | capped cp loss of the **final** move — the existing objective |
| `depth` | depth of the last completed iteration — free, no teacher |
| `settle` | fraction of the log-window already spent on the move finally declared |

`D_log` is the measure derived twice over — once as an occupation measure, once
as Luka's `Σ_{i>c} 1[agrees]/i`; the two sum to `log(N/c)`. Its appeal was that
it is bounded, needs no epsilon, and is already denominated in doublings, hence
in Elo at 007's ~50–70 Elo per doubling.

## The confound check, first

Before reading any metric: **do the arms actually differ?** 800 positions,
60k nodes, `c = 1000`.

| arm | depth | Δdepth | t |
|---|---|---|---|
| base | 10.00 | — | — |
| nopolicy | 7.09 | **−2.91** | **−70** |
| over2.4 | 13.20 | **+3.20** | **+40** |

Emphatically yes. Deleting the price list costs **2.9 plies at the same node
count**. This is not a subtle arm.

## The result

Same run. Paired differences against `base`, `t = Δ/se`; `|t| > 2` is resolved.

| arm | `D_log` Δ (t) | `D_cp` Δ (t) | `regret` Δ (t) | agree | settle |
|---|---|---|---|---|---|
| base | — | — | — | 0.515 | 0.656 |
| nopolicy | **−0.072 (−1.5)** | −0.20 (−0.3) | +0.69 (+0.6) | 0.496 | 0.623 |
| over2.4 | +0.163 (+3.0) | +2.11 (+3.2) | +4.86 (+3.7) | 0.450 | 0.647 |

* On **over-pruning** all three agree and none beats plain regret.
* On **under-pruning** all three are unresolved, and both trajectory metrics
  have the **wrong sign** — they rate the policy-free search *better*.

Restricting the window to the tail makes it worse, not better:

| `c` (N = 60k) | span | `nopolicy` `D_log` Δ (t) |
|---|---|---|
| 1000 | 5.91 | −0.072 (−1.5) |
| 5000 | 3.58 | −0.077 (−2.0) |
| 15000 | 2.00 | −0.066 (−2.6) |
| 30000 | 1.00 | **−0.052 (−3.8)** |

At `t = −3.8` this is not noise. The occupation measure **significantly prefers
the search that is 2.9 plies shallower and worth 80–150 Elo less.**

## Two explanations tested and killed

**"Log weighting upweights early search, which is policy-invariant."** Killed by
the `c` sweep: restricting to the last doubling *strengthens* the wrong-sign
effect. It is not an early-search artifact.

**"Occupancy rewards stasis — the shallower arm freezes and holds."** Killed by
`settle`: `nopolicy` is *less* settled (0.623 vs 0.656; 0.816 vs 0.889 at
`c = 30000`), i.e. it revises more, not less.

## What actually explains it

The three numbers only fit together one way. `nopolicy` **spends more log-time
holding the teacher's move, ends on it less often, and revises more.** It
reaches the right move and then talks itself out of it.

Which is the whole problem, because **only the final declaration is played.**
An occupation measure credits *visiting* the right answer; the game pays for
*committing* to it. So the trajectory family is a recall measure over the time
axis, and it inherits precisely the precision blindness 007 found on the value
axis — one level down, for more machinery.

There is no repair inside the family. Weighting late time fixes the sign only by
degenerating toward the final answer, which is regret; weighting early time is
what fails here. **The trajectory carries no information about under-pruning
that the final answer does not already carry better.**

## What works instead

**Depth at fixed nodes**, `t = −70` where every teacher-based metric sits at 1.
It is free, it is already in `SearchResult`, and it needs no labels at all —
under-pruning *is* "fewer plies for the same nodes", with no proxy step.

It is degenerate alone, and this run measures that too rather than assuming it:
`over2.4` is **+3.2 plies deeper and worse on every other axis**. Depth bought
by pruning is not depth. So it is the **constraint** and regret is the
**objective**, exactly as `005` recommendation #1 argued — now with the σ to
back it.

And do not naively add them in Elo: `over2.4` scores +3.2 plies ≈ +176 Elo
against +4.86 cp ≈ −131 Elo, i.e. a linear combination *rewards* it. The pair
must be read as a constraint and an objective, not summed.

## The finding with actual leverage

Two label sets, so the teacher's own strength is controlled rather than assumed.

**A. 8000 positions, teacher 400k:**

| student | base depth | Δdepth (t) | Δregret (t) | `D_log` (t) |
|---|---|---|---|---|
| 15k | 7.05 | −1.95 (−195) | +0.29 (+0.7) | −0.111 (**−10.4**) |
| 60k | 9.73 | −2.91 (−231) | **+1.37 (+3.4)** | −0.090 (**−6.2**) |

**B. 2000 positions, teacher 4M**, so the teacher still leads by 5.7× at the
deepest student — which A could not do:

| student | teacher edge | base depth | Δdepth (t) | Δregret (t) | `D_log` (t) | agree |
|---|---|---|---|---|---|---|
| 60k | 66.7× | 9.68 | −2.89 (−116) | +1.65 (+2.0) | −0.119 (−4.5) | 0.481 |
| 240k | 16.7× | 12.64 | −4.07 (−117) | +1.85 (+2.2) | −0.081 (−2.4) | 0.542 |
| 700k | 5.7× | 15.03 | **−5.03** (−116) | +1.71 (+2.2) | −0.007 (−0.2) | 0.583 |

**007 measured at the shallowest rung.** At 15k regret sees nothing, exactly
reproducing it; by 60k it resolves the arm at `t = 3.4`. Below about depth 7 the
root move is not a function of the pruning policy at all.

**But regret saturates, and that is the real finding.** From 60k to 700k the
depth gap **widens monotonically from 2.89 to 5.03 plies while regret stays
pinned near 1.7 cp**. The 240k plateau was not the 400k teacher running out of
authority — it survives a teacher 5.7–66.7× stronger. So the attenuation gets
*worse* with depth, not better.

I had this wrong an hour earlier, on 800 positions, where the ladder appeared to
grow to +1.80 at 240k. That was noise (±1.06), and "reductions compound with
depth" was the wrong mechanism — it predicts monotone growth, which does not
happen.

**The mechanism that does fit:** the root move is a *bounded-information
channel*. Agreement climbs (0.481 → 0.583) and the regret gap does not, because
one move out of ~35 cannot express more of a policy's value however much depth
separates the arms. This is 007's result and `005`'s "the root is the wrong
place to measure", now with a number: **the root channel caps out near 1.7 cp**
— about 45 Elo at the 007 exchange rate, against a policy worth 80–150.

**The practical consequence saves compute rather than costing it: score at
60k.** Going to 700k costs 12× and buys *zero* extra signal. The natural
assumption was that the honest target is the operating point. It is not.

**And the consequence that matters for the roadmap:** no root-based objective
can be repaired by spending more on it. The remaining headroom is per-child
supervision against a full-width oracle — `005` category C — which is already
track 1 of the tuning plan.

## One more explanation tested and killed

*"Regret saturates because the chosen move falls outside the labelled top-6,
where it clips to `best − worst` and therefore cancels in the paired
difference."* This would have been a different disease with a different cure —
label more moves, not more positions.

Measured: the off-list rate is **4.6–10.5%** across every arm and every budget.
Not clipping. Most disagreements are near-ties *inside* the top six, which is
also why regret stays small while `agree` sits near 0.5.

## Consequences

1. **Score at 60k.** Not 15k, where the objective is genuinely blind; not
   deeper, which costs 12× for nothing.
2. **`Report` now carries `depth`, and `descend` takes `--depth-floor`**
   (default 0.25 plies), rejecting a candidate that falls that far below the
   starting depth *before* comparing its regret. Regret is attenuated ~3× in
   this direction, so over a descent run the tuner can trade real depth for
   sample noise and the objective will thank it. Depth is free, needs no
   labels, and separates the arms at 116–247σ.
   It is a **floor, never a term**: `over2.4` is +3.1 to +4.7 plies deeper and
   worse everywhere, so any objective *containing* depth is maximised by the
   known-bad arm.
   **It binds on the first real descent run**, which validates it better than
   the constructed arms do. Two free parameters at 60k, unguarded, iteration 2
   lowers `c_rank` 300 → 200 — *less* pricing, the direction regret is weak in —
   buying 0.35 cp of training regret (a quarter of its own ±1.39 stderr) for
   **0.72 plies of real depth**. Held-out moves 0.10 cp against ±1.65: the
   regret gain does not exist. With the floor on, three candidates are blocked
   and the run stops at the honest optimum.
3. **Drop the trajectory family as an objective.** `trajectory()` and
   `metrics()` stay in `tune.rs` as diagnostics — `settle` and `flips` explain
   *why* a price list behaves as it does — but nothing is fitted against them.
4. The teacher in `label_one` still runs `Params::default()`, so it shares the
   student's policy and its blind spots. This confound would make regret
   *over*-detect `nopolicy`, not under-detect it, so it does not explain the
   saturation — but it must be fixed before anything is fitted.
