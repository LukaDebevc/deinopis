**014 · The optimum is budget-invariant; low budgets have the wrong sign** · 2026-08-25
Asked two things at once: (a) can agreement rate — "did the cheap search pick
the teacher's move", a 0/1 loss instead of a centipawn loss — replace regret,
and (b) does the best `c_rank` depend on the student's budget. `c_rank` swept at
five budgets against the same fixed 400k teacher, 4000 positions.

| `c_rank` | 0 | 125 | 250 *(def)* | 375 | 500 | 750 | 1000 |
|---|---|---|---|---|---|---|---|
| regret @1k | 32.33 | 33.51 | 32.72 | **31.84** | 32.42 | 32.32 | 32.29 |
| @4k | 27.15 | 27.61 | 27.17 | **26.59** | 27.37 | 27.26 | 28.22 |
| @15k | 23.23 | 22.85 | 21.94 | **21.57** | 22.04 | 24.39 | 24.90 |
| @60k | 17.97 | 17.82 | **16.67** | 16.85 | 17.63 | 18.53 | 19.88 |
| @240k | 13.41 | 12.99 | **12.81** | **12.81** | 13.23 | 13.49 | 15.55 |

**(a) Agreement is a valid instrument and adds nothing.** It picks the same
optimum as regret at every budget. It is regret with a coarser loss, at the same
price. Useful for reporting, not a new signal.
**(b) The optimum does not drift.** It is a flat plateau at **250-375 across a
240x span of budget**. An earlier reading of these data claimed a downward drift
(375,375,375,250); 240k returns 250 and 375 *tied to the last digit*, so that was
one point of noise over-read and is withdrawn.
**What does change with budget is the penalty for over-pruning**, and it is
monotone. `c_rank=1000` against the row's best, as a fraction of that best:
**1.4% (1k) -> 6.1% (4k) -> 15.4% (15k) -> 19.3% (60k) -> 21.4% (240k)**.
**At 1k it is negative** — over-pruning scores *better* than the default. The
search reaches depth 3.0 at 1k, and at depth 3 there is no tree to over-prune.
**Consequence — do not tune cheap by shrinking the student.** Contrast-to-noise,
(worst-best)/stderr: **1.9 (1k), 2.0 (4k), 4.6 (15k), 4.9 (60k), 4.8 (240k)**.
The 1k student makes *twice* the raw error of the 60k student and discriminates
the parameter **2.5x worse**, because the extra error is common to all arms and
cancels in the pairing. Off-list rate compounds it: 22.4% at 1k against 7.0% at
60k, so at 1k a fifth of positions have their regret clipped to the cap.
**The instrument saturates upward too**: contrast stops improving above 15k.
**15k is the operating point** — 4x cheaper than 60k, same discrimination, same
answer. (240k is attenuated: the 400k teacher leads it by only 1.67x. It agrees
with 60k, which is what it was run to check.)
**Open, and now the only way to break the tie:** at every budget `c_rank=375`
buys ~0.8 plies over 250 at equal regret (60k: 9.81 -> 10.56; 240k: 12.85 ->
13.72). Equal-regret-more-depth is not the `over2.4` trap (deeper *and worse*),
but depth is a constraint and not a term, so it cannot be scored. This goes to
SPRT or it does not go.

---
