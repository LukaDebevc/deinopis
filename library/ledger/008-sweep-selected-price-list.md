**008 · Sweep-selected price list** · 2026-08-25
From 007's sweep: `c_rank` and `c_rank_depth` are already at a local optimum
(both directions worse), `c_hist` moves only 234 of 14000 decisions, `c_gap`
is *better at zero* (−0.248 ± 0.204) than at its guessed 300, and the ceiling
was the binding constraint — loosening `max_base` from −2000 to +1000 is worth
−0.443 ± 0.264 cp. Composite `max_base=1000, c_gap=0`: **−0.475 ± 0.258 cp**,
which at the 007 exchange rate predicts **+12 to +14 Elo**. bench 661128 ->
**567321**. The prediction was recorded before the match — this is the proxy's
first falsifiable claim about Elo.
**Setup:** 008 vs the **006** binary (not 005 — 006 isolates what the sweep
itself bought), 600 games at 10+0.1, 43-opening book, colour-reversed pairs,
both sides as subprocesses, 5 concurrent, 32 MB hash. Both binaries rebuilt
from source and identified by bench fingerprint (567321 / 661128) before play.
**Result: +10.4 Elo [−8.2, +29.2]**, +138 =342 −120, LOS 86.3%.
**Reading, and it is the careful one:** the point estimate lands on the
prediction and the interval contains it, but the interval also contains zero,
so **this does not confirm the exchange rate — it fails to falsify it.** The
match is underpowered *by construction*: half-width ±18.7 against a predicted
effect of +13. A 600-game match cannot adjudicate a 13-Elo claim. Resolving it
to 2 sigma from zero needs a half-width near ±6.5, i.e. about 8x the games —
~5000, roughly 7 hours at this TC and concurrency.
What the match does rule out is the failure mode `library/005` predicted: a
recall-only objective cannot find an interior optimum, so the sweep should have
walked to the **boundary** of what the metric tolerates, and a boundary walk
shows up as a point estimate at or below zero. It came out at +10.4. That is
weak evidence, honestly labelled, not a result.
**Standing:** 006 was −12.7 [−32, +7] vs 005; 008 is +10.4 vs 006, so 008 vs
005 is about −2 ± 27 by transitivity — indistinguishable. **The price-list
refactor is now approximately free.** It has cost nothing measurable and bought
a parameterisation that can be fitted, which was the whole point of 006. It is
still **not a checkpoint**: the gate requires beating 005 directly and 008 has
never been played against 005.
`c_gap = 0` remains a **negative result for the OCBA-motivated gap term** — the
reason the price was made a function of centipawns rather than of rank. It
falsifies this feature, not the argument; `library/004` measures why (the
feature explains r² = 0.032 of the true gap, and 008's proposed SEE fix reaches
only 0.080 — a quiescence search of the child reaches 0.712).
**Decision:** do not spend 5000 games sharpening this number. "Not falsified"
is all a match of affordable size can return here, and the same machine-hours
buy far more as 009. Two cheap follow-ups instead: (a) step further in the same
direction and check whether the proxy keeps improving while Elo does not, which
is the direct test of the boundary hypothesis; (b) 009's trajectory metrics,
which are dense and do not need a match to score a candidate at all.

---
