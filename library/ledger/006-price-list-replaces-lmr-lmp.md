**006 · Price list replaces LMR + LMP** · 2026-08-25 · `search::Pricing`
Deleted `lmr_table()`, `lmr_min_depth`, `lmr_min_moves`, `lmp_max_depth`,
`lmp_base` and the "don't reduce captures" rule. Replaced with one continuous
price per child: `c0 + c_rank*L(rank) + c_gap*L(1+gap/gap_unit) + c_hist*H +
c_rank_depth*L(rank)*L(depth)`, clamped between a floor and a depth-dependent
ceiling. Twelve named parameters, none of them fitted — the coefficients were
chosen to sit near the old table's shape and nothing more.
**Result:** bench 630367 -> 661128 nodes (+4.9%). **−12.7 Elo [−32, +7]**,
+123 =332 −145 over 600 games at 10+0.1 against the 005 binary, both sides as
subprocesses. LOS 9.9%, i.e. consistent with zero and not resolved well enough
to call it a loss.
**Decision:** kept as a research arm, **not** a checkpoint — it has not beaten
the previous version and the gate would rightly refuse it. Removing a hand-fit
table and a pruning rule for guessed coefficients at a cost of 0–30 Elo is the
price of a parameterisation that can be fitted at all, and fitting it is 007.
Design and the OCBA argument for the `log(gap)` term:
`library/003-budget-search.md`.

---
