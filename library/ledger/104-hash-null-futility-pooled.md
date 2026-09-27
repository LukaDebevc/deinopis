# 104 · No hash interaction; futility pooled is +4.6 over 6000 — on the bound

_2026-09-18/19. The 2×2 hash experiment 103 ordered, plus the nps control it
implied. Run dir `~/chess-runs/20260918-hash/` (one frozen split binary
`625981d4`, m1-b1 `ce1f2656`), cells finished 23:18 before the reboot and were
read 09-19._

## The 2×2: hash is not the explanation

Same frozen split binary both arms, `--set fut_max_depth=3` vs base,
8+0.08 conc 5, 1000 fixed games per cell (estimates, not SPRTs).

| cell | score (fut) | Elo | draws | wall |
|---|---|---|---|---|
| hash 32 (replicates the gate) | +155 =719 −126 | **+10.1 [−1.3, +21.5]** | 71.9% | 3506 s |
| hash 64 (replicates the SR-1 SPRT) | +168 =702 −130 | **+13.2 [+1.3, +25.1]** | 70.2% | 3501 s |

Zero engine errors, zero forfeits, 500 pairs each.

- [FACT] Difference between cells **+3.1 ± 8.4** — no interaction. 103's
  hypothesis is dead.
- [FACT] **Mixed-hash comparisons on record are therefore NOT suspect**, which
  103 had flagged as the expensive branch of this test. Nothing to re-run.

## The nps control: the split refactor is free

103 left a second difference unexamined. The gate compared the **tree
(split + flag)** against the **cp-0005 binary**; the cells compared **flag
on/off inside one binary**. The split commit `0e8ac04` lands *after* cp-0005,
so the gate carried two changes, not one. Bench-exact (219718 both) means same
nodes — it does not mean same speed, and games are played on the clock.

`taskset -c 5 chess bench --wdl m1-b1`, 5 runs each, same `.cargo/config.toml`
(`target-cpu=native`) both:

| build | mean nps | runs |
|---|---|---|
| cp-0005 tag | 889,640 | 895323 885386 881797 895604 890089 |
| HEAD (split, flag 0) | 886,721 | 880346 890522 888978 892616 881145 |

- [FACT] **0.33% apart against a ±0.8% within-build spread** — no cost. The
  split is behaviour-identical *and* speed-identical, so the gate and the cells
  measured the same intervention and pool legitimately.

## Pooled: futility skip-quiets is worth about +5

The 102 SPRT (+24.6, 806g) is excluded — it stopped on H1, which stops on an
upward run. The three unbiased measurements:

| measurement | n | Elo | SE |
|---|---|---|---|
| gate (103, hash 32) | 4000 | +0.9 | 3.06 |
| cell h32 | 1000 | +10.1 | 5.83 |
| cell h64 | 1000 | +13.2 | 6.07 |
| **inverse-variance pooled** | **6000** | ~~**+4.6 [−0.2, +9.4]**~~ | 2.47 |

🔴 **This pooling is wrong — corrected by LEDGER 109 (2026-09-20).** The gate
is a **two-binary** comparison; the two cells are flag-on/off inside one
binary. The nps control below rules out a *speed* difference between those
binaries, not a *play* difference, so it never licensed the pool. Four
one-binary measurements of the flag (the two cells, plus 108's R1 and R2 FUT
cells at n=6000 and n=12000) are homogeneous, χ² = 1.31 on 3 dof (p ≈ 0.73),
and pool to **+12.28 ± 1.50**. Adding the gate pushes χ² to 12.47 on 4 dof
(p ≈ 0.014). The split has since been measured directly and is **neutral**
(−1.22 ± 2.31 at the anchor), so the gate's +0.9 remains unexplained and is
excluded, not pooled.

- [FACT] Heterogeneity χ² = 4.36 on 2 dof, p ≈ 0.11. The scatter is what three
  measurements of one quantity look like; no arm needs an explanation.
- [INFERENCE] The true effect sits **on the [0,5] SPRT bound**. That is the
  whole story of 102/103: an SPRT cannot decide a hypothesis it is centred on,
  and 4000 games at SE 3.1 was never going to.

## Decision

- ~~**Not shipped, not killed.** It is a **ship candidate parked for a bundle
  partner**~~ — **superseded by LEDGER 109.** At **+12.28 ± 1.50** the change
  clears the [0,5] bound on its own and needs no partner. `fut_max_depth`
  stays 0 only until the deliverable (tree with the flag on, vs the cp-0005
  binary) is measured cleanly, which the gate never did. The 094 precedent
  below applies to genuinely small positives, which this is not:
  the 094 precedent is exactly this: three items at +3.2/+4.1/+5.7 that no
  single [0,5] SPRT could pass went +11.2 H1 together and became cp-0005.
  There is currently no second live small positive to bundle with.
- [DECISION] The instrument, not the idea, is the binding constraint. Deciding
  a +5 change at 95% needs SE ≈ 2.5, i.e. ~6000 games — 6 h of desk box at
  8+0.08, which is what we just spent to get an interval that still contains
  zero. **Throughput is now the critical path**; see LEDGER 105.
