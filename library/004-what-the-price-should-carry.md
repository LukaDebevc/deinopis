# 004 · What the price should carry

_2026-08-25. Measured in a scratch copy while another session held the working
tree, so the tools are not in `examples/` yet — `gapstat.rs`, `features.rs`,
`sigma.rs`, reproduced from the listings in this file. Positions: `.ladder/*.pgn`,
skip 16 plies, stride 7, top 8 moves per position, reference search 512k nodes.
Companion: `005-what-to-measure.md`, on objectives rather than features._

## The question

LEDGER 008 found `c_gap` measures **better at zero** than at its guessed 300,
and read that as falsifying the feature rather than the OCBA argument. This
asks the same question directly rather than through the proxy: **do the
features the price is built from carry any information at all**, and what does
allocation theory say the coefficients should be?

It corroborates 008 and supplies the mechanism. One of the two load-bearing
features explains 3% of the variance in what it predicts; the fix is cheap and
large, and is **not** the SEE change 008 proposes; and a third quantity the
price list does not contain at all is the strongest signal in the data.

## Why the objective has to change before the parameters can grow

The intent is for `Pricing` to absorb more indicators over time, eventually
becoming a learned controller. That intent is what rules out the current
objective, not just makes it slow.

Root regret gives **one scalar per position**, and it is zero in the ~70% of
positions where two price lists agree. Its standard error over 1500 positions
is ~1 cp against a dynamic range of a couple of cp. A finite-difference
gradient costs `2n+1` evaluations. Both scale badly in `n`: the noise floor
does not improve as parameters are added, and the cost per gradient grows
linearly. Twelve parameters is already at the edge; forty is not reachable.

A depth-8 tree contains ~10^5–10^6 pricing decisions. Labelling each one
against a full-width oracle turns the problem into supervised learning with
millions of examples, where a hundred features are identifiable for free.
**The dense-supervision route is not a faster way to do the same fit — it is
the precondition for the parameter set ever growing.**

## The allocation law

Child with predicted gap `g` below the running best, search error s.d.
`σ(b) = σ₀ 2^(-γ b/PLY)` after budget `b`, mistake probability `Φ(-g/σ(b))`,
node cost `2^(b/PLY)`, mistake cost `C`. Minimising
`Σ C_i Φ(-z_i) + λ Σ 2^(b_i/PLY)` with `z = g/σ(b)` gives

    C φ(z) z γ = λ 2^(b/PLY)

Marginal benefit peaks at `z ≈ 1` and dies for large `z`: sampling is worth
something only while the outcome is in doubt. On the meaningful branch the
solution sits near `z* = O(1)`, giving

    b ≈ (PLY/γ) [ log₂σ − log₂g + log₂z* ] + (PLY/γ) log₂C

Three structural claims, all testable:

* the coefficient on `log₂(gap)` is **PLY/γ**, with γ independently measurable;
* a **`log₂(σ)` term exists with exactly the opposite coefficient** — only the
  ratio `g/σ` enters. `Pricing` has no σ term at all;
* a standalone **`log₂(depth)`** term exists, representing `log C`, the cost of
  an error at this node. `Pricing` has only the rank×depth *interaction*.

## Measurement 1 · γ, the error-decay rate

`sd(V_n(c) − V_ref(c))` against `n`, ref = 512k nodes, 300 positions × 8 moves,
mate scores dropped, deviations clamped to ±1500 cp.

| nodes | sd (cp) | mean abs (cp) | depth reached |
|---|---|---|---|
| 2 000 | 205.6 | 105.8 | 4.21 |
| 8 000 | 178.1 | 87.1 | 6.17 |
| 32 000 | 155.1 | 67.8 | 8.45 |
| 128 000 | 113.6 | 41.6 | 11.03 |

`sd ~ nodes^-0.138`, empirical EBF **1.88**, so **γ = 0.126 per ply — search
error halves every 7.9 plies.** Hence `PLY/γ ≈ 7900 milli-plies`.

The top rung is contaminated: `V_ref` carries its own error and is strongly
correlated with `V_n`, which makes the measured decay *faster* than the truth
near the top. Fitting only the first three rungs gives γ ≈ 0.091 and
`PLY/γ ≈ 11000`. Both are far above the current `c_gap = 300`.

## Measurement 2 · the gap feature is almost pure noise

`gap` in `search.rs` is `best_score − static_eval − move_gain(b, mv)`, and
`move_gain` is a PST delta plus the raw victim value — no exchange resolution.
Scoring four candidate predictors of the true gap, in the same log space
`price()` uses:

| predictor | slope (true~pred) | r | **r²** | usable c_gap |
|---|---|---|---|---|
| `move_gain` (current) | 0.232 | 0.178 | **0.032** | 1836 |
| `move_gain` + SEE veto | 0.359 | 0.282 | 0.080 | 2842 |
| **qsearch of the child** | 0.879 | 0.844 | **0.712** | 6954 |
| 1000-node search of the child | 0.959 | 0.908 | 0.825 | 7583 |

**The current feature explains 3% of the variance in the quantity it is
supposed to predict.** A quiescence search of the child — tens to low hundreds
of nodes — explains 71%, and reaches 86% of what a 1000-node search gets.

This resolves the branch cleanly: **the OCBA argument is not what is broken.
The feature is.** And it explains why `c_gap = 300` looked plausible next to a
hand-fit table — under classical errors-in-variables the optimal coefficient on
a noisy feature is the ideal one times the regression slope, and that slope is
0.23. The attenuation was doing the work the coefficient was credited with.

### The structural consequence

OCBA is a **two-stage** procedure: sample everything cheaply, *then* allocate
by `(σ/gap)²`. `Pricing` attempts stage 2 with a static estimate and no stage 1.
The measured fix is to run stage 1 — a qsearch pass over the children — and
price from its output. Cost is affordable where reductions matter (high
remaining depth) and self-defeating near the leaves, which makes the crossover
itself a parameter.

## Measurement 3 · σ is strongly predictable, and is not in the price list

Binning by candidate, sd of `V_2k − V_ref` within each quintile. 3785 child
samples, overall sd 196.3 cp. Regressing on single `|deviations|` — the first
thing tried — is the wrong instrument: `|d|/σ` is itself random with
`Var(ln|z|) = π²/8`, so even a perfect predictor caps out near r² = 0.29.

| candidate | Q1 | Q2 | Q3 | Q4 | Q5 | Q5/Q1 |
|---|---|---|---|---|---|---|
| non-pawn material | 374.4 | 163.8 | 110.0 | 93.7 | 49.5 | **0.13** |
| legal move count | 362.9 | 162.5 | 134.5 | 91.7 | 78.1 | 0.22 |
| `\|qsearch − 1k search\|` | 122.6 | 135.3 | 142.1 | 185.3 | 319.7 | **2.61** |
| sibling spread of `move_gain` | 271.0 | 198.1 | 185.3 | 136.5 | 159.3 | 0.59 |
| `\|static − qsearch\|` | 214.7 | 213.0 | 202.5 | 143.8 | 175.0 | 0.82 (non-monotone) |

Material, move count and sibling spread are mutually correlated, so each was
re-binned **within** material quartiles:

| candidate, controlled for material | npm 0–4 | 4–6 | 6–10 | 10–14 | pooled |
|---|---|---|---|---|---|
| `\|qsearch − 1k\|` high/low ratio | 1.77 | 2.12 | 1.67 | 1.87 | **1.82** |
| sibling spread high/low ratio | 0.80 | 1.43 | 1.07 | 1.44 | 0.88 |
| legal moves high/low ratio | 0.58 | 0.80 | 0.86 | 0.94 | 0.63 |

Fitted model on `ln|dev|`:

| model | r² | fraction of achievable ceiling |
|---|---|---|
| material only | 0.196 | 0.76 |
| `\|qsearch − 1k\|` only | 0.089 | — |
| material + `\|qsearch − 1k\|` | **0.232** | **0.80** |
| + move count + spread | 0.235 | 0.81 |

**σ ∝ npm^-0.89 · |qsearch − 1k|^0.18**, capturing ~80% of what is
theoretically capturable. Material alone spans a **7× range in σ** (sd 35 cp in
heavy middlegames, 438 cp in light endgames).

Three conclusions:

1. **The dominant σ driver is game phase, inverted from intuition:** more
   material means *less* shallow-search error. The mechanism is about *this*
   evaluator, not about chess — PST captures middlegame material imbalance
   well and endgame technique not at all. **Expect this coefficient to move
   substantially after NNUE**; it is not a durable constant.
2. **`|qsearch − 1k|` is genuine, independent local uncertainty** — it survives
   the material control at 1.67–2.12× in every bin.
3. **The sibling-spread result from `library/001` is not confirmed.** Its raw
   0.59 ratio is a material confound; within material bins the effect is
   non-monotone and flips sign (0.80, 1.43, 1.07, 1.44). Recording this as
   **inconclusive**, and noting that the uncontrolled version pointed the wrong
   way — this is exactly the confound the analysis was built to catch.

### Where the σ term belongs

Naively, a 2.5-doubling range in σ times `PLY/γ ≈ 7900` implies ±20 plies of
budget difference, which is absurd. The resolution is that λ is a Lagrange
multiplier for a *global* node budget, so it adapts across positions and only
*within-node* σ variation belongs in the per-child price. The discovery
therefore splits:

* **across positions → a time-management signal.** Uncertainty varies 7×
  between positions and time management currently ignores it entirely. Separate
  experiment, cheap, plausibly worth real Elo.
* **between siblings at a node → the per-child `log σ` term**, fed by
  `|qsearch − shallow|`, which the two-stage restructuring above already pays
  for.

## What this says about the design

| current | measured verdict |
|---|---|
| `gap` from `static_eval + move_gain` | r² = 0.03. Replace with a qsearch pass. |
| `c_gap = 300` | 6–25× too small even after attenuation. |
| no σ term | σ is the strongest signal measured; ~80% predictable. |
| no phase conditioning | phase spans a 7× σ range. |
| `c_rank`, `c_rank_depth` | untested here — needs the search's *real* ordering, not a static rank. |
| `fare` and `c0` | exactly degenerate wherever the clamp is not binding (`search.rs:243-252`); `pv_discount` partially so. A flat direction the finite-difference tuner will waste evaluations on. |
| hard `clamp(…, max_base + max_slope·depth)` | theory says the saturation should be soft; the `−z²/2ln2` correction bends, it does not hit a wall. |

## Plan

Objectives and their ordering now live in `005-what-to-measure.md`. What this
file adds to that plan is the corpus: a **full-width fixed-depth alpha-beta
oracle** (bound cutoffs kept, all heuristic pruning off) computes exactly
`V_d`, which is order- and policy-independent, unlike labels produced by our
own search policy. Run it at d = 6,7,8 over ~1000 positions and dump **every
priced child** — features, price charged, and `rel(c) = 1[V_d(c) > alpha]`.
Cost is ~2-15 s/position at d=8; 1-4 h once on 6 cores.

That corpus turns fitting into logistic regression with ~10^6 examples, where a
hundred features are identifiable — which is the point, given that `Pricing` is
meant to keep absorbing indicators. Root regret gives one scalar per position,
zero wherever two price lists agree, and costs `2n+1` evaluations per gradient;
its noise floor does not improve as parameters are added. **The dense route is
not a faster version of the same fit, it is the precondition for the parameter
set ever growing.**

One risk it does not remove: search control is scale-dependent, so a fit
against a d=8 oracle is an extrapolation to game depths. Fit at d = 6,7,8,
check the coefficients move along a smooth 1-D curve, and verify at d=10 on a
subset before relying on the pipeline.

## Immediate, independent of the above

The qsearch-gap result does not need any of the phases. It is a feature swap
with a 22× information gain, measured. Two cheap arms worth an SPRT on their
own, in order of cost:

1. **SEE veto on `move_gain`** (r² 0.03 → 0.08) — a few lines, no node cost.
2. **qsearch-sourced gap above a depth threshold** (r² → 0.71), with `c_gap`
   raised toward ~7000 to match the reduced attenuation. Larger change, larger
   expected effect.

Raising `c_gap` *without* fixing the feature would be actively harmful — the
attenuation is what currently keeps a noise feature from doing damage.
