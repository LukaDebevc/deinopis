# 062 · A feature earns its place against an information-free control

_2026-09-07. `chess tune compare`, 1000 fishpack positions, 15000 nodes,
paired SE ±2.02 cp (LEDGER 061). Engine b986f5b + working tree. Net
`gate-3ep-q8.nnue`. Every number below is regret in centipawns, **lower is
better**, and `b - a` negative means the candidate won._

## The setup

`Pricing` grew three new coefficients and `Params` grew one, all declared
through the `params!` macro so they are tunable, documented and UCI-visible
from birth. The price function now takes a `Feats` struct rather than five
positional arguments — a new indicator is a field and a term, not another
argument threaded through the move loop.

| name | fires when | group |
|---|---|---|
| `c_see` | SEE says the move loses material | gap |
| `c_recap` | the move recaptures on the square the previous move captured on | sigma |
| `c_offpv` | per off-PV decision on the path from the root | cost of error |
| `check_extension` | the node is in check (this was a hardcoded `budget += PLY`) | cost of error |

The grouping is not decoration. `library/004` derives the budget a child
deserves as

```text
b ~ (PLY/gamma) * [ log2(sigma) - log2(gap) + log2(C) ]
```

so a feature earns its place by estimating one of those three quantities, and
that is what fixes its sign before any measurement is taken.

**All three new coefficients default to zero and each feature is computed only
if its coefficient is non-zero.** `chess bench` is **334110 nodes**, exact,
`cargo test --release` passes, `perft-suite` ALL PASS, and the tuner reproduces
the baseline to the digit (regret 41.46 ± 2.02, depth 7.03, 10551 nodes). The
refactor is proven inert rather than argued to be.

## The instrument this entry adds: an information-free control curve

Every result on this objective has the same confound. Any term that prunes
harder buys depth at fixed nodes, and depth buys regret. A feature that
carries no information at all will therefore *win* if you compare it against
the default.

`c0` is the intercept — a flat price charged to every priced child, identical
for all of them, carrying **zero** information. Sweeping it traces out the free
trade between depth and regret:

| `c0` | depth | b − a |
|---|---|---|
| 200 | 7.39 | −2.213 ± 1.699 |
| 400 | 7.70 | −1.585 ± 1.812 |
| 600 | 7.88 | +0.532 ± 1.881 |
| 800 | 8.04 | +2.888 ± 1.990 |

That curve is the bar. A new feature has to beat it **at matched depth**, and
the comparison must be run paired and directly (`--a c0=400 --b c_see=2500`),
not by differencing two comparisons against the default.

## `c_see` — monotone on both channels, and it clears the bar

| `c_see` | b − a | depth |
|---|---|---|
| 500 | −1.405 ± 1.577 | +0.27 |
| 1000 | −1.845 ± 1.650 | +0.40 |
| 1500 | −2.498 ± 1.702 | +0.54 |
| 2500 | −3.470 ± 1.794 | +0.67 |
| 4000 | −4.576 ± 1.789 | +0.74 |
| 6000 | −4.451 ± 1.789 | +0.72 |
| **9000** | **−4.617 ± 1.797** | **+0.74** |
| 13000 / 20000 | −4.617 ± 1.797 | +0.73 |

Monotone over five steps on both channels, then flat. The plateau is
mechanical and was predicted: past ~9000 milli-plies the term drives every
SEE-losing move into `max_base + max_slope*depth`, the price stops depending on
the coefficient, and the limit is *"an SEE-losing move gets the largest
reduction the ceiling allows"* — which for most of them is below `skip_below`,
so it is **SEE pruning in the main search**, a rule this engine only had inside
quiescence.

Against the information-free control, paired directly:

| a | b | depth a → b | b − a |
|---|---|---|---|
| `c0=400` | `c_see=2500` | 7.70 → 7.70 | −1.885 ± 1.914 |
| `c0=600` | **`c_see=4000`** | 7.88 → 7.77 | **−5.108 ± 1.931** |
| `c0=400` | `c_offpv=300` | 7.70 → 7.52 | +3.212 ± 1.947 |

**`c_see` wins by 5.1 ± 1.9 cp against a flat price while giving up 0.11 plies
of depth**, |t| = 2.6. "It just prunes more" is falsified.

This is `library/004` measurement 2 cashed. That measurement found `move_gain`
explains r² = 0.032 of the gap it is supposed to predict and an SEE veto takes
it to 0.080, and 004 read `c_gap`'s measured optimum of zero as falsifying
*the feature, not the OCBA argument*. The theory made a prediction — fix the
feature and the gap term becomes large and positive — and this is the first
time it has been tested. It came out the way the theory said.

## `c_offpv` — a clean negative, with a mechanism

| `c_offpv` | b − a | depth |
|---|---|---|
| 75 | −0.601 ± 1.631 | +0.15 |
| 150 | −1.658 ± 1.786 | +0.29 |
| 300 | +1.627 ± 1.866 | +0.49 |
| 600 | +0.019 ± 1.955 | +0.62 |

Depth rises monotonically, regret does not follow, and against the control at
matched depth it **loses by 3.2 ± 1.9 cp**. The two channels disagree, which is
the signature of a term that prunes without informing.

The mechanism is that it is nearly redundant by construction. The budget a node
holds is already `root_budget − sum of every price paid on the path`, so the
accumulated cost of being far from the PV is *already* carried additively by
the currency. A plain count of off-PV steps adds only the part the prices do
not distinguish, and that part appears to be nothing. **Path-shaped features
have to say what kind of path it was, not how long it was.**

## `c_recap` — inconclusive by construction, not measured at zero

| `c_recap` | positions that differ | b − a |
|---|---|---|
| −1500 | 22 / 1000 | +0.238 ± 0.536 |
| −500 | 18 / 1000 | +0.517 ± 0.468 |
| +500 | 68 / 1000 | +0.551 ± 0.825 |

A recapture at rank > 0 under a priced parent is rare enough that n=1000 cannot
see it. This is a statement about the corpus, **not** a verdict on the feature:
record it as unmeasured, and either broaden the definition (the previous move
was a capture, anywhere) or raise `--n`, which the nested sample makes cheap.

## `check_extension` — the hand-picked constant was already right

| value | b − a | depth |
|---|---|---|
| 0 (no extension) | +4.528 ± 1.664 | **+0.49** |
| 500 | −0.466 ± 1.617 | +0.25 |
| 750 | +1.203 ± 1.503 | +0.13 |
| **1000 (default, = `PLY`)** | — | — |
| 1500 | +1.245 ± 1.801 | −0.48 |
| 2000 | +6.427 ± 1.960 | −1.43 |

Deleting the check extension **gains 0.49 plies of depth and loses 4.5 ± 1.7
cp**, which is the sharpest available demonstration that this objective is not
simply rewarding depth. The default sits at the optimum the instrument can
resolve; nothing to buy here, and the constant is now in the registry where the
project's rules say it belongs.

## Decision

- Keep all four in the registry at their identity defaults. Nothing ships from
  a proxy.
- **`c_see` is the candidate for an SPRT.** At LEDGER 007's exchange rate
  (1 cp ≈ 25-30 Elo) 4.6 cp reads as ~115 Elo, which nobody should believe:
  that rate has been tested exactly once, in LEDGER 008, where it predicted
  +12-14 and delivered +10.4 [−8, +29] — not falsified rather than confirmed.
  The number worth having is the SPRT, at `c_see` somewhere in 2500-6000, and
  it should be run at fixed *work* as well as fixed nodes because the term
  moves the main:q mix.
- `c_offpv`: negative, mechanism understood, do not retry in this form.
- `c_recap`: unmeasured. Broaden or enlarge the corpus before judging it.
- The **information-free control curve is the standing protocol** for every
  future feature on this objective. Screen against `c0` at matched depth,
  paired and direct. Without it, a feature that carries nothing will look like
  a win.
