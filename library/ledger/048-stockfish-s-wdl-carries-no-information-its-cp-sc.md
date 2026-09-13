## 048 — Stockfish's WDL carries no information its cp score does not, but the three-way *outcome* does

`nnue/attic/wdl_dim2.py`, `wdl_shape.py`, `wdl_teacher_cubic.py`,
`wdl_teacher_buckets.py`, `wdl_ecal.py`, 2026-09-03. Corpus is
`chess-data/all.data` (893,500,985 records); samples are contiguous chunks at
2% / 35% / 68% of the file, 2/3 train and 1/3 held out by chunk. Stockfish
master `06675f7`.

**The proposal was to relabel with Stockfish's WDL instead of its cp score.
For Stockfish specifically that is a no-op, and the source says so.**
`win_rate_model` (`uci.cpp:537-602`) is

    material = P + 3N + 3B + 5R + 9Q,  m = clamp(material,17,78)/58
    W = sig((v-a(m))/b(m)),  L = sig((-v-a(m))/b(m)),  D = 1-W-L

with `a`, `b` cubics in `m` and eight hard-coded constants. `W,D,L` are a
deterministic function of the internal value and the material count — no
search information enters that `v` did not already carry. Stockfish's own
comment in `to_cp` states the converse: cp **is** the WDL log-odds score,

    logit(W) - logit(L) = 2v/b      exactly
    cp = 100 v / a

so switching from cp to a WDL log-odds score multiplies the target by
`2a/(100b)`, which ranges 0.080–0.109 over the material range. **A <=36%
material-dependent rescale is the entire content of "use WDL instead of cp".**

**But the outcome distribution is genuinely two-dimensional, and the second
axis is the larger one.** n=6M, at `|s| < 20`:

| material | n | W | D | L | E=W+D/2 | log(WL/D²) |
|---|---|---|---|---|---|---|
| 0–16 | 480,901 | 0.004 | 0.991 | 0.005 | 0.4994 | −10.85 |
| 32–42 | 50,188 | 0.094 | 0.796 | 0.110 | 0.4923 | −4.11 |
| 72+ | 64,020 | 0.292 | 0.389 | 0.319 | 0.4860 | −0.49 |

Expected score moves **0.013** across that range; the draw rate moves **0.60**.
By pawn count it is starker: 0–2 pawns is 1.000 draw, 12+ is 0.479, at the same
expected score. So a scalar target discards the larger axis — but only a real
three-way head can recover it, because Stockfish's WDL cannot (see above).

**The basis matters, and the obvious switch-on-sign forms are not monotone.**
Scores of the shape `(w>l)·f(W,D) − (w<l)·g(D,L)` measure distance from the
draw pole, which is a magnitude: they are discontinuous at `W=L` and rank a
+100cp position below a −100cp one. `A/D^p` is antisymmetric but fails at
`D→1`, where `A` does not go to zero: 2.2% of 30,628 group pairs are ordered
backwards vs expected score at the best `p`, and the least-squares and rank
fits disagree on `p` (0.16 vs 2.0), which is what a misspecified family looks
like. The coordinates that diagonalise a side swap are

    A = log(W/L)        negates under a swap
    S = log(W L / D²)   invariant under a swap
    (1-D)/D = 2 cosh(A/2) exp(S/2)          exact inverse
    E = W + D/2 = 1/2 + (1-D) tanh(A/2) / 2  exact, zero free parameters

`E` is a monotone function of `W − L` alone, so "maximise W against D" at fixed
`L` already **is** expected-score maximisation. The only real deviation is
trading `L` for `W`: `(1-D)^q tanh(A/2)` with `q>1`. That is a risk preference,
**it is not fittable on outcome data** (outcome data can only say `q=1`), and
draw utility 1/2 is the unique one keeping `f(W,D,L) = -f(L,D,W)` — so like
contempt it belongs at the root with the root side's sign fixed, never in a
leaf score both sides read.

**Stockfish's shipped WDL constants are unusable on this corpus.** Three-way
NLL, n=9M, held out. (The refit below was measured here and then *not* wired
into `train.py`'s `--wdl-a` / `--wdl-b` defaults, which stayed at values
scoring 1.20245 until 2026-09-04 — see LEDGER 049.)

| teacher | params | test NLL |
|---|---|---|
| SF shipped constants | — | **1.19934** |
| flat `a,b` (no material) | 2 | 0.73569 |
| cubic `a(m), b(m)`, refit | 8 | 0.65232 |
| per-`b_sym`-bucket (material × pawns) | 80 | **0.64018** |

1.84x the NLL of a refit of the *same functional form*. They are fit to LTC
Fishtest games from the root; this is mid-game plies of fast selfplay, and the
refit wants `b` about 3x larger (much less confident).

**Material dependence is worth 0.05% for `E` and 11.3% for the distribution.**
Same data, same model family. A nine-parameter free-K-per-material-bin model
beats a one-parameter global sigmoid by 0.00027 nats on expected score
(0.56324 → 0.56297) — nothing. On the three-way outcome it is 0.736 → 0.652.
The material story is worthless for the scalar and dominant for the draw
split, which is *why* the head is split into `A` and `S` at all.

**The finding that changed the experiment design: the WDL teacher's `E` is
about 2x better calibrated than the target the scalar arm already trains on.**
n=3M held out, per-score-bin `|E_pred − E_empirical|`:

| | n-weighted per-bin \|dE\| | RMSE vs outcome | NLL vs outcome |
|---|---|---|---|
| WDL bucket teacher | **0.0118** | **0.26436** | **0.55921** |
| `sigmoid(s/K)`, K=258.7 | 0.0222 | 0.26581 | 0.55991 |

So a two-arm design (scalar vs WDL head) varies **head shape and teacher
quality at once** and cannot attribute a win. Three arms are required: A =
scalar head + `sigmoid(s/K)`; B = scalar head + the WDL teacher's `E`; C = WDL
head + the three-way teacher. `C−B` is the head, `B−A` is the teacher.

**Instrument note, recorded because it nearly inverted a conclusion.** Mean
`|E − z|` with `z ∈ {0, 0.5, 1}` is minimised by predicting the *median*
outcome, which is a draw, so any model biased toward 0.5 scores well on it
regardless of accuracy. It ranked Stockfish's constants above the refit
teacher, contradicting every per-bin row. It is not a proper scoring rule;
RMSE and NLL both agree with the rows. **Do not score expected-score models
with MAE.**

**Decision.** The relabelling proposal is dead: it needs no Stockfish run,
because WDL is a closed-form function of cp. The head, three-arm design and
dual yardstick are in `train.py` (`--experiment wdl`); round-trip,
antisymmetry and `b_sym` flip-invariance verified to float32.

**The six-arm run was STOPPED partway and the head is SHELVED.** It reached
arm A complete (val 0.020066, outcome NLL 0.619415) and arm B step 2000 of
8000 before being killed; seed 2 never started, so there is no noise floor and
**no gap here is a result**. The one suggestive row, at matched step 800 and
worth nothing on its own: B was worse on val (0.022927 vs 0.022750, expected,
since val scores against the target A trains on) and better on outcome NLL
(0.625119 vs 0.625493).

Reasons for stopping, in order of weight. (1) The pre-registered prediction
was C ~ B on the scalar readout, because `D` cancels out of `E` and `E` is all
the readout consumes — so the expected headline was a null. (2) The payoff was
deferred to the search interaction (`f` sweep, `Pricing` re-fit per `f`,
SPRTs), which is days of work on an unconfirmed bet. (3) **The trunk was
wrong.** These arms are 1,041,226 params on 524M positions landing at 0.020;
the models that matter are 6.4M–12.7M params on 1.0–2.5B positions landing at
0.0152 (LEDGER 043), so a head result here might not transfer. (4) The
accumulator (STATE.md Next #2) is a known-size win — 0.017344, −18% against
the bar — blocked only on engineering, and nothing from the eval study can
deploy without it.

**What is worth keeping.** Arm B is the free part: a target measured ~2x
better calibrated on `E` than `sigmoid(s/K)`, at zero inference cost and zero
architecture change. It was never tested to completion. If this is revisited,
run A vs B alone — two arms, no WDL head — and on the 6.4M-param trunk, not
this one.

**Not decidable by val loss, if the head is ever picked up again.** The
search's margins are absolute cp, so a monotone reparametrisation of the
readout changes which nodes are pruned while changing no ordering. The `q`
sweep and the readout comparison go to `chess tune` regret with `Pricing`
re-fit per readout, then SPRT — never to val loss.
