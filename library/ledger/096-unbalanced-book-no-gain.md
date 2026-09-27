# 096 · An unbalanced opening book cuts draws and buys nothing

_2026-09-16. Run dir `~/chess-runs/20260916-bookpilot/` (binaries, books,
PGNs, `analyse.py`). Commit: book support in `src/book.rs`, tree at 199115._

## The question

Our SPRTs draw 71–72% of their games (FX-1a 2133/3000, FX-3/4 gate
1877/2632, both 8+0.08). A draw carries less information than a decisive
game, so the standard move — fishtest made it in 2022 — is to start games
from positions one side is already winning, which converts draws into
wins for whoever is better at converting. The hope was fewer games per
verdict.

**It does not work here.** The draw rate falls from 72% to 58% and the
number of games to a decision does not move.

## Setup

One manipulation: the opening book. Everything else is held fixed —
`cp-0004` vs `cp-0003` (a real search change, quiescence TT sharing,
LEDGER 087), the same `m1-b1` net named on both sides, 8+0.08, concurrency
5, 64 MB hash, both arms as subprocesses of the same driver binary.
Books are interleaved in 200-game chunks so a load change on the box hits
every arm equally; each chunk of an unbalanced book uses its own 100
positions. Bench verified before the run: cp3 243605, cp4 199091.

| book | what it is |
|---|---|
| builtin | the 43 balanced theory lines, 4–8 plies, what we use today |
| uho8 | `8mvs_big_+80_+109.epd`, 16 plies, Stockfish scores white +0.80..+1.09 |
| x120 | `UHO_XXL_2022_+120_+149.pgn`, 16 plies, white +1.20..+1.49 |
| lich | `UHO_Lichess_4852_v1.epd`, fishtest's current default, either side to move |

All three are Stockfish's (CC0), shuffled once with a recorded seed, and
**not** selected by our own eval — a book of "positions our eval likes"
would select for the thing under measurement.

## Result

`delta` is the effect size per pair, `(mean pair score − ½) / sd`. Games to
a decision scale as `1/delta²` when the SPRT bounds are stated in those
units. Our SPRT fixes its bounds in **Elo**, so what it actually costs goes
as `(mean − ½)/variance` — the last column, and the operational number.

| book | pairs | draw % | adv. side W/D/L | Elo | delta | delta ratio [95%] | games at [0,5] |
|---|---|---|---|---|---|---|---|
| builtin | 600 | 71.8 | 19/72/9 | +32.5 | 0.262 | 1 | 1 |
| uho8 | 600 | 57.5 | 33/58/10 | +43.7 | 0.288 | 1.10 [0.71, 1.71] | 1.10 |
| x120 | 300 | 58.3 | 33/58/8 | +37.2 | 0.259 | 0.99 [0.54, 1.70] | 1.16 |
| lich | 300 | 55.2 | 36/55/9 | +31.9 | 0.216 | 0.83 [0.36, 1.46] | 1.44 |

Wall clock per 200 games: builtin 755 s, uho8 731 s, x120 715 s. Unbalanced
games are ~12% shorter (median 151 plies vs 172), worth ~4% more games per
hour, which does not change the conclusion.

**Nothing here beats the balanced book.** The interval on uho8 admits a real
gain of up to 1.7x in delta, so this rules out a large win, not a small one.

## Why, mechanically

Model a game as a latent normal `N(b + Δ, 1)` cut at `±θ`: above `+θ` the
side to move wins, below `−θ` it loses, between them a draw. `Δ` is the
engine gap, `b` the book's head start. Fit `θ` and `b` **per book** from
that row's W/D/L — they are not the same across books, which is the point:

| book | draws | `θ` | `b` |
|---|---|---|---|
| builtin | 72% | 1.109 | — (the 0.231 fitted here is `Δ`, = +32.5 Elo) |
| uho8 | 57% | **0.868** | 0.419 |
| x120 | 59% | 0.915 | 0.484 |
| lich | 55% | 0.850 | 0.491 |

The unbalanced positions are not only skewed, they are **sharper**: the draw
band narrows from 1.11σ to 0.87σ. The Fisher information per game about `Δ`,
`I = Σ (∂pᵢ/∂Δ)²/pᵢ` over win/draw/loss, evaluated at the true `Δ` and
averaged over which side the engine draws:

| book | builtin | uho8 | x120 | lich |
|---|---|---|---|---|
| `I` | 0.697 | 0.761 | 0.748 | 0.758 |
| ratio | 1.00 | 1.09 | 1.07 | 1.09 |

Predicted 1.08x against a measured 1.10 — but the interval is wide, so the
agreement is not the argument. **The argument is the ceiling.** Maximising
`I` over all `(θ, b)` gives **0.810**, at a balanced book with ~46% draws.
No book can beat that, so a balanced book's draw rate fixes the entire
headroom:

| balanced draw rate | `θ` | `I` | most *any* book can buy |
|---|---|---|---|
| 50% | 0.674 | 0.808 | 1.00x |
| **72% (us)** | 1.080 | 0.708 | **1.14x** |
| 80% | 1.282 | 0.616 | 1.31x |
| 85% | 1.440 | 0.534 | 1.52x |
| 88% (fishtest LTC, unverified) | 1.555 | 0.473 | 1.71x |
| 92% | 1.751 | 0.371 | 2.18x |

This is why Stockfish gains and we do not, and it is one-sided: it does not
depend on choosing the right book, only on our own draw rate. At 72% there
are at most 14% of the games on the table from any opening book that exists.

**And the skew is not what helps.** At `Δ = 0`, isolating the two changes:

| | `θ` | `b` | `I` | |
|---|---|---|---|---|
| balanced | 1.109 | 0 | 0.696 | |
| uho8's band, no skew | 0.868 | 0 | **0.778** | sharpness alone, 1.12x |
| uho8 as it is | 0.868 | 0.419 | 0.765 | plus the skew, 1.10x |

The imbalance is a small **negative**; every bit uho8 gained came from the
positions being sharp. A *balanced* sharp book would collect the same 1.12x
with no change to the Elo scale — see the decision below.

The scale change shows up in the Elo column: the gap reads +43.7 on uho8
against +32.5 on the builtin book (1.34x, itself only 1.4σ). Of that 1.34x,
1.22x is inflated pair variance and 1.10x is real.

## Two things this also settles

**A Stockfish +1 position is not a 50/50 win for us.** Our engine wins
those positions 33% of the time and draws 58%. Raising the band to
+1.20..+1.49 changed that by less than the noise (33/58 both), so the 50/50
balance point — if a book could reach it — is far beyond +1.5. Stockfish's
"+1.00 means a 50% win" calibration is a statement about Stockfish's own
play, not about a 3250-rated engine's.

**FX-12 / MS-5: the 43-line book is not replaying games — but the
conclusion drawn from that was wrong.** 600 games off the 43 lines contain
**598 distinct games**; clock jitter decorrelates repeats. What this entry
then concluded — "the worry that a small book narrows our intervals was
unfounded" — **is false, and was corrected by LEDGER 111 and 112.** The
intervals *are* narrowed, by up to 1.8x at R1, and a duplicate-game check
cannot see it: the mechanism is that a change is worth different amounts in
different openings, so ~70 replays of each of 43 lines are correlated even
when every game is distinct. A large book is needed, and is now the default.

## Decision

Keep the balanced builtin book as the default for `chess match`, which also
keeps the gauntlet's absolute ratings on the same footing as before.

Keep the book *support* — EPD and FEN starts, `--book <file>`, the
compiled-in `--book uho`, `checkpoint.sh --book`. It costs nothing, it is
what made this measurement possible, and the conclusion is draw-rate
dependent: at a longer time control, or after the engine gains a few
hundred Elo, `θ` rises and this is worth re-running. `books/README.md`
says how.

**Do not switch the book and keep `--sprt 0,5`.** This one needs no model.
The bounds are in Elo. On uho8 the same engine gap reads +43.7 against
+32.5, so H1 at 5 Elo on that scale is H1 at **3.7 Elo** on the old one:
switching the book without rescaling the bounds accepts patches worth 3.7
as if they were 5. Rescale them honestly and the gain is exactly the
1.10 [0.71, 1.71] measured above. The apparent speedup is the units.

**Open, and the version of this idea worth testing: a sharp *balanced*
book.** The decomposition says the skew is a small negative and all of
uho8's gain came from `θ` narrowing. A balanced book with uho8's sharpness
would collect the same 1.12x with no scale change, no rescaled bounds and
no gauntlet discontinuity. Whether a balanced book can reach 58% draws for
us is unmeasured — same design as this study, one evening. Still capped at
1.14x, so it ranks below the time control.

## A side effect: cp-0004 is worth about +33, not +61

The builtin arm is 1200 games of `cp-0004` vs `cp-0003` under the gate's own
conditions, and it reads **+32.5** where LEDGER 087 and `CHECKPOINTS.md`
record +61.4 [+34, +90] over 246 games. Nothing disagrees: 087 was an SPRT
that stopped as soon as it crossed H1, and a test that stops on crossing
reports the crossing, so its point estimate is biased upward. 1200 fixed
games are the better estimate of the same quantity.

This is worth knowing whenever a checkpoint's Elo is quoted as the engine's
gain — every gate number in `CHECKPOINTS.md` carries the same upward bias,
and the smaller the game count the larger it is.

## What would actually cut games

**What this study does not cover.** It measures games-to-a-decision on one
search change. Fishtest's other reason for UHO is not speed but *visibility*
— at 85%+ draws, patches that only cash out in sharp positions produce
almost no decisive games at any sample size. The ceiling argument above says
nothing about that, and if the question is "does a sharp book make different
patches measurable" this run is silent. Different experiment.

The honest next lever for game count: games per hour. A shorter
time control buys them directly (8+0.08 → 4+0.04 is ~2x), at the cost of
measuring a slightly different engine. Same instrument, same design as this
study — one manipulation, interleaved arms, `delta²` per hour as the
score — and it would answer it in an evening.
