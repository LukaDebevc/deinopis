# The eval architecture study — findings

A long proxy-only study of what the eval should look like: which blocks of the
quadratic form matter, extra input features, bucket routing, a first NNUE head,
how to design routing rules, and a three-way outcome head. **Nothing in it has
played a game.** Headline numbers are against the `base r=512` bar of 0.021095.

This is the synthesis; each claim's full setup and caveats are in the ledger
entry it cites. It was lifted out of `STATE.md` on 2026-09-03, where it had
grown to 104 lines and pushed the handover file past being cheap to read.

- **Read-side bucketing is FREE; only `vbuck` pays a refresh.** `Bucketed` is
  `psqt(x) + <act(Vx), R[b(x)]>` and `a = Vx` does not depend on the bucket, so
  a bucket change costs a different reader row (nothing), a 512-wide shift swap
  (~10% of a piece move), or nothing at all for the bucketed PSQT. Every price
  in LEDGER 031/034 is a price for `vbuck`-style conditioning, which is the one
  design never trained. Read conditioning is worth −13.4% and is free; king
  routing costs the most and saturates at −5.8%.
- **Rank was never the constraint — the single reader was.** r=128 to 1024 buys
  −2.4% with one reader and **−9.6%** with 584. Best number in the study:
  **0.017344** (x584 material+count, r=1024). LEDGER 036.
- The NNUE head beats the quadratic form by 11.6% with 3.6x fewer parameters.
- Material read buckets are worth −13.4%; any king routing saturates at −5.8%
  by 121–256 buckets, so HalfKP's 4,096 buy nothing over Luka's 11 regions.
- A rule's predictive power can be scored WITHOUT training it (`nnue/attic/power.py`,
  Spearman 0.943 against six trained arms, 0.949 against a random-projection
  control). Balance alone predicts it at Spearman 0.883 — a good first filter,
  not a sufficient one.
- **The refresh price was wrong twice, and both fixes were Luka's.** First it
  was counted in the mover's perspective only, so a rule reading the opponent's
  half measured free, and the rule search found exactly that hole (LEDGER 031).
  Then it was measured on GAME positions: the tree's nodes carry **16.6 pieces
  against 19.4**, which nearly doubles the king-move rate, and move ordering
  changes which pieces move on top of that. Priced on the moves the search
  really makes (`chess movedump`, `nnue/treeprice.py`), per ply across both
  accumulators: **HalfKP 40.34%, material 24^2 22.55%** — 2-3x every number on
  record (LEDGER 034). Move ordering suppresses king moves by only 8%; the
  position effect is ~2x and had been missed entirely.
- **The designed coarsening lost.** `merged256` (184.6 effective buckets) fits
  WORSE than the one-line hand rule `material 24^2` (~62 effective) and costs
  more: 0.018583 vs 0.018264 at 24.31% vs 22.55%/ply. The merged curve is flat
  from 256 to 1024, so bucket count was never the constraint. Cause: only half
  the criterion was implemented — cut reduction, no value term — so the merge
  splits capacity between material and the king, and king routing is known to
  saturate at -5.8% (LEDGER 035). One cheap claim left: put the value term
  back. If a hand rule still wins, learned routing on this axis is finished.
- A coarsening must be TRAINED, not distilled from a trained fine model:
  post-hoc clustering to 16 buckets keeps -78% of the gain (LEDGER 033).
- Tying accumulator halves between perspectives does nothing (LEDGER 032).

- **A noise floor, at last: run-to-run val-loss spread is <=0.018%** on 1B
  positions (two seeds, five arms, LEDGER 044). So the 1B adapter study's 0.7%
  span was real and the 2.5B joint arms' 0.026% was not. Quote nothing under
  ~0.05% as a ranking.
- **`rand64` — 64 buckets with no chess in them — is worth −0.14% by itself.**
  Score a rule against `rand64`, not against plain psqr.
- **The stabler pawn hash wins.** Two 6-bit magics, same everything else: the
  one chosen for low refresh beats the one chosen for coverage by 0.85% of val
  loss (3.24% vs 2.42% against `rand64`), *even though* it is blind to the
  whole 4th rank — and rank 4 is the most expensive rank to discard,
  H(rank|rest) = 2.913 bits. Information about the structure and usefulness as
  a regime label are not the same quantity. LEDGER 044.

- **A learned pawn rule beats the best magic.** 10 annealed predicates
  `popcount(pawns & MASK) > k` -> 1024 states -> 64 buckets: **0.027432**, i.e.
  −3.75% against `rand64` where `newmagic` gets −3.25%, closing most of the gap
  to `king6` (−4.09%). But 67x more annealing bought 0.036% — **the greedy
  1024->64 merge is the ceiling, not the search** — and score-per-bucket-bit is
  still *rising* at 6 bits, unlike king routing. LEDGER 045.
- **A loss can buy bucket stability, and one form is nearly free.** Training
  on `-I(B;game)` over real game tape holds a bucket for 8 plies and beats
  `king6` on the quantity `king6` was best at (3.45 vs 1.43 bits), but taxes
  val at every weight tried (+1.5 / +2.1 / +5.2%). `tv` (squared change in the
  bucket posterior over real same-side move pairs) buys 2.6 points of flip2 for
  no measurable val cost. LEDGER 046. The two ends of that trade-off are now
  the seed of a search: `nnue/losslab/` has an LLM propose loss terms and ranks
  them by Pareto dominance on (val, flip2) — no scalar objective, because the
  exchange rate between eval accuracy and accumulator refreshes is what the
  search is meant to inform. `nnue/losslab/README.md`.
- **Score rules with I(B;G), not the Renyi-2 collision form.** The collision
  score cannot separate two magics that training separates by ~50x the noise
  floor; I(B;G) orders all four pawn rules monotonically with trained val loss.

- **Stockfish's WDL is a closed-form function of its cp score, so relabelling
  with it is a no-op** -- `uci.cpp:537-602` derives W/D/L from the internal
  value and the material count alone, and SF's own `to_cp` comment says cp
  *is* the WDL log-odds score. But the three-way *outcome* is genuinely
  two-dimensional and the second axis is the larger one: at a fixed score,
  expected score moves 0.013 across material while the draw rate moves 0.60.
  A real three-way head can recover that; Stockfish's WDL cannot. The head is
  built on the coordinates that diagonalise a side swap (`A = log(W/L)`
  negates, `S = log(WL/D^2)` does not), so W<->L symmetry is structural rather
  than learned. **The run was stopped partway and the head is SHELVED** -- the
  prediction was a null on the scalar readout (`D` cancels out of `E`), the
  payoff was deferred to an expensive search-side sweep, and the trunk was
  wrong anyway: 1.04M params at 0.020 against the 6.4M-param models at 0.0152
  that actually matter. One free thing was left untested: the WDL teacher's
  `E` is ~2x better calibrated than the `sigmoid(s/K)` the scalar arm trains
  on, at zero inference cost and no architecture change. LEDGER 048.

Full record, including everything queued on the GPU and what is known broken:
`library/009-eval-architecture.md` (an index; sections are files beside it).
Live bucket-rule work, with the next runs and what is still unproven:
`nnue/runs/SESSION-magic.md` and `nnue/runs/SESSION-router.md`.
