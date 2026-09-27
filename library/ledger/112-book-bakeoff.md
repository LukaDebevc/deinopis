# 112 · The book bake-off: 4000 lines cut the honest R1 interval 1.7x, and the default is now `lich.epd`

_2026-09-20. `<cluster scratch>`, four cells, 3.2 h.
Scripts `bakeoff.py`, `loo.py`, `balance2.py`, `xcorr.py` in
`<cluster scratch>`. One manipulation: the opening book. Same binary
(`a873e77` + FEN book support, bench 201835), same net, same TC, same change
under test (`fut_max_depth=3`)._

## Result

| cell | Elo | SE naive | SE honest | deff | shuffled | wall | SE @ 1 h |
|---|---|---|---|---|---|---|---|
| builtin R1 | +11.82 | 2.44 | **4.41** | 3.27 | 1.03 | 1.30 h | **5.03** |
| lich R1 | +11.12 | 2.61 | **2.61** | 1.00 | 1.00 | 1.28 h | **2.95** |
| builtin R3 | +10.77 | 2.18 | 2.56 | 1.38 | 0.79 | 0.32 h | 1.46 |
| lich R3 | +8.77 | 2.23 | 2.40 | 1.16 | 1.13 | 0.31 h | 1.33 |

Pre-registered: builtin R1 ~4.9, lich R1 ~3.3, lich wins ~1.5x; falsified if
lich were no better. **Measured 4.41 vs 2.61 — 1.69x, not falsified.** In time
terms lich is 2.9x cheaper at R1, 1.2x at R3.

- [FACT] The win decomposes exactly: `sqrt(3.27) = 1.81x` from removing the
  clustering, `× 0.935x` lost per game to sharper positions, `= 1.69x`.
- [FACT] **lich carries 12% less information per game** (naive delta 4.26 vs
  4.84). That replicates LEDGER 096's 0.83 [0.36, 1.46] with a tight interval.
  096's Fisher-information ceiling stands: sharpness buys ≤1.14x and we are
  not collecting it.
- [INFERENCE] **All of the gain is independence, none of it is sharpness.** A
  4000-line *balanced* book would collect ~1.6x of the 1.69x with no scale
  change. That is the better book, and it remains unsourced — but it is worth
  ~6%, not worth a night.

## Two caveats on the table above

- **lich's `deff = 1.00` at R1 is by construction, not measured.** 3000 pairs
  over 4000 lines means each opening appears once, so the cluster estimator
  degenerates to the naive one identically. The structural argument is sound —
  nothing repeats, so there is no within-opening clustering to hide — but no
  measurement was made. At R3 (m = 1.5) the shuffled control (1.13) is as
  large as the real estimate (1.16): that cell has no resolution either.
- **`deff` on 43 groups is noisy to roughly ±25%.** 108 read R1-FUT at 4.27
  and R3-FUT at 2.26; this run reads 3.27 and 1.38 for the same cells. Use
  LEDGER 111's rule of thumb as an order of magnitude, not a coefficient.

## What was ruled out

- [FACT] **`book[15]` was not carrying the design effect.** Leave-one-opening-out
  on 108's own PGNs: the largest single contributor is `book[18]` at −0.89 of
  4.27, and `book[15]` is not in the top six. The clustering is spread across
  ~40 openings. The drop from 108's `deff` to this run's is noise, not the fix
  working. (My hypothesis; wrong.)
- [FACT] **`lich.epd` is UHO-class on our own eval** (depth 10, n = 400):
  median |cp| **153.5**, 90.5% of positions over 100 cp, against the builtin
  43 at 56.0 / 16.3% and a startpos anchor of +31. Pohl.epd, which 111
  rejected, reads 149. This is fishtest's deliberate design, not a defect —
  and by the decomposition above it is where the 12% per-game loss comes from.
- [INFERENCE] **No detectable Elo-scale shift.** FUT reads +11.12 lich vs
  +11.82 builtin at R1 and +8.77 vs +10.77 at R3; pooled difference
  **1.58 ± 2.90 Elo**, ratio 0.86 ± 0.26. Consistent with 1, but a 25% shift
  is not excluded. Unlike 096's uho8, which inflated 1.34x — probably because
  lich is two-sided (60.8% white-favoured, not ~100%). Watch it; do not
  rescale SPRT bounds on this evidence.
- [FACT] **A factorial is not doubly exposed to the book.** Per-opening effect
  profiles of different changes at R3 correlate raw r ≈ +0.20 (disattenuated
  ≈ +0.45), with the AA null reading |r| ≤ 0.38 as a control. A contrast
  between two cells therefore carries ~1.1x a single cell's opening variance,
  not 2x.

## Decision

**`books/lich.epd` is the default book for `chess match`, compiled in.**
`--book builtin` restores the 43 lines, which is what every number before this
entry was measured on. Compiled in rather than read from disk because the
binary is copied to the cluster alone, and a default that needs a file beside
it is the same failure mode as `chess bench` reporting two node counts
depending on whether `quad.nnue` was there (LEDGER 109).

Tests, perft suite and bench are unchanged (63 pass, perft exact, bench
201835). `default_book_is_legal_and_distinct` checks the compiled-in book
parses, is legal, has no repeated position and has ≥1500 lines.

**Not done, deliberately:** the 15-cell level fan and the 16-cell factorial
(10.3 h) were staged and are cancelled. At ~3250 Elo the patches in them are
worth +5 to +15 each; ten hours to resolve differences among them is the
+1 Elo grind this project cannot win. Bundle and gate instead.
