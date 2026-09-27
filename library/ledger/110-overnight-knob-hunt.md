# 110 · Overnight knob hunt: 30 cells, four winners, the pricing layer is at an optimum

_2026-09-20. Run dir `<cluster scratch>`, drivers `hunt.sh`
(part 1, classical search knobs) and `hunt2.sh` (part 2, the pricing layer).
Binary `chess-head` (a873e77), net `m1-b1`, conc 32, hash 64, `nice 19`.
**354,000 games in 10.6 h**, zero engine errors and zero forfeits in all 30
cells. Screens are R3 `1.4+0.014`, 12,000 games; the gate cell is R1
`11.5+0.115`, 6,000 games._

## Why this was run

LEDGER 065 screened most of these coordinates and found "every direction is
worse" — but on the **root-regret proxy**, which floors at ~4.1 cp (061) and
cannot resolve a +5 Elo change. None of them had ever been screened on games,
because 12,000 games cost 12 h on the desk and cost 20 min here (108).

Pre-registered before the table was seen, because ~25 screens at SE 2.18 would
throw up ~0.6 cells above +4.4 by chance alone:

- **≥ +7 at R3 → earns an R1 confirm.** Expected false positives ~0.03.
- **+4 to +7 → bundling candidate**, not confirmed alone.
- **< +4 → recorded null.** Nothing ships on a screen regardless.

## Controls

| cell | n | Elo | θ | |
|---|---|---|---|---|
| A00_AA (null, opens the batch) | 12,000 | +1.53 [−2.8, +5.8] | 0.704 | ✓ |
| A30_AA_end (null, closes it) | 12,000 | −3.68 [−8.0, +0.6] | 0.692 | ✓ |
| A01_FUT (positive) | 12,000 | +8.02 [+3.8, +12.2] | 0.725 | ✓ 0.9σ from 108's +10.80 |
| A02_GATE | 6,000 | — | 1.069 | ✓ R1 target 1.06 |

- [FACT] θ is recoverable from W/D/L alone, so deleting the PGNs cost no θ.
  It cost everything else — see the caveat below.

## The gate: 103 was wrong

**A02_GATE = +15.36 [+10.71, +20.00]**, 6,000 games at R1, LOS 100%.

| | Elo | SE | vs. observed |
|---|---|---|---|
| observed | **+15.36** | 2.37 naive / **~4.9 honest** | — |
| component prediction (109) | +11.06 | 2.75 | 1.2σ — consistent |
| the gate as 103 measured it | +0.90 | 3.06 | **3.7σ — refuted** |

- [INFERENCE] With hash (104's h32 cell) and the split (109) both excluded,
  and the same comparison reading +15.4 when run **outside**
  `tools/checkpoint.sh`, the gate's own machinery during the 103 exit-code-bug
  window is the remaining explanation.
- [FACT] The 3.7σ against 103 survives the book correction of LEDGER 111 and
  the +15.36 does not. Same change, same 43 openings, same effective TC, so
  the opening effects are common to both measurements and cancel in the
  difference. Clustering inflates "is this > 0", not "do two measurements of
  the same thing agree".
- The +4.3 above prediction is partly real: the head binary also carries 106's
  psqt row padding (+1.6% nps ≈ +1 Elo), which the component cells did not.

## The screens

Honest σ applies LEDGER 111's R3 design effect (~1.8, so SE ×1.34). It is an
**estimate from the ladder's R3 cells, not a per-cell measurement** — these
PGNs were deleted by the driver and cannot be re-analysed.

| cell | change | Elo [95% naive] | σ naive | σ honest | verdict |
|---|---|---|---|---|---|
| A16 | `delta_margin` 100→200 | **+39.43** [+35.2, +43.7] | 18.3 | **13.6** | confirm |
| A24 | `c_see` 4000→2000 | **+13.15** [+8.9, +17.4] | 6.1 | **4.6** | confirm |
| A13 | `nmp_min_depth` 3→2 | **+10.25** [+6.0, +14.5] | 4.8 | **3.6** | confirm |
| A09 | `rfp_margin` 75→40 | +7.99 [+3.8, +12.2] | 3.7 | 2.8 | **demoted to bundling** |
| A15 | `nmp_base_reduction` 3→2 | +4.46 [+0.2, +8.7] | 2.1 | 1.5 | null |
| A04 | `iid_min_depth/nonpv`=2 | +1.82 | 0.8 | 0.6 | null |
| A26 | `c_recap`=500 | −2.58 | −1.2 | −0.9 | null |
| A28 | `c_usleave`=300 | −2.63 | −1.3 | −1.0 | null |
| A06 | `qevade_see`=1 | −2.78 | −1.3 | −1.0 | null |
| A22 | `pv_discount`=500 | −3.33 | −1.5 | −1.1 | null |
| A03 | `iid_min_depth/nonpv`=4 | −3.94 | −1.8 | −1.3 | null |
| A07 | `qsee_thresh`=−50 | −4.34 | −2.0 | −1.5 | null |
| A05 | `se_min_depth`=4 | −6.34 | −2.9 | −2.2 | negative |
| A10 | `rfp_margin`=110 | −9.85 | −4.5 | −3.4 | negative |
| A11 | `check_extension`=0 | −12.34 | −5.6 | −4.2 | negative |
| A23 | `pv_discount`=2000 | −12.98 | −5.8 | −4.4 | negative |
| A21 | `max_slope`=500 | −14.02 | −6.4 | −4.8 | negative |
| A14 | `nmp_min_depth`=5 | −16.92 | −7.7 | −5.8 | negative |
| A27 | `c_offpv`=200 | −16.89 | −7.7 | −5.8 | negative |
| A20 | `floor`=−200 | −22.03 | −9.8 | −7.3 | negative |
| A25 | `c_see`=6000 | −38.67 | −17.7 | −13.2 | negative |
| A19 | `skip_below`=−256 | −61.19 | −26.9 | −20.1 | negative |
| A08 | `rfp_max_depth`=0 | −82.41 | −34.8 | −26.0 | negative |
| A18 | `fare`=256 | −86.70 | −35.3 | −26.3 | negative |
| A12 | `check_extension`=2000 | −125.32 | −50.6 | −37.8 | negative |
| A17 | `fare`=64 | −266.52 | −88.2 | −65.8 | negative |

Five candidates were dropped before launch as inert — raising `se_min_depth`,
`nmp_base_reduction`, `rfp_max_depth`, `hist_cap`, `q_max_ply` or `c_hist`
above default is a no-op while lowering binds, the same "a second cap binds
first" signature 065 found for `max_base`. Screening an inert param reads ~0
whether or not the idea is right. `se_margin` and `se_depth_slack` failed the
liveness filter even with singular extensions on, which more likely means SE
fires rarely on those 4 positions than that the params are dead; they are
**not** claimed inert.

## What the winners have in common

- [FACT] Each winner has its opposite direction measured, and each opposite
  loses monotonically: `rfp_margin` 40 → +8.0 / 110 → −9.9; `nmp_min_depth`
  2 → +10.3 / 5 → −16.9; `c_see` 2000 → +13.2 / 6000 → −38.7. That is a
  slope, not four lucky cells.
- [INFERENCE] One story, not four: **prune harder on static-eval evidence
  (RFP, NMP), prune softer on material evidence (qsearch delta, SEE pricing).**
  Consistent with a qsearch node costing 0.275 of a main node — pruning the
  cheap thing to save time is a bad trade.
- [FACT] `delta_margin` moves the search by ±15% at fixed depth 12 over 5
  positions (0 → 548,682 nodes; 100 → 413,490; 200 → 467,051; 400 → 391,341),
  so +39 Elo is not a dead knob reading noise. `delta_margin=0` is the *most*
  expensive setting, because destroying qsearch accuracy wrecks move ordering
  upstream. **`chess bench` ignores `--set`** — six identical node counts was
  the tell; the measurement had to go through UCI.

## The pricing layer is at a local optimum, and one dormant feature is not dormant

- [FACT] Every price-formula knob except `c_see` loses, steeply, in the one
  direction tried: `fare` 64 → −266.5 and 256 → −86.7 (default 128),
  `skip_below` −61.2, `floor` −22.0, `max_slope` −14.0, `pv_discount` −13.0.
  These were never screened on games before; `library/013`'s defaults hold up.
- [FACT] `c_recap` and `c_usleave` read null at the magnitudes tried.
  **`c_offpv=+200` reads −16.89 at 5.8σ honest.** A coefficient with an effect
  that large is live and pointed the wrong way, not dormant.
- [FACT] `c_usleave=300` is the only cell whose θ is off-band (0.758 vs
  0.70 ± 0.05): it raises draws 51.8% → 55.1% at zero Elo. θ is a tripwire
  only on cells whose treatment is neutral.

## Decision

- **`delta_margin`, `c_see`, `nmp_min_depth` earn R1 confirms**; `rfp_margin`
  and `nmp_base_reduction` are bundling candidates only.
- **Nothing ships on a screen.** 109's screen/confirm disagreement stands as a
  warning even though 111 largely explains it.
- **Do not sum the winners.** `delta_margin` and `c_see` both act on
  material-losing moves and are expected to overlap. The combination is a
  factorial, not an addition — see `ROADMAP.md`.
- **Next candidates, each one 20 min:** `c_offpv=-200`; `c_see` at 1000/0;
  `delta_margin` at 150/300/400/600, since only one point on that curve is
  known and the node counts hint 400 may beat 200.
- 🔴 **Keep the PGNs.** This batch deleted them to save 40 MB a cell, which is
  why every row above carries an estimated rather than a measured design
  effect. That was the single most expensive decision of the night.
