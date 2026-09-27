# 121 — R1 confirms of the hunt survivors: IIR and hmc hold and ship (cp-0010, cp-0011), material moves-to-go does not

**Date:** 2026-09-27 · **Base:** `71be205` (cp-0009, bench 151742) · **Net:** `m1-b1.nnue`
· **Run dir:** `the GPU cluster:<cluster scratch>` (`runner.sh`, `queue.txt`, `cells.log`, PGNs kept)

## Setup

Four cells from LEDGER 117/118, each as one `--set` on cp-0009 against cp-0009
defaults: same binary (`x86-64-v3`) both sides, 8+0.08, 6000 games, conc 32,
hash 64, nice 19. **Pre-registered** in `runner.sh`: a candidate goes to
`tools/checkpoint.sh` only if the lower 95% bound is above 0, one change per
gate, largest first. `tm_mat_pct=70` was there only to show whether 50 is an
interior point, and was never a candidate to ship.

## Results

| cell | R3 (LEDGER 117/118, 12k) | **R1, 6000 games** | W / L / D |
|---|---|---|---|
| `iir_min_depth=3` | +12.4 | **+14.0 [+9.1, +18.8]** | 1280 / 1039 / 3681 |
| `hmc_scale=200` | +5.6 / +4.1 | **+8.7 [+3.8, +13.5]** | 1257 / 1107 / 3636 |
| `tm_mat_pct=50` | +4.9 [+0.9, +9.0] | −1.3 [−6.1, +3.5] | 1162 / 1185 / 3653 |
| `tm_mat_pct=70` | — | −3.4 [−8.3, +1.4] | 1110 / 1169 / 3721 |

- **IIR and hmc do not shrink R3 → R1.** They slightly grow, unlike the TM
  split's ×1.33 shrink (118). This fits a search change whose value does not
  come from using clock the old code left unspent.
- **Material moves-to-go is dead** [negative result]. The R3 +4.9 was a 1.2σ
  reading out of a sweep, and at R1 both values sit at or below zero. It stays
  dormant (`tm_mat_pct` default off). Do not re-run it without a new idea about
  where the time should go.

## Gates

Both at 8+0.08, SPRT [0, 10], desk conc 5, one change each, IIR first.

| gate | base | result |
|---|---|---|
| `iir_min_depth` 0 → 3 | cp-0009 | **H1 at 1872 games, +10.6 [+2, +19] → cp-0010** (`2852f72`, bench 148668). A first attempt reached +18.9 [−2, +40] at 350 games, LLR 1.20, and was killed by a machine shutdown. It was restarted from zero, not resumed. |
| `hmc_scale` 0 → 200 | cp-0010 | **H1 at 2604 games, +8.9 [+2, +16] → cp-0011** (`57a80ea`, bench 143603). `--games 8000` was set before the start, because at a true +8.7 a [0, 10] SPRT often runs out of games at 4000. It didn't need the extra games. |

Quote the R1 numbers (fixed n), not the gate's, which stop at a boundary.

## Status

cp-0011 is current: cp-0009 + IIR + hmc, bench 143603. The two R1 numbers
were each measured on cp-0009 alone, so **+22.7 for the pair is an upper
guess, not a measurement**, because the two can overlap. cp-0011 has not been
gauntleted. The last rating is cp-0009's 3349 ± 38 (LEDGER 120), and about 72%
of self-play gain carried over there.
