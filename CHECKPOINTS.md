# Checkpoints

Every row is a version that passed the gate in `tools/checkpoint.sh`: correct
(perft + tests exact), and beaten-the-previous-checkpoint at an SPRT, played
between the two binaries over the same UCI path. Each row has a matching
annotated git tag, so any of them can be rebuilt and re-tested:

```
tools/checkpoint.sh -m "what changed"     run the gate; push only if it passes
tools/checkpoint.sh --restore-baseline    rebuild the reference from the last tag
git tag -l 'cp-*'                         every kept version
```

A failed gate writes nothing and pushes nothing. Failures are not recorded
here — they belong in `LEDGER.md`, with the same care as the successes.

**The bench column is net-dependent, so it names the net that was deployed at
the time** — the sha is in the annotated tag. A gate that benches one eval and
plays another is exactly how cp-0006's row came to be wrong for a day
(LEDGER 114); `checkpoint.sh` now resolves the net from `publish/net.sha256`
and hash-checks it, so the two cannot drift apart again.

| tag | date | what changed | bench nodes | strength vs previous |
|---|---|---|---|---|
| cp-0001 | 2026-09-06 | WDL eval by default, 2.2x faster kernel, contempt as a readout | 396299 | first checkpoint — no reference to compare against |
| cp-0002 | 2026-09-08 | i16 accumulator, c_see pricing, m1-a1 net | 219127 | 380 games: +95 =229 -56   +35.8 Elo  [+16, +56]   LOS 100.0%   LLR 2.96 [-2.94, 2.94] |
| cp-0003 | 2026-09-09 | o1 SEE-split ordering, m1-b1 net | 243605 | 684 games: +135 =455 -94   +20.9 Elo  [+7, +35]   LOS 99.8%   LLR 2.99 [-2.94, 2.94] |
| cp-0004 | 2026-09-12 | quiescence TT sharing | 199091 | 246 games: +65 =159 -22   +61.4 Elo  [+34, +90]   LOS 100.0%   LLR 2.95 [-2.94, 2.94] |
| cp-0005 | 2026-09-16 | bundle persist-all + conth + easy-move (LEDGER 094) [0,5] | 219718 | 3776 games: +543 =2772 -461   +7.5 Elo  [+2, +13]   LOS 99.5%   LLR 2.95 [-2.94, 2.94] |
| cp-0006 | 2026-09-20 | bundle: fut_max_depth=3, delta_margin=200, c_see=2000, nmp_min_depth=2, psqt 16-byte stride | 179364 | **6000 games: +1997 =3050 -953   +61.1 Elo  [+55, +67]** at 8+0.08, fixed n (LEDGER 114). Its own 448-game gate read +37.4 [+17, +58] but played `gate-3ep-q8.nnue` on both sides, not the shipping net — corrected here, not appended |
| cp-0007 | 2026-09-21 | delta_margin 200->1200: qsearch delta pruning does not pay (LEDGER 113) | 151742 | 440 games: +106 =269 -65   +32.5 Elo  [+14, +51]   LOS 100.0%   LLR 3.00 [-2.94, 2.94] — **fixed-n confirm: +40.02 [+35.02, +45.03] over 6000 games** (113); quote that, not the gate |

**Absolute rating is measured separately, against CCRL-rated opponents**, not
by this gate — `tools/ladder.sh run`. Latest: **cp-0007 at about 3293 ± 29 on
the CCRL Blitz scale**, 7 anchors × 200 games at 10+0.1 (LEDGER 115). cp-0004
was 3246 ± 33 (088), so cp-0005..0007 bought +47 against foreign opponents
where the self-play rows above claim +108.6 between them.
| cp-0008 | 2026-09-22 | time management: tm_moves_to_go 30->24, tm_inc_pct 75->90 (LEDGER 114, 1x confirm) | 151742 | 508 games: +114 =316 -78   +24.7 Elo  [+9, +41]   LOS 99.9%   LLR 2.95 [-2.94, 2.94] |
| cp-0009 | 2026-09-27 | time management: hard stop at 5x the allotment + instability/falling-score stretch of the start gate (tm_hard_pct 100->500, tm_instab 0->100, tm_fall 0->20; R1 +59.3 [54.4, 64.2] n=6000); q-Exact TT stores no longer displace deeper entries; dormant hunt flags | 151742 | 296 games: +78 =178 -40   +44.9 Elo  [+22, +68]   LOS 100.0%   LLR 3.01 [-2.94, 2.94] |
| cp-0010 | 2026-09-27 | search: internal iterative reduction at depth >= 3 (iir_min_depth 0->3; R1 +14.0 [9.1, 18.8] n=6000) | 148668 | 1872 games: +367 =1195 -310   +10.6 Elo  [+2, +19]   LOS 99.3%   LLR 3.01 [-2.94, 2.94] |
| cp-0011 | 2026-09-28 | search: static eval drifts toward draw with the halfmove clock (hmc_scale 0->200; R1 +8.7 [3.8, 13.5] n=6000) | 143603 | 2604 games: +512 =1647 -445   +8.9 Elo  [+2, +16]   LOS 99.3%   LLR 3.03 [-2.94, 2.94] |
