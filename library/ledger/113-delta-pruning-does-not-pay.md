# 113 · Delta pruning does not pay: the rule made the tree bigger

_2026-09-21. Screens in `<cluster scratch>` (`hunt-dm.sh`, cells
C0–C2) and `<cluster scratch>` (`hunt.sh`, cells H06–H08).
Confirm on the desk, `~/chess-runs/20260921-dm1200/`, `desk-tomorrow/run.sh`.
Binary cp-0006 (`499b4d6`) against itself, one `--set` apart; net `m1-b1`
(sha `ce1f2656`); book `lich.epd`, 4000 lines. Screens R3 `1+0.01`, conc 32,
hash 64, 12,000 games. Confirm R1 `8+0.08`, conc 5, hash 32, 6,000 games,
fixed n — never an SPRT._

## The claim

`delta_margin` is not a constant that wanted tuning. The rule it controls —
one line at `search.rs:1766` — costs the search about 15% of its nodes and
roughly **+40 Elo at the gate's own time control**, and the level fan never
turns around anywhere inside its declared box. This is the change cp-0007
ships.

## The level fan

Every cell is cp-0006 against cp-0006 with one override, so the manipulation
is the constant and nothing else. R3 screens, 12,000 games each:

| `delta_margin` | R3 `1+0.01` | R1 `8+0.08` | ratio |
|---|---|---|---|
| 150 | **−19.01** [−23.70, −14.33] | — | — |
| 200 (cp-0006) | baseline | baseline | — |
| 300 | +22.38 [+17.77, +27.00] | — | — |
| 400 | +49.02 [+44.49, +53.57] | **+24.83** [+19.56, +30.10] | 1.97 |
| 600 | +67.61 [+63.01, +72.23] | — | — |
| 800 | +80.75 [+75.90, +85.64] | — | — |
| **1200** (box max) | **+84.06** [+79.32, +88.84] | **+40.02** [+35.02, +45.03] | 2.10 |

Monotone across the whole box, saturating from 600 up — 800 and 1200 differ by
3.3 ± 6.8, which is nothing. Because a *larger* margin prunes *less*, 1200 is
the rule switched off in all but name: it still fires, but only when the static
eval is twelve pawns below alpha.

This is why C0–C2 were run as a structural question rather than a tuning one.
A flat top means the answer is "the rule should not exist", not "the constant
should be 700".

## The mechanism: it was never a saving

The obvious story — pruning less buys accuracy and pays for it in nodes — is
wrong, and `chess bench` says so without playing a game. Depth 9, 12
positions, net `m1-b1`, same binary one `--set` apart:

| `delta_margin` | bench nodes | nps (mean of 4) | Δ nodes | time to d9 |
|---|---|---|---|---|
| 200 | 179,364 | 817k | — | 0.2195 s |
| 400 | 153,566 | 806k | −14.4% | 0.1905 s (−13.2%) |
| 800 | 152,918 | 792k | −14.7% | 0.1931 s (−12.0%) |
| 1200 | 151,742 | 809k | **−15.4%** | 0.1875 s (**−14.6%**) |

nps is flat — the 3% spread across the four is inside the 3–4% run-to-run
spread of any one of them — so the looser filter costs nothing per node, and
the tree gets **smaller**.

⚠️ **Build note, because this table was wrong once.** The first run of it read
~413k nps across all four. Those binaries were built in a scratch tree that had
been given `src/` and `Cargo.toml` but *not* `.cargo/config.toml`, so they
compiled to baseline x86-64 with no popcnt, BMI1 or AVX — about half speed on a
node that is 85% eval. Node counts are unaffected by this (they are
deterministic, and the non-native and native builds agree exactly), and the
flat-nps conclusion survived the rebuild, but the absolute column did not. A
bench whose nps is ~2x off the repo binary's is the symptom to look for.

The move mix says where the saving goes (200 → 1200):

| | 200 | 1200 | Δ |
|---|---|---|---|
| main quiet | 104,043 | 87,080 | −16.3% |
| main capture | 37,067 | 29,001 | −21.8% |
| q capture | 23,329 | 23,482 | **+0.7%** |
| q quiet | 3,364 | 2,883 | −14.3% |

Quiescence does the same amount of work and returns better bounds, so the main
search takes its cutoffs sooner. Delta pruning at 200 cp was not buying nodes
back — it was spending them. [INFERENCE on the causal story; the node and Elo
numbers are measurements]

**What the bench does not explain.** Nodes are already flat at
`delta_margin=400` (−14.4% vs −15.4% at 1200) while Elo climbs another +35
over that same stretch. So the top half of the fan is quiescence returning
*better values*, not cheaper ones, and `bench` is blind to it. A prediction
made from the node count alone would have stopped at 400 and left +15 Elo at R1
on the table.

## The confirm

6,000 games, 8+0.08, conc 5, hash 32, desk idle, both arms cp-0006 on `m1-b1`:

**+40.02 Elo [+35.02, +45.03], LOS 100%** — `+1585 =3518 −897`.

Checked before being believed:

| check | result |
|---|---|
| colour balance | 2967 / 2967 exact |
| independent recount off the PGN | +39.8 at the time of the read, matches the log |
| **time forfeits** | **zero**, either side — the bigger-tree arm never flagged |
| terminations | 4503 adjudication · 1361 repetition · 36 fifty-move · 27 insufficient · 8 stalemate · 1 mate |
| gain by colour | +487 net as white, +190 net as black — present on both |

The forfeit check is the one that mattered. Arm B searches a 15% bigger tree,
so "the loser was losing on the clock" was the live boring explanation for an
effect this size. It is dead: not one time loss in 6,000 games.

## Transfer

The R3→R1 ratio is **1.97 at dm=400 and 2.10 at dm=1200**. This was not
guaranteed and I predicted it might not hold: since the node-count mechanism
covers only the bottom of the fan, the top half could plausibly have transferred
at a different rate. It did not. The quality half and the speed half decay
alike. (Asterisk: the two ratios come from different boxes — C5 ran conc 32 /
hash 64 on the cluster, the desk confirm conc 5 / hash 32.)

STATE.md's standing "R3 over-predicts its gate by 2–3x" should read **~2.0x**;
see 114 for the other two measurements of the same ratio.

## The gate

`tools/checkpoint.sh`, 2026-09-21: tests 63/63, perft suite exact, bench
151742 @ ~810k nps on `m1-b1`, then SPRT [0, 10] vs cp-0006 as a subprocess.

**H1 accepted at 440 games: +106 =269 −65, +32.5 Elo [+14, +51], LLR 3.00.**
Tagged `cp-0007`, pushed.

The gate's +32.5 and the confirm's +40.02 are the same measurement at very
different n — the gate's interval [+14, +51] contains +40.02 comfortably, and
440 games cannot separate them. **Quote the fixed-n +40.02 [+35.02, +45.03];
the gate is a pass/fail, not an estimate.** Note it landed *below* the
fixed-n value, where an SPRT's stopping rule biases away from zero — that is
noise at this n, not a countersignal.

## What shipped

`delta_margin` default 200 → **1200**, knob kept, box unchanged. cp-0007
(`b82423e`), tag carries net sha `ce1f265647c060a2`.

Kept as a tunable rather than deleted, deliberately: the gate should test what
was measured, and what was measured is the constant. Deleting the branch is a
near neighbour nobody has played a game on, and the knob is part of the
`search::Params` surface the learned controller is meant to replace.

**Open, and worth naming:** the default now equals the top of its own box, so
the box can no longer express "more". Either the box widens or the line goes.
Neither is measured; both are follow-ups, not this entry.

## Not bundled

`tm_moves_to_go=24` + `tm_inc_pct=90` is +18.20 [+12.76, +23.64] at the same
TC and n (114), and was deliberately left out of cp-0007. Its confirm ran at
3x box load and its own A/A null came back **−4.69** [−11.02, +1.64] rather
than ~0, on the knob most sensitive to contention there is. That is an open
confound, and it does not belong inside the largest clean result this project
has recorded. It gets a 1x confirm and its own checkpoint.
