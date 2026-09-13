## 046 — I(B;game) as a loss: the router loses the mirror, and pays 2.9% for it

**Setup.** 100M positions/arm, `all.data`, batch 65536, lr 1e-3, seed 0,
`--init-psqt ckpt-big/psqt_768-_64-_16-_1.pt`, `router bits6 st all`.
Launcher `nnue/runs/router-scripts/run_gametape.sh`, log
`nnue/runs/router_gametape.log`. Trajectory numbers are a separate n=1,500,000
pass of `nnue/attic/routergames.py` over the `tono_games.data` tail (12,553 games,
99.2% of steps one real ply), not the in-training probe.

Two new router penalties, both reading a `GameTape` — contiguous runs of
`tono_games.data`, where the game boundary and the ply are stored explicitly,
so grouping by game is read off rather than inferred:

- `ginfo=w`: `loss += w*(H_within - H_pool)` = `-w * I(B;game)`. "Spread across
  games, constant inside one" as one trainable number.
- `tv=w`: `loss += w * mean ||p(b|x_{t+2}) - p(b|x_t)||^2` over REAL same-side
  move pairs. Replaces `swing`, which priced a bucket change on an empty-board
  move graph with no legality, no captures and every piece equally likely.

| arm | hard val | vs none | flip2 | flip1 | runlen | eff_game | I(B;game) |
|---|---|---|---|---|---|---|---|
| king6 bar | 0.027550 | +6.8% | 18.5% | 78.2% | 1.28 | 7.3 | 1.43 |
| none | 0.025802 | — | 41.2% | 98.4% | 1.02 | 14.7 | 1.59 |
| tv=3e-3 | **0.025630** | **-0.67%** | 38.5% | 95.8% | 1.04 | 13.4 | 1.78 |
| tv=1e-2 | 0.025713 | -0.34% | 32.5% | 92.2% | 1.08 | 11.6 | 1.86 |
| ginfo=3e-4 | 0.026183 | +1.5% | — | — | — | — | 2.73 |
| ginfo=1e-3 | 0.026345 | +2.1% | 24.2% | 17.5% | 5.73 | 6.1 | 3.31 |
| ginfo=3e-3 | 0.027131 | +5.2% | — | — | — | — | 2.97 |
| ginfo=1e-3,tv=1e-2 | 0.026252 | +1.7% | — | — | — | — | 3.25 |
| ginfo=3e-3,tv=3e-2 | 0.026546 | +2.9% | 16.5% | **12.5%** | **7.99** | 5.2 | **3.45** |

**The result.** `ginfo` more than doubles king6's I(B;game) (3.45 vs 1.43) and
holds a bucket for **8.0 plies** instead of 1.0. The coupling is required: `tv`
alone reaches flip2 32.5%, `ginfo` alone 24.2%, the two together 16.5%. This is
the first thing that beats a hand rule on game-level structure.

**The surprise.** flip1 falls 98.4% -> **12.5%**. Per-ply flipping was
dominated by the side-to-move board mirror and no rule had escaped it (king6
78.2%). A bucket that must survive a whole game cannot depend on a quantity
that flips every ply, so `ginfo` forces mirror-invariance. That was queued as a
separate structural change (weight tying `w[our p, sq] = w[their p, mirror sq]`)
and the loss produced it unasked.

**The negative.** `ginfo` costs val monotonically — +1.5 / +2.1 / +5.2% at
3e-4 / 1e-3 / 3e-3 — and no weight escapes it. **Do not ship a `ginfo` arm at
these weights**: 3% of val for stability nothing currently charges for is a
straight loss today, because the deployed CoreLora v6 head
(`src/deepeval.rs:517`) shares the accumulator across buckets, so a flip
rebuilds nothing.

That is a statement about the CURRENT head and nothing more. It was first
written here as "a capability with no buyer", which overstated it: the router
exists to replace the king-bucket scheme that sits BEFORE the accumulator, and
in that design a flip is precisely what a refresh costs. The Elo consequence
has never been measured either way — no engine was built with a bucketed
accumulator, so "buys zero Elo" is an inference from reading the head, not a
result. At runlen 8.0 the refresh arithmetic changes by about an order of
magnitude, and every price on record for that design is for a design never
trained (031/034).

**Keep:** `tv=3e-3`, 0.025630, -0.67% val, and mildly MORE stable than no
penalty. **Corrected 2026-09-02:** the "-0.67%, ~37x the noise floor" reading
used 044's 0.018% val spread, which is not this arm's noise. Four replicates of
the no-penalty router at exactly this config give a single-run val sd of
**0.000061** (0.24% relative) and a flip2 sd of **1.13 points**
(`nnue/losslab/baselines.log`). Averaged over 3 seeds, `tv=3e-3` is 0.025684 vs
0.025775 — inside the val margin. Its flip2 gain (39.8% vs 42.5%) is real and
its val gain is not: `tv=3e-3` does not dominate the no-penalty router, it sits
beside it on the frontier.

**Superseded — `bal` was the wrong instrument** (`nnue/runs/router_balcross.log`,
same config). `bal` pushes the MARGINAL to uniform over all 64 buckets, which is
not conditional structure: eff buckets 46.6 -> 63.5 and val +2.1..+2.9% at every
weight from 0.003 to 0.03, crossed with `swing` at 1e-4 and 1e-3. king6, the
best rule on record, uses eff 20.6. Use `ginfo`, not `bal`.

**Instrument note.** The in-training flip2 probe was 8 windows x 256 plies and
put `none` at 30.9% where n=1.5M says 41.2% — wrong by more than the effect
being measured, because plies inside a game are correlated. Raised to 64x512.
