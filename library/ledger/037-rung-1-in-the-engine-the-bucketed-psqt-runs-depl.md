## 037 — Rung 1 in the engine: the bucketed PSQT (`runs/deploy.log`)

**Setup.** The one thing in the eval study that needs NO accumulator. `Bucketed`
is `psqt(x) + <act(Vx), R[b(x)]>`; conditioning the LINEAR term keeps the 32
gathers per eval and only widens the table, because the linear term has no
accumulator to rebuild (LEDGER 019). Two arms at r=768 — the rank the engine
already runs — identical except for that table, so the match measures it and
nothing else. `--experiment deploy`, 8000 steps, lr 1e-3, K = 258.7.

| arm | val loss | bench nps |
|---|---|---|
| deploy base | 0.021062 | 2.26 M |
| deploy psqt x material+count | **0.019805** (−5.97%) | 1.95 M (**−14%**) |

−5.97% replicates the −6.08% measured at r=512, so it survives the rank change.

**File format.** `export.py` writes version 4: v3's padding word becomes a
family count, so v3 files still load and re-exporting the deployed net is
**byte-identical**. Families are summed, not crossed — 576 rows + 8 rows, not
4608 — as `Bucketed` trains them. 584 x 768 i32 = 1.8 MB. `qeval.rs` folds the
bucket rows into the linear term, keeping the single rounding at the end.

**Verification.** The bucket index is the one part of the eval the engine
RE-IMPLEMENTS rather than reads from the file, so a mismatch is silent: the
eval stays plausible and every game is played from the wrong table.
- `chess evalfen` now prints the bucket indices. Checked against `rulestats.py`
  on **20,000 real search positions**, 134 distinct material buckets hit:
  **0 mismatches**.
- `nnue/verify.py` extended for v4 (`CHESS_PFAMS`): **0 feature mismatches,
  0 bucket mismatches**, engine − model **+0.019 cp mean / 0.661 max** on 400
  positions — unbiased, inside the 0.14 cp quantisation budget.
- `chess bench` **334,110 nodes**, unchanged through every commit.

**Result: no effect.** SPRT `--sprt 0,5` at 10+0.1, 32 MB hash, 4 concurrent,
both arms the same frozen binary copy with different `--quad` files. Stopped by
hand at **2574 games** (the LLR was wandering near zero and would not have
resolved inside the 3000 cap): **+572 =1416 −586, −1.9 Elo [−11, +7], LOS 34%,
LLR −1.10 [−2.94, 2.94]**. An early +30.3 at 92 games was noise; do not quote it.

So a **−5.97% proxy-loss improvement bought 0 Elo**, at a **14% nps cost** the
bucketed table also charges. This is the clearest example yet of the standing
rule that a proxy metric winning is not a result: the eval got measurably
better at predicting the training target and the engine did not get stronger.
Read it as a warning about the whole loss-driven study, not about this net.

**Two engine changes worth keeping.** `chess match --engine "<path> <args>"`
passes arguments to the subprocess, which is what lets one binary play both
sides of an eval SPRT; and a leading flag is no longer read as a subcommand, so
`chess --quad net.nnue` is the UCI engine rather than a help screen.
