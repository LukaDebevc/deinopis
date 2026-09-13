# 050 — The engine can run the WDL net, and the 10B run was saving nothing

_2026-09-04. `src/wdleval.rs`, `nnue/export_wdl.py`, `nnue/verify_wdl.py`._

## What was built

`src/wdleval.rs` runs the `combo` arm of the `runs/wdlnet` sweep — the design
the sweep converged on — with an incremental accumulator:

```
768 psq --ft--> 256 -> crelu
256  --down[king]--> 16      each side, one shared table
[16|16] --mid[sym]--> 32     no activation
32   --up[mat]--> 32         each side -> crelu
[32|32] --l2[mat]--> 32      each side, same input -> crelu
[32|32] --head[mat]--> 3 logits (L, D, W)  + skip + psqt
```

Selected with `--wdl <path>` or `$CHESS_WDL`, and it wins over `--deep` and
`--quad`; only the winner's accumulator is built, so a loser is never pushed.
`export_wdl.py` folds `base + delta` and refuses any arm outside this topology
by name. `chess fendump N [seed]` is new: random legal play from the start
position, so a verifier's FEN corpus is reproducible instead of a file on one
machine.

**The search still reads one number.** `cp = K * logit(W + D/2)` with
`K = 288.5` (`fit_k`, the constant the scalar arms use), which is the readout
`nnue/wdl.py` derives and which deliberately throws away the draw-rate axis
LEDGER 048 measured. `K` is in the file, not compiled in. Reading `D` in the
search is the next question and is a separate change.

## What was measured

`chess fendump 1500 12345`, a **randomly initialised** `combo` — random rather
than trained on purpose, so an exporter bug that drops a zero-initialised table
(`skiph`, the psqt, every delta) cannot hide:

| | |
|---|---|
| feature mismatches | **0** / 1500 |
| bucket mismatches (all five indices) | **0** / 1500 |
| logits, engine vs torch | mean 3.8e-7, max **6.4e-7** |
| cp, engine vs torch | max **0.4995** — exactly the rounding bound |

`CHESS_ACC_CHECK=1 chess bench 4 --wdl`: 7,031,842 evals, **35 rebuilds
(0.0005%)**, and **20 mismatches, worst 1 cp**. The mismatches are f32 drift
between the incremental add/sub sequence and the from-scratch rebuild landing
on opposite sides of a rounding boundary — a wrong row or a missed update
cannot produce 20 errors in 7M that are all exactly one centipawn. `deepeval`
has the same property by construction.

`chess bench 9` with no net is **334110 nodes**, unchanged: nothing on the
existing paths moved.

No Elo, no nps and no strength claim: there are no trained weights yet, and the
random net's evals blow qsearch up, so every timing taken through it measures
the tree and not the evaluator.

## Negative: the 10B run produced no weights

`wdlarms.py` had no `torch.save` anywhere in it. The `combo-10b` run
(`--positions 10000000000`, started 13:41, ~5.4 h in at discovery, val
`ce_out` 0.7353 against `combo-1ep`'s 0.7413) was going to finish with a
scaling curve and nothing deployable. The script now writes
`ckpt-wdlnet/<arm>.pt` — cfg, state, seed, lam and `k_cp` — at **every val
point**, not once at the end, so an interrupted run still leaves weights.

Editing the file does not reach the running process. The run is reproducible
(`--seed` seeds numpy as of 2026-09-03), so the weights cost one more 6.8-hour
re-run and no thinking.
