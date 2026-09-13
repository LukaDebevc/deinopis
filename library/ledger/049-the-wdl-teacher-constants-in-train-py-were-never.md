## 049 — The WDL teacher constants in `train.py` were never the refit, and one sweep was voided by it

`nnue/wdlarms.py`, `nnue/wdlnet.py`, 2026-09-04. Corpus is
`chess-data/all.data` (893,500,985 records, Stockfish `test79-2022-04`
binpacks via `nnue/extract`). Fit on 2M positions at 30% of the file, held out
on 2M at 70%, stride 40 on both.

**The `--wdl-a` / `--wdl-b` defaults were as bad as Stockfish's shipped
constants, which LEDGER 048 had already ruled out.**

| teacher | test 3-way NLL | predicted D | actual D |
|---|---|---|---|
| the defaults that were in `train.py` | **1.20245** | 0.687 | 0.495 |
| SF shipped constants (048, other slice) | 1.19934 | — | — |
| refit here, same 8-parameter cubic form | **0.64859** | 0.495 | 0.497 |
| refit in 048, never wired in | 0.65232 | — | — |

The base-rate entropy of the outcome on this corpus is **1.04095 nats**
(L/D/W = 0.2567 / 0.4980 / 0.2453, n = 1M). The old defaults therefore scored
*worse than predicting the base rate*, while claiming a 68.7% draw rate against
an actual 49.5%. The number is flat across the file — 1.1995 / 1.2037 / 1.1955
/ 1.1993 at 2% / 35% / 68% / 95% — so it is the constants, not a slice.

The failure is silent by construction: 048 measured the refit and reported it,
but the value that went into the CLI default was not it. Nothing in any run
scores the teacher, so a target worse than the base rate produced no warning.

**What it cost.** An 18-arm architecture sweep at 250M positions each was run
against that teacher at `lam = 0.9`, i.e. 90% of every training target was a
distribution with the draw rate wrong by +19 points. All 18 arms finished
between 0.937 and 0.945 outcome CE. With the refit teacher the same `base` arm
reaches **0.902 after 300 steps (5M positions)** — better than any of the 18
after 250M. The sweep is void and was re-run; its logs are kept under
`nnue/runs/wdlnet-stale-teacher/` rather than deleted, because the *rankings*
in it are a free second sample of the same comparisons under a different
(bad) target.

**Fixes.**
- `a = [219.5003, -471.8433, 164.5011, 319.7684]`,
  `b = [472.3219, -1141.005, 998.184, -121.0777]` are now the defaults in both
  `train.py` and `nnue/wdlarms.py`.
- `wdlarms.py` prints **the teacher's own ce_out and onll on the val set before
  any arm runs**. A target nobody scores is a target nobody can trust.
- `loader.Batcher` gained `val_stride` (default 1, no change for existing
  callers). The old validation tail was 1M positions from ~20k games; for a
  label that is a property of the *game*, that is ~20k independent samples, not
  1M. At `val_stride=40` the same 1M positions come from ~800k games and it is
  still a clean game-level split.

**Note the disagreement that found it.** The bad teacher's *expected score* was
fine — onll 0.57931, better than any trained arm — while its three-way CE was
worse than the base rate. A pipeline that only ever looks at E cannot see this
failure at all, which is the same point 048 makes about the second axis.
