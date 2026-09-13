# nnue — the eval training stack

Trains the evaluation networks and exports them into the Rust engine. Python +
PyTorch; nothing here is compiled into `chess`, only the exported net is.

Data lives **outside the repo** at `data/all.data`
(binpack, stm-canonical). `--data` is required and has no default.

**Use `nnue/.venv/bin/python`, not `python3`.** The system interpreter's torch
has no CUDA, so every run silently falls back to "Torch not compiled with CUDA
enabled". `.venv` is a symlink to a local CUDA venv
(torch 2.7.1+cu126, RTX 4060 Ti) and is gitignored.

**The validation set is frozen.** `freeze_val.py` writes the held-out records
to a file of their own, because `Batcher` otherwise takes them from the tail of
`--data` -- so appending a month to the pool would silently replace the val set
and quietly break every number ever recorded. Pass both flags together; the
tool prints them:

```
--val-file data/val_frozen.data --pool-end 853500985
```

`--pool-end` is not optional: `--val-file` frees the tail back into the pool,
so without it a run trains on its own validation set. Verified transparent --
a 20M-position `gate` run scores 0.76712 either way, to every digit.

```
python3 train.py --data data/all.data \
                 --experiment psqtbuckets --steps 3000 --batch 16384
python3 test_features.py && python3 test_round2.py     # after any change here
```

## The live path

`train.py` is the driver: it owns the training loop, `build_arms` (which
experiment builds which models) and `main`. Everything else it needs is a
module beside it.

| module | what |
|---|---|
| `loader.py` | binpack reader, GPU unpacking, `NFEAT`/`PAD` |
| `common.py` | `loss_fn`, `outcome_nll`, `fit_k`, `bag` |
| `features.py` | input features, `side_split`, `flip_feat` |
| `buckets.py` | bucket rules and the `FAMILIES` registry |
| `rulestats.py` | statistics over bucket rules (used by `buckets.py`) |
| `movegraph.py` | move-adjacency graph over squares |
| `models.py` | every model arm — the big one |
| `router.py` | learned bucket router, `GameTape`, the stability losses |
| `wdl.py` | the WDL basis, `b_sym`, and the refit Stockfish teacher (LEDGER 048/049) |
| `wdlnet.py` | the bucketed trunk with a three-way head — the `combo` design |
| `wdlarms.py` | its sweep driver; writes `ckpt-wdlnet/<arm>.pt` at every val point |
| `fen.py` | FEN parsing for the tests |
| `quant.py` | weight quantisation: prices any bit width / granularity on the frozen val set, and bakes an RTN checkpoint (LEDGER 053) |
| `qat.py` | fine-tunes through a straight-through fake quantiser, against the real target or as KL to the fp32 net |
| `test_features.py`, `test_round2.py` | the only tests; both print `ALL PASS` |

Deployment into the engine: `export.py` / `export_deep.py` / `export_wdl.py`
write the `.nnue` file, `verify.py` / `verify_deep.py` / `verify_wdl.py` check
the exported net against the trainer on real positions. The WDL pair is the
live one:

```
python3 export_wdl.py --ckpt ckpt-wdlnet/combo-10b.pt --out nets/combo-10b.nnue
../target/release/chess fendump 2000 > /tmp/fens.txt
CHESS_WDL=nets/combo-10b.nnue CHESS_CKPT=ckpt-wdlnet/combo-10b.pt \
    python3 verify_wdl.py /tmp/fens.txt          # prints ALL PASS
```

`chess fendump N [seed]` is new and exists because the old verifiers were run
against a hand-kept FEN file that is not in the repo, so their numbers could
not be re-run. `treeprice.py` prices an accumulator refresh against the
moves the search actually makes (LEDGER 034) — that is the cost side of any
"is this feature worth it" question.

## Everything else

| directory | what |
|---|---|
| `losslab/` | **active**: an LLM proposes router loss terms, ranked by Pareto dominance on (val, flip2). Has its own README. |
| `runs/` | logs. Top level = study in flight, `old/` = already written up. Has its own README. |
| `propose/` | the LLM proposal client shared with `losslab`. |
| `extract/` | Rust binpack extractor (its own crate; `cargo build --release` inside it). |
| `attic/` | code from finished studies. Not maintained. Has its own README. |
| `ckpt*/`, `nets/` | checkpoints and exported nets. **Gitignored** — 817 MB, all regenerable. |

## Rules

- **Run both tests after any change to the live path.** They are cheap and they
  are the only thing standing between a refactor and a silently wrong net.
- **A number without its config is a rumour.** Quote steps, batch, `--val-cap`
  and the seed. Run-to-run val spread is ≤0.018%; nothing under ~0.05% is a
  ranking (LEDGER 044).
- **`--seed` seeds numpy too.** It did not until 2026-09-03, which made every
  `GameTape` arm unreproducible; see `library/ledger/047`.
- **When a study ends, move its scripts to `attic/` in the same change as its
  ledger entry.** See `attic/README.md`.
- No file over ~600 lines. `train.py` was 4,561 and is now eight modules.
  `build_arms` currently dispatches 43 experiments and is the next thing over
  the line.
