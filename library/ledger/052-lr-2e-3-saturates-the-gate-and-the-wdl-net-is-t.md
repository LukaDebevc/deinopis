# 052 — lr 2e-3 saturates the gate and freezes the ft table (the 128-Elo loss was the kernel, see 056)

**Date** 2026-09-05 · **Corpus** `all.data` = cluster `chess893.data` (identical
md5 on the frozen val slice), 893,500,985 records, val = last 40M strided by 40
→ 1M positions (`teacher alone on val: ce_out 0.64714 onll 0.56100`)
**Matches** 10+0.1, built-in 43-opening book, `--concurrency 5`, both arms the
same binary as subprocesses

## The engine can run a gated net

`src/wdleval.rs` only did `clamp(0,1)` on the accumulator, so no arm using the
`gate` fold could load. Added `FLAG_GATE`, file format 7 → 8, and the fold done
in place (`clamp(a,-1,1) * clamp(b,0,1)` into the front half; every read is at
or above the write, so no copy is needed). `export_wdl.py` writes the flag and
sizes `down` at `width/2`.

Verified: 0/2000 feature mismatches, 0/2000 bucket mismatches, logits match
torch to **2.4e-6** on three gate nets. The crelu path still verifies, and the
default-eval bench is **334110 nodes**, unchanged, so the change is neutral
where it should be.

## The deployed net lost 128 Elo, and it is speed, not accuracy

> **CORRECTED 2026-09-06 (LEDGER 056).** The nps column below was measured on a
> kernel carrying a bounds check in every matvec inner loop. Removing it took
> the WDL net 315,814 → 700,250 nps (LEDGER 055), and the same design then beat
> the quadratic eval by **+84.6 Elo [+61, +109]**. The −127.7 below is a fact
> about that kernel, not about the net. The section's *reasoning* stands and is
> what made the fix findable: the loss was speed, not accuracy.

`gatew1024-1ep` had the best val loss of any net trained (0.72752) and lost
**−127.7 Elo [−156, −101], 324 games** to the quadratic eval.

| eval | read/eval | nps | 
|---|---|---|
| quadratic | — | 2,231,428 |
| `gate` (512→256, b16) | 69.5 KB | 292,726 |
| `gatew1024b8` | 55.5 KB | 217,600 |
| `gatew1024` (1024→512, b16) | 97.5 KB | 183,358 |

**12.2x slower per node.** A better eval that searches a twelfth of the tree is
not a better engine. `gatew1024` also *doubled* the 32.8 KB L1 read that
LEDGER 038/040 already named as the binding cost.

Head to head, `gate` beat `gatew1024` by **+69.0 Elo [+40, +99], 260 games** —
33% cheaper *and* stronger, for 0.0026 *worse* val loss. **Val loss ranked these
two backwards.** i8 would take `gate` to 20.4 KB (3.4x); halving the accumulator
to 2×128 buys only 26%, because `l2`, `mid` and `up` are set by `hidden`, not
by width.

## Why the 20B run failed: not overfitting, not repetition

The 20B `gatew1024` run ended at **0.77118**, worse than the same arm's
one-pass **0.72752**. Three explanations were checked and two are wrong.

**Not repetition.** `combo-10b` ran 11.7 passes over *the same 853.5M pool*
(`all.data` and `chess893.data` are the same file) at lr 1e-3 and descended
monotonically to 0.73467 with no degradation at all.

**Not overfitting.** `train − val` in the 20B run stays between +0.001 and
+0.008 throughout. At its worst point (6.75B) **train 0.79013 is worse than val
0.78440** — a memorising net has train far below val. It got worse at fitting
the data it was training on.

**It is the learning rate, via clamp saturation.** Measured on 1500 real
positions:

| net | epochs | ft \|w\| mean | \|z\| value/gate | both clamped |
|---|---|---|---|---|
| `gate-1ep` | 1 | 0.144 | 0.69 / 0.77 | 16.4% |
| `gate-3ep` | 3 | 0.192 | 0.98 / 0.99 | 25.4% |
| `gatew1024-1ep` | 1 | 0.155 | 0.78 / 0.97 | 22.2% |
| `gatew1024-20b` | 23 | **0.874** | **4.73 / 6.09** | **81.0%** |

`gate-1ep` vs `gate-3ep` is a clean single-variable comparison — same arch,
lr, data, only length differs — and gives a monotone dose-response. At 23
epochs **81% of gate units sit outside both clamps, where the gradient is
exactly zero**, which is why *training* loss degraded. Saturation is
self-reinforcing: past the clamp a weight gets no gradient, so it is frozen at
its large value and cannot shrink back. `AdamW(weight_decay=0.0)` left nothing
to pull it in.

**The LR was chosen on runs 80x too short.** 2e-3 won a 250M-position sweep and
was then applied to a 20B run. The optimum for a short schedule is always
higher. The one-pass runs escape it only because their cosine is compressed
into 52k steps.

**Clipping is the wrong fix; decay is the right one.** The inflation is in the
BULK (mean 0.144 → 0.874, and √32 × 0.874 ≈ 4.9 ≈ the observed 4.73), not the
tail. Stockfish's clip of 127/64 = 1.98 touches **0.009%** of a healthy net and
would not have stopped this. Weight decay shrinks the bulk proportionally.
The sparsity objection does not apply: at batch 16,384 every ft row that occurs
gets gradient every step (rarest resolvable row ≈ 8 activations/batch, none
below 1), and the decay timescale at lr 6.7e-4 / wd 1e-2 is ~150k steps.

## The validation set was a landmine

`loader.py` built val as the **tail of `--data`**, and `chess-data/fetch.sh`
**appends**. Adding a month would have silently replaced the held-out set and
broken comparability with all 51 previous entries, with no error. `freeze_val.py`
now pins it to its own file; `--pool-end` is mandatory alongside `--val-file`
because the freed tail would otherwise re-enter the pool and the run would
train on its own validation set. Verified transparent: a 20M-position `gate`
run scores **0.76712 either way, to every digit**.

## Decisions

- `gate` (512→256, bneck 16) is the shape, not `gatew1024`. Val loss is not a
  ranking for deployment while the engine is memory-bound.
- Retrain at **lr 6.7e-4 with wd 1e-2**, 10B, snapshots every 1B (the 20B run's
  best point, step 61,032, no longer exists on disk — `save()` overwrites).
- **i8 the `down` read before spending more GPU time on val loss.** It is worth
  3.4x; every architecture choice on the table is worth tens of percent.
- All 893M positions are ONE month at stride 1. LEDGER 021 said to buy distinct
  games instead and it was never done; `linrock/test79` has Mar and Apr unused.

---
