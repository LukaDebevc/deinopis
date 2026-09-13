## 7. Can bucket tables be merged? (scratchpad `merge.py`, `merge2.py`)

Post-hoc on trained `ckpt-buckets/kings_x4096_read_pre.pt`: cluster the learned
readers, replace each with its centroid, re-measure. Bucket gain for reference:
0.021095 -> 0.019934, so buckets are worth 0.001161.

| clusters | Euclidean (gain kept) | data-weighted (gain kept) |
|---|---|---|
| 1,024 | 0.020022 (92%) | 0.020032 (92%) |
| 256 | 0.020277 (70%) | 0.020309 (68%) |
| 64 | 0.020898 (17%) | 0.020759 (29%) |
| 16 | 0.021792 (worse than no buckets) | 0.022005 (worse) |

The data-weighted metric (`d^2 = (R_i - R_j)^T M (R_i - R_j)`, `M = E[a a^T]`)
barely changes the answer, because the **activation gram's condition number is
only 4.2e3** — the accumulator excites its directions fairly evenly, so
Euclidean was already close to right. This is the opposite of what the rank
analysis in section 2 found, and worth remembering: check, do not assume.

**The conclusion above is superseded. Luka objected that post-hoc merging asks
the wrong question, and he was right.** Merging asks whether the learned
readers of a *trained fine* model are interchangeable. It does not ask whether
the fine granularity was needed, because a coarse model trained from scratch
puts a different function in each of its fewer buckets. `kingcoarse` is the
properly posed version — every coarsening trained from scratch, one run, same
seed and same block size, with an in-run material anchor
(`runs/kingcoarse.log`, 8000 steps, r=512, block 2M):

| arm | buckets | params | val | vs base |
|---|---|---|---|---|
| base r=512 x1 | 1 | 395,010 | 0.021084 | — |
| kings 4^2 (quadrants) | 16 | 410,882 | 0.020292 | **-3.76%** |
| **kings 11^2 (Luka's regions)** | 121 | 518,402 | **0.019901** | **-5.61%** |
| kings 16^2 (2x2 tiles) | 256 | 656,642 | 0.019851 | -5.85% |
| kings 64^2 (HalfKP) | 4,096 | 4,588,802 | 0.019863 | -5.79% |
| material 24^2 (anchor) | 576 | 984,322 | 0.018266 | -13.36% |

Two results.

**King routing saturates at 121-256 buckets.** 4,096 buckets are no better than
256 — 0.019863 vs 0.019851, a 0.06% difference against a ~0.1% noise floor
(LEDGER 023) — for 7x the parameters. Luka's 11 regions reach 96% of the
saturated gain with 518k parameters against 4.59M. The boring explanation
(4,096 is undertrained, ~145k rows per visited bucket) does not fit: the train
losses are also equal, 0.019645 vs 0.019637. An undertrained arm would be worse
on both, an overfitting arm better on train and worse on val. It is neither, so
the extra resolution is simply not being used.

**And merging understated coarse buckets badly.** 16 merged clusters came out
*worse than no buckets*; 16 trained buckets get 64% of the saturated gain. Do
not use post-hoc merging to decide granularity — it measures reader
interchangeability in a model that never had to economise.

One thing this does not settle: each bucket only picks a read vector over a
*shared* 512-dim accumulator, so the saturation may be V-bound rather than
bucket-bound. That is the same open question as section 5 and it is what the
accumulator-conditioning arm is for.

The in-run material anchor at -13.36% also validates the cross-run comparison
used elsewhere in this file (-13.40% from `runs/buckets.log`): **material
routing beats any king routing by 2.3x**, and the gap is not a run artefact.
