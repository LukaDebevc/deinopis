## 055 — A bounds check in the matvec inner loop was costing 2.2x

_2026-09-06. `src/wdleval.rs`, `src/deepeval.rs`, `chess evalprof`._

**Setup.** Build the profiler ROADMAP 4b asked for, then follow it. `chess
evalprof` prices each stage of an evaluation as the *difference of two
prefixes* of `logits_upto`, a const-generic `STOP` bound so a prefix compiles to
exactly the work in it. The corpus is 512 positions sampled every 97th eval
from a real depth-8 search on the bench FENs — tree positions, not a random
walk. Single thread, Ryzen 5 5500 (Zen 3), `target-cpu=native`.

**The instrument checks out.** Profiler total 3059 ns/eval against 3100 ns
derived from `bench` (WDL 3.145 µs/node minus default-eval 0.466, over 0.864
evals/node): **1.3% apart, with no clock constant assumed anywhere.**

**What it found.** `down` was 57% of the evaluation at 0.213 ns/MAC while
`l2` — the same kind of operation — ran at 0.082. Two controls said it was not
memory:

| | down, ns |
|---|---|
| i8 weights (4 KB/bucket) | 1748 |
| f32 weights, same values (16 KB/bucket) | 1738 |
| corpus of 4 positions (one or two buckets, L1-resident) | 1748 |
| corpus of 512 positions | 1744 |

4x the bytes for the same time, and bucket variety worth nothing. The
disassembly said why:

```
vmulps  %ymm0,%ymm1,%ymm0
vaddps  (%rbx),%ymm0,%ymm0      <- load z[k]
vmovups %ymm0,(%rbx)            <- store it back, every iteration
```

`&w[j * DO..j * DO + DO]` is a **conditional panic inside the loop**. LLVM has
to leave the destination coherent at every possible exit, so the accumulator
could not live in a register and every multiply-add paid a store-to-load
forward. Monomorphising on the inner trip count (LEDGER 040) was necessary and
not sufficient; `chunks_exact` does the one check up front and the loop body
then has no exit.

**The changes, each gated on `bench` node count.**

| | WDL nps | ns/eval |
|---|---|---|
| start | 315,814 | 3059 |
| `chunks_exact` instead of an indexed row slice | 527,689 | 1556 |
| `skip=false` on `mid`/`up` — no activation precedes them, so the zeros it looks for cannot exist | 586,319 | 1325 |
| `skip=false` on `l2`/`head`/`skiph` too | 602,126 | 1210 |
| same fix on the gate activation's `ft_bias` reads | 608,389 | 1185 |
| FMA + four partial accumulators | **700,250** | **949** |

Node count **396,299 throughout**, so nothing above changed which positions get
searched. `perft-suite` ALL PASS, 33 tests pass, default-eval bench 334,110
unchanged, `verify_wdl.py` ALL PASS on 1500 positions for both the i8 and f32
files (max 0.5 cp, which is the engine rounding to whole centipawns).

**FMA alone made `down` 28% *slower*** — 401 → 513 ns — which is how the
dependency chain got identified rather than guessed. FMA is 4-cycle latency
against `vaddps`'s 3, and `down` is 256 inputs into 16 outputs, so two ymm
registers carry 256 serial adds each: predicted 4/3 = 1.33, measured 1.28. Four
partial accumulators cut the chain to 64 and FMA became a win. Eight was worse
than four (302 vs 272 ns on `down`) — 16 accumulator registers is the whole
file.

An `[[f32; DO]; U]` indexed by the unroll counter is **7x slower** than one
accumulator: LLVM will not promote a stack array indexed by a loop variable.
Four separately-named accumulators is the entire trick.

**`deepeval` had the identical bug**: `574,076 → 813,140 nps (+41.6%)`, 395,106
nodes both sides, and `evalfen` **bit-identical on 1500 positions**.

**Decision.** Keep all of it. The WDL net is now **3.14x** slower per node than
the quadratic eval (2,197,420 vs 698,672 nps), not 7.0x. LEDGER 040's closing
claim — that the arithmetic was never the bottleneck and i8 is motivated by
footprint — is **wrong and is corrected there**: its 512 KB-of-intervening-
traffic column is not what a real tree does (LEDGER 054), and fixing the
arithmetic bought 41.6% in the engine it said would show nothing.

The controlled i8-vs-f32 pair (same weights, same 396,299-node tree) is now
661,767 vs 687,961 nps, **+4.0%** — up from +3.1% and still small. i8 remains
free accuracy (LEDGER 053) and marginal speed.

**Left on the table.** `down` is 0.033 ns/MAC ≈ 0.14 cycles/MAC, against ~0.06
for two 256-bit FMAs per cycle, so roughly 2x remains in the f32 kernel and
more in a true int8 one (`vpdpbusd` is 32 MACs per instruction against our 8).
And `bench` now implies ~1137 ns/eval against the profiler's 949: the ~190 ns
gap is the incremental accumulator update and the eval dispatch, neither of
which `evalprof` measures yet.
