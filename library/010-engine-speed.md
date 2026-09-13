# Engine speed: cores, and what a node costs

Depth material for LEDGER 039 and 040. `STATE.md` has the summary.

Machine throughout: AMD Ryzen 5 5500, Zen 3, **6 physical cores / 12 threads**,
32 KB L1D and 512 KB L2 per core, 16 MB shared L3, AVX2 and FMA but no AVX-512
and no VNNI. `target-cpu=native`. Nothing else running.

## 1. Lazy SMP

### What it is, and why the "dangerous" idea is the right one

The design is: **N threads search the whole tree from the root, over one shared
transposition table, with no other synchronisation.** No work decomposition, no
split points, no locks. Threads read each other's half-written table on
purpose. One thread's stored entry is another thread's instant cutoff, so the
tree each of them has left to walk keeps shrinking under it.

This is threads, not processes, and the difference is the whole mechanism.
Processes would have to `mmap` a shared region and hand-place the table in it;
threads share the address space for free, and the table is the *only* thing
that needs sharing. Everything else — killers, history, the PV, the repetition
key stack — is per-thread and must stay that way, which is what `ThreadData`
already was.

The races are load-bearing rather than tolerated. What has to hold is exactly
one thing: **a probe must never return a different position's entry.** The TT
gets that from Hyatt's XOR checksum, which this engine has had since the start
— each slot stores `key ^ data` beside `data`, so a torn write fails the check
and is discarded. A stale or slightly-wrong score for the *right* position is
already something the search copes with, because that is what an aged table is.

The alternative designs (Young Brothers Wait, Dynamic Tree Splitting) need a
work queue over subtrees, and a subtree's value depends on the alpha-beta
window it inherits, which changes while it sits in the queue. The shared state
is then far larger than a table and far harder to keep correct. Lazy SMP wins
because it needs nothing this engine did not already have.

### What it costs in determinism

Gone, and it cannot be bought back: two runs visit different nodes and can
return different moves. So **node-limited searches are forced to one thread**
(`Engine::search_threads`). `go nodes N` is how the tuner and every
reproducibility check ask for a fixed unit of work; under SMP that would be N
nodes on *some* thread with the others racing it, which is a different search
every run. Time-limited searches were already nondeterministic and lose
nothing. `bench` is single-threaded and its fingerprint is unaffected.

### Measured

`chess smp --depth 16 --reps 2`, 24 searches, 256 MB hash. Time-to-depth is
scored only on the searches where every arm returned at the requested depth —
an arm that proved a mate early did less work than its depth suggests, and
different arms prove it at different depths.

| threads | nps | nps × | time-to-depth × |
|---|---|---|---|
| 1 | 2.91 M | 1.00 | 1.00 |
| 2 | 5.98 M | 2.06 | 1.96 |
| 4 | 11.95 M | 4.11 | 1.98 |
| 6 | 16.03 M | 5.52 | 2.32 |
| 12 | 19.48 M | 6.70 | 2.20 |

Read the last column, not the middle one. **nps scaling is nearly free and
nearly meaningless**: N threads searching the same tree N times over would
report a perfect N-fold nps and be worth zero Elo. Time-to-depth is the one
that pays. 5.5x the nodes buys about **2.3x the effective speed** at 6 threads
— the rest is duplicated work, which is the known and accepted price of the
design.

12 threads is worse than 6. The second thread on a core shares one set of
execution units and one L1, and this search is not latency-bound enough to fill
the gaps. **Do not run more threads than physical cores.**

### Diversification: the depth-skip table loses

Whether helper threads should run a *different* schedule of iterative-deepening
depths, so the threads sit at different depths at any instant. `smp_skip=1` is
the classic Stockfish phase/size table; `smp_skip=0` gives every thread the
same schedule and lets the TT be the only thing that makes them differ.

At 6 threads, time-to-depth 16, n=3 runs each:

| | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|
| `smp_skip=0` | 2.51 | 2.86 | 2.74 | **2.70** |
| `smp_skip=1` | 2.55 | 2.63 | 1.97 | 2.38 |

No skipping also holds a higher nps (5.6x against 5.3x). Skipping makes the
helpers spend their time on budgets the main thread has not reached, which is
worth less than spending it on the one it is on. Same conclusion Stockfish
reached when it deleted its skip table. **Default is 0.** Three runs, and
time-to-depth is a proxy — no SPRT has been run on this.

### What has NOT been measured

**Elo.** Time-to-depth is a proxy and this project's rule about proxies applies
to it. The honest test is 6 threads against 1 at a fixed time control with
`--concurrency 1`, which costs the whole machine for the duration.

### A roadmap claim that is wrong

ROADMAP #2 said Lazy SMP is worth "6x the SPRT throughput — which is the real
bottleneck". **It is not.** `chess match` already runs games in parallel
(`--concurrency`, default cores/2 = 6 here) and the match runner explicitly
sends `setoption name Threads value 1` to every subprocess engine. The cores
are already saturated by games. Lazy SMP multiplies Elo per game, not games per
hour, and using both at once would divide the cores between them. Corrected in
ROADMAP.

## 2. What a node costs, with the deep net

Continues LEDGER 038, which left the deep net at 4.2x the quadratic form's nps
and named the remaining suspects. Uncontended numbers here; 038's were measured
with an SPRT running on other cores.

### Constant loop bounds: +27.6% nps, bit-identical

`width` and `hidden` are read from the net file, so every inner loop in
`deepeval.rs` was `for k in 0..w` with a trip count LLVM could not see. It
cannot keep the destination in registers across the outer loop, and it emits a
vector body behind a runtime length check that often loses to the scalar tail.

Fixed by monomorphising the inner loops on the dimension (`add_n<N>`,
`sub_n<N>`, `matvec_n<H>`) and dispatching on the runtime value over the shapes
the trainer emits, with the old dynamic loop as a fallback so an odd-width
experimental net still runs.

`chess bench 12 --deep nnue/nets/deep.bin`: **478,000 → 609,880 nps, +27.6%**,
with 2,041,983 nodes / 1,764,016 evals / 115 rebuilds — every one identical
before and after, which is the point. Same operations, same order, same f32
rounding. `bench 9` on the quad net (334,110) and PeSTO (bench 12, 1,927,076)
both unchanged; 33 tests pass; `perft-suite` ALL PASS.

### Where the remaining time goes

`CHESS_ABL` marginal costs, `bench 11`, ns per node. Total went 2105 → 1651.

| stage | before | after |
|---|---|---|
| extras gather | 423 | **189** |
| l1 | 728 | **708** |
| l2 | 108 | 90 |
| bucket-weight cache saves | — | 177 |

The gather more than halved. **l1 did not move at all** — and l1 is the biggest
single item.

### Why l1 did not move: it is footprint-bound, not arithmetic-bound

LEDGER 038 inferred this from a coincidence — l1's weights are 32.8 KB and Zen
3's L1D is 32 KB. Here is the measurement (`scratchpad/l1cache.rs`): the same
kernels, with a strided walk over a buffer between calls, standing in for what
a real node does to the caches between two evals (a TT probe into 256 MB, a
128 KB accumulator stack, the movegen tables).

ns per l1 call, net of the walk itself:

| intervening traffic | runtime dims | const dims | avx2 int8 |
|---|---|---|---|
| none | 927 | **243** | 77 |
| 64 KB | 910 | 420 | 94 |
| 512 KB | 1559 | **1486** | 1012 |

With nothing else touching the caches, constant bounds are worth 3.8x. At
64 KB of intervening traffic the same code costs 420 instead of 243 — it is
now paying to re-fetch its own weights. **At 512 KB the constant bounds are
worth nothing at all**, which is exactly what the engine measured. The
arithmetic was never the bottleneck; refilling 32.8 KB of f32 weights from L2
on every evaluation is.

So the gather sped up (its rows are small and it was genuinely scalar-bound)
and l1 did not (it was already waiting on memory).

**This settles what i8 is for.** Not the 10x arithmetic — the 4x footprint.
32.8 KB → 8.2 KB fits in L1D beside everything else the node touches, and the
int8 column above is the only one that stays ahead under pressure. The
arithmetic speed-up is a bonus that mostly will not be visible.

### Ranked, for whoever does this next

1. **i8 l1** — the largest remaining item and the one with a measured
   mechanism. `_mm256_maddubs_epi16` + `_mm256_madd_epi16`; Zen 3 has AVX2 but
   no VNNI, so no `dpbusd` shortcut. Needs the accumulator scale, l1's input
   range and the crelu bounds to agree; 038's warning stands, that when they do
   not the eval stays plausible and every game is played slightly wrong.
   `nnue/verify_deep.py` is the instrument that catches it.
2. **The extras gather, 189 ns.** Still ~30 random rows of 15 KB. Making
   `pawnfile`/`pawnpair` incremental was rejected in 038 for good reasons; a
   cheaper option is to fold them into the same i8 accumulator.
3. Everything else is now under 100 ns per node.
