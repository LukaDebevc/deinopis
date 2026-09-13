# 059 — The node profiler: eval and the accumulator are 85% of the search

**Setup.** `chess nodeprof`, new, behind `--features nodeprof`. It reads the
invariant TSC around every non-recursive leaf operation inside `negamax` and
`quiescence` and accumulates cycles per zone. No zone contains another, so
self-time is inclusive time and the parts add up; the remainder is the search's
own control flow and is printed as `control` rather than hidden.

Workload is deliberately `bench`'s: the same 12 FENs, depth 9, one thread.
Commit `93cada2` plus this change. Eval is `--wdl nnue/nets/gate-3ep-q8.nnue`,
which is what `tools/checkpoint.sh` gates with.

## The instrument is honest

Three checks, none of them optional:

| check | result |
|---|---|
| does it change the tree? | **396299 nodes, identical** to the plain build (334110 on quad, also identical) |
| does the overhead correction work? | plain `bench` search time **102.9 ms**, `nodeprof`'s corrected total **103.3 ms** — **0.4%** |
| does it agree with the other instrument? | `nodeprof` prices one eval at **950 ns**, `evalprof` at **938.5 ns** — **1.2%** |

The correction matters: two `rdtsc` reads cost 36 cycles (10.0 ns) and bracket
every call, which on the quad workload is **21.6%** of the measured total. That
is subtracted per call and also removed from the denominator, or the correction
would land in `control` and the search's own overhead would read 50% when it is
28%. Zones whose measured cost is at or below the probe pair print `<probe`
rather than a number — the instrument's floor is 10 ns and saying so is the
difference between a measurement and a claim.

The 46 ms `bench` spends outside `go` is one 64 MB TT clear per position. It is
setup, not search, so `bench` nps understates search speed by 44% on this
workload. Both numbers are now printed.

## Where the time goes (WDL net, the deploy config)

396299 nodes (151701 main, 244598 q), **504.3 ms of search, 1272.6 ns/node**.

| zone | site | calls/node | ns/call | share |
|---|---|---|---|---|
| eval | q | 0.98 | 948.5 | **44.9%** |
| eval | main | 0.69 | 950.1 | **19.6%** |
| acc push | main | 2.17 | 259.6 | **17.0%** |
| acc push | q | 0.23 | 291.9 | **3.2%** |
| movegen | main | 0.40 | 122.8 | 1.5% |
| make | main | 2.17 | 17.2 | 1.1% |
| movegen | q | 0.27 | 84.1 | 1.1% |
| order | main | 0.40 | 48.5 | 0.6% |
| price | main | 3.70 | 3.9 | 0.4% |
| everything else | | | | <1% |
| control | | | | 10.2% |

`draw`, `prefetch`, `acc pop`, `tt probe`, `tt store`, `history` and `observe`
are at or below the 10 ns probe floor.

**eval + accumulator push = 84.7% of the search.** Movegen, make/unmake, the
TT, move ordering and the price list together are **5.4%**. That answers the
question the roadmap left open — the search's own bookkeeping is not where the
time is, and no amount of work on movegen or make/unmake is worth anything.

## Two things nobody had priced

**1. The accumulator push is 20.2% of the search and appears nowhere in the
docs.** ROADMAP 4b noted that `bench` implied ~1137 ns/eval against
`evalprof`'s 949 and called the ~190 ns gap "outside the instrument". This is
that gap, and it is bigger than the estimate: 101.9 ms of 504.3 ms.

The mechanism is not the feature update, it is the copy. `wdleval.rs:1162`
does `next.copy_from_slice(...)` over `2 * width` **f32** — 4096 bytes per
push — before touching the ~4 changed rows, each another 2 KB read. About 12 KB
of traffic per made move, at ~46 GB/s, which is roughly L1/L2 bandwidth. The
code is not slow; it is moving four times more bytes than it needs to. An i16
accumulator halves every one of those numbers.

**2. Quiescence never reads the transposition table.** It is 44.9% of search
time and effectively all of it is static eval. Probed and discarded (so the
tree stays bit-identical):

- **1.2%** of q-nodes already have a stored static eval in the hash. The main
  search's stores do not cover quiescence — negative result, that idea is dead.
- **26.9%** of q-nodes repeat a key quiescence already visited this search.
  Upper bound: the witness is direct-mapped, so collisions inflate it.

So the reuse is there, but only if quiescence **stores**, which it does not.
That is a different experiment from the one the 1.2% killed, and it is worth
running.

## Also on record

- `head` (0.212 ns/MAC) and `psqt` (0.376) run 6-11x worse per MAC than `down`,
  `mid`, `up` and `l2` (0.033-0.046). By `evalprof`'s own rule — "a stage far
  off the others is a code-generation problem" — those two are unfinished.
  Together 140 ns of 938, so **15% of the eval, 10% of the search**.
- **This CPU (Ryzen 5 5500, Zen 3) has AVX2 + FMA but no VNNI and no AVX-512.**
  ROADMAP 4 and STATE both price a true int8 kernel at "`vpdpbusd` does 32
  MACs per instruction against our 8". `vpdpbusd` does not exist on this
  machine. The AVX2 path is `vpmaddubsw` + `vpmaddwd` + add, about 10.7 MACs
  per instruction, so the *arithmetic* argument for int8 is ~1.3x here, not 4x.
  The **memory** argument is untouched and is the real one: `down` reads 32.8 KB
  of f32 weights per evaluation and is bandwidth-bound, not issue-bound.
  Corrected at the source in ROADMAP.

## Decision

Node-level attribution is done and both instruments agree. The search's own
code is 5.4% and is closed as an optimisation target. The three ranked
candidates, all in the eval path, are in ROADMAP 4. No Elo claimed here —
this is a profile, and a profile is not a result.
