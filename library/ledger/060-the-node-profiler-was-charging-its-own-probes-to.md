# 060 — The node profiler was charging its own probes to `control`

**Negative result about an instrument.** `chess nodeprof` as committed at
`b986f5b` — the build LEDGER 059 was written from — reports a search that is
37% slower than the same code without it, and puts all of that in `control`,
the residual bucket that means "the search's own flow control".

## Setup

Same workload as 059: `bench`'s 12 FENs at depth 9, one thread,
`--wdl nnue/nets/gate-3ep-q8.nnue`, `taskset -c 5`. Three repeats per arm.

## The disagreement

| build | search time | ns/node | `control` |
|---|---|---|---|
| `nodeprof` as committed | 688–697 ms | 1737–1758 | **33.5%** |
| plain `bench`, same commit | ~503 ms | ~1270 | — |
| `nodeprof` with the probes gated off | 479–493 ms | 1210–1245 | **10.3–10.6%** |

Ruled out first, in this order: run-to-run noise (3 repeats, 1.2% spread); an
engine regression (plain `bench` reproduces 059's 504.3 ms to 0.3%, so the
search did not get slower); and a broken correction (the *attributed* time
matched between arms almost exactly, 453.7 vs 452.9 ms — only the denominator
was wrong).

## Cause

Two diagnostic probes in quiescence sat outside every `zone!` block: a
`tt.probe` per q-node, and a write into a 32 MB direct-mapped witness table.
Both were on the `nodeprof` feature. Being outside every zone, their cost was
not attributed to any of them, so it fell into the residual — and the residual
is `control`.

They are the probes that produced 059's own "1.2% of q-nodes have a stored
static eval / 26.9% repeat a key" numbers, which are unaffected: those are
counts, and the tree is bit-identical either way.

## Fix

Split into a `qprobe` feature, which implies `nodeprof`. The 32 MB allocation
is gated on it too, so an ordinary profiling build no longer evicts the eval
weights to hold a table it never reads. `nodeprof` alone now reproduces 059:
control 10.3–10.6% against 059's 10.2%, total 479–493 ms against 504.3 ms.

## Decision

**059's table stands; the instrument that produced it had stopped
reproducing it.** Had a price table been frozen from the broken build, every
zone's share would have been scaled by 0.66 and `control` would have read as a
third of the search — pointing optimisation work at the one area 059 correctly
closed as a target (the search's own code is 5.4%).

Also on record: the profiler's cost has to be *inside* the taxonomy or
*outside* the measurement. A probe that is neither is indistinguishable from
the thing being measured, and it lands in whichever bucket is defined as "what
is left over".
