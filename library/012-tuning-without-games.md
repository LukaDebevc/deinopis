# 012 — Tuning without games

How to use the position-level tuning loop: what each command does, what the
numbers mean, and what this instrument can and cannot decide.

`src/tune.rs` explains *why* the objective is shaped this way and the three
ways it can lie. This file is the operating manual.

## The loop in one screen

```bash
# 0. once per corpus: pull FENs out of a binpack (137 s for the 5.6 GB file)
nnue/extract/target/release/nnue-extract \
    --input data/fishpack32.binpack \
    --output /dev/null --mark --game-stride 401 \
    --fen-dump data/tune/fishpack-gs401.fens

# 1. once per teacher: label the positions expensively        (~9 min for 1000)
./target/release/chess tune label \
    --fens data/tune/fishpack-gs401.fens \
    --n 1000 --seed 0 --ref-nodes 400000 --top 5 \
    --out data/tune/fishpack-1k.labels \
    --wdl nnue/nets/gate-3ep-q8.nnue

# 2. as often as you like: score a candidate                        (3 s / 1000)
./target/release/chess tune eval    --labels <f> --nodes 15000 --wdl <net>
./target/release/chess tune compare --labels <f> --a c_rank=250 --b c_rank=375 ...
./target/release/chess tune search  --labels <f> --free c_rank,c_hist --iters 15
```

Step 2 is the whole point: **a full 1000-position pass takes 3.0 seconds**, so
a parameter sweep that would be a week of SPRTs is an afternoon.

## Step 0 — the corpus

`--game-stride 401` keeps one game in 401 and every ply of it. On
`fishpack32.binpack` that read 2.56 billion positions from 21.4 million games
and wrote **6,379,672 FENs** (341 MB). `--mark` writes positions that fail
Stockfish's training filter as well as those that pass; we want in-check and
mid-exchange positions available, because the sampler applies its own filter.

Use a binpack, not our own PGNs. `positions_from_pgn` still exists and still
samples the distribution *this engine already reaches*, which is one of the
three biases in the `tune.rs` header. The binpack fixes that one. It does not
fix the other two.

## Step 1 — labels

The teacher searches each of the top `--top` root moves at `--ref-nodes` nodes
and records what each is worth. That is the reference opinion the cheap search
is scored against.

**The sample is nested in `--n`.** Positions are ranked by `hash(seed, fen)` and
the lowest `n` kept, an ordering that does not depend on `n`. So the 1000-
position corpus is a strict subset of the 14000-position one at the same seed,
and an existing `--out` file is used as a **cache**: raising `--n` relabels only
the positions that are new. Changing `--seed` draws a fresh corpus.

**The cache key is the entire teacher.** The header line

```
# teacher 400000 nodes, top 5, engine b986f5b bin:9fa50e6def37dc09, net nnue/nets/gate-3ep-q8.nnue
```

is compared verbatim, and any difference relabels everything. `bin:` is a hash
of the actual executable, because the commit is not enough — `-dirty` gives
every uncommitted tree the same name, which is useless during exactly the phase
when the search is being changed — and because the binary also covers the
feature flags and the compiler. `--reuse` overrides the check when you know the
diff cannot reach the search.

That strictness is what makes the promotion workflow safe:

> Label from the current best engine at high compute. Tune against those
> labels. When a candidate wins its SPRT and becomes the new best, relabel from
> it and carry on.

A stronger engine's opinion of a position **is a different opinion at the same
node count**, so mixing its labels with the old ones would make the objective
depend on which positions happened to be labelled before the promotion. The
cache key makes that mistake impossible rather than merely discouraged.

Two things the teacher is deliberately *not*:

- **Not budgeted in work.** A label set is reused for months; a node count is
  the same search forever, while a work budget shifts the moment anyone
  re-freezes the price table in `work.rs`.
- **Not relaxed.** `label_one` uses `Params::default()`, so the teacher is the
  student's own price list at more nodes and shares its blind spots. This is a
  known hole — STATE's "Next" item 6 — and the promotion workflow only fixes
  half of it (a better engine, still the same pricing).

## Step 2 — scoring

### The budget, four ways to say it

| flag | meaning |
|---|---|
| `--nodes N` | N nodes per position. What every number on record used. |
| `--work N` | N work units per position. Needs `--features work`. |
| `--total-nodes N` | N for the whole pass, divided by the corpus size. |
| `--total-work N` | likewise. |

A **work unit** is one main-search node at the deploy configuration
(`src/work.rs`). The distinction from nodes is not pedantry: a quiescence node
costs **0.275** of a main-search node (`library/007`), so at a fixed *node*
budget a price list can be credited for buying cheap nodes. `compare` prints
`nodes a -> b` so the shift is visible either way.

Measured, n=1000 at 15000, `c_rank` 250 → 1000: **+2.49 ± 2.06 cp at fixed
nodes against +2.23 ± 2.09 at fixed work.** The two budgets agree to well
inside the noise, so **fixed work is a correctness fix that currently changes
no answer.** Keep using it — it costs nothing and it is immune to machine load
— but do not claim it bought anything.

The `--total-*` forms make a pass cost the same wall time whatever `--n` is,
which is the knob that trades coverage against resolution. It has a floor and
the floor is measured: below ~15000 per position the tool prints the LEDGER 014
warning, because contrast-to-noise falls from 4.6 at 15k to 1.9 at 1k and at 1k
the known-bad over-pruning arm scores *better* than the default. Buying
coverage below that floor buys more coverage of a metric with the wrong sign.

### What the columns mean

```
1000 positions @ 15000 nodes: regret 41.46 +/- 2.02 cp   depth 7.03
                              nodes 10551   agree 46.5%   off-list 11.7%   (3.0s)
```

- **regret** — the objective. Mean centipawns the teacher thinks the cheap
  search's move gave up, capped per position at `--cap` (default 200).
- **depth** — a **constraint, not a second objective**. Never sum it into
  regret: pricing harder buys depth *and* loses Elo, so any linear combination
  rewards the known-bad arm. `tune search` enforces it as a floor
  (`--depth-floor`, default 0.25 plies).
- **nodes** — constant under a node budget; under a work budget it is the free
  readout of where the candidate moved the money.
- **off-list** — the cheap search picked a move the teacher never scored. Those
  positions get charged the worst label, which is a floor, so a large off-list
  means regret is understated.

### Parameter overrides

`--set`, `--a`, `--b` and `--free` accept **all 20 registered names**, not just
the 12 price-list ones — `params!` generates a `set` that sees through to the
nested struct. `chess params` lists them.

`tune search`'s default free list is still the price list alone, on purpose:
the finite-difference step is `max(|v|/8, 25)` milli-plies, which is right for
a price and absurd for `nmp_min_depth`, whose entire range is 1..12. Name such
a parameter in `--free` deliberately if you want it, and read the trajectory
rather than trusting the step.

## What 1000 positions can and cannot decide

Measured on this corpus, at 15000, paired: **standard error 2.06 cp**, so the
smallest difference resolvable at |t| = 2 is about **4.1 cp**.

| effect | positions needed |
|---|---|
| 4.1 cp | 1000 |
| 2.5 cp — `c_rank` 250→1000, the known-bad arm | 2700 |
| 1.37 cp — deleting the *entire* price list (LEDGER 010) | 9000 |
| 1.0 cp | 16900 |

**1000 is a deliberate choice, not a limitation to work around.** At LEDGER
007's exchange rate (1 cp ≈ 25–30 Elo) a 4.1 cp floor is a ~100 Elo floor, and
this project is looking for structural wins of that order rather than +1 Elo
patches. A change that does not resolve at 1k is a small change, and small
changes are not what we are shopping for yet. When that stops being true, the
corpus is nested — lift `--n` and only the new positions cost anything.

One caveat on reading `compare`: `tune.rs` argues that pairing removes most of
the variance because two price lists usually choose the same move. On a **large**
perturbation that is not true — `c_rank` 250→1000 moved 436 of 1000 positions
and pairing bought only **1.4×** over independent means. The argument holds for
the small steps a descent takes; it does not hold for a big A/B.

## What this decides, and what it does not

It **proposes**. `tools/checkpoint.sh` decides: tests, perft, bench, then an
SPRT against the previous checkpoint's binary. A proxy metric winning is not a
result, and regret is a proxy with a measured ceiling of ~1.7 cp (LEDGER
009/010) and a documented blind spot in the under-pruning direction.

The accept rule that follows from that: **regret may reject a candidate but
never promote one on its own.** Read it together with the depth floor and, when
the budget is work, the node column. Promotion is an SPRT.

## Files

| | |
|---|---|
| `src/tune.rs` | corpus, labels, objective, descent — and why it can lie |
| `src/work.rs` | the deterministic cost model and the frozen price table |
| `nnue/extract/` | the binpack reader, `--fen-dump` |
| `data/tune/` | corpora and label files (outside the repo) |
