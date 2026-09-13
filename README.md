# chess

A chess engine in Rust, written from scratch with no dependencies, with a
neural evaluation trained by its own PyTorch pipeline. It rates **about 3246
on the CCRL Blitz scale** (±33, 800 games at 10+0.1 against four CCRL-rated
engines; [ledger 088](library/ledger/088-cp-0004-gauntlet-3246-ccrl-blitz.md)).

> **Work in progress.** This is an active research project. The engine is
> stable and strong; the research question below is still open.

**The question:** can the hand-tuned pruning and reduction rules in an
alpha-beta search be replaced by ones that are learned and measured? Instead of
a depth counter with a reduction table, every node here holds a *budget*, and
each child pays a price that depends on how unpromising the move looks. The
price is a short list of named, tunable terms, so a learned model can replace
them one at a time ([ENGINE.md](ENGINE.md), [library/003](library/003-budget-search.md)).

This follows on from my bachelor's thesis on machine learning in chess engines
(SPSA tuning, NNUE evaluation).

## Quick start

```sh
cargo build --release                    # needs a CPU with AVX2 for full speed
./target/release/chess --wdl nets/m1-b1.nnue            # UCI on stdin/stdout
./target/release/chess serve --wdl nets/m1-b1.nnue      # web board at http://127.0.0.1:8080
./target/release/chess bench --wdl nets/m1-b1.nnue      # prints 199091 nodes
./target/release/chess help                             # every other subcommand
cargo test --release
```

In a GUI (Cute Chess, Arena, Banksia) add the engine with the argument
`--wdl <path>/nets/m1-b1.nnue`. **Without it the engine silently falls back to
a hand-written piece-square eval and is several hundred Elo weaker**;
`chess bench` tells you which one ran (199091 nodes = the net, 233103 = the
fallback). UCI options: `Hash`, `Threads`, `DrawValue`, `Contempt*`.

`chess serve` is also how I watch a search: it streams depth, score, nodes and
the principal variation live, with the PV drawn as arrows on the board.

## Strength over time

Every rating below comes from the same gauntlet (`tools/ladder.sh`): 200 games
per opponent at 10+0.1, opponents built from source at the exact version CCRL
rated. Treat them as "about X on the CCRL Blitz scale": CCRL's own conditions
(hardware, book, time control) add a systematic offset on top of the interval.

| version | date | what changed | rating | entry |
|---|---|---|---|---|
| baseline | 2026-08-24 | alpha-beta, hand-written eval | ~2550 ±60 | [004](library/ledger/004-baseline-strength-anchored.md) |
| — | 2026-08-27 | learned quadratic eval | ~2971 ±43 | [025](library/ledger/025-absolute-rating-after-the-quadratic-eval-about-2.md) |
| — | 2026-09-07 | win/draw/loss neural eval (a build between `cp-0001` and `cp-0002`) | ~3049 ±46 | [071](library/ledger/071-cp-0002-gauntlet-about-3176-ccrl-blitz.md) |
| `cp-0002` | 2026-09-08 | i16 accumulator, SEE term in the price | ~3176 ±42 | [071](library/ledger/071-cp-0002-gauntlet-about-3176-ccrl-blitz.md) |
| `cp-0003` | 2026-09-09 | move ordering split by SEE, better net | ~3199 ±56 | [076](library/ledger/076-cp-0003-gauntlet-about-3199-ccrl-blitz.md) |
| `cp-0004` | 2026-09-12 | quiescence shares the hash table | **~3246 ±33** | [088](library/ledger/088-cp-0004-gauntlet-3246-ccrl-blitz.md) |

Consecutive gauntlet ratings often overlap. The claim that each version is
better than the last rests on a head-to-head SPRT, listed in
[CHECKPOINTS.md](CHECKPOINTS.md).

## What is in it

- **Move generation.** Magic bitboards, legal move generation, a copy-make
  board. Exact on a published perft suite (`chess perft-suite`).
- **Search.** Iterative deepening with aspiration windows, principal variation
  search, a lockless transposition table, null move, reverse futility,
  quiescence, lazy SMP. Reductions and late-move pruning are replaced by the
  budget-and-price scheme above.
- **Evaluation.** An NNUE-style network with a win/draw/loss head: a 768→512
  feature transformer kept incrementally as an i16 accumulator, a king-bucketed
  bottleneck, material-bucketed layers above it. Trained in `nnue/` on
  Stockfish-scored positions, with a Leela network as the teacher for the
  current net.
- **Measurement.** A match runner with SPRT and pentanomial statistics, a
  gauntlet against rated engines, a deterministic work meter (a stand-in for
  wall-clock time that doesn't change with machine load), and a tuner that
  scores search parameters by regret against a deep search instead of games.

## How it is measured

Nothing is kept because it looks better. `tools/checkpoint.sh` is the only way
a version becomes the new best: it runs the tests, the full perft suite and
the bench, then plays an SPRT against the previous checkpoint's binary, both
sides as subprocesses. It tags and commits only if the test accepts. The four
`cp-*` tags are the versions that passed.

Every experiment gets an entry in the [ledger](LEDGER.md): setup, result,
decision. Most of them are negative, and those are recorded with the same
care. A few that show how it works:

- [064](library/ledger/064-the-first-learned-search-feature-to-pass-an-sprt.md): the first search feature to pass an SPRT, +51 Elo from SEE in the price.
- [051](library/ledger/051-the-architecture-sweep-was-measuring-the-optimis.md): an architecture sweep that turned out to be measuring the optimiser.
- [068](library/ledger/068-eqnodes-and-eqtime-disagree-eqnodes-retired.md): equal-nodes and equal-time matches disagree, so equal-nodes was retired.
- [024](library/ledger/024-a-gauntlet-anchor-voided-by-our-own-uci-parser.md): a gauntlet result voided by a bug in our own UCI parser.
- [085](library/ledger/085-corr-phase-1-killed-zero.md): correction history, a standard technique, measured at exactly 0.0 over 586 games here and dropped.

## How it was built

With AI coding agents. [CLAUDE.md](CLAUDE.md) is the rule set they work
under: perft before anything, a bench after every change, no version kept
without the gate, every result in the ledger. I choose the questions, design
the experiments and decide what is kept. Session handover notes, the working
backlog and raw run logs stay in a private working copy, so some ledger
entries cite scripts and logs that are not in this repository.

## Layout

| path | what |
|---|---|
| `src/` | the engine |
| `tests/` | search and harness correctness |
| `nets/` | the net the current checkpoint plays with |
| `ENGINE.md` | what the engine does, with the actual formulas |
| `ARCHITECTURE.md` | how the code is laid out and why |
| `CHECKPOINTS.md` | every version that passed the gate |
| `LEDGER.md` | one line per experiment; full entries in `library/ledger/` |
| `library/` | longer write-ups |
| `tools/` | the gate, the gauntlet, match helpers |
| `nnue/` | the eval trainer (Python, PyTorch; its own README) |

## License

MIT, see [LICENSE](LICENSE).
