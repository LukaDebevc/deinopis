# Working on this engine

## Read first
`STATE.md` (what's happening), then `ARCHITECTURE.md` (why the code is shaped
this way). If the work is in the trainer, `nnue/README.md` too. Depth lives in
`library/`, reached by pointer — never read a whole study to answer a question
about one part of it.

Each top-layer file has one job and a size budget. Past the budget it stops
being cheap to open, which is the whole point of it:

| file | job | budget |
|---|---|---|
| `STATE.md` | handover: where we are, what is running, what is next | ~150 lines |
| `ARCHITECTURE.md` | why the code is shaped this way; changes rarely | ~200 |
| `ROADMAP.md` | the ordered backlog | ~100 |
| `LEDGER.md` | **index only** — one line per experiment, full entry in `library/ledger/NNN-slug.md` | 1 line/entry |
| `CHECKPOINTS.md` | every version that passed the gate, with its tag | 1 line/version |
| `nnue/README.md` | what the trainer is, what is live in it, how to run it | ~80 |
| `library/` | the write-ups. No budget; this is where depth goes. | — |

A finding goes in the ledger. A synthesis across findings goes in `library/`.
`STATE.md` gets a **pointer** to it, never the prose — that is how it reached
237 lines and grew a section labelled stale by its own text.

## The point of the project
Not another Stockfish clone — we cannot win the SPRT-throughput race on 6 cores
and should not try. The bet is **learned search control**: replacing hand-tuned
reduction and pruning heuristics with trained ones. A structural win of +50 Elo
is resolvable in ~2000 games; the +1 Elo patch grind is not open to us. Design
accordingly: prefer changes that could be large over changes that are safe.

## Rules that keep this honest
- **Perft before anything.** Any change to movegen or `make_move` must leave
  `cargo test --release` and `chess perft-suite` exact. No exceptions.
- **`chess bench` after every change.** If the node count moved on a change
  that was meant to be behaviour-neutral, it wasn't. Investigate before moving on.
- **A proxy metric winning is not a result.** Node count, eval accuracy and
  synthetic-tree regret are all proxies. Elo in real games is the thing.
- **`tools/checkpoint.sh` decides what is kept.** Nothing gets committed as a
  new best version without passing it: tests, perft suite, bench, then an SPRT
  against the previous checkpoint's binary, both sides as subprocesses. It
  pushes on H1 and writes nothing otherwise. Do not hand-tag a version around
  it — a version nobody measured is a version nobody can trust later.
- **Match numbers come with their conditions.** Elo without the time control,
  the game count and the interval is not a result, it is a rumour. Absolute
  ratings additionally carry a systematic offset from the CCRL conditions; say
  "about X on the CCRL Blitz scale", never "we are X".
- **Record negative results** with the same care as positive ones: the full
  entry goes in `library/ledger/NNN-slug.md`, one new line in `LEDGER.md`.
- **Correct conclusions at the source.** If something in LEDGER or ARCHITECTURE
  turns out wrong, edit it. Do not append a contradicting entry and leave both.
- Keep the top-layer docs short. Push depth into `library/`. A file past about
  600 lines is a token tax on every session that opens it — split it into an
  index plus parts, as `LEDGER.md` and `library/009-eval-architecture.md` are.
  `nnue/train.py` was 4,561 lines for the same reason and is now eight modules.

## Rules that keep the repo navigable
- **Source is committed; artefacts are ignored.** `nnue/` is 819 MB, of which
  2 MB is source — checkpoints, nets, logs and `.npz` are all regenerable and
  all in `.gitignore`. A clean `git status` therefore means nothing is
  forgotten, and that is the only reason it is worth keeping clean.
- **Finished studies move to the attic, in the same change as their ledger
  entry.** `nnue/attic/` holds code that a ledger entry cites and nobody
  maintains; `nnue/runs/old/` holds the matching logs. Nothing stays at the top
  level unless it is on the training path or the deploy path. Skipping this is
  how `nnue/` reached 51 files that could not be ranked by relevance.
- **The attic stays runnable.** Every archived script carries a four-line path
  bootstrap so its imports still resolve. Archived is not broken; a measurement
  nobody can re-run is a measurement nobody can check.
- **A new directory ships with a README.** One screen: what it is, what is live
  in it, what is finished. `nnue/`, `nnue/attic/`, `nnue/runs/`, `nnue/losslab/`
  and `library/004-tools/` each have one.

## Conventions
- No dependencies. Not dogma — the web GUI and the TT would each pull in a
  hundred crates for things that are 100 lines of `std`. Revisit for the NNUE
  trainer, which is a separate concern.
- `unsafe` is allowed in the hot path (`get_unchecked` on tables with
  statically-known bounds) and must carry a comment naming the invariant.
- New search constants go in `search::Params`, never inline. That struct is the
  interface the learned controller will replace.
- **New *information* goes in the `features!` block**, never as an inline `if`
  in the move loop. One line gives it a coefficient, a box, a registry entry,
  lazy evaluation and a price term; a dormant feature costs less than this box
  can measure, so the bar for declaring one is low and the bar for switching it
  on is an SPRT. Declare its **group** (`Sigma` / `Gap` / `Cost`) — that is the
  claim being made about it, and it fixes the sign before anything is measured.
  `library/013`.
