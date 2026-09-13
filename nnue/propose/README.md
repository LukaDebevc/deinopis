# Handing the rule search to Gemini

The eval study has an expensive question with a cheap scorer: **which routing
rule should sit in front of the accumulator?** `power.py` scores a rule without
training it and `refresh.py` prices it per legal move, so the marginal cost of
one more candidate is an index computation, not a training run. That is the
only reason this is worth automating — the search is bottlenecked on *ideas*,
not on machine time.

So: the model writes rules, this harness measures them, and nothing is believed
until it appears in the table next to the twenty hand-written baselines that
were scored in the same pass on the same positions.

## Where the key goes

    echo 'AIza...' > nnue/propose/.key      # gitignored, along with everything below

or `export GEMINI_API_KEY=...`. Get one at https://aistudio.google.com/apikey.

## Running it

    python3 run.py models                   # which ids this key sees, and quota left
    python3 run.py probe                    # once: cache 8192 positions for validation
    python3 run.py loop --rounds 40 --model gemma-4-31b-it --score-every 8
    python3 run.py report

`propose --dry-run` prints the prompt without spending a request.

## What the model can and cannot do

It emits one function and nothing else:

```python
def rule(b):
    """One line saying what distinction this draws."""
    return b.combine([(term, size), ...])
```

`b` is a batch of positions viewed as bitboards. **The generated code cannot
touch the filesystem, the network, or any object other than `b`** — the source
is parsed before it runs and rejected if it contains an import, a dunder, an
attribute on anything but `b`, or a call to any builtin outside a whitelist of
eight pure ones. There is no container because there is nothing to contain: the
rule is a pure tensor expression. The only process writing files is `run.py`,
and it only appends to the three files in this directory.

Every primitive on `b` is one or two bitboard instructions in the engine, so
"cheap to compute" is structural rather than checked afterwards. `b.cost` adds
them up and a rule over 200 ops is refused. Sliding-piece attacks are absent on
purpose: they need magic lookups and they change on nearly every move.

## The gates, in order

| gate | rejects |
|---|---|
| AST walk | imports, dunders, foreign attributes, stray statements |
| execution | wrong return shape, out-of-range index, non-determinism |
| cost | more than 200 bitboard ops per recompute |
| balance | fewer than 2 effective buckets, or one bucket holding >90% |
| partition hash | a rule that splits the probe positions exactly like one already in the pool |
| `power.py` | nothing — it just measures, and the number goes in the table |

The partition hash is the one that earns its keep. Models restate their own
ideas constantly; canonicalising the induced partition catches a restatement
however differently the arithmetic is written.

## The feedback channel

Each round's prompt carries the current leaderboard, the names already tried,
and last round's rejections **with their reasons**. That is the whole of the
"agentic" part, and it is deliberately simple: best-N in the prompt, no island
population, no MAP-Elites. If the naive version plateaus, the population
bookkeeping is the thing to upgrade — the scorer does not change.

`brief.md` is the standing situation brief. It carries what LEDGER already
knows, which is mostly a list of what *not* to propose: king routing saturates
near 5.8%, bucket count was never the constraint, the designed coarsening lost
to a one-line hand rule. Keep it current — a stale brief spends quota on
questions already answered.

## Quotas

Three limits per model, and they differ by an order of magnitude. The strong
text models allow tens of requests a day; the Gemma models allow ~14k but cap
tokens per minute, which bounds the prompt rather than the count. The prompt is
~1.3k tokens, so Gemma sustains roughly ten rounds a minute. `client.MODELS`
holds the declared numbers — check them against `run.py models` before a long
run, since a wrong model id fails on the first call.

## What this cannot tell you

`power.py` is a lower bound measured on a frozen accumulator, and it ranks
rather than levels. A rule winning here has earned a *training run*, not a
place in the engine. The chain to Elo is unchanged and still long.
