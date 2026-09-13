# losslab — searching for the router's loss function

An LLM proposes AUXILIARY LOSS TERMS for the learned bucket router; the harness
trains a real router with each one and ranks them by **Pareto dominance on
(validation loss, flip2)**. Sibling of `~/Desktop/LAB/new_optim`, which does the
same thing for optimizers, and it reuses that project's chain-prompting,
AST-gated sandbox and JSONL log.

## Why a search and not more hand-designed terms

LEDGER 046 built two game-structured penalties by hand (`ginfo`, `tv`) and found
the two ends of a trade-off: `tv` is nearly free but weak, `ginfo` is strong but
taxes val at every weight tried. What is missing is the middle, and the space of
functional forms is much larger than the space of things worth hand-writing.
A proposal costs ~2.5 minutes of GPU, so a day buys about 500 of them.

## The ranking, and why there is no score

    dominates(A, B)  <=>  A.val < B.val - eps_val  and  A.flip2 < B.flip2 - eps_flip
    rank(A)          =    how many other arms dominate A
    frontier         =    rank 0

There is no weighted sum anywhere. A scalar objective would fix an exchange rate
between eval accuracy and accumulator refreshes, and that rate is the thing this
is supposed to inform, not assume. The output is a set of live options, and
choosing among them is a separate decision that needs the refresh price.

`eps` is not decoration. Two runs of the same loss at different seeds came out
**3.9 flip2 points apart** on a 300-step screen. Without margins the frontier
fills with arms that are ahead by an RNG draw. `run.py` therefore measures the
noise floor FIRST, from replicates of the no-penalty baseline, and sizes the
margins as `2 * sd * sqrt(2/n_seeds)` — the standard error of the difference of
two seed-averaged arms, not the single-run spread.

## What a candidate looks like

One function, in its own file, loaded by `train.py --router-loss-file`:

```python
KNOBS = {"W": [1e-3, 3e-3]}        # optional: the harness sweeps the grid

def penalty(ctx):
    tp, pi = ctx["tp"], ctx["pair"]
    d = tp[pi + 2] - tp[pi]        # same side to move, two real plies apart
    return W * d.pow(2).sum(1).mean()
```

`ctx` carries the shuffled training batch (`p`, `z`) and a **game tape**
(`tp`, `tz`, `thard`, `gid`, `cnt`, `pair`) — 128 runs of 64 consecutive plies
of real games, in order. That is exactly what the hand-written `ginfo` and `tv`
terms see, so a candidate can reproduce either of them and the comparison is
like-for-like. `lab/prompts.py` documents it for the model.

## Severity — what stops a candidate cheating

- **flip2 is scored on games the penalty never trained on.** The tape is split
  by record range: the loss reads the first 80% of `tono_games.data`, the probe
  reads the last 20% (`--tape-train-hi`). Otherwise a candidate could drive
  flip2 down by memorising the games it was penalised on.
- **flip2 is measured on HARD buckets**, while every penalty acts on the soft
  posterior. The metric is not a restatement of the loss.
- **val is the deployed number**: hard-bucket validation loss, the same number
  every LEDGER entry quotes.
- **The degenerate corners cannot win.** Collapsing to one bucket gives
  flip2 = 0 with a ruined val; ignoring stability gives the best val at
  flip2 ≈ 41%. Both are dominated-by-nothing, so they sit on the frontier as
  anchors — which is correct, and is why `eff` (effective buckets used) is
  printed next to every arm: it is how you see a collapse for what it is.

## Staging

| stage | steps | seeds | cost | what it decides |
|---|---|---|---|---|
| noise | 1526 | 4 | ~4 min once | the margins every comparison below uses |
| baselines | 1526 | 2 | ~10 min once | the anchors a domination count is relative to |
| screen | 300 | 1 | ~15 s | drops only candidates clearly worse on BOTH |
| dev | 1526 | 2 | ~2 min | the numbers that rank it |

The screen gate is deliberately loose (`screen_slack = 3` noise widths). A
300-step ranking is a cheap proxy for the 1526-step one and **has not been shown
to preserve the order** — tightening it would trade compute for silently
discarded candidates.

## Running it

```bash
export GOOGLE_API_KEY=...
python run.py --baselines-only     # noise floor + hand-written terms, no LLM
python run.py --smoke              # `tv` through the candidate path, end to end
python run.py --hours 24           # search for a day
python run.py --resume --hours 12  # continue an existing log
```

Ctrl-C is safe: every candidate is written to `losslab_run.jsonl` as it
finishes, and `--resume` rebuilds the board from it.

## Layout

```
run.py                 CLI
lab/config.py          RunConfig — staging, seeds, paths
lab/evaluator.py       runs nnue/train.py as a subprocess, one arm per seed
lab/pareto.py          domination count, frontier, noise margins, the board
lab/sandbox.py         AST whitelist + KNOBS extraction
lab/prompts.py         system prompt: the setting, the two objectives, and
                       every term already tried with how it came out
lab/llm.py             Gemma/Gemini client, PLAN→CODE chain, model fallback
lab/search.py          controller: baselines → noise → propose/screen/dev loop
candidates/            every proposal's source, named as it appears on the board
results/               scratch for per-run JSON
```

## Hooks this needed in `train.py`

- `--router-loss-file FILE` — wires `penalty(ctx)` in as the aux term `custom`
- `--router-json PATH` — the metrics the report prints, machine-readable
- `--tape-train-hi F` — the train/score split of the game tape
- `GameTape(lo=, hi=)` — a record range, so the two halves are different games
