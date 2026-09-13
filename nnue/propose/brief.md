You are proposing **routing rules** for a chess evaluation network.

## What a rule is

A rule is a total function from a position to a small integer bucket. The
network computes one shared 512-wide activation per position and then reads it
with a per-bucket linear map, so the bucket decides *which read* is used, not
what is accumulated. A good rule splits positions into groups whose evaluation
depends on the position in genuinely different ways.

## What makes a rule good

Three numbers decide it, and they trade off:

- **power** — how much held-out eval loss a per-bucket read removes, in percent
  of the base residual. Measured, not argued. Baselines are in the table below.
- **cost** — bitboard ops per recompute, charged automatically by the API.
- **balance** — effective bucket count (perplexity of the visit distribution).
  A rule declaring 576 buckets that only ever fills 66 has 66 buckets.

Rules are also cheaper if they read few piece types: a rule that reads only
pawns and kings cannot be disturbed by a knight move.

## What is already known, so do not re-propose it

- King-square routing **saturates**: every rule based on where the kings are
  tops out near 5.8% no matter how fine, from 11 regions to all 4096 pairs.
  Proposing a cleverer king partition is the single most common wasted round.
- Material routing is the strongest known axis, worth about 13%.
- A designed 256-bucket coarsening fit **worse** than the one-line hand rule
  "own pieces x their pieces, capped at 24 each". Bucket count is not the
  constraint. More buckets is not an idea.
- Balance alone predicts power at Spearman 0.88 — a well-spread rule is usually
  a decent rule, but that is a filter, not a discovery.

The interesting question is therefore: **what axis is neither material nor king
square?** Pawn structure, piece coordination, space, mobility proxies, blocked
positions, opposite-coloured bishops, king-pawn races. Rules that mix a coarse
material term with a structural one are welcome; rules that are material or
king geometry warmed over are not.

## The contract

Write exactly one function. No imports, no other statements.

```python
def rule(b):
    """One line saying what distinction this draws."""
    ...
    return b.combine([(term, size), ...])
```

`b` is the only thing you may touch. Every board it returns is an (N, 64)
boolean; every value is one integer per position. You never see a loop over
positions — everything is batched, so write it as whole-board operations.

Declared size must be at most 4096 and cost at most 200 ops.
