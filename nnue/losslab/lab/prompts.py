"""What the model is told. The system prompt carries the real prior knowledge:
what has already been tried on this exact problem and how it came out, so the
search does not spend the day rediscovering that a load-balance term is the
wrong instrument.
"""

_SYSTEM_TEMPLATE = r"""
You are designing AUXILIARY LOSS FUNCTIONS for a learned bucket router inside a
chess neural network evaluation. You write a few lines of PyTorch. Be concrete
and mechanistic; a re-skin of a term already on the list is a failure even if it
scores well.

## The setting

A chess position is a sparse binary feature vector (768 rows: side x piece type
x square, plus king-square features). A ROUTER reads it and picks one of 64
buckets:

    z = W . x + b            (6 logits; W is what you are shaping)
    bucket = the 6 sign bits of z, so b(x) in 0..63
    p(b|x) = the factorised sigmoid distribution over those 64 buckets

The bucket selects which weight table the network then uses. Today this
replaces a hand-written rule (`king6` = 6 bits of the king square, the best
hand rule on record). The goal is to replace the king-bucket scheme that
HalfKA-style nets use.

## Why this is hard -- the actual tension

The bucket must be SPECIFIC enough that per-bucket tables differ usefully, and
STABLE enough that consecutive positions in a game keep the same bucket. Those
pull opposite ways. HalfKA has the same contradiction: when its input is cheap
to update it does not discriminate, and when it discriminates it is expensive
to update.

The cost of a bucket change is concrete. The network keeps an incrementally
updated accumulator. A move that changes the bucket forces a full rebuild
instead of two adds. So the number that matters is

    flip2 = P( b(x_{t+2}) != b(x_t) )   over REAL games, SAME side to move.

flip2 is the real cost. Per-ply flipping (flip1) is NOT a cost here: the two
sides keep separate accumulators anyway, so a rule that flips every ply but is
constant for each side is fine. Do not optimise flip1.

## The two objectives -- and they are NOT combined

  val    the network's validation loss with HARD buckets. Lower is better.
         Scale: a no-penalty router is about 0.0258; `king6` is 0.0276.
         A 0.5% move is large. Run-to-run noise is about 0.02%.
  flip2  the bucket flip rate on HELD-OUT games. Lower is better.
         A no-penalty router is about 41%; `king6` is about 18%.

Candidates are ranked ONLY by how many other arms beat them on BOTH numbers at
once. Anything nothing else beats outright is kept. There is no exchange rate
between val and flip2 and you must not assume one -- a proposal that buys a
huge flip2 drop for a small val cost is interesting, and so is one that buys a
val gain at no flip2 cost. What is NOT interesting is a proposal that is worse
on both.

Ignoring stability gives the best val at flip2 41%; that corner is on the board
as `none` and cannot be won. The OTHER corner is not ranked at all:

  A collapsed router is REJECTED before it reaches the board. `eff` in the
  table is the effective number of buckets in use (exp of the usage entropy,
  64 max; a no-penalty router sits near 49). An arm below 4 is dropped,
  because flip2 = 0 is then true by construction -- there are no bucket
  changes left to count -- and nothing can beat a zero, so it would hold a
  frontier slot for ever. A zero there is not a result.

  Penalising sign AGREEMENT of the logits across plies is the usual way to
  fall in. sigmoid(-z_t * z_{t+2}), softplus(-z_t * z_{t+2}) and relatives are
  all minimised most cheaply by a router that simply stops routing, and two
  proposals have already died exactly this way. Before you propose a term,
  ask what a router that always picks bucket 0 would score on it. If the
  answer is zero, the term measures silence, not stability.

__BAND_SECTION__

## What has already been tried on this exact problem

  info=w    w * (E_x H(p(b|x)) - H(E_x p))  = -I(B;X). Sharpens the router.
  bal=w     w * (log 64 - H(E_x p)). Pushes bucket USAGE toward uniform.
            NEGATIVE RESULT: costs 2.1-2.9% val at every weight from 0.003 to
            0.03, and the mechanism is visible -- effective buckets used goes
            46.6 -> 63.5. Uniform usage is not structure. `king6` uses only
            20.6 effective buckets and is the best hand rule. Do not propose
            marginal-entropy-to-uniform again.
  margin=w  w * relu(m - |z|).mean(). Pushes logits away from the sign flip.
  swing=w   penalises the logit change over a synthetic one-piece-move graph
            (no legality, no captures, no game). A surrogate for flip2. At
            1e-4 it was worth -0.5% val; alone at higher weight it collapses
            the router to ~3 buckets.
  l1=w      w * |W|.mean() on the router weights.
  ginfo=w   w * (H(B|game) - H(B)) = -w * I(B;game), on real game tape.
            RESULT: flip2 41% -> 24% at w=1e-3 and 16.5% combined with tv, and
            it more than doubles king6's I(B;game) (3.45 vs 1.43 bits). It also
            destroyed the side-to-move board mirror as a side effect (flip1
            98% -> 12.5%). BUT it costs val monotonically: +1.5% / +2.1% /
            +5.2% at w = 3e-4 / 1e-3 / 3e-3. Nothing found a weight that
            escapes that. The open question is whether a DIFFERENT functional
            form gets the same stability cheaper.
  tv=w      w * ||p(b|x_{t+2}) - p(b|x_t)||^2 over real same-side move pairs.
            RESULT: at 3e-3 it is worth -0.67% val AND slightly lower flip2 --
            the only free win so far. At 1e-2, flip2 32.5% for -0.34% val.

So: penalising the SOFT posterior change (tv) is nearly free but weak.
Penalising conditional entropy over whole games (ginfo) is strong but taxes
val. The interesting region is between them, and probably not on the straight
line joining them.

## Your actual brief

Invent. The ONLY fixed things are the two objectives and the interface. You are
not restricted to variations of the terms above, to information-theoretic
quantities, to penalties on the posterior, or to anything else on this page.
Nobody knows what the right loss is -- that is why this is a search and not an
implementation task. A term built from an idea nobody here has had is the
outcome worth having; every proposal gets trained and measured, so a wrong
guess costs two minutes and is still informative. Propose the thing you would
actually want to test.

The list below is what the problem looks like to people who have stared at it,
offered so you do not re-derive it. It is NOT a menu to pick from, and copying
one of these is worth less than a bad idea that is genuinely yours.

- The soft posterior p is a poor proxy for a SIGN FLIP. What actually flips a
  bucket is one logit crossing zero. A term that prices the crossing directly
  (margin relative to the per-move logit movement, i.e. |z| against |dz|) is a
  different object from ||dp||^2 and might buy the same stability with less
  distortion of where the boundary sits.
- Per-BIT structure: the 6 bits are independent. A rule can be stable in 5 bits
  and volatile in 1, which costs a full flip. Terms that treat the bits
  separately, or that spend the instability budget unevenly across them, have
  not been tried at all.
- Stability is only expensive if it fights the eval. A term GATED on positions
  where the bucket choice barely matters (small logit magnitude, or small
  spread between per-bucket predictions) could be free.
- Asymmetric penalties: it may be fine to flip rarely and expensively rather
  than often and cheaply, since the cost is per rebuild. Penalising the
  EXPECTED NUMBER OF RUNS rather than the flip probability is a different
  objective with the same units.
- Anything that makes the boundary depend on slow-moving material rather than
  fast-moving squares.

__IDEA_BANK__

## Interface

Write exactly one top-level function:

    def penalty(ctx):
        ...
        return scalar   # a 0-dim torch tensor, ADDED to the loss

`ctx` is a dict:

  p      (B, 64)  soft bucket posterior on a shuffled TRAINING batch (B=65536)
  z      (B, 6)   router logits on that batch
  tp     (T, 64)  soft posterior on a GAME TAPE: T=8192 rows that are 128 runs
                  of 64 CONSECUTIVE plies of real games, in order
  tz     (T, 6)   logits on the tape
  thard  (T,)     hard bucket index on the tape (integer; not differentiable)
  gid    (T,)     which run/game each tape row belongs to (0..G-1)
  cnt    (G,)     number of rows in each run
  pair   (P,)     indices i into the tape where (i, i+2) is a valid SAME-SIDE
                  consecutive move pair. Use tp[pair+2] against tp[pair].
  nb     64       number of buckets
  mode   "bits"
  w      (768, 6) the masked router weight matrix
  gain   float    logit multiplier
  torch, F (=torch.nn.functional), math

Rules:
  - Return a scalar tensor with a gradient. Not a float, not a tensor of shape
    (n,).
  - The value is multiplied by an outer weight before it reaches the loss, but
    set your own scale so the natural weight is near 1. The main loss is about
    0.026, so a penalty whose raw value is ~1 needs an internal factor of
    roughly 1e-3 to matter without dominating.
  - Only torch/math/numpy. No file, network, or process access.
  - Index safety: `pair` is already filtered, `pair + 2` is in range. `cnt`
    sums to T. Do not assume G is fixed.
  - Cheap: this runs every one of 1526 training steps.

Optionally declare a parametric family with a literal grid, and the harness
sweeps it for you at no token cost:

    KNOBS = {"W": [1e-3, 3e-3], "TAU": [0.5, 2.0]}

Read those as module globals inside `penalty` (e.g. `return W * term`). Keep it
to at most ~4 combinations.

## Output format

    PLAN: <2-5 sentences. What quantity are you penalising, and WHY would it
    buy flip2 more cheaply than ginfo does? Name the mechanism.>

    ```python
    <the code>
    ```
"""


def code_prompt(plan):
    return f"""Implement this plan as the `penalty(ctx)` function, exactly to the
interface in the system prompt. Output ONLY a python code block. No prose, no
explanation, no test harness. Include a `KNOBS` dict only if the plan calls for
a swept parameter.

PLAN:
{plan}
"""


def explore_plan_prompt(report, ledger, nudge=""):
    return f"""Propose a NEW auxiliary loss for the router.

{nudge}

Current board (ranked by how many arms beat them on BOTH val and flip2 -- 0
means nothing beats it outright, so it is a live option):

{report}

What has been proposed so far in this run, and how it went:

{ledger}

Propose something MECHANISTICALLY different from everything above. State which
region of the (val, flip2) plane you are aiming at and why your term should
land there. Do not restate a term that is already on the board with a new name
or a new constant -- a swept constant is what KNOBS is for. Arms that land
above flip2 25% are not worth a run whatever they cost.

Output PLAN: then one python code block.
"""


def synthesis_plan_prompt(a, b):
    return f"""Two loss functions that are both on the Pareto frontier, reached
by DIFFERENT mechanisms. Combine what makes each work into one term -- not by
adding them (the harness can already add two terms), but by finding the shared
idea and expressing it once.

=== A: {a['name']}  (val {a['val']:.6f}, flip2 {100 * a['flip2']:.1f}%) ===
plan: {a.get('plan', '')}
```python
{a['code']}
```

=== B: {b['name']}  (val {b['val']:.6f}, flip2 {100 * b['flip2']:.1f}%) ===
plan: {b.get('plan', '')}
```python
{b['code']}
```

Say in PLAN what the shared mechanism is and why the combination should reach a
point NEITHER of them reaches, rather than landing between them.

Output PLAN: then one python code block.
"""


def refine_prompt(name, plan, code, detail):
    return f"""This loss is on the Pareto frontier but has a specific weakness.
Diagnose it, then fix it.

=== {name} ===
plan: {plan}
```python
{code}
```

Measured:
{detail}

Answer in this shape:

WEAKNESS: <one or two sentences -- which term in the code causes the number
that is bad, mechanically>
FIX: <what you change and why it addresses that term specifically>

PLAN: <the fixed design in 2-3 sentences>

```python
<the fixed code>
```

Change the mechanism, not just a constant. A constant sweep is what KNOBS does.
"""


def fix_prompt(plan, code, error):
    return f"""This candidate crashed. Repair it with the SMALLEST change that
keeps the intended mechanism.

PLAN: {plan}

```python
{code}
```

ERROR:
{error}

Common causes here: returning a non-scalar (reduce with .mean() or .sum());
indexing `pair` on a tensor of the wrong length (tp has T rows, pair indexes
into T); using `thard` in a way that needs a gradient (it has none -- use it
only for masks or counts); a shape mismatch between z (T, 6) and p (T, 64).

PLAN: <the same plan, one line>

```python
<the fixed code>
```
"""


# ---------------------------------------------------------------- the target

def _band_section(lo, hi):
    """The target region, written for the model.

    Kept as a function of the config band so there is ONE place that says where
    the search is aiming. The numbers quoted alongside it are measured arms, so
    the model can see that the band is below everything on the board rather
    than being told it is easy.
    """
    return f"""## The target region

**The goal is flip2 between {100*lo:.0f}% and {100*hi:.0f}%, at the lowest val
it can be had for.** Read that as a band, not a threshold. Above {100*hi:.0f}%
is not yet interesting; below {100*lo:.0f}% is very likely a collapsed or
near-collapsed router rather than a win, and buying flip2 below {100*lo:.0f}%
with more val is spending on an axis that has stopped paying.

Be clear about how far this is from anything measured. `none` is 41%. `king6`,
the best hand-written rule anyone has, is 18%. The best arm on the board is
`ginfo=3e-3,tv=3e-2` at 16.7% for +3.08% val, eff 51. **Nothing has ever been
measured inside the band.** A term that lands at 8% at eff > 40 for any val
cost under about +5% would be the best result this search has produced; one
that lands there for +1% would settle the question the search exists to answer.

Two consequences for what you propose:

- Arms at flip2 25-42% are no longer interesting however cheap they are. That
  end of the front is covered by `none`, `tv` and `ginfo` already. A proposal
  whose honest expectation is "about like tv but different" is not worth a run.
- The band is roughly a 5x reduction in flip rate from `none`. A gentle
  regulariser added to the existing objective will not get there. Expect to
  need a term that changes the STRUCTURE of what the bucket depends on -- what
  the router is allowed to be a function of -- not one that taxes an existing
  quantity a bit harder.

The eff floor is what stops this being trivial: flip2 = 0 is free if the router
stops routing, so an arm below eff 4 is rejected outright and arms below eff 30
are not built on. **Aim for the band at eff > 40.** Say in your PLAN what your
term scores on a router that always picks bucket 0 -- if the answer is "the
best possible score", the term measures silence and will be thrown away.

Read this before proposing another movement penalty. At MATCHED flip2 (~25%),
`ginfo` costs +2.49% val at eff 53, while the best "penalise logit movement"
candidate costs +3.21% val at eff 10. Every movement-suppression family
measured so far walks eff DOWN as it pushes flip2 down -- 37 -> 25 -> 17 across
a single weight sweep -- which means it is buying flip2 by quietly shrinking
the router: it costs val and it heads for the rejection gate. Both arms that
actually reach the low-flip2 region do so at eff ~51.

The reading, which is a hypothesis and not a settled fact, so attack it if you
can: down here flip2 is bought by making the bucket depend on SLOW, game-level
structure -- what -I(B;game) rewards -- and not by taxing per-ply logit motion.
"""


# Concrete starting points. NOT a menu to work through: the bank exists so the
# search does not spend its first hours re-deriving the obvious, and so that
# `idea_nudge` can push different parallel explorers down different branches
# instead of all six converging on the same term. Each one names a MECHANISM,
# not a formula -- the formula is the model's job.
IDEAS = [
    ("hysteresis / deadband on the crossing",
     "A bucket bit flips when a logit crosses zero. Price the crossing itself: "
     "penalise |z| being small ONLY on tape rows, so the router is pushed to "
     "keep every logit far from its own boundary along a real game, while "
     "leaving the shuffled-batch boundary free to sit wherever the eval wants."),
    ("per-bit budget",
     "The 6 bits are independent and a flip in any one costs a full rebuild. "
     "Measure the flip rate per bit and penalise the WORST bit, or spend the "
     "instability budget unevenly on purpose: 5 bits frozen hard to slow "
     "features, 1 bit left free to carry fast information."),
    ("slow-feature input shaping",
     "Penalise the router weight on rows of the 768-dim input that change "
     "often within a game (piece-square occupancy of light pieces) relative to "
     "rows that change rarely (material counts, king squares, pawn structure). "
     "This is a term on `w` and the tape statistics, not on the posterior, and "
     "it costs nothing at inference."),
    ("run-length, not flip rate",
     "The cost is per rebuild, so flipping rarely and expensively beats "
     "flipping often and cheaply. Penalise the expected NUMBER OF RUNS of a "
     "constant bucket along a game -- same units, different objective, and it "
     "does not reward a router that jitters symmetrically."),
    ("gate the penalty on where it is cheap",
     "Stability only costs val where the bucket choice actually matters. Gate "
     "any stability term on positions where the per-bucket predictions barely "
     "differ, or where the logit is already large, so the term stops charging "
     "for the positions the eval cares about."),
    ("predict the future bucket",
     "Make p(b|x_t) predict b(x_{t+2}) rather than merely agree with it: a "
     "cross-entropy from the current posterior to the DETACHED future hard "
     "bucket. Asymmetric where tv is symmetric, and it cannot be minimised by "
     "shrinking both sides toward each other."),
    ("commit to the game's modal bucket",
     "Compute the mode (or the mean posterior) of the bucket over each game "
     "run, detach it, and pull every row in that run toward it. This is "
     "ginfo's mechanism without the entropy: it rewards agreeing with a "
     "slow-moving target instead of penalising within-game spread."),
    ("temporal low-pass on the logits",
     "Penalise the high-frequency component of z along a run -- a second "
     "difference, or the residual after smoothing z over a window -- while "
     "leaving the low-frequency component completely free. A bucket that "
     "tracks a slow trend is cheap; one that tracks the trend plus jitter is "
     "not."),
    ("margin against measured movement",
     "The right scale for |z| is not a constant, it is how much z actually "
     "moves in one same-side move. Penalise relu(k * |dz| - |z|) with |dz| "
     "detached, so the required margin is set per position by the observed "
     "motion instead of by a hand-picked m."),
    ("distance to the hand rule, as structure not imitation",
     "king6 gets 18% flip2 from 6 bits of the king square. Do not copy it -- "
     "use the fact: penalise the router for being LESS predictable from slow "
     "state than king6 is, e.g. by requiring high mutual information between "
     "the bucket and a slow summary (material signature, king zone) while "
     "leaving which function of it entirely free."),
]


def _idea_bank():
    lines = ["## Ten starting points",
             "",
             "These are ideas nobody has run yet, offered so the search does "
             "not spend its first hours re-deriving the obvious. They are NOT "
             "a menu to work through in order, and an idea of your own that "
             "nobody here has had is worth more than a faithful implementation "
             "of one of these. Two of them COMBINED is often the interesting "
             "object -- a gate from one applied to the quantity of another -- "
             "and combinations are explicitly wanted.",
             ""]
    for i, (name, body) in enumerate(IDEAS, 1):
        lines.append(f"{i}. **{name}** -- {body}")
    lines += ["",
              "The functional form is yours. The same quantity behaves very "
              "differently under different shaping: a hinge charges nothing "
              "until a threshold, a square charges everywhere and mostly for "
              "the outliers, a log charges the first bit of instability most, "
              "a max over bits charges only the worst one, a detached weight "
              "turns a penalty into a reweighting. Say in the PLAN which "
              "shape you chose and what it charges for that a plain mean "
              "would not."]
    return "\n".join(lines)


def configure(flip2_lo, flip2_hi):
    """Rebuild SYSTEM_PROMPT for the configured target band.

    Called by the controller at startup, so config.py stays the single place
    that says where the search is aiming.
    """
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = (_SYSTEM_TEMPLATE
                     .replace("__BAND_SECTION__", _band_section(flip2_lo, flip2_hi))
                     .replace("__IDEA_BANK__", _idea_bank()))
    return SYSTEM_PROMPT


def idea_nudge(rng):
    """A per-call push toward a different corner of the idea bank.

    Six explorer threads share one system prompt, so without this they all read
    the same list and tend to return the same term with different constants.
    Each call gets its own draw: sometimes one idea, sometimes a cross of two,
    sometimes an explicit instruction to ignore the bank entirely -- that last
    branch is kept because the bank is a prior, not the answer.
    """
    r = rng.random()
    if r < 0.25:
        return ("For THIS proposal, ignore the ten starting points entirely "
                "and propose a mechanism that is on none of the lists above.")
    if r < 0.65:
        a, b = rng.sample(IDEAS, 2)
        return (f"For THIS proposal, build a term that combines these two "
                f"ideas into ONE quantity -- not a sum of two penalties:\n"
                f"  - {a[0]}: {a[1]}\n  - {b[0]}: {b[1]}\n"
                f"Say in the PLAN what the shared mechanism is.")
    a = rng.choice(IDEAS)
    return (f"For THIS proposal, start from this idea and take it further than "
            f"the sketch does -- a specific functional form, and a reason that "
            f"form and not another:\n  - {a[0]}: {a[1]}")


configure(0.02, 0.10)
