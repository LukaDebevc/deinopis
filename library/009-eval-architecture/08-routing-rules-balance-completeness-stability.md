## 8. Routing rules: balance, completeness, stability (`nnue/rulestats.py`,
`nnue/attic/refresh.py`)

The three criteria a routing rule must satisfy, all measured. Refresh rate uses
the engine's own movegen via a new `chess moves` subcommand: 20,000 positions,
**580,957 legal moves** (29.0 per position). The frame is anchored to the
player who was to move and does **not** swap with the move — an accumulator
belongs to a player, not to a ply.

### The legal-move mix is not the made-move mix

Rating a rule by its rate over *legal* moves is wrong, and wrong by a large
factor. Counters in `make_move` (`ThreadData::movemix`, printed by
`chess bench`; six increments, node count unchanged at 334,110) give the mix
the tree actually plays at d9 over 324,010 made moves:

| | legal moves | made moves |
|---|---|---|
| quiet | 94.4% | 69.0% |
| capture | 5.4% | **30.2%** |
| promotion | 0.2% | 1.1% |

**Captures are 5.6x over-represented among made moves** — they are searched
first and quiescence searches almost nothing else. Reweighting the per-class
rates by that mix is the `in search` column below, and it reverses the ranking:
material-reading rules get ~6x more expensive, king-reading rules ~25% cheaper.

> **Correction, later the same session: every number in this table is HALF
> the price.** `refresh.py` applies one move in the *mover's* perspective and
> asks whether the mover's bucket changed. The records are stm-canonical, so
> the opponent's pieces never move in that view. A real engine keeps two
> accumulators, one per perspective, and both must stay current — so the cost
> of a rule is the number of accumulators a ply dirties, counted in *both*
> views. For a side-symmetric rule (material, phase, counts, both kings) the
> per-ply price is exactly **2x** the column below, so the ordering here
> survives. For a rule that reads only one side's half it is worthless: such a
> rule measures as free here and costs full price in the other accumulator.
> See "The price was half" below. Measured per ply: HalfKP **16.20%**,
> material 24^2 **8.00%** (n = 60k positions).

| rule set | declared | perp | naive /move | **in search (per mover acc)** | quiet | capture |
|---|---|---|---|---|---|---|
| phase x3 | 3 | 3 | 0.14% | **0.80%** | 0.00% | 2.07% |
| phase x4 | 4 | 4 | 0.20% | **1.18%** | 0.00% | 2.58% |
| pieces (rook x queen)^2 | 36 | 8 | 0.29% | **1.74%** | 0.00% | 4.05% |
| phase x5 | 5 | 5 | 0.41% | **2.39%** | 0.00% | 6.02% |
| **castle state 4^2** | 16 | 4 | 3.32% | **2.46%** | 3.51% | 0.13% |
| **king region 11^2** (Luka's) | 121 | 17 | 3.48% | **2.63%** | 3.66% | 0.33% |
| centre state x3 (open/mobile/locked) | 3 | 3 | 0.69% | **3.07%** | 0.16% | 9.52% |
| phase x8 | 8 | 7 | 0.58% | **3.37%** | 0.00% | 9.10% |
| **material 24^2** | 576 | 62 | 0.68% | **3.94%** | 0.00% | 10.53% |
| phase x imbalance | 40 | 10 | 0.79% | 4.56% | 0.00% | 12.33% |
| centre pawns 3^2 | 9 | 5 | 1.86% | 5.77% | 1.00% | 16.81% |
| **kings 64x64 (HalfKP)** | 4,096 | 207 | 10.98% | **8.22%** | 11.59% | 0.76% |
| count x8 | 8 | 8 | 1.70% | 9.44% | 0.00% | 31.26% |
| shelter (castle x cover)^2 | 256 | 36 | 11.64% | 13.73% | 11.19% | 19.38% |
| wing majority x9 | 9 | 7 | 2.52% | 14.11% | 0.00% | 44.33% |
| **pawn struct (centre x wing)** | 27 | 20 | 3.14% | **16.84%** | 0.16% | 52.73% |
| shelter x material | 147,456 | 874 | 12.30% | 17.52% | 11.19% | 29.82% |

Still an underestimate for material rules: the tree's captures are MVV-LVA
ordered and SEE-filtered, so they take valuable pieces more often than a
uniform capture does, and a bigger capture crosses a bin edge more often.
Underestimate for king rules too, for a different reason: castling rights and
en passant are not stored in the record, so castling — the largest king
relocation there is — is never generated.

### Rules cross additively

Measured, not assumed. `phase x5` 2.39% + `castle 4^2` 2.46% = 4.85%, and the
crossed rule measures **4.85%**. Same for three-way crosses. So a rule budget
can be spent component by component.

### What a 2% in-search budget buys

**One rule.** Phase at 3-5 levels (0.80-2.39%) or the king's castle state
(2.46%), not both. Luka's guess of "3 or 5 game states through the game" is
exactly the range the budget allows.

**Pawn structure is the worst thing on the board to route on** if it is read as
counts or majorities: 16.84%, because any pawn capture anywhere flips a wing
majority. The only near-affordable pawn read is centre state — locked / open /
mobile, read from the d and e files only — at 3.07%.

### Where the loss lives, and where routing helps (`nnue/attic/phaseloss.py`)

Luka's guess was "most buckets around midgame". Per-position loss on the 1M
validation set, binned by phase, each bucketed model's mean loss in the bin
relative to the unbucketed base:

| phase | share | base loss | material x576 | count x8 | kings x4096 |
|---|---|---|---|---|---|
| 0 (least material) | 11.1% | 0.026134 | **-25.93%** | -11.44% | -4.99% |
| 1 | 24.0% | 0.018686 | -14.52% | -11.42% | -5.36% |
| 2 | 10.1% | 0.022865 | -15.11% | -8.69% | -3.71% |
| 3 | 11.7% | 0.022680 | -11.27% | -4.28% | -4.82% |
| 4 | 6.2% | 0.027186 | -12.09% | -5.71% | -6.66% |
| 5 | 9.0% | 0.024715 | -9.48% | -4.89% | -6.00% |
| 6 | 9.1% | 0.025249 | -8.40% | -6.25% | **-7.06%** |
| 7 (full material) | 18.8% | 0.013533 | -6.66% | -6.53% | -6.04% |
| all | 100% | 0.021095 | -13.40% | -7.93% | -5.50% |

**Material routing's gain is monotone decreasing in material**, 4x stronger at
phase 0 than phase 7; half of it comes from the 35% of positions with the least
material. King routing is flat to slightly *rising* with material. Binned by
raw piece count instead, at 2-5 pieces material routing is **-46.5%** and king
routing is **+3.2%** — HalfKP-style king routing actively hurts in deep
endgames.

So the answer to "most buckets around midgame" is: not for material. **Material
buckets pay in the endgame, king buckets pay in the middlegame.** They are
complementary along the phase axis, which is the argument for crossing them
rather than choosing one — and it is the first evidence in this file for the
phase x game-type factorisation as a shape.

### Designing the coarsening from the transition graph (`nnue/attic/coarsen.py`)

> **VOID — the numbers in this subsection are not real.** The spectral cut was
> free to build king groups that read only the *opponent's* king, which the
> one-perspective harness prices at zero. `kings graph-64 at 0.23%` actually
> costs about 7.6% per ply. The *idea* survives and is rebuilt correctly in
> "Luka's merge, rebuilt on an honest price" below; the frontier does not.

Luka: "can we get more between-game difference and less within-game
difference", and "join some of them so we don't have too many switches". Those
are one instruction. A rule costs a refresh exactly when a made move carries
the index from cell i to cell j, so build the transition matrix over made moves
(weighted by the measured move-class mix) and merge cells so transitions land
*inside* a merged cell. Minimising the cut of T is minimising the refresh rate.
Normalised spectral embedding, k-means, cells the sample never visited filled
in from their nearest visited neighbour in the rule's own coordinates.

| rule | buckets | perp | in-search refresh |
|---|---|---|---|
| kings 64^2 (HalfKP) | 4,096 | 207 | 8.22% |
| kings 11^2 (hand-designed) | 121 | 17 | 2.63% |
| **kings graph-121** | 121 | **26** | **1.29%** |
| **kings graph-64** | 64 | 16 | **0.23%** |
| kings graph-16 | 16 | 8 | 0.20% |
| kings graph-256 | 256 | 77 | 5.14% |
| material 24^2 | 576 | 62 | 3.94% |
| material graph-64 | 64 | 32 | 2.00% |
| material graph-32 | 32 | 19 | 1.19% |

**The king axis quotients beautifully and the material axis does not.** graph-64
matches the hand-designed 11 regions on effective bucket count (16 vs 17) at
**1/11 the refresh**; graph-121 has more effective buckets than the hand rule
(26 vs 17) at half the refresh. Material only halves.

The mechanism is the asymmetry Luka was reaching for. A king move is *local* --
it steps to an adjacent square -- so the dense part of the transition graph is
short-range wandering, and quotienting it out costs little of what the rule
knows. A capture is not local and it is not incidental: the capture edges are
exactly the distinctions a material rule exists to draw, so cutting along them
throws away the signal. **Between-versus-within is winnable on the king axis
and structurally not on the material axis.**

Two things to hold onto:

* Top group holds ~31% of visits for the king coarsenings. Low cut is partly
  bought by pooling the middlegame wandering states into one lump.
* **Cut-minimisation is free to merge cells the evaluation needed apart.**
  Nothing in this table says the coarsening is *good*. The merge study
  (section 7) is the standing warning; the only honest test is to train it,
  which is what `runs/rules.log` is for.

### A 7% budget

> **VOID.** Set against half-price numbers, and two of its rows use the void
> king coarsenings. In per-ply units the same budget is ~14%, and the
> combinations below cost twice what they say. The additivity result is
> unaffected — rates still add exactly, in either unit.

Raised from 2% on the evidence above. In-search rates add exactly, so:

| combination | rate |
|---|---|
| kings graph-64 + material 24^2 | **4.17%** |
| kings graph-121 + material 24^2 | 5.23% |
| kings 11^2 + phase x8 | 6.00% |
| kings 11^2 + material 24^2 | 6.57% |
| kings 64^2 (HalfKP) alone | 8.22% -- does not fit |
| HalfKP + material 24^2 | 12.16% -- does not fit |

**At 7% a king rule and a material rule fit together and HalfKP alone does
not.** That is the head-to-head: our factorisation is cheaper than HalfKP and
routes on two complementary axes (the phase-binned loss above) rather than one.
Note that two families in `Bucketed` are *additive* readers, not a cross, so
kings + material is 4,096 + 576 readers rather than 2.4M -- which is Luka's
"multiple parallel game types", and it is what makes the combination affordable
in parameters as well as in refreshes.

### Scoring a rule's predictive power without training it (`nnue/attic/power.py`)

A search that only knows stability merges away everything that mattered -- k=1
has a perfect refresh rate. So: freeze the base model's accumulator. A
read-bucketed model is then "give each bucket its own linear read of the frozen
activation", and the best such read has a closed form. Linearise the sigmoid
loss about the base prediction (`s = sigmoid(pred/K)`, `g = s(1-s)/K`,
`r = target - s`) and per bucket it is weighted least squares:
`G_b = sum g^2 Z Z^T`, `c_b = sum g r Z`, reduction `c_b^T G_b^-1 c_b`.

Two properties make it the right tool: **it is additive over cells**, so once
per-cell statistics exist any coarsening is scored by summing them -- no second
pass, which is what makes searching over partitions possible; and it is
**held out**, statistics accumulated on two disjoint halves with the read
fitted on A and scored on B, because otherwise more buckets always wins.

Validated against the six arms of `runs/kingcoarse.log`, which were trained
under identical conditions:

| family | buckets | proxy | trained |
|---|---|---|---|
| none | 1 | 0.05% | 0.00% |
| kings 4^2 | 16 | 0.55% | 3.76% |
| kings 11^2 | 121 | 1.26% | 5.61% |
| kings 16^2 | 256 | 1.63% | 5.85% |
| kings 64^2 | 4,096 | 3.18% | 5.79% |
| material 24^2 | 576 | 5.87% | 13.36% |

**Spearman 0.943, Pearson 0.93-0.98**, stable across dim in {16, 48, 96} and
ridge over four decades. The single rank inversion is `kings16` vs `kings`, the
one pair training says is a tie.

Two things it is not. It is a **lower bound** -- the real model retrains V
jointly and the proxy projects to `dim` PCA directions, so it reads 2-7x low.
And the underestimate is **worse for coarse rules** (kings4 is 6.8x low,
material 2.3x), because reshaping V is exactly what a coarse-trained model
gets to do. Use it to rank candidates and train the finalists; expect it to
under-rate coarse rule sets.

One implementation trap, because it cost a full debugging cycle and looks
right: scaling the ridge by *each bucket's own* diagonal gives a five-sample
bucket a five-sample-sized prior, so it stays underdetermined and its read
explodes on the held-out half. Every rule scored NEGATIVE and the correlation
was -0.77. A single absolute prior on a per-sample scale fixes it.

### Predictive power and stability together (`nnue/attic/rulesearch.py`)

Twenty bitboard-cheap predicates across Luka's three axes, both axes measured.
n = 655,360 positions, dim 48, ridge 1e3.

| rule | declared | perp | power | refresh | power/refresh |
|---|---|---|---|---|---|
| mat: full 24^2 | 576 | 62 | **2.430%** | 3.94% | **0.62** |
| mat: rook x queen 6^2 | 36 | 8 | 0.850% | 1.74% | 0.49 |
| mat: majors 4^2 | 16 | 6 | 0.600% | 1.79% | 0.34 |
| king: wing 3^2 | 9 | 5 | 0.276% | 1.42% | 0.19 |
| pawn: passed 3^2 | 9 | 2 | 0.351% | 1.92% | 0.18 |
| king: region 11^2 | 121 | 17 | 0.468% | 2.63% | 0.18 |
| mat: imbalance 40 | 40 | 10 | 0.767% | 4.56% | 0.17 |
| king: halfkp 64^2 | 4,096 | 207 | 1.076% | 8.22% | 0.13 |
| mat: phase 4 | 4 | 4 | 0.151% | 1.18% | 0.13 |
| mat: phase 8 | 8 | 7 | 0.370% | 3.37% | 0.11 |
| mat: bishop pair 2^2 | 4 | 2 | 0.087% | 0.86% | 0.10 |
| mat: minors 4^2 | 16 | 7 | 0.484% | 5.15% | 0.09 |
| king: rank 3^2 | 9 | 3 | 0.101% | 1.37% | 0.07 |
| pawn: count 4 | 4 | 4 | 0.204% | 4.48% | 0.05 |
| king: shelter 16^2 | 256 | 36 | 0.591% | 13.73% | 0.04 |
| king: castle 4^2 | 16 | 4 | 0.103% | 2.46% | 0.04 |
| pawn: centre4 3^2 | 9 | 5 | 0.202% | 5.77% | 0.03 |
| pawn: advance 3^2 | 9 | 5 | 0.174% | 6.09% | 0.03 |
| pawn: centre 3 | 3 | 3 | 0.076% | 3.07% | 0.02 |
| pawn: files 3^2 | 9 | 2 | 0.074% | 3.89% | 0.02 |

**Material dominates both columns.** The top three on power-per-refresh are all
material; the pawn predicates are the weakest axis in the pool, with passed
pawns the one exception.

**Is balance enough on its own?** Luka's hypothesis was that even splits plus
stability would be sufficient, and evenness would find the game types by
itself. Spearman(power, perplexity) over these 20 rules is **0.883** -- a
good predictor, not a sufficient one. At matched perplexity the spread is
large:

| perp | rule | power |
|---|---|---|
| 2 | pawn: passed 3^2 | **0.351%** |
| 2 | mat: bishop pair 2^2 | 0.087% |
| 2 | pawn: files 3^2 | 0.074% |
| 4 | mat: phase 4 | 0.151% |
| 4 | king: castle 4^2 | 0.103% |
| 5 | king: wing 3^2 | 0.276% |
| 5 | pawn: centre4 3^2 | 0.202% |

**4.7x spread at identical balance**, and the largest disagreement in the whole
table is HalfKP (perplexity 207, efficacy 1.076%) against material 24^2
(perplexity 62, efficacy 2.430%) -- 3.3x fewer effective buckets and 2.3x more
power, which is exactly what training found. So: use balance as the cheap first
filter when sweeping a large space, and predictive power for the final ranking.

**These numbers are individual and do NOT add.** Two rules that carry the
same information score well separately and gain nothing together -- the
psqt-versus-read result is the standing example. Combining requires
accumulating the statistics on the joint signature, which is the next build.

### Is the proxy just reading back what the accumulator was trained to keep?

Luka's objection, and it is the right one to raise: a frozen accumulator can
only show information it kept, so the score may measure "what V remembers"
rather than "what the rule knows". Features V was trained to represent would
look good by construction.

Control: replace the accumulator with a RANDOM projection of the same scale --
which carries no training bias, because a random projection preserves
information generically -- and rescore the pool. The prediction, and therefore
the residual being fitted, still comes from the trained model, because that is
the error we are trying to reduce.

| rule | trained V | random V |
|---|---|---|
| mat: full 24^2 | 2.437% | 1.331% |
| king: halfkp 64^2 | 1.085% | 0.723% |
| mat: rook x queen 6^2 | 0.843% | 0.381% |
| mat: imbalance 40 | 0.756% | 0.336% |
| king: shelter 16^2 | 0.597% | 0.369% |
| mat: majors 4^2 | 0.594% | 0.232% |
| king: region 11^2 | 0.440% | 0.221% |
| mat: phase 8 | 0.368% | 0.137% |
| pawn: passed 3^2 | 0.355% | **0.261%** |
| king: wing 3^2 | 0.263% | 0.130% |
| pawn: count 4 | 0.224% | **0.041%** |
| pawn: files 3^2 | 0.069% | 0.008% |

**Spearman 0.949.** Levels roughly halve -- a random representation is a worse
one -- but the ordering is essentially unchanged, so the ranking is not an
artefact of V's training. The objection is real in principle and does not bite
in practice.

Where it does bite is visible in the ratios. `pawn: passed 3^2` keeps 74% of
its score under a random projection: its information is nearly a per-bucket
offset and needs no learned representation. `pawn: count 4` keeps 18% and
`pawn: files` 12% -- those genuinely depend on what V learned. So the proxy is
mildly biased toward representation-dependent features, which is the direction
Luka predicted, but not enough to reorder the table.

One implementation note, because the first version of this control was
worthless: swapping V also changes the model's own prediction, so every rule
"fixed" a broken model by ~27% and the control measured nothing. The
prediction has to stay the trained model's; only the readable representation is
randomised.

### Staged rule construction (`nnue/attic/stagedrules.py`)

> **VOID as a result, kept as a record of two mistakes worth not repeating.**
> The prices are half-price, and the search's best rules exploited the missing
> half. Separately, the merge ORDER was wrong: joining the pair with the lowest
> combined mass joins them across the *lightest* edge, which is the least
> refresh saved per bucket spent. Merging material 2,916 cells to 128 removed
> **13%** of the cut; the corrected rule removes **62%**. Both fixes are in
> `nnue/attic/merge2.py`.

Luka's design, and the part he thinks is the important idea: do not build the
full product and prune it. Prune each axis first -- kings, material, pawns --
then cross the pruned axes and prune the product. The raw product is
4096 x 2916 x 4096 cells, so crossing first is not an option anyway, and it
would spend the budget on distinctions each axis already showed were not worth
keeping. Target: about 512 rules, roughly 4 bits of king + 7 of material +
5 of pawns before the joint prune.

His merge rule, which is different from `coarsen.py`'s spectral cut and better:
repeatedly take the pair of cells that are **connected** -- reachable from one
another in a single made move -- whose **combined visit probability is
lowest**, and merge them. Rare cells get eaten first; a merged cell carries the
sum of its parts and stops being cheap; only connected cells merge, so a
transition that used to cost a refresh ends up inside a bucket. Cells with no
transitions attach to the smallest group; cells never visited attach to their
nearest visited neighbour in the axis's own coordinates.

**On the king axis it works beautifully.** 4,096 cells to 16 groups: effective
count 11.5 at **0.14% refresh**, against the fine rule's 8.22%. That beats the
spectral coarsening on both axes (0.20%, effective 8).

**But "even by necessity" is false, and the failure mode is instructive.** On
the pawn axis, 32 groups came out with an effective count of **1.5** -- one
bucket. The transition graph is star-shaped: rare pawn structures are not
connected to *each other*, only to the common one, so the only connected pair
on offer is always leaf-to-hub and the hub eats the board. Merging the lowest
pair does not balance when the graph is a star.

The fix is one line and in the spirit of the design: refuse any merge that
would take a group past `cap / k` of the mass. Evenness becomes a constraint
instead of a hope. With cap = 2.0 over 105,536 positions:

| axis | fine | seen | -> | effective | refresh | fine refresh |
|---|---|---|---|---|---|---|
| kings 64^2 | 4,096 | 2,816 | 16 | 13.4 | 1.18% | 8.02% |
| material 54^2 (N,B,R,Q per side) | 2,916 | 750 | 128 | 53.0 | 6.22% | 7.88% |
| pawns 64^2 (per-wing counts) | 4,096 | 1,692 | 32 | 22.9 | 17.19% | 24.17% |

Joint 16 x 128 x 32 pruned to 512: **effective 419.6 of 512, 82% even** --
exactly the property Luka wanted -- but **refresh 15.96%**, well over budget.

**The cap is a trade: it buys evenness with stability.** Uncapped kings were
0.14% at effective 11.5; capped they are 1.18% at effective 13.4.

**Pawns are the problem axis, and three independent measurements now agree.**
Worst predictive power in the 20-predicate pool; worst refresh of the three
fine spaces (24.17%); and the only axis whose transition graph is star-shaped.
The king/pawn/material trio may be right about chess and wrong about routing --
a pawn structure that changes character is exactly a pawn structure whose index
moves.

A sweep over seven configurations (including pawns switched off entirely and
pawns at 8 groups, caps 1.5 / 3.0 / infinity, all pruned to 512, 500k
positions) was still running when this was written. What it decides: whether
512 rules under 7% is reachable at all, and what keeping the pawn axis costs.

### The price was half, and the search found the missing half

Luka spotted it from the numbers alone: the staged search produced 16 king
buckets with an effective count of 11.2 and a refresh rate of **0.00%**. His
argument — if the mass really is spread over 11 buckets, a walk that starts
somewhere has to leave, so a zero cut is impossible on a connected graph. The
king-move graph on (our king, their king) pairs *is* connected, so 0.00% could
only be a measurement artefact.

It was. Printing the group map as a function of both king squares:

```
group(our king, their king), rows = our king a1.. , cols = their king
0  5  3  0  3  5  1  5  2  7  5  3  5  8  0 12
0  5  3  0  3  5  1  5  2  7  5  3  5  8  0 12
0  5  0  0  3  5  1  5  2  7  5  3  5  8  0 12
```

Constant down the columns: **the rule ignores our own king and buckets on
theirs.** `refresh.py` applies the mover's move in the mover's perspective, and
the records are stm-canonical, so the opponent's pieces never move in that
view. A rule reading only the opponent's half is free by construction — in that
view. Verified against a second accumulator built by mirroring the record
(swap the side bit, flip the rank), n = 60k positions:

| rule | mover's acc | other acc | **per ply** |
|---|---|---|---|
| kings 64^2 (HalfKP) | 8.10% | 8.10% | **16.20%** |
| material 24^2 | 4.00% | 4.00% | **8.00%** |
| the staged 16-king rule | **0.00%** | 7.65% | **7.65%** |

So the rule the search was proudest of costs about what HalfKP costs per
accumulator, and buys 11 effective buckets instead of 207. It bought nothing.

The general lesson is not about chess. **A search will find any direction the
cost function cannot see**, and the way to catch it is a conservation argument
of exactly the kind Luka used, not a bigger sample.

### Luka's merge, rebuilt on an honest price (`nnue/attic/merge2.py`)

Two corrections at once. Price: both accumulators, per ply. Order, which is
Luka's own fix and separates the two decisions the old rule conflated —

    while more than k non-empty cells:
        i <- the LEAST-VISITED live cell          (rarity picks what to spend)
        j <- the neighbour of i joined by the      (connection strength picks
             HEAVIEST transition                    where it goes)
        merge i into j
    then fold the never-visited cells into the final buckets

Zero-mass cells are held out of the merge entirely so they cannot act as a
rare-cell dumping ground that flatters the mass profile. They are placed at the
end by Luka's rule for that too: count which coordinate each real merge crossed
— those are the walls that broke down, so they are the cheap directions to
travel — and snap each unvisited cell to the nearest visited one under a
distance that makes those directions cheap.

n = 173,216 positions, both accumulators, made-move weighted:

| kings (un-merged 16.04%/ply) | effective | refresh/ply | cut removed |
|---|---|---|---|
| 8 | 6.5 | 3.57% | 78% |
| 16 | 11.2 | 4.98% | 69% |
| 32 | 21.4 | 7.98% | 50% |
| 64 | 36.3 | 9.24% | 42% |
| 128 | 59.7 | 11.01% | 31% |

| material N/B/R/Q per side (un-merged 15.79%/ply) | effective | refresh/ply | cut removed |
|---|---|---|---|
| 8 | 7.3 | 1.61% | **90%** |
| 16 | 12.9 | 2.32% | 85% |
| 32 | 24.5 | 3.56% | 78% |
| 64 | 42.5 | 4.98% | 69% |
| 128 | 64.1 | 5.98% | **62%** |

**The axis ranking reverses.** The old study said kings quotient beautifully
and material does not. With the price fixed and the merge order fixed, material
is about half the price of kings at every matched effective count: ~12
effective costs material 2.32% and kings 4.98%; ~24 effective costs 3.56%
against 7.98%. The old conclusion was an artefact of both bugs pointing the
same way — the cheap king rules were the asymmetric ones, and the material
merge was crippled by joining across light edges.

Joint (cross two pruned axes, prune the product with the same rule):

| joint | cells | effective | refresh/ply |
|---|---|---|---|
| K16 x M32 -> 256 | 512 | 158.6 | 7.75% |
| **K32 x M64 -> 256** | 2,048 | **184.6** | **7.79%** |
| K32 x M64 -> 512 | 2,048 | 309.0 | 9.76% |
| K64 x M128 -> 512 | 8,192 | 337.9 | 9.40% |
| K64 x M128 -> 1024 | 8,192 | 569.4 | 11.14% |

Against the strongest hand rule — material 24^2, ~62 effective buckets at
8.00%/ply — **K32 x M64 -> 256 gives three times the effective buckets at the
same price.** That is the first time the machinery has been ahead of a hand
rule on an honest price, and it is what `--experiment merged` is training.

#### What the rule still reads

A mutual-information column, added before Luka clarified that the direction
histogram was meant for placing unvisited cells: how much of each coordinate's
entropy the final bucket still carries. It answers a different question and is
worth keeping for it.

| material buckets | our N | our B | our R | our Q |
|---|---|---|---|---|
| 8 | 33.5% | 34.7% | 53.0% | **90.6%** |
| 16 | 39.1% | 47.0% | 71.7% | **91.2%** |
| 32 | 58.4% | 65.8% | 77.9% | 91.9% |

At 8 buckets the merge keeps 91% of the queen bit and throws away two thirds of
knight count. It rediscovered, from transition statistics alone, roughly what
`b_material` was hand-built to say — queens matter, knight count is the most
disposable. On kings all four coordinates come out at 25-35% and none is
disposable: king position is genuinely four-dimensional.

### Tier assignment

The naive table said material was the cheap pre-accumulator rule and the king
was the expensive one. **Corrected: in the real tree it is the other way
round.** A king rule is nearly free on captures (0.13-0.76%) and costs only on
king moves; a material rule is free on quiet moves and costs on every capture,
which is where the tree spends 30% of its make_move calls.

| tier | rule | effective | in-search refresh |
|---|---|---|---|
| stable, pre-accumulator | castle state 4^2, or king region 11^2 | 4 / 17 | 2.46% / 2.63% |
| dynamic, post-accumulator | material 24^2 x shelter | 874 | free here |

Cost of a refresh, for calibrating the budget [INFERENCE]: an incremental
update is ~2-3 row operations, a full rebuild ~24 in the middlegame, so a
refresh is ~10x an update and the expected accumulator cost is `1 + 9r`. At
r = 2% that is +18% on accumulator time — and the accumulator is only part of
eval, so less than that overall. The 10x and the accumulator's share of eval
are both estimates; neither has been measured on this engine, which has no
accumulator yet (LEDGER 019).

**But nothing above is a reason to spend the budget yet.** Every accuracy
number in this file comes from *post-accumulator* conditioning, which costs
zero refresh: in `Bucketed` the rule picks the read and a pre-activation shift,
both after `Vx`. The one pre-accumulator datapoint is section 5, and it argues
against the whole axis: psqt x (material+count) gave -6.08%, read x
(material+count) gave -16.08%, and the two did not add — the same information
entered twice. The experiment that decides whether the refresh budget matters
at all is conditioning **V itself** (rule x feature transformer) and measuring
the gain *on top of* an already read-conditioned model with the same rule. If
that is small, every rule belongs behind the accumulator where it is free.
