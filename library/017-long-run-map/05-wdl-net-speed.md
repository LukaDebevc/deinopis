# 017/05 · WDL (WD), the net (NN), engine speed (SP)

## WD · Does WDL do anything a cp eval would not?

What is known. The search reads one scalar, `cp = 288.5·logit(W + D/2)`.
D is used only by contempt code, which lost in every shape (STATE
2026-09-10: −189, −64, −46) and by the risk-sensitive readouts (066: −50 to
−254). Training on lc0's 3-way labels beats training on a sigmoid of cp
(058, "shape, not smoothing"). New (092): **the net's D is ~96% a function
of (E, material)**; its residual carries a modest stability signal (drawish
beyond material → ~20–30% less shallow-search error).

So the question splits three ways: (1) does the 3-way *head* help learn E?
(2) does D carry search-useful information beyond material? (3) is E the
right scalar to read? WD-1 answers (1) and is the cleanest test of "WDL vs
cp". WD-2..WD-7 answer (2). (3) is mostly answered by 066.

**WD-1 · GPU → SPRT · scalar head vs 3-way head, same labels.** Two arms,
identical trunk, data, lc0 labels, steps, seed set: a. 3-way CE on
(W, D, L) (today); b. one sigmoid output trained with BCE on
`E_lc0 = W + D/2`. Metric everyone can read: BCE of the predicted E against
`E_lc0` and against the game outcome, on the frozen val set; 3 seeds each at
~2B positions (seed spread ~0.0008, 058). Then the winner of each at full
length and **equal-time games** (`netcmp.sh`). If b ≈ a on E and in games,
the WDL head is decoration and the 3-way target's value was the teacher,
not the head. Variant c: 3-way head with the loss on E only (tests whether
the parameterisation or the loss carries it).

**WD-2 · PROBE · redo 092 on tree positions.** 092 sampled fishpack
positions; the search evaluates tree positions (fewer pieces, mid-exchange:
memory "refresh price needs tree positions", LEDGER 022). Dump 20k eval
calls with `--features movedump`, rerun `probes/wdl_probe.py` and
`sigma_an.py`. If D's residual predicts σ more strongly there, WD-3/4 rise.

**WD-3 · SPRT · a node-level drawishness term.** Price children of a node
by the *node's* D-residual (the child's D is unknown before make): budget
−= `c_drawres · max(0, Dres − 0.05)` at non-PV nodes. Must be compared with
a **material-only control arm** (same form with a material proxy for
drawishness), since 92–96% of D is material. Variants: coefficient
500/1000/2000 mp.

**WD-4 · SPRT · D-scaled pruning margins.** RFP and futility margins ×
`(1 − k·Dres)` — prune more readily where the net thinks the position is
dead. k = 0.3/0.6. Material-only control arm again.

**WD-5 · SPRT · TT-stored D for children.** Store D (8 bits) in the TT entry
(with FX-5's new layout) so a child's D is known before searching it when
the TT has it; then a true per-child `Sigma` feature `c_child_draw`.
Only if WD-3 shows any signal.

**WD-6 · SPRT · D in time management.** Root `Dres` high → spend less time
(allocation × (1 − ½·Dres)). Cheap, one arm, TU-6's driver can tune it.

**WD-7 · PROBE · does the *teacher's* D know more than the student's?**
Run 092's regression on lc0's labels (`nnue/lc0/`): R² of D_lc0 on (E_lc0,
material). If lc0's D is much less material-determined than the net's, the
student is failing to learn position-specific drawishness — a capacity or
input-feature problem (fortresses, opposite bishops), which makes NN-3/NN-4
more interesting.

**WD-8 · SPRT · sharpness-gated contempt** (the one form the contempt study
left open). Weight the asymmetric contempt bonus by `|static − qsearch|`
or LC-9's σ̂ instead of D, so only live positions get fought for. Price it
in self-play *and* against the gauntlet anchors, where contempt is supposed
to pay.

**WD-9 · PROBE · readout geometry.** Margins in cp are margins in logit(E).
Try margins in E itself (`E_static − m·depth ≥ E_beta`) via a readout
switch for pruning only. 066 changed the returned *value*; this changes only
where pruning thresholds sit. Low prior; proxy-screen only unless it
resolves.

## NN · The net

**NN-1 · GPU · unique games.** 083: more passes buy nothing (+2.4 over
3000 games); 021/one-epoch rule: buy unique positions. ~106 GB never used.
Do: fetch 6–12 more months with `--game-stride`, relabel with lc0 (15.5
GPU-h per 893M, 058), train one epoch at the m1 recipe (lr 6.7e-4, wd 1e-2,
052), `steps × batch ≤ unique`. Variants: a. same total positions, 2x
distinct games (stride within games); b. 2x total positions; c. a. plus
m1-b1 warm start.

**NN-2 · GPU → SPRT · width 256 on the m1 recipe.** `gateh` (256) led on
Elo in 056 despite worse val, with +26% nps; never retrained on lc0 labels.
Variants: 256, 384, 512 (control), all on NN-1's data. Equal-time games only.

**NN-3 · GPU → SPRT · king-bucketed input.** STATE: HalfKA lost at 250M —
"a budget result, not a verdict". Variants: a. 4 king buckets on the ft
(mirrored); b. 8; c. 16; price the refresh with `treeprice.py` on tree
positions (034). Now 30B+ positions are affordable.

**NN-4 · GPU · pawn features into the deep head** (009/20: "the obvious
next arm", never run).

**NN-5 · GPU · train on the positions the search evaluates.** Mix tree
positions (movedump, labelled by lc0) into training at 0/25/50%. Tree
positions have 16.6 pieces vs 19.4 and mostly a capture available (022).

**NN-6 · GPU · blend in search targets.** Target = `(1−λ)·lc0_WDL +
λ·sigmoid(V_64k/K)` from our own search; λ = 0/0.25/0.5. Standard in other
engines; tests whether our search adds to lc0's view.

**NN-7 · GPU → SPRT · int8 activations** (ROADMAP 4, library/010: `down`
reads 32.8 KB f32 per eval — memory, not arithmetic). QAT via `qat.py`,
envelope from 053. Speed item as much as a net item.

**NN-8 · GPU · a stronger or ensembled teacher.** lc0 net size/version as
a variable: relabel 50M with two lc0 nets and train short arms (058's
method). If the better lc0 net trains a better student, NN-1 should use it.

## SP · Engine speed (the eval is ~70% of search time)

Rule of thumb [GUESS]: +10% nps ≈ +5–8 Elo at blitz. SPRT `[0,3]` for pure
speed (P4). Measure with interleaved A/B pairs, `taskset`, ≥3 pairs
(OPTIMIZATIONS methodology).

**SP-1 · lazy accumulator.** 091: 34% of pushes are never read (tune pass;
re-measure on bench). Record the delta on push, apply on the first eval
that needs it. Neutral in nodes; nps A/B; then `[0,3]`.

**SP-2 · head and psqt codegen** (ROADMAP 4c.3: 0.212 and 0.376 ns/MAC vs
0.033 elsewhere; ~10% of search; 087 estimated +7–11 Elo).

**SP-3 · int8 `down` activations** (with NN-7).

**SP-4 · f32 kernel headroom** (~2x on paper, 0.14 vs ~0.06 cycles/MAC).

**SP-5 · PROBE · an eval cache separate from the TT** in the engine. After
the Q-store, how many q-node evals repeat a key whose TT entry was
overwritten? `--features qprobe` counter. Build only if > 5% of evals.

**SP-6 · refresh cost of king moves** — Finny-table style cached
accumulators per king bucket, once NN-3 adds buckets.

**SP-7 · PROBE · a re-profile after every checkpoint** (`nodeprof`,
`--emit-prices`, re-emit `work.rs`; STATE Next 2 is still queued).
