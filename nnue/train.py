"""Trainer for the quadratic-form eval, and the experiments that gate it.

Everything is trained in win-probability space:

    loss = ( sigmoid(f/K) - [ lam*sigmoid(s/K) + (1-lam)*z ] )^2

f is the model's centipawn output, s the teacher's search score, z the game
result. The sigmoid is not cosmetic: in raw centipawn space the model would
spend most of its capacity on the difference between +900 and +1100, which
never changes a game. In win-probability space that region is saturated, the
gradient vanishes, and capacity moves to balanced positions where eval accuracy
actually decides games.

K is fitted from the data by maximum likelihood, not copied from another
pipeline. On this dataset it comes out near 259 cp, not the 400 that circulates.
"""

import json
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import Batcher, GpuUnpacker, NFEAT, PAD   # noqa: E402

# `import router` and not `from router import *`: main() sets this module's
# tape and custom-loss settings by ATTRIBUTE assignment (router.TAPE_STRIDE =
# ...), which only works through the module object. A `global TAPE_STRIDE` here
# would bind a name in train.py that nothing reads, and every run would finish
# normally at the default.
import router                                         # noqa: E402
from common import fit_k, bag, loss_fn, outcome_nll   # noqa: E402
from features import *                                # noqa: E402,F403
from buckets import *                                 # noqa: E402,F403
from movegraph import move_graph, graph               # noqa: E402
from models import *                                  # noqa: E402,F403
from wdl import *                                     # noqa: E402,F403
from router import (Router, LogitNorm, RoutedPSQTBucket, GameTape,   # noqa: E402
                    parse_aux, flip_proxy, router_report, merge_ladder)



@torch.no_grad()
def evaluate(model, b, K, lam):
    """Val loss in the units the SCALAR arm reports, for every arm.

    A WDL arm's cross-entropy is a different number from the scalar arm's
    squared error and the two cannot be ranked against each other. So a WDL arm
    is scored here on its cp readout, K logit(W + D/2), pushed through the SAME
    `loss_fn` against the SAME target. That is the pre-registered comparison:
    does predicting the richer target build a better trunk, judged on the
    scalar the engine actually consumes. Its own cross-entropy is reported
    separately by `evaluate_wdl` and is for monitoring, not for ranking."""
    model.eval()
    tot, onll, n = 0.0, 0.0, 0
    for feat, s, z in b.val_batches():
        out = model(feat)
        if getattr(model, "wdl", False):
            E = wdl_expected(*out)
            pred = wdl_cp(*out, K)
        else:
            pred = out
            E = torch.sigmoid(out / K)
        tot += loss_fn(pred, s, z, K, lam).item() * len(feat)
        onll += outcome_nll(E, z).item() * len(feat)
        n += len(feat)
    model.train()
    return tot / n, onll / n


@torch.no_grad()
def evaluate_wdl(model, b, teacher, lam):
    """The WDL arm's own three-way loss. Monitoring only -- NOT comparable to
    the scalar arm, which is exactly why `evaluate` exists in the form it does."""
    model.eval()
    tot, n = 0.0, 0
    for feat, s, z in b.val_batches():
        A, S = model(feat)
        tot += loss_wdl(A, S, s, z, teacher, feat, lam).item() * len(feat)
        n += len(feat)
    model.train()
    return tot / n


def flip_batch(feat, score, result, gen):
    """Show a random half of the batch from the other chair, labels negated.

    The records are stm-canonical: the mover is always at the bottom in white.
    This undoes that for half the batch, so the model sees each position in
    whichever orientation the coin picked and has to learn both. It is not the
    "absolute colours" encoding -- the data no longer records which colour was
    really to move, so that one is not reachable -- it prices the *consistency*
    of the canonicalisation, which is the part that costs table space.
    """
    m = torch.rand(feat.shape[0], device=feat.device, generator=gen) < 0.5
    f = torch.where(m.unsqueeze(1), flip_feat(feat), feat)
    return f, torch.where(m, -score, score), torch.where(m, 1.0 - result, result)


def log_points(steps, every):
    """Where to evaluate: dense and geometric early, then every `every`.

    A run that only reports every 2000 steps cannot tell a model that is slow
    to start from one that is short of capacity, which is the whole thing this
    methodology change is for. The early points cost one val pass each.
    """
    pts = {steps}
    p = 25
    while p < every:
        pts.add(p)
        p *= 2
    pts.update(range(every, steps + 1, every))
    return sorted(p for p in pts if p <= steps)


def train(model, b, K, lam, steps, lr, opt_name="adamw", log_every=200, tag="",
          augment=False, warm_buckets=0.0, curve_path="", warmup=0, teacher=None,
          tie_after_step=None):
    model.to(b.device).train()
    # The router usually wants a different step size from the table it routes
    # to, and raising the ARM's lr to get it would also move the delta table
    # and confound the comparison with the fixed-rule arms. So it is a separate
    # parameter group: everything else keeps exactly the lr the other arms use.
    mult = getattr(model, "rlr_mult", 1.0)
    rwd = getattr(model, "router_wd", 0.0)
    if mult != 1.0 or rwd:
        rp = [p for n, p in model.named_parameters() if n.startswith("router.")]
        op = [p for n, p in model.named_parameters()
              if not n.startswith("router.")]
        params = [{"params": op, "lr": lr, "weight_decay": 0.0},
                  {"params": rp, "lr": lr * mult, "weight_decay": rwd}]
    else:
        params = model.parameters()
    opt = (torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
           if opt_name == "adamw" else
           torch.optim.SGD(params, lr=lr))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps - warmup))
    sched = (torch.optim.lr_scheduler.SequentialLR(
                 opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.02, 1.0, warmup), cos],
                 milestones=[warmup])
             if warmup else cos)

    release = int(steps * warm_buckets)
    if release:
        model.force_base = True
        print(f"  curriculum: bucket 0 only for {release} steps", flush=True)
    gen = torch.Generator(device=b.device).manual_seed(1234)

    curve = None
    if curve_path:
        os.makedirs(os.path.dirname(curve_path) or ".", exist_ok=True)
        curve = open(curve_path, "w", buffering=1)
        curve.write("step,positions,train,val,outcome_nll,lr,phase,seconds\n")

    want = set(log_points(steps, log_every))
    hist, t0 = [], time.time()
    try:
        for i, (feat, s, z) in enumerate(b.train_batches(steps), 1):
            if release and i == release + 1:
                model.release_buckets()
                print(f"  curriculum: buckets released at step {i}", flush=True)
            if augment:
                feat, s, z = flip_batch(feat, s, z, gen)
            if getattr(model, "wdl", False):
                A_, S_ = model(feat)
                l = loss_wdl(A_, S_, s, z, teacher, feat, lam)
            else:
                # arm B: same scalar head, better-calibrated teacher.
                tch = teacher if getattr(model, "use_teacher", False) else None
                l = loss_fn(model(feat), s, z, K, lam, tch, feat)
            # A model may ask for an extra term (the router's load-balance
            # penalty). It is added to what is BACKPROPPED and kept out of what
            # is PRINTED, so the logged train loss stays comparable across arms.
            aux = getattr(model, "_aux", None)
            opt.zero_grad(set_to_none=True)
            (l if aux is None else l + aux).backward()
            opt.step()
            if tie_after_step is not None:
                tie_after_step(model)
            sched.step()
            if i in want:
                v, onll = evaluate(model, b, K, lam)
                hist.append((i, v, onll))
                dt = time.time() - t0
                phase = "locked" if (release and i <= release) else "free"
                if curve:
                    curve.write(f"{i},{i * b.batch},{l.item():.6f},{v:.6f},"
                                f"{onll:.6f},"
                                f"{opt.param_groups[0]['lr']:.3e},{phase},{dt:.1f}\n")
                print(f"  {tag:<26} step {i:>6}  train {l.item():.6f}  val {v:.6f}"
                      f"  onll {onll:.6f}  {i * b.batch / dt:,.0f} pos/s", flush=True)
    finally:
        if curve:
            curve.close()
    return hist


def build_arms(args):
    """(name, factory, lr) -- factories so each arm is built under its own seed."""
    r, lr = args.rank, args.lr
    if args.experiment == "depth":
        return [("768->1", lambda: DeepLinear(()), lr),
                ("768->16->1", lambda: DeepLinear((16,)), lr),
                ("768->64->16->1", lambda: DeepLinear((64, 16)), lr)]
    if args.experiment == "scale":
        # Does the flat model's disadvantage vanish once it can reach centipawn
        # scale? Same function class and same output distribution at init in
        # every arm; only the step geometry differs.
        return [("768->1 scale=1 lr=0.03", lambda: DeepLinear((), 1.0), 0.03),
                ("768->1 scale=1 lr=0.3", lambda: DeepLinear((), 1.0), 0.3),
                ("768->1 scale=400", lambda: DeepLinear((), 400.0), lr),
                ("768->64->16->1 (ref)", lambda: DeepLinear((64, 16)), lr)]
    if args.experiment == "ceiling":
        return [(f"quadratic r={r}", lambda: Quadratic(r, args.scale), lr),
                (f"relu h={r}", lambda: ReLUNet(r, args.scale), lr)]
    if args.experiment == "psqt":
        return [("psqt 768->64->16->1", lambda: DeepLinear((64, 16), args.scale), lr)]
    if args.experiment == "warmstart":
        return [(f"quadratic r={r}", lambda: Quadratic(r, args.scale), lr)]
    if args.experiment == "compare":
        # The gate on the whole idea. `psqt` is the best any model linear in x
        # can do, so it is the floor the quadratic form must beat. `relu` is
        # the control for the degree-2 ceiling: a quadratic form cannot express
        # any three-way interaction, and king safety is genuinely three-way.
        return ([("psqt 768->64->16->1",
                  lambda: DeepLinear((64, 16), args.scale), lr)] +
                [(f"quadratic r={k}", (lambda k=k: Quadratic(k, args.scale)), lr)
                 for k in (64, 256)] +
                [(f"relu h={k}", (lambda k=k: ReLUNet(k, args.scale)), lr)
                 for k in (64, 256)])
    if args.experiment == "groups":
        # Deployability screen: which *subset* keeps the gain at a size that
        # fits in cache. Each group is "fam+fam[:ro]"; ":ro" drops the
        # pre-activation shift, which the bishops ablation measured at 0.09%.
        arms = []
        for g in args.groups.split(","):
            g = g.strip()
            if not g:
                continue
            ro = g.endswith(":ro")
            fams = tuple(g[:-3].split("+") if ro else g.split("+"))
            n = sum(FAMILIES[f][1] for f in fams)
            arms.append((f"{'+'.join(fams)} x{n}{' read only' if ro else ' read+pre'}",
                         (lambda fams=fams, ro=ro: Bucketed(args.rank, args.scale,
                                                            families=fams, pre=not ro)),
                         lr))
        return arms
    if args.experiment == "buckets":
        # Baselines first, and both matter. r=512 with one bucket isolates the
        # bucket effect at fixed rank; r=768 with one bucket is the *deployed*
        # model and the maximum of the whole single-bucket quadratic family --
        # beating it is the only thing that means anything.
        fams = [f.strip() for f in args.families.split(",") if f.strip()]
        arms = [("base r=512 x1", lambda: Bucketed(512, args.scale, families=("none",)), lr),
                ("base r=768 x1 (deployed)",
                 lambda: Bucketed(768, args.scale, families=("none",)), lr)]
        for f in fams:
            arms.append((f"{f} x{FAMILIES[f][1]} read+pre",
                         (lambda f=f: Bucketed(512, args.scale, families=(f,),
                                               pre=True)), lr))
        # Ablation: is the pre-activation shift doing anything, or is the read
        # alone the whole effect? Same family, one difference.
        if fams:
            arms.append((f"{fams[0]} x{FAMILIES[fams[0]][1]} read only",
                         (lambda f=fams[0]: Bucketed(512, args.scale,
                                                     families=(f,))), lr))
        if len(fams) > 1:
            arms.append((f"all: {'+'.join(fams)} read+pre",
                         (lambda: Bucketed(512, args.scale,
                                           families=tuple(fams), pre=True)), lr))
        return arms
    if args.experiment == "rules":
        # Luka's two questions at once.
        #
        # 1. Smarter rules: does a coarsening designed from the TRANSITION
        #    GRAPH train as well as a hand-designed spatial one at the same
        #    bucket count, while costing far less refresh? (kgraph121 is 1.29%
        #    against kings11's 2.63%; kgraph64 is 0.23% at the same
        #    perplexity.) Cut-minimisation is free to merge cells the eval
        #    needed apart, so this is the only test that means anything.
        #
        # 2. Our idea against HalfKP, under a 7% refresh budget. Note that two
        #    families in `Bucketed` are ADDITIVE readers, not a cross -- which
        #    is exactly Luka's "multiple parallel game types", and is why
        #    kings + material is 4096+576 readers rather than 2.4M.
        b = lambda *f: (lambda: Bucketed(512, args.scale, families=f, pre=True))
        return [("material 24^2 (anchor)", b("material"), lr),
                ("kings 11^2 (hand, 2.63%)", b("kings11"), lr),
                ("kings graph 121 (1.29%)", b("kgraph121"), lr),
                ("kings graph 64 (0.23%)", b("kgraph64"), lr),
                ("kings 64^2 HalfKP (8.22%)", b("kings"), lr),
                ("kings11 + material (6.57%)", b("kings11", "material"), lr),
                ("kgraph64 + material (4.17%)", b("kgraph64", "material"), lr),
                ("HalfKP + material (12.16%)", b("kings", "material"), lr)]
    if args.experiment == "staged":
        # Train the staged rules instead of trusting the proxy. The proxy said
        # the plain hand rule beats every staged one (2.450% against 1.718%),
        # and the proxy has a known bias against merged rules -- it reads coarse
        # rules 2-7x low. So this is the only measurement that settles it.
        #
        # Matched declared size is the point: material 24^2 is 576 readers at
        # 3.94% refresh, staged512 is 512 at 5.56%. If the staged rule does not
        # win here it is not paying for its machinery, whatever the proxy says.
        b = lambda *f: (lambda: Bucketed(512, args.scale, families=f, pre=True))
        return [("base, no buckets", b("none"), lr),
                ("material 24^2 (576, 3.94%)", b("material"), lr),
                ("staged 512 (5.56%)", b("staged512"), lr),
                ("staged 1024 (6.12%)", b("staged1024"), lr),
                ("staged512 + material (additive)", b("staged512", "material"), lr)]
    if args.experiment == "merged":
        # Does a coarsening designed from the transition graph train as well as
        # a hand rule at the same PRICE? Prices are per ply across both
        # accumulators (nnue/merge2.py); the old half-price numbers let an
        # earlier search win by reading only the opponent's half.
        #
        #   material 24^2   576 declared,  ~62 effective, 8.00%/ply   <- anchor
        #   merged256       256 declared, 184.6 effective, 7.79%/ply
        #   merged512       512 declared, 337.9 effective, 9.40%/ply
        #   merged1024     1024 declared, 569.4 effective, 11.14%/ply
        #   kings 64^2     4096 declared,             --, 16.20%/ply
        #
        # merged256 is the arm that matters: same price as the anchor, three
        # times the effective buckets. If it does not beat the anchor, the
        # search is not buying anything and the direction is dead.
        b = lambda *f: (lambda: Bucketed(512, args.scale, families=f, pre=True))
        return [("base, no buckets", b("none"), lr),
                ("material 24^2 (8.00%/ply)", b("material"), lr),
                ("merged 256 (7.79%/ply)", b("merged256"), lr),
                ("merged 512 (9.40%/ply)", b("merged512"), lr),
                ("merged 1024 (11.14%/ply)", b("merged1024"), lr),
                ("kings 64^2 HalfKP (16.20%/ply)", b("kings"), lr)]
    if args.experiment == "vbuck":
        # The experiment that decides whether the refresh budget matters at all.
        #
        # Everything measured so far conditions the READ, which happens after
        # the accumulator and therefore costs zero refreshes. Section 5 of
        # library/009 is the one pre-accumulator datapoint and it is
        # discouraging: psqt x (material+count) gained 6.08% where read x
        # (material+count) gained 16.08%, and the two did not add.
        #
        # So: same rule, three arms. Read-conditioned only; read-conditioned
        # with 16 extra accumulator entries that are NOT bucketed (the width
        # control -- extra capacity is extra capacity); read-conditioned with
        # those 16 entries bucketed. If arm 3 does not beat arm 2, no rule
        # belongs in front of the accumulator and the whole refresh-rate study
        # is a cost with no benefit to price.
        k = 16
        arms = []
        for label, fam in (("material 24^2", "material"),
                           ("kings 11^2", "kings11")):
            arms += [
                (f"{label} read+pre r=512",
                 (lambda f=fam: Bucketed(512, args.scale, families=(f,), pre=True)), lr),
                (f"{label} read+pre r={512 + k} (width control)",
                 (lambda f=fam: Bucketed(512 + k, args.scale, families=(f,), pre=True)), lr),
                (f"{label} read+pre r=512 + {k} bucketed acc",
                 (lambda f=fam: Bucketed(512, args.scale, families=(f,), pre=True,
                                         vbuck=k)), lr),
            ]
        # Severity control: same table size, same parameters, random routing.
        for label, fam in (("random x576", "rand576"), ("random x121", "rand121")):
            arms.append(
                (f"{label} read+pre r=512 + {k} bucketed acc",
                 (lambda f=fam: Bucketed(512, args.scale, families=(f,), pre=True,
                                         vbuck=k)), lr))
        return arms
    if args.experiment == "ratio":
        # Luka's allocation sweep, in the only form that has content. Total
        # accumulator width is held at `--width`; the sweep is over how much of
        # it is spent on a *tied* perspective pair rather than a free single
        # accumulator. The (all free) end is the deployed model at that width.
        w = args.width
        pairs = [(w, 0), (3 * w // 4, w // 8), (w // 2, w // 4),
                 (w // 4, 3 * w // 8), (0, w // 2)]
        return [(f"tied {2 * t / w:.0%}  free={f} tied={t}x2",
                 (lambda f=f, t=t: PerspSplit(f, t, args.scale)), lr)
                for f, t in pairs]
    if args.experiment == "shrink":
        # Three ways to ease into a sparse conditioning, on the family where
        # sparsity actually bites: `kings` declares 4096 buckets and the visit
        # distribution has a perplexity of 238, so most rows see almost no data.
        #  - independent: every bucket learns only from its own positions.
        #  - shared component: add family "none", whose single row takes
        #    gradient from every position. In function space this is exactly
        #    redundant -- a constant vector the per-bucket rows could absorb --
        #    so any difference is optimisation, the same phenomenon LEDGER 017
        #    measured for the deep-linear PSQT.
        #  - curriculum: route everything to bucket 0 for a while, then copy it
        #    into every bucket and release.
        fams = tuple(f.strip() for f in args.families.split(",") if f.strip())
        mk = lambda f: (lambda: Bucketed(args.rank, args.scale, families=f, pre=True))
        return [(f"{'+'.join(fams)} independent", mk(fams), lr),
                (f"{'+'.join(fams)} + shared component", mk(("none",) + fams), lr),
                (f"{'+'.join(fams)} curriculum", mk(fams), lr),
                (f"{'+'.join(fams)} + shared component curriculum",
                 mk(("none",) + fams), lr)]
    if args.experiment == "pairs":
        # Luka's quadratic feature transformer, in the only form that is
        # deployable. A full x^T W_k x per output would need 768*768*128 = 75M
        # weights and, worse, would stream ~196 KB of table per square touched
        # on every incremental update -- a third of the entire current model per
        # move. Factorising each quadratic unit as a product of two LINEAR
        # accumulator entries costs 768*2m weights and an ordinary NNUE
        # accumulator update, with m multiplies at read time.
        #
        # `square` is the diagonal-only case a_k^2 and already lost to crelu by
        # 2.15%; a_i * a_j is strictly richer, so it gets its own arms. Width is
        # held at 128 for the like-for-like comparison (same table, same update
        # cost, only the read differs -- and the pair read then feeds l1 half as
        # many inputs, so a win there comes with FEWER parameters). The 256 arms
        # restore l1's input width so the comparison can also be made at equal
        # head size, at double the accumulator.
        mk = lambda a, w: (lambda a=a, w=w: Deep(w, args.hidden, args.scale,
                                                 families=("none",), act1=a))
        return [("crelu  w=128 (reference)",  mk("crelu", 128),    lr),
                ("square w=128 (reference)",  mk("square", 128),   lr),
                ("pairmul  w=128",            mk("pairmul", 128),  lr),
                ("pairclip w=128",            mk("pairclip", 128), lr),
                ("pairmul  w=256",            mk("pairmul", 256),  lr),
                ("pairclip w=256",            mk("pairclip", 256), lr)]
    if args.experiment == "kingcoarse":
        # Luka's objection to the post-hoc merge test, and he is right that it
        # was the wrong question. Collapsing a TRAINED 4096-bucket table asks
        # whether the learned readers are interchangeable; it does not ask
        # whether the fine granularity was needed, because a coarse model
        # trained from scratch can put a different function in each of its
        # fewer buckets. Only training the coarsenings answers that.
        #
        # `material` rides along as an in-run anchor: it is the family that beat
        # `kings` by 2.4x on the same bar, and reading the king ladder against a
        # bar from another invocation would import that invocation's noise.
        fams = [("base r=512 x1", "none"), ("kings 4^2 (x16)", "kings4"),
                ("kings 11^2 (x121)", "kings11"), ("kings 16^2 (x256)", "kings16"),
                ("kings 64^2 (x4096)", "kings"), ("material 24^2 (x576)", "material")]
        return [(n, (lambda f=f: Bucketed(args.rank, args.scale, families=(f,),
                                          pre=(f != "none"))), lr)
                for n, f in fams]
    if args.experiment == "rankbuckets":
        # Does the rank saturation survive bucketing? `persp` found 512 -> 1024
        # worth only -0.19%, but every arm there had ONE reader. If the single
        # reader is the binding constraint, extra rank has nothing to do and the
        # curve is flat for a reason that has nothing to do with rank; give the
        # model 584 readers and the same extra rank might start to pay. Two
        # curves over the same ranks, differing only in the read conditioning.
        # 768 is a hard algebraic ceiling, not an empirical one: with act=square
        # and reader R, the model is x^T (V^T diag(R) V) x, and every symmetric
        # 768x768 W has a spectral decomposition with 768 terms. r=1024 is kept
        # as a control -- it can express nothing new, so anything it wins is
        # optimisation from over-parameterisation (the LEDGER 017 effect), and
        # anything it wins by more than noise means this algebra is wrong.
        ranks = (128, 256, 512, 768, 1024)
        arms = [(f"x1  r={k}", (lambda k=k: Bucketed(k, args.scale,
                                                     families=("none",))), lr)
                for k in ranks]
        arms += [(f"x584 material+count  r={k}",
                  (lambda k=k: Bucketed(k, args.scale,
                                        families=("material", "count"),
                                        pre=True)), lr)
                 for k in ranks]
        return arms
    if args.experiment == "ranksweep":
        # Prices the capacity confound in-run. Every extra-feature arm adds
        # embedding rows, so a win could be parameters rather than the feature;
        # this curve says what parameters alone are worth under exactly these
        # conditions, instead of borrowing LEDGER 023's single 512 -> 768 point.
        return [(f"base r={k} x1", (lambda k=k: Bucketed(k, args.scale,
                                                         families=("none",))), lr)
                for k in (256, 512, 768, 1024, 2048)]
    if args.experiment == "sides":
        # Which side's pieces have to multiply which. `tied` is the deployed
        # model: one accumulator, one reader, all three blocks of W sharing a
        # coefficient. The rest either untie the blocks or delete them.
        arms = [(f"{b} r={r}", (lambda b=b: SideQuadratic(r, args.scale, blocks=b)), lr)
                for b in ("tied", "all", "same", "cross", "us", "them")]
        # What the stm-canonical orientation is worth. Same model as `tied`;
        # the only difference is that half its training positions arrive seen
        # from the other chair with the label negated, so the table has to hold
        # both orientations. Validation stays canonical in every arm.
        arms.append(("tied r=512 orientation scrambled",
                     lambda: SideQuadratic(r, args.scale), lr))
        return arms
    if args.experiment == "persp":
        # Two controls, and both are needed: the tie halves the parameters at
        # equal accumulator width, so it must beat the model with its parameter
        # count and not lose badly to the one with its width.
        return [("single r=512 (deployed)",
                 lambda: SideQuadratic(512, args.scale), lr),
                ("single r=1024 (2x params)",
                 lambda: SideQuadratic(1024, args.scale), lr),
                ("persp r=256 (0.5x params)",
                 lambda: Perspective(256, args.scale), lr),
                ("persp r=512 (1x params)",
                 lambda: Perspective(512, args.scale), lr),
                ("persp r=512 no cross",
                 lambda: Perspective(512, args.scale, cross=False), lr),
                ("antisym r=512",
                 lambda: Antisym(SideQuadratic(512, args.scale)), lr)]
    if args.experiment == "wdl":
        # The pre-registered pair for the three-way head. The trunk, the rank,
        # the bucketed PSQT, the batches and the init draw are identical; the
        # ONLY difference is whether the head emits one number or two. Both are
        # scored by `evaluate` on the cp readout against the scalar target, so
        # the ~0.018% run-to-run floor (LEDGER 044) applies to the comparison.
        # `args.scale` is K here: main sets it to the fitted K when --scale is
        # not passed, which is how every arm in this file already gets its
        # centipawn units.
        trunk = lambda: Bucketed(args.rank, args.scale, families=("none",),
                                 pfams=("material", "count"))

        def teacher_scalar():
            m = trunk()
            m.use_teacher = True
            return m

        # Three arms, because the head and the teacher are TWO manipulations
        # and a two-arm design cannot separate them. C - B is the head,
        # B - A is the teacher (measured 2x better calibrated on E, so it may
        # win on its own -- that would be a free result unrelated to WDL).
        return [("A scalar head, sigmoid teacher", trunk, lr),
                ("B scalar head, wdl teacher E", teacher_scalar, lr),
                ("C wdl head A+S", lambda: WDL(trunk(), args.scale), lr)]

    if args.experiment == "deploy":
        # The two arms of the deployment SPRT, at the rank the engine already
        # runs. Identical except for the bucketed PSQT, so the match measures
        # that and nothing else. Conditioning the LINEAR term is the one thing
        # in this study that needs NO accumulator: still 32 gathers, from a
        # 584 x 769 table instead of a 769 one.
        return [("deploy base",
                 lambda: Bucketed(args.rank, args.scale, families=("none",)), lr),
                ("deploy psqt x material+count",
                 lambda: Bucketed(args.rank, args.scale, families=("none",),
                                  pfams=("material", "count")), lr)]

    if args.experiment == "psqtbuckets":
        # Conditioning the LINEAR term instead of the read. Same 32 gathers at
        # inference, a bigger table, and a degree-3 interaction for free.
        arms = [("base r=512 x1 (bar)",
                 lambda: Bucketed(args.rank, args.scale, families=("none",)), lr)]
        for spec in args.pfams.split(","):
            spec = spec.strip()
            if not spec:
                continue
            pf = tuple(spec.split("+"))
            n = sum(FAMILIES[f][1] for f in pf)
            arms.append((f"psqt x {spec} (x{n})",
                         (lambda pf=pf: Bucketed(args.rank, args.scale,
                                                 families=("none",), pfams=pf)), lr))
        # Is a conditioned PSQT additive with a conditioned read, or the same
        # information twice? The pair below is the only way to tell.
        arms += [("read x material+count",
                  lambda: Bucketed(args.rank, args.scale,
                                   families=("material", "count"), pre=True), lr),
                 ("read x material+count + psqt x material",
                  lambda: Bucketed(args.rank, args.scale,
                                   families=("material", "count"), pre=True,
                                   pfams=("material",)), lr)]
        return arms
    if args.experiment == "extras":
        # Extra INPUT features: they enter the accumulator, so the quadratic
        # form multiplies them against every piece. A bucket cannot do that.
        fams = tuple(f.strip() for f in args.families.split(",") if f.strip())
        base = fams if fams != ("none",) else ("none",)
        tag = "+".join(base)
        arms = [(f"bar [{tag}]",
                 lambda: Bucketed(args.rank, args.scale, families=base,
                                  pre=base != ("none",)), lr)]
        for spec in args.extras.split(","):
            spec = spec.strip()
            if not spec:
                continue
            ex = tuple(spec.split("+"))
            arms.append((f"+{spec}",
                         (lambda ex=ex: Bucketed(args.rank, args.scale,
                                                 families=base,
                                                 pre=base != ("none",),
                                                 extras=ex)), lr))
        return arms
    if args.experiment == "deep":
        # A composing head on top of the same features. The quadratic form is
        # the reference, not the floor: it is what is deployed.
        fams = tuple(f.strip() for f in args.families.split(",") if f.strip())
        w, h = args.width, args.hidden
        ex = tuple(f for f in args.extras.split("+") if f and f != "none")
        return [("quadratic r=512 (reference)",
                 lambda: Bucketed(512, args.scale, families=("none",)), lr),
                (f"deep {w}x2->{h} crelu x1",
                 lambda: Deep(w, h, args.scale, families=("none",),
                              act1="crelu", extras=ex), lr),
                (f"deep {w}x2->{h} square x1",
                 lambda: Deep(w, h, args.scale, families=("none",),
                              act1="square", extras=ex), lr),
                (f"deep {w}x1->{h} crelu x1 (no perspective)",
                 lambda: Deep(w * 2, h, args.scale, families=("none",),
                              act1="crelu", persp=False, extras=ex), lr),
                (f"deep {w}x2->{h} crelu x {'+'.join(fams)}",
                 lambda: Deep(w, h, args.scale, families=fams,
                              act1="crelu", extras=ex), lr),
                (f"deep {w}x2->{h} crelu x {'+'.join(fams)} curriculum",
                 lambda: Deep(w, h, args.scale, families=fams,
                              act1="crelu", extras=ex), lr)]
    if args.experiment == "head":
        # Is the HEAD the bottleneck? Every `deep` arm ever run used h=32 --
        # 94 of them, no exceptions -- so the default was never examined.
        #
        # This is the cheap version of Luka's question. The head is where the
        # arithmetic lives, and today's bench says arithmetic is free on this
        # machine while memory is not: PST (32 lookups/eval) and the quadratic
        # (1024) benched identically, yet a 1.8 MB bucketed PSQT cost 14% nps.
        # The bucketed l2 in this very net is 576 x 1056 floats = 2.4 MB, read
        # one random 4 KB row per eval; the same arithmetic in a SHARED head is
        # 4 KB total and never leaves L1.
        #
        # So: if a wider shared head matches the bucketed narrow one, we trade a
        # 2.4 MB random-access table for registers at equal loss. If it does
        # not, the head is not starved and a richer read (attention or
        # otherwise) has no headroom to claim either.
        w = args.width
        fams = tuple(f.strip() for f in args.families.split(",") if f.strip())
        mk = lambda h, f: (lambda h=h, f=f: Deep(w, h, args.scale, families=f,
                                                 act1="crelu"))
        return [(f"deep {w}x2->32 x1 (bar)",          mk(32, ("none",)), lr),
                (f"deep {w}x2->64 x1",                mk(64, ("none",)), lr),
                (f"deep {w}x2->128 x1",               mk(128, ("none",)), lr),
                (f"deep {w}x2->256 x1",               mk(256, ("none",)), lr),
                (f"deep {w}x2->32 x {'+'.join(fams)} (bucketed bar)",
                 mk(32, fams), lr),
                (f"deep {w}x2->64 x {'+'.join(fams)}", mk(64, fams), lr)]
    if args.experiment == "wide":
        # `head` showed a SHARED h=256 head (0.016951) beating the BUCKETED h=32
        # head (0.017358) with 3.2x fewer parameters. Two things follow.
        #
        # First, this is the design that gets cheaper when we optimise. The
        # bucketed head's cost in the engine is a random 4.2 KB row out of a
        # 2.4 MB table -- memory, which quantisation barely helps. The shared
        # head's cost is arithmetic, which i8 SIMD cuts 4-8x.
        #
        # Second, bucketing a wide head is not even implementable here: the
        # per-sample weight gather is [n, h, h] regardless of bucket count,
        # which is 17 GB at h=256, n=65536. So the two are not stackable at this
        # size and the comparison above is the whole question.
        #
        # These arms SAVE, unlike `head` -- the point is to export one and play
        # it, since LEDGER 037 showed a -5.97% loss win buying 0 Elo.
        w = args.width
        ex = ("pawnfile", "pawnpair")
        return [(f"deep {w}x2->128 +pawn x1",
                 lambda: Deep(w, 128, args.scale, families=("none",), act1="crelu", extras=ex), lr),
                (f"deep {w}x2->256 +pawn x1",
                 lambda: Deep(w, 256, args.scale, families=("none",), act1="crelu", extras=ex), lr),
                (f"deep {w}x2->512 x1",
                 lambda: Deep(w, 512, args.scale, families=("none",), act1="crelu"), lr)]
    if args.experiment == "stack":
        # Luka's 2x2: {base features, +pawn features} x {no buckets, material
        # buckets}, all on the same deep head. The question is not whether
        # either helps -- both already did, separately, on two DIFFERENT model
        # families -- but whether they ADD. Buckets reweight a fixed
        # representation; extras enlarge it. If they are two routes to the same
        # information the 2x2 collapses, and section 5 of library/009 already
        # saw exactly that happen once (psqt buckets + read buckets did not
        # add).
        #
        # The two no-extras cells reproduce runs/deep.log (0.018644 and
        # 0.017410), so they are a harness check as well as a baseline:
        # `Batcher.reset()` hands every arm the identical batch stream, so all
        # four numbers are differenceable and the interaction term is readable.
        w, h = args.width, args.hidden
        ex = tuple(f for f in args.extras.split("+") if f and f != "none")
        fams = tuple(f.strip() for f in args.families.split(",") if f.strip())
        exn, bn = "+".join(ex) or "none", "+".join(fams)
        out = []
        for xl, xf in (("base", ()), (f"+{exn}", ex)):
            for bl, bf in ((" x1", ("none",)), (f" x {bn}", fams)):
                out.append((f"deep {w}x2->{h} {xl}{bl}",
                            (lambda bf=bf, xf=xf: Deep(w, h, args.scale,
                                                       families=bf,
                                                       act1="crelu",
                                                       extras=xf)), lr))
        # Not part of the 2x2. runs/pairs.log compared `pairclip w=256`
        # (0.017557) against `crelu w=128` (0.018559) and called the pair read
        # the winner, but those two arms differ in TWO things: the read, and an
        # accumulator table of twice the size. This is the missing control.
        out.append((f"deep {w * 2}x2->{h} base x1 (width control)",
                    lambda: Deep(w * 2, h, args.scale, families=("none",),
                                 act1="crelu"), lr))
        return out
    if args.experiment == "corelora":
        # One arm, on purpose. This is not a comparison -- it is a long run of
        # one design to a converged curve, so that the curve itself is the
        # deliverable (LEDGER: every ranking before this was an 8000-step
        # ranking, and 8000 steps was nowhere near converged).
        ex = tuple(args.extras.split("+")) if args.extras else ()
        fam = args.families.split(",")[0]

        def mk(w, h, f):
            return lambda: CoreLora(width=w, hidden=h, core_frac=args.core_frac,
                                    scale=args.scale, family=f, extras=ex)

        arms = [(f"corelora w{args.width} h{args.hidden} x {fam}",
                 mk(args.width, args.hidden, fam), lr)]
        if args.controls:
            # A 2x2 on (accumulator width) x (buckets), same data, same
            # batches, same steps, same curve. Width is the axis that costs
            # real nps in the engine; buckets are free to read and cost only
            # memory. The 2x2 is what separates "the buckets earned it" from
            # "any extra capacity would have earned it".
            for w in (args.width, 2 * args.width):
                for f in (fam, "none"):
                    if w == args.width and f == fam:
                        continue                    # that is the main arm
                    arms.append((f"corelora w{w} h{args.hidden} x {f}",
                                 mk(w, args.hidden, f), lr))
        return arms

    if args.experiment == "attn":
        # Luka's pairwise-attention read, against the two models he asked for:
        # the linear one it is warm started from, and the NNUE head that is the
        # current shape of "a net on top of PSQT". Nothing else -- the deployed
        # quadratic form is a legitimate bar but it is not what was asked.
        #
        # The comparison that matters is NOT loss at equal d. It is loss at
        # equal COST, and the two models spend their cost in different places:
        # `deep` spends it on a 768 x width table (memory) and a 2*width -> h
        # matvec (arithmetic, 32.8 KB of weights re-fetched every eval, which
        # library/010 measured as the single biggest item in the engine);
        # `attn` spends it on P^2 pairs of a d-vector with a table of only
        # 768 x 4d. At d=32 the tables are the same size (98k parameters) and
        # that is the honest head-to-head; d=8 is 4x smaller than anything in
        # the study so far and is here to find the floor of the curve, not to
        # win.
        ds = [int(x) for x in args.dims.split(",") if x.strip()]
        w, h = args.width, args.hidden
        arms = [("psqt 768->64->16->1", lambda: DeepLinear((64, 16), args.scale), lr),
                (f"deep {w}x2->{h} x1 (nnue style)",
                 lambda: Deep(w, h, args.scale, families=("none",),
                              act1="crelu"), lr)]
        for second in ("none", "crelu"):
            for d in ds:
                tag = "2crelu" if second == "crelu" else "1crelu"
                arms.append((f"attn d={d} {tag}",
                             (lambda d=d, s=second: PairAttn(
                                 d, args.scale, qk_std=args.qk_std,
                                 second_act=s, chunk=args.attn_chunk,
                                 ckpt=args.attn_ckpt)), lr))
        return arms
    if args.experiment == "attninit":
        # The clamp is the whole model, so where the pre-clamp products land at
        # init decides whether it starts as a rank-d bilinear form (products
        # tiny, clamp never bites) or as noise (products huge, half the pairs
        # dead and the other half saturated). Cheap sweep, short runs.
        return [(f"attn d=8 1crelu qk_std={s}",
                 (lambda s=s: PairAttn(8, args.scale, qk_std=s,
                                       chunk=args.attn_chunk)), lr)
                for s in (0.2, 0.5, 1.0, 2.0)]
    if args.experiment == "halfka_tono":
        # User asked: 2 long runs one halfka (king+psqr bar) and one tono
        # (turn-on-but-not-off latch flags) with identical inn->2x256->1
        # persp backbone, both sides like halfkp. Loss logged via curve CSV.
        return [
            ("halfka inn->2x256->1 (king+psqr)", lambda: HalfKAvsTonoMLP("halfka", args.scale), lr),
            ("tono inn->2x256->1 (latch+psqr)", lambda: HalfKAvsTonoMLP("tono", args.scale), lr),
        ]
    if args.experiment == "tree_halving":
        # psqr -> relu(acc) -> halving left+relu(a*left+b*right) until 1
        # same data/batches/scale as halfka_tono for comparability
        W = args.width if args.width >= 16 else 256
        # ensure power of 2
        import math
        p = 1 << (W - 1).bit_length()
        if p != W:
            p >>= 1
        return [(f"tree_halving psqr->{p} halving", lambda p=p: TreeHalving(p, args.scale), lr)]
    if args.experiment == "tree_halving_heads":
        W = args.width if args.width >= 16 else 256
        import math
        p = 1 << (W - 1).bit_length()
        if p != W:
            p >>= 1
        return [(f"tree_halving_heads psqr->{p} per-level", lambda p=p: TreeHalvingHeads(p, args.scale), lr)]
    if args.experiment == "rank":
        return [(f"quadratic r={k}", (lambda k=k: Quadratic(k, args.scale)), lr)
                for k in (16, 64, 256, 768)]
    if args.experiment == "magic_arms":
        # Does a more stable pawn hash train better?  One manipulation per arm:
        # the SAME model (psqr + 64-bucket PSQT delta), only the bucket fn moves.
        # `rand64` is the capacity control (same table, no chess) and appears
        # twice under different seeds -- that pair is the noise floor, which
        # this project has never measured and without which none of these
        # differences can be read.
        names = register_magics(args.magics, args.magic_bits) if args.magics else []
        names += register_rules(args.rules_npz) if args.rules_npz else []
        names += register_cross(args.cross) if args.cross else []
        names += [f.strip() for f in args.pure_fams.split(",") if f.strip()]
        for n in names:
            assert n in FAMILIES, f"--pure-fams/{n!r} not in FAMILIES"
        out = [("pure psqr only", lambda: PurePSQTBucket(args.scale, pfams=()), lr),
               ("pure psqr + rand64", lambda: PurePSQTBucket(args.scale, pfams=("rand64",)), lr),
               ("pure psqr + king6", lambda: PurePSQTBucket(args.scale, pfams=("king6",)), lr),
               ("pure psqr + rand32", lambda: PurePSQTBucket(args.scale, pfams=("rand32",)), lr),
               ("pure psqr + king32", lambda: PurePSQTBucket(args.scale, pfams=("king32",)), lr),
               ("pure psqr + castle_phase", lambda: PurePSQTBucket(args.scale, pfams=("castle_phase",)), lr)]
        for n in names:
            out.append((f"pure psqr + {n}",
                        (lambda n=n: PurePSQTBucket(args.scale, pfams=(n,))), lr))
        if args.only:
            keep = [x.strip() for x in args.only.split(",") if x.strip()]
            out = [o for o in out if any(k in o[0] for k in keep)]
            assert out, f"--only {args.only!r} matched no arm"
        # Dropping arms is safe: the seed and the batcher are reset per arm
        # below, so an arm's result does not depend on what ran before it.
        return out


    if args.experiment == "router_arms":
        # Is the bucket rule better LEARNED than SEARCHED?
        #
        # Same model as magic_arms (psqr + 64-bucket PSQT delta, same parameter
        # count in the delta table), same warm start, same batches. The only
        # thing that moves is where b(x) comes from: a magic hash, a hand rule,
        # an annealed predicate set, or gradient descent.
        #
        # `router bits pawns` is the like-for-like arm -- it reads exactly the
        # `our pawns` bitboard the annealed rules read and its rule has exactly
        # their shape (k thresholded weighted counts), so it differs from
        # `anneal` in one thing: how the rule was chosen. `router bits all` is
        # the same machinery with the whole board visible, and it CAN represent
        # king6 exactly, so losing to king6 would be an optimisation result.
        names = register_magics(args.magics, args.magic_bits) if args.magics else []
        names += register_rules(args.rules_npz) if args.rules_npz else []
        k = args.router_bits
        out = [("rand64 control",
                lambda: PurePSQTBucket(args.scale, pfams=("rand64",)), lr),
               ("king6 bar",
                lambda: PurePSQTBucket(args.scale, pfams=("king6",)), lr)]
        for n in names:
            out.append((f"fixed {n}",
                        (lambda n=n: PurePSQTBucket(args.scale, pfams=(n,))), lr))
        modes = [x.strip() for x in args.router_modes.split(",") if x.strip()]
        inps = [x.strip() for x in args.router_inputs.split(",") if x.strip()]
        gains = [float(x) for x in args.router_gain.split(",") if x.strip()]
        rlrs = [float(x) for x in args.router_lrmult.split(",") if x.strip()]
        auxs = (args.router_aux.split(";") if args.router_aux else [""])
        sfx = (lambda g, rl, a: (f" g{g:g}" if len(gains) > 1 else "")
               + (f" x{rl:g}" if len(rlrs) > 1 else "")
               + ((" [" + (a or "none") + "]") if len(auxs) > 1 else ""))
        for tm in modes:
            for inp in inps:
                for g in gains:
                    for rl in rlrs:
                        for a in auxs:
                            out.append((f"router bits{k} {tm} {inp}"
                                        + sfx(g, rl, a)
                                        + (" sym" if args.router_sym else ""),
                                        (lambda g=g, rl=rl, tm=tm, inp=inp, a=a:
                                         RoutedPSQTBucket(
                                            args.scale, nbits=k, mode="bits",
                                            train_mode=tm, inp=inp,
                                            balance=args.router_balance, gain=g,
                                            rlr_mult=rl, aux=parse_aux(a),
                                            rank=args.router_rank,
                                            rwd=args.router_wd,
                                            scalar=args.router_scalar,
                                            bn=args.router_norm,
                                            eigfloor=args.router_eigfloor,
                                            sym=args.router_sym)), lr))
        if args.router_freeze:
            for inp in inps:
                for g in gains:
                    out.append((f"router bits{k} FROZEN {inp}"
                                + (f" g{g:g}" if len(gains) > 1 else ""),
                                (lambda g=g, inp=inp: RoutedPSQTBucket(
                                    args.scale, nbits=k, mode="bits",
                                    train_mode="st", inp=inp, gain=g,
                                    freeze=True)), lr))
        if args.router_flat:
            for tm in modes:
                for inp in inps:
                    for g in gains:
                        for rl in rlrs:
                            for a in auxs:
                                out.append((f"router flat{1 << k} {tm} {inp}"
                                            + sfx(g, rl, a)
                                            + (" sym" if args.router_sym else ""),
                                            (lambda g=g, rl=rl, tm=tm, inp=inp, a=a:
                                             RoutedPSQTBucket(
                                                args.scale, nbits=1 << k,
                                                mode="flat", train_mode=tm,
                                                inp=inp,
                                                balance=args.router_balance,
                                                gain=g, rlr_mult=rl,
                                                aux=parse_aux(a),
                                                rank=args.router_rank,
                                                rwd=args.router_wd,
                                                scalar=args.router_scalar,
                                                bn=args.router_norm,
                                                eigfloor=args.router_eigfloor,
                                                sym=args.router_sym)), lr))
        if args.router_grouped:
            gg, _, hh = args.router_grouped.partition("x")
            G, H = int(gg), int(hh)
            assert G >= 2 and H >= 2, "--router-grouped wants GxH, e.g. 4x8"
            for tm in modes:
                for inp in inps:
                    for g in gains:
                        for rl in rlrs:
                            for a in auxs:
                                out.append((f"router grouped{G}x{H} {tm} {inp}"
                                            + sfx(g, rl, a)
                                            + (" sym" if args.router_sym else ""),
                                            (lambda g=g, rl=rl, tm=tm, inp=inp, a=a:
                                             RoutedPSQTBucket(
                                                args.scale, nbits=0,
                                                mode="grouped", train_mode=tm,
                                                inp=inp,
                                                balance=args.router_balance,
                                                gain=g, rlr_mult=rl,
                                                aux=parse_aux(a),
                                                rank=args.router_rank,
                                                rwd=args.router_wd,
                                                scalar=args.router_scalar,
                                                bn=args.router_norm,
                                                eigfloor=args.router_eigfloor,
                                                sym=args.router_sym,
                                                groups=G, ways=H)), lr))
        if args.only:
            keep = [x.strip() for x in args.only.split(",") if x.strip()]
            out = [o for o in out if any(x in o[0] for x in keep)]
            assert out, f"--only {args.only!r} matched no arm"
        return out

    if args.experiment == "scalar_bucket":
        # psqt + c[b(x)]: how much of a bucketed PSQT's win is just a
        # per-bucket OFFSET? `none` (nb = 1) is the floor -- plain psqt with a
        # single global bias -- and every rand{n} arm is the memorisation
        # control at that nb.
        fams = [x.strip() for x in args.scalar_fams.split(",") if x.strip()]
        return [(f"scalar {f} (nb={FAMILIES[f][1]})",
                 (lambda f=f: ScalarBucket(args.scale, fam=f)), lr)
                for f in fams]

    if args.experiment == "king_magic":
        # psqr + 4096 bucketed PSQT where bucket = king6 || magic6 (12 bits)
        # This is the adapter training: base PSQT + per-bucket delta (only relative difference)
        # Use PurePSQTBucket for pure adapter (no quadratic confound), plus Bucketed control
        return [
            ("pure psqr only", lambda: PurePSQTBucket(args.scale, pfams=()), lr),
            ("pure psqr + king_magic 4096", lambda: PurePSQTBucket(args.scale, pfams=("king_magic",)), lr),
            ("pure psqr + rand4096", lambda: PurePSQTBucket(args.scale, pfams=("rand4096",)), lr),
            ("bucketed psqr+quad 4096", lambda: Bucketed(args.rank, args.scale, families=("none",), pfams=("king_magic",)), lr),
        ]
    if args.experiment == "king_magic_halfka":
        # (rule x psqr) ->256 comparison: HalfKA bar vs king_magic 32-cluster rule (but here we use full 4096 for direct comparison first)
        # For now compare halfka vs king_magic as extra family in HalfKAvsTonoMLP style
        # We will add km32 after clustering; this experiment is the 4096 direct vs halfka
        return [
            ("halfka psqr->256", lambda: HalfKAvsTonoMLP("halfka", args.scale, hidden=256), lr),
            ("king_magic psqr->256", lambda: HalfKAvsTonoMLP("king_magic", args.scale, hidden=256), lr),
        ]
    if args.experiment == "km32_vs_halfka":
        # 32-cluster rule from king_magic adapters vs halfka, both (rule x psqr)->256
        # plus ablations: king6 only, magic6 only
        return [
            ("halfka 128 psqr->256", lambda: HalfKAvsTonoMLP("halfka", args.scale, hidden=256), lr),
            ("km32 32 psqr->256", lambda: HalfKAvsTonoMLP("km32", args.scale, hidden=256), lr),
            ("king6 64 psqr->256", lambda: HalfKAvsTonoMLP("king6", args.scale, hidden=256), lr),
            ("magic6 64 psqr->256", lambda: HalfKAvsTonoMLP("magic6", args.scale, hidden=256), lr),
        ]
    if args.experiment == "halfka_km32_5b":
        # 5B main comparison only: halfka vs km32 (for user request) — additive version
        return [
            ("halfka 128 psqr->256", lambda: HalfKAvsTonoMLP("halfka", args.scale, hidden=256), lr),
            ("km32 32 psqr->256", lambda: HalfKAvsTonoMLP("km32", args.scale, hidden=256), lr),
        ]
    if args.experiment == "rule_piece":
        # JOINT (rule,piece) vs (king,piece): 32×(768×256) vs 64×(768×256) then 256→256→1
        # This is the correct architecture user requested: rule selects piece table
        return [
            ("halfka (king,piece) 64x768x256", lambda: RulePieceMLP("halfka", args.scale, hidden=256), lr),
            ("km32 (rule,piece) 32x768x256", lambda: RulePieceMLP("km32", args.scale, hidden=256), lr),
        ]
    if args.experiment == "rule_piece_5b":
        # 5B / 2.5B long run for rule_piece
        return [
            ("halfka (king,piece) 64x768x256", lambda: RulePieceMLP("halfka", args.scale, hidden=256), lr),
            ("km32 (rule,piece) 32x768x256", lambda: RulePieceMLP("km32", args.scale, hidden=256), lr),
        ]
    return [(f"quadratic r={r}", lambda: Quadratic(r, args.scale), lr)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--experiment", default="depth",
                    choices=["depth", "scale", "ceiling", "compare",
                             "warmstart", "psqt", "rank", "single",
                             "buckets", "groups", "sides", "persp",
                             "psqtbuckets", "extras", "deep", "ranksweep", "shrink", "ratio",
                                 "rankbuckets", "pairs", "kingcoarse", "vbuck",
                                 "rules", "staged", "merged", "deploy", "wdl", "stack", "head", "wide",
                             "corelora", "attn", "attninit", "halfka_tono", "tree_halving", "tree_halving_heads",
                             "king_magic", "king_magic_halfka", "km_vs_halfka", "km32_vs_halfka", "halfka_km32_5b", "rule_piece", "rule_piece_5b", "magic_arms",
                             "router_arms", "scalar_bucket"])
    ap.add_argument("--magics", default="",
                    help="magic_arms: 'name=0xhex,...' pawn-hash buckets to compare")
    ap.add_argument("--magic-bits", type=int, default=6)
    ap.add_argument("--router-bits", type=int, default=6,
                    help="router_arms: bits in the learned rule; 2**k buckets")
    ap.add_argument("--router-modes", default="st",
                    help="router_arms: comma list of soft|st|sample")
    ap.add_argument("--router-inputs", default="all,pawns",
                    help="router_arms: comma list of all|pawns -- which feature "
                         "rows the router may read")
    ap.add_argument("--router-freeze", action="store_true",
                    help="router_arms: also run the router at its random init, "
                         "never trained -- the control that separates the rule "
                         "FAMILY from the learning of it")
    ap.add_argument("--router-scalar", action="store_true",
                    help="the adapter is ONE number per bucket instead of a "
                         "PSQT delta table per bucket: nb parameters, not "
                         "nb * 768")
    ap.add_argument("--scalar-fams", default="none,rand64,rand4096,rand65536",
                    help="--experiment scalar_bucket: which bucket families to "
                         "run as psqt + c[b(x)]")
    ap.add_argument("--router-norm", default="none",
                    choices=["none", "center", "bn", "white", "stat"],
                    help="normalise the router logits before the threshold, "
                         "all of it free at export (an affine map on the "
                         "logits folds into the rule matrix). center: subtract "
                         "the mean, so every bit splits 50/50 and none can go "
                         "constant -- the only part of BatchNorm that can "
                         "change the partition at all, since sign(z/sigma) = "
                         "sign(z). bn: full BatchNorm, for comparison. white: "
                         "ZCA, Cov^-1/2 (z - mu), which decorrelates the bits "
                         "so all 2^k sign patterns are about equally likely -- "
                         "even buckets for free, at the risk of promoting a "
                         "flat direction to a bit that flips every move")
    ap.add_argument("--router-eigfloor", type=float, default=1e-2,
                    help="--router-norm white only: floor on each covariance "
                         "eigenvalue, RELATIVE to the mean one, before the "
                         "inverse square root. It caps how far a flat "
                         "direction can be amplified, so it is a continuous "
                         "dial towards center, but NOT reaching it at 1: "
                         "clamp(min=f*mean) raises the flat directions and "
                         "leaves every direction ABOVE the floor divided by "
                         "its own sigma, so Cov^-1/2 is a single scalar (i.e. "
                         "center) only once the floor exceeds the LARGEST "
                         "eigenvalue. Measured 2026-09-03, 300 steps, 12 bits: "
                         "0 and 0.1 both give eff 3084-3092, 1 gives 1389, and "
                         "center is eff 390 -- so the center end is far above "
                         "1. flip2 did not move outside noise across that "
                         "range (n=1; the 300-step flip2 sd is ~2-4 points).")
    ap.add_argument("--router-pretrain", type=int, default=0,
                    help="train the ROUTER ALONE on its aux penalty for this "
                         "many steps, then FREEZE it and fit the tables to it. "
                         "Decouples the two gradients: the deployed rule's "
                         "stability is then set by stage 1 and stage 2 cannot "
                         "trade it away.")
    ap.add_argument("--router-pretrain-lr", type=float, default=1e-2)
    ap.add_argument("--router-merge", default="",
                    help="after training, k-means the learned bucket tables "
                         "down this ladder (e.g. '2048,1024,512,256,128,64,32,"
                         "16') and re-measure val/flip2/eff at every level. "
                         "Nested: each level clusters the previous level's "
                         "centroids. Router untouched -- only how many "
                         "distinct tables its buckets point at changes.")
    ap.add_argument("--refit-ckpt", default="",
                    help="refit a trained router arm under a MERGED table: load "
                         "this ckpt's full state into the (matching) router arm, "
                         "freeze the router, tie the delta columns per --refit-"
                         "assign and retrain the tables. That is router -> LUT "
                         "-> tables with the tables fitted to the LUT, so the "
                         "val is the TRUE merged loss, not merge_ladder's "
                         "no-refit upper bound.")
    ap.add_argument("--refit-assign", default="",
                    help="'router_json:k' -- take merge level k's `assign` "
                         "(raw bucket -> table) from that file and tie delta "
                         "columns sharing a table. Empty = freeze the router "
                         "only (control: how much of the gap is the freeze).")
    ap.add_argument("--refit-recluster", type=int, default=0,
                    help="instead of --refit-assign, re-run weighted k-means "
                         "on the LOADED (already refitted) deltas down to this "
                         "many tables, then tie and retrain. That is round 2+ "
                         "of cluster -> refit -> cluster -> refit.")
    ap.add_argument("--router-flat", action="store_true",
                    help="router_arms: also run a flat softmax over 2**k "
                           "buckets (no factorised bit structure, 2**k x more "
                           "router parameters)")
    ap.add_argument("--router-grouped", default="",
                    help="router_arms: also run a GROUPED router 'GxH' (G "
                         "H-way argmaxes, H**G raw states, e.g. '4x8' = 12 "
                         "bits in four 8-ary groups, 4096 raw states at 32 "
                         "logits). A fitted LUT (--router-merge + --refit-*) "
                         "takes it back down to the deployed table count.")
    ap.add_argument("--router-sym", action="store_true",
                    help="router_arms: tie the router weights chair-symmetric "
                           "(their piece = mirrored our piece), so the logits "
                           "are invariant under swapping the side to move. "
                           "Kills the stm-mirror flip1 component by "
                           "construction; folds out at export for free.")
    ap.add_argument("--router-aux", default="",
                    help="router_arms: semicolon-separated ARMS, each a comma "
                         "list of router penalties 'name=weight[:param]' from "
                         "info|bal|margin|swing|l1|ginfo|tv. Each arm is one more model "
                         "in the same process on the same batches, e.g. "
                         "'--router-aux \";swing=1e-3;margin=0.02:2\"' runs "
                         "no-penalty, swing and margin against each other")
    ap.add_argument("--tape", default="128x64",
                    help="router_arms: shape of the game tape the ginfo/tv "
                         "penalties read, as CHUNKSxPLIES. 128x64 = 128 random "
                         "runs of 64 consecutive plies per step. Longer runs "
                         "estimate H(B|game) better and cost a forward pass")
    ap.add_argument("--tape-stride", type=int, default=0,
                    help="router_arms: spread each tape chunk out instead of "
                         "taking clen consecutive plies. 0 = consecutive (the "
                         "shape every arm on record used). s > 0 = quadruples "
                         "[a,a+1,a+2,a+3] every s plies from a random offset, "
                         "so the `pair+2` contract is unchanged but the chunk "
                         "spans s*clen/4 plies of game. Halves the pairs per "
                         "chunk and cuts their autocorrelation")
    ap.add_argument("--router-loss-file", default="",
                    help="router_arms: a .py file defining penalty(ctx) -> "
                         "scalar, wired in as the aux term `custom`. The search "
                         "loop in nnue/losslab writes candidate losses here, so "
                         "a proposal never has to edit this file. ctx carries "
                         "the batch posterior p/z and the GAME TAPE tp/tz/gid/"
                         "cnt/pair -- exactly what ginfo and tv see")
    ap.add_argument("--router-json", default="",
                    help="write {arm: metrics} as JSON here. Same numbers the "
                         "report prints; for a caller that has to rank arms")
    ap.add_argument("--tape-train-hi", type=float, default=0.8,
                    help="fraction of the game tape the PENALTY may train on. "
                         "The flip2 probe reads the rest, so the metric a "
                         "candidate is ranked on is out-of-sample")
    ap.add_argument("--router-balance", type=float, default=0.0,
                    help="router_arms: weight on the KL of the marginal bucket "
                         "distribution to uniform; 0 = measure collapse first")
    ap.add_argument("--router-rank", type=int, default=0,
                    help="factorise the router matrix as (rows,r) @ (r,k). "
                         "0 = off. r >= k is the same function class and the "
                         "same deployed rule, only a different gradient path; "
                         "r < k is a strictly smaller family")
    ap.add_argument("--router-wd", type=float, default=0.0,
                    help="weight decay on the ROUTER parameters only. With "
                         "--router-rank this is nuclear-norm regularisation on "
                         "the product, so the rule goes low-rank")
    ap.add_argument("--router-gain", default="1",
                    help="router_arms: comma list of multipliers on the router "
                         "logits; >1 sharpens the sigmoids towards a hard rule. "
                         "At gain 1 and init sd 0.05 a 26-piece board lands "
                         "logits near 0, i.e. a nearly uniform mixture, which "
                         "is the state the soft mode has no pressure to leave")
    ap.add_argument("--router-lrmult", default="1",
                    help="router_arms: comma list of MULTIPLIERS on --lr for "
                         "the router's own parameters only. A multiplier keeps "
                         "the delta table on the same lr as every fixed-rule "
                         "arm, so the arms still differ in one thing")
    ap.add_argument("--rules-npz", default="",
                    help="magic_arms: 'name=anneal_rules.npz,...' annealed predicate rules")
    ap.add_argument("--only", default="",
                    help="magic_arms: comma-separated substrings; run only "
                         "matching arms (per-arm reseed makes this safe)")
    ap.add_argument("--cross", default="",
                    help="magic_arms: 'name=king6*rule,...' product of two families")
    ap.add_argument("--pure-fams", default="",
                    help="magic_arms: comma list of FAMILIES keys to train as "
                         "PurePSQTBucket arms (e.g. 'matsum8,count'). The six "
                         "fixed bars always run; --only still filters.")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=65536)
    # Refit on 2M positions at 30%% of all.data, held out on 2M at 70%%:
    # 3-way NLL 0.64859 against 1.20245 for the constants that used to be
    # here, which predicted a 68.7%% draw rate against an actual 49.5%% and
    # scored WORSE than the 1.04095-nat base rate. LEDGER 048 measured the
    # same refit at 0.65232 and its numbers were never wired in here.
    ap.add_argument("--wdl-a", default="[219.5003,-471.8433,164.5011,319.7684]",
                    help="cubic coefficients of a(material/58) in the teacher "
                         "WDL model. Default is Stockfish's shipped fit, which "
                         "is NOT right for this corpus -- pass the refit values")
    ap.add_argument("--wdl-b", default="[472.3219,-1141.005,998.184,-121.0777]",
                    help="cubic coefficients of b(material/58); see --wdl-a")
    ap.add_argument("--wdl-table", default="",
                    help="npy of shape (2, 40): teacher a and b per b_sym "
                         "bucket. Overrides --wdl-a/--wdl-b, and is what should "
                         "actually be passed -- the cubic misses the draw rate "
                         "by 7 points in the largest bin")
    ap.add_argument("--block", type=int, default=4_000_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--scale", type=float, default=0.0)  # 0 => use K
    ap.add_argument("--opt", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--K", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--val-cap", type=int, default=1_000_000)
    ap.add_argument("--save", default="", help="directory to write checkpoints to")
    ap.add_argument("--groups", default="",
                    help="comma-separated bucket groups, e.g. "
                         "'material+count:ro,material+count+kings:ro'")
    ap.add_argument("--families", default="bishops,kings,queens,material,count",
                    help="bucket families to try, comma separated")
    ap.add_argument("--extras", default="",
                    help="comma-separated extra-input feature sets, "
                         "each '+'-joined, e.g. 'pawnfile,pawnfile+centre'")
    ap.add_argument("--pfams", default="count,material",
                    help="comma-separated bucket families for the PSQT term")
    ap.add_argument("--width", type=int, default=128, help="deep: accumulator")
    ap.add_argument("--hidden", type=int, default=32, help="deep: hidden layer")
    ap.add_argument("--warm-buckets", type=float, default=0.0,
                    help="deep: fraction of steps routed to bucket 0 before "
                         "bucket 0 is copied into every bucket and released")
    ap.add_argument("--augment-flip", action="store_true",
                    help="train on a random half of positions seen from the "
                         "other chair, labels negated -- destroys the "
                         "stm-canonical orientation the records were built with")
    ap.add_argument("--controls", action="store_true",
                    help="corelora: also run the no-bucket control and the "
                         "double-width no-bucket control")
    ap.add_argument("--core-frac", type=float, default=0.5,
                    help="corelora: fraction of each perspective's accumulator "
                         "that feeds the SHARED core; the rest feeds the "
                         "per-bucket adapter")
    ap.add_argument("--order", default="seq", choices=["seq", "block"],
                    help="seq: every record used at most once, blocks tiled and "
                         "shuffled. block: the old random-overlapping-blocks "
                         "sampler, which repeats ~25%% of a 524M-draw run and "
                         "never visits 56%% of the corpus")
    ap.add_argument("--stride", type=int, default=1,
                    help="seq: read every Nth record, rotating the offset each "
                         "pass. Decorrelates within a pass; does NOT change how "
                         "many unique positions the run can draw")
    ap.add_argument("--data-frac", type=float, default=1.0,
                    help="use only this fraction of the training pool; the "
                         "validation tail is unchanged")
    ap.add_argument("--allow-repeat", action="store_true",
                    help="let the run lap the corpus instead of raising")
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear LR warmup steps before the cosine decay")
    ap.add_argument("--curve-dir", default="",
                    help="write a per-arm CSV of the loss curve here")
    ap.add_argument("--dims", default="8,16,32",
                    help="attn: comma-separated q/k/v/e widths to sweep")
    ap.add_argument("--qk-std", type=float, default=0.7,
                    help="attn: init sd of q and k (products ~ qk_std^2)")
    ap.add_argument("--attn-chunk", type=int, default=0,
                    help="attn: channels per pair block, 0 = all at once")
    ap.add_argument("--attn-ckpt", action="store_true",
                    help="attn: recompute pair blocks in backward")
    ap.add_argument("--init-psqt", default="",
                    help="checkpoint of a DeepLinear model to warm-start the "
                         "quadratic form's linear term from")
    args = ap.parse_args()

    router.TAPE_NCHUNK, router.TAPE_CLEN = (int(x) for x in args.tape.split("x"))
    router.TAPE_STRIDE = args.tape_stride
    router.TAPE_TRAIN_HI = args.tape_train_hi
    if args.router_loss_file:
        # Bind torch/F/math/np before the exec. A proposal that writes a bare
        # `torch.sigmoid(...)` is the common case, and an empty namespace turns
        # that into a NameError only AFTER a full startup + first step -- the
        # same cost as a real arm, for a typo. ctx carries the same names, so
        # both `torch.x` and `ctx["torch"].x` work.
        ns = {"torch": torch, "nn": nn, "F": nn.functional,
              "math": math, "np": np}
        # The caller (nnue/losslab) has already AST-gated this file. Running it
        # here rather than importing keeps it a plain function with no package.
        exec(compile(open(args.router_loss_file).read(),
                     args.router_loss_file, "exec"), ns)
        router.CUSTOM_LOSS = ns["penalty"]

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    # GameTape.sample() draws its chunk starts from the GLOBAL numpy RNG, which
    # nothing seeded before 2026-09-03. So every tape arm was unreproducible at
    # a fixed --seed: three runs of one arm at --seed 0 came out 0.028746 /
    # 0.028795 / 0.028771 (120 steps, batch 16384). That spread was on record as
    # "GPU nondeterminism"; most of it is this. Seeding it makes --seed mean
    # what it says and can only NARROW the measured noise floor, so the losslab
    # margins calibrated against the old spread stay conservative.
    np.random.seed(args.seed)
    b = Batcher(args.data, dev, batch=args.batch, block=args.block,
                seed=args.seed, val_cap=args.val_cap, order=args.order,
                stride=args.stride, allow_repeat=args.allow_repeat,
                data_frac=args.data_frac)
    print(f"{b}  device={dev}", flush=True)

    K = args.K
    if K <= 0:
        _, s, z = next(iter(b.val_batches()))
        K, nll = fit_k(s, z)
        print(f"fitted K = {K:.1f} cp  (NLL {nll:.5f}, n={len(s):,})", flush=True)
    if args.scale <= 0:
        args.scale = K
    print(f"lambda={args.lam}  opt={args.opt}  steps={args.steps}  "
          f"batch={args.batch:,}  output scale={args.scale:.1f}\n", flush=True)

    # The teacher-side WDL distribution, so a three-way arm's `lam` still means
    # what it means for the scalar arm. Stockfish's shipped constants are ~3x
    # overconfident on this corpus, so these are refit here (nnue/teacher.py).
    wdl_tab = None
    if args.wdl_table:
        wdl_tab = torch.tensor(np.load(args.wdl_table), device=dev,
                               dtype=torch.float32)
        assert wdl_tab.shape == (2, B_SYM_N), wdl_tab.shape
        print(f"teacher WDL: per-bucket table {args.wdl_table}", flush=True)
    teacher = TeacherWDL(torch.tensor(json.loads(args.wdl_a), device=dev),
                         torch.tensor(json.loads(args.wdl_b), device=dev),
                         table=wdl_tab)

    results, reports = {}, {}
    for name, factory, lr in build_arms(args):
        # Same init draw AND the same batches in the same order for every arm.
        # Without resetting both, later arms see different data and the
        # comparison silently confounds arm with data.
        torch.manual_seed(args.seed)
        b.reset()
        model = factory()
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        extra = f"  extras={model.ex}" if hasattr(model, "ex") and model.ex.names else ""
        print(f"{name}  ({n_par:,} parameters, lr={lr}){extra}", flush=True)
        # Anything that can accept a linear warm start gets one. Gating this on
        # a fixed isinstance list silently cold-started every model class added
        # after it was written, which is a confound with the arm.
        if args.init_psqt and hasattr(model, "init_psqt"):
            ck = torch.load(args.init_psqt, map_location=dev, weights_only=False)
            model.to(dev).init_psqt(ck["psqt_w"].to(dev), ck["psqt_b"].to(dev))
            print(f"  warm start from {args.init_psqt}", flush=True)

        # Both switches are per-arm so a control and its treatment can live in
        # one invocation and draw the same batches.
        # The name gate keeps the old multi-arm experiments honest (only the
        # arm called "curriculum" gets one); a single-arm run just gets it.
        warm = (args.warm_buckets
                if "curriculum" in name or args.experiment == "corelora" else 0.0)
        aug = args.augment_flip or "scrambled" in name
        safe = "".join(c if c.isalnum() or c in "-_=." else "_" for c in name)
        curve = os.path.join(args.curve_dir, safe + ".csv") if args.curve_dir else ""
        if args.router_pretrain and isinstance(model, RoutedPSQTBucket):
            model.to(dev)
            print(f"  stage 1: router alone, {args.router_pretrain} steps "
                  f"@ lr {args.router_pretrain_lr:g}", flush=True)
            model.pretrain_router(args.router_pretrain,
                                  lr=args.router_pretrain_lr,
                                  log_every=max(1, args.router_pretrain // 10))
        tie_fn, refit_assign = None, None
        if args.refit_ckpt:
            assert isinstance(model, RoutedPSQTBucket), \
                "--refit-ckpt needs a router arm"
            assert not args.router_pretrain, "refit x pretrain not combined"
            assert not model.scalar, "refit needs delta tables, not cvec"
            ck = torch.load(args.refit_ckpt, map_location=dev,
                            weights_only=False)
            model.to(dev)
            model.load_state_dict(ck["state"])
            rw, rb = ck["rule_w"], ck["rule_b"]
            mw, mb = model.router.rule_table()
            assert mw.shape == rw.shape and mb.shape == rb.shape, (
                f"refit arm {tuple(mw.shape)} != ckpt {tuple(rw.shape)}: "
                "build the same mode/bits/inputs/sym the ckpt trained")
            print(f"  refit from {args.refit_ckpt} val {ck.get('val', float('nan')):.6f}",
                  flush=True)
            for p in model.router.parameters():
                p.requires_grad_(False)
            # A frozen router cannot act on the penalty, so paying a tape
            # forward per step buys nothing (same reason pretrain empties it).
            model.aux = {}
            assert not (args.refit_assign and args.refit_recluster), \
                "one assign source: --refit-assign or --refit-recluster"
            if args.refit_recluster:
                kk = int(args.refit_recluster)
                rr0 = router_report(model, b, K, args.lam, tag="round-base")
                lv = merge_ladder(model, b, K, args.lam, [kk],
                                  rr0["counts"], seed=args.seed)
                refit_assign = torch.tensor(lv[0]["assign"], dtype=torch.long,
                                            device=dev)
                print(f"  reclustered {model.nb} raw buckets -> {kk} tables "
                      f"on the loaded deltas; router frozen, tables retrained",
                      flush=True)
            if args.refit_assign:
                jp, _, kk = args.refit_assign.partition(":")
                d = json.load(open(jp))
                m = next(iter(d.values()))
                for lvl in m.get("merge", []):
                    if int(float(lvl.get("k"))) == int(kk):
                        refit_assign = torch.tensor(
                            lvl["assign"], dtype=torch.long, device=dev)
                        break
                assert refit_assign is not None, \
                    f"no merge level {kk} in {jp}"
            if refit_assign is not None:
                k = int(refit_assign.max()) + 1
                with torch.no_grad():
                    W = model.delta.weight.detach().clone()
                    for t in range(k):
                        cols = (refit_assign == t).nonzero().flatten()
                        if len(cols):
                            W[:, cols] = W[:, cols].mean(1, keepdim=True)
                    model.delta.weight.copy_(W)
                a = refit_assign
                def tie_fn(model, _a=a):
                    with torch.no_grad():
                        W = model.delta.weight
                        for t in range(int(_a.max()) + 1):
                            cols = (_a == t).nonzero().flatten()
                            if len(cols):
                                W[:, cols] = W[:, cols].mean(1, keepdim=True)
                print(f"  tied {model.nb} raw buckets -> {k} tables; "
                      f"router frozen, tables retrained", flush=True)
            else:
                print("  router frozen, tables retrained untied (control)",
                      flush=True)
        results[name] = train(model, b, K, args.lam, args.steps, lr,
                              args.opt, args.log_every, tag=name,
                              augment=aug, warm_buckets=warm,
                              curve_path=curve, warmup=args.warmup,
                              teacher=teacher, tie_after_step=tie_fn)
        if curve:
            print(f"  curve -> {curve}", flush=True)
        rr = router_report(model, b, K, args.lam) if refit_assign is None else \
            router_report(model, b, K, args.lam, remap=refit_assign,
                          tag="refit", nstat=int(refit_assign.max()) + 1)
        # The router logit covariance spectrum, which is what --router-eigfloor
        # acts on: the floor lifts every eigenvalue below f * mean and leaves
        # the rest alone, so the spread here says how far up a floor has to go
        # before it touches the directions that carry the signal.
        bn = getattr(getattr(model, "router", None), "bn", None)
        if isinstance(bn, LogitNorm):
            ev = torch.linalg.eigvalsh(bn.rcov).clamp(min=0).flip(0)
            m = ev.mean().clamp(min=1e-12)
            r = (ev / m).tolist()
            print("  logit spectrum (eigenvalue / mean, descending):",
                  " ".join(f"{v:.3g}" for v in r), flush=True)
            print(f"  spectrum: max/mean {r[0]:.3g}  min/mean {r[-1]:.3g}"
                  f"  frac below mean {sum(1 for v in r if v < 1)}/{len(r)}",
                  flush=True)
            if rr is not None:
                rr["spectrum"] = r
        reports[name] = rr
        if args.router_merge and rr is not None and not model.scalar:
            lv = [int(x) for x in args.router_merge.split(",") if x.strip()]
            rr["merge"] = merge_ladder(model, b, K, args.lam, lv,
                                       rr["counts"], seed=args.seed)

        if args.save:
            os.makedirs(args.save, exist_ok=True)
            ck = {"arm": name, "state": model.state_dict(), "K": K,
                  "scale": args.scale, "lam": args.lam, "steps": args.steps,
                  "lr": lr, "val": results[name][-1][1]}
            if rr is not None:
                ck["router"] = rr
                ck["rule_w"], ck["rule_b"] = model.router.rule_table()
            if refit_assign is not None:
                ck["assign"] = refit_assign.detach().cpu().tolist()
            if isinstance(model, DeepLinear):
                w, bb = model.collapse()
                ck["psqt_w"], ck["psqt_b"] = w.cpu(), bb.cpu()
            path = os.path.join(args.save, safe + ".pt")
            torch.save(ck, path)
            print(f"  saved {path}", flush=True)
        print(flush=True)

    # Two yardsticks on purpose. `outcome NLL` is PRIMARY -- unbiased across
    # arms that train on different targets. `val loss` is the study's existing
    # metric (LEDGER 044's <=0.018% floor) and is biased against any arm not
    # trained on sigmoid(s/K), so it is the conservative read.
    print(f"{'arm':<32} {'outcome NLL':>13} {'best':>10} "
          f"{'val loss':>12} {'best':>10}")
    for name, h in results.items():
        print(f"{name:<32} {h[-1][2]:>13.6f} {min(o for _, _, o in h):>10.6f} "
              f"{h[-1][1]:>12.6f} {min(v for _, v, _ in h):>10.6f}")

    if args.router_json:
        out = {}
        for name, h in results.items():
            rec = {"val": h[-1][1], "best": min(v for _, v, _ in h),
                   "outcome_nll": h[-1][2],
                   "outcome_nll_best": min(o for _, _, o in h)}
            rr = reports.get(name)
            if rr is not None:
                rec.update({k: float(v) for k, v in rr.items()
                            if k not in ("counts", "merge", "spectrum")})
                rec["counts"] = [int(x) for x in rr["counts"]]
                if "spectrum" in rr:
                    rec["spectrum"] = [float(x) for x in rr["spectrum"]]
                if "merge" in rr:
                    rec["merge"] = [
                        {k: ([int(x) for x in v] if isinstance(v, list)
                             else float(v))
                         for k, v in m.items() if k != "counts"}
                        for m in rr["merge"]]
            out[name] = rec
        with open(args.router_json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
