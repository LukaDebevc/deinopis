"""Driver for the WDL-head architecture sweep (`wdlnet.py`).

Separate from `train.py` on purpose: the loss is a three-way cross-entropy, the
head is not a scalar, and `build_arms` there already dispatches 43 experiments.
Nothing here touches the live path that `test_features.py` guards.

    python3 wdlarms.py --data data/all.data \
                       --positions 500_000_000 --arms base,base-s1,nobuckets

Every arm varies ONE thing from `base`, which is the architecture as specified.
Two seeds of `base` are an arm, because without the run-to-run spread none of
the other gaps can be called.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import Batcher                                   # noqa: E402
from wdl import TeacherWDL                                   # noqa: E402
from wdlnet import mean_entropy, WdlNet, ce, expected, target_dist         # noqa: E402


BASE = dict(width=256, hidden=32, b1="king", b2="mat", head_b="none",
            reuse=True, code="delta", depth=2, psqt="plain", skip=False,
            bneck=0, bn_act=False, bn_bd="king", bn_bu="mat", bn_mid="",
            route=0, route_hard=True, route_proj=False)

_COMBO = dict(bneck=16, bn_mid="sym", head_b="mat", skip=True, lam=0.5)

ARMS = {
    "base":      {},                                  # the specified design
    "base-s1":   {"seed": 1},                         # the noise floor
    "nobuckets": {"b1": "none", "b2": "none"},        # do buckets help at all
    "nol1":      {"b1": "none"},                      # which of the two
    "nol2":      {"b2": "none"},
    "invert":    {"b1": "mat", "b2": "king"},         # material first
    "onehot":    {"code": "onehot"},                  # base+delta vs independent
    "separate":  {"reuse": False},                    # share us/them weights?
    "shallow":   {"depth": 1},                        # depth
    "deeper":    {"depth": 3},
    "w128":      {"width": 128},                      # width
    "w512":      {"width": 512},
    "h64":       {"hidden": 64},
    "nopsqt":    {"psqt": "none"},                    # the passthrough
    "psqtmat":   {"psqt": "mat"},
    "skip":      {"skip": True},
    "headmat":   {"head_b": "mat"},                   # bucket the output layer
    "headsym":   {"head_b": "sym"},
    # --- batch 2: confirm the winners, price the target, push the width ---
    "h64-s1":    {"hidden": 64, "seed": 1},
    "w512-s1":   {"width": 512, "seed": 1},
    "w512h64":   {"width": 512, "hidden": 64},
    "w1024":     {"width": 1024},
    "h128":      {"hidden": 128},
    "lam0":      {"lam": 0.0},          # pure game outcome, no teacher
    "lam05":     {"lam": 0.5},
    "lam1":      {"lam": 1.0},          # pure teacher, no outcome at all
    # --- batch 3: a discrete passthrough off the accumulator ---
    # argmax over the accumulator picks one row of a `route` x 3 logit table,
    # added to the output the way psqt is. Backward pretends it was a softmax.
    "argmax":    {"route": 256},                       # hard, straight-through
    "softrt":    {"route": 256, "route_hard": False},  # the same path, soft: is
                                                       # discreteness the point?
    "argmaxp":   {"route": 64, "route_proj": True},    # route off a 256->64 proj
    # --- batch 3: L1 as a squeeze, king down and material back up ---
    # down projection chosen by each king, the two halves concatenated, up
    # projection chosen by each side's material -- so our hidden vector still
    # sees their king. `bnecknob` is the control that removes the buckets and
    # keeps the shape; without it a win here is unattributable.
    "bneck":     {"bneck": 16},
    "bneckd8":   {"bneck": 8},
    "bneckwide": {"bneck": 64},
    "bneckact":  {"bneck": 16, "bn_act": True},        # crelu at the waist
    "bnecknob":  {"bneck": 16, "bn_bd": "none", "bn_bu": "none"},
    "bneckunt":  {"bneck": 16, "reuse": False},        # untie us/them
    "chain":     {"bneck": 16, "bn_mid": "sym"},       # three bucketed matrices
                                                       # multiplied, no activation
    # --- batch 5: WHY does a bucket help -- routing, capacity, or an offset?
    # Controls (same params, same MACs, no information in the index):
    "hashk":     {"b1": "hash64"},
    "hashm":     {"b2": "hash54"},
    # If a 2-way random split buys most of what a 54-way random split buys,
    # the gain is not the partition at all: it is the `delta` reparameterisation
    # W_b = W_base + dW[b], which under Adam moves the shared direction at
    # roughly twice the rate because base and delta are each normalised to ~lr.
    "hash2":     {"b2": "hash2"},
    # Dose-response: routing predicts monotone, capacity does not.
    "king4":     {"b1": "king4"},
    "king16":    {"b1": "king16"},
    "mat6":      {"b2": "mat6"},
    # Additive instead of multiplicative: 2 KB against 512 KB.
    "kingadd":   {"b1": "none", "kadd": "king"},
    "kingboth":  {"kadd": "king"},
    # --- batch 6: queen squares as a bucket family, and where the
    # nonlinearity actually earns its place. The eight `nl` arms are the full
    # 2^3 over the three crelu positions in
    #   psqt -> [a0] -> king layer -> [a1] -> queen layer -> [a2] -> head+psqt
    # at identical shapes, so only the nonlinearity differs. "000" is a pure
    # product of bucketed matrices: one linear map per (king, queen) pair.
    "qbuck":     {"b2": "queen"},
    "nl000":    {"stack": "kq", "b1": "king", "b2": "queen", "acts": "000"},
    "nl001":    {"stack": "kq", "b1": "king", "b2": "queen", "acts": "001"},
    "nl010":    {"stack": "kq", "b1": "king", "b2": "queen", "acts": "010"},
    "nl011":    {"stack": "kq", "b1": "king", "b2": "queen", "acts": "011"},
    "nl100":    {"stack": "kq", "b1": "king", "b2": "queen", "acts": "100"},
    "nl101":    {"stack": "kq", "b1": "king", "b2": "queen", "acts": "101"},
    "nl110":    {"stack": "kq", "b1": "king", "b2": "queen", "acts": "110"},
    "nl111":    {"stack": "kq", "b1": "king", "b2": "queen", "acts": "111"},
    # --- batch 7: 768 -> 64 -> 64 -> 64 -> 3, where does the nonlinearity pay?
    # Per side: 768 features -> ft(64) -> king layer(64) -> queen layer(64),
    # then the two sides concat to 128 and the head reads 3 logits, with the
    # psqt passthrough added as everywhere else. Same 2^3 over crelu positions
    # as the `nl` arms but at a different shape, so a conclusion that holds in
    # both is a conclusion about the nonlinearity and not about 256x32.
    "s64-000":   {"stack": "kq", "b1": "king", "b2": "queen", "acts": "000",
                 "width": 64, "hidden": 64},
    "s64-001":   {"stack": "kq", "b1": "king", "b2": "queen", "acts": "001",
                 "width": 64, "hidden": 64},
    "s64-010":   {"stack": "kq", "b1": "king", "b2": "queen", "acts": "010",
                 "width": 64, "hidden": 64},
    "s64-011":   {"stack": "kq", "b1": "king", "b2": "queen", "acts": "011",
                 "width": 64, "hidden": 64},
    "s64-100":   {"stack": "kq", "b1": "king", "b2": "queen", "acts": "100",
                 "width": 64, "hidden": 64},
    "s64-101":   {"stack": "kq", "b1": "king", "b2": "queen", "acts": "101",
                 "width": 64, "hidden": 64},
    "s64-110":   {"stack": "kq", "b1": "king", "b2": "queen", "acts": "110",
                 "width": 64, "hidden": 64},
    "s64-111":   {"stack": "kq", "b1": "king", "b2": "queen", "acts": "111",
                 "width": 64, "hidden": 64},
    # With acts="000" and head_b="none" the whole net is ONE linear map of the
    # accumulator per (king, queen) pair, and `skip` adds a second unbucketed
    # linear map of the very same vector to the logits: W_head + W_skip. That is
    # an exact reparameterisation -- identical expressivity, not one extra
    # function representable -- so any difference is Adam, not capacity. It is
    # the same structure as the `hash` puzzle with the information removed
    # entirely, which makes it the cleaner test of the two.
    "s64-000-skip": {"stack": "kq", "b1": "king", "b2": "queen", "acts": "000",
                     "width": 64, "hidden": 64, "skip": True},
    "s64-111-skip": {"stack": "kq", "b1": "king", "b2": "queen", "acts": "111",
                     "width": 64, "hidden": 64, "skip": True},
    # --- batch 8: king in the INPUT or king in the OUTPUT bucket?
    # All three are acc -> crelu -> logits with nothing in between, so the only
    # difference is where the king square is allowed to act. `flat0` is the
    # control that neither arm has a king at all -- without it a win by either
    # is unattributable to the king rather than to the extra parameters.
    #   flat0    768 -> 64 -> crelu -> 3            49k ft, 1 head
    #   flatka   64x768 -> 64 -> crelu -> 3        3.19M ft, 1 head
    #   flatkb   768 -> 64 -> crelu -> 3 per king   49k ft, 64 heads
    "flat0":   {"stack": "flat", "width": 64, "psqt": "none", "head_b": "none"},
    "flatka":  {"stack": "flat", "width": 64, "psqt": "none", "head_b": "none",
                "ft_mode": "halfka"},
    "flatkb":  {"stack": "flat", "width": 64, "psqt": "none", "head_b": "king"},
    # --- batch 9: what the accumulator activation should be.
    # All four are `combo` with only the accumulator activation changed, run at
    # lr 2e-3 where combo itself measures 0.74712 -- that run IS the clamp(0,1)
    # arm, so it is the anchor and does not need repeating.
    # `pair` halves the width, so it gets two controls rather than one: `pair`
    # doubles the accumulator to keep the downstream width at 256 (2x the ft
    # table), and `pairh` keeps the table and halves downstream instead. If
    # pairwise wins both, it is the activation and not the shape.
    "act11":     {"width": 256, "act": "sym", **_COMBO},
    "pair":      {"width": 512, "act": "pair", **_COMBO},
    "pairh":     {"width": 256, "act": "pair", **_COMBO},
    # one half signed, the other a (0,1) gate -- the shape LLMs use. `pair`
    # can only make non-negative products, so it cannot represent "this
    # feature, negatively, when that one is present"; this can.
    "gate":      {"width": 512, "act": "gate", **_COMBO},
    "gateh":     {"width": 256, "act": "gate", **_COMBO},
    # Does a king-conditioned input table pay in a net deep enough to use it?
    # Same as `gate` in every other respect, so the feature set is the only
    # difference. Mirrored to 32 king squares: 24,576 rows, not 49,152.
    "gateka":    {"width": 512, "act": "gate", "ft_mode": "halfkam", **_COMBO},
    # Halve the accumulator with the fold and double the layer that reads it,
    # so the weights actually read per evaluation stay put: 128 x 32 = 4,096
    # where combo has 256 x 16 = 4,096. Same price, different shape.
    "foldkeep":  {**_COMBO, "width": 256, "act": "gate", "bneck": 32},
    # Accumulator width was worth 0.0053 at 250M; the fold lets us buy it
    # without widening what the engine reads every evaluation.
    "gatew1024": {"width": 1024, "act": "gate", **_COMBO},
    # The cost-matched version of the above, which is the one that answers the
    # question: 1024 folds to 512, and bneck 16 -> 8 keeps the down projection
    # at 512x8 = 4,096, the same as combo's 256x16. Total 7,104 weights read
    # per evaluation against combo's 8,384 -- CHEAPER, with 4x the accumulator.
    "gatew1024b8": {**_COMBO, "width": 1024, "act": "gate", "bneck": 8},
    # Halve the WAIST instead of the accumulator. `gateh` and this arm buy a
    # similar saving two different ways: `gateh` shrinks what `down` reads,
    # this shrinks what it writes. 43.5 KB/eval against gateh's 51.5 and
    # gate's 69.5, so if it holds val it is the cheapest net on the table.
    "gateb8":    {**_COMBO, "width": 512, "act": "gate", "bneck": 8},
    # the generous control: same 512 accumulator as `pair` but clamp(0,1), so
    # it keeps all 512 downstream and has strictly more parameters everywhere.
    "act01w512": {"width": 512, **_COMBO},
    # --- batch 4: the four cheap winners stacked, which no arm above tested ---
    # chain + headmat + skip + lam05. Every one of them was measured as a
    # single change from base; together they are a PREDICTION, and the point of
    # this arm is that stacking is not additive until someone runs it.
    "combo":     {"bneck": 16, "bn_mid": "sym", "head_b": "mat", "skip": True,
                  "lam": 0.5},
    "combobig":  {"width": 1024, "bneck": 64, "bn_mid": "sym", "head_b": "mat",
                  "skip": True, "lam": 0.5},
}


def build(name, seed_base, lam_default):
    """(model kwargs, seed, lam). `seed` and `lam` are arm overrides that are
    not model arguments -- an arm may vary the target it trains on, and `ce_out`
    is scored against the game outcome either way, so those arms still rank."""
    cfg = dict(BASE)
    over = dict(ARMS[name])
    seed = seed_base + over.pop("seed", 0)
    lam = over.pop("lam", lam_default)
    cfg.update(over)
    return cfg, seed, lam


@torch.no_grad()
def evaluate(model, b, teacher, lam, temp=1.0):
    """(mixed-target CE, pure-outcome CE, binary NLL of E) on the val tail.

    `ce_out` is the primary: it is a proper scoring rule against the thing we
    actually care about and it does not depend on lam or on the teacher, so it
    ranks arms that were trained on different targets without bias. `onll` is
    the same number the scalar arms in the eval study report, so this sweep can
    be put beside them."""
    model.eval()
    mix = out = onll = 0.0
    n = 0
    for bat in b.val_batches():
        (feat, s, z), ext = bat[:3], (bat[3] if len(bat) > 3 else None)
        lg = model(feat)
        m = len(feat)
        mix += ce(lg, target_dist(s, z, teacher, feat, lam, ext, temp)).item() * m
        out += ce(lg, target_dist(s, z, teacher, feat, 0.0)).item() * m
        E = expected(lg).clamp(1e-6, 1 - 1e-6)
        onll += -(z * E.log() + (1 - z) * (1 - E).log()).mean().item() * m
        n += m
    model.train()
    return mix / n, out / n, onll / n


def log_points(steps, every):
    """Every `every` steps, plus geometric points early.

    The regular grid is what a scaling projection needs -- loss against
    positions on a fixed spacing, so an arm's curve can be extrapolated and
    compared against another's at every budget, not only at the end. The early
    geometric points cost one val pass each and separate "slow to start" from
    "short of capacity"."""
    pts = {steps}
    p = 25
    while p < every:
        pts.add(p)
        p *= 2
    pts.update(range(every, steps + 1, every))
    return sorted(p for p in pts if p <= steps)


def run(name, args, b, teacher):
    cfg, seed, lam = build(name, args.seed, args.lam)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WdlNet(**cfg).to(args.device)
    if args.init:
        # A continuation is a warm restart: weights only, fresh optimiser,
        # fresh cosine. The cfg check is the whole safety of it -- `state`
        # would load happily into a net built from a different arm as long as
        # the shapes lined up, and nothing downstream would say so.
        ck = torch.load(args.init, map_location=args.device, weights_only=False)
        assert ck["cfg"] == cfg, (
            f"--init {args.init} was arm {ck.get('arm')!r} with a different "
            f"cfg; refusing to continue it as {name!r}.\n"
            f"  ckpt: {ck['cfg']}\n  here: {cfg}")
        model.load_state_dict(ck["state"])
        print(f"init from {args.init}  (arm {ck.get('arm')!r}, "
              f"{ck.get('positions', 0):,} positions)", flush=True)
    nparam = sum(p.numel() for p in model.parameters())
    b.seed = seed
    b.reset()

    # Weight decay goes on the weight MATRICES only, never on biases -- a
    # decayed bias just shifts the whole layer toward zero for no benefit.
    #
    # Why decay at all: LEDGER 052 measured the 20B run's ft table growing from
    # |w| mean 0.144 to 0.874, which put |z| at ~4.7 and left 81% of the gate
    # units outside both clamps, where the gradient is exactly zero. The growth
    # is in the BULK, so clipping the tail does not touch it -- decay does.
    # At lr 6.7e-4 and wd 1e-2 the pullback is 6.7e-6/step against a measured
    # growth of ~1.5e-6/step.
    decay = [q for q in model.parameters() if q.dim() > 1]
    plain = [q for q in model.parameters() if q.dim() <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": plain, "weight_decay": 0.0}], lr=args.lr)
    # Warmup is a fraction of STEPS, so at a fixed position budget a 16x
    # larger batch gets 16x fewer warmup steps for the same 5M positions.
    # Adam's moment estimates settle in steps, not positions, which is why a
    # large-batch run may want this raised.
    warm = max(1, int(args.steps * args.warmup_frac))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.02, 1.0, warm),
              torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps - warm)],
        milestones=[warm])

    print(f"\n=== {name}  {cfg}  seed={seed}  lam={lam}  params={nparam:,}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    ckpt_path = f"{args.ckpt}/{name}{args.suffix}.pt"

    def save():
        """Weights, and everything `export_wdl.py` needs to rebuild the model.

        Written at every val point rather than once at the end: a 10B-position
        run is seven hours, and a run whose weights only exist after the last
        step is a run that produces a scaling curve and nothing deployable if
        anything interrupts it. Saving costs ~10 ms against a val pass that
        costs seconds."""
        torch.save({"cfg": cfg, "state": model.state_dict(), "seed": seed,
                    "lam": lam, "arm": name + args.suffix, "k_cp": args.k_cp},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
    def snapshot(step):
        """A checkpoint that is NOT overwritten, every `--ckpt-every` positions.

        `save()` writes one file and replaces it, so a run that degrades leaves
        only its worst weights behind -- which is exactly what happened to the
        20B run, whose best point (step 61,032) no longer exists on disk."""
        torch.save({"cfg": cfg, "state": model.state_dict(), "seed": seed,
                    "lam": lam, "arm": name + args.suffix, "k_cp": args.k_cp,
                    "step": step, "positions": step * b.batch},
                   f"{args.ckpt}/{name}{args.suffix}.{step * b.batch // 10**9:03d}b.pt")

    snap_every = max(1, args.ckpt_every // b.batch) if args.ckpt_every else 0
    curve = open(f"{args.out}/{name}{args.suffix}.csv", "w", buffering=1)
    curve.write("step,positions,train,ce_mix,ce_out,onll,lr,seconds\n")
    want = set(log_points(args.steps, args.log_every))
    t0 = time.time()
    last = None
    try:
        for i, bat in enumerate(b.train_batches(args.steps), 1):
            (feat, s, z), ext = bat[:3], (bat[3] if len(bat) > 3 else None)
            lg = model(feat)
            l = ce(lg, target_dist(s, z, teacher, feat, lam, ext, args.temp))
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            # Bound the ft table the way Stockfish does (its clip is 127/64 =
            # 1.98, and it clips because it quantises to int8). Non-binding on
            # a healthy net -- 0.009% of `gate-1ep`'s weights sit above it --
            # so this is a rail against runaway growth and preparation for the
            # i8 read, not a training pressure.
            if args.clip > 0:
                with torch.no_grad():
                    model.ft.weight.clamp_(-args.clip, args.clip)
                    model.ft_bias.clamp_(-args.clip, args.clip)
            sched.step()
            if snap_every and i % snap_every == 0:
                snapshot(i)
            if i in want:
                mix, out, onll = evaluate(model, b, teacher, lam, args.temp)
                dt = time.time() - t0
                last = (mix, out, onll)
                curve.write(f"{i},{i * b.batch},{l.item():.6f},{mix:.6f},"
                            f"{out:.6f},{onll:.6f},"
                            f"{opt.param_groups[0]['lr']:.3e},{dt:.1f}\n")
                save()
                print(f"  {name:<12} {i:>6}/{args.steps}  train {l.item():.5f}"
                      f"  mix {mix:.5f}  out {out:.5f}  onll {onll:.5f}"
                      f"  {i * b.batch / dt:,.0f} pos/s", flush=True)
    finally:
        curve.close()
        save()
    rec = dict(arm=name + args.suffix, cfg={k: v for k, v in cfg.items()}, seed=seed,
               params=nparam, steps=args.steps, batch=b.batch,
               positions=args.steps * b.batch, lam=lam, lr=args.lr,
               stride=b.stride, val_stride=b.val_stride,
               ce_mix=last[0], ce_out=last[1], onll=last[2],
               seconds=time.time() - t0)
    with open(f"{args.out}/results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", default="base")
    ap.add_argument("--positions", type=int, default=250_000_000)
    ap.add_argument("--steps", type=int, default=0, help="overrides --positions")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--block", type=int, default=1_000_000,
                    help="pool entries per read block; x stride records of file")
    ap.add_argument("--stride", type=int, default=8,
                    help="read every stride-th ply, so a pass spans every game")
    ap.add_argument("--val-stride", type=int, default=40,
                    help="spread the val tail over 40x the records, so the same "
                         "1M positions come from ~40x as many games")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=1.0,
                    help="temperature on the SIGMOID teacher only (>1 softer). "
                         "A control for --labels: an external teacher differs "
                         "in entropy as well as in shape, and this is how the "
                         "two are told apart.")
    ap.add_argument("--labels", default=None,
                    help="external (N,3) float16 (L,D,W) .npy, one row per "
                         "record of --data, replacing the sigmoid teacher. "
                         "Written by nnue/lc0/label.py. `ce_out` still scores "
                         "against the game outcome, so an arm trained on these "
                         "is directly comparable with one that was not.")
    ap.add_argument("--lam", type=float, default=0.9,
                    help="teacher weight in the three-way target; 0 = outcome only")
    ap.add_argument("--warmup-frac", dest="warmup_frac", type=float, default=0.02,
                    help="linear warmup as a fraction of total steps")
    ap.add_argument("--val-file", dest="val_file", default=None,
                    help="pin the validation set to this file instead of the "
                         "tail of --data, so the pool can grow without "
                         "silently changing what every number means")
    ap.add_argument("--pool-end", dest="pool_end", type=int, default=0,
                    help="train only on records [0, N); use with --val-file so "
                         "the pool cannot reach the frozen val records")
    ap.add_argument("--wd", type=float, default=0.0,
                    help="AdamW weight decay on weight matrices (not biases)")
    ap.add_argument("--clip", type=float, default=0.0,
                    help="clamp ft weights to +/- this after every step; "
                         "1.98 is Stockfish's 127/64. 0 disables")
    ap.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=0,
                    help="also write a NON-overwritten snapshot every N positions")
    ap.add_argument("--init", default=None,
                    help="start from this checkpoint's weights instead of a "
                         "fresh init. The optimiser state is NOT restored -- "
                         "Adam's moments are rebuilt over the warmup, which is "
                         "why a continuation wants one. `cfg` must match the "
                         "arm exactly; a silent shape mismatch would train a "
                         "different net under the same name.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-cap", type=int, default=1_000_000)
    ap.add_argument("--log-positions", type=int, default=10_000_000,
                    help="val pass every this many positions (scaling curve)")
    ap.add_argument("--allow-repeat", action="store_true",
                    help="let a run exceed one pass over the unique records. "
                         "The loader refuses by default because eval training "
                         "here was measured to want a single pass; a run that "
                         "uses this needs a 1-epoch control beside it.")
    ap.add_argument("--out", default="runs/wdlnet")
    ap.add_argument("--ckpt", default="ckpt-wdlnet",
                    help="where the weights go; one file per arm, rewritten at "
                         "every val point")
    ap.add_argument("--k-cp", dest="k_cp", type=float, default=288.5,
                    help="cp = K * logit(W + D/2) in the engine. `fit_k` on "
                         "this corpus; stored in the checkpoint so the "
                         "exporter cannot use a different one by accident")
    ap.add_argument("--suffix", default="", help="appended to the arm name in "
                    "the output files, so a budget ladder does not collide")
# Refit on 2M positions at 30%% of all.data, held out on 2M at 70%%:
# 3-way NLL 0.64859 against 1.20245 for the constants that used to be
# here, which predicted a 68.7%% draw rate against an actual 49.5%% and
# scored WORSE than the 1.04095-nat base rate. LEDGER 048 measured the
# same refit at 0.65232 and its numbers were never wired in here.
    ap.add_argument("--wdl-a", default="[219.5003,-471.8433,164.5011,319.7684]")
    ap.add_argument("--wdl-b", default="[472.3219,-1141.005,998.184,-121.0777]")
    args = ap.parse_args()
    if not args.steps:
        args.steps = args.positions // args.batch
    args.log_every = max(1, args.log_positions // args.batch)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(args.device)
    b = Batcher(args.data, dev, batch=args.batch, block=args.block,
                val_cap=args.val_cap, seed=args.seed, order="seq",
                stride=args.stride, val_stride=args.val_stride,
                allow_repeat=args.allow_repeat, val_file=args.val_file,
                pool_end=args.pool_end, labels=args.labels)
    print(b, flush=True)
    # The refit cubic teacher of LEDGER 048. Stockfish's shipped constants are
    # 1.84x worse in NLL on this corpus and are not an option.
    teacher = TeacherWDL(torch.tensor(json.loads(args.wdl_a), device=dev),
                         torch.tensor(json.loads(args.wdl_b), device=dev))
    # The teacher's OWN score on the val set, printed before any arm runs.
    # A target nobody scores is a target nobody can trust: the first version of
    # this sweep trained every arm against a distribution that was worse than
    # the base rate, and nothing in the output said so.
    tot = te = ent = m = 0
    with torch.no_grad():
        for bat in b.val_batches():
            (feat, s_, z_), ext = bat[:3], (bat[3] if len(bat) > 3 else None)
            tgt = target_dist(s_, z_, teacher, feat, 1.0, ext, args.temp)
            oh = target_dist(s_, z_, teacher, feat, 0.0)
            tot += ce(tgt.clamp(min=1e-7).log(), oh).item() * len(feat)
            ent += mean_entropy(tgt).item() * len(feat)
            E = (tgt[:, 2] + tgt[:, 1] / 2).clamp(1e-6, 1 - 1e-6)
            te += -(z_ * E.log() + (1 - z_) * (1 - E).log()).mean().item() * len(feat)
            m += len(feat)
    print(f"teacher alone on val: ce_out {tot / m:.5f}  onll {te / m:.5f}"
          f"  H {ent / m:.5f}   (base-rate entropy is 1.04095)", flush=True)

    rows = []
    for name in args.arms.split(","):
        name = name.strip()
        if name:
            rows.append(run(name, args, b, teacher))
    print(f"\n{'arm':<12} {'params':>12} {'ce_out':>10} {'onll':>10} {'ce_mix':>10}")
    for r in sorted(rows, key=lambda r: r["ce_out"]):
        print(f"{r['arm']:<12} {r['params']:>12,} {r['ce_out']:>10.5f}"
              f" {r['onll']:>10.5f} {r['ce_mix']:>10.5f}")


if __name__ == "__main__":
    main()
