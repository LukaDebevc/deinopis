"""Quantisation-aware fine-tuning of a trained WDL net.

Round-to-nearest is a guess about which weights matter; this measures it. The
net is fine-tuned with the target tensors passed through a straight-through
fake quantiser -- forward sees `Q(w)`, backward sees the identity -- so the
weights move to wherever the *quantised* net is best, rather than to wherever
the fp32 net was best and then getting rounded.

Two objectives, and they answer different questions:

  `--distill 0`  train on the real target (the LEDGER 048 teacher mixed with
                 the game outcome, exactly as `wdlarms.py` does). The quantised
                 net is then free to end up BETTER than the fp32 one, which it
                 can, because 8 bits is not the binding constraint at this
                 width -- the data is.
  `--distill 1`  match the fp32 net's own logits (KL). This is the AWQ/GPTQ
                 objective: minimise the disagreement with the teacher network
                 rather than the loss. It needs no labels and it is what to use
                 when the fp32 net is the thing being replaced and any drift
                 from it is a risk, not an improvement.

The scale is recomputed from the current weights every forward. That is a
moving target for the optimiser, but the alternative -- freezing the fp32 net's
scale -- pins the clipping range to weights that are about to move, and with
`col` granularity the scale is a single max per output column and moves very
little in practice.

`base + delta` is folded before anything and the base is frozen at zero, so the
tensor being quantised is exactly what `export_wdl.py` writes.

    python3 qat.py --ckpt ckpt-wdlnet/gate-3ep.pt --bits ft=8 --steps 3000 \
        --data data/all.data \
        --val-file data/val_frozen.data \
        --pool-end 853500985 --out ckpt-wdlnet/gate-3ep-q8.pt
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import Batcher                                    # noqa: E402
from wdl import TeacherWDL                                    # noqa: E402
from wdlnet import ce, target_dist                            # noqa: E402
from quant import (fold_inplace, layer_specs, fake_quant,     # noqa: E402
                   load, measure, cp_of, val_cache, with_ref, parse_bits)


class FakeQuant(nn.Module):
    """`w -> w + (Q(w) - w).detach()`: the straight-through estimator."""

    def __init__(self, bits, mode, group, clip=0.0):
        super().__init__()
        self.bits, self.mode, self.group, self.clip = bits, mode, group, clip

    def forward(self, w):
        dq, _, _ = fake_quant(w, self.bits, self.mode, self.group, self.clip)
        return w + (dq - w).detach()


def attach(model, spec, mode, group, clip=0.0):
    """Parametrise the named tensors, and freeze the folded-away base."""
    names = dict((n, m) for n, m, _ in layer_specs(model))
    live = []
    for name, bits in spec.items():
        if not bits:
            continue
        mod = names[name]
        if isinstance(mod, torch.Tensor):                 # ft / psqt weight
            owner = model.ft if name == "ft" else model.psqt
            attr = "weight"
        elif hasattr(mod, "nb"):
            owner, attr = mod, ("base" if mod.nb == 1 else "w")
        else:                                             # skiph, an nn.Linear
            owner, attr = model.skiph, "weight"
        P.register_parametrization(owner, attr,
                                   FakeQuant(bits, mode, group,
                                             clip if name == "ft" else 0.0))
        live.append(f"{name}@int{bits}")
    for _, mod, _ in layer_specs(model):
        if hasattr(mod, "nb") and mod.nb > 1 and mod.base is not None:
            mod.base.requires_grad_(False)
            mod.bbias.requires_grad_(False)
    return live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/all.data")
    ap.add_argument("--val-file", dest="val_file",
                    default="data/val_frozen.data")
    ap.add_argument("--pool-end", dest="pool_end", type=int, default=853_500_985)
    ap.add_argument("--bits", required=True, help="e.g. ft=8,down=8")
    ap.add_argument("--mode", default="col",
                    choices=["col", "row", "group", "bucket", "tensor"])
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--clip", type=float, default=0.0,
                    help="hard-clip the ft table inside the quantiser, so the "
                         "net TRAINS against the clipped range instead of "
                         "having it imposed afterwards")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--block", type=int, default=1_000_000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--distill", type=float, default=0.0,
                    help="0 = the real target, 1 = KL to the fp32 net's logits")
    ap.add_argument("--val-cap", type=int, default=400_000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wdl-a", default="[219.5003,-471.8433,164.5011,319.7684]")
    ap.add_argument("--wdl-b", default="[472.3219,-1141.005,998.184,-121.0777]")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ref_model, ck = load(args.ckpt, dev)
    fold_inplace(ref_model)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    model, _ = load(args.ckpt, dev)
    fold_inplace(model)
    live = attach(model, parse_bits(args.bits), args.mode, args.group, args.clip)

    cache = val_cache(args.val_file, dev, args.batch, args.val_cap)
    ref_cp = cp_of(ref_model, cache)
    cache = with_ref(cache, ref_cp)
    teacher = TeacherWDL(torch.tensor(json.loads(args.wdl_a), device=dev),
                         torch.tensor(json.loads(args.wdl_b), device=dev))

    ce_fp, _, _, _ = measure(ref_model, cache)
    print(f"{args.ckpt}: fp32 ce_out {ce_fp:.6f} on {len(ref_cp):,} positions")
    print(f"quantised: {' '.join(live)}  mode={args.mode} group={args.group}")
    c, rmse, p99, mae = measure(model, cache)
    print(f"  step      0   ce {c:.6f} ({c - ce_fp:+.5f})   "
          f"cp rmse {rmse:7.2f}  mae {mae:6.2f}  p99 {p99:7.2f}   [RTN]")

    b = Batcher(args.data, dev, batch=args.batch, block=args.block,
                seed=args.seed, order="seq", stride=args.stride,
                val_file=args.val_file, pool_end=args.pool_end, val_cap=1)
    train = [q for q in model.parameters() if q.requires_grad]
    decay = [q for q in train if q.dim() > 1]
    plain = [q for q in train if q.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": plain, "weight_decay": 0.0}], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    best = (rmse if args.distill > 0 else c, 0)
    t0 = time.time()
    for i, (feat, s, z) in enumerate(b.train_batches(args.steps), 1):
        lg = model(feat)
        loss = 0.0
        if args.distill < 1.0:
            loss = loss + (1 - args.distill) * ce(
                lg, target_dist(s, z, teacher, feat, args.lam))
        if args.distill > 0.0:
            with torch.no_grad():
                tgt = F.softmax(ref_model(feat), -1)
            loss = loss + args.distill * ce(lg, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if i % args.every == 0 or i == args.steps:
            c, rmse, p99, mae = measure(model, cache)
            key = rmse if args.distill > 0 else c
            mark = ""
            if key < best[0]:
                best, mark = (key, i), " *"
            print(f"  step {i:6d}   ce {c:.6f} ({c - ce_fp:+.5f})   "
                  f"cp rmse {rmse:7.2f}  mae {mae:6.2f}  p99 {p99:7.2f}   "
                  f"{time.time() - t0:5.0f}s{mark}", flush=True)

    # Bake the quantised values in: `leave_parametrized` writes Q(w) into the
    # plain parameter, so what is saved is exactly what an integer exporter
    # would read, and `quant.py --bits ...` on the result is a no-op.
    for owner, attr in _parametrized(model):
        P.remove_parametrizations(owner, attr, leave_parametrized=True)
    torch.save({"cfg": ck["cfg"], "state": model.state_dict(),
                "seed": args.seed, "lam": args.lam,
                "arm": ck.get("arm", "") + "-qat", "k_cp": ck.get("k_cp", 288.5),
                "quant": {"bits": args.bits, "mode": args.mode,
                          "group": args.group, "clip": args.clip, "steps": args.steps,
                          "distill": args.distill, "from": args.ckpt}},
               args.out)
    c, rmse, p99, mae = measure(model, cache)
    print(f"wrote {args.out}   ce {c:.6f} ({c - ce_fp:+.5f})  "
          f"cp rmse {rmse:.2f}  mae {mae:.2f}  p99 {p99:.2f}")


def _parametrized(model):
    out = []
    for mod in model.modules():
        if P.is_parametrized(mod):
            out += [(mod, k) for k in list(mod.parametrizations.keys())]
    return out


if __name__ == "__main__":
    main()
