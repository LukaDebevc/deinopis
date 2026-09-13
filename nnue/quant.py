"""Weight quantisation for the WDL net, and the harness that prices it.

**Why weight-only, and why footprint.** LEDGER 040 measured that the integer
*arithmetic* speed-up is invisible under real memory pressure while the
*footprint* cut is not: `down` alone is 32.8 KB of f32 read per evaluation and
Zen 3's L1D is 32 KB. So the thing being bought here is bytes, not multiplies,
and that makes the first cut of the problem much smaller than a full integer
pipeline:

  * weights are quantised, activations stay f32;
  * the accumulation stays f32, so there is no overflow question at all and no
    clipping range to fit -- the `-32..32` worry does not arise until we go for
    the arithmetic too (see `ACTIVATIONS` at the bottom);
  * dequantisation is one multiply per output, `z[k] = s[k]*acc[k] + b[k]`,
    which is free next to a `din x dout` matvec.

**Everything is measured against the folded net.** `BucketLinear` stores
`base + delta`; the engine stores the sum. Quantising the two halves separately
would quantise a parameterisation the engine never sees, so `fold_inplace`
collapses them first and the folded net is the fp32 reference for every number
here. Folding is exact and `--check-fold` asserts it.

Granularity, in the order it costs the engine nothing:
  `col`    one scale per (bucket, output column) -- the default. Dequantise at
           the end of the matvec; the scale vector is `dout` floats.
  `group`  additionally split the INPUT axis into groups of `g`, one scale per
           (bucket, group, column). Costs a partial-sum flush every `g` inputs.
  `bucket` one scale per bucket. `tensor` one for the whole layer.

Usage:
    python3 quant.py --ckpt ckpt-wdlnet/gate-3ep.pt --sweep
    python3 quant.py --ckpt ... --bits down=8,l2=8,ft=8 --out ckpt-wdlnet/q.pt
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import Batcher, PAD                      # noqa: E402
from wdlnet import WdlNet, expected, ce, target_dist  # noqa: E402
from wdl import TeacherWDL                      # noqa: E402


# The tensors the engine reads, in the order it reads them. `per_eval` is the
# bytes ONE evaluation touches at f32: two buckets for every per-side layer,
# because the two perspectives pick different rows. `ft` is not in that column
# because the accumulator is incremental -- its cost is the table's residency,
# not a per-node read.
def layer_specs(model):
    b = model.bot
    out = [("ft", model.ft.weight, 0),
           ("down", b.down_us, 2),
           ("mid", b.mid, 2),
           ("up", b.up_us, 2),
           ("l2", model.l2_us, 2),
           ("head", model.head, 1)]
    if model.psqt_mode != "none":
        out.append(("psqt", model.psqt.weight, 0))
    if model.skip:
        out.append(("skiph", model.skiph.weight, 1))
    return [(n, m, k) for n, m, k in out if m is not None]


def fold_inplace(model):
    """Collapse `base + delta` into absolute per-bucket weights.

    Exact: `forward` computes `w + base` when `base is not None`, so moving the
    base into `w` and zeroing it changes nothing. After this the tensor being
    quantised is byte-for-byte what `export_wdl.fold` writes into the net file.
    """
    for _, mod, _ in layer_specs(model):
        if not hasattr(mod, "nb") or mod.nb == 1 or mod.base is None:
            continue
        with torch.no_grad():
            mod.w += mod.base
            mod.b += mod.bbias
            mod.base.zero_()
            mod.bbias.zero_()
    return model


def weight_of(mod):
    """The tensor a spec entry quantises, and a setter for it."""
    if isinstance(mod, torch.Tensor):
        return mod
    return mod.base if mod.nb == 1 else mod.w


# ------------------------------------------------------------------ quantise

def fake_quant(w, bits, mode="col", group=0, clip=0.0):
    """Symmetric round-to-nearest, returned dequantised. Also returns the
    integer tensor and the scales, so a caller can write a real file.

    The input axis is the SECOND-to-last for a bucketed weight `(nb, din, dout)`
    and for a plain `(din, dout)`; the embedding tables are `(rows, width)` and
    are quantised per output column too, which for `ft` means one scale per
    accumulator unit -- the natural choice, because a unit's scale is a free
    reparameterisation the next layer's column absorbs exactly.
    """
    qmax = 2 ** (bits - 1) - 1
    if clip:
        w = w.clamp(-clip, clip)
    x = w if w.dim() == 3 else w[None]                 # (nb, din, dout)
    nb, din, dout = x.shape
    if mode == "group" and group and din % group == 0:
        x = x.view(nb, din // group, group, dout)
        red = 2
    elif mode == "col":
        x = x.view(nb, 1, din, dout)
        red = 2
    elif mode == "row":
        # One scale per INPUT row, shared across the output columns. For `ft`
        # this is one scale per (piece, square), which is the granularity an
        # accumulator update can actually use: `acc[k] += s[f] * q[f][k]` is one
        # broadcast multiply per added feature, against `col`'s scale vector
        # which cannot be applied until the sum is finished -- and the sum is
        # never finished, because the accumulator is incremental.
        x = x.view(nb, din, dout, 1)
        red = 2
    elif mode == "bucket":
        x = x.view(nb, 1, din * dout, 1)
        red = 2
    elif mode == "tensor":
        x = x.view(1, 1, nb * din * dout, 1)
        red = 2
    else:
        raise ValueError(f"{mode}/{group} does not divide din={din}")
    s = x.abs().amax(dim=red, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.round(x / s).clamp(-qmax - 1, qmax)
    return (q * s).view(w.shape), q, s


def apply_spec(model, spec, mode="col", group=0):
    """`spec` maps layer name -> bits. Missing or 0 means leave it in f32."""
    names = dict((n, m) for n, m, _ in layer_specs(model))
    with torch.no_grad():
        for name, bits in spec.items():
            if not bits:
                continue
            if name not in names:
                raise KeyError(f"{name}: not a layer of this net "
                               f"({sorted(names)})")
            w = weight_of(names[name])
            dq, _, _ = fake_quant(w.data, bits, mode, group)
            if name == "ft":
                dq[PAD].zero_()      # the padding row is held at zero
            w.data.copy_(dq)
    return model


# -------------------------------------------------------------------- metrics

@torch.no_grad()
def measure(model, batches, ref=None, lam=0.0, teacher=None):
    """(outcome CE, cp RMSE vs `ref`, cp p99 abs error, mean |cp|).

    Two numbers because they answer different questions. The CE is what the
    sweep is scored on elsewhere in this repo and is comparable with every val
    number on record. The cp error against the *fp32 net itself* is the one
    that matters for a drop-in replacement: it is the disagreement the search
    would see, with the label noise divided out.
    """
    model.eval()
    tot = n = 0.0
    de = []
    for feat, s, z, cpref in batches:
        lg = model(feat)
        tot += ce(lg, target_dist(s, z, teacher, feat, lam)).item() * len(feat)
        n += len(feat)
        E = expected(lg).clamp(1e-6, 1 - 1e-6)
        cp = model.k_cp * torch.log(E / (1 - E))
        if cpref is not None:
            de.append(cp - cpref)
    if not de:
        return tot / n, None, None, None
    d = torch.cat(de)
    return (tot / n, d.pow(2).mean().sqrt().item(),
            d.abs().quantile(0.99).item(), d.abs().mean().item())


@torch.no_grad()
def cp_of(model, batches):
    out = []
    for feat, _, _, _ in batches:
        E = expected(model(feat)).clamp(1e-6, 1 - 1e-6)
        out.append(model.k_cp * torch.log(E / (1 - E)))
    return torch.cat(out)


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = WdlNet(**ck["cfg"]).to(device)
    m.load_state_dict(ck["state"])
    m.k_cp = ck.get("k_cp") or 288.5
    m.eval()
    return m, ck


def footprint(model):
    """Bytes per layer at f32, and the per-evaluation read."""
    rows = []
    for name, mod, k in layer_specs(model):
        w = weight_of(mod)
        nb = w.shape[0] if w.dim() == 3 else 1
        per = w.numel() // nb
        rows.append((name, w.numel(), per * 4 * k))
    return rows


# ---------------------------------------------------------------------- main

VAL = "data/val_frozen.data"


def val_cache(path, device, batch, cap):
    """The frozen val set, unpacked once and kept on the GPU."""
    b = Batcher(path, device, batch=batch, val_file=path, pool_end=1,
                val_cap=cap, val_frac=1.0)
    out = []
    seen = 0
    for feat, s, z in b.val_batches():
        out.append((feat, s, z, None))
        seen += len(feat)
        if seen >= cap:
            break
    return out


def with_ref(cache, cp):
    i = 0
    out = []
    for feat, s, z, _ in cache:
        out.append((feat, s, z, cp[i:i + len(feat)]))
        i += len(feat)
    return out


def parse_bits(s):
    d = {}
    for part in s.split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        d[k.strip()] = int(v)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--val-cap", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bits", default="", help="e.g. down=8,l2=8")
    ap.add_argument("--mode", default="col",
                    choices=["col", "row", "group", "bucket", "tensor"])
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="one layer at a time at 8/6/4 bits, then all together")
    ap.add_argument("--check-fold", action="store_true")
    ap.add_argument("--out", default="",
                    help="write the quantised weights as a checkpoint "
                         "`export_wdl.py` can read. Weight-only quantisation "
                         "dequantises to f32, so the v8 f32 net file exported "
                         "from this plays EXACTLY the quantised net -- which is "
                         "how the accuracy cost gets an SPRT before any integer "
                         "kernel exists to be wrong.")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model, ck = load(args.ckpt, dev)
    cache = val_cache(args.val, dev, args.batch, args.val_cap)
    n = sum(len(f) for f, _, _, _ in cache)

    ce0, _, _, _ = measure(model, cache)
    fold_inplace(model)
    ce1, _, _, _ = measure(model, cache)
    print(f"{args.ckpt}  n={n:,}  fp32 ce_out {ce0:.6f}   "
          f"folded {ce1:.6f}   drift {ce1 - ce0:+.2e}")
    if args.check_fold:
        assert abs(ce1 - ce0) < 1e-6, "folding is not exact"

    ref_cp = cp_of(model, cache)
    cache = with_ref(cache, ref_cp)
    ref = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print(f"reference |cp| mean {ref_cp.abs().mean():.1f}  "
          f"p99 {ref_cp.abs().quantile(0.99):.1f}")

    print("\nlayer     params      f32 B/eval")
    tot = 0
    for name, npar, per in footprint(model):
        tot += per
        print(f"  {name:<7} {npar:>9,}  {per / 1024:>8.1f} KB")
    print(f"  {'TOTAL':<7} {'':>9}  {tot / 1024:>8.1f} KB per evaluation")

    def run(spec, label):
        model.load_state_dict(ref)
        apply_spec(model, spec, args.mode, args.group)
        c, rmse, p99, mae = measure(model, cache)
        print(f"  {label:<28} ce {c:.6f} ({c - ce1:+.5f})   "
              f"cp rmse {rmse:7.2f}  mae {mae:6.2f}  p99 {p99:7.2f}")
        return c, rmse

    names = [n for n, _, _ in footprint(model)]
    if args.sweep:
        print(f"\none layer at a time  (mode={args.mode} group={args.group})")
        for bits in (8, 6, 4):
            for name in names:
                run({name: bits}, f"{name} @ int{bits}")
            print()
        for bits in (8, 6, 4):
            run({k: bits for k in names}, f"EVERYTHING @ int{bits}")
    if args.bits:
        print()
        run(parse_bits(args.bits), args.bits)
        if args.out:
            torch.save({"cfg": ck["cfg"], "state": model.state_dict(),
                        "seed": 0, "lam": ck.get("lam", 0.5),
                        "arm": ck.get("arm", "") + "-rtn",
                        "k_cp": ck.get("k_cp", 288.5),
                        "quant": {"bits": args.bits, "mode": args.mode,
                                  "group": args.group, "rtn": True,
                                  "from": args.ckpt}}, args.out)
            print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
