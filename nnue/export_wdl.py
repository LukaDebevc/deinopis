"""Write a `wdlnet.WdlNet` into the engine's version-7 net file.

Only the `combo` topology is exportable, and deliberately so: the engine's
`src/wdleval.rs` implements exactly `ft -> down[fam] -> mid[fam] -> up[fam] ->
l2[fam] -> head[fam] (+ skip, + psqt)`, and an arm outside that shape has no
Rust to run in. Every constraint below is checked rather than assumed -- a net
that loads with the wrong topology evaluates plausibly and plays every game
wrong.

Two things happen here that the engine therefore never has to do:

  * `base + delta` is FOLDED, so the file holds absolute per-bucket weights.
    `code="onehot"` arms fold to the same layout with no base to add.
  * `BucketLinear` already stores `(din, dout)`, which is the input-major order
    the engine wants, so nothing is transposed except `skiph`, which is an
    `nn.Linear` and stores `(dout, din)`.

Usage:
    python3 export_wdl.py --ckpt ckpt-wdlnet/combo-10b.pt --out nets/combo-10b.nnue
"""

import argparse
import struct
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from wdlnet import WdlNet, FAM     # noqa: E402
from loader import NFEAT           # noqa: E402

MAGIC = 0x51554144
VERSION_F32 = 8
# Version 9 stores the five bucketed layers as int8 with a per-(bucket, output
# column) f32 scale, which is what LEDGER 053 priced at -0.00002 val CE on
# 409,600 held-out positions -- below the noise floor, and on the good side of
# zero. `ft` and `psqt` stay f32 in both versions: int8 on the ft table costs
# 4x every other layer combined and its per-node read is zero anyway.
VERSION_Q8 = 9
NROWS = NFEAT + 1

FAM_ID = {"none": 0, "king": 1, "mat": 2, "sym": 3}
FLAG_SKIP, FLAG_PSQT, FLAG_BN_ACT, FLAG_GATE = 1, 2, 4, 8

# The accumulator activations the engine implements. `gate` splits the
# accumulator in half and multiplies clamp(a,-1,1) by clamp(b,0,1), so the
# layer below it reads width/2 -- which is why the activation has to be in
# the file rather than implied: the same `width` means two different `down`
# shapes. `sym` and `pair` are trained arms with no Rust behind them.
ACTS = {"crelu": 0, "gate": FLAG_GATE}

# `fit_k` on the same corpus: P(win) = sigmoid(score / K). The engine reads
# `cp = K * logit(W + D/2)`, so this is what puts the WDL net's centipawns on
# the same scale every search constant was tuned against.
DEFAULT_K = 288.5


def check(cfg):
    """Refuse anything `src/wdleval.rs` does not implement."""
    bad = []
    if not cfg.get("bneck"):
        bad.append("bneck=0: the engine has no plain-L1 path for the WDL net")
    if not cfg.get("reuse", True):
        bad.append("reuse=False: the engine shares one table between the sides")
    if cfg.get("depth") != 2:
        bad.append(f"depth={cfg.get('depth')}: the engine implements depth 2")
    if cfg.get("route"):
        bad.append("route != 0: the discrete passthrough is not in the engine")
    if cfg.get("psqt") not in ("none", "plain"):
        bad.append(f"psqt={cfg.get('psqt')!r}: only 'none' and 'plain' are in the engine")
    if cfg.get("act", "crelu") not in ACTS:
        bad.append(f"act={cfg.get('act')!r}: the engine implements "
                   f"{sorted(ACTS)} only")
    if cfg.get("ft_mode", "plain") != "plain":
        bad.append(f"ft_mode={cfg.get('ft_mode')!r}: the engine reads the plain "
                   "768 table, not a king-conditioned one")
    if cfg.get("stack"):
        bad.append(f"stack={cfg.get('stack')!r}: the stacked arms are not in the engine")
    for k in ("b1", "bn_bd", "bn_bu", "bn_mid", "b2", "head_b"):
        v = cfg.get(k, "")
        if v and v not in FAM_ID:
            bad.append(f"{k}={v!r} is not a family the engine knows")
    if bad:
        raise SystemExit("cannot export this arm:\n  " + "\n  ".join(bad))


def fold(bl):
    """One `BucketLinear` as `(nb, din, dout)` weights and `(nb, dout)` biases,
    with `base + delta` already added together."""
    if bl.nb == 1:
        return bl.base.detach()[None], bl.bbias.detach()[None]
    w, b = bl.w.detach(), bl.b.detach()
    if bl.base is not None:                      # code="delta"
        w = w + bl.base.detach()
        b = b + bl.bbias.detach()
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--int8", action="store_true",
                    help="write a version-9 file: the five bucketed layers as "
                         "int8 plus per-column scales. 65.5 KB read per "
                         "evaluation becomes 16.4 KB, which fits L1D.")
    ap.add_argument("--K", type=float, default=0.0,
                    help=f"cp = K * logit(W + D/2); 0 = the checkpoint's, else {DEFAULT_K}")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    check(cfg)
    model = WdlNet(**cfg)
    model.load_state_dict(ck["state"])
    model.eval()

    K = args.K or ck.get("k_cp") or DEFAULT_K
    flags = 0
    if cfg.get("skip"):
        flags |= FLAG_SKIP
    if cfg.get("psqt", "plain") != "none":
        flags |= FLAG_PSQT
    if cfg.get("bn_act"):
        flags |= FLAG_BN_ACT
    act = cfg.get("act", "crelu")
    flags |= ACTS[act]

    d, hidden, width = cfg["bneck"], cfg["hidden"], cfg["width"]
    # What `down` actually reads. `wdlnet.WdlNet` calls this `wpost`.
    wpost = width // 2 if act == "gate" else width
    mid_fam = cfg.get("bn_mid") or ""
    layers = [
        ("down", model.bot.down_us, cfg["bn_bd"]),
        ("mid", model.bot.mid, mid_fam),
        ("up", model.bot.up_us, cfg["bn_bu"]),
        ("l2", model.l2_us, cfg["b2"]),
        ("head", model.head, cfg["head_b"]),
    ]

    buf = bytearray()
    put_u32 = lambda v: buf.extend(struct.pack("<I", int(v)))
    put_f32 = lambda v: buf.extend(struct.pack("<f", float(v)))

    def put_arr(t, shape):
        t = t.contiguous().float()
        assert tuple(t.shape) == tuple(shape), f"{tuple(t.shape)} != {tuple(shape)}"
        buf.extend(t.numpy().astype("<f4").tobytes())

    put_u32(MAGIC)
    put_u32(VERSION_Q8 if args.int8 else VERSION_F32)
    put_u32(width)
    put_u32(d)
    put_u32(hidden)
    put_u32(flags)
    put_f32(K)
    for name, mod, fam in layers:
        if name == "mid" and not fam:
            put_u32(0)          # family unused
            put_u32(0)          # 0 buckets == "this layer is absent"
        else:
            put_u32(FAM_ID[fam])
            put_u32(FAM[fam])

    put_arr(model.ft.weight.detach(), (NROWS, width))
    put_arr(model.ft_bias.detach(), (width,))
    dims = {"down": (wpost, d), "mid": (2 * d, 2 * d), "up": (2 * d, hidden),
            "l2": (2 * hidden, hidden), "head": (2 * hidden, 3)}
    for name, mod, fam in layers:
        if name == "mid" and not fam:
            continue
        w, b = fold(mod)
        di, do = dims[name]
        if args.int8:
            # Exactly `quant.fake_quant(w, 8, "col")`: symmetric, round to
            # nearest, one scale per (bucket, output column). It has to be
            # exactly that, because that is the function whose accuracy cost
            # was measured -- a different rounding rule here would be an
            # unmeasured net wearing a measured net's number.
            s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
            q = torch.round(w / s).clamp(-128, 127).to(torch.int8)
            assert tuple(q.shape) == (FAM[fam], di, do), q.shape
            buf.extend(q.contiguous().numpy().tobytes())
            put_arr(s.squeeze(1), (FAM[fam], do))
        else:
            put_arr(w, (FAM[fam], di, do))
        put_arr(b, (FAM[fam], do))
    if flags & FLAG_SKIP:
        put_arr(model.skiph.weight.detach().t(), (2 * hidden, 3))
    if flags & FLAG_PSQT:
        put_arr(model.psqt.weight.detach(), (NROWS, 3))

    with open(args.out, "wb") as f:
        f.write(buf)
    print(f"wrote {args.out}: {len(buf):,} bytes  "
          f"{'int8 layers' if args.int8 else 'f32'}  "
          f"width {width} (act {act} -> down reads {wpost}) "
          f"bneck {d} hidden {hidden} K {K:.1f} flags {flags}")
    for name, mod, fam in layers:
        nb = 0 if (name == "mid" and not fam) else FAM[fam]
        print(f"  {name:<5} fam {fam or '-':<5} buckets {nb}")


if __name__ == "__main__":
    main()
