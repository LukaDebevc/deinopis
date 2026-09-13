"""Write a dual-tail net file (version 10): one accumulator, two tails.

The rung-3 deployment format (LEDGER 080): the main search runs the big
tail, quiescence behind `--dual` runs the small one, both reading the SAME
incrementally maintained accumulator — which is why the two checkpoints
must share width/activation/hidden and every bucket family, and why the
file stores one `ft`/`ft_bias` and two tails. The tier is picked per node
through the existing `set_small` seam; without `--dual` the file behaves
exactly like its big tail alone.

Layout (all little-endian):

    magic, ver=10,
    width, hidden, k_cp,                                  # shared head
    ft, ft_bias,                                          # f32, shared
    big d, flags, 5x(fam, nb), down, [mid], up, l2, head, [skiph], [psqt],
    small d, flags, 5x(fam, nb), down, [mid], up, l2, head, [skiph], [psqt]

Both tails are int8 (version 9's scheme, per-column scales): LEDGER 053
priced that at -0.00002 val CE, and the SPRT prices the deployed thing.
There is no f32 dual format; the single-tail v8 file is the control arm.

Every sharing constraint is checked rather than assumed — a file whose
tails read different accumulators would evaluate plausibly and play every
game wrong:

    nnue/.venv/bin/python nnue/export_dual.py \
        --big nnue/runs/b1-20260909/m1-b1.pt \
        --small nnue/runs/distill/tail-b8-warm.pt \
        --out nnue/runs/distill/dual-b8.nnue
"""

import argparse
import struct
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from wdlnet import WdlNet, FAM                              # noqa: E402
from loader import NFEAT                                    # noqa: E402
from export_wdl import (fold, FAM_ID, FLAG_SKIP, FLAG_PSQT,  # noqa: E402
                        FLAG_BN_ACT, FLAG_GATE, MAGIC, DEFAULT_K)

VERSION_DUAL_Q8 = 10
NROWS = NFEAT + 1

# Bits that change shapes or the code path: the tails must agree on all of
# them, since they share one accumulator and one forward pass. SKIP/PSQT
# only add a term and may differ per tail.
SHAPE_FLAGS = FLAG_BN_ACT | FLAG_GATE


def flags_of(cfg):
    f = 0
    if cfg.get("skip"):
        f |= FLAG_SKIP
    if cfg.get("psqt", "plain") != "none":
        f |= FLAG_PSQT
    if cfg.get("bn_act"):
        f |= FLAG_BN_ACT
    if cfg.get("act", "crelu") == "gate":
        f |= FLAG_GATE
    return f


def check_pair(big, small):
    """Refuse anything the engine cannot run as one accumulator + two tails."""
    bad = []
    for k in ("width", "hidden", "act", "b1", "bn_bd", "bn_bu", "bn_mid",
              "b2", "head_b", "psqt", "depth", "reuse", "route", "stack",
              "ft_mode"):
        if big.get(k) != small.get(k):
            bad.append(f"{k}: big={big.get(k)!r} small={small.get(k)!r}")
    if not big.get("bneck"):
        bad.append("big bneck=0: the engine has no plain-L1 path")
    if not small.get("bneck"):
        bad.append("small bneck=0: the engine has no plain-L1 path")
    if big.get("bneck") == small.get("bneck"):
        bad.append("same waist: a dual file with identical tails is a bug")
    if bad:
        raise SystemExit("tails cannot share one accumulator:\n  " + "\n  ".join(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", required=True)
    ap.add_argument("--small", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=float, default=0.0,
                    help="0 = the big checkpoint's K (both must agree)")
    args = ap.parse_args()

    bck = torch.load(args.big, map_location="cpu", weights_only=False)
    sck = torch.load(args.small, map_location="cpu", weights_only=False)
    check_pair(bck["cfg"], sck["cfg"])
    big, small = WdlNet(**bck["cfg"]), WdlNet(**sck["cfg"])
    big.load_state_dict(bck["state"])
    small.load_state_dict(sck["state"])
    big.eval()
    small.eval()

    kb = bck.get("k_cp") or DEFAULT_K
    ks = sck.get("k_cp") or DEFAULT_K
    if abs(kb - ks) > 1e-9:
        raise SystemExit(f"K differs: big {kb} small {ks}")
    K = args.K or kb

    # The distilled tail trained its own psqt from a warm start; the file
    # keeps each tail's own tables, which is exactly what `verify_wdl.py`
    # checked on the standalone exports.
    if not torch.equal(big.ft.weight, small.ft.weight) or \
            not torch.equal(big.ft_bias, small.ft_bias):
        raise SystemExit("ft tables differ: not one accumulator, refusing")

    buf = bytearray()
    put_u32 = lambda v: buf.extend(struct.pack("<I", int(v)))
    put_f32 = lambda v: buf.extend(struct.pack("<f", float(v)))

    def put_arr(t, shape):
        t = t.contiguous().float()
        assert tuple(t.shape) == tuple(shape), f"{tuple(t.shape)} != {tuple(shape)}"
        buf.extend(t.numpy().astype("<f4").tobytes())

    def put_tail(model, cfg):
        d, hidden, width = cfg["bneck"], cfg["hidden"], cfg["width"]
        wpost = width // 2 if cfg.get("act") == "gate" else width
        mid_fam = cfg.get("bn_mid") or ""
        layers = [
            ("down", model.bot.down_us, cfg["bn_bd"]),
            ("mid", model.bot.mid, mid_fam),
            ("up", model.bot.up_us, cfg["bn_bu"]),
            ("l2", model.l2_us, cfg["b2"]),
            ("head", model.head, cfg["head_b"]),
        ]
        put_u32(d)
        put_u32(flags_of(cfg))
        for _, mod, fam in layers:
            if mod is None:
                put_u32(0)
                put_u32(0)
            else:
                put_u32(FAM_ID[fam])
                put_u32(FAM[fam])
        dims = {"down": (wpost, d), "mid": (2 * d, 2 * d), "up": (2 * d, hidden),
                "l2": (2 * hidden, hidden), "head": (2 * hidden, 3)}
        for name, mod, fam in layers:
            if mod is None:
                continue
            w, b = fold(mod)
            di, do = dims[name]
            s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
            q = torch.round(w / s).clamp(-128, 127).to(torch.int8)
            assert tuple(q.shape) == (FAM[fam], di, do), (name, q.shape)
            buf.extend(q.contiguous().numpy().tobytes())
            put_arr(s.squeeze(1), (FAM[fam], do))
            put_arr(b, (FAM[fam], do))
        if flags_of(cfg) & FLAG_SKIP:
            put_arr(model.skiph.weight.detach().t(), (2 * hidden, 3))
        if flags_of(cfg) & FLAG_PSQT:
            put_arr(model.psqt.weight.detach(), (NROWS, 3))

    bcfg = bck["cfg"]
    put_u32(MAGIC)
    put_u32(VERSION_DUAL_Q8)
    # Shared head: the accumulator width, the hidden width both tails read,
    # and the cp scale. Everything else (waist, flags, families, weights)
    # lives in the two tail blocks below, so a mismatch fails in the tail
    # checks rather than loading silently.
    put_u32(bcfg["width"])
    put_u32(bcfg["hidden"])
    put_f32(K)
    put_arr(big.ft.weight.detach(), (NROWS, bcfg["width"]))
    put_arr(big.ft_bias.detach(), (bcfg["width"],))
    put_tail(big, bcfg)
    put_tail(small, sck["cfg"])

    with open(args.out, "wb") as f:
        f.write(buf)
    print(f"wrote {args.out}: {len(buf):,} bytes  dual int8  "
          f"width {bcfg['width']} hidden {bcfg['hidden']} K {K:.1f}  "
          f"big bneck {bcfg['bneck']} + small bneck {sck['cfg']['bneck']}")


if __name__ == "__main__":
    main()
