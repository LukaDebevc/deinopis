"""Write a trained `Deep` net in the form the engine reads.

Everything here is f32 on purpose. Quantising a crelu net is a separate job
with its own failure mode -- the accumulator's scale, l1's input scale and the
clipping range all have to agree, and when they do not the eval stays plausible
and every game is played slightly wrong. Getting the ARCHITECTURE into the
engine and getting it QUANTISED are two problems, and mixing them means a
mismatch in `verify.py` could be either. In floats a mismatch is a bug.

Layout (version 5, little-endian throughout):

    u32 magic, u32 version, u32 nrows, u32 width, u32 hidden,
    u32 n_extra, u32 n_fam, f32 scale
    n_extra x (u32 kind, u32 off, u32 size)     # extra input families
    n_fam   x (u32 fam,  u32 nbuck)             # read-bucket families
    f32 v     [nrows][width]                    # accumulator table
    f32 psqt  [nrows]                           # psqt skip, stm side only
    f32 l1_w  [hidden][2*width]                 # shared, both perspectives
    f32 l1_b  [hidden]
    per family: f32 l2 [nbuck][hidden*hidden + hidden]   # w row-major [k][h]
    per family: f32 l3 [nbuck][hidden + 1]
    f32 bias

The two bucket families are SUMMED, not crossed -- 576 rows plus 8 rows rather
than 4608 -- which is the same convention the version-4 psqt tables use.
"""
import argparse, struct
import torch

MAGIC = 0x51554144
VERSION = 5
# Must match train.py's EXTRAS and the engine's `deepeval::Extra`.
EXTRA_ID = {"pawnfile": (0, 8 * 64), "pawnpair": (1, 7 * 16)}
FAMILY_ID = {"material": (0, 576), "count": (1, 8)}
NFEAT = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--extras", default="",
                    help="'+'-joined, in the order Deep built them")
    ap.add_argument("--families", default="",
                    help="comma separated, in the order Deep built them")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st, scale = ck["state"], float(ck["scale"])
    ex = [x for x in a.extras.split("+") if x and x != "none"]
    fams = [x.strip() for x in a.families.split(",") if x.strip() and x != "none"]

    v = st["v.weight"].float()                       # (nrows, width)
    nrows, width = v.shape
    l1w = st["l1.weight"].float()                    # (hidden, 2*width)
    hidden = l1w.shape[0]
    assert l1w.shape[1] == 2 * width, (l1w.shape, width)

    # The table must be exactly 768 piece rows + 1 padding row + the extras the
    # caller named, or the engine will index a different feature than the model
    # was trained on and nothing downstream will notice.
    want = NFEAT + 1 + sum(EXTRA_ID[n][1] for n in ex)
    assert nrows == want, f"table has {nrows} rows, --extras implies {want}"
    assert (v[NFEAT] == 0).all(), "padding row is not zero"
    assert len(fams) == len([k for k in st if k.startswith("l2.")]), \
        "--families does not match the checkpoint's l2 tables"

    head = struct.pack("<IIIIIIIf", MAGIC, VERSION, nrows, width, hidden,
                       len(ex), len(fams), scale)
    off = NFEAT + 1
    for n in ex:
        kind, size = EXTRA_ID[n]
        head += struct.pack("<II", kind, off)
        head += struct.pack("<I", size)
        off += size
    for n in fams:
        fid, nb = FAMILY_ID[n]
        assert st[f"l2.{fams.index(n)}.weight"].shape[0] == nb
        head += struct.pack("<II", fid, nb)

    def blob(t):
        return t.contiguous().float().view(-1).numpy().astype("<f4").tobytes()

    body = blob(v) + blob(st["psqt.weight"]) + blob(l1w) + blob(st["l1.bias"])
    for k in range(len(fams)):
        body += blob(st[f"l2.{k}.weight"])
    for k in range(len(fams)):
        body += blob(st[f"l3.{k}.weight"])
    body += blob(st["bias"])

    with open(a.out, "wb") as f:
        f.write(head + body)
    print(f"{a.out}: {len(head) + len(body):,} bytes  "
          f"nrows={nrows} width={width} hidden={hidden} scale={scale:.4g}")
    print(f"  extras   {'+'.join(ex) or 'none'}")
    print(f"  families {','.join(fams) or 'none'}  val={ck.get('val')}")


if __name__ == "__main__":
    main()
