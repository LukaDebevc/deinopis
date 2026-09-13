"""Collapse a trained quadratic model to one i16 matrix and write it for the engine.

For binary x, x_i^2 = x_i, so the linear term folds into the diagonal:

    x^T W x + b.x + c  =  sum_{i!=j} W_ij x_i x_j + sum_i (W_ii + b_i) x_i + c
                       =  x^T M x + c        with M_ii = W_ii + b_i

So the entire model is one symmetric 768x768 matrix and a scalar. The rank used
for training does not survive into the file -- W = V diag(lam) V^T is dense
whatever r was -- which is why full rank is free at inference.

The diagonal is stored separately and at higher precision. It is the PSQT --
"opponent queen on e4" is -1635 cp -- while the off-diagonal peaks at 452, so a
single i16 scale for both would be set by the diagonal and waste 5 bits on every
pairwise term. Splitting them takes the quantisation step from 0.050 cp to
0.014 cp for 3 KB.

A version-4 file may also carry a BUCKETED PSQT: extra diagonals, one set per
bucket of a routing rule, added to `diag` for the bucket the position falls in.
This is the one conditioning in the study that costs the engine nothing --
still 32 gathers, from a 584 x 768 table instead of a 768 one -- because the
linear term has no accumulator to rebuild (LEDGER 019).

Output file layout (little-endian, no dependencies to read it):
    magic    u32   0x51554144 ("QUAD")
    version  u32   3 or 4
    q_off    i32   off-diagonal scale
    q_diag   i32   diagonal scale
    bias     i32   cp * q_diag  (NOT whole centipawns: rounding it to an
                  integer put a systematic -0.46 cp into every eval)
    nfam     i32   v3: 0 padding. v4: number of bucketed-PSQT families.
    v4 only, nfam times:
      fam    u32   0 = material (576 buckets), 1 = count (8)
      nbuck  u32
    diag     i32[768]        cp * q_diag
    off      i16[768*768]    cp * q_off, row-major, diagonal entries zero
    v4 only, nfam times:
      pq     i32[nbuck*768]  cp * q_diag, row-major by bucket

    eval_cp = round( ( 2*sum_{i<j in S} off_ij * q_diag
                     + sum_{i in S} (diag_i + sum_f pq_f[b_f][i]) * q_off
                     + bias * q_off )
                   / (q_off * q_diag) )

One rounding, at the end. Rounding each term separately cost up to 0.5 cp
apiece and the engine is only accurate to 1 cp in total.

The families are summed, not crossed, exactly as `Bucketed` trains them: a
576-bucket table and an 8-bucket table, not a 4608-bucket one.
"""

import argparse
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import NFEAT   # noqa: E402

MAGIC = 0x51554144


# Bucket family -> (id in the file, bucket count). Must match `qeval.rs`.
FAMILY_ID = {"material": (0, 576), "count": (1, 8)}


def collapse(ck):
    st, scale = ck["state"], ck["scale"]
    V = st["v.weight"][:NFEAT].double()                 # (768, r)
    # `Quadratic` stores the reader as `lam`; `Bucketed` with one bucket keeps
    # the same numbers in its single reader row, and the two are the same model
    # (see the class docstring in train.py).
    lam = (st["lam"] if "lam" in st else st["readers.0.weight"][0]).double()
    W = (V * lam) @ V.T                                 # V diag(lam) V^T
    if "psqt.weight" in st:
        W += torch.diag(st["psqt.weight"][:NFEAT, 0].double())
    W *= scale
    W = 0.5 * (W + W.T)                                 # kill float asymmetry
    return W, float(st["bias"].item() * scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pfams", default="",
                    help="comma-separated bucketed-PSQT families in the "
                         "checkpoint, in the order Bucketed built them")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    M, bias = collapse(ck)

    d = torch.diagonal(M).clone()
    off = M - torch.diag(d)
    q_off = int(32767 // max(float(off.abs().max()), 1e-9))
    q_diag = 1024
    offq = torch.round(off * q_off).clamp(-32768, 32767).to(torch.int16)
    diagq = torch.round(d * q_diag).to(torch.int32)

    # How much did quantisation cost? Measure it on the thing we care about --
    # the eval of a realistic position -- not on the matrix entries.
    rng = np.random.default_rng(0)
    err = []
    for _ in range(2000):
        S = torch.from_numpy(rng.choice(NFEAT, 32, replace=False))
        exact = M[S][:, S].sum().item() + bias
        approx = (offq[S][:, S].to(torch.int64).sum().item() / q_off
                  + diagq[S].to(torch.int64).sum().item() / q_diag
                  + round(bias * q_diag) / q_diag)
        err.append(abs(exact - approx))
    err = np.array(err)

    # The bucketed PSQT, if the checkpoint has one. Stored at q_diag like the
    # diagonal it is added to, so the engine sums them before a single divide.
    fams = [x.strip() for x in args.pfams.split(",") if x.strip()]
    pq = []
    for k, name in enumerate(fams):
        key = f"pq.{k}.weight"
        if key not in ck["state"]:
            sys.exit(f"--pfams says {name} but the checkpoint has no {key}")
        fid, nb = FAMILY_ID[name]
        w = ck["state"][key].double().reshape(-1) * ck["scale"]
        rows = w.numel() // nb
        w = w.reshape(nb, rows)[:, :NFEAT]              # drop the padding row
        if rows != NFEAT + 1:
            sys.exit(f"{key}: {rows} rows per bucket, expected {NFEAT + 1}")
        pq.append((name, fid, nb, torch.round(w * q_diag).to(torch.int32)))

    version = 4 if pq else 3
    with open(args.out, "wb") as f:
        f.write(struct.pack("<IIiiii", MAGIC, version, q_off, q_diag,
                            int(round(bias * q_diag)), len(pq)))
        for _, fid, nb, _ in pq:
            f.write(struct.pack("<II", fid, nb))
        f.write(diagq.numpy().astype("<i4").tobytes())
        f.write(offq.numpy().astype("<i2").tobytes())
        for _, _, _, t in pq:
            f.write(t.numpy().astype("<i4").tobytes())

    print(f"arm            {ck['arm']}")
    print(f"val loss       {ck['val']:.6f}   (K={ck['K']:.1f}, scale={ck['scale']:.1f})")
    print(f"diag peak      {float(d.abs().max()):.1f} cp  -> q_diag = {q_diag} "
          f"(resolution {1/q_diag:.4f} cp)")
    print(f"off  peak      {float(off.abs().max()):.1f} cp  -> q_off  = {q_off} "
          f"(resolution {1/q_off:.4f} cp)")
    print(f"bias           {bias:+.2f} cp")
    print(f"quantisation error on 32-piece positions:")
    print(f"               mean {err.mean():.4f} cp   p99 {np.percentile(err,99):.4f} cp"
          f"   max {err.max():.4f} cp")
    for name, fid, nb, t in pq:
        print(f"psqt bucket    {name} (id {fid}): {nb} x {NFEAT}, "
              f"peak {float(t.abs().max())/q_diag:.1f} cp")
    import os
    print(f"wrote          {args.out}  ({os.path.getsize(args.out):,} bytes, "
          f"version {version})")


if __name__ == "__main__":
    main()
