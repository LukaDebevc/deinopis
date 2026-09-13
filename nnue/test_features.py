"""Independent checks on the new feature families and side/flip machinery.

Every extra-input family is re-derived here from a FEN by a second, deliberately
dumb implementation and compared against the batched GPU one. A wrong feature is
invisible in a loss curve -- it just trains something else -- so nothing runs
until this passes.
"""
import sys
import torch

sys.path.insert(0, ".")
from loader import NFEAT, PAD                                    # noqa: E402
import train as T
# _PLUS and _QUARTERS are module-private in features.py, so `import *` into
# train.py skips them; reach them at their real home.
import features as _F
T._PLUS, T._QUARTERS = _F._PLUS, _F._QUARTERS                                                # noqa: E402

from fen import feats_from_fen, decode, FENS          # noqa: E402


feat = torch.tensor([feats_from_fen(f) for f in FENS])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feat = feat.to(dev)
fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    if not ok:
        fails += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        print(f"       got  {got}\n       want {want}")


# ---------------------------------------------------------------- flip
print("flip_feat")
f2 = T.flip_feat(feat)
check("involution", T.flip_feat(f2).tolist(), feat.tolist())
check("pad preserved", (f2 == PAD).tolist(), (feat == PAD).tolist())
# every piece keeps its type, swaps side, mirrors rank
for row, frow in zip(feat.tolist(), f2.tolist()):
    for a, b in zip(row, frow):
        if a == PAD:
            continue
        sa, pa, ra, fa = a // 384, (a % 384) // 64, (a % 64) // 8, a % 8
        sb, pb, rb, fb = b // 384, (b % 384) // 64, (b % 64) // 8, b % 8
        assert (sb, pb, rb, fb) == (1 - sa, pa, 7 - ra, fa), (a, b)
print("  ok   type kept, side swapped, rank mirrored")

# ---------------------------------------------------------------- pawnfile
print("x_pawnfile")
got = T.x_pawnfile(feat, NFEAT + 1).cpu().tolist()
want = []
for row in feat.cpu().tolist():
    ours = {}
    them = {}
    for side, pt, rk, fl in decode(row):
        if pt != 0:
            continue
        (ours if side == 0 else them).setdefault(fl, []).append(rk)
    w = []
    for fl in range(8):
        u = max(ours.get(fl, [0]))
        t = min(them.get(fl, [7]))
        w.append(NFEAT + 1 + fl * 64 + u * 8 + t)
    want.append(w)
check("per-file advanced pawn ranks", got, want)

# ---------------------------------------------------------------- pawnpair
print("x_pawnpair")
got = T.x_pawnpair(feat, 0).cpu().tolist()
want = []
for row in feat.cpu().tolist():
    u = [0] * 8
    t = [0] * 8
    for side, pt, rk, fl in decode(row):
        if pt == 0:
            (u if side == 0 else t)[fl] = 1
    want.append([fl * 16 + ((u[fl] * 2 + u[fl + 1]) * 2 + t[fl]) * 2 + t[fl + 1]
                 for fl in range(7)])
check("adjacent-file pawn presence", got, want)

# ---------------------------------------------------------------- centre
print("x_centre")
got = T.x_centre(feat, 0).cpu().tolist()
want = []
for row in feat.cpu().tolist():
    m = [0, 0]
    for side, pt, rk, fl in decode(row):
        if (rk, fl) in T._PLUS:
            m[side] |= 1 << T._PLUS.index((rk, fl))
    want.append([m[0], (1 << 12) + m[1]])
check("central plus occupancy", got, want)

# ---------------------------------------------------------------- tiles
for tname, (h, w, sv, sh) in {"tile23": (2, 3, 2, 1), "tile22": (2, 2, 1, 1),
                              "tile43": (4, 3, 2, 1)}.items():
    print(f"{tname}  ({h}x{w} stride {sv},{sh})")
    fn = T.EXTRAS[tname]
    origins = [(r, f) for r in range(0, 8 - h + 1, sv)
               for f in range(0, 8 - w + 1, sh)]
    nt, k = len(origins), h * w
    check("size", fn.size, 2 * nt * (1 << k))
    check("active", fn.active, 2 * nt)
    got = fn(feat, 0).cpu().tolist()
    want = []
    for row in feat.cpu().tolist():
        v = []
        for side in (0, 1):
            for t, (r0, f0) in enumerate(origins):
                m = 0
                for s, pt, rk, fl in decode(row):
                    if s == side and r0 <= rk < r0 + h and f0 <= fl < f0 + w:
                        m |= 1 << ((rk - r0) * w + (fl - f0))
                v.append((side * nt + t) * (1 << k) + m)
        want.append(v)
    check("occupancy masks", got, want)

# ---------------------------------------------------------------- kingpair
print("x_kingpair")
got = T.x_kingpair(feat, 0).cpu().tolist()
want = []
for row in feat.cpu().tolist():
    ks = {s: rk * 8 + fl for s, pt, rk, fl in decode(row) if pt == 5}
    want.append([ks[0] * 64 + ks[1]])
check("king square pair", got, want)

# ---------------------------------------------------------------- Extras
print("Extras bookkeeping")
ex = T.Extras(("pawnfile", "centre", "kingpair"))
check("rows", ex.rows, NFEAT + 1 + 512 + 8192 + 4096)
check("active", ex.active, 8 + 2 + 1)
out = ex(feat)
check("concat width", out.shape[1], 32 + 11)
check("indices in range", bool((out < ex.rows).all()), True)
check("base untouched", out[:, :32].tolist(), feat.tolist())
# offsets must not collide between families
off = out[:, 32:].cpu()
check("pawnfile block", bool(((off[:, :8] >= NFEAT + 1) &
                              (off[:, :8] < NFEAT + 1 + 512)).all()), True)
check("centre block", bool(((off[:, 8:10] >= NFEAT + 1 + 512) &
                            (off[:, 8:10] < NFEAT + 1 + 512 + 8192)).all()), True)

# --------------------------------------------- tied SideQuadratic == deployed
print("SideQuadratic tied == Bucketed(none)")
torch.manual_seed(0)
a = T.SideQuadratic(64, 258.7).to(dev).eval()
torch.manual_seed(0)
b = T.Bucketed(64, 258.7, families=("none",)).to(dev).eval()
with torch.no_grad():
    b.v.weight.copy_(a.v.weight)
    b.psqt.weight.copy_(a.psqt.weight)
    b.readers[0].weight.copy_(a.r[:1])
    b.bias.copy_(a.bias)
    d = (a(feat) - b(feat)).abs().max().item()
check("outputs agree", d < 1e-3, True)
print(f"       max |diff| = {d:.2e} cp")

# ------------------------------------------- blocks sum back to the full form
print("SideQuadratic blocks")
torch.manual_seed(0)
m = T.SideQuadratic(64, 258.7, blocks="all").to(dev).eval()
with torch.no_grad():
    # tie the three readers as (R, R, 2R) and it must reproduce `tied` with R
    m.r[0].copy_(a.r[0]); m.r[1].copy_(a.r[0]); m.r[2].copy_(2 * a.r[0])
    m.v.weight.copy_(a.v.weight); m.psqt.weight.copy_(a.psqt.weight)
    m.bias.copy_(a.bias)
d = (m(feat) - a(feat)).abs().max().item()
check("uu + tt + 2*ut reproduces the tied form", d < 1e-2, True)
print(f"       max |diff| = {d:.2e} cp")

if __name__ == "__main__":
    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURE(S)'}")
    sys.exit(1 if fails else 0)
