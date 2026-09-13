"""Second, dumb implementation of the round-two families, checked against the
batched one. Same rule as test_features.py: nothing runs until this passes."""
import sys
import torch

sys.path.insert(0, ".")
from loader import PAD                                          # noqa: E402
import train as T
# _PLUS and _QUARTERS are module-private in features.py, so `import *` into
# train.py skips them; reach them at their real home.
import features as _F
T._PLUS, T._QUARTERS = _F._PLUS, _F._QUARTERS                                               # noqa: E402
from fen import feats_from_fen, decode, FENS                    # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feat = torch.tensor([feats_from_fen(f) for f in FENS]).to(dev)
fails = 0


def check(name, got, want):
    global fails
    if got != want:
        fails += 1
        print(f"  FAIL {name}\n       got  {got}\n       want {want}")
    else:
        print(f"  ok   {name}")


print("centre2 == centre (the general region helper reproduces the special case)")
check("identical", T.EXTRAS["centre2"](feat, 0).tolist(),
      T.EXTRAS["centre"](feat, 0).tolist())

print("randcell is a different 12 squares of the same shape")
check("size", T.EXTRAS["randcell"].size, T.EXTRAS["centre"].size)
check("differs", T.EXTRAS["randcell"](feat, 0).tolist()
      != T.EXTRAS["centre"](feat, 0).tolist(), True)

print("x_pawnrow")
got = T.x_pawnrow(feat, 0).cpu().tolist()
want = []
for row in feat.cpu().tolist():
    w = []
    for fl in range(8):
        code = 0
        for s, pt, rk, f in decode(row):
            if pt == 0 and f == fl and 1 <= rk <= 6:
                code += (s + 1) * 3 ** (rk - 1)
        w.append(fl * 729 + code)
    want.append(w)
check("per-file base-3 pawn column", got, want)

print("x_pawnrow contains x_pawnfile")
# Decoding the base-3 code back to the most advanced pawn of each side has to
# reproduce pawnfile exactly, or one of the two is wrong.
pr = T.x_pawnrow(feat, 0).cpu().tolist()
pf = T.x_pawnfile(feat, 0).cpu().tolist()
ok = True
for prow, frow in zip(pr, pf):
    for fl in range(8):
        code = prow[fl] - fl * 729
        cells = [(code // 3 ** i) % 3 for i in range(6)]
        us = max([i + 1 for i, c in enumerate(cells) if c == 1] or [0])
        them = min([i + 1 for i, c in enumerate(cells) if c == 2] or [7])
        ok &= frow[fl] - fl * 64 == us * 8 + them
check("decodes to the same advanced-pawn pair", ok, True)

print("x_kingzone")
got = T.x_kingzone(feat, 0).cpu().tolist()
want = []
for row in feat.cpu().tolist():
    ks = {s: (rk, fl) for s, pt, rk, fl in decode(row) if pt == 5}
    w = [0, 0, 0, 0]
    for j, who in enumerate((0, 1)):
        kr, kf = ks[who]
        for s, pt, rk, fl in decode(row):
            if abs(rk - kr) <= 1 and abs(fl - kf) <= 1:
                w[j * 2 + s] |= 1 << ((rk - kr + 1) * 3 + (fl - kf + 1))
    want.append([i * 512 + v for i, v in enumerate(w)])
check("3x3 around each king, by side", got, want)
# the king is always the centre of its own zone
for row in got:
    assert (row[0] - 0) & (1 << 4), row          # our king in our zone
    assert (row[3] - 3 * 512) & (1 << 4), row    # their king in their zone
print("  ok   each king sits at bit 4 of its own zone")

print("flanks")
fn = T.EXTRAS["flanks"]
check("size", fn.size, 8 * 4096)
check("active", fn.active, 8)
got = fn(feat, 0).cpu().tolist()
want = []
for row in feat.cpu().tolist():
    w, off = [], 0
    for sq_set in T._QUARTERS:
        m = [0, 0]
        for s, pt, rk, fl in decode(row):
            if (rk, fl) in sq_set:
                m[s] |= 1 << sq_set.index((rk, fl))
        w += [off + m[0], off + 4096 + m[1]]
        off += 8192
    want.append(w)
check("four quarters, per side", got, want)



print("PerspSplit endpoints reproduce the models they should")
torch.manual_seed(0)
sq = T.SideQuadratic(64, 258.7).to(dev).eval()
ps = T.PerspSplit(64, 0, 258.7).to(dev).eval()
with torch.no_grad():
    ps.vf.weight.copy_(sq.v.weight)
    ps.psqt.weight.copy_(sq.psqt.weight)
    ps.r[0].copy_(sq.r[0])
    ps.bias.copy_(sq.bias)
check("all-free end == single accumulator",
      (ps(feat) - sq(feat)).abs().max().item() < 1e-3, True)

torch.manual_seed(0)
pv = T.Perspective(32, 258.7).to(dev).eval()
ps2 = T.PerspSplit(0, 32, 258.7).to(dev).eval()
with torch.no_grad():
    ps2.vt.weight.copy_(pv.v.weight)
    ps2.psqt.weight.copy_(pv.psqt.weight)
    ps2.r[0].copy_(pv.r[0]); ps2.r[1].copy_(pv.r[1]); ps2.r[2].copy_(pv.r[2])
    ps2.bias.copy_(pv.bias)
check("all-tied end == the tied perspective pair",
      (ps2(feat) - pv(feat)).abs().max().item() < 1e-3, True)

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
