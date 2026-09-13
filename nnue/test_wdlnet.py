"""Structural tests for the WDL-head sweep. Prints ALL PASS or raises.

These are the ones that would silently poison a 500M-position run: a bucket
index that does not mean what the docstring says, a bucketed layer that does
not equal its own loop reference, and a passthrough that is not antisymmetric.
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import PAD
from fen import feats_from_fen, FENS
from features import flip_feat
from wdlnet import (BucketLinear, WdlNet, king_sqs, mat_buckets, ce,
                    target_dist, expected, NB_MAT, FAM)

DEV = "cpu"
F = torch.tensor([feats_from_fen(f) for f in FENS], dtype=torch.long)


def test_king_sqs():
    us, them = king_sqs(F)
    # from the FENs directly: our king is (side 0, type 5), theirs (side 1, 5)
    for r in range(len(FENS)):
        row = [int(i) for i in F[r] if i != PAD]
        u = [i % 64 for i in row if i < 384 and (i % 384) // 64 == 5]
        t = [i % 64 for i in row if 384 <= i < 768 and (i % 384) // 64 == 5]
        assert len(u) == 1 and len(t) == 1, FENS[r]
        assert int(us[r]) == u[0], (FENS[r], int(us[r]), u[0])
        assert int(them[r]) == t[0] ^ 56, (FENS[r], int(them[r]), t[0] ^ 56)
    # the flipped board swaps the two, and the mirror is what makes the same
    # L1 table apply to both perspectives
    fu, ft = king_sqs(flip_feat(F))
    assert torch.equal(fu, them) and torch.equal(ft, us)
    print("  king_sqs                ok")


def test_mat_buckets():
    mu, mt = mat_buckets(F)
    assert int(mu.max()) < NB_MAT and int(mt.min()) >= 0
    # startpos: 8 pawns -> bin 2, 4 minors -> bin 2, 2 rooks -> bin 2, queen -> 1
    want = (((2 * 3 + 2) * 3 + 2) * 2 + 1)
    assert int(mu[0]) == want == int(mt[0]), (int(mu[0]), want)
    # K+P vs K (FENS[5], white to move): 1 pawn -> 0, no minors/rooks/queens
    assert int(mu[5]) == 0 and int(mt[5]) == 0
    # 3 pawns -> bin 1 for the side that has them (FENS[6]: white 2, black 2)
    fu, ft = mat_buckets(flip_feat(F))
    assert torch.equal(fu, mt) and torch.equal(ft, mu)
    print("  mat_buckets             ok")


def test_bucket_linear():
    torch.manual_seed(0)
    for code in ("delta", "onehot"):
        bl = BucketLinear(5, 7, 3, code)
        with torch.no_grad():           # give the deltas something to do
            bl.w.normal_(std=0.3)
            bl.b.normal_(std=0.3)
        x = torch.randn(11, 7)
        idx = torch.randint(0, 5, (11,))
        got = bl(x, idx)
        for i in range(11):
            w, b = bl.w[idx[i]], bl.b[idx[i]]
            if bl.base is not None:
                w, b = w + bl.base, b + bl.bbias
            ref = x[i] @ w + b
            assert torch.allclose(got[i], ref, atol=1e-5), (code, i)
    # delta init: every bucket is the base, so the layer is bucket-free at step 0
    bl = BucketLinear(5, 7, 3, "delta")
    x, idx = torch.randn(11, 7), torch.randint(0, 5, (11,))
    assert torch.allclose(bl(x, idx), x @ bl.base + bl.bbias, atol=1e-6)
    # nb == 1 is a plain linear
    bl1 = BucketLinear(1, 7, 3, "delta")
    assert torch.allclose(bl1(x, torch.zeros(11, dtype=torch.long)),
                          x @ bl1.base + bl1.bbias, atol=1e-6)
    print("  BucketLinear            ok")


def test_psqt_antisymmetry():
    """With the trunk silenced, the passthrough must satisfy f(flip) = swap(f)."""
    for mode in ("plain", "mat"):
        m = WdlNet(width=16, hidden=4, psqt=mode)
        with torch.no_grad():
            m.psqt.weight.normal_(std=0.3)
            m.psqt.weight[PAD].zero_()
            m.head.base.zero_(); m.head.bbias.zero_()
        a, b = m(F), m(flip_feat(F))
        assert torch.allclose(a, -b.flip(-1), atol=1e-5), (mode, a[0], b[0])
    print(f"  psqt passthrough        ok")


def test_shapes():
    for over in ({}, {"b1": "none", "b2": "none"}, {"b1": "none"},
                 {"b1": "mat", "b2": "king"}, {"head_b": "mat"},
                 {"head_b": "sym"}, {"depth": 1}, {"depth": 3},
                 {"reuse": False}, {"code": "onehot"}, {"skip": True},
                 {"psqt": "none"}, {"psqt": "mat"}, {"width": 32, "hidden": 8}):
        cfg = dict(width=16, hidden=4)
        cfg.update(over)
        out = WdlNet(**cfg)(F)
        assert out.shape == (len(FENS), 3), (over, out.shape)
        assert torch.isfinite(out).all()
    print("  every arm builds        ok")


def test_loss():
    torch.manual_seed(0)
    lg = torch.randn(9, 3)
    z = torch.tensor([0., .5, 1., 0., .5, 1., 0., .5, 1.])
    s = torch.randn(9) * 100
    teacher = lambda score, feat: (torch.full_like(score, .3),
                                   torch.full_like(score, .4),
                                   torch.full_like(score, .3))
    t = target_dist(s, z, teacher, None, 0.9)
    assert torch.allclose(t.sum(1), torch.ones(9), atol=1e-6)
    # lam = 0 is the one-hot outcome, and ce against it is plain cross-entropy
    oh = target_dist(s, z, teacher, None, 0.0)
    cls = (z * 2).round().long()
    assert torch.allclose(ce(lg, oh), nn.functional.cross_entropy(lg, cls), atol=1e-6)
    # a uniform prediction scores log 3 and E = 1/2
    u = torch.zeros(9, 3)
    assert abs(ce(u, oh).item() - torch.tensor(3.0).log().item()) < 1e-6
    assert torch.allclose(expected(u), torch.full((9,), 0.5), atol=1e-6)
    print("  loss / target           ok")


for fn in (test_king_sqs, test_mat_buckets, test_bucket_linear,
           test_psqt_antisymmetry, test_shapes, test_loss):
    fn()
print("ALL PASS")
