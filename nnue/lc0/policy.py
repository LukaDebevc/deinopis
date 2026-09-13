"""An lc0 SE-resnet, CLASSICAL_1858 policy head only, in PyTorch.

Companion to net.py (value head). Transcribed from lc0's own source
(blobless clone at /tmp/lc0, HEAD ~2026-09):

  backends/network_tf_cc.cc  classical branch: 1x1 conv (BN folded, relu) ->
                             flatten NHWC -> FC(1858) + bias. MakeConvBlock
                             defaults relu=true; the classical policy call
                             passes no override.
  neural/encoder.cc          kMoveStrs: the 1858 move strings in NN-index
                             order (/tmp/bilinear/lc0moves.txt). Knight promos
                             fold into the queen slot (no 'n' entries exist).
                             Black-to-move positions are rank-mirrored before
                             lookup, matching the plane encoder.

Maia-1900 is a relu net (no activation flag in its format block), so every
activation here is relu -- including inside the SE unit (se_unit.cc applies
the default activation there too).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loadw import read, convblock, layer
from .pbwire import walk

EPS = 1e-5

with open('/tmp/bilinear/lc0moves.txt') as f:
    MOVES = [l.strip() for l in f]
assert len(MOVES) == 1858, len(MOVES)
MOVE_TO_IDX = {m: i for i, m in enumerate(MOVES)}


def uci_to_idx(uci, black_to_move):
    """'e7e5' (+stm) -> 1858 index. Rank-mirror for black, knight promo -> q."""
    if black_to_move:
        tr = str.maketrans('12345678', '87654321')
        uci = uci.translate(tr)
    if len(uci) == 5 and uci[4] == 'n':
        uci = uci[:4] + 'q'
    return MOVE_TO_IDX[uci]


def _fold(cb, out_ch, in_ch, k):
    w = cb['w'].reshape(out_ch, in_ch, k, k).copy()
    b = cb.get('b', np.zeros(out_ch, np.float32)).copy()
    if 'mean' not in cb:
        return w, b
    gamma = cb.get('gamma', np.ones(out_ch, np.float32))
    beta = cb.get('beta', np.zeros(out_ch, np.float32))
    g = gamma / np.sqrt(cb['std'] + EPS)          # 'std' is the variance
    return w * g[:, None, None, None], -g * (cb['mean'] - b) + beta


class SEBlock(nn.Module):
    def __init__(self, c, se):
        super().__init__()
        self.c = c
        for n, t in se.items():
            self.register_buffer(n, torch.from_numpy(np.ascontiguousarray(t)))

    def forward(self, z, skip):
        p = z.mean(dim=(2, 3)) + self.cb2
        h = F.relu(p @ self.w1.view(-1, self.c).T + self.b1)
        s = h @ self.w2.view(-1, h.shape[1]).T + self.b2
        scale, shift = s[:, :self.c], s[:, self.c:]
        g = torch.sigmoid(scale)[:, :, None, None]
        return F.relu(g * (z + self.cb2[None, :, None, None])
                      + shift[:, :, None, None] + skip)


class LC0Policy(nn.Module):
    """Input conv -> N SE-residual blocks -> 1858 policy logits."""

    def __init__(self, path):
        super().__init__()
        nf, w = read(path)
        assert nf.get(1) == 1, f'input format {nf.get(1)}, want CLASSICAL_112'
        assert nf.get(4) == 2, f'policy format {nf.get(4)}, want CLASSICAL_1858'
        assert 2 in w and 27 not in w, 'not a residual tower'

        ib = convblock(w[1][0])
        self.filters = c = ib['mean'].size
        wt, bs = _fold(ib, c, 112, 3)
        self.register_buffer('iw', torch.from_numpy(wt))
        self.register_buffer('ib', torch.from_numpy(bs))

        self.blocks = nn.ModuleList()
        self.cw, self.cb = [], []
        for raw in w[2]:
            r = walk(raw)
            c1, c2 = convblock(r[1][0]), convblock(r[2][0])
            w1, b1 = _fold(c1, c, c, 3)
            w2, b2 = _fold(c2, c, c, 3)
            se = walk(r[3][0])
            d = {n: layer(se[f][0]) for f, n in ((1, 'w1'), (2, 'b1'), (3, 'w2'), (4, 'b2'))}
            d['cb2'] = b2
            blk = SEBlock(c, d)
            blk.register_buffer('w1c', torch.from_numpy(w1))
            blk.register_buffer('b1c', torch.from_numpy(b1))
            blk.register_buffer('w2c', torch.from_numpy(w2))
            self.blocks.append(blk)

        pb = convblock(w[3][0])
        self.pch = pb['mean'].size if 'mean' in pb else None
        # Conv policy head (POLICY_CONVOLUTION): policy1 is field 11
        # (3x3, c->c, BN, default activation); policy is field 3
        # (3x3, c->80, bias only, NO activation); then a gather through
        # kConvPolicyMap (73x64 -> 1858). Verified against cudnn + blas
        # backends, which agree with each other line for line here.
        p1 = convblock(w[11][0])
        c1 = p1['mean'].size
        w1, b1 = _fold(p1, c1, c, 3)
        self.register_buffer('p1w', torch.from_numpy(w1))
        self.register_buffer('p1b', torch.from_numpy(b1))
        p2w = layer(walk(w[3][0])[1][0]).reshape(80, c, 3, 3).copy()
        p2b = layer(walk(w[3][0])[2][0]).copy()
        self.register_buffer('p2w', torch.from_numpy(p2w.copy()))
        self.register_buffer('p2b', torch.from_numpy(p2b.copy()))
        convmap = np.loadtxt('/tmp/bilinear/convmap.txt', dtype=np.int64)
        assert convmap.shape == (73 * 64,), convmap.shape
        # Map position i is (channel, square) = (i // 64, i % 64) in the
        # backends' channel-major reading; our flat is NHWC, i.e. index
        # square * 80 + channel (cf. the TF backend's policy_map build).
        # Using i directly here was a silent total scramble (mass on legal
        # exactly at the chance rate) -- this translation IS the mapping.
        gather = np.full(1858, -1, np.int64)
        for i, j in enumerate(convmap):
            if j >= 0:
                assert gather[j] == -1, j
                gather[j] = (i % 64) * 80 + (i // 64)
        assert (gather >= 0).all()
        self.register_buffer('gather',
                             torch.from_numpy(gather.copy()).long())

    def forward(self, x):
        """x: [B, 112, 8, 8] float -> policy logits [B, 1858]."""
        h = F.relu(F.conv2d(x, self.iw, self.ib, padding=1))
        for b in self.blocks:
            y = F.relu(F.conv2d(h, b.w1c, b.b1c, padding=1))
            z = F.conv2d(y, b.w2c, None, padding=1)
            h = b(z, h)
        p = F.relu(F.conv2d(h, self.p1w, self.p1b, padding=1))
        p = F.conv2d(p, self.p2w, self.p2b, padding=1)
        # NHWC flatten: index (sq*80 + c), sq = h*8+w, matching the TF/CPU
        # backend's flat layout over the same plane order net.py verifies.
        p = p.permute(0, 2, 3, 1).reshape(p.shape[0], -1)
        return p[:, self.gather]
