"""An lc0 SE-resnet, value head only, in PyTorch.

Why reimplement instead of driving the lc0 binary: we need 10^8 forward passes
in batches. lc0's interface is one position at a time through UCI, which is the
wrong shape by four orders of magnitude.

Everything here is transcribed from lc0's own reference code, not from memory:

  network_legacy.cc  `bn_stddivs` holds the VARIANCE, not a standard deviation
                     and not its reciprocal. Folding is
                     g = gamma / sqrt(var + 1e-5),
                     w <- w * g,  b <- -g * (mean - b) + beta.
  se_unit.cc         out = act( sigmoid(scale) * (conv2 + bias) + shift + skip )
                     with [scale, shift] the two halves of the second FC.
  network_blas.cc    value head: 1x1 conv 512->32, mish; FC 2048->128, mish;
                     FC 128->3; softmax -> (W, D, L) in that order.

Only the value head is built. The policy head on this net is attention-based
and we are not asking it anything.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loadw import read, convblock, layer
from .pbwire import walk

EPS = 1e-5


def _fold(cb, out_ch, in_ch, k):
    """ConvBlock -> (weight [out,in,k,k], bias [out]), batch norm folded in."""
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
        # z is conv2's output WITHOUT its bias; cb2 is that bias.
        p = z.mean(dim=(2, 3)) + self.cb2                     # [B, C]
        h = F.mish(p @ self.w1.view(-1, self.c).T + self.b1)  # [B, se]
        s = h @ self.w2.view(-1, h.shape[1]).T + self.b2      # [B, 2C]
        scale, shift = s[:, :self.c], s[:, self.c:]
        g = torch.sigmoid(scale)[:, :, None, None]
        return F.mish(g * (z + self.cb2[None, :, None, None])
                      + shift[:, :, None, None] + skip)


class LC0Value(nn.Module):
    """Input conv -> N SE-residual blocks -> WDL value head."""

    def __init__(self, path):
        super().__init__()
        nf, w = read(path)
        assert nf.get(1) == 1, f'input format {nf.get(1)}, want CLASSICAL_112'
        assert nf.get(5) == 2, f'value format {nf.get(5)}, want WDL'
        assert nf.get(7) == 1, 'expected mish as the default activation'
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
            d['cb2'] = b2                       # conv2 bias, applied inside SE
            blk = SEBlock(c, d)
            blk.register_buffer('w1c', torch.from_numpy(w1))
            blk.register_buffer('b1c', torch.from_numpy(b1))
            blk.register_buffer('w2c', torch.from_numpy(w2))
            self.blocks.append(blk)

        vb = convblock(w[6][0])
        self.vch = vb['mean'].size
        vw, vbias = _fold(vb, self.vch, c, 1)
        self.register_buffer('vw', torch.from_numpy(vw))
        self.register_buffer('vb', torch.from_numpy(vbias))
        self.register_buffer('h1w', torch.from_numpy(layer(w[7][0])))
        self.register_buffer('h1b', torch.from_numpy(layer(w[8][0])))
        self.register_buffer('h2w', torch.from_numpy(layer(w[9][0])))
        self.register_buffer('h2b', torch.from_numpy(layer(w[10][0])))
        self.nhid = self.h1b.numel()

    def forward(self, x):
        """x: [B, 112, 8, 8] float -> WDL logits [B, 3] (softmax not applied)."""
        h = F.mish(F.conv2d(x, self.iw, self.ib, padding=1))
        for b in self.blocks:
            y = F.mish(F.conv2d(h, b.w1c, b.b1c, padding=1))
            z = F.conv2d(y, b.w2c, None, padding=1)
            h = b(z, h)
        v = F.mish(F.conv2d(h, self.vw, self.vb))
        v = v.reshape(v.shape[0], -1)
        v = F.mish(v @ self.h1w.view(self.nhid, -1).T + self.h1b)
        return v @ self.h2w.view(3, -1).T + self.h2b
