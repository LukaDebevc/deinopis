"""A standard-shaped NNUE trunk with a three-way outcome head.

Different from everything else in `models.py` in two ways that matter.

**The head is a real distribution.** LEDGER 048 measured that the outcome is
two-dimensional and that the scalar throws away the larger axis: at |score|<20
the expected score moves 0.013 across material while the draw rate moves
0.99 -> 0.39. It also measured that Stockfish's own WDL output cannot recover
that axis, because `win_rate_model` is a deterministic function of the cp score
and the material count. So the second axis has to be *learned from outcomes*,
which is what a softmax over (L, D, W) trained by cross-entropy does.

**The trunk is bucketed layer by layer, not table by table.** Shape:

    768 psq  --ft-->  256 us | 256 them          (one table, two perspectives)
    256      --L1[king]-->  32     x2, weights shared, their king mirrored
    64       --L2[material]--> 32   x2, weights shared, one bucket per side
    64       --L3-->  3 logits      (L, D, W)
    768      --psqt--> 3 logits     passthrough, added to the above

A bucketed layer is parameterised as `base + delta[b]` rather than as one
independent weight matrix per bucket. Every sample then trains the base, so the
gradient a bucket receives does not thin out as the bucket count grows -- with
64 x 54 cells and independent weights, a cell sees 1/3456 of the batch. `code`
selects between the two so the choice is measured rather than assumed.

Why the buckets are indexed the way they are: `flip_feat` maps (side, sq) to
(1-side, sq^56), so in the flipped view their king sits at `sq^56` with side 0.
Feeding that square to the SAME L1 table is what "reuse the weights and flip
the other king" means, and it is exact, not an approximation.

Bucketed layers are evaluated by computing every bucket densely and selecting
one. It wastes `nb` times the multiplies, but the alternative -- gathering a
per-sample weight matrix -- materialises n x 256 x 32 floats (537 MB at
n = 16384) and is memory-bound. The dense form is a single GEMM.
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import NFEAT, PAD          # noqa: E402
from common import bag                 # noqa: E402
from features import _parts, flip_feat # noqa: E402
from wdl import b_sym, B_SYM_N          # noqa: E402


NB_KING = 64
NB_MAT = 3 * 3 * 3 * 2          # pawns x minors x rooks x queen = 54

# Every bucket family a layer may read. "sym" is `wdl.b_sym`, the
# side-swap-invariant (material x pawns) partition LEDGER 048 fits the draw
# rate on -- the natural candidate for the output layer, which predicts a
# distribution over an outcome and not a side-relative score.
FAM = {"none": 1, "king": NB_KING, "mat": NB_MAT, "sym": B_SYM_N,
       # Controls for *why* a bucket helps. `hash64`/`hash54` keep the bucket
       # count, the parameter count and the arithmetic identical to `king`/
       # `mat` but carry no information in the index: the bucket is a hash of
       # the whole position. If an arm on these matches its real-family twin,
       # the family was buying parameters, not routing. `king4`/`king16` and
       # `mat6` are the same partitions at lower resolution -- a dose-response
       # curve, which routing predicts should be monotone and capacity does not.
       "hash64": 64, "hash54": 54, "hash2": 2, "king4": 4, "king16": 16, "mat6": 6,
       # 64 squares + 1 for "no queen". With two queens the lowest-numbered
       # square wins: a random pick would make the same position score
       # differently on two visits, which breaks the engine's transposition
       # table and any reproducible measurement.
       "queen": 65}


# ------------------------------------------------------------------ buckets

def king_sqs(feat):
    """(our king square, their king square mirrored into our frame)."""
    valid, side, pt, sq = _parts(feat)
    is_k = valid & (pt == 5)
    us = (sq * (is_k & (side == 0))).sum(1).clamp(0, 63)
    them = (sq * (is_k & (side == 1))).sum(1).clamp(0, 63) ^ 56
    return us.long(), them.long()


def _count(feat, mask, side_id):
    """How many pieces matching `mask` belong to side `side_id`."""
    valid, side, pt, sq = _parts(feat)
    return (mask & valid & (side == side_id)).sum(1)


def mat_buckets(feat):
    """(our material bucket, their material bucket), each 0..53.

    pawns (0-2 | 3-5 | 6-8) x minors (0 | 1-2 | 3+) x rooks (0 | 1 | 2+)
    x queen (0 | 1+). Luka's partition, one index per side.
    """
    valid, side, pt, sq = _parts(feat)
    out = []
    for s in (0, 1):
        mine = valid & (side == s)
        p = (mine & (pt == 0)).sum(1)
        mi = (mine & ((pt == 1) | (pt == 2))).sum(1)
        r = (mine & (pt == 3)).sum(1)
        q = (mine & (pt == 4)).sum(1)
        pb = (p // 3).clamp(max=2)
        mb = torch.where(mi == 0, 0, torch.where(mi <= 2, 1, 2))
        rb = r.clamp(max=2)
        qb = (q > 0).long()
        out.append((((pb * 3 + mb) * 3 + rb) * 2 + qb).long())
    return out[0], out[1]


_ZOB = None


def hash_buckets(feat, nb):
    """A deterministic, information-free bucket: Zobrist hash of the piece list.

    The first version of this multiplied each piece code by one constant and
    summed, which factors to `(C * sum(code)) mod nb` -- a relabelling of a
    single scalar, not a hash, and it leaked piece count through the `+1` per
    piece. A per-(piece, square) random table XOR-reduced is the standard fix
    and is what the engine's own transposition key uses. XOR is order-invariant
    and each (piece, square) occurs at most once, so nothing cancels.

    This is the control that separates "the index carries information" from
    "54 matrices are more matrices than 1", so it has to be measurably
    information-free -- see the permutation null in LEDGER 050.
    """
    global _ZOB
    valid, side, pt, sq = _parts(feat)
    code = ((side * 6 + pt) * 64 + sq + 1).long() * valid.long()
    if _ZOB is None or _ZOB.device != code.device:
        g = torch.Generator().manual_seed(0x9E3779B9)
        t = torch.randint(-(2 ** 62), 2 ** 62, (12 * 64 + 1,),
                          generator=g, dtype=torch.int64)
        t[0] = 0                                  # empty slots contribute nothing
        _ZOB = t.to(code.device)
    v = _ZOB[code]
    h = v[:, 0]
    for i in range(1, v.shape[1]):
        h = torch.bitwise_xor(h, v[:, i])
    h = torch.bitwise_xor(h, torch.bitwise_right_shift(h, 32))
    return (torch.bitwise_and(h, 0x7FFFFFFF) % nb).long()


def queen_sqs(feat):
    """(our queen square, their queen square mirrored), 64 meaning no queen."""
    valid, side, pt, sq = _parts(feat)
    is_q = valid & (pt == 4)
    out = []
    for s_id, mirror in ((0, False), (1, True)):
        m = is_q & (side == s_id)
        q = torch.where(m, sq.long(), torch.full_like(sq.long(), 64))
        lo = q.min(1).values
        has = lo < 64
        v = lo.clamp(max=63)
        if mirror:
            v = v ^ 56
        out.append(torch.where(has, v, torch.full_like(v, 64)))
    return out[0], out[1]


def coarse_king(feat, nb):
    """King buckets at reduced resolution: 4 quadrants or 16 4x4 blocks."""
    us, them = king_sqs(feat)
    if nb == 4:
        f = lambda x: (x // 32) * 2 + (x % 8) // 4
    elif nb == 16:
        f = lambda x: (x // 16) * 4 + (x % 8) // 2
    else:
        raise ValueError(nb)
    return f(us), f(them)


def coarse_mat(feat):
    """Material at 6 buckets instead of 54: pawns (0-2|3-5|6-8) x queen."""
    us, them = mat_buckets(feat)
    # the 54-index is (((pb*3 + mb)*3 + rb)*2 + qb); drop minors and rooks.
    f = lambda x: (x // 18) * 2 + (x % 2)
    return f(us), f(them)


# ------------------------------------------------------------- bucketed layer

class BucketLinear(nn.Module):
    """din -> dout, with the weights chosen by a bucket index.

    code="delta":  W_b = W_base + dW[b], dW init 0. Every sample trains the
                   base, so a bucket only has to learn its *difference*.
    code="onehot": W_b = W[b], each bucket independent. The control.

    nb == 1 collapses to a plain nn.Linear and takes the same path either way.
    """

    def __init__(self, nb, din, dout, code="delta"):
        super().__init__()
        self.nb, self.din, self.dout, self.code = nb, din, dout, code
        s = din ** -0.5
        self.base = None
        if nb == 1 or code == "delta":
            self.base = nn.Parameter(torch.empty(din, dout).uniform_(-s, s))
            self.bbias = nn.Parameter(torch.empty(dout).uniform_(-s, s))
        if nb > 1:
            self.w = nn.Parameter(torch.zeros(nb, din, dout)
                                  if code == "delta" else
                                  torch.empty(nb, din, dout).uniform_(-s, s))
            self.b = nn.Parameter(torch.zeros(nb, dout)
                                  if code == "delta" else
                                  torch.empty(nb, dout).uniform_(-s, s))

    def forward(self, x, idx):
        if self.nb == 1:
            return x @ self.base + self.bbias
        w, b = self.w, self.b
        if self.base is not None:
            w = w + self.base
            b = b + self.bbias
        # (n, din) @ (din, nb*dout) -> (n, nb, dout), then pick the row.
        y = (x @ w.permute(1, 0, 2).reshape(self.din, self.nb * self.dout))
        y = y.view(-1, self.nb, self.dout) + b
        return y.gather(1, idx.view(-1, 1, 1).expand(-1, 1, self.dout)).squeeze(1)


class Bottle(nn.Module):
    """L1 as a squeeze: width -> d (bucketed one way) -> hidden (bucketed another).

    The point of the shape is that the two halves are concatenated *before* the
    up projection, so our side's hidden vector sees their king's contribution
    even though the down projection was chosen by one king each. With `tie` the
    same two tables serve both sides and the their-side concat is reversed,
    which keeps the us/them symmetry exact rather than fitted.

    `mid` inserts a further bucketed linear with no activation around it -- the
    "several bucketed matrices multiplied together" arm. Without any activation
    the whole stack is still one linear map per (bucket, bucket, bucket) cell,
    which is the interesting bit: it is a low-rank factorisation of a table far
    too big to store, not a deeper network.
    """

    def __init__(self, width, d, hidden, fd, fu, fm, code, act, tie):
        super().__init__()
        self.d, self.act = d, act
        mk = lambda nb, di, do: BucketLinear(nb, di, do, code)
        self.down_us = mk(FAM[fd], width, d)
        self.down_them = self.down_us if tie else mk(FAM[fd], width, d)
        self.mid = mk(FAM[fm], 2 * d, 2 * d) if fm else None
        self.mid_them = (self.mid if tie else mk(FAM[fm], 2 * d, 2 * d)) if fm else None
        self.up_us = mk(FAM[fu], 2 * d, hidden)
        self.up_them = self.up_us if tie else mk(FAM[fu], 2 * d, hidden)

    def forward(self, au, at, du_i, dt_i, uu_i, ut_i, mu_i, mt_i):
        a = self.down_us(au, du_i)
        b = self.down_them(at, dt_i)
        if self.act:
            a, b = crelu(a), crelu(b)
        cu = torch.cat([a, b], -1)
        ct = torch.cat([b, a], -1)
        if self.mid is not None:
            cu, ct = self.mid(cu, mu_i), self.mid_them(ct, mt_i)
        return self.up_us(cu, uu_i), self.up_them(ct, ut_i)


# ------------------------------------------------------------------- the net

def crelu(z):
    return torch.clamp(z, 0.0, 1.0)


class WdlNet(nn.Module):
    """The architecture in the module docstring. Emits 3 logits (L, D, W).

    Every knob here is an arm of the experiment:

    `width`     accumulator per perspective (the "256 x 2")
    `hidden`    the per-side layer width (the "32 x 2")
    `b1`, `b2`  which family buckets L1 and L2: "king" (64, per side),
                "mat" (54, per side), "sym" (40, side-invariant), "none" (1).
                Swapping the two is the "material first" arm.
    `head_b`    whether the 64 -> 3 output layer is bucketed too
    `reuse`     one L1/L2 table shared by both sides, or one table each
    `code`      "delta" (base + per-bucket delta) or "onehot" (independent)
    `depth`     2 = both layers, 1 = L1 then straight to the logits,
                3 = an extra shared 64 -> 64 stage before the head
    `psqt`      "none" | "plain" | "mat": the psq -> 3-logit passthrough,
                optionally bucketed by our material
    `skip`      also wire the 64-wide L1 concat straight to the logits
    """

    def __init__(self, width=256, hidden=32, b1="king", b2="mat",
                 head_b="none", reuse=True, code="delta", depth=2,
                 psqt="plain", skip=False, std=0.05,
                 bneck=0, bn_act=False, bn_bd="king", bn_bu="mat", bn_mid="",
                 route=0, route_hard=True, route_proj=False, kadd="",
                 stack="", acts="111", ft_mode="plain", act="crelu"):
        super().__init__()
        # `stack` is the per-side chain: accumulator -> a `b1`-bucketed layer
        # -> a `b2`-bucketed layer -> concat -> head. It differs from the
        # default trunk in that the two sides never mix until the head, which
        # is what makes the all-linear member of the family readable: the whole
        # net is then one linear map per (b1, b2) pair, and `acts` says which of
        # the three crelu positions is switched on. depth is forced to 1 because
        # this mode supplies its own second layer.
        self.stack, self.acts = stack, acts
        if stack:
            depth = 1
        self.width, self.hidden, self.depth = width, hidden, depth
        self.b1, self.b2, self.head_b, self.reuse = b1, b2, head_b, reuse
        king, mat = FAM[b1], FAM[b2]
        self.psqt_mode, self.skip = psqt, skip
        self.wdl = True

        # "halfka": the accumulator table is indexed by (own king square,
        # piece, square) instead of (piece, square), so the king enters through
        # the INPUT rather than through a later layer's bucket. That is 64x768
        # rows instead of 768 -- 3.1M parameters against 49k -- and it is the
        # whole point of the comparison. Rows 0..PAD keep their old meaning so
        # the frozen padding row still works.
        # What happens to the accumulator before the first layer reads it.
        #   crelu  clamp(z, 0, 1)                      -- what we have used
        #   sym    clamp(z, -1, 1)                     -- keeps the sign
        #   pair   split in half, clamp each to (0,1), MULTIPLY -- Stockfish's
        #          fold. It halves the width and makes the first layer see
        #          products of accumulator entries, so the net gets a quadratic
        #          term the other two cannot express at any width.
        # Only the accumulator is affected; later layers stay clamp(0,1), which
        # is what Stockfish does too.
        self.act = act
        self.ft_mode = ft_mode
        wpost = width // 2 if act in ("pair", "gate") else width
        self.ft = bag(width, std,
                      rows=NFEAT + 1 + (32 if ft_mode == "halfkam" else 64) * NFEAT
                      if ft_mode.startswith("halfka") else None)
        self.ft_bias = nn.Parameter(torch.zeros(width))

        # The cheap alternative to bucketing L1: let the bucket shift L1's
        # OUTPUT instead of choosing its weights. `king` costs 64 x hidden
        # parameters against the bucketed layer's 64 x width x hidden -- 2 KB
        # against 512 KB -- and no arithmetic either way. If this recovers most
        # of what `b1="king"` buys, the table is memory we are paying for
        # nothing.
        self.kadd = kadd
        self.kadd_emb = nn.Embedding(FAM[kadd], hidden) if kadd else None
        if self.kadd_emb is not None:
            nn.init.zeros_(self.kadd_emb.weight)

        mk = lambda nb, di, do: BucketLinear(nb, di, do, code)
        self.bneck = bneck
        self.bn_bd, self.bn_bu = bn_bd, bn_bu
        if bneck:
            self.bn_mid = bn_mid
            self.bot = Bottle(wpost, bneck, hidden, bn_bd, bn_bu, bn_mid,
                              code, bn_act, reuse)
            self.l1_us = self.l1_them = None
        elif stack == "kq":
            self.bot = None
            self.l1_us = self.l1_them = None
            self.k_us = mk(FAM[b1], wpost, hidden)
            self.k_them = self.k_us if reuse else mk(FAM[b1], wpost, hidden)
            self.q_us = mk(FAM[b2], hidden, hidden)
            self.q_them = self.q_us if reuse else mk(FAM[b2], hidden, hidden)
        elif stack == "flat":
            # accumulator -> crelu -> logits and nothing else, so the ONLY
            # place king information can enter is `ft_mode` (input side) or
            # `head_b` (output side). That is the minimal pair.
            self.bot = None
            self.l1_us = self.l1_them = None
        else:
            self.bot = None
            self.l1_us = mk(king, wpost, hidden)
            self.l1_them = self.l1_us if reuse else mk(king, wpost, hidden)
        d1 = 2 * wpost if stack == "flat" else 2 * hidden
        if depth >= 2:
            self.l2_us = mk(mat, d1, hidden)
            self.l2_them = self.l2_us if reuse else mk(mat, d1, hidden)
            dout = 2 * hidden
        else:
            self.l2_us = self.l2_them = None
            dout = d1
        self.l25 = nn.Linear(dout, dout) if depth >= 3 else None
        self.head = BucketLinear(FAM[head_b], dout, 3, code)
        with torch.no_grad():
            self.head.base.normal_(std=0.01)
            self.head.bbias.zero_()
            if FAM[head_b] > 1 and code == "onehot":
                self.head.w.normal_(std=0.01)
                self.head.b.zero_()
        if skip:
            self.skiph = nn.Linear(d1, 3, bias=False)
            nn.init.zeros_(self.skiph.weight)
        self.route, self.route_hard = route, route_hard
        if route:
            # A discrete passthrough: the accumulator (or a projection of it)
            # picks ONE row of a `route` x 3 logit table. The forward pass is a
            # hard argmax; the backward pass pretends it was a softmax, which is
            # the straight-through estimator. `softrt` keeps the softmax in the
            # forward pass too and is the control for whether the discreteness
            # is what matters or merely the extra path. Table starts at zero so
            # the arm begins exactly at `base`.
            self.route_lin = nn.Linear(width, route, bias=False) if route_proj else None
            if self.route_lin is None and route != width:
                raise ValueError("route must equal width unless route_proj")
            self.route_tab = nn.Parameter(torch.zeros(route, 3))
        if psqt != "none":
            nb = NB_MAT if psqt == "mat" else 1
            self.psqt = bag(3 * nb, 0.0)
            self.psqt_nb = nb

    # `_swap` turns their-side psqt logits into ours: a side swap exchanges
    # W and L and leaves D alone, so the passthrough is antisymmetric by
    # construction rather than by fitting.
    @staticmethod
    def _swap(p):
        return p.flip(-1)

    def _route(self, z):
        """One row of the logit table, argmax in the forward pass and softmax
        in the backward one (straight-through)."""
        if self.route_lin is not None:
            z = self.route_lin(z)
        p = F.softmax(z, -1)
        if self.route_hard:
            hard = F.one_hot(z.argmax(-1), z.shape[-1]).to(p.dtype)
            p = (hard - p).detach() + p
        return p @ self.route_tab

    def _idx(self, feat, name):
        """(our index, their index) for a bucket family. `sym` is invariant
        under a side swap by construction, so both sides read the same row."""
        if name == "none":
            z = torch.zeros(feat.shape[0], dtype=torch.long, device=feat.device)
            return z, z
        if name == "king":
            return king_sqs(feat)
        if name == "mat":
            return mat_buckets(feat)
        if name == "sym":
            s = b_sym(feat)
            return s, s
        if name.startswith("hash"):
            h = hash_buckets(feat, FAM[name])
            return h, h
        if name in ("king4", "king16"):
            return coarse_king(feat, FAM[name])
        if name == "mat6":
            return coarse_mat(feat)
        if name == "queen":
            return queen_sqs(feat)
        raise KeyError(name)

    def forward(self, feat):
        flip = flip_feat(feat)
        i1u, i1t = self._idx(feat, self.b1)
        i2u, i2t = self._idx(feat, self.b2)
        mu, mt = mat_buckets(feat)

        zu = self.ft(self._ftin(feat)) + self.ft_bias
        zt = self.ft(self._ftin(flip)) + self.ft_bias

        if self.stack == "flat":
            h = z = torch.cat([crelu(zu), crelu(zt)], -1)
            return self._out(feat, flip, h, z, zu, zt, mu, mt)
        if self.stack == "kq":
            a0, a1, a2 = (c == "1" for c in self.acts)
            au = crelu(zu) if a0 else zu
            at = crelu(zt) if a0 else zt
            ku_, kt_ = self.k_us(au, i1u), self.k_them(at, i1t)
            if a1:
                ku_, kt_ = crelu(ku_), crelu(kt_)
            qu_, qt_ = self.q_us(ku_, i2u), self.q_them(kt_, i2t)
            if a2:
                qu_, qt_ = crelu(qu_), crelu(qt_)
            # `h` is the L1 concat everywhere else in this file, so here it is
            # the KING-layer output, not the queen-layer output. The difference
            # is the whole point of `skip` in this stack: the linear net is
            # z0 K_b Q_c H, and the product K_b Q_c = KQ + K dQ_c + dK_b Q +
            # dK_b dQ_c ties the same dK_b across all 65 queen buckets. That
            # 64x65 array of maps is a multiplicative factorisation, and an
            # additive bypass is not in its image -- so a skip off `h` really
            # does add functions. Wiring it off the queen output instead makes
            # it W_head + W_skip, one matrix, and tests nothing.
            h = torch.cat([ku_, kt_], -1)
            z = torch.cat([qu_, qt_], -1)
            return self._out(feat, flip, h, z, zu, zt, mu, mt)

        au, at = self._act(zu), self._act(zt)
        if self.bot is not None:
            du_i, dt_i = self._idx(feat, self.bn_bd)
            uu_i, ut_i = self._idx(feat, self.bn_bu)
            mu_i, mt_i = self._idx(feat, self.bn_mid or "none")
            hu, ht = self.bot(au, at, du_i, dt_i, uu_i, ut_i, mu_i, mt_i)
        else:
            hu, ht = self.l1_us(au, i1u), self.l1_them(at, i1t)
        if self.kadd_emb is not None:
            ku, kt = self._idx(feat, self.kadd)
            hu = hu + self.kadd_emb(ku)
            ht = ht + self.kadd_emb(kt)
        h = torch.cat([crelu(hu), crelu(ht)], -1)
        z = h
        if self.depth >= 2:
            z = torch.cat([crelu(self.l2_us(h, i2u)),
                           crelu(self.l2_them(h, i2t))], -1)
        if self.l25 is not None:
            z = crelu(self.l25(z))
        return self._out(feat, flip, h, z, zu, zt, mu, mt)

    def _act(self, z):
        """The accumulator activation. `pair` returns HALF the input width."""
        if self.act == "sym":
            return torch.clamp(z, -1.0, 1.0)
        if self.act == "pair":
            a, b = z.chunk(2, dim=-1)
            return torch.clamp(a, 0.0, 1.0) * torch.clamp(b, 0.0, 1.0)
        if self.act == "gate":
            # value x gate, as in the gated units LLMs use: the first half keeps
            # its sign in (-1, 1) and the second half is a non-negative gate in
            # (0, 1). `pair` cannot represent a negative product; this can.
            a, b = z.chunk(2, dim=-1)
            return torch.clamp(a, -1.0, 1.0) * torch.clamp(b, 0.0, 1.0)
        return crelu(z)

    def _ftin(self, f):
        """Feature indices to read the accumulator table with.

        Plain: the 768 (piece, square) indices the loader already produces.
        halfka: shifted past the padding row and offset by 64 x (own king
        square), which is what makes the table king-conditioned. `f` is always
        one perspective's own view, so `king_sqs(f)[0]` is that perspective's
        own king, mirrored for the flipped board exactly as everywhere else.
        """
        if self.ft_mode not in ("halfka", "halfkam"):
            return f
        k = king_sqs(f)[0]
        g = f.clamp(max=NFEAT - 1)
        if self.ft_mode == "halfkam":
            # Mirror the whole board when our king is on files a-d, so the king
            # is always on the right. That halves the table to 32 x 768 and
            # makes the two mirror images of a position share weights instead
            # of learning it twice. `x ^ 7` flips the file: it touches only the
            # low three bits, so piece type and side are untouched.
            m = (k % 8) < 4
            k = torch.where(m, k ^ 7, k)
            g = torch.where(m.unsqueeze(1), g ^ 7, g)
            k = (k // 8) * 4 + (k % 8) - 4          # 0..31
        idx = NFEAT + 1 + k.unsqueeze(1) * NFEAT + g
        return torch.where(f != PAD, idx, torch.full_like(f, PAD))

    def _out(self, feat, flip, h, z, zu, zt, mu, mt):
        """Head, skip, route and psqt passthrough -- shared by every trunk."""
        out = self.head(z, self._idx(feat, self.head_b)[0])
        if self.skip:
            out = out + self.skiph(h)
        if self.route:
            out = out + self._route(zu) - self._swap(self._route(zt))
        if self.psqt_mode != "none":
            pu = self.psqt(feat).view(-1, self.psqt_nb, 3)
            pt_ = self.psqt(flip).view(-1, self.psqt_nb, 3)
            if self.psqt_nb == 1:
                out = out + pu[:, 0] - self._swap(pt_[:, 0])
            else:
                g = lambda p, i: p.gather(1, i.view(-1, 1, 1).expand(-1, 1, 3))[:, 0]
                out = out + g(pu, mu) - self._swap(g(pt_, mt))
        return out


# --------------------------------------------------------------- loss / eval

def wdl_from_logits(logits):
    """(W, D, L) from logits ordered (L, D, W)."""
    p = F.softmax(logits, -1)
    return p[:, 2], p[:, 1], p[:, 0]


def expected(logits):
    W, D, _ = wdl_from_logits(logits)
    return W + D / 2


def target_dist(score, result, teacher, feat, lam, ext=None, temp=1.0):
    """lam * teacher distribution + (1-lam) * the game outcome, as (L, D, W).

    Mixing rather than training on the outcome alone is deliberate: the outcome
    is one bit per *game* smeared over ~50 correlated plies, while the teacher
    score is a per-position label. lam keeps the same meaning it has everywhere
    else in this repo, so a WDL arm is comparable with the scalar arms.

    `ext` replaces the teacher with a precomputed (L, D, W) per record -- a
    Leela net's own distribution, say. It enters at exactly the point the
    sigmoid teacher would have, so the two differ in the distribution and in
    nothing else. `lam` and the outcome mix are untouched, which is what makes
    the arms comparable."""
    cls = (result * 2).round().long()
    oh = F.one_hot(cls, 3).float()
    if lam <= 0.0:
        return oh
    if ext is not None:
        t = ext
    else:
        tW, tD, tL = teacher(score, feat)
        t = torch.stack([tL, tD, tW], 1)
        # `temp` > 1 flattens the teacher toward uniform, < 1 sharpens it. It
        # exists as a CONTROL, not as a knob to tune: an external teacher is
        # both a different shape and a different entropy, and a student that
        # improves only because its target got softer has learned nothing
        # about the teacher. Matching the entropy and re-running separates the
        # two. It is applied to the teacher only -- the outcome one-hot is the
        # ground truth and has no temperature.
        if temp != 1.0:
            t = F.softmax(t.clamp_min(1e-9).log() / temp, dim=1)
    return lam * t + (1 - lam) * oh


def mean_entropy(t):
    """Mean entropy of a batch of target distributions, in nats.

    Printed beside every teacher's ce_out because "the student got better"
    and "the target got softer" are different claims and this is the number
    that tells them apart."""
    return -(t.clamp_min(1e-9) * t.clamp_min(1e-9).log()).sum(1).mean()


def ce(logits, tgt):
    return -(tgt * F.log_softmax(logits, -1)).sum(1).mean()
