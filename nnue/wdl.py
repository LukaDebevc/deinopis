"""The win/draw/loss head and the teacher that supervises it."""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import NFEAT, PAD   # noqa: E402


from common import bag   # noqa: E402
from features import _parts   # noqa: E402


# ------------------------------------------------------------------ WDL head
#
# The outcome is three-way and the scalar throws away the larger of the two
# axes: at |score| < 20 the expected score moves 0.013 across material while
# the draw rate moves 0.99 -> 0.39 (n=6M, measured 2026-09-03).
#
# The basis matters. Rather than a softmax over (W, D, L), the head emits the
# two coordinates that diagonalise a side swap:
#
#     A = log(W/L)       negates when the sides swap   <- the existing scalar
#     S = log(W L / D^2) invariant when the sides swap  <- one new number
#
# so W <-> L symmetry is STRUCTURAL rather than learned. `flip_batch` already
# trains the scalar to be antisymmetric; in this basis a softmax head's freedom
# to violate that does not exist. The inversion is closed form:
#
#     (1 - D) / D = 2 cosh(A/2) exp(S/2),   W = (1-D) sig(A),  L = (1-D) sig(-A)
#
# and the readout the engine consumes is the risk-neutral one, with no fitted
# constants beyond the K the scalar already uses:
#
#     E = W + D/2 = 1/2 + (1-D) tanh(A/2) / 2,      cp = K logit(E)
#
# A deliberate risk preference is (1-D)**q * tanh(A/2) with q != 1. It is not
# fittable on outcome data -- outcome data can only ever say q = 1 -- so it is
# a search parameter scored by an SPRT, not a training target. It also breaks
# the zero-sum property (draw utility 1/2 is the unique one that keeps
# f(W,D,L) = -f(L,D,W)), so like contempt it belongs at the root with the root
# side's sign fixed, never in a leaf score both sides read.

PIECE_VALUE = (1, 3, 3, 5, 9, 0)


def mat_value(feat):
    """Total material 1/3/3/5/9 over BOTH sides -- a side swap leaves it alone."""
    valid, _, pt, _ = _parts(feat)
    w = torch.tensor(PIECE_VALUE, device=feat.device, dtype=torch.float32)
    return (w[pt] * valid).sum(1)


def b_sym(feat):
    """Side-symmetric coarse bucket for the sharpness head: material x pawns.

    Symmetry is the point. `b_material` codes us and them into separate digits,
    so a side swap changes its index; S must not change. Total material and
    total pawn count are both invariant, and the measurement says they carry
    nearly all of the draw rate (D spans 0.99 -> 0.39 across material and
    1.00 -> 0.48 across pawn count at a fixed score)."""
    valid, _, pt, _ = _parts(feat)
    w = torch.tensor(PIECE_VALUE, device=feat.device, dtype=torch.float32)
    mat = (w[pt] * valid).sum(1)
    pawns = ((pt == 0) & valid).sum(1)
    mb = torch.bucketize(mat, torch.tensor([16., 24., 32., 42., 52., 62., 72.],
                                           device=feat.device))
    pb = torch.bucketize(pawns.float(), torch.tensor([3., 6., 9., 12.],
                                                     device=feat.device))
    return mb * 5 + pb


B_SYM_N = 8 * 5


class WDL(nn.Module):
    """Any scalar model + a side-symmetric sharpness head.

    `inner` is untouched and keeps emitting one number, which is now read as
    A = log(W/L) in units of K. `sharp` is a 40-entry table over (material,
    pawns) -- deliberately tiny, because the whole claim is that S is a
    property of the side-symmetric part of the position and needs no
    accumulator. If a 40-entry table is not enough, that is a measurement, and
    the next step is a wider symmetric head, not a softmax."""

    def __init__(self, inner, K):
        super().__init__()
        self.inner = inner
        self.K = K
        self.wdl = True
        self.sharp = nn.Embedding(B_SYM_N, 1)
        nn.init.constant_(self.sharp.weight, -4.0)   # start drawish, as the data is
        self.scale = getattr(inner, "scale", 1.0)

    def release_buckets(self):
        if hasattr(self.inner, "release_buckets"):
            self.inner.release_buckets()

    def forward(self, feat):
        A = self.inner(feat) / self.K
        S = self.sharp(b_sym(feat)).squeeze(-1)
        return A, S


def wdl_probs(A, S):
    """(W, D, L) from the two coordinates. Exact inverse of the basis above."""
    # (1-D)/D = 2 cosh(A/2) exp(S/2); work in logs for the tails.
    log_odds = math.log(2.0) + torch.log(torch.cosh(A.clamp(-30, 30) / 2)) + S / 2
    u = torch.sigmoid(log_odds)                   # u = 1 - D
    W = u * torch.sigmoid(A)
    L = u * torch.sigmoid(-A)
    return W, (1 - u), L


def wdl_expected(A, S):
    """E = W + D/2, the risk-neutral summary. q = 1 by construction."""
    W, D, L = wdl_probs(A, S)
    return W + D / 2


def wdl_cp(A, S, K):
    """The scalar the engine consumes. No fitted constants beyond K."""
    E = wdl_expected(A, S).clamp(1e-6, 1 - 1e-6)
    return K * torch.log(E / (1 - E))


class TeacherWDL:
    """P(W, D, L | teacher score, material), Stockfish's functional form refit.

    Needed so that `lam` keeps its meaning. The scalar loss blends the teacher
    score with the game result at lam = 0.9; the three-way loss must blend a
    teacher *distribution* with the result, or switching head silently also
    switches to a nearly pure-result target and the arms stop being comparable.

    Stockfish's shipped constants are NOT usable here: on this corpus they are
    about 3x overconfident (b wants a factor 3.19), because they are fit to LTC
    Fishtest games from the root and this is mid-game plies of fast selfplay.
    Material dependence is essential despite being worth nothing for E: it is
    the whole of D.

    Two forms. `cubic` is Stockfish's shape with a(m), b(m) refit here. `table`
    is a (2, B_SYM_N) tensor giving a and b per `b_sym` bucket -- the SAME
    partition the S head reads, so teacher and head have matching resolution
    and neither is limited by the other's bucketing."""

    def __init__(self, a, b, table=None):
        self.a, self.b, self.table = a, b, table

    def __call__(self, score, feat):
        if self.table is not None:
            k = b_sym(feat)
            a, b = self.table[0][k], self.table[1][k].clamp(min=5.0)
        else:
            m = (mat_value(feat).clamp(17, 78) / 58.0)
            poly = lambda c: ((c[0] * m + c[1]) * m + c[2]) * m + c[3]
            a, b = poly(self.a), poly(self.b).abs().clamp(min=5.0)
        W = torch.sigmoid((score - a) / b)
        L = torch.sigmoid((-score - a) / b)
        return W, (1 - W - L).clamp(min=1e-6), L


def loss_wdl(A, S, score, result, teacher, feat, lam):
    """Cross-entropy against lam * teacher distribution + (1-lam) * outcome.

    `result` arrives as 0 / 0.5 / 1 (the scalar convention), so the three-way
    label is round(2 * result) exactly -- and `flip_batch`'s `1 - result` is
    the correct L <-> W swap with no extra code."""
    W, D, L = wdl_probs(A, S)
    tW, tD, tL = teacher(score, feat)
    cls = (result * 2).round().long()
    oh = torch.nn.functional.one_hot(cls, 3).float()          # (B, 3) = L, D, W
    tgt = lam * torch.stack([tL, tD, tW], 1) + (1 - lam) * oh
    pred = torch.stack([L, D, W], 1).clamp(min=1e-7)
    pred = pred / pred.sum(1, keepdim=True)
    return -(tgt * torch.log(pred)).sum(1).mean()


