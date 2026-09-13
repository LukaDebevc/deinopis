"""Scaling, the shared embedding table, and the loss everything is fitted in.

Split out of train.py because router.py and models.py both need `bag` and
`loss_fn`, and importing train.py for them would be circular."""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import NFEAT, PAD   # noqa: E402


# ---------------------------------------------------------------- scaling K

def fit_k(score, result):
    """Fit K in P(win) = sigmoid(score/K) by maximum likelihood on the results."""
    s = score.double()
    z = result.double()
    logk = torch.tensor(math.log(400.0), dtype=torch.float64,
                        device=s.device, requires_grad=True)
    opt = torch.optim.LBFGS([logk], max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(s / logk.exp())
        eps = 1e-9
        nll = -(z * (p + eps).log() + (1 - z) * (1 - p + eps).log()).mean()
        nll.backward()
        return nll

    nll = opt.step(closure)
    return logk.exp().item(), float(nll.detach())


# ---------------------------------------------------------------- pieces

def bag(width, std, rows=None):
    """EmbeddingBag over the 768 features plus one frozen padding row.

    `rows` extends the table past the padding row for extra input features
    (see EXTRAS); the padding row keeps its index either way.

    Positions have between 3 and 32 pieces; the unpacker pads every position to
    32 slots pointing spare slots at PAD, whose row is held at zero and gets no
    gradient. That keeps the batch a fixed shape, which is what lets the whole
    unpack run on the GPU without a device sync.
    """
    e = nn.EmbeddingBag(rows or NFEAT + 1, width, mode="sum", padding_idx=PAD)
    nn.init.normal_(e.weight, std=std)
    with torch.no_grad():
        e.weight[PAD].zero_()
    return e




# ---------------------------------------------------------------- train loop

def loss_fn(pred, score, result, K, lam, teacher=None, feat=None):
    """Scalar loss. `teacher` swaps the teacher-side term for the WDL model's
    expected score, which is measured to be about 2x better calibrated on E
    than sigmoid(s/K) (per-bin |dE| 0.0118 vs 0.0222, n=3M held out).

    That swap exists so the WDL head can be tested WITHOUT confounding it with
    a better teacher: arm B is this scalar model on the WDL teacher's E, so
    C - B isolates the head and B - A isolates the teacher."""
    if teacher is None:
        tgt_teacher = torch.sigmoid(score / K)
    else:
        tW, tD, _ = teacher(score, feat)
        tgt_teacher = tW + tD / 2
    target = lam * tgt_teacher + (1 - lam) * result
    return ((torch.sigmoid(pred / K) - target) ** 2).mean()


def outcome_nll(E, result):
    """NLL of the actual game outcome under the arm's expected score.

    The PRIMARY yardstick. `loss_fn` scores against sigmoid(s/K), which is the
    target arm A trains on, so it is biased against every arm that trains on
    anything else. This one is unbiased across arms and is a proper scoring
    rule; it is also noisier, so it needs its own floor (two seeds of arm A)
    before any gap in it counts."""
    E = E.clamp(1e-6, 1 - 1e-6)
    return -(result * torch.log(E) + (1 - result) * torch.log(1 - E)).mean()
