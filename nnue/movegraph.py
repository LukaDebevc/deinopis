"""The empty-board move graph, and the lazily built singleton around it."""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import NFEAT, PAD   # noqa: E402


# ---------------------------------------------------------------- move graph

def _rays(sq, steps, sliding):
    """Squares reachable from `sq` by `steps` on an empty board."""
    r, f, out = sq // 8, sq % 8, []
    for dr, df in steps:
        rr, ff = r + dr, f + df
        while 0 <= rr < 8 and 0 <= ff < 8:
            out.append(rr * 8 + ff)
            if not sliding:
                break
            rr, ff = rr + dr, ff + df
    return out


_STEPS = {
    1: ([(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)], False),
    2: ([(1, 1), (1, -1), (-1, 1), (-1, -1)], True),
    3: ([(1, 0), (-1, 0), (0, 1), (0, -1)], True),
    4: ([(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)], True),
    5: ([(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)], False),
}


def move_graph():
    """`from row -> reachable rows` for every piece feature, on an empty board.

    This is the object a stability penalty needs.  A quiet move changes the
    router's logit by exactly `w[to] - w[from]`, so "how much does one move
    move the logit" is a sum over the edges of THIS graph and nothing else --
    not a variance over squares, which would also charge the rule for
    distinguishing squares no single move connects.

    Records are stm-canonical, so side 0 is the mover and its pawns go up the
    board; side 1's pawns go down.  Castling is absent (rights are not stored),
    which understates king edges exactly as `refresh.py` says it does.

    Returns (edges, dest, deg): `edges` (M,2) row pairs for the penalty;
    `dest` (rows, D) and `deg` (rows,) for the flip diagnostic.
    """
    per = {}
    for side in (0, 1):
        for pt in range(6):
            for sq in range(64):
                base = side * 384 + pt * 64
                if pt == 0:                       # pawn
                    if sq < 8 or sq >= 56:
                        tos = []
                    else:
                        d = 8 if side == 0 else -8
                        r = sq // 8
                        tos = [sq + d]
                        if (side == 0 and r == 1) or (side == 1 and r == 6):
                            tos.append(sq + 2 * d)
                        f = sq % 8
                        if f > 0:
                            tos.append(sq + d - 1)
                        if f < 7:
                            tos.append(sq + d + 1)
                        tos = [t for t in tos if 0 <= t < 64]
                else:
                    steps, sliding = _STEPS[pt]
                    tos = _rays(sq, steps, sliding)
                per[base + sq] = [base + t for t in tos]
    rows = NFEAT + 1
    D = max(len(v) for v in per.values())
    dest = np.zeros((rows, D), dtype=np.int64)
    deg = np.zeros(rows, dtype=np.int64)
    edges = []
    for r, tos in per.items():
        deg[r] = len(tos)
        for i, t in enumerate(tos):
            dest[r, i] = t
        edges += [(r, t) for t in tos]
    return (torch.tensor(edges, dtype=torch.long),
            torch.from_numpy(dest), torch.from_numpy(deg))


_GRAPH = None


def graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = move_graph()
    return _GRAPH


