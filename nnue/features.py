"""Extra input features, and the two perspectives.

EXTRAS is mutated at import time by make_regions(); Extras() reads it. Both
live here so a caller only ever mutates it through a function in this module."""

import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import NFEAT, PAD   # noqa: E402

def _parts(feat):
    """(valid, side, piece type, square) from padded feature indices."""
    valid = feat != PAD
    f = feat.clamp(max=NFEAT - 1)
    return valid, f // 384, (f % 384) // 64, f % 64


def _present(feat, mask, key, width):
    """Per-row count of `mask` grouped into `width` slots by `key`."""
    n = feat.shape[0]
    key = torch.where(mask, key, torch.zeros_like(key))
    out = torch.zeros(n, width, device=feat.device, dtype=torch.float32)
    out.scatter_add_(1, key, mask.float())
    return out


# ---------------------------------------------------------------- extra inputs

"""Extra *input* features, appended past the 768 piece-square ones.

The distinction from a bucket matters. A bucket picks which reader row scales an
accumulator that has already been computed, so it can only rescale what the
piece features found. An input feature enters the accumulator itself, so the
quadratic form multiplies it against every piece on the board: "our rook on d1
given the d-file is half-open" is a term the bucketed model cannot write and
this one can.

The price is that these indices are not slow-moving. A base feature changes
twice per move; a tile-occupancy feature changes on any move into or out of the
tile, and tiles overlap. The number of *active* extra features is the number of
extra gathers per accumulator update, and is recorded on each family.

Every family returns (N, k) indices already offset into the shared table.
"""


def _rank_file(feat):
    valid, side, pt, sq = _parts(feat)
    return valid, side, pt, sq // 8, sq % 8


def x_pawnfile(feat, off):
    """Per file: how far our most advanced pawn has come, and theirs.

    8 features, one per file, each 8x8 = 64 states: our furthest pawn's rank
    (0 = none, a pawn is never on rank 0) crossed with their furthest (7 = none).
    Passed, backward, blocked and half-open are all functions of exactly this
    pair, and a PSQT can see none of them. 8 gathers.
    """
    valid, side, pt, rk, fl = _rank_file(feat)
    n, dev = feat.shape[0], feat.device
    ours = (valid & (pt == 0) & (side == 0)).long()
    them = (valid & (pt == 0) & (side == 1)).long()
    up = torch.zeros(n, 8, device=dev, dtype=torch.long)
    up.scatter_reduce_(1, fl * ours, rk * ours, "amax")
    dn = torch.full((n, 8), 7, device=dev, dtype=torch.long)
    dn.scatter_reduce_(1, fl * them, torch.where(them > 0, rk, torch.full_like(rk, 7)), "amin")
    return off + torch.arange(8, device=dev) * 64 + up * 8 + dn


x_pawnfile.size = 8 * 64
x_pawnfile.active = 8


def x_pawnpair(feat, off):
    """Adjacent-file pawn presence, crossed: 7 pairs x 16 states.

    (our pawn on f, ours on f+1, theirs on f, theirs on f+1). Chains and
    isolated pawns are two-file facts; `pawnfile` sees each file on its own.
    """
    valid, side, pt, rk, fl = _rank_file(feat)
    n, dev = feat.shape[0], feat.device
    is_p = valid & (pt == 0)
    has = torch.zeros(n, 16, device=dev)
    has.scatter_add_(1, torch.where(is_p, side * 8 + fl, torch.zeros_like(fl)),
                     is_p.float())
    has = (has > 0).long()
    u, t = has[:, :8], has[:, 8:]
    code = ((u[:, :7] * 2 + u[:, 1:]) * 2 + t[:, :7]) * 2 + t[:, 1:]
    return off + torch.arange(7, device=dev) * 16 + code


x_pawnpair.size = 7 * 16
x_pawnpair.active = 7


# The 12-square plus centred on the middle four: d4 e4 d5 e5, their orthogonal
# neighbours d3 e3 d6 e6, and c4 c5 f4 f5. Ranks are from the mover's side.
_PLUS = [(3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4),
         (5, 3), (5, 4), (3, 2), (4, 2), (3, 5), (4, 5)]


def x_centre(feat, off):
    """2^12 occupancy patterns over the central plus, one feature per side.

    Piece identity is dropped on purpose. The quadratic form can multiply the
    pattern against a specific piece-square feature, so identity comes back
    where it earns its keep while the pattern itself stays combinatorial: 4096
    states for two gathers, against the 12 features a linear model would get.
    """
    valid, side, pt, sq = _parts(feat)
    n, dev = feat.shape[0], feat.device
    bitof = torch.full((64,), -1, device=dev, dtype=torch.long)
    for i, (r, f) in enumerate(_PLUS):
        bitof[r * 8 + f] = i
    b = bitof[sq]
    inside = valid & (b >= 0)
    out = torch.zeros(n, 2, device=dev)
    out.scatter_add_(1, torch.where(inside, side, torch.zeros_like(side)),
                     torch.where(inside, torch.pow(2.0, b.clamp(min=0).float()),
                                 torch.zeros(1, device=dev)))
    return off + torch.tensor([0, 1 << 12], device=dev) + out.long()


x_centre.size = 2 * (1 << 12)
x_centre.active = 2


def make_tiles(h, w, sv, sh):
    """Overlapping h x w windows over the board, each an occupancy mask per side.

    A stride smaller than the window is the point: every square sits in several
    windows, so a pattern straddling one window's edge is still seen whole by
    another. Cost is `active` extra gathers per accumulator rebuild, and unlike
    the piece features these change on any move into or out of a window -- so a
    quiet move touches ~2 base features and up to 2 * (h/sv)(w/sh) tile ones.
    """
    origins = [(r, f) for r in range(0, 8 - h + 1, sv)
               for f in range(0, 8 - w + 1, sh)]
    nt, k = len(origins), h * w

    def fn(feat, off):
        valid, side, pt, sq = _parts(feat)
        n, dev = feat.shape[0], feat.device
        rk, fl = sq // 8, sq % 8
        out = torch.zeros(n, 2 * nt, device=dev)
        zero = torch.zeros(1, device=dev)
        for t, (r0, f0) in enumerate(origins):
            inside = (valid & (rk >= r0) & (rk < r0 + h)
                      & (fl >= f0) & (fl < f0 + w))
            bit = (rk - r0) * w + (fl - f0)
            out.scatter_add_(
                1, torch.where(inside, side * nt + t, torch.zeros_like(side)),
                torch.where(inside, torch.pow(2.0, bit.clamp(min=0).float()), zero))
        return off + torch.arange(2 * nt, device=dev) * (1 << k) + out.long()

    fn.size = 2 * nt * (1 << k)
    fn.active = 2 * nt
    return fn


def x_kingpair(feat, off):
    """Our king square x theirs as an INPUT feature rather than a bucket.

    Same 4096 index as the `kings` bucket family. As a bucket it rescales the
    accumulator; as an input it multiplies against every piece, which is the
    king-relative indexing an NNUE buys with HalfKP -- for one extra gather
    instead of a 4096-fold table on every feature.
    """
    valid, side, pt, sq = _parts(feat)
    is_k = valid & (pt == 5)
    us = (sq * (is_k & (side == 0))).sum(1)
    them = (sq * (is_k & (side == 1))).sum(1)
    return (off + us * 64 + them).unsqueeze(1)


x_kingpair.size = 64 * 64
x_kingpair.active = 1


EXTRAS = {
    "pawnfile": x_pawnfile,
    "pawnpair": x_pawnpair,
    "centre":   x_centre,
    "kingpair": x_kingpair,
    "tile23":   make_tiles(2, 3, 2, 1),      # 24 windows x 64 states
    "tile22":   make_tiles(2, 2, 1, 1),      # 49 windows x 16 states
    "tile43":   make_tiles(4, 3, 2, 1),      # 18 windows x 4096 states
}


def make_region(squares, seed=None):
    """Occupancy bitmask over a fixed square set, one feature per side.

    The generalisation of `x_centre`. `seed`, if given, replaces the square set
    with a random one of the same size -- the control for "is it *this* region,
    or would any region of this size do", which is the only confound the arms
    themselves cannot separate.
    """
    if seed is not None:
        import random
        squares = [(s // 8, s % 8) for s in
                   random.Random(seed).sample(range(64), len(squares))]
    k = len(squares)

    def fn(feat, off):
        valid, side, pt, sq = _parts(feat)
        n, dev = feat.shape[0], feat.device
        bitof = torch.full((64,), -1, device=dev, dtype=torch.long)
        for i, (r, f) in enumerate(squares):
            bitof[r * 8 + f] = i
        b = bitof[sq]
        inside = valid & (b >= 0)
        out = torch.zeros(n, 2, device=dev)
        out.scatter_add_(1, torch.where(inside, side, torch.zeros_like(side)),
                         torch.where(inside, torch.pow(2.0, b.clamp(min=0).float()),
                                     torch.zeros(1, device=dev)))
        return off + torch.tensor([0, 1 << k], device=dev) + out.long()

    fn.size = 2 * (1 << k)
    fn.active = 2
    return fn


def make_regions(sets):
    """Several regions side by side, sharing one offset block."""
    parts = [make_region(s) for s in sets]

    def fn(feat, off):
        out, o = [], off
        for p in parts:
            out.append(p(feat, o))
            o += p.size
        return torch.cat(out, dim=1)

    fn.size = sum(p.size for p in parts)
    fn.active = sum(p.active for p in parts)
    return fn


def x_pawnrow(feat, off):
    """Luka's pawn rows: per file, the full occupancy pattern down the file.

    Six ranks a pawn can stand on, each empty / ours / theirs (both is
    impossible on one square), so 3^6 = 729 states per file and 8 gathers. This
    strictly contains `pawnfile`, which keeps only the most advanced pawn of
    each side, so the pair measures what doubled pawns and the pawns behind the
    front one are worth.
    """
    valid, side, pt, rk, fl = _rank_file(feat)
    n, dev = feat.shape[0], feat.device
    on = valid & (pt == 0) & (rk >= 1) & (rk <= 6)
    cell = torch.zeros(n, 48, device=dev)
    cell.scatter_add_(1, torch.where(on, fl * 6 + (rk - 1), torch.zeros_like(fl)),
                      torch.where(on, (side + 1).float(), torch.zeros(1, device=dev)))
    p3 = (3 ** torch.arange(6, device=dev))
    code = (cell.long().clamp(max=2).view(n, 8, 6) * p3).sum(-1)
    return off + torch.arange(8, device=dev) * 729 + code


x_pawnrow.size = 8 * 729
x_pawnrow.active = 8


def x_kingzone(feat, off):
    """The 3x3 around each king, occupancy by side: 4 features of 2^9.

    (our king's neighbourhood as filled by our pieces / by theirs, then the same
    around their king). King safety is the standard example of an interaction a
    PSQT cannot see, and this is the cheapest honest encoding of it: 4 gathers.

    The king square itself is not in the index, so "no piece to my west" and
    "the board ends to my west" look the same here. `kingpair` supplies the
    missing square and the quadratic form can multiply the two together, which
    is why the pair is worth running alongside each other.
    """
    valid, side, pt, rk, fl = _rank_file(feat)
    n, dev = feat.shape[0], feat.device
    is_k = valid & (pt == 5)
    krk = (rk * (is_k & (side == 0))).sum(1, keepdim=True)
    kfl = (fl * (is_k & (side == 0))).sum(1, keepdim=True)
    trk = (rk * (is_k & (side == 1))).sum(1, keepdim=True)
    tfl = (fl * (is_k & (side == 1))).sum(1, keepdim=True)
    out = torch.zeros(n, 4, device=dev)
    for j, (zr, zf) in enumerate(((krk, kfl), (trk, tfl))):
        dr, df = rk - zr, fl - zf
        near = valid & (dr.abs() <= 1) & (df.abs() <= 1)
        bit = (dr + 1) * 3 + (df + 1)
        out.scatter_add_(1, torch.where(near, j * 2 + side, torch.zeros_like(side)),
                         torch.where(near, torch.pow(2.0, bit.clamp(min=0).float()),
                                     torch.zeros(1, device=dev)))
    return off + torch.arange(4, device=dev) * 512 + out.long()


x_kingzone.size = 4 * 512
x_kingzone.active = 4


# Four 3x4 quarters: our queenside and kingside, then theirs. Luka's "flanks",
# 12 squares each, so 2^12 states per region per side.
_QUARTERS = [[(r, f) for r in rs for f in fs]
             for rs in ((0, 1, 2, 3), (4, 5, 6, 7))
             for fs in ((0, 1, 2), (5, 6, 7))]

EXTRAS.update({
    "pawnrow":  x_pawnrow,
    "kingzone": x_kingzone,
    "flanks":   make_regions(_QUARTERS),
    "centre2":  make_region(_PLUS),                  # must equal `centre`
    "randcell": make_region(_PLUS, seed=7),          # control for `centre`
})


class Extras:
    """Bookkeeping for a chosen set of extra-input families.

    They share the embedding table with the piece features, starting one row
    past the padding row, so a family enters the linear term and the accumulator
    with no second code path and the warm start (which writes only rows 0..767)
    still leaves them at zero.
    """

    def __init__(self, names):
        self.names = [n for n in names if n and n != "none"]
        self.offs, o = [], NFEAT + 1
        for n in self.names:
            self.offs.append(o)
            o += EXTRAS[n].size
        self.rows = o
        self.active = sum(EXTRAS[n].active for n in self.names)

    def __call__(self, feat):
        if not self.names:
            return feat
        return torch.cat([feat] + [EXTRAS[n](feat, o)
                                   for n, o in zip(self.names, self.offs)], dim=1)

    def __repr__(self):
        return (f"{'+'.join(self.names) or 'none'} "
                f"(+{self.rows - NFEAT - 1:,} rows, {self.active} gathers)")




# ---------------------------------------------------------------- sides

def side_split(feat):
    """(ours, theirs): two copies of the padded piece list, other side PADded out.

    Features are laid out side*384 + type*64 + square with side 0 = the player to
    move, so the split is a comparison on the index and nothing is recomputed."""
    pad = torch.full_like(feat, PAD)
    return (torch.where(feat < 384, feat, pad),
            torch.where((feat >= 384) & (feat != PAD), feat, pad))


def flip_feat(feat):
    """The same board seen from the other chair: colours swapped, ranks mirrored.

    This is a permutation of the 768 indices, not a symmetry of the label.
    Records are stm-canonical, so flipping maps a position to the one where the
    *other* player is to move -- the two differ by a tempo, worth tens of
    centipawns. It is the map a two-perspective accumulator applies to build its
    second half, which is why that tie is a prior and not a free lunch."""
    f = feat.clamp(max=NFEAT - 1)
    side, pt, sq = f // 384, (f % 384) // 64, f % 64
    g = (1 - side) * 384 + pt * 64 + (sq ^ 56)
    return torch.where(feat == PAD, feat, g)


