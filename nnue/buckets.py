"""Bucket RULES: fixed functions from a position to a bucket index.

FAMILIES, _COARSE, _STAGED and _MERGED are mutated in place by the
register_* helpers below. Callers must go through those helpers -- rebinding
the name from another module would leave this module's readers on the old
dict."""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import NFEAT, PAD   # noqa: E402
from features import _parts, _present   # noqa: E402


# ---------------------------------------------------------------- buckets

"""Bucket families.

A bucket is a hard function of the position that selects *which* output weights
to read. The rule that makes it deployable: the bucket may only change the
**read**, never the accumulator. `vec = act(V x)` stays bucket-independent, so
it is still incrementally updatable; a capture that changes the bucket costs a
different lookup, not a rebuild.

Every index is computed from the same (N,32) feature tensor the model sees, so
there is no second data path to keep in sync.
"""


def b_none(feat):
    return torch.zeros(feat.shape[0], dtype=torch.long, device=feat.device)


def b_bishops(feat):
    """(none | light | dark | both)^2, canonicalised under a light/dark swap.

    Mirroring files maps legal positions to legal positions and flips every
    square's colour, so "light" carries no meaning by itself -- only *which
    bishops share a colour* does. Folding each index against its colour-swapped
    twin makes the bucket exactly invariant and collapses 16 slots to 10 used.
    """
    valid, side, pt, sq = _parts(feat)
    is_b = valid & (pt == 2)
    col = ((sq // 8) + (sq % 8)) & 1
    has = _present(feat, is_b, side * 2 + col, 4) > 0
    us = has[:, 0].long() + 2 * has[:, 1].long()
    them = has[:, 2].long() + 2 * has[:, 3].long()
    swap = lambda b: (b & 1) * 2 + (b >> 1)
    return torch.minimum(us * 4 + them, swap(us) * 4 + swap(them))


def b_kings(feat):
    """64 x 64 on the two king squares -- the NNUE king-bucket idea, un-coarsened."""
    valid, side, pt, sq = _parts(feat)
    is_k = valid & (pt == 5)
    us = (sq * (is_k & (side == 0))).sum(1)
    them = (sq * (is_k & (side == 1))).sum(1)
    return us * 64 + them


def b_queens(feat):
    valid, side, pt, sq = _parts(feat)
    is_q = valid & (pt == 4)
    us = (is_q & (side == 0)).sum(1).clamp(max=2)
    them = (is_q & (side == 1)).sum(1).clamp(max=2)
    return us * 3 + them


def b_material(feat):
    """[rooks 0/1/2+] x [light bishop] x [dark bishop] x [queen], squared."""
    valid, side, pt, sq = _parts(feat)
    col = ((sq // 8) + (sq % 8)) & 1
    rook = _present(feat, valid & (pt == 3), side, 2).clamp(max=2).long()
    queen = (_present(feat, valid & (pt == 4), side, 2) > 0).long()
    bis = _present(feat, valid & (pt == 2), side * 2 + col, 4) > 0
    code = lambda s: ((rook[:, s] * 2 + bis[:, 2 * s].long()) * 2
                      + bis[:, 2 * s + 1].long()) * 2 + queen[:, s]
    return code(0) * 24 + code(1)


def b_count(feat):
    """Standard NNUE output bucketing: total pieces, 8 bins. Control arm."""
    return (((feat != PAD).sum(1) - 2).clamp(min=0) // 4).clamp(max=7)


def _king_pair(feat, coarse):
    """Both king squares, each mapped through `coarse` to a region id."""
    valid, side, pt, sq = _parts(feat)
    is_k = valid & (pt == 5)
    us = (sq * (is_k & (side == 0))).sum(1)
    them = (sq * (is_k & (side == 1))).sum(1)
    n = int(coarse(torch.arange(64, device=feat.device)).max()) + 1
    return coarse(us) * n + coarse(them)


def _c16(sq):
    """2x2 tiles: the board as a 4x4 grid of quads."""
    return (sq // 16) * 4 + (sq % 8) // 2


def _c4(sq):
    """Quadrants."""
    return (sq // 32) * 2 + (sq % 8) // 4


_FILEG = None
_RANKG_OUT = None
_RANKG_MID = None


def _c11(sq):
    """Luka's regions: {abc},{fgh} x {12,34,56,78}  +  {de} x {123,45,678}."""
    global _FILEG, _RANKG_OUT, _RANKG_MID
    if _FILEG is None or _FILEG.device != sq.device:
        _FILEG = torch.tensor([0, 0, 0, 2, 2, 1, 1, 1], device=sq.device)
        _RANKG_OUT = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=sq.device)
        _RANKG_MID = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=sq.device)
    f, r = sq % 8, sq // 8
    fg = _FILEG[f]
    return torch.where(fg == 2, 8 + _RANKG_MID[r], fg * 4 + _RANKG_OUT[r])


def b_kings16(feat):
    return _king_pair(feat, _c16)


def b_kings11(feat):
    return _king_pair(feat, _c11)


def b_kings4(feat):
    return _king_pair(feat, _c4)


def b_rand(feat, n):
    """A router with the right shape and no chess in it.

    The severity control for `vbuck`: bucketing the accumulator multiplies the
    parameter count, so a win could be capacity rather than routing. This routes
    on a hash of the position -- same number of buckets, same table size, same
    parameters, no information about the evaluation. If the random router gains
    as much as the material one, the gain was capacity.

    Deterministic per position, which is all a router has to be here; it is not
    stable under a move and is not meant to be deployed."""
    h = (feat.long() * 2654435761) ^ (feat.long() << 13)
    return (h.sum(1).abs() % n)


_COARSE = {}


def b_mapped(feat, fine, name):
    """A coarsening designed from the transition graph (`nnue/coarsen.py`).

    The hand-designed king families above group squares by where they are. This
    groups fine cells by how often a made move carries the index from one to
    the other, so the merges are the ones that cost a refresh. Whether that is
    a good grouping for the *evaluation* is what training it answers -- cut is
    free to merge cells the eval needed apart."""
    if name not in _COARSE:
        import numpy as np, os
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "coarsenings.npz")
        _COARSE.update({k: torch.as_tensor(v) for k, v in np.load(f).items()})
    return _COARSE[name].to(feat.device)[fine(feat)]


_STAGED = {}


def b_staged(feat, name, k1):
    """A staged rule: fine kings and fine material, each pruned, then the
    product pruned again (`nnue/stagedjoint.py`). The whole chain is two table
    lookups and a multiply, so it costs nothing at inference; what it costs is
    a refresh whenever a made move changes the bucket, which is the number the
    search was minimising."""
    if not _STAGED:
        import numpy as np, os
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stagedjoint.npz")
        _STAGED.update({k: torch.as_tensor(v) for k, v in np.load(f).items()})
    import rulestats as R
    ax = _STAGED["axes_" + name].to(feat.device)
    grp = _STAGED[name].to(feat.device)
    return grp[ax[0][R.r_kings(feat)[0]] * k1 + ax[1][R.r_matfine(feat)[0]]]


S512 = "K64c100xM256c1.5->512c1.5"
S1024 = "K32c100xM256c1.5->1024c1.5"


_MERGED = {}


def b_merged(feat, name, k1):
    """A coarsening from `nnue/merge2.py`: the least-visited live cell joins the
    neighbour it transitions to most, until k buckets remain; never-visited
    cells fold in afterwards. Priced per ply across BOTH accumulators, which is
    what the engine actually pays."""
    if not _MERGED:
        import numpy as np, os
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merged.npz")
        _MERGED.update({k: torch.as_tensor(v) for k, v in np.load(f).items()})
    import rulestats as R
    ax = _MERGED["axes_" + name].to(feat.device)
    grp = _MERGED[name].to(feat.device)
    return grp[ax[0][R.r_kings(feat)[0]] * k1 + ax[1][R.r_matfine(feat)[0]]]


# --- king + pawn magic (6+6 bits) ---
# Pawn structure hash: our pawns only (side==0 & pt==0), bb * MAGIC >>58
# MAGIC found by nnue/magic_search.py maximizing H/n - beta*E/n and H*stab^2
# Best on tono_games 500k/300k: 0x2c0c8fcfcfedecbb  H=5.935 H/n=0.989 E/n=0.073 stab=0.927 perp 61.2
# Also good: 0x608dee52cc1dd10f etc. We keep the H*stab2 optimum.
MAGIC_PAWN = 0x2c0c8fcfcfedecbb  # <2^63, positive, GPU-friendly
# Alternative for ~90% bucket stability target could be 0x... but 0x2c already 92.7% Hamming ~82% bucket, close.

def _pawn_bb(feat):
    """Our pawn bitboard per row as uint64 (fits in int64 <2^63 for most, but use int64)."""
    valid, side, pt, sq = _parts(feat)
    is_pawn = valid & (side == 0) & (pt == 0)
    # Build bb via per-slot scatter; feat is (N,32) long, we iterate 32 Slots (tiny)
    N = feat.shape[0]
    bb = torch.zeros(N, dtype=torch.int64, device=feat.device)
    for i in range(feat.shape[1]):
        sq_i = sq[:, i]
        m = is_pawn[:, i]
        if m.any():
            # 1 << sq_i per row where m
            # torch's << with tensor shift is elementwise
            bit = torch.where(m, torch.ones_like(sq_i, dtype=torch.int64) << sq_i, torch.zeros_like(bb))
            bb = bb | bit
    return bb  # int64, high bit may be set for pawn on h8 (>=63? max sq 63 => 1<<63 = min int64, still okay)

def b_king6(feat):
    """Our king square 0..63, 6 bits. In stm-canonical, side==0 king."""
    valid, side, pt, sq = _parts(feat)
    is_k = valid & (pt == 5) & (side == 0)
    # sq where is_k, else 0 (will be masked but there is exactly one king per side)
    # Use sum: exactly one true per row
    ks = (sq * is_k).sum(1).clamp(0, 63)
    return ks.long()

def b_magic6(feat):
    """6-bit pawn hash: (pawn_bb * MAGIC) >>58 &63. Our pawns only."""
    bb = _pawn_bb(feat)  # (N,) int64
    # Compute idx via numpy uint64 for correct logical shift (torch int64 mul wraps same low bits, but logical shift differs for negative)
    # For our chosen MAGIC <2^63, bb <2^64, product low 64bits same for signed/unsigned, but we need logical >>58.
    # Do on CPU with numpy uint64 then back to device for correctness and simplicity (65k per batch cheap).
    # Move to CPU
    bb_np = bb.detach().cpu().numpy().astype(np.uint64)
    magic_np = np.uint64(MAGIC_PAWN)
    idx_np = (bb_np * magic_np) >> np.uint64(58)
    idx_np = idx_np & np.uint64(63)
    return torch.from_numpy(idx_np.astype(np.int64)).to(feat.device).long()

def make_magic(magic, nbits=6):
    """A 6-bit (or n-bit) pawn hash for an ARBITRARY magic, as a bucket fn.

    `b_magic6` above is hard-wired to MAGIC_PAWN. To ask whether a *better*
    magic trains better we need several in one process, so the magic becomes an
    argument. Same arithmetic: top `nbits` of (our_pawn_bb * magic) as an
    unsigned 64-bit product. numpy does the shift because torch's int64 >> is
    arithmetic and the product's sign bit is meaningful here."""
    m_np = np.uint64(magic)
    sh = np.uint64(64 - nbits)
    msk = np.uint64((1 << nbits) - 1)

    def f(feat):
        bb = _pawn_bb(feat).detach().cpu().numpy().astype(np.uint64)
        idx = ((bb * m_np) >> sh) & msk
        return torch.from_numpy(idx.astype(np.int64)).to(feat.device).long()
    return f


def make_rules(npz_path):
    """Bucket fn for an annealed predicate rule set (nnue/anneal_rules.py).

    bit_j = popcount(our_pawns & mask_j) > thr_j, packed into a 2^R state, then
    `lut` maps that to the merged buckets the model actually sees. Unlike a
    magic hash this is readable: each bit is "do I have more than k pawns in
    this region", so a winning rule can be looked at."""
    d = np.load(npz_path)
    masks, thr, lut = d["masks"], d["thr"], d["lut"].astype(np.int64)
    pw2 = np.array([1 << j for j in range(len(masks))], dtype=np.int64)

    def f(feat):
        bb = _pawn_bb(feat).detach().cpu().numpy().astype(np.uint64)
        bits = np.stack([(np.bitwise_count(bb & np.uint64(m)) > t)
                         for m, t in zip(masks, thr)], 1).astype(np.int64)
        return torch.from_numpy(lut[bits @ pw2]).to(feat.device).long()
    return f, int(lut.max()) + 1


def register_rules(spec):
    """'name=path.npz,...' -> FAMILIES entries."""
    names = []
    for part in filter(None, (x.strip() for x in spec.split(","))):
        name, _, path = part.partition("=")
        FAMILIES[name] = make_rules(path)
        names.append(name)
    return names


def register_magics(spec, nbits=6):
    """'name=0xdeadbeef,other=0x...' -> FAMILIES entries of size 2**nbits."""
    names = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, hx = part.partition("=")
        FAMILIES[name] = (make_magic(int(hx, 16), nbits), 1 << nbits)
        names.append(name)
    return names


def b_pawncount(feat):
    """How many pawns we have, 0..8.

    The control for the annealed predicate rules. A rule bit is
    popcount(pawns & mask) > k, so with a near-full mask and a middling
    threshold the family CONTAINS pawn count -- and on the first smoke run
    I(B; pawn count) was 2.007 of the 3.838 bits the rule knew, i.e. 52% of it
    was just counting. If an annealed rule does not beat this arm, it found
    material bucketing, not pawn structure."""
    bb = _pawn_bb(feat).detach().cpu().numpy().astype(np.uint64)
    return torch.from_numpy(np.bitwise_count(bb).astype(np.int64)).to(feat.device).long().clamp(0, 8)


def make_cross(a_name, b_name):
    """Cartesian product of two families: index = ia * size_b + ib.

    The k-means on the 4096 king x magic adapters came out 87.7% mutual
    information with the king square, so the king half carries the signal and
    should stay at full 6 bits while the pawn half is compressed. That is a
    product, not a sum: `pfams=(king6, rule)` would only add two independent
    PSQT deltas."""
    fa, na = FAMILIES[a_name]
    fb, nb = FAMILIES[b_name]
    return (lambda feat: fa(feat) * nb + fb(feat)), na * nb


def register_cross(spec):
    """'name=king6*rule,...' -> FAMILIES entries."""
    names = []
    for part in filter(None, (x.strip() for x in spec.split(","))):
        name, _, rhs = part.partition("=")
        a_name, _, b_name = rhs.partition("*")
        FAMILIES[name] = make_cross(a_name, b_name)
        names.append(name)
    return names


def b_king_magic(feat):
    """12-bit combined: king6 <<6 | magic6  => 0..4095"""
    k = b_king6(feat)
    m = b_magic6(feat)
    return (k << 6) | m


def b_king32(feat):
    """Our king square folded by file-mirror to 32 buckets: min(sq, mirror(sq)).

    The 32-way king bar: same information as king6 at half the buckets, and
    mirror-invariant, so the file-mirror half of king motion is free. What it
    does NOT fix is the stm chair swap (which king is "ours" alternates) --
    that is what the tied router weights are for.
    """
    valid, side, pt, sq = _parts(feat)
    is_k = valid & (pt == 5) & (side == 0)
    ks = (sq * is_k).sum(1).clamp(0, 63)
    f, r = ks % 8, ks // 8
    mf = torch.minimum(f, 7 - f)
    return (r * 4 + (mf // 2).clamp(max=3)).long()


def b_castle_phase(feat):
    """Castle state (4x4=16, both kings) x material half (2) = 32 buckets.

    The slow fixed bar: castle state flips only when a king leaves its region
    (6.5%/ply both accs on tree moves) and the phase half flips only on a
    capture crossing the material midpoint (~2-3%). If a learned 32-way rule
    cannot beat this on val at equal-or-better stability, it found nothing.
    """
    valid, side, pt, sq = _parts(feat)
    is_k = valid & (pt == 5)
    ku = (sq * (is_k & (side == 0))).sum(1)
    kt = (sq * (is_k & (side == 1))).sum(1)
    f = lambda s: (s % 8).clamp(0, 7)
    r = lambda s: (s // 8).clamp(0, 7)
    st = lambda s: torch.where(r(s) > 1, torch.full_like(s, 3),
                torch.where(f(s) >= 6, torch.zeros_like(s),
                torch.where(f(s) <= 2, torch.ones_like(s),
                            torch.full_like(s, 2))))
    castle = st(ku) * 4 + st(kt)
    n = (valid & ((pt == 1) | (pt == 2))).sum(1) \
        + 2 * (valid & (pt == 3)).sum(1) + 4 * (valid & (pt == 4)).sum(1)
    half = (n * 2 // 25).clamp(max=1)
    return (castle * 2 + half).long()


def make_matsum(nb):
    """Total material P1 N3 B3 R5 Q9 (both sides, 0..78) in `nb` uniform bins.

    The slow symmetric rule: total material moves only on captures and
    promotions, so it is quiet in the main search and busy in quiescence
    (measured 2026-09-07 on 227k tree plies, both-accs %/ply, main/q):
    nb=2 -> 3.4 (1.7/13.5), nb=4 -> 10.4 (5.3/40.3),
    nb=8 -> 20.3 (11.0/75.1), nb=32 -> 45.0 (28.2/144.2).
    Uniform bins (no calibration table); symmetric so both accs flip together.
    """
    def f(feat):
        valid, side, pt, sq = _parts(feat)
        v = torch.zeros_like(sq)
        v = torch.where(pt == 0, torch.ones_like(v), v)
        v = torch.where((pt == 1) | (pt == 2), torch.full_like(v, 3), v)
        v = torch.where(pt == 3, torch.full_like(v, 5), v)
        v = torch.where(pt == 4, torch.full_like(v, 9), v)
        s = (v * valid.long()).sum(1)
        return ((s * nb) // 79).clamp(max=nb - 1).long()
    return f


FAMILIES = {
    "none":     (b_none, 1),
    "merged256":  (lambda f: b_merged(f, "K32xM64->256", 64), 256),
    "merged512":  (lambda f: b_merged(f, "K64xM128->512", 128), 512),
    "merged1024": (lambda f: b_merged(f, "K64xM128->1024", 128), 1024),
    "staged512":  (lambda f: b_staged(f, S512, 256), 512),
    "staged1024": (lambda f: b_staged(f, S1024, 256), 1024),
    "kgraph16":  (lambda f: b_mapped(f, b_kings, "kings64^2_16"), 16),
    "kgraph64":  (lambda f: b_mapped(f, b_kings, "kings64^2_64"), 64),
    "kgraph121": (lambda f: b_mapped(f, b_kings, "kings64^2_121"), 121),
    "mgraph64":  (lambda f: b_mapped(f, b_material, "material24^2_64"), 64),
    "bishops":  (b_bishops, 16),
    "kings":    (b_kings, 64 * 64),
    "kings16":  (b_kings16, 16 * 16),
    "kings11":  (b_kings11, 11 * 11),
    "kings4":   (b_kings4, 4 * 4),
    "queens":   (b_queens, 9),
    "material": (b_material, 24 * 24),
    "rand576":  (lambda f: b_rand(f, 576), 576),
    "rand121":  (lambda f: b_rand(f, 121), 121),
    "count":    (b_count, 8),
    "king6":    (b_king6, 64),
    "king32":   (b_king32, 32),
    "castle_phase": (b_castle_phase, 32),
    "magic6":   (b_magic6, 64),
    "king_magic": (b_king_magic, 4096),
    "rand4096": (lambda f: b_rand(f, 4096), 4096),
    "rand16384": (lambda f: b_rand(f, 16384), 16384),
    "rand65536": (lambda f: b_rand(f, 65536), 65536),
    "rand32":   (lambda f: b_rand(f, 32), 32),
    "rand64":   (lambda f: b_rand(f, 64), 64),
    "pawncount": (b_pawncount, 9),
    "matsum2":  (make_matsum(2), 2),
    "matsum4":  (make_matsum(4), 4),
    "matsum8":  (make_matsum(8), 8),
    "matsum32": (make_matsum(32), 32),
}


