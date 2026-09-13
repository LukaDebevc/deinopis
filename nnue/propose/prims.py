"""Bitboard-shaped primitives: everything a generated rule is allowed to touch.

A candidate rule is a pure function of a `Board`. The Board exposes only
operations the engine can already do cheaply on bitboards -- mask, shift,
popcount, fill, table lookup -- so "cheap to compute in the engine" is true by
construction instead of being checked after the fact. Sliding-piece attacks are
deliberately absent: they need magic lookups and they change on almost every
move, which is the opposite of what a routing rule needs.

Every primitive charges an op count. `b.cost` is what one recompute would cost
in the engine, in rough bitboard instructions, and it is one of the three
numbers a candidate is judged on.

`b.reads` is INFERRED from which piece types the rule actually asked for, not
declared by the proposer. `refresh.py` only perturbs the pieces a rule reads,
so a wrong declaration would silently understate the refresh rate.
"""
import torch

US, THEM = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
NFEAT = PAD = 768

# Squares are 0..63 with square = rank * 8 + file, rank 0 = the mover's own
# back rank. Records are side-to-move canonical, so US always moves up.
_R = torch.arange(64) // 8
_F = torch.arange(64) % 8


def _named_masks():
    m = {}
    for i in range(8):
        m[f"file_{'abcdefgh'[i]}"] = (_F == i)
        m[f"rank_{i + 1}"] = (_R == i)
    m["centre4"] = ((_F >= 3) & (_F <= 4) & (_R >= 3) & (_R <= 4))
    m["centre16"] = ((_F >= 2) & (_F <= 5) & (_R >= 2) & (_R <= 5))
    m["kingside"] = (_F >= 5)
    m["queenside"] = (_F <= 2)
    m["centrefiles"] = ((_F >= 3) & (_F <= 4))
    m["edge"] = ((_F == 0) | (_F == 7) | (_R == 0) | (_R == 7))
    m["light"] = (((_F + _R) % 2) == 1)
    m["dark"] = (((_F + _R) % 2) == 0)
    m["ourhalf"] = (_R <= 3)
    m["theirhalf"] = (_R >= 4)
    m["all"] = torch.ones(64, dtype=torch.bool)
    return m


MASKS = _named_masks()


class Budget(Exception):
    """A rule spent more ops than the cost ceiling allows."""


class Board:
    """A batch of positions, viewed as bitboards.

    Every method returns either an (N, 64) boolean board -- the tensor stand-in
    for a u64 bitboard -- or an (N,) integer tensor, one value per position.
    """

    def __init__(self, feat, max_cost=200):
        valid = feat != PAD
        f = feat.clamp(max=NFEAT - 1)
        side, pt, sq = f // 384, (f % 384) // 64, f % 64
        n = feat.shape[0]
        dev = feat.device
        flat = torch.zeros(n, 12 * 64 + 1, dtype=torch.bool, device=dev)
        dump = torch.full_like(sq, 12 * 64)
        flat.scatter_(1, torch.where(valid, (side * 6 + pt) * 64 + sq, dump), True)
        self._bb = flat[:, : 12 * 64].view(n, 12, 64)
        self.n = n
        self.dev = dev
        self.cost = 0
        self.reads = set()
        self._max_cost = max_cost
        self._r = _R.to(dev)
        self._f = _F.to(dev)

    # ------------------------------------------------------------- accounting
    def _charge(self, ops):
        self.cost += ops
        if self.cost > self._max_cost:
            raise Budget(f"rule exceeded {self._max_cost} ops")

    # ---------------------------------------------------------------- sources
    def pieces(self, side, pt):
        """The bitboard of one side's pieces of one type. Free: it is a field."""
        self.reads.add(int(pt))
        return self._bb[:, int(side) * 6 + int(pt)]

    def occupied(self, side=None):
        """Every piece, or every piece of one side. One OR per piece type."""
        self.reads.update(range(6))
        self._charge(5 if side is not None else 11)
        if side is None:
            return self._bb.any(1)
        return self._bb[:, int(side) * 6 : int(side) * 6 + 6].any(1)

    def mask(self, name):
        """A compile-time constant board. Free -- it is an immediate."""
        if name not in MASKS:
            raise KeyError(f"unknown mask {name!r}; have {sorted(MASKS)}")
        return MASKS[name].to(self.dev).expand(self.n, 64)

    # ------------------------------------------------------------- bit ops
    def AND(self, a, b):
        """Squares in both boards."""
        self._charge(1)
        return a & b

    def OR(self, a, b):
        """Squares in either board."""
        self._charge(1)
        return a | b

    def ANDNOT(self, a, b):
        """a without b."""
        self._charge(1)
        return a & ~b

    def XOR(self, a, b):
        """Squares in exactly one of the two boards."""
        self._charge(1)
        return a ^ b

    def shift(self, bb, dfile, drank):
        """Slide a whole board, dropping bits that fall off. One shift + mask."""
        self._charge(2)
        x = bb.view(self.n, 8, 8)
        out = torch.zeros_like(x)
        sr, dr = max(0, -drank), max(0, drank)
        sf, df = max(0, -dfile), max(0, dfile)
        h, w = 8 - abs(drank), 8 - abs(dfile)
        if h > 0 and w > 0:
            out[:, dr : dr + h, df : df + w] = x[:, sr : sr + h, sf : sf + w]
        return out.view(self.n, 64)

    def fill_up(self, bb):
        """Smear every bit toward rank 8 (Kogge-Stone). Spans, open files."""
        self._charge(6)
        x = bb
        for k in (1, 2, 4):
            x = x | _raw_shift(x, 0, k, self.n)
        return x

    def fill_down(self, bb):
        """Smear every bit toward rank 1."""
        self._charge(6)
        x = bb
        for k in (1, 2, 4):
            x = x | _raw_shift(x, 0, -k, self.n)
        return x

    # --------------------------------------------------------- cheap attacks
    def pawn_attacks(self, side):
        """Squares that side's pawns attack. Two shifts and an OR."""
        self.reads.add(PAWN)
        self._charge(3)
        p = self._bb[:, int(side) * 6 + PAWN]
        d = 1 if int(side) == US else -1
        return _raw_shift(p, -1, d, self.n) | _raw_shift(p, 1, d, self.n)

    def king_ring(self, side):
        """The eight squares around that side's king."""
        self.reads.add(KING)
        self._charge(8)
        k = self._bb[:, int(side) * 6 + KING]
        out = torch.zeros_like(k)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df or dr:
                    out = out | _raw_shift(k, df, dr, self.n)
        return out

    # ------------------------------------------------------------- reductions
    def popcount(self, bb):
        """How many squares are set -- one count per position."""
        self._charge(1)
        return bb.sum(1)

    def any(self, bb):
        """1 if the board has any bit set, else 0 -- one per position."""
        self._charge(1)
        return (bb.sum(1) > 0).long()

    def square_of(self, side, pt):
        """The square of a unique piece -- meant for the king. One ctz."""
        self.reads.add(int(pt))
        self._charge(1)
        bb = self._bb[:, int(side) * 6 + int(pt)]
        return (bb.long() * torch.arange(64, device=self.dev)).sum(1)

    def count(self, side, pt):
        """Shorthand for popcount(pieces(side, pt))."""
        self.reads.add(int(pt))
        self._charge(1)
        return self._bb[:, int(side) * 6 + int(pt)].sum(1)

    def file_of(self, sq):
        """File 0..7 (a..h) of a square value."""
        self._charge(1)
        return sq % 8

    def rank_of(self, sq):
        """Rank 0..7 of a square value, 0 = the mover's own back rank."""
        self._charge(1)
        return sq // 8

    # ---------------------------------------------------------- combinators
    def clamp(self, v, lo, hi):
        """Squash a count into a small range. Cheap and usually necessary."""
        self._charge(1)
        return v.clamp(int(lo), int(hi)) - int(lo)

    def table(self, v, lut):
        """Map a small integer through a lookup table -- the bits->group map.

        `lut` is a plain list of ints; the result is in 0..max(lut). This is the
        one primitive that lets a rule express a partition no arithmetic gives."""
        self._charge(1)
        t = torch.tensor([int(x) for x in lut], device=self.dev, dtype=torch.long)
        return t[v.long().clamp(0, len(lut) - 1)]

    def bits(self, flags):
        """Pack boolean tests into one integer -- the bitmask -> rule index map."""
        self._charge(len(flags))
        out = torch.zeros(self.n, dtype=torch.long, device=self.dev)
        for i, fl in enumerate(flags):
            out = out + (fl.long() << i)
        return out, 1 << len(flags)

    def combine(self, terms):
        """Mixed radix over (value, size) pairs -> (index, total size).

        Every rule ends here. Values are clamped into their declared size, so an
        off-by-one produces a merged bucket rather than an out-of-range index."""
        self._charge(2 * len(terms))
        idx = torch.zeros(self.n, dtype=torch.long, device=self.dev)
        size = 1
        for v, s in terms:
            s = int(s)
            if s < 1:
                raise ValueError("term size must be >= 1")
            idx = idx * s + v.long().clamp(0, s - 1)
            size *= s
        return idx, size


def _raw_shift(bb, dfile, drank, n):
    x = bb.view(n, 8, 8)
    out = torch.zeros_like(x)
    sr, dr = max(0, -drank), max(0, drank)
    sf, df = max(0, -dfile), max(0, dfile)
    h, w = 8 - abs(drank), 8 - abs(dfile)
    if h > 0 and w > 0:
        out[:, dr : dr + h, df : df + w] = x[:, sr : sr + h, sf : sf + w]
    return out.view(n, 64)
