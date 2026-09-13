"""Price routing rules on the moves the search actually makes.

Everything before this priced a rule on TRAINING positions, enumerating their
legal moves and reweighting the per-move rates by the tree's move-CLASS mix
(69% quiet / 30% capture / 1% promo, from `chess bench`). Three biases survive
that correction, and Luka named the one that matters: move ordering. Captures,
killers and history-rich moves are searched first and most interior nodes cut
after one move, so the tree plays a very different mix of PIECES than a
class-reweighted legal-move mix does. King moves are history-poor and ordered
late. If the tree rarely moves a king, every king-reading rule -- HalfKP above
all -- has been priced too high.

`chess movedump --rand` writes two streams from the same sampled positions:
the move the tree chose (site 0 main, 1 quiescence) and one drawn uniformly
from the legal list (site 2). Holding the positions fixed, the gap between
them is move ordering and nothing else.

Prices are per PLY across BOTH accumulators, which is what a ply costs an
engine that keeps one accumulator per perspective. Both FENs come from the
engine, so nothing here reimplements en passant, castling or promotion --
which is where the last pricing bug lived.
"""
import sys, time
import numpy as np
import torch

sys.path.insert(0, './nnue')
from loader import NFEAT, PAD, MAX_PIECES
import rulestats as R

PC = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}


def parse_board(bp):
    """FEN board field -> list of (colour, piece type, square), a1 = 0."""
    out, sq = [], 56
    for ch in bp:
        if ch == '/':
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            out.append((0 if ch.isupper() else 1, PC[ch.upper()], sq))
            sq += 1
    return out


def load(path, limit=None):
    """-> site, and four (N,32) int16 feature tables.

    An accumulator belongs to a PLAYER, not to a ply, so each position is
    encoded twice: once with White as side 0 and once with Black as side 0
    (squares flipped). Both have to stay current, so a ply's price is the
    number of the two that the move dirties.
    """
    lines = open(path).read().splitlines()
    if limit:
        lines = lines[:limit]
    n = len(lines)
    pw = np.full((n, MAX_PIECES), PAD, np.int16)
    pb = np.full((n, MAX_PIECES), PAD, np.int16)
    cw = np.full((n, MAX_PIECES), PAD, np.int16)
    cb = np.full((n, MAX_PIECES), PAD, np.int16)
    site = np.zeros(n, np.int8)
    mover = np.zeros(n, np.int8)          # piece type that moved
    t = time.time()
    for i, ln in enumerate(lines):
        p, c, uci, s = ln.split('\t')
        site[i] = int(s)
        for fen, aw, ab in ((p, pw, pb), (c, cw, cb)):
            for k, (col, pt, sq) in enumerate(parse_board(fen.split(' ', 1)[0])):
                aw[i, k] = col * 384 + pt * 64 + sq
                ab[i, k] = (1 - col) * 384 + pt * 64 + (sq ^ 56)
        frm = (ord(uci[0]) - 97) + 8 * (ord(uci[1]) - 49)
        for col, pt, sq in parse_board(p.split(' ', 1)[0]):
            if sq == frm:
                mover[i] = pt
                break
        if i % 100000 == 0:
            print(f"\r  {i:,}/{n:,}  {time.time()-t:.0f}s", end="", flush=True)
    print(f"\r  {n:,} plies parsed in {time.time()-t:.0f}s     ")
    return site, mover, pw, pb, cw, cb


def price(fn, pw, pb, cw, cb, mask, chunk=100000):
    """Refreshes per ply, summed over the two accumulators, and the parent's
    visit distribution over buckets (White's accumulator only, to keep the
    perplexity a count of buckets rather than of buckets x perspective)."""
    idx = np.nonzero(mask)[0]
    hit = 0.0
    size = None
    counts = None
    for s in range(0, len(idx), chunk):
        j = idx[s:s + chunk]
        for a, b, first in ((pw, cw, True), (pb, cb, False)):
            base, size, _ = fn(torch.as_tensor(a[j].astype(np.int64)))
            new, _, _ = fn(torch.as_tensor(b[j].astype(np.int64)))
            hit += (base != new).sum().item()
            if first:
                if counts is None:
                    counts = torch.zeros(size, dtype=torch.float64)
                counts += torch.bincount(base, minlength=size).double()
    n = len(idx)
    p = counts / counts.sum()
    nz = p[p > 0]
    perp = float(torch.exp(-(nz * nz.log()).sum()))
    return 100.0 * hit / max(n, 1), size, int((counts > 0).sum()), perp


def main(path, limit=None):
    site, mover, pw, pb, cw, cb = load(path, limit)
    tree = site <= 1
    rand = site == 2
    print(f"\n{tree.sum():,} tree plies, {rand.sum():,} uniform-legal plies "
          f"from the same positions\n")

    names = "pawn knight bishop rook queen king".split()
    print(f"{'moved piece':<12}{'tree':>9}{'uniform':>9}{'ratio':>8}")
    print("-" * 38)
    for pt in range(6):
        a = (mover[tree] == pt).mean() * 100
        b = (mover[rand] == pt).mean() * 100
        print(f"{names[pt]:<12}{a:>8.2f}%{b:>8.2f}%{a/max(b,1e-9):>8.2f}")

    hdr = (f"\n{'rule':<30}{'buckets':>9}{'seen':>7}{'perp':>7}"
           f"{'tree %/ply':>12}{'uniform':>10}{'ratio':>8}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    res = []
    for name, fn in R.RULES.items():
        a, size, seen, perp = price(fn, pw, pb, cw, cb, tree)
        b, _, _, _ = price(fn, pw, pb, cw, cb, rand)
        res.append((name, size, seen, perp, a, b))
    for name, size, seen, perp, a, b in sorted(res, key=lambda r: r[4]):
        print(f"{name:<30}{size:>9,}{seen:>7,}{perp:>7.0f}"
              f"{a:>11.2f}%{b:>9.2f}%{a/max(b,1e-9):>8.2f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
