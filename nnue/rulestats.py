"""Score candidate routing rules on the three things Luka asked for.

A rule set is a total function position -> bucket. Three properties decide
whether it is usable, and only the first is usually measured:

  BALANCE      perplexity of the visit distribution, and the largest share.
               The declared count is a fiction: a fixed router gets whatever
               chess gives it, and `material` declares 576 and gets 66.

  COMPLETE     every position lands somewhere. True by construction for a
               total function; asserted anyway, because an index built by
               arithmetic on piece counts is easy to push out of range.

  STABLE       how often the index changes when a piece moves. This is the one
               that decides whether the rule can sit in front of the
               accumulator, and it is measured here rather than argued: real
               positions from the corpus, with the king stepped to each legal
               adjacent square and each pawn pushed, asking whether the index
               moved. Pieces the rule does not read cannot change it, so only
               king and pawn moves are enumerated -- for a rule that reads
               other pieces, that is a LOWER bound on the change rate and the
               table says so.

Not legal-move-weighted: a king step is counted once per reachable adjacent
square, not by how often engines play it. The bias is identical across rule
sets, so the comparison is fair even though the absolute rate is not a
per-move probability.
"""
import sys
import torch
sys.path.insert(0, './nnue')
from loader import Batcher, NFEAT, PAD

US, THEM = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)


def parts(feat):
    """(valid, side, piece type, square) for every slot, PAD-safe."""
    valid = feat != PAD
    f = feat.clamp(max=NFEAT - 1)
    return valid, f // 384, (f % 384) // 64, f % 64


def occupancy(feat):
    """(N, 64) bool: any piece, and (N, 64) bool: one of ours."""
    valid, side, _, sq = parts(feat)
    occ = torch.zeros(feat.shape[0], 64, dtype=torch.bool, device=feat.device)
    ours = torch.zeros_like(occ)
    occ.scatter_(1, sq, valid)
    ours.scatter_(1, sq, valid & (side == US))
    return occ, ours


def king_sq(feat, s=US):
    valid, side, pt, sq = parts(feat)
    m = valid & (pt == KING) & (side == s)
    return (sq * m).sum(1)


def king_slot(feat, s=US):
    """Index of the slot holding that side's king, for surgery."""
    valid, side, pt, _ = parts(feat)
    m = valid & (pt == KING) & (side == s)
    return m.float().argmax(1)


def has(feat, s, pt):
    valid, side, p, _ = parts(feat)
    return ((valid & (side == s) & (p == pt)).sum(1) > 0).long()


def count(feat, s, pt):
    valid, side, p, _ = parts(feat)
    return (valid & (side == s) & (p == pt)).sum(1)


# ------------------------------------------------------------------ rule sets
# Each returns (index, declared_size, reads) where `reads` names the piece
# types the rule looks at -- the stability enumeration below only has to move
# those, and anything unnamed provably cannot change the index.

FILEG = torch.tensor([0, 0, 0, 2, 2, 1, 1, 1])       # a b c | d e | f g h
RANKG_OUT = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
RANKG_MID = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])


def king_region(sq):
    """Luka's 11 regions: {abc},{fgh} x {12,34,56,78}  +  {de} x {123,45,678}."""
    f, r = sq % 8, sq // 8
    fg = FILEG.to(sq.device)[f]
    return torch.where(fg == 2, 8 + RANKG_MID.to(sq.device)[r],
                       fg * 4 + RANKG_OUT.to(sq.device)[r])


def r_kings(feat):
    """HalfKP's own bucket, un-coarsened: both king squares."""
    return king_sq(feat, US) * 64 + king_sq(feat, THEM), 64 * 64, (KING,)


def r_kingregion(feat):
    return (king_region(king_sq(feat, US)) * 11
            + king_region(king_sq(feat, THEM))), 121, (KING,)


def _castle_state(sq):
    """Where the king actually ended up, in the four states play produces.

    0 kingside shelter (g/h, ranks 1-2)   1 queenside shelter (a/b/c, ranks 1-2)
    2 still central and at home (d/e/f, ranks 1-2)   3 off the back rank."""
    f, r = sq % 8, sq // 8
    back = r <= 1
    ks = back & (f >= 6)
    qs = back & (f <= 2)
    return torch.where(~back, torch.full_like(sq, 3),
                       torch.where(ks, torch.zeros_like(sq),
                                   torch.where(qs, torch.ones_like(sq),
                                               torch.full_like(sq, 2))))


def _shelter(feat, s, ksq):
    """Own pawns on the king's file and its neighbours, within two ranks ahead.

    Capped at 3. For `them` the board is still stm-canonical, so their pawns
    advance downward -- the sign of the rank offset flips with the side."""
    valid, side, pt, sq = parts(feat)
    m = valid & (side == s) & (pt == PAWN)
    kf, kr = (ksq % 8).unsqueeze(1), (ksq // 8).unsqueeze(1)
    df = (sq % 8 - kf).abs()
    dr = (sq // 8 - kr) if s == US else (kr - sq // 8)
    near = m & (df <= 1) & (dr >= 1) & (dr <= 2)
    return near.sum(1).clamp(max=3)


def r_shelter(feat):
    """Luka's example: where the king went x how much pawn cover it has."""
    ku, kt = king_sq(feat, US), king_sq(feat, THEM)
    us = _castle_state(ku) * 4 + _shelter(feat, US, ku)
    them = _castle_state(kt) * 4 + _shelter(feat, THEM, kt)
    return us * 16 + them, 256, (KING, PAWN)


def r_castle(feat):
    """Shelter's coarse half alone: 4 x 4. The stability floor to beat."""
    return (_castle_state(king_sq(feat, US)) * 4
            + _castle_state(king_sq(feat, THEM))), 16, (KING,)


def _centre_pawns(feat, s):
    """Own pawns on d4 e4 d5 e5 -- Luka's 'two pawns in the centre four'."""
    valid, side, pt, sq = parts(feat)
    m = valid & (side == s) & (pt == PAWN)
    c = ((sq == 27) | (sq == 28) | (sq == 35) | (sq == 36))
    return (m & c).sum(1).clamp(max=2)


def r_centre(feat):
    return (_centre_pawns(feat, US) * 3 + _centre_pawns(feat, THEM)), 9, (PAWN,)


def r_shelter_centre(feat):
    a, na, _ = r_shelter(feat)
    b, _, _ = r_centre(feat)
    return a * 9 + b, na * 9, (KING, PAWN)


def r_count(feat):
    return ((((feat != PAD).sum(1) - 2).clamp(min=0) // 4).clamp(max=7), 8,
            (PAWN, KNIGHT, BISHOP, ROOK, QUEEN))


def r_material(feat):
    """[rooks 0/1/2+] x [light bishop] x [dark bishop] x [queen], squared."""
    valid, side, pt, sq = parts(feat)
    col = ((sq // 8) + (sq % 8)) & 1
    out = []
    for s in (US, THEM):
        m = valid & (side == s)
        rook = (m & (pt == ROOK)).sum(1).clamp(max=2)
        queen = ((m & (pt == QUEEN)).sum(1) > 0).long()
        bl = ((m & (pt == BISHOP) & (col == 0)).sum(1) > 0).long()
        bd = ((m & (pt == BISHOP) & (col == 1)).sum(1) > 0).long()
        out.append(((rook * 2 + bl) * 2 + bd) * 2 + queen)
    return out[0] * 24 + out[1], 576, (BISHOP, ROOK, QUEEN)


def r_pieces36(feat):
    """A stable ~32: [rooks 0/1/2+] x [queen], squared. Reads no king, no pawn."""
    valid, side, pt, _ = parts(feat)
    out = []
    for s_ in (US, THEM):
        m = valid & (side == s_)
        rook = (m & (pt == ROOK)).sum(1).clamp(max=2)
        queen = ((m & (pt == QUEEN)).sum(1) > 0).long()
        out.append(rook * 2 + queen)
    return out[0] * 6 + out[1], 36, (ROOK, QUEEN)


def _phase(feat, s):
    """Standard game-phase weight: N=B=1, R=2, Q=4, pawns ignored.

    Wide bins on a monotone quantity are the stable way to read material: a
    capture moves the total by 1-4, so the index changes only when that step
    crosses a bin edge, instead of every time a specific piece count changes.
    Max is 12 per army, so 24 for the position."""
    valid, side, pt, _ = parts(feat)
    m = valid & (side == s)
    return ((m & ((pt == KNIGHT) | (pt == BISHOP))).sum(1)
            + 2 * (m & (pt == ROOK)).sum(1)
            + 4 * (m & (pt == QUEEN)).sum(1))


def r_phase4(feat):
    """Total non-pawn material, 4 wide bins. Sides summed, not crossed --
    a phase is a property of the position, not of one army."""
    t = _phase(feat, US) + _phase(feat, THEM)
    return (t * 4 // 25).clamp(max=3), 4, (KNIGHT, BISHOP, ROOK, QUEEN)


def r_phase8(feat):
    t = _phase(feat, US) + _phase(feat, THEM)
    return (t * 8 // 25).clamp(max=7), 8, (KNIGHT, BISHOP, ROOK, QUEEN)


def r_phase_imbal(feat):
    """Phase x who is up material, in pawn-equivalents, coarsely binned."""
    ph, _, _ = r_phase8(feat)
    d = _phase(feat, US) - _phase(feat, THEM)
    b = (d.clamp(-4, 4) + 4) // 2                    # 0..4
    return ph * 5 + b, 40, (KNIGHT, BISHOP, ROOK, QUEEN)


def r_phase_shelter(feat):
    a, _, _ = r_phase8(feat)
    b, nb, _ = r_shelter(feat)
    return a * nb + b, 8 * nb, (KNIGHT, BISHOP, ROOK, QUEEN, KING, PAWN)


def r_shelter_material(feat):
    a, na, _ = r_shelter(feat)
    b, nb, _ = r_material(feat)
    return a * nb + b, na * nb, (KING, PAWN, BISHOP, ROOK, QUEEN)


RULES = {
    "kings 64x64 (HalfKP)": r_kings,
    "king region 11^2": r_kingregion,
    "castle state 4^2": r_castle,
    "centre pawns 3^2": r_centre,
    "shelter (castle x cover)^2": r_shelter,
    "shelter x centre": r_shelter_centre,
    "material 24^2": r_material,
    "phase x4 (wide bins)": r_phase4,
    "phase x8": r_phase8,
    "phase x imbalance (x40)": r_phase_imbal,
    "phase x shelter (x2048)": r_phase_shelter,
    "count x8": r_count,
    "pieces (rook x queen)^2": r_pieces36,
    "shelter x material": r_shelter_material,
}


# ------------------------------------------------ game type: pawn structure
# Luka's split: phase says WHERE in the game we are, game type says WHICH game
# it is. Phase is material. Game type is meant to come from the king and the
# pawns -- the two things that do not change on a random capture.

def _centre_state(feat):
    """Open / mobile / locked, read from the d and e files only.

    0 open   -- no pawn of either colour on d or e
    2 locked -- our pawn directly behind an enemy pawn there (a blocked pair)
    1 mobile -- anything else
    Reads four squares' worth of pawns, so a capture elsewhere cannot move it."""
    valid, side, pt, sq = parts(feat)
    p = valid & (pt == PAWN) & (((sq % 8) == 3) | ((sq % 8) == 4))
    n = p.sum(1)
    us = torch.zeros(feat.shape[0], 64, dtype=torch.bool, device=feat.device)
    th = torch.zeros_like(us)
    us.scatter_(1, sq, p & (side == US))
    th.scatter_(1, sq, p & (side == THEM))
    locked = (us[:, :56] & th[:, 8:]).any(1)          # canonical: we move up
    return torch.where(locked, torch.full_like(n, 2),
                       torch.where(n == 0, torch.zeros_like(n),
                                   torch.ones_like(n)))


def _wing(feat):
    """Pawn majority on each wing: sign(ours - theirs) on a-c and on f-h."""
    valid, side, pt, sq = parts(feat)
    p = valid & (pt == PAWN)
    f = sq % 8
    out = []
    for lo, hi in ((0, 2), (5, 7)):
        z = p & (f >= lo) & (f <= hi)
        d = (z & (side == US)).sum(1) - (z & (side == THEM)).sum(1)
        out.append(d.clamp(-1, 1) + 1)
    return out[0] * 3 + out[1], 9


def r_centrestate(feat):
    return _centre_state(feat), 3, (PAWN,)


def r_wing(feat):
    i, n = _wing(feat)
    return i, n, (PAWN,)


def r_struct(feat):
    """Game type from pawns alone: centre state x the two wing majorities."""
    w, nw = _wing(feat)
    return _centre_state(feat) * nw + w, 3 * nw, (PAWN,)


def r_phase3(feat):
    t = _phase(feat, US) + _phase(feat, THEM)
    return (t * 3 // 25).clamp(max=2), 3, (KNIGHT, BISHOP, ROOK, QUEEN)


def r_phase5(feat):
    t = _phase(feat, US) + _phase(feat, THEM)
    return (t * 5 // 25).clamp(max=4), 5, (KNIGHT, BISHOP, ROOK, QUEEN)


# ------------------------------------------------------ phase x game type
# The shape Luka proposed. Crossing two rules multiplies the declared count and
# -- if the two read disjoint pieces -- should add their change rates. Whether
# it actually adds is measured, not assumed.

def _cross(*rs):
    idx, size, reads = 0, 1, ()
    for i, n, rd in rs:
        idx = idx * n + i
        size *= n
        reads = reads + tuple(r for r in rd if r not in reads)
    return idx, size, reads


def r_pc(feat):
    return _cross(r_phase5(feat), r_castle(feat))


def r_pcc(feat):
    return _cross(r_phase5(feat), r_castle(feat), r_centrestate(feat))


def r_pcs(feat):
    return _cross(r_phase5(feat), r_castle(feat), r_struct(feat))


def r_pk(feat):
    return _cross(r_phase5(feat), r_kingregion(feat))


def r_p8cc(feat):
    return _cross(r_phase8(feat), r_castle(feat), r_centrestate(feat))


def r_pcsh(feat):
    return _cross(r_phase5(feat), r_shelter(feat))


RULES.update({
    "phase x3": r_phase3,
    "phase x5": r_phase5,
    "centre state x3": r_centrestate,
    "wing majority x9": r_wing,
    "pawn struct (centre x wing)": r_struct,
    "P5 x castle (x80)": r_pc,
    "P5 x castle x centre (x240)": r_pcc,
    "P8 x castle x centre (x384)": r_p8cc,
    "P5 x castle x struct (x2160)": r_pcs,
    "P5 x kingregion (x605)": r_pk,
    "P5 x shelter (x1280)": r_pcsh,
})


# ------------------------------------------------------- the predicate pool
# Luka's three axes: king macro position, pawn macro position, material. Each
# predicate is something a bitboard engine can answer in a few instructions --
# a popcount, a shift, a mask -- because a router that costs more than the
# evaluation it routes is not a router.

def _pawn_masks(feat):
    valid, side, pt, sq = parts(feat)
    p = valid & (pt == PAWN)
    return p & (side == US), p & (side == THEM), sq % 8, sq // 8


def r_kingwing(feat):
    """Which third of the board each king lives on: a-c / d-e / f-h.

    The chess content is opposite-side castling, which changes the evaluation
    more than almost anything else and is a 2-instruction test."""
    z = lambda s: FILEG.to(s.device)[s % 8]
    return z(king_sq(feat, US)) * 3 + z(king_sq(feat, THEM)), 9, (KING,)


def r_kingrank(feat):
    """How far up the board each king is: home / middle / advanced.

    Reads king activity, which is what separates an endgame from a middlegame
    with the same material."""
    b = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2])
    ru = b.to(feat.device)[king_sq(feat, US) // 8]
    rt = b.to(feat.device)[7 - king_sq(feat, THEM) // 8]
    return ru * 3 + rt, 9, (KING,)


def _passed(feat, s, chunk=32768):
    """Own pawns with no enemy pawn ahead on their file or either neighbour.

    Chunked, and the pairwise tensors are int8: the natural one-shot version
    builds an (N, 32, 32) int64 intermediate, which is 4.8 GB at N = 580k moves
    and was killing the process."""
    out = []
    for i in range(0, feat.shape[0], chunk):
        us, them, f, r = _pawn_masks(feat[i:i + chunk])
        f, r = f.to(torch.int8), r.to(torch.int8)
        mine, foe = (us, them) if s == US else (them, us)
        d = (r.unsqueeze(1) - r.unsqueeze(2)) if s == US else (r.unsqueeze(2) - r.unsqueeze(1))
        near = (f.unsqueeze(1) - f.unsqueeze(2)).abs() <= 1
        block = foe.unsqueeze(1) & near & (d < 0)
        out.append((mine & ~block.any(2)).sum(1).clamp(max=2))
    return torch.cat(out)


def r_passed(feat):
    return _passed(feat, US) * 3 + _passed(feat, THEM), 9, (PAWN,)


def r_pawncount(feat):
    """Total pawns, 4 wide bins. The cheapest pawn-structure summary there is."""
    us, them, _, _ = _pawn_masks(feat)
    n = us.sum(1) + them.sum(1)
    return (n * 4 // 17).clamp(max=3), 4, (PAWN,)


def r_pawnfiles(feat):
    """How many files each side's pawns occupy: a spread/compactness read."""
    out = []
    for s in (US, THEM):
        us, them, f, _ = _pawn_masks(feat)
        m = us if s == US else them
        occ = torch.zeros(feat.shape[0], 8, dtype=torch.bool, device=feat.device)
        occ.scatter_(1, f, m)
        out.append((occ.sum(1) * 3 // 9).clamp(max=2))
    return out[0] * 3 + out[1], 9, (PAWN,)


def r_advance(feat):
    """Rank of each side's most advanced pawn, 3 bins. Space, cheaply."""
    us, them, _, r = _pawn_masks(feat)
    a = (r * us).max(1).values
    b = (7 - r).masked_fill(~them, -1).max(1).values.clamp(min=0)
    g = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    return g.to(feat.device)[a] * 3 + g.to(feat.device)[b], 9, (PAWN,)


def r_bishoppair(feat):
    valid, side, pt, sq = parts(feat)
    col = ((sq // 8) + (sq % 8)) & 1
    out = []
    for s in (US, THEM):
        m = valid & (side == s) & (pt == BISHOP)
        out.append((((m & (col == 0)).sum(1) > 0)
                    & ((m & (col == 1)).sum(1) > 0)).long())
    return out[0] * 2 + out[1], 4, (BISHOP,)


def r_majors(feat):
    """Rooks+queens per side, capped: the coarsest material read that matters."""
    valid, side, pt, _ = parts(feat)
    out = []
    for s in (US, THEM):
        m = valid & (side == s) & ((pt == ROOK) | (pt == QUEEN))
        out.append(m.sum(1).clamp(max=3))
    return out[0] * 4 + out[1], 16, (ROOK, QUEEN)


def r_minors(feat):
    valid, side, pt, _ = parts(feat)
    out = []
    for s in (US, THEM):
        m = valid & (side == s) & ((pt == KNIGHT) | (pt == BISHOP))
        out.append(m.sum(1).clamp(max=3))
    return out[0] * 4 + out[1], 16, (KNIGHT, BISHOP)


POOL = {
    "king: wing 3^2": r_kingwing,
    "king: rank 3^2": r_kingrank,
    "king: castle 4^2": r_castle,
    "king: region 11^2": r_kingregion,
    "king: shelter 16^2": r_shelter,
    "king: halfkp 64^2": r_kings,
    "pawn: centre 3": r_centrestate,
    "pawn: count 4": r_pawncount,
    "pawn: passed 3^2": r_passed,
    "pawn: files 3^2": r_pawnfiles,
    "pawn: advance 3^2": r_advance,
    "pawn: centre4 3^2": r_centre,
    "mat: phase 4": r_phase4,
    "mat: phase 8": r_phase8,
    "mat: bishop pair 2^2": r_bishoppair,
    "mat: majors 4^2": r_majors,
    "mat: minors 4^2": r_minors,
    "mat: rook x queen 6^2": r_pieces36,
    "mat: full 24^2": r_material,
    "mat: imbalance 40": r_phase_imbal,
}


# ----------------------------------------------------------------- stability
def king_moves(feat):
    """Every position repeated once per legal-geometry king step.

    Returns (perturbed, origin) so a changed index can be attributed. A step is
    kept if the target is on the board and not occupied by one of our own
    pieces; that is exactly the king's move set apart from castling and from
    moving into check, neither of which shifts the rate much."""
    N = feat.shape[0]
    ksq = king_sq(feat, US)
    slot = king_slot(feat, US)
    occ, ours = occupancy(feat)
    kf, kr = ksq % 8, ksq // 8
    out, src = [], []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            nf, nr = kf + df, kr + dr
            ok = (nf >= 0) & (nf < 8) & (nr >= 0) & (nr < 8)
            tgt = (nr.clamp(0, 7) * 8 + nf.clamp(0, 7))
            ok = ok & ~ours.gather(1, tgt.unsqueeze(1)).squeeze(1)
            g = feat.clone()
            g.scatter_(1, slot.unsqueeze(1).long(), (KING * 64 + tgt).unsqueeze(1))
            out.append(g[ok]); src.append(torch.nonzero(ok).squeeze(1))
    return torch.cat(out), torch.cat(src)


def pawn_moves(feat):
    """Every position repeated once per single pawn push (one or two squares)."""
    valid, side, pt, sq = parts(feat)
    occ, _ = occupancy(feat)
    m = valid & (side == US) & (pt == PAWN)
    out, src = [], []
    for step in (8, 16):
        tgt = sq + step
        ok = m & (tgt < 64)
        if step == 16:
            ok = ok & (sq // 8 == 1) & ~occ.gather(1, (sq + 8).clamp(max=63))
        ok = ok & ~occ.gather(1, tgt.clamp(max=63))
        rows, slots = torch.nonzero(ok, as_tuple=True)
        g = feat[rows].clone()
        g.scatter_(1, slots.unsqueeze(1),
                   (PAWN * 64 + tgt[rows, slots]).unsqueeze(1))
        out.append(g); src.append(rows)
    return torch.cat(out), torch.cat(src)


def captures(feat):
    """Every position repeated once per enemy piece removed.

    An UPPER bound on capture-driven change: it treats every enemy piece as
    capturable this move, which is far more than are actually en prise. A rule
    reading only material still changes on some real fraction of captures, and
    this column says which rules are exposed to that at all."""
    valid, side, pt, _ = parts(feat)
    m = valid & (side == THEM) & (pt != KING)   # kings are not capturable
    rows, slots = torch.nonzero(m, as_tuple=True)
    g = feat[rows].clone()
    g.scatter_(1, slots.unsqueeze(1), torch.full_like(slots.unsqueeze(1), PAD))
    return g, rows


def stability(fn, feat):
    """Fraction of king steps / pawn pushes that move the index."""
    base, _, _ = fn(feat)
    res = {}
    for name, gen in (("king", king_moves), ("pawn", pawn_moves),
                      ("capture", captures)):
        g, src = gen(feat)
        if len(g) == 0:
            res[name] = (0.0, 0)
            continue
        idx, _, _ = fn(g)
        res[name] = ((idx != base[src]).double().mean().item() * 100, len(g))
    return res


def main(nmax=200_000, dev="cpu"):
    b = Batcher("data/all.data", dev, batch=65536)
    fs, n = [], 0
    for feat, *_ in b.val_batches():
        fs.append(feat.to(dev)); n += len(feat)
        if n >= nmax:
            break
    feat = torch.cat(fs)[:nmax]
    print(f"n = {len(feat):,} real positions from the validation tail\n")
    hdr = (f"{'rule set':<28}{'declared':>9}{'seen':>8}{'perp':>7}"
           f"{'top%':>7}{'king mv':>9}{'pawn mv':>9}{'capture':>9}"
           f"{'reads':>26}")
    print(hdr); print("-" * len(hdr))
    for name, fn in RULES.items():
        idx, size, reads = fn(feat)
        assert int(idx.min()) >= 0 and int(idx.max()) < size, f"{name} out of range"
        c = torch.bincount(idx, minlength=size).double()
        p = c / c.sum(); nz = p[p > 0]
        perp = torch.exp(-(nz * nz.log()).sum()).item()
        st = stability(fn, feat)
        names = {PAWN: "pawn", KNIGHT: "knight", BISHOP: "bishop",
                 ROOK: "rook", QUEEN: "queen", KING: "king"}
        rd = "+".join(names[r] for r in reads)
        print(f"{name:<28}{size:>9,}{int((c>0).sum()):>8,}{perp:>7.0f}"
              f"{p.max()*100:>6.1f}%{st['king'][0]:>8.1f}%{st['pawn'][0]:>8.1f}%"
              f"{st['capture'][0]:>8.1f}%{rd:>26}")
    print(f"\nking steps enumerated: {stability(r_kings, feat)['king'][1]:,}"
          f"   pawn pushes: {stability(r_kings, feat)['pawn'][1]:,}")


if __name__ == "__main__":
    main()


# ------------------------------------------------- fine spaces for staging
# One per axis, deliberately finer than anything usable, because the pruning
# is what picks the buckets. Each is a few popcounts on bitboards.

def r_matfine(feat):
    """[knights 0-2][bishops 0-2][rooks 0-2][queens 0-1] per side: 54^2."""
    valid, side, pt, _ = parts(feat)
    out = []
    for s in (US, THEM):
        m = valid & (side == s)
        n = (m & (pt == KNIGHT)).sum(1).clamp(max=2)
        b = (m & (pt == BISHOP)).sum(1).clamp(max=2)
        r = (m & (pt == ROOK)).sum(1).clamp(max=2)
        q = ((m & (pt == QUEEN)).sum(1) > 0).long()
        out.append(((n * 3 + b) * 3 + r) * 2 + q)
    return out[0] * 54 + out[1], 54 * 54, (KNIGHT, BISHOP, ROOK, QUEEN)


def r_pawnfine(feat):
    """Pawns per file zone (a-c / d-e / f-h), capped at 3, per side: 64^2.

    Where the pawns are rather than how many: this is the "pawn macro
    position" axis, and it is three popcounts against three file masks."""
    valid, side, pt, sq = parts(feat)
    p = valid & (pt == PAWN)
    z = FILEG.to(feat.device)[sq % 8]          # 0 = a-c, 1 = f-h, 2 = d-e
    out = []
    for s in (US, THEM):
        m = p & (side == s)
        c = [(m & (z == k)).sum(1).clamp(max=3) for k in (0, 2, 1)]
        out.append((c[0] * 4 + c[1]) * 4 + c[2])
    return out[0] * 64 + out[1], 64 * 64, (PAWN,)
