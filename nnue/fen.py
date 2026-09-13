"""FEN -> the 32-slot stm-relative feature list the loader produces.

A second implementation on purpose: the tests and the cost measurements have to
derive the features from the FEN independently of `loader.GpuUnpacker`, or they
would only be testing that the code agrees with itself. Kept in step with
verify.py, which is the version cross-checked against the engine.
"""
from loader import PAD

ORDER = "PNBRQK"


def feats_from_fen(fen):
    board, stm = fen.split()[0], fen.split()[1]
    out = []
    for r, row in enumerate(board.split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            sq = (7 - r) * 8 + f
            if stm == "b":
                sq ^= 56
            side = 0 if (("w" if ch.isupper() else "b") == stm) else 384
            out.append(side + 64 * ORDER.index(ch.upper()) + sq)
            f += 1
    out = sorted(out)
    return out + [PAD] * (32 - len(out))


def decode(row):
    """(side, piece type, rank, file) for every live slot."""
    return [(i // 384, (i % 384) // 64, (i % 64) // 8, i % 8)
            for i in row if i != PAD]


FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pp2pppp/2n2n2/2pp4/3P4/2N1PN2/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "8/5k2/8/3p4/3P4/8/5K2/8 b - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/pp4k1/8/8/8/8/5KPP/8 w - - 0 1",
]
