"""FEN -> the 112 input planes of INPUT_CLASSICAL_112_PLANE.

For verification only. The production path (`label.py`) builds the same planes
straight from the packed training records on the GPU; this exists so the model
can be checked against positions a human can reason about.

Layout, transcribed from lc0's `encoder.cc`:
  0..103   8 history steps x 13 planes: ours P N B R Q K, theirs p n b r q k,
           then a repetition flag. History we do not have is filled by
           repeating the current position, which is what lc0 itself does when
           it is handed a bare FEN (`FillEmptyHistory::FEN_ONLY`).
  104..107 we_can_000, we_can_00, they_can_000, they_can_00
  108      all ones if WE are black
  109      the rule-50 ply count, raw (not /100 -- that is the hectoplies
           formats, and this net is not one)
  110      zeros            111      ones

Squares are lc0's bit order: bit i -> element i, so index 0 is a1. When black
is to move the board is mirrored vertically and the colours are exchanged, so
"ours" is always at the bottom -- the same convention our own records use.
"""
import numpy as np

PIECES = 'PNBRQK'


def planes_from_fen(fen):
    parts = fen.split()
    rows = parts[0].split('/')
    stm = parts[1]
    castle = parts[2] if len(parts) > 2 else '-'
    r50 = int(parts[4]) if len(parts) > 4 else 0

    board = {}
    for r, row in enumerate(rows):            # rows[0] is rank 8
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                board[(7 - r) * 8 + f] = ch
                f += 1

    black = stm == 'b'
    p = np.zeros((112, 64), np.float32)
    for sq, ch in board.items():
        s = sq ^ 56 if black else sq          # mirror rank when black moves
        mine = ch.isupper() != black
        idx = PIECES.index(ch.upper()) + (0 if mine else 6)
        p[idx, s] = 1.0
    p[:13] = p[:13]                            # step 0 is already written
    for i in range(1, 8):                      # history fill: repeat step 0
        p[i * 13:i * 13 + 13] = p[:13]

    we, they = ('kq', 'KQ') if black else ('KQ', 'kq')
    p[104] = 1.0 if we[1] in castle else 0.0   # we_can_000  (queenside)
    p[105] = 1.0 if we[0] in castle else 0.0   # we_can_00
    p[106] = 1.0 if they[1] in castle else 0.0
    p[107] = 1.0 if they[0] in castle else 0.0
    p[108] = 1.0 if black else 0.0
    p[109] = float(r50)
    p[111] = 1.0
    return p.reshape(112, 8, 8)
