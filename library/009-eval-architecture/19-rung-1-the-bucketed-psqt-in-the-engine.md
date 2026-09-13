## Rung 1: the bucketed PSQT, in the engine

The only thing here that needs no accumulator. Conditioning the linear term
keeps the 32 gathers and widens the table; file version 4 carries 584 x 768
extra diagonals (1.8 MB), summed into `diag` for the position's bucket.

| arm (r=768) | val | bench nps |
|---|---|---|
| deploy base | 0.021062 | 2.26 M |
| deploy psqt x material+count | 0.019805 (−5.97%) | 1.95 M (−14%) |

The bucket index is the one part of the eval the engine re-implements rather
than reads from the file — a mismatch would be silent — so it is checked two
ways: `chess evalfen` prints it, and it agrees with `rulestats.py` on 20,000
real search positions (134 distinct material buckets, 0 mismatches);
`verify.py` with `CHESS_PFAMS` re-derives the whole eval in float torch from
the checkpoint rather than the file (0 mismatches, +0.019 cp mean).

SPRT in flight. **−3.9 Elo [−37, +29] at 178 games**; an early +30.3 at 92
games was noise. See LEDGER 037.
