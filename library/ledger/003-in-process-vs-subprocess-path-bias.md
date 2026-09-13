**003 · In-process vs subprocess path bias** · 2026-08-24
The match runner can play the engine under test in-process or over a pipe.
These are different code paths, so a match between them could carry a
systematic bias — and the checkpoint gate would otherwise compare a new build
in-process against an old binary over a pipe, adding that bias to every
decision.
**Setup:** 400 games, 100ms/move, the same build against itself, in-process vs
subprocess, 43-line book, colour-reversed pairs.
**Result:** **+4.3 Elo [-18, +27]**, +103 =199 -98. Consistent with zero, but
only resolved to about ±22 — the same order as the effects we care about, so
"it's probably fine" is not good enough.
**Decision:** the checkpoint gate runs **both sides as subprocesses**, where
the asymmetry cancels by construction instead of being assumed small. The
in-process player is kept for `Params` experiments, where both arms are
in-process and it cancels there too. Write-up:
`library/002-measuring-strength.md` §7.

---
