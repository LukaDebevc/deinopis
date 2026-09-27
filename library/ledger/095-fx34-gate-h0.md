# 095 · FX-3/4 gate: full TT + fail-low own-move only, H0 (−4.6 over 2622)

_2026-09-15. FX-3 (TT `resize` allocates the full requested hash — the old
code halved unconditionally, so `--hash 64` dealt 32 MB effective) + FX-4
(fail-low keeps the stored move only when the entry is ours — without the
key check a fail-low inherits another position's move), commit `0c28087`
with F1 tests (`table_size_matches_request`, `fail_low_store_keeps_*`).
Bench 199091 → **199115** (deterministic, 2 identical runs). Gate:
`chess-new` (199115, HEAD) vs `chess-old` (199091, FX-1a frozen,
sha-verified), same m1-b1 net (`ce1f2656`) both sides, SPRT [0,5] @8+0.08
conc 5 --hash 64, builtin 43-line book. Run dir
`~/chess-runs/20260915-fx34gate/` (both binaries, net, SHA256+bench,
`sprt.log`, `games.pgn`)._

## Result

| games | score | Elo | LLR |
|---|---|---|---|
| 2622 | +360 =1867 −395 | **−4.6 [−11, +2]** | −2.98 (bounds ±2.94) |

H0 accepted, LOS 9.0%, 1311 pairs, 9815 s wall. Draws 71.2% — the usual
rate. Zero engine errors, zero time forfeits.

## Reading

- Clean reject, not inconclusive: the interval's upper edge (+2) excludes
  the +5 H1 threshold.
- Attribution is muddy by design — two changes, one gate. Neither had an
  Elo prior: FX-3 doubles effective hash at this TC, where hash pressure is
  low; FX-4 is a correctness fix (the F1 test proves the foreign-move
  leak) that is rare by construction. The gate was for shipping a
  bench-moving commit, not for a hoped gain.
- The +24-node bench move is real (deterministic ×2); which half moves it
  is unseparated. The nps split in SHA256 (686k vs 823k) is box load, not
  a measurement — ignore it.
- The old arm carries the foreign-move bug through 1311 pairs with zero
  errors, consistent with H0: the bug is rare in practice.

## Decision

- **Not shipped.** Revert `0c28087`; the ship tree goes back to bench
  **199091** — the tree S1/S2, FX-1a and the bundle (094) ran on.
- FX-3/4 stay out unless re-tested solo. FX-4's correctness case alone is
  not an Elo case.
