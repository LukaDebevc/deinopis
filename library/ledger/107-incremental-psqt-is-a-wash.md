# 107 · An incremental psqt accumulator is a wash: −0.3% nps

_2026-09-19, desk box, on top of 106's padding. Built, verified, measured,
reverted. The code is not in the tree; this entry is what it cost to learn._

## The idea

106 measured `psqt`'s gather at **3.3% of the search** (n=12 paired nps): 54
random rows out of a 768-row table, on every evaluation. But psqt sums over
**the same feature set the ft accumulator already maintains incrementally**.
So carry the two perspective sums on the accumulator stack and update them in
the bit walk `push` already runs — ~6 row reads per made move instead of 54
gathers per evaluation, a 9x reduction in rows touched.

## Built and verified

- `WdlNet::fill_psqt` as the single reference gather; `WdlEval::pacc`,
  `[slot][2][PSQ]`, carried down the stack by the same `copy_from_slice` +
  add/sub loop as `acc`; `logits_upto` takes the two sums instead of gathering.
- The small (`--dual`) tail keeps the gather: it has its own psqt table and
  `--dual` lost 43 Elo and is dormant (081/082).
- [FACT] **`CHESS_ACC_CHECK=1`: 0 mismatches over 111,970 evals**, and bench
  came out **219718 — unchanged**, so the reordered f32 sum never moved a
  rounded centipawn across 219,718 nodes. 55 tests pass.
- [FACT] The check is severe, verified by negative control: dropping the
  black-perspective half of the update gives **111,685 mismatches, worst
  56 cp**. Restoring it gives 0.

## And it bought nothing

| | nps (14 paired runs, depth 12) |
|---|---|
| 106's padding only | 983,128 |
| + incremental psqt | 980,211 |
| | **−0.30%** (median −0.33%) |

Node counts identical in all 14. `evalprof` says the eval side did exactly
what it was meant to — psqt 64.2 -> 13.8 ns, eval −4.1% — and the search did
not get faster, so the cost landed in `push`.

- [INFERENCE] The two sides cancel because the cost is **cache misses, not row
  count**. The ft push streams `width * 2` bytes — 1 KB per changed square —
  through L1, which evicts the 12 KB psqt table between accesses. Six random
  rows then miss about as often as fifty-four do, and moving work from a
  per-eval loop to a per-move loop moves the misses without removing them.
- [GUESS] The fix this implies is to put each psqt row **in the same cache line
  as the ft row it accompanies**, so it rides a stream that is already paid
  for. That is blocked: `addsub!` dispatches on `width` (64/128/256/512) and a
  `width + 4` row falls into the dynamic path, which is the bounds-check trap
  LEDGER 055 removed. It would need the specialisation widened first, and it
  quantises psqt at `ACC_SCALE`, so it is a real change needing an SPRT, not a
  refactor. Not attempted.

## Decision

- **Reverted.** Neutral speed for ~90 lines across the accumulator, the loader
  and three call paths, plus an f32 drift channel that did not exist before.
  Same reasoning that sent 106's `qwf` back.
- `psqt`'s 3.3% stays on the table, with the one cheap idea for it now ruled
  out and the one expensive idea written down above.
- [DECISION] Worth keeping as method: the eval side was verified to 0
  mismatches with a negative control **before** anything was measured for
  speed, which is why the negative result took one nps run rather than a
  debugging session.
