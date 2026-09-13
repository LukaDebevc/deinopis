# 087 · cp-0004: quiescence TT sharing, +61 over cp-0003

_2026-09-12. `tools/checkpoint.sh -m "quiescence TT sharing" --net
nnue/runs/b1-20260909/m1-b1.nnue`. Full gate, pushed. Net sha
`ce1f265647c060a2` (same m1-b1 as cp-0003 — the gate measures the code
delta with the net held equal)._

## Result

Build ok, `cargo test --release` pass (45 tests), `perft-suite` ALL PASS,
bench **199091 nodes** (new fingerprint; was 243605 at cp-0003).

```
SPRT vs baseline [0,10]   8+0.08, conc 5, same m1-b1 net both sides
246 games: +65 =159 -22   +61.4 Elo  [+34, +90]   LOS 100.0%   LLR 2.95
SPRT: H1 accepted
```

Committed, tagged `cp-0004`, pushed (origin is the local bare repo);
baseline for the next gate is this build.

## What shipped

The measured delta is the Q-store: quiescence probes the shared TT and
stores twice per node (stand-pat fail-high as Lower, loop result with the
usual bound), with score cutoffs fenced by the `Q_DEPTH = 255` marker so
they cross only between quiescence nodes; move + static eval cross freely,
which is what orders quiescence by TT move now. Bench 243605 → 199091
(−18%), nps +29%. Agrees with the frozen-binary Q-store SPRT (+51.0 over
508 games, STATE item 5).

Riders (all dormant / bench-exact, verified by the unchanged fingerprint):
pawn-key field + correction-history table (`--corr`, killed in 085),
`fut_max_depth`/`fut_margin` params + gate (proxy-killed in 084), dual-file
seam + small-tail tier (`--dual`, killed in 081/082), asymmetric-contempt
knobs (killed STATE 2026-09-10), `--threads-a/--threads-b` + PGN `[A]`/`[B]`
tags + `elo --combine`, util counters, `matvec` width-12 arms. No rider
moves the bench by a node.

## Incidents

- First gate invocation died at 20 games: the launcher kept the tool call
  alive past its timeout and the timeout killed the gate's process group.
  Relaunched under `setsid` (own session) and polled with short calls —
  background matches must be detached from the launching call, never slept
  in. Nothing was committed by the dead run; the relaunch re-ran the full
  gate from scratch.
- The day's three kills ride along as dormant code + ledger entries, not as
  behavior: corr phase 1 (085, 586g 0.0), c0=200 (086, 422g −23),
  whole-node futility (084, proxy +4.6..+22cp).

## Decision

- cp-0004 is the baseline. Next Elo work starts from its binary.
- Ranked next (all queued behind this gate): continuation history for
  ordering, easy-move time management, then the unscreened-param queue
  (c_rank, max_base<1000, rfp_max_depth, delta_margin=50) under 086's bar
  (|t| > 2 vs default AND a smooth neighborhood). Eval-speed head/psqt
  remainder (~+7-11, thousands of games to resolve) stays deferred.
