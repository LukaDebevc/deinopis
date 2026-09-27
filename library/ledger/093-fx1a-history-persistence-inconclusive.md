# 093 · FX-1a history persistence across moves: +5.7 over 3000, inconclusive, not shipped

_2026-09-15. Engine owns one `ThreadData` per thread, reused across moves,
cleared on `ucinewgame`; node-limited searches stay fresh (bench-exact,
default 0). Modes behind `--persist-hist`: 1 persist-all, 2 halve, 3
killers-cleared (`3e907cd`). SPRT [0,5] @8+0.08 conc 5, --hash 64, same
frozen binary both arms (bench 199091, pre-FX-3 tree), A `--persist-hist=1`
vs B base, m1-b1 net (`ce1f2656`) both sides, builtin book. Run dir
`~/chess-runs/20260915-fx1a/` (first durable dir, partial HK-4). With the
FX-3 bug unfixed at freeze time, --hash 64 dealt 32 MB effective — equal
both sides, so the comparison is fair. Mechanism check passed before the
match: repeated depth-8 search 12138 → 9045 nodes (−25%, same PV)._

## Result

| games | score | Elo | LLR |
|---|---|---|---|
| 3000 | +458 =2133 −409 | **+5.7 [−1, +12]** | 1.41 (bounds ±2.94) |

Inconclusive at the 3000-game cap. LOS 95.4%. PGN re-score agrees
(games Δ=0, Elo Δ=0.02 via `tools/rescore.sh`). Zero engine errors.

Trajectory never faded the 085/089 way, but never climbed either: −33 at
42g → +10.8 at 834g → +6.6 at 2542g → +5.7 at 3000g, LLR flat 1.5–1.7
since ~2400g. Same verdict class as S2 easy-move (090, +4.1 [−2,+10]).

## Reading

Persisted history is real (the −25% node drop proves the tables do work
across moves) and worth small-positive Elo — the interval's upper edge
(+12) excludes the GUESS-scale win (+5–15 in FX-1's text reads high now),
and MS-3 says [0,5] cannot ship a +3–6 effect inside 3000 games anyway.
[INFERENCE] The 089 suspects rhyme here: ordering is already good (O1),
so better-informed history has little left to buy.

## Decision

- **Not shipped.** `--persist-hist` stays dormant, default 0, bench-exact.
- FX-1b (halve) / FX-1c (killers-cleared) on hold: a +5.7 persist-all gives
  no reason to expect either to pass [0,5] alone. Candidate for a later
  small-positives bundle (MS-3), not for solo re-runs.
- RV-1/RV-2 do not need FX-1 shipped to run: both arms can carry
  `--persist-hist=1`, same binary — persistence as test harness, not as
  shipped behavior. That is the next history work, after the FX-3/4 gate.
- Box moves to the FX-3/4 gate (`~/chess-runs/20260915-fx34gate/`).
