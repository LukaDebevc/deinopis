# 084 · Whole-node futility-return-static proxy-killed, +4.6 to +22cp

_2026-09-12. One-line mirror of RFP on the alpha side, new `fut_max_depth`
(default 0, bench-exact) + `fut_margin` (default 100) in `search::Params`,
gate after NMP in the whole-node block: `corr_eval + margin*depth <= alpha`
returns `static_eval` as a fail-soft upper bound. Screened on the relabelled
1k-see corpus @15k nodes, m1-b1 net, `--b fut_max_depth=D[,fut_margin=M]`._

## Result

| arm | b − a (cp, negative = better) | depth |
|---|---|---|
| D=2 | **+12.70 ± 2.02** | 8.11 → 8.24 |
| D=3 | **+14.38 ± 2.14** | 8.11 → 8.35 |
| D=5 | **+22.32 ± 2.30** | 8.11 → 8.31 |
| D=1 | **+8.25 ± 1.86** | 8.11 → 8.07 |
| D=1, M=50 | **+7.81 ± 1.80** | 8.11 → 7.98 |
| D=1, M=200 | **+4.57 ± 1.81** | 8.11 → 8.14 |

Larger margins only asymptote to 0 from above — monotonically bad, no
interior optimum. |t| > 2.5 everywhere, so this is refuted, not unresolved.

## Why

A node failing low is exactly where the search must find the tactic that
gets back, and returning static forfeits captures too. RFP's mirror is not
symmetric: on the beta side the opponent still has to prove the refutation
in the null-move search that runs first; on the alpha side there is no
second opinion — the return skips the capture search that recovers. The
standard form (skip quiets, still search captures) is a different mechanism
and is NOT refuted by this; it needs move-loop surgery, not this gate.

## Decision

- Do not SPRT this form. Gate + params stay dormant in the tree (bench
  199091 exact, 45 tests pass), commented with this verdict, same deal as
  IID's dormant params (074). The param names are the right ones if the
  skip-quiets form is ever built.
- Same screen confirmed the live defaults are locally optimal: RFP 50/60,
  NMP ±, check_extension 500/2000, c_see 2000/6000/8000, delta 200 all read
  ≥ 0. Only `c0=200` hints positive (−1.36 ± 1.63) — queued for an SPRT
  behind the correction-history match.
