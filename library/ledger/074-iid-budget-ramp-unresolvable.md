# 074 — Internal iterative deepening as a budget ramp: unresolvable, off

## Question
When the PV fails low (or any believed-best is wrong), the true best may have
been ranked late and searched at a heavily reduced budget or priced out. Should
a node with no TT move first climb a ramp of small budgets (1, 2, 4, ... plies,
capped below the full budget), carrying ordering through the TT, and stop
stepping when a bound is struck? I.e. root-style iterative deepening applied
locally, recursively.

## Setup
`src/search.rs`, behind two new `Params` (both default 0 = off):
`iid_min_depth` (PV-node gate, tried 3/4/6), `iid_min_nonpv` (non-PV gate,
tried 5/6/7), sharing the existing `iid_reduction` (plies the deepest probe
sits below the full budget, tried 1/2/3). Probe window is the node's own
(alpha, beta); the probe score is never returned — a shallow fail-high proves
nothing — it only ends the ramp early, and the probe PV is cleared so a
fail-low below cannot leak a shallow line. Recursion self-suppresses: after
the first probe stores a TT move, deeper same-node calls skip the ramp.

Bug found by the experiment: `iid_reduction=1` overflowed the stack. The probe
node re-adds the check extension on entry, so in check `budget − 1ply` came
back as an *equal* budget at the same ply — infinite recursion. Fixed at the
use site (`cap + ext < budget`), the same strict-decrease rule the move loop
enforces with `.min(budget − 1)`.

## Result
Root-regret proxy (`tune compare`, 1000 pos @ 60k nodes, `fishpack-1k-see`;
negative = IID better), all within noise (SE ~0.9–1.3):

| arm | Δ regret | differing |
|---|---|---|
| PV min4 r2 (single-shot scaffold) | −1.05 ± 1.12 | 118 |
| PV ramp min4 (r2 cap) | −0.05 ± 0.90 | 102 |
| + nonpv 7 / 6 / 5 | +0.29 / −0.09 / **+1.39** ± ~1.0–1.2 | 108 / 131 / 178 |
| PV ramp min4 r1 / r3 | −0.46 ± 0.97 / +0.96 ± 0.90 | 129 / 108 |
| PV ramp min6 | +0.13 ± 0.58 | 34 |

Fixed-depth time-to-depth (`trace`, same best moves everywhere): startpos d10
+0.8%, Italian d10 −0.2%, KID d10 +7.3%, Italian d13 +0.5%, KID d13 +5.7%.
No depth reaches further per node at any setting; sharp positions cost most.

## Mechanism (hypothesis, not measured)
This engine's 30%-of-nodes PVS re-searches already pay for ordering mistakes
after the fact: a late move that fails high gets re-searched at full budget
with updated killers/history/TT. IID front-loads that payment but does not
clearly reduce the total, so the net is ~zero minus probe overhead. If true,
IID can only win where the *first* move choice is systematically wrong and
expensive — worth checking via first-move fail-high rate before any retry.

## Decision
Default stays off (bench-exact at 218770 with the gate closed). No SPRT —
two proxy instruments and TTD agree there is nothing to promote. Seam kept:
null-window cut probes and higher gates are the only variants not yet
measured, and only a first-move-fail-high probe justifies them.
