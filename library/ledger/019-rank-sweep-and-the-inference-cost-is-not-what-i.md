**019 · Rank sweep, and the inference cost is not what I first computed** · 2026-08-27
All arms warm-started from the trained PSQT (018), 8000 steps x 65536.

| r | params | val loss |
|---|---|---|
| 16 | 13,090 | 0.025643 |
| 64 | 50,050 | 0.022650 |
| 256 | 197,890 | 0.021424 |
| 768 (full rank) | 592,130 | **0.021267** |

The spectrum is concentrated: **r=256 captures 99.3% of full rank** with a third
of the parameters, and 16 -> 64 is a far bigger step than 256 -> 768.
**Cost, corrected — Luka's point, and it overturns the analysis this entry was
going to record.** eval = sum over ordered pairs of active features, so lifting
a piece off square p costs `-2*sum_{i in S} W_pi + W_pp`: **|S| ~ 32 gathers,
not 768**. There is no accumulator. `a = Wx` is only needed when a nonlinearity
sits downstream and has to see every hidden unit; a quadratic form has none.
Verified exact against from-scratch evaluation over 4000 toggles.

| | ops / quiet move | state per ply |
|---|---|---|
| quadratic, direct | **64** | **none** |
| quadratic, accumulator | 1536 | 3 KB |
| ReLU h=256 | ~1024 | 1 KB |

Two consequences. Cost **scales with piece count** — 64 ops in the opening, 12
in a K+P ending — where an NNUE is flat. And with no accumulator there is no
per-ply state, which retires ARCHITECTURE.md's "the NNUE accumulator (4 KB) is
far too big to copy per node" and the `push`/`pop` on `Evaluator` that exists
for it. Honest deduction: 32 scattered gathers inside a 1536-byte row touch ~18
of its 24 cache lines, so *bytes moved* are similar to streaming the row — the
win is ALU work and the absence of state, not memory traffic.
**Decision:** deploy the **full-rank W** as a 768x768 i16 table, 1.18 MB. The
W-form cost does not depend on rank, so low rank buys nothing at inference and
full rank is strictly free. Low rank remains the right *training*
parameterisation.
**Consequence for 016's ceiling result:** warm-started quadratic r=256
(0.021424) beats cold ReLU h=256 (0.021500) at ~1/16 the per-node cost. The
degree-2 gap measured in 016 was partly a warm-start deficit, not an
architectural one. 016's "ReLU leads by 2-4%" should be read as an upper bound
on the ceiling, not a measurement of it.

---
