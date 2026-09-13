# 066 · Risk-sensitive WDL collapse loses three ways

_2026-09-07. Three SPRTs [0,5] at 8+0.08, same frozen binary both sides
(`nnue/runs/wdlrisk-20260907/chess`, bench 227236), same net named explicitly,
one CLI flag apart. Engine b986f5b + working tree, code since reverted (kept
nowhere — this entry is the record)._

## Question

The eval collapses W/D/L to `cp = 288.5 · logit(W + D/2)`, discarding the
draw-rate axis LEDGER 048 showed is the larger one. Does reweighting
win-vs-draw by game situation buy Elo — ahead maximise win-over-draw, behind
maximise draw-over-loss?

## Arms

- **root**: `k·(s·(w−d) + (1−s)·(d−l))` with `s = sigmoid(w−l)` fixed at the
  root, scored root-relative and negated to STM so minimax stays coherent.
  k=350, scale-matched (raw-risk std 0.91 vs cp std 321 on 3000 game FENs;
  corr(cp, risk) = 0.77, so it genuinely reorders).
- **node**: same formula, side recomputed per node (no root state, TT-clean,
  negates exactly across a ply).
- **contempt**: same idea through the standard readout — draw worth `1−s` to
  root, no scale. Takes over from `DrawValue`.
- Proven draws priced consistently per mode (level 0, ahead down to −2500,
  behind up to +2500); adjudication needs both engines' agreement, so one
  arm's inflated scores cannot manufacture results.

## Results

| arm | games | Elo | verdict |
|---|---|---|---|
| root, k=350 | 646 | **−62 [−83, −41]** | H0 |
| contempt-auto | 628 | **−50 [−69, −31]** | H0 |
| node, k=350 | 154 | **−254 [−324, −198]** | H0 |

Speed is not the cause (interleaved bench passes overlap on nps; node counts
deterministic per config: 227236 / 296971 / 231274 / 220952).

## Mechanism (exploratory — PGN rows, base's scores as neutral ruler)

All three arms *reach* +200cp positions less often than base (chances 259 vs
391, 215 vs 363, 27 vs 142) and convert/hold worse when they do. The
distortion steers ordinary play into worse positions, not just endings.
[INFERENCE] At level scores the formula is `k·0.5·log(W/L)` — the draw mass
cancels, so it systematically prefers sharp positions over sound ones. It pays
for variance. Node mode adds intransitivity (a different utility at every
node = minimax on nothing), which is why it is catastrophic rather than bad.

## Decision

Collapse-reweighting of this form is dead: two independent implementations of
the same direction (matched-scale logit-linear, scale-free through E) both
lose ~50, so the direction fails, not the k — no k-sweep was run. The
surviving hypothesis is using D in **search control** (extensions, pricing),
not in the eval number. Caveat: root/contempt scores are root-dependent and
cached across moves in the TT (second-order staleness); node mode is immune
and lost worst, so staleness does not explain the losses.
