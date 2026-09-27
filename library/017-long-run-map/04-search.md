# 017/04 · Search pieces (SR) and learned control (LC)

## SR · Standard search pieces

Each is one SPRT (P4) behind a default-off `Params` flag (P2). Information
that should affect non-first moves goes in `features!`; anything that must
reach the first move (extensions, whole-node rules) is a node-level
`Params` term (013: a feature can never reach rank 0). After ~3 of these
ship, re-run SPSA campaign A (TU-5).

**SR-1 · futility, skip-quiets form** (`fut_max_depth`; since the split,
this param enables ONLY the skip-quiets form — the proxy-killed whole-node
gate additionally needs `fut_whole_node=1`, default 0, never SPRT on.
`--set fut_max_depth=3` is the pure SR-1 arm: proxy-neutral −0.3±0.7 on
fresh 4k, 881 positions differ).
Variants: a. depth 3, margin 100 (queued); b. depth 2, margin 75; c. depth
5, margin 150; d. margin with a base: `fut_base + fut_margin·depth` (add
`fut_base`, try 50/100). Must not fire at PV nodes, in check, or before a
move has set `best_score` (already coded).

**SR-2 · singular extensions** (built, `se_min_depth`). Variants: a. min
depth 8, margin 3 cp/ply; b. min depth 6, margin 2; c. a. + double
extension when the verification fails low by > 3·margin (cap 2 per root
path; add a per-path counter to `PathInfo`); d. a. + negative extension
(TT move gets −½ ply) when the verification *fails high* above beta
(multi-cut: return `sbeta` if `sbeta ≥ beta`). Singular work is depth-heavy:
if a. is flat at 8+0.08, rerun it at 20+0.2 before killing (MS-2).

**SR-3 · evasion ordering by SEE** (built, `qevade_see`). Bench must move at
freeze time or the flag is not reaching anything.

**SR-4 · razoring** — the fixed version of 084. 084 returned static eval
and lost because it skipped the captures that recover; razoring *searches*
them. At non-PV, not in check, depth ≤ `raz_max_depth`, `static +
raz_margin·depth ≤ alpha` → `v = quiescence(alpha, alpha+1)`; if `v ≤ alpha`
return `v`, else continue the normal search. Variants: a. depth ≤ 2, margin
250; b. depth ≤ 3, margin 200; c. depth 1 only, margin 300. Proxy-screen
first (P5): 084's proxy reading (+4.6..+22) is the thing to beat.

**SR-5 · "improving" flag.** `improving = static_eval(ply) >
static_eval(ply−2)` (false if either side was in check). Needs a
`static_eval` field in `PathInfo`. Variants, separate arms: a. RFP margin ×
(1 − ½·improving); b. a `features!` Cost/Sigma term `c_notimproving` (price
higher when not improving); c. futility (SR-1) margin scaled likewise; d.
null-move allowed only if improving or `static ≥ beta + 50`. [GUESS] +5–15.

**SR-6 · ProbCut.** At non-PV, depth ≥ 5, |beta| not mate: for captures
with SEE ≥ `pc_margin`, search at budget − 4 plies with window
`[beta+pc_margin−1, beta+pc_margin]`; a fail-high returns. Variants:
margin 150/200/250; depth gate 5/6. [GUESS] +5–15.

**SR-7 · null move reduction by eval gap.** `R = 3 + depth/4 +
min((static − beta)/nmp_eval_div, 3)`. Variants: divisor 150/200/250.
Also a. verification search at depth ≥ 12 (zugzwang guard).

**SR-8 · expected cut-node term** (`features!`, Cost, free). A node
reached as a non-first child searched with a null window expects to cut;
reduce more there. Pass `cut_node: bool` down (child of a cut node's first
move is an all-node, etc.). Variants: coefficient 250/500/1000 mp.

**SR-9 · history-based pruning of late quiets.** Price already has `c_hist`;
add a hard rule variant: at depth ≤ 3, a quiet with history < −4000·depth
is skipped. Compare against raising `c_hist` via SPSA (TU-5); if SPSA
already gets the Elo, drop the rule.

**SR-10 · quiet checks at the first quiescence ply.** Generate quiet
checking moves at `qd == 0` only, SEE ≥ 0. Weak prior (O8 neutral in the
main search), different mechanism.

**SR-11 · recapture extension** (node-level, Sigma): the child of a
recapture on the same square gets +¼..½ ply. `c_recap` fires too rarely as a
price term (065); as an extension it reaches the first move.

**SR-12 · passed-pawn push to the 7th** as an extension (node-level).
Variants: +½ ply; +1 ply only if SEE ≥ 0.

**SR-13 · aspiration window growth and re-centering.** In TU-1/TU-5; a
structural variant: on fail-low, keep beta and reset alpha to −∞ after two
failures.

**SR-14 · Lazy SMP Elo** (never measured): same binary, `--threads-a 6
--threads-b 1`, `--concurrency 1`, 10+0.1, SPRT [0,50]. Then thread voting
(root move by summed depth-weighted votes) as an SPRT at 6 threads.

**SR-15 · mate and draw details.** a. 50-move rule proximity: scale eval
toward 0 as `halfmove_clock` → 100 (`eval·(200 − hmc)/200`); b. an
insufficient-material and KvKP-trivial draw recogniser before eval.
Both cheap; (a) is standard and often worth a few Elo.

## LC · Learned control — the project's bet

The price list is a linear policy fitted by a tool that can resolve ~10
parameters. Growing it needs one label per priced child (004, 005 category
C). This is the stated binding constraint and has never been built.

**LC-1 · PROBE → tool · the per-child corpus.** New `chess oracle`
subcommand. For each of ~1000 positions (4k corpus FENs): run a
**full-width fixed-depth alpha-beta** at d = 6, 7, 8 (price list off,
`skip_below` off, RFP/NMP/futility off, TT on, alpha-beta cutoffs kept),
recording at every node where the *real* search would price a child: the
node's features (every `features!` value, rank, depth, static eval, D,
material, `|static − qsearch|`), the price the real search would charge, and
the label `rel = 1[V_d(child) > alpha]` plus `V_d(child) − alpha`. Output
one row per priced child (CSV or `.npz`). Check: the d=8 value of the root
equals a plain full-width search; run time per position recorded
(004 estimated 2–15 s at d=8). Then EC-5.

**LC-2 · SPRT · two-stage gap: price from a quiescence search of the
child.** 004 measured r² 0.03 (today's `move_gain` gap) → **0.71**
(qsearch of the child). Above a budget threshold, run `qsearch(child)` for
every non-first child before the loop, use `gap = best − (−qsearch)` in
the price, and reorder by it. Variants: a. threshold 5 plies, `c_gap` 2000;
b. threshold 3 plies, `c_gap` 4000; c. threshold 7 plies, `c_gap` 7000
(004's attenuation-corrected value); d. a. with the qsearch result also
used for ordering only (price unchanged) — separates the ordering effect
from the pricing effect. Pay attention to the cost: nodeprof before and
after; a node budget cannot see it (013).

**LC-3 · SPRT · a σ term from stage 1.** With LC-2's qsearch in hand,
`|static(child) − qsearch(child)|` is the best σ proxy on record (092:
Q5/Q1 2.06, survives material control). `features!` entry, Sigma group,
`Log` kind. Variants: coefficient −1000/−2000/−4000 mp per log2 (sign: high
σ → cheaper price → more budget).

**LC-4 · PROBE → SPRT · fit the price by logistic regression on LC-1.**
Target: `rel`. Features: everything LC-1 records. Model: logistic, L2,
fit at d=6,7,8 separately; check each coefficient moves along a smooth
curve with d (004's scale warning) and verify at d=10 on 100 positions.
Convert the fitted log-odds to milli-plies through γ (004: PLY/γ ≈ 7900) and
SPRT it against the SPSA-tuned price list (TU-5), not the default.

**LC-5 · SPRT · a non-linear price.** Same corpus, a small GBM or a 2-layer
MLP (≤ 32 hidden, integer-quantised) over the same features. Must be cheap
per child: budget its ns with nodeprof before training anything.

**LC-6 · PROBE · the relaxed teacher** (STATE Next 9). Label the proxy
corpus with pruning *off* (RFP/NMP/futility/skip_below off, 400k nodes) and
compare with today's teacher on the same positions: fraction of positions
where the best move differs. If > ~10%, the proxy has been rewarding
self-agreement; switch teachers.

**LC-7 · PROBE · sibling spread in real positions** (001, open; 004 found
it confounded with material on PeSTO). On LC-1 rows, within material bins:
does the spread of children's `V_d` predict how often the shallow search's
root move differs from the d=8 one?

**LC-8 · SPRT · soft backup at PV nodes only** (001: 35–47% regret cut in
simulation). At PV nodes (all children searched with open windows anyway),
return `τ·logsumexp(v/τ)` instead of `max`. Variants: τ = 5/15/30 cp; only
at root; only at plies ≥ 2. Bound handling: only on exact scores.

**LC-9 · GPU → SPRT · an uncertainty head.** Train a small head on the net's
features to predict `ln|static − V(64k)|` (092 method, scaled up to ~10M
positions with EC-4-style labelling). Use it as the σ term (LC-3) or to
scale RFP/futility margins (`margin ∝ σ̂`). Compare against LC-3's free
proxy, which needs no training.

**LC-10 · SPRT · σ in time management** (004: "across positions → a TM
signal"). Root σ̂ from `|static − V(depth 4)|` at the root, or from
LC-9's head; allocate `time × clamp(σ̂/σ̄, 0.6, 1.6)`. Variants: the root
quantity; the average over the root's top 3 moves.
