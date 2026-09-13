## 058 — The worse teacher trains the better student, and it is not smoothing

_2026-09-06. `nnue/wdlarms.py --labels/--temp`, `nnue/loader.py`,
`nnue/lc0/label.py`. Teacher `512x15-t79_9-swa-2016000`. One A100, GPU 2;
each arm is 80 s._

**Question.** LEDGER 057 measured the lc0 teacher as a *predictor* and found it
0.029 worse than the incumbent sigmoid, and concluded straight distillation was
not worth training on — while explicitly flagging that a teacher-quality number
cannot decide a student question. This asks the student question directly: hold
everything fixed and change only the distribution the teacher term supplies.

**Setup.** 52M records (`lc0/pilot.data`), 43M positions trained, the `gate`
arm, batch 16,384, lr 1e-3, lam 0.5, val_stride 8 (8M records held out to
score 1M positions). The two arms differ in `--labels` and in nothing else:
same data, same seeds, same steps, same val set. Three seeds each, because the
seed spread is 0.0008 and a single pair cannot see a 0.004 effect.

The referee is **`ce_out`**, cross entropy against the pure game outcome.
`target_dist` returns the one-hot at `lam=0` and never reaches the teacher, so
it is teacher-independent by construction — the only metric that can rank two
nets trained on different targets without favouring one of them.

**Alignment check, run before believing anything.** The "teacher alone on val"
line reproduces each teacher's own score through the training path: **0.65120**
for the sigmoid and **0.68034** for lc0, against **0.65450 / 0.68328** measured
by `teachercmp.py` on a different slice with different code. A label file that
had drifted out of lockstep with the records would not do that. The labels are
concatenated onto each record *before* anything shuffles, so there is no second
index to get wrong.

**Result.**

| teacher | its own ce_out | its H | student ce_out | n seeds |
|---|---|---|---|---|
| sigmoid, T=1.00 | 0.65120 | — | 0.76937 | 3 |
| sigmoid, T=1.10 — **entropy-matched** | 0.65192 | 0.67327 | 0.77000 | 1 |
| sigmoid, T=1.25 | 0.65708 | 0.71408 | 0.77111 | 1 |
| sigmoid, T=1.45 | 0.66824 | 0.76028 | 0.77362 | 1 |
| sigmoid, T=1.70 — **accuracy-matched** | 0.68535 | 0.80701 | 0.77752 | 1 |
| **lc0 512x15** | 0.68034 | 0.67277 | **0.76551** | 3 |

Per seed: lc0 0.76507 / 0.76558 / 0.76587, sigmoid 0.76963 / 0.76917 / 0.76932.
Every lc0 seed beats every sigmoid seed; the arms do not overlap.

**The teacher that predicts the outcome 0.029 WORSE trains a student 0.0039
BETTER.** Teacher quality and student quality point in opposite directions
here, which is exactly why 057 could not settle this and said so.

**The control that matters.** An external teacher differs from the sigmoid in
two ways at once — *where* it puts its mass and *how much* it spreads it — and
a softer target is label smoothing, a well-known free win that has nothing to
do with Leela. That is the boring explanation, and it is dead:

- lc0's mean target entropy is **0.67277**; the sigmoid at T=1.10 is
  **0.67327**. At that entropy match the sigmoid student scores **0.77000**,
  which is no better than unsoftened — lc0 wins by **0.0045**.
- Softening the sigmoid makes the student **monotonically worse** across four
  temperatures, 0.76937 → 0.77752. Smoothing is not merely absent, it is the
  wrong direction.
- Matching the teacher's *accuracy* instead (T=1.70, ce_out 0.68535 against
  lc0's 0.68034) gives 0.77752 — lc0 wins by **0.0120**.

So the gain is in the shape of the distribution: which of L, D and W the mass
sits on given the position. That is the thing the sigmoid structurally cannot
have, because its D is a function of (score, material) alone (LEDGER 048).

**Decision.** Relabel the full 893.5M corpus with this net and train a real
arm. Cost is **15.5 GPU-hours** at the measured 16,060 pos/s — the whole budget
is the forward pass; binpack decode runs at 4.4M entries/s and is free.

**What this does NOT show.** `ce_out` is a proxy and this project's own rule is
that a proxy winning is not a result — val loss has ranked these nets backwards
against Elo three times, most recently `gateh` (LEDGER 056). 43M positions is
also a short run and the ranking could move at 10B. The claim is "worth the 15
hours", not "distillation is better". The temperature arms are one seed each;
the effect is 5-15x the 0.0008 seed spread and the trend is monotone over four
points, but they are not three-seed results.

**Corrects LEDGER 057's decision**, which is edited at the source.
