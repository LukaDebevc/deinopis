## 057 — An lc0 teacher is 0.05 worse on the scalar, and its D does not transfer

> **CORRECTED 2026-09-06 (LEDGER 058).** The measurements below stand; the
> DECISION drawn from them was wrong. Training a student on these labels
> beats the incumbent sigmoid teacher by 0.0039 ce_out over three seeds,
> and an entropy-matched sigmoid does not reproduce it — so the worse
> predictor is the better teacher, and the full relabel IS worth its 15.5
> GPU-hours. This entry is the reason it got asked properly: everything
> here is about the teacher as a predictor, which was never the question.

_2026-09-06. `nnue/lc0/`, `nnue/extract --aux`. Teacher
`512x15-t79_9-swa-2016000` (lczero.org `more_nets`), a 15-block SE-resnet from
run T79 — the run that generated this corpus. One A100, GPU 2._

**Question.** Our labels are one Stockfish cp score plus one game outcome, and
the WDL "teacher" is a fixed sigmoid of that scalar (LEDGER 048), so `D` is a
function of (score, material) and can never say *this* ending is drawn. Does a
real Leela net's WDL carry anything the sigmoid cannot?

**Setup.** 52M records extracted from `test79-2022-04-apr-12tb7p.min-v2`
(the same `linrock/test79` source family as the corpus), plus a new `--aux`
side file carrying the castling, side-to-move and rule-50 state the 32-byte
record drops. Teachers are scored on 1M held-out records by `ce_out`, cross
entropy against the game outcome — teacher-independent by construction, and the
same number `wdlarms.evaluate` reports. Slice-to-slice noise on `ce_out` is
**±0.003** (0.65450 vs 0.65166 on two disjoint 1M slices), so the effects below
are 6-7x noise.

**Two instrument checks first.** `planes.py` (Rust bit-packing + a GPU argsort
decode) and `encode.py` (Python FEN parsing) share no code and produce
**bit-identical** 112-plane inputs on 4000 records. The K that best maps lc0's
log-odds onto the Stockfish cp scale came out at **260** on a 20-cp grid,
against LEDGER 016's independently ML-fitted **258.7**.

**Result.** ce_out on 1M held-out records, lower is better:

| teacher | scalar E from | D from | ce_out | onll |
|---|---|---|---|---|
| incumbent | Stockfish | sigmoid | **0.65450** | 0.55749 |
| hybrid | Stockfish | lc0 | 0.67364 | 0.55749 |
| lc0 raw | lc0 | lc0 | 0.68328 | 0.57257 |
| lc0 -> sigmoid, K=240 | lc0 | sigmoid | 0.70438 | 0.57257 |

**The scalar is where the loss is.** Holding the shape fixed, lc0's E costs
**+0.0499** against Stockfish's. One forward pass of a 512x15 net is a much
worse outcome predictor than a Stockfish search, which is not a surprise and is
most of the -0.029 total.

**The shape gain is an artefact, and this is the part worth remembering.**
Holding lc0's scalar, lc0's D beats a sigmoid of that same scalar by
**+0.0211** — which reads like dark knowledge. But holding *Stockfish's*
scalar, lc0's D **loses** to the sigmoid by **-0.0191**. The two are symmetric
to within noise, which is the signature of a swap penalty: each teacher's D is
calibrated to its own E and neither transfers. Reporting only the first
direction would have been a false positive.

**Miscalibration was ruled out, not assumed.** lc0 over-predicts D by 0.01-0.02
in nearly every bin of the draw table. Fitting a temperature and a draw-logit
bias on a *disjoint* 1M slice returns (1.00, -0.10) and buys **0.0007**. The
numbers above are with that calibration applied.

**Decision — SUPERSEDED, see LEDGER 058.** What was written here: "straight
distillation from a 1-node lc0 net is not worth training on; it replaces a
better scalar with a worse one and its D buys nothing once paired with a scalar
it was not calibrated against."

Both halves of that sentence are still true *as statements about the teacher*,
and the conclusion drawn from them was still wrong. The caveat recorded in the
same breath — that distillation is known to help even when soft labels are less
accurate than hard ones, because the value is in the relative probabilities
rather than the argmax, so `ce_out` of a *teacher* cannot see it — turned out
to be the whole story. The 52M labels were computed and kept
(`lc0/pilot.lc0.npy`) precisely so the student question could be asked
directly, and when it was, the lc0-taught student won.

The lesson to keep: a measurement of the teacher is not a measurement of the
student, and no amount of care taken over the first one makes it into the
second.

**What would change the answer.** More nodes per position (the scalar gap is
what dominates, and search is what fixes it), or a policy target, which is a
genuinely new signal rather than a reshaping of one we have. Both are still
worth having; neither was needed to make the labels pay.

**Loose ends.** `chess893.data`'s exact source binpack is unidentified — its
recipe was not recorded and none of the eight `linrock/test79` files reproduce
its first record, so these labels are for a fresh 52M extract, not for the
existing corpus. `nnue/extract` also silently drops the first game of any
binpack (`ply <= prev_ply` is false for the first entry); one game in 801k,
documented in place and deliberately not fixed, since fixing it would shift
every record and break alignment with data already on disk.
