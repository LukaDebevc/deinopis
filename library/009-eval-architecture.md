# 009 — Eval architecture: sides, features, buckets, and the shape of the net

Session of 2026-08-27. Everything here is **validation loss on held-out
positions**, which is a proxy. Nothing in this document has played a game.
Bar for the whole session: `base r=512 x1` = **0.021095**, replicated at
0.021083 / 0.021085 / 0.021086 / 0.021093 / 0.021095 / 0.021096 across six
separate invocations — a 0.06% spread, consistent with the 0.1% noise floor
from LEDGER 023. Warm-start PSQT floor (the linear model) = **0.031219**.

Common config for every run: `--data all.data --steps 8000 --lr 0.001
--K 258.7 --init-psqt ckpt-big/psqt_768-_64-_16-_1.pt`, batch 65,536,
lambda 0.9, AdamW, one epoch discipline (8000 x 65536 = 524M = 0.59 epoch of
the 894M pool). Python is `python3`.

---

This document is an INDEX. Each section below is its own file, so a
question about one part of the study costs one section and not 1,267
lines.

- [1. Which blocks of the quadratic form matter (`runs/sides.log`)](009-eval-architecture/01-which-blocks-of-the-quadratic-form-matter.md) · 24 lines
- [2. Perspective, rank, and the algebraic ceiling (`runs/persp.log`)](009-eval-architecture/02-perspective-rank-and-the-algebraic-ceiling.md) · 48 lines
- [3. Extra input features — the "psqt++" question (`runs/extras.log`)](009-eval-architecture/03-extra-input-features-the-psqt-question.md) · 36 lines
- [4. Extra features *on top of* bucketing (`runs/extras-buck.log`)](009-eval-architecture/04-extra-features-on-top-of-bucketing.md) · 20 lines
- [5. Bucketing the PSQT instead of the read (`runs/psqtbuckets.log`)](009-eval-architecture/05-bucketing-the-psqt-instead-of-the-read.md) · 19 lines
- [6. The NNUE head (`runs/deep.log`)](009-eval-architecture/06-the-nnue-head.md) · 35 lines
- [7. Can bucket tables be merged? (scratchpad `merge.py`, `merge2.py`)](009-eval-architecture/07-can-bucket-tables-be-merged-scratchpad-merge-p.md) · 61 lines
- [8. Routing rules: balance, completeness, stability (`nnue/rulestats.py`,](009-eval-architecture/08-routing-rules-balance-completeness-stability.md) · 556 lines
- [9. Luka's proposed architecture — what the arithmetic says](009-eval-architecture/09-luka-s-proposed-architecture-what-the-arithmet.md) · 67 lines
- [Tools built this session](009-eval-architecture/10-tools-built-this-session.md) · 31 lines
- [Verification that the refactor is honest](009-eval-architecture/11-verification-that-the-refactor-is-honest.md) · 15 lines
- [Bugs found and fixed this session](009-eval-architecture/12-bugs-found-and-fixed-this-session.md) · 43 lines
- [What is running / queued](009-eval-architecture/13-what-is-running-queued.md) · 39 lines
- [Known issues](009-eval-architecture/14-known-issues.md) · 19 lines
- [The price was measured on the wrong positions](009-eval-architecture/15-the-price-was-measured-on-the-wrong-positions.md) · 75 lines
- [The designed coarsening lost (`runs/merged.log`)](009-eval-architecture/16-the-designed-coarsening-lost.md) · 63 lines
- [What actually costs a refresh (and what does not)](009-eval-architecture/17-what-actually-costs-a-refresh-and-what-does-no.md) · 27 lines
- [Rank was never the constraint; the single reader was](009-eval-architecture/18-rank-was-never-the-constraint-the-single-reade.md) · 19 lines
- [Rung 1: the bucketed PSQT, in the engine](009-eval-architecture/19-rung-1-the-bucketed-psqt-in-the-engine.md) · 20 lines
- [Open questions](009-eval-architecture/20-open-questions.md) · 12 lines
