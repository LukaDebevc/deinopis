## 032 — Tying accumulator halves does nothing (`runs/ratio.log`)

**Setup.** Rank 1024 accumulator, sweep what fraction of it is tied between the
two perspectives (0 / 25 / 50 / 75 / 100%), 8000 steps, lr 1e-3, K = 258.7,
warm-started from `ckpt-big/psqt_768-_64-_16-_1.pt`.

**Result.** val 0.021051 / 0.021038 / 0.021029 / 0.021021 / 0.021060. A 0.19%
spread on a ~0.1% noise floor: **flat**. Prediction on record was flat; it is.

**Decision.** Do not spend parameters on tying. Negative, recorded.
