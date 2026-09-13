## 5. Bucketing the PSQT instead of the read (`runs/psqtbuckets.log`)

| arm | val | vs bar |
|---|---|---|
| base r=512 x1 (bar) | 0.021093 | — |
| psqt x count (x8) | 0.020650 | −2.10% |
| psqt x kings (x4096) | 0.020877 | −1.02% |
| psqt x material (x576) | 0.020147 | −4.48% |
| **psqt x material+count (x584)** | **0.019810** | **−6.08%** |
| read x material+count | 0.017701 | −16.08% |
| read x material+count + psqt x material | 0.017676 | −16.20% |

**The two conditionings are the same information, not additive.** psqt x
material alone is −4.48% but adds only 0.12% on top of an already-conditioned
read — at the noise floor. LEDGER 023's "crossing beats summing" does **not**
extend to conditioning two parts of the model on the same bucket.

**Actionable:** psqt x material+count is −6.08% and needs no accumulator — 32
gathers from a 0.9 MB table on the existing accumulator-free `src/qeval.rs`.
