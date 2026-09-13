## 033 — A coarsening must be trained, not distilled (`runs/shrink.log`)

**Setup.** Take the trained kings x 4096 read-bucketed model (val 0.019934) and
cluster its bucket tables post-hoc into k groups; compare against training the
coarse model directly.

**Result.** Post-hoc clustering keeps **-78.4%** of the gain at 16 clusters
(val 0.022005, worse than no buckets at all), 28.9% at 64, 67.7% at 256, 91.6%
at 1,024. A *trained* 16-bucket king model reaches 0.020292, better than no
buckets. So a coarsening is not a summary of a fine model.

Also in the same run: adding a bucket-independent shared read alongside the
bucketed read does nothing (0.017257 vs 0.017220), and a curriculum that trains
bucket 0 alone for the first 2,000 of 8,000 steps **hurts** (0.018027 vs
0.017220) — at a fixed step budget the warm-up costs more than it gives.

**Decision.** Never evaluate a candidate coarsening by clustering a trained
fine model. Train it. No shared component, no bucket curriculum.

**Scope (added by 043).** This forbids *distilling weights* — taking the fine
model's bucket tables as the coarse model's tables. It does **not** forbid
using a trained fine model as an *embedding to choose the partition* and then
retraining that partition end to end, which is a different procedure and is
what 043 does. The failure here is inheriting the weights, not consulting the
fine model.
