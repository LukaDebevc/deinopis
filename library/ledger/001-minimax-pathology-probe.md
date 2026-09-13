**001 · Minimax pathology probe** · 2026-08-24 · `examples/pathology.rs`
Synthetic depth-10 tree, tunable sibling correlation, noisy frontier eval.
Asked whether minimax pathology and PV instability are real, and whether
alternative backup operators fix them.
**Result:** Pathology is governed by one ratio, `sigma_eval / sibling_spread(K)`.
Chess is safe because sibling spread is large, not because minimax is sound.
Soft (softmax-weighted) backup cuts regret 35-47% at mid depth and cuts PV flip
rate 3-5x. Double-estimator backup is clearly **worse** — it fixes value bias
when the problem is selection variance.
**Decision:** Park it. Soft backup is incompatible with alpha-beta pruning
except at PV nodes. Revisit as roadmap #5 once there is a measured baseline to
test against. Full write-up: `library/001-minimax-pathology.md`.

---
