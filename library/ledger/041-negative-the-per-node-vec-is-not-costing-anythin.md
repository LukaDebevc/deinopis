## 041 — Negative: the per-node `Vec` is not costing anything

`negamax` heap-allocates `quiets_tried: Vec<Move>` at the first quiet move of
every interior node — a `malloc` in the hottest loop in the program, and an
obvious-looking thing to move to the stack. Replaced with `[Move; 64]` plus a
count: node counts identical (334,110 at depth 9, 1,474,112 at depth 12, so the
cap never binds on the corpus) and speed **2,910,934 against 2,876,775 nps,
+1.2%, n=3 each — inside the run-to-run spread**.

Reverted. glibc caches an allocation this size and the list is short, so the
change buys nothing measurable and costs a cap, past which the history penalty
silently stops accumulating. Recorded so nobody spends the afternoon on it
twice. If the allocator ever *does* show up, it will be under Lazy SMP with
many threads, and it should be measured there rather than at one thread.

---
