**002 · target-cpu=native** · 2026-08-24
The default x86-64 baseline emits **zero** `popcnt` instructions; `count_ones()`
becomes a shift-and-mask sequence.
**Result:** perft 213.8 -> 227.9 Mnps (+6.6%), search 3.51 -> 3.64 Mnps (+3.8%).
Node counts byte-identical, so it is pure speed. Smaller than expected — the
code is not popcount-bound.
**Decision:** Kept, in `.cargo/config.toml`. Worth ~2-4 Elo. Note it binds the
binary to this machine; use `-C target-cpu=x86-64-v3` for anything portable.

---
