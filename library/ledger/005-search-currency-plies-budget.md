**005 · Search currency: plies -> budget** · 2026-08-25
Replaced integer `depth` in the search with a fine-grained `budget` in units of
`PLY = 128`, with the price of every child set to exactly the old LMR table.
Groundwork for 006; on its own it should do nothing at all.
**Result:** `bench` **630367 nodes — byte-identical**, full perft suite exact,
tests pass. Which is the point: a refactor of the search that claims to be
behaviour-neutral is only believable if the fingerprint does not move.
**Decision:** kept. Also fixed a latent bug found on the way: `go nodes N` was
parsed into `Limits::nodes` and then never enforced, so `--tc nodes=N` was
silently an unbounded search. Now checked every node — not every 2048 — because
a node-limited search is the tuning objective's unit of work and has to be
exactly reproducible.

---
