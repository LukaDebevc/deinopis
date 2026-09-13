## 031 — Routing rules: the refresh price was half, and a search found the hole

**Setup.** Search for a bucket rule ("routing rule") that splits games into
types cheaply: score candidate coarsenings by *refresh rate* (how often a made
move changes the bucket, so an accumulator must be rebuilt) and by effective
bucket count. Coarsenings built by merging cells of a fine axis using the
transition graph over made moves. `nnue/rulestats.py`, `coarsen.py`,
`stagedrules.py`, then `merge2.py`.

**Result — negative, and it invalidates earlier entries.** `nnue/attic/refresh.py`
applies each move in the *mover's* perspective, and the training records are
stm-canonical, so the opponent's pieces never move in that view. A rule that
reads only the opponent's half therefore measures as **free**. The staged
search found exactly that hole: its best 16-bucket king rule ignored our own
king, scored 0.00%, and actually costs 7.65% per ply. Verified by mirroring the
record and counting both accumulators (n = 60k positions): HalfKP **16.20%**
per ply, material 24^2 **8.00%**, the "0.00%" rule **7.65%**.

Every refresh number recorded before this is **per mover-accumulator**, i.e.
half. Side-symmetric rules double and keep their ordering; asymmetric rules are
unpriced. VOID: the `coarsen.py` king frontier (graph-64 "0.23%"), the 7%
budget as stated, and every rule in `stagedjoint.npz`.

Found by Luka from a conservation argument — even mass over 11 buckets plus a
connected walk cannot give a zero cut — not from a larger sample.

**Second, independent error.** The merge joined the connected pair with the
*lowest combined mass*, which is the pair with the *weakest* transition: the
least refresh saved per bucket spent. Material 2,916 -> 128 removed **13%** of
the cut. Joining across the heaviest edge instead removes **62%**.

**Decision.** Both fixed in `nnue/attic/merge2.py`. Rule: least-visited live cell
joins the neighbour it transitions to most; never-visited cells are held out
and folded in afterwards. Corrected result (n = 173,216 positions, per ply,
both accumulators): the axis ranking **reverses** — material is about half the
price of kings at every matched effective count (~12 effective: 2.32% vs 4.98%;
~24 effective: 3.56% vs 7.98%). The old "kings quotient beautifully, material
does not" was an artefact of both bugs pointing the same way.

Best joint: **K32 x M64 -> 256**, effective 184.6 at **7.79%/ply**, against the
best hand rule (material 24^2, ~62 effective at 8.00%/ply) — three times the
effective buckets at the same price. Training it (`--experiment merged`) is the
only thing that settles whether that is worth anything; a proxy metric winning
is not a result.

Depth: `library/009-eval-architecture.md`, sections "The price was half" and
"Luka's merge, rebuilt on an honest price".
