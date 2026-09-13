## 042 — The extractor destroyed game structure; `--game-stride` and `--mark`

The learned-search-control study needs whole games (a latched feature is a
running set over a side's plies), and `all.data` cannot supply one. Two
separate causes, both in `nnue/extract`:

- **Filtered positions were dropped, not flagged.** `if !keep(&e) { continue; }`
  discards 46% of plies, and the 32-byte record carries no game id and no ply,
  so boundaries would have to be guessed from piece-count jumps — which merges
  games, the exact leak the study is trying to measure.
- **`--stride` strides on the wrong axis.** `kept % stride` is a *global*
  counter over kept positions (021 is corrected accordingly), so it thins plies
  *within* games rather than dropping whole games.

**The boundary signal is exact, not a heuristic.** `plyprobe` over a 200 MB
prefix of `fishpack32` (95.7M entries, 801,017 games): every within-game ply
delta is **+1, 94,857,841 of 94,857,841**, and every game starts at **ply 1,
801,017 of 801,017**. Zero exceptions, so `ply <= prev_ply` is a boundary.

**Where the metadata goes.** `bulletformat::ChessBoard.extra` is `pub`, zeroed
by `from_raw` and read nowhere in the crate — 3 free bytes per record.
`extra[0]` bit0 = passes the filter, bit1 = first ply of a game; `extra[1..3]` =
u16 ply. Nothing that consumes the old format is affected; a trainer wanting
only the filtered set masks on bit0.

**Result.** `--mark --game-stride 24` over the 5.58 GB `fishpack32`:
**2,555,358,353 entries → 891,669 games / 106,523,069 positions, 3.41 GB, 245 s
(10.4M entries/s)**. 66.4% of written plies pass the filter; **12.6% of the file
is `ply < 16`**, which the old path could not represent at all. That is **19x
the 46,840 games** all local lc0 data provides, and the pool the 9-arm
comparison was too noisy to resolve on.

**Decision:** keep both flags. `--stride` is unchanged for pools consumed as
loose positions; `--game-stride` + `--mark` is the setting whenever game
structure matters. A truncated trailing chunk now warns and stops instead of
panicking, so a byte-range prefix of a binpack is usable directly.

**Also:** the dataset builder stored the latched set explicitly, 80 wide, at
**270 B/position** — 26 GB at this scale. The set is a running *prefix* of the
order in which a side first visits states, so it is now stored as that order
plus a per-position length: **23.1 B/position, 11.7x smaller**, verified
element-for-element against the old representation including where the cap
binds. Features are not stored at all; the 32-byte record is the feature store
and decodes at 0.62M positions/s, which is cheaper than keeping a second copy.
