**021 · 250M positions was data-limited, not under-trained** · 2026-08-27
Training the full-rank model for 40000 steps (2.6B positions, ~10 epochs) over
the 250M pool pushed the train/val gap from 0.7% at 8000 steps to **4% at 4000
steps of the long run** — the model was memorising, not learning.
**Decision:** extracted the *whole* T79 May file rather than the first 250M
records: **894M positions, 27 GB**. On that pool the same model at 16000 steps
runs train 0.021095 / val 0.021273, a **0.8% gap** — the overfitting is gone.
**Corrected — the train/val gap is the wrong criterion.** This entry originally
said to add months "only if the train/val gap says so". Luka's rule is
stricter and better: train **one epoch**, and buy *unique* positions rather
than extra passes. Even keeping every ply of one game is close to worthless,
because positions inside a game are highly correlated. The 40000-step run drew
2.62B samples from 894M unique positions — **~2.9 passes** — which is already
the wrong shape even though the gap stayed at 0.8%.
**Consequence for the data plan.** `nnue/extract --stride N` keeps one position
per N; `chess-data/fetch.sh` now threads `STRIDE` through. **Corrected:** the
counter is global over kept positions, not per game — it does not reset at a
game boundary. For thinning a pool that is consumed as loose positions this is
the same thing, but it makes `--stride` unusable whenever game structure
matters, because it punches holes in games. See 042 for `--game-stride`. At
stride 8 a month costs ~3.6 GB instead of 28.6, so the same 53 GB free holds
~14 months (~1.5B positions from 14x as many distinct games) instead of one.
Size the next run so `steps x batch <= unique positions`.

---
