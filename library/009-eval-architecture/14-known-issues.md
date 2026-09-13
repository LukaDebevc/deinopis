## Known issues

- **`extras-buck` exited 134** after printing its full results table:
  `terminate called without an active exception`, a daemon-thread destructor,
  almost certainly `loader.py`'s block-reader thread not being joined. Every
  earlier run exited 0, so it is timing-dependent and the concurrent
  `kingcoarse` stream is the likely trigger. Results are complete and valid.
  **Not patched deliberately** — editing `loader.py` mid-queue would leave the
  remaining experiments running different loader code from the earlier ones.
  Fix after the queue drains.
- **`STATE.md` is stale**: it predates LEDGER 020–025, and its recorded bench
  fingerprint of 567,321 nodes does not match the working tree's 334,110 (the
  uncommitted eval work changed it, not the `chess moves` addition).
- **`gather.rs` has not been run.** It is the microbenchmark of one 512-wide
  i16 gather across table sizes x index dwell — Luka's "dummy testing that
  compute and retrieval time are similarish", and the test of LEDGER 023's
  cache *argument* that a slow-moving bucket index keeps a big table free.
  Deliberately deferred: the training reader thread pollutes L3 and would
  confound it. **Run when the GPU queue is idle.**
