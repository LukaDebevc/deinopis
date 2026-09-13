## Verification that the refactor is honest

- `SideQuadratic(tied)` is **output-identical** to the deployed `Bucketed(none)`.
- The untied `all` form reproduces `tied` exactly under the coefficient tie
  `(R, R, 2R)` — max diff 1.28e-09.
- `PerspSplit(W,0)` == `SideQuadratic(W, tied)`; `PerspSplit(0,r)` ==
  `Perspective(r)`.
- `tied r=512` = 0.021085 against LEDGER 023's `base r=512 x1` = 0.021095.
- Every extra feature family is cross-checked against an independent
  FEN-derived implementation. ALL PASS.
- After adding `chess moves`: `cargo test --release` 6/6 pass, `chess
  perft-suite` **ALL PASS**, bench 334,110 nodes.
- After adding `ThreadData::movemix`: 33 tests pass, `chess perft-suite`
  **ALL PASS**, bench still **334,110 nodes** — counters cannot move the
  fingerprint, and it did not move.
