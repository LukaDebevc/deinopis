# Measurement tools for `library/004`

Drop into `examples/` and `cargo build --release --example <name>`. Run from
the repo root; they read `.ladder/*.pgn`.

| tool | what it measures |
|---|---|
| `gapstat.rs` | search-error decay vs nodes (gamma, EBF) and the true-gap regression |
| `features.rs` | r2 of four candidate gap predictors, plus a first pass at sigma |
| `sigma.rs` | sigma estimators binned by quintile; dumps `sigma.tsv` for control analysis |

**They need one engine change:** `Searcher::quiescence` must be `pub`. It is
private in `src/search.rs`. Nothing else is modified.

`sigma.tsv` columns: `qs_1k  st_qs  spread  nmoves  npm  dev`. The nested
control analysis in 004 (binning within material quartiles) was done on that
file; redo it there rather than trusting the uncontrolled quintiles, which
point the wrong way for sibling spread.

These were run on a 12-core box **while no match was in flight**. They use
`--threads 6` by default and will perturb a running time-controlled SPRT.
Check `ps` before launching.
