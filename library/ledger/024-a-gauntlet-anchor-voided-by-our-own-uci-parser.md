**024 · A gauntlet anchor voided by our own UCI parser** · 2026-08-27
The Simbelmyne 1.10.0 anchor in the 894M gauntlet reported +47.0% score and an
implied 3217. Re-run clean, the same opponent implies **2983**. Including the
contaminated anchor put the combination at 3017; the correct six-anchor value
is **2971** (LEDGER 025).
It was a measurement of our parser.
**Cause:** Simbelmyne emits promotions with an **uppercase** piece letter
(`a7a8Q`). UCI specifies lowercase, so it is technically wrong, but tolerating
either case is standard and cutechess does. `MoveList::find_uci` compared
strings exactly, so every promotion by Simbelmyne was scored an illegal move
and forfeited **in our favour**: 28 games lost to `illegal move ...Q`, plus 31
to `write failed: Broken pipe` and 1 timeout -- **59 of 200 games, 30%**.
**Fix:** `find_uci` now compares with `eq_ignore_ascii_case`. Nothing else in a
UCI move is case-bearing, so this cannot make an illegal move look legal;
`tests/harness.rs::promotion_uci_is_case_insensitive` asserts both directions.
Bench **334110 nodes, unchanged**.
**Second fix, the one that matters more.** The match runner *did* print
"60 game(s) ended in an engine error - treat the result with suspicion", and
the ladder combined the anchor anyway. `tools/ladder.sh` now counts
`Termination "engine error` from the PGN, carries the count in the results
file, and **excludes** any anchor with a non-zero count, loudly. A warning
nobody's code reads is not a safeguard.
**The other five anchors had zero engine errors**, so the cluster below stands.

---
