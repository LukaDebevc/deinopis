# 017/00 · Progress: item status and who owns what

_Status of every map item, updated when an item changes state. The map
part files (`01`–`06`) are the specs — they don't repeat status. When an
item closes, mark it there too as `[done → LEDGER NNN]` per 01-protocol._

_Last updated: 2026-09-19._

## Resources right now

| resource | state |
|---|---|
| desk box (games) | **free** — SR-1 closed (104). TU-5 (pair 362) resumes once the ladder picks a TC |
| cluster CPU (games) | BUSY: TC-transfer ladder, 32 cores, ~10 h from 2026-09-19 12:47 (`<cluster scratch>`) |
| GPU (cluster) | idle — NN-1 not started |
| desk (code, probes) | free — the desk box is idle, so proxy-CPU work no longer steals game CPU |

## Phase 0 (in progress)

| item | status | evidence / note |
|---|---|---|
| HK-1 commit tree | done | `873f4b5` 089/090 entries, `e847188` dormant flags, `6a47fab` map + 091/092; bench 199091 at each |
| FX-1a persist-all SPRT | done → LEDGER 093 | 3000g: +5.7 [−1,+12], LLR 1.41, inconclusive at cap; rescore agrees (Δ=0); not shipped, stays dormant default-off |
| FX-1b halve / FX-1c killers-cleared | built, ON HOLD | 093: no reason to expect either to pass [0,5] alone; bundle candidate only |
| FX-3/4 gate SPRT | done → LEDGER 095 | H0, −4.6 [−11,+2], 2622g, LLR −2.98 (`~/chess-runs/20260915-fx34gate/`, chess-new 199115 vs chess-old 199091); not shipped, to be reverted |
| TU-1 params / TU-2 `--set` / TU-3 `spsa.py` | done | `d372758`; bench-exact at defaults; sign logic + resume smoke-tested live |
| MS-4 `rescore.sh` / HK-8 `verify.sh` | done | `e01db4e`; rescore agrees with live SPRT to 0.01 Elo; verify default fingerprint 199115 |
| HK-5 label inventory | done | `chess-data/tune/README.md`; all three label files stale (name `gate-3ep-q8`/`b986f5b`) → MS-6 still needed |
| HK-4 durable run dir | partial | dir exists, FX-1a frozen in it (binary, net, SHA256+bench); README written 2026-09-15; `tools/*.sh` defaults still point at `/tmp` habits — pending |
| HK-10 bench table | partial | `ENGINE.md` already carries both 199091 and 199115 rows; `OPTIMIZATIONS.md` agreement unchecked |
| FX-12 book probe | done → 096 | 34 distinct 2-move openings, heaviest 9.3% — and the repeats cost nothing: 600 games are 598 distinct games |
| MS-5 opening book | done → 096 | unbalanced book cuts draws 72%→58%, games to a decision unchanged (ratio 1.10 [0.71,1.71]); support kept on branch `uho-book`, default unchanged |
| MS-6 relabel (cp-0004 + m1-b1, lift to 4k) | done | `fishpack-4k-cp5.labels`: 4000×top5@400k from cp-0005 tree + m1-b1, 2142 s wall, log `~/chess-runs/20260916-ms6/label.log`; inventory in `chess-data/tune/README.md` |
| TU-4 `c_end` screen | done → LEDGER 098 | 16 params × ~8 steps vs default, fresh 4k @15k; SE ~0.85; c_end set (3 proxy-blind fallbacks); max_base/slope +side inert reproduces 065 |
| SR-2 singular (`se_min_depth=8`) | done → LEDGER 101 | 3000g: −2.5 [−9,+4], inconclusive at cap, rescore agrees; ran to cap pre-shutdown, no resume needed; not shipped, stays dormant |
| SR-1 futility skip-quiets (`fut_max_depth=3`) | done → LEDGER 104 | 2×2 hash cells: no interaction (+3.1 ± 8.4); split refactor costs 0.33% nps, i.e. nothing. Pooled over 6000 unbiased games **+4.6 [−0.2, +9.4]** — on the [0,5] bound. Not shipped, not killed: **ship candidate parked for a bundle partner** |
| TX-1 TC-transfer ladder | RUNNING → LEDGER 105 | 5 arms × 4 rungs on the GPU cluster, 32 cores, from 09-19 12:47. Answers the "shape evidence" question 099 left open and picks the screening TC. R4 smoke: 65,000 games/h, A/A −0.0, zero forfeits, median depth 7.0 vs 14.0 |
| TU-5 campaign A | PARKED 2026-09-17 (pair 362) | spsa.csv kept — resume: `tools/spsa.py ~/chess-runs/20260916-tu5/campaign-a.params --pairs 20000 --tc 2+0.02 --concurrency 3 --bin ./target/release/chess --wdl nnue/runs/b1-20260909/m1-b1.nnue --dir ~/chess-runs/20260916-tu5/campaign-a --seed 16` |
| SP-1 speedup study | done → LEDGER 099 | draw rule 12cp/12 shipped (bench-exact); resign-500 rejected (bias); TC ladder: campaign TC = 2+0.02 |

## Blocked on the box, in launch order

1. ~~FX-1a verdict~~ done (093, inconclusive, not shipped).
2. FX-3/4 gate verdict: done → LEDGER 095 (H0, −4.6 [−11, +2], 2622g).
3. ~~BUNDLE~~ done: **H1, +11.2 [+4, +19], 2318g** (094, on the GPU cluster). Gate
   done: **H1, +7.5 [+2, +13], 3776g** (097, `cp-0005` pushed, bench 219718).
4. ~~TU-4 `c_end` from proxy~~ done (098). TU-5 campaign A PARKED at pair 362.
5. RV-1/RV-2 — runnable with `--persist-hist=1` on both arms (093).

The list above is desk-box history. The desk box is now free and the queue is
the cluster's; see `STATE.md` "Next, in order".

## First-ten checklist (map §"First ten, in order")

1. HK-1 — done. 2. FX-1 — done → 093 (inconclusive); RV-1/RV-2 still to run.
3. FX-3+FX-4 gate — done → 095 (H0, reverted). 4. MS-6 — done.
5. TU-1..3 — done. 6. TU-5 — PARKED at pair 362, waiting on the ladder's TC.
7. SR-1 futility — done → 102/103/**104** (+4.6 pooled, parked for a bundle).
8. SR-2 singular — done → 101 (inconclusive, not shipped).
9. LC-1 — not started. 10. NN-1 — not started.

Off-list and running: **TX-1**, the TC-transfer ladder (LEDGER 105).

## Notes worth keeping

- FX-1a's frozen binary predates FX-3/4 (bench 199091 in its SHA256), so the
  persist-all verdict is clean of the TT-size change — and it doubles as the
  gate's old arm (sha-verified copy in the fx34gate dir). Do not "update" it.
- FX-1a closed inconclusive at cap (093): +5.7 [−1,+12], LLR 1.41, rescore
  Δ=0. No silent extends — P4 hand-kill needs a written reason, and the cap
  verdict stands as recorded.
- `pgrep -fa 'chess match'` before any build; `CARGO_TARGET_DIR=/tmp/chess-target`
  while the box is loaded.
- Cluster binaries build with `-C target-cpu=x86-64-v3`, never `native`: the
  desk is Zen 3 and the GPU cluster is Zen 2. Verify the bench fingerprint of every
  cluster build against the desk before it plays a game.
