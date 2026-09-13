## 039 — Lazy SMP: 5.5x the nodes, 2.3x the search (`library/010`)

Six threads over one shared TT, no other synchronisation. `Shared`/`ThreadData`
already split correctly and the TT was already lockless with a Hyatt XOR
checksum, so this is a driver (`search::go_parallel`) and a diversification
knob, not a restructuring.

`chess smp --depth 16 --reps 2`, 24 searches, 256 MB hash, time-to-depth scored
only where every arm returned at the requested depth:

| threads | 1 | 2 | 4 | 6 | 12 |
|---|---|---|---|---|---|
| nps × | 1.00 | 2.06 | 4.11 | **5.52** | 6.70 |
| time-to-depth × | 1.00 | 1.96 | 1.98 | **2.32** | 2.20 |

Read the second row. nps scaling is nearly free and nearly meaningless — N
threads searching the same tree N times over would report a perfect N-fold nps
and be worth zero Elo. 12 threads (SMT) is worse than 6; do not exceed the
physical core count.

**Negative: the depth-skip table loses.** Helpers running a different schedule
of iterative-deepening depths (`smp_skip=1`, the classic Stockfish phase/size
table) measured **2.38x** against **2.70x** for no skipping at all, n=3 runs
each at 6 threads, and lower nps with it (5.3x vs 5.6x). Default is 0.

**Node-limited searches are forced to one thread.** `go nodes N` is the tuner's
unit of work and the method rests on it being exactly reproducible; under SMP
it would be N nodes on *some* thread with the others racing it. `bench` is
unaffected: 334,110 at depth 9, unchanged.

**No Elo.** Time-to-depth is a proxy. The honest test is 6 threads vs 1 at a
fixed TC with `--concurrency 1`, which costs the whole machine while it runs.

**ROADMAP #2 was wrong about why this matters.** It claimed "6x the SPRT
throughput — which is the real bottleneck". `chess match` already runs games in
parallel and sends `Threads 1` to every subprocess engine, so the cores are
already saturated by games. Lazy SMP multiplies Elo per game, not games per
hour. Corrected in ROADMAP.
