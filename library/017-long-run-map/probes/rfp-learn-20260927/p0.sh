#!/bin/bash
# P0 of the uncertainty-head probe, 2026-09-27. cpd = cp-0009 + `nodedump`
# (bench 151742, tree unchanged), x86-64-v3. Samples pruning-decision nodes
# from self-play, then labels each with qsearch + a 64k-node search + the
# net's inner layers (chess nodelabel). Writes DONE / FAIL lines to p0.log.
set -u
D=<cluster scratch>; cd "$D" || exit 1
N=$D/m1-b1.nnue
say() { echo "$(date +%H:%M:%S) $*" >> p0.log; }
say "START pid $$"
mkdir -p dump shards
CHESS_NODEDUMP=$D/dump/n CHESS_NODEDUMP_EVERY=${EVERY:-2000} nice -n 19 ./cpd match \
  --engine "$D/cpd --wdl $N" --opponent "$D/cpd --wdl $N" --name-a a --name-b b \
  --tc 1+0.01 --concurrency 16 --games ${GAMES:-2000} --hash 64 --pgn $D/selfplay.pgn > selfplay.log 2>&1
say "SELFPLAY $(grep -o 'RESULT.*' selfplay.log | tail -1) files=$(ls dump | wc -l)"
# game id = the dumping process; one engine process plays one game side.
i=0; for f in dump/*.tsv; do i=$((i+1)); awk -v g=$i '{print $0"\t"g}' "$f"; done > nodes.tsv
say "NODES $(wc -l < nodes.tsv)"
split -n l/32 -d -a 2 nodes.tsv shards/s
pids=()
for f in shards/s??; do
  nice -n 19 ./cpd nodelabel "$f" "$f.lab" --wdl $N --nodes 65536 --hash 16 > /dev/null 2> "$f.err" & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
cat shards/s??.lab.tsv > lab.tsv; cat shards/s??.lab.f32 > lab.f32
say "LABELLED rows=$(wc -l < lab.tsv) f32_bytes=$(stat -c %s lab.f32) expect=$(( $(wc -l < lab.tsv) * 163 * 4 )) rc=$rc"
say "DONE"
