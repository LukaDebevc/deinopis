#!/bin/bash
# Play every net against every other net, once, at one time control, and
# turn the crosstable into one rating per net.
#
# Why this exists next to netcmp.sh: netcmp is a STAR. Every challenger plays
# one reference and nothing else, so the answer it gives is "better than the
# incumbent, yes or no". That is the shipping question, and for the shipping
# question a star spends every game on it. But a star cannot rank the
# challengers against each other -- gateh-C vs gateb8-D was only ever inferred
# through their common opponent, and STATE.md has been carrying that caveat
# since LEDGER 052. A round robin measures those pairings directly.
#
# The cost of that is real: 5 nets is 10 pairings against a star's 4, so at a
# fixed game budget each pairing gets less than half the games. Use netcmp when
# you want to know whether to ship one net; use this when you want the order.
#
# `chess match` is two engines and `chess elo` cannot combine a multi-opponent
# result (STATE.md), so the pairings are a loop here and the aggregation is
# tools/rr_rate.py. The W/D/L come from the match runner's own summary line and
# never from the PGN: both arms report the same UCI name, so the PGN path
# scores White instead of the engine (STATE.md).
#
# Usage:
#   tools/roundrobin.sh [opts] a.{pt,nnue} b.{pt,nnue} ...
#
#   --games N        per pairing            (default 300)
#   --tc SPEC        time control           (default 10+0.1)
#   --concurrency K  parallel games         (default 4 of 12 threads)
#   --anchor NAME    rating zero point      (default: the first net)
set -u
cd "$(dirname "$0")/.."

GAMES=300; TC=10+0.1; CONC=4; ANCHOR=""
NETS=()
while (( $# )); do
  case "$1" in
    --games)       GAMES=$2; shift 2 ;;
    --tc)          TC=$2; shift 2 ;;
    --concurrency) CONC=$2; shift 2 ;;
    --anchor)      ANCHOR=$2; shift 2 ;;
    -*)            echo "unknown option $1"; exit 2 ;;
    *)             NETS+=("$1"); shift ;;
  esac
done
(( ${#NETS[@]} >= 2 )) || { sed -n '2,33p' "$0"; exit 2; }
SPEC="$TC"

OUT=nnue/runs/rr-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
echo "round robin -> $OUT"

# Freeze the binary. Everything below runs this copy, never ./target: a
# `cargo build` mid-match silently swaps one arm (match-harness notes).
cargo build --release 2>&1 | grep -E "^error" && exit 1
BIN="$OUT/chess"
cp target/release/chess "$BIN"
git rev-parse --short HEAD > "$OUT/commit.txt" 2>/dev/null
git status --porcelain > "$OUT/dirty.txt" 2>/dev/null

# .pt -> .nnue, int8. Free accuracy: -0.00002 val CE (LEDGER 053).
FILES=(); NAMES=()
for n in "${NETS[@]}"; do
  base=$(basename "$n"); base=${base%.pt}; base=${base%.nnue}
  if [[ "$n" == *.pt ]]; then
    f="$OUT/$base.nnue"
    echo "export $n -> $f"
    python3 nnue/export_wdl.py --ckpt "$n" --out "$f" --int8 >> "$OUT/export.log" 2>&1 \
      || { echo "  EXPORT FAILED, see $OUT/export.log"; exit 1; }
  else
    f="$n"
  fi
  FILES+=("$f"); NAMES+=("$base")
done
[ -z "$ANCHOR" ] && ANCHOR="${NAMES[0]}"

# --- cost table ------------------------------------------------------------
# THREE runs per net, not one. A single `bench` right after `cargo build` reads
# several percent low and the first net in the loop reads worst -- that is how
# gate-A got written down as 653k nps when it is 702k. nps is a wall-clock
# metric; one sample of it is not a measurement.
{
  printf '%-24s %10s %10s %10s %10s %10s\n' net nodes r1 r2 r3 mean
  for i in "${!FILES[@]}"; do
    a=(); nd=""
    for r in 1 2 3; do
      o=$("$BIN" bench --wdl "${FILES[$i]}" 2>&1 | grep -oP '^\d+ nodes \d+ nps' | tail -1)
      nd=${o%% *}; a+=("$(grep -oP '\d+(?= nps)' <<<"$o")")
    done
    printf '%-24s %10s %10s %10s %10s %10s\n' "${NAMES[$i]}" "$nd" \
           "${a[0]}" "${a[1]}" "${a[2]}" "$(( (a[0]+a[1]+a[2])/3 ))"
  done
} | tee "$OUT/speed.txt"

# --- pairings --------------------------------------------------------------
N=${#FILES[@]}
TOTAL=$(( N * (N - 1) / 2 ))
echo
echo "$TOTAL pairings x $GAMES games at $SPEC, concurrency $CONC"
: > "$OUT/results.tsv"
k=0
for ((i = 0; i < N; i++)); do
  for ((j = i + 1; j < N; j++)); do
    k=$((k + 1))
    a="${NAMES[$i]}"; b="${NAMES[$j]}"
    echo
    echo "=== [$k/$TOTAL] $a vs $b   ($SPEC, $GAMES games)  $(date +%T) ==="
    log="$OUT/$a-vs-$b.log"
    "$BIN" match \
      --engine   "$BIN --wdl ${FILES[$i]}" \
      --opponent "$BIN --wdl ${FILES[$j]}" \
      --name-a "$a" --name-b "$b" \
      --games "$GAMES" --tc "$SPEC" --concurrency "$CONC" --hash 32 \
      --pgn "$OUT/$a-vs-$b.pgn" 2>&1 | tee "$log" | tail -3
    # Last summary line wins; it is the only complete one.
    s=$(grep -oP '^\d+ games: \+\d+ =\d+ -\d+' "$log" | tail -1)
    if [ -z "$s" ]; then
      echo "  NO RESULT PARSED from $log"; continue
    fi
    ng=$(grep -oP '^\d+'      <<<"$s")
    w=$(grep -oP '(?<=\+)\d+' <<<"$s")
    d=$(grep -oP '(?<= =)\d+' <<<"$s")
    l=$(grep -oP '(?<= -)\d+' <<<"$s")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$a" "$b" "$w" "$d" "$l" "$ng" >> "$OUT/results.tsv"
    echo "  -> $a +$w =$d -$l  ($ng games)"
  done
done

# --- ratings ---------------------------------------------------------------
echo
python3 tools/rr_rate.py --anchor "$ANCHOR" "$OUT/results.tsv" | tee "$OUT/ratings.txt"
echo
echo "all results under $OUT"
