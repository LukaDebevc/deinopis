#!/bin/bash
# Rank a batch of WDL nets against each other. First net is the reference;
# every other one plays it head to head, at EQUAL TIME.
#
# Equal time is the only condition, on purpose. Equal nodes would measure eval
# quality, which the val loss already ranks; the shipping question is whether
# a net is worth what it costs on the clock. A net that sees better per node
# but searches a bigger tree can still lose on time (m2-a2 vs m1-a1,
# 2026-09-08: +23.5 at equal nodes, -12.2 at equal time) -- and time is what
# we ship. The equal-nodes condition was removed for spending games on a
# question nobody acts on.
#
# Same binary on both sides, frozen as a copy: `chess match` runs both arms as
# subprocesses and a `cargo build` mid-match would silently swap one of them
# (see the match-harness notes). The ONLY difference between arms is --wdl.
#
# Usage:
#   tools/netcmp.sh [opts] ref.{pt,nnue} chal.{pt,nnue} ...
#
#   --speed-only     just the cost table (seconds); no games
#   --games N        per pairing                (default 400)
#   --tc SPEC        time control               (default 10+0.1)
#   --concurrency K  parallel games             (default 4 of 12 threads)
#
# .pt arguments are exported to int8 (v9) net files first. int8 is free:
# -0.00002 val CE across all five per-eval layers (LEDGER 053).
set -u
cd "$(dirname "$0")/.."

GAMES=400; TC=10+0.1; CONC=4
SPEED_ONLY=0
NETS=()
while (( $# )); do
  case "$1" in
    --speed-only)   SPEED_ONLY=1; shift ;;
    --games)        GAMES=$2; shift 2 ;;
    --tc)           TC=$2; shift 2 ;;
    --concurrency)  CONC=$2; shift 2 ;;
    -*)             echo "unknown option $1"; exit 2 ;;
    *)              NETS+=("$1"); shift ;;
  esac
done
(( ${#NETS[@]} >= 1 )) || { sed -n '2,30p' "$0"; exit 2; }

OUT=nnue/runs/netcmp-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
echo "netcmp -> $OUT"

# Freeze the binary. Everything below runs this copy, never ./target.
cargo build --release 2>&1 | grep -E "^error" && exit 1
BIN="$OUT/chess"
cp target/release/chess "$BIN"
git rev-parse --short HEAD > "$OUT/commit.txt" 2>/dev/null

# .pt -> .nnue, int8.
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

# --- cost table ------------------------------------------------------------
# A different net searches a different tree, so nps across two nets is not a
# controlled comparison of kernel speed -- but it IS the honest cost number,
# because the tree is part of what the net costs you.
{
  printf '%-24s %10s %12s %10s %s\n' net nodes nps rebuild size
  for i in "${!FILES[@]}"; do
    o=$("$BIN" bench --wdl "${FILES[$i]}" 2>&1)
    nd=$(grep -oP '^\d+(?= nodes)' <<<"$o" | tail -1)
    np=$(grep -oP '\d+(?= nps)' <<<"$o" | tail -1)
    rb=$(grep -oP '(?<=rebuilds \()[0-9.]+(?=%\))' <<<"$o" | tail -1)
    printf '%-24s %10s %12s %9s%% %s\n' "${NAMES[$i]}" "${nd:-?}" "${np:-?}" \
           "${rb:-?}" "$(stat -c %s "${FILES[$i]}")"
  done
} | tee "$OUT/speed.txt"
(( SPEED_ONLY )) && exit 0

# --- matches ---------------------------------------------------------------
REF="${FILES[0]}"; REFN="${NAMES[0]}"
run() {  # name tcspec tag
  local name=$1 tc=$2 tag=$3 i=$4
  echo
  echo "=== $name vs $REFN   $tag ($tc, $GAMES games) ==="
  "$BIN" match \
    --engine   "$BIN --wdl ${FILES[$i]}" \
    --opponent "$BIN --wdl $REF" \
    --name-a "$name" --name-b "$REFN" \
    --games "$GAMES" --tc "$tc" --concurrency "$CONC" --hash 32 \
    --pgn "$OUT/$name-vs-$REFN-$tag.pgn" 2>&1 | tee "$OUT/$name-$tag.log" | tail -8
}
for i in "${!FILES[@]}"; do
  (( i == 0 )) && continue
  run "${NAMES[$i]}" "$TC" eqtime "$i"
done
echo; echo "all results under $OUT"
