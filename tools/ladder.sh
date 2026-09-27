#!/usr/bin/env bash
#
# The Elo anchor.
#
# A match only ever measures a *difference*. To turn that into a rating you
# need opponents whose rating is already known on some published scale, so this
# script builds a ladder of open-source engines that CCRL has rated, plays a
# gauntlet against them, and combines the results into one number.
#
# Two things are deliberate:
#
#   * **Exact rated versions.** Each engine is checked out at the precise tag
#     CCRL tested. Building master would give a binary with no published
#     rating, which is the same as having no anchor at all.
#   * **Ratings are fetched, not typed.** The numbers come from the live CCRL
#     Blitz list and are matched by the exact engine name string. A rating
#     typed from memory is a silent systematic error in every result that ever
#     cites it.
#
# What this cannot remove: CCRL rates its engines with a 12-move book and
# 6-piece tablebases at 2'+1" on an i7-4770K, and we do none of those things.
# Hardware cancels (both engines run here, on the same box), but the pool and
# the conditions do not, so the absolute number carries a systematic offset of
# order tens of Elo on top of the statistical interval. See
# library/002-measuring-strength.md.
#
# Usage:
#   tools/ladder.sh build              clone at the rated tags, build, verify
#   tools/ladder.sh ratings            (re)fetch the CCRL list
#   tools/ladder.sh list               what is installed and what it is rated
#   tools/ladder.sh run [--games N] [--tc TC] [--concurrency K] [--only k1,k2]
#
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LADDER="$ROOT/.ladder"
BIN="${CHESS_BIN:-$ROOT/target/release/chess}"

# The engine picks its eval silently: $CHESS_WDL, else `quad.nnue` beside the
# binary, else PeSTO -- with only an `info string` to say which. A gauntlet run
# without $CHESS_WDL therefore measures a stale net and reports it as a rating,
# and on 2026-09-21 one did: cp-0007 came back 3047 instead of ~3350 because it
# played an August `quad.nnue`. LEDGER 071/076/088 were all run with $CHESS_WDL
# exported by hand, which is a convention no file enforced.
#
# So resolve it here, the same way tools/checkpoint.sh does: from the one place
# that names the deployed net, in git, hash-checked.
if [ -z "${CHESS_WDL:-}" ]; then
  PIN="$ROOT/publish/net.sha256"
  if [ -f "$PIN" ]; then
    read -r PIN_SHA PIN_PATH < "$PIN"
    if [ -f "$ROOT/$PIN_PATH" ] \
       && [ "$(sha256sum "$ROOT/$PIN_PATH" | cut -d' ' -f1)" = "$PIN_SHA" ]; then
      export CHESS_WDL="$ROOT/$PIN_PATH"
    else
      echo "publish/net.sha256 names a net that is missing or has the wrong sha" >&2; exit 1
    fi
  else
    echo "no \$CHESS_WDL and no publish/net.sha256 -- refusing to gauntlet on an unnamed eval" >&2; exit 1
  fi
fi
CCRL_URL="https://computerchess.org.uk/ccrl/404/rating_list_all.html"

# key | repo | tag | exact CCRL name | build recipe | binary produced
ENGINES=(
  "bbc|maksimKorzh/BBC|1.1|BBC 1.1 64-bit|bbc|bbc-1.1"
  "goldfish|bsamseth/Goldfish|v2.1.1|Goldfish 2.1.1 64-bit|cargo-goldfish|target/release/goldfish"
  "cinnamon|gekomad/Cinnamon|v2.4|Cinnamon 2.4 64-bit|cinnamon|src/cinnamon"
  "tantabus|analog-hors/tantabus|v2.0.0|Tantabus 2.0.0 64-bit|cargo|target/release/tantabus-uci"
  "blunder|deanmchris/blunder|v8.5.5|Blunder 8.5.5 64-bit|go|blunder-8.5.5"
  "byteknight|ptsouchlos/byte-knight|v4.0.0|byte-knight 4.0.0 64-bit|cargo-byteknight|target/release/byte-knight"
  "4ku|kz04px/4ku|v5.1|4ku 5.1 64-bit|make|4ku"
  "inanis|Tearth/Inanis|v1.6.0|Inanis 1.6.0 64-bit|cargo|target/release/inanis"
  "simbelmyne|sroelants/simbelmyne|v1.10.0|Simbelmyne 1.10.0 64-bit|cargo|target/release/simbelmyne"
  # Added 2026-09-21. Every anchor above was BELOW us, so the rating was an
  # extrapolation off the top of the ladder -- and the implied rating rises with
  # anchor strength (slope +0.34 on cp-0007), so a ladder of weak anchors
  # systematically understates. These three bracket us from above. LEDGER 115.
  "frozenight|MinusKelvin/frozenight|v6.0.0|Frozenight 6.0.0 64-bit|cargo|target/release/frozenight-uci"
  "stash|mhouppin/stash-bot|v37.0|Stash 37.0 64-bit|stash|src/stash"
  "marvin|bmdanielsson/marvin-chess|v6.3.0|Marvin 6.3.0 64-bit|make|marvin"
)

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*"; exit 1; }

# ---------------------------------------------------------------- build
build_one() {
  local key="$1" repo="$2" tag="$3" recipe="$4" out="$5"
  local dir="$LADDER/src/$key"
  if [ ! -d "$dir" ]; then
    echo "cloning $repo @ $tag"
    git clone --quiet --depth 1 --branch "$tag" --recurse-submodules \
      "https://github.com/$repo.git" "$dir" || { warn "clone failed: $repo"; return 1; }
  fi
  # Opponents are built for this machine, exactly as we build ourselves. A
  # baseline-x86-64 opponent would be slower than the binary CCRL rated, and
  # that difference would show up as strength we do not have.
  (
    cd "$dir"
    case "$recipe" in
      cargo)          RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet ;;
      cargo-goldfish) RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet --bin goldfish ;;
      cargo-byteknight) RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet --bin byte-knight ;;
      make)           make -j4 >/dev/null ;;
      go)             GOARCH=amd64 GOAMD64=v3 go build -o "$out" blunder/main.go ;;
      bbc)            gcc -O3 -march=native -o "$out" src/bbc_1.1.c -lm ;;
      cinnamon)       ( cd src && make -j4 cinnamon64-modern-AMD >/dev/null 2>&1 ) ;;
      stash)          ( cd src && make -j4 >/dev/null ) ;;
      *)              echo "no recipe $recipe"; exit 1 ;;
    esac
  ) || { warn "build failed: $key"; return 1; }

  [ -f "$dir/$out" ] || { warn "$key: expected binary $out not produced"; return 1; }
  cp "$dir/$out" "$LADDER/bin/$key"
  # A binary that does not answer `uci` is not an opponent, it is a hang.
  local id
  id="$(printf 'uci\nquit\n' | timeout 15 "$LADDER/bin/$key" 2>/dev/null | grep -m1 '^id name ' || true)"
  [ -n "$id" ] || { warn "$key: no UCI handshake"; return 1; }
  echo "  ok  ${id#id name }"
}

cmd_build() {
  mkdir -p "$LADDER/src" "$LADDER/bin"
  local ok=0 failed=()
  for e in "${ENGINES[@]}"; do
    IFS='|' read -r key repo tag ccrl recipe out <<< "$e"
    say "$key  ($ccrl)"
    if build_one "$key" "$repo" "$tag" "$recipe" "$out"; then
      ok=$((ok+1))
    else
      failed+=("$key")
    fi
  done
  echo
  echo "$ok/${#ENGINES[@]} opponents built."
  [ ${#failed[@]} -eq 0 ] || warn "failed: ${failed[*]} — the gauntlet will simply skip these."
  cmd_ratings
}

# ---------------------------------------------------------------- ratings
cmd_ratings() {
  mkdir -p "$LADDER"
  say "CCRL Blitz ratings"
  if curl -sL --max-time 120 -o "$LADDER/ccrl-blitz.html" "$CCRL_URL"; then
    echo "fetched $(date -u +%Y-%m-%d) from $CCRL_URL"
  else
    [ -f "$LADDER/ccrl-blitz.html" ] || die "cannot fetch the CCRL list and no cached copy exists"
    warn "fetch failed — using the cached list, which may be out of date"
  fi
  python3 - "$LADDER" "${ENGINES[@]}" <<'PY'
import re, html, sys, pathlib, datetime
ladder = pathlib.Path(sys.argv[1])
wanted = {}
for spec in sys.argv[2:]:
    key, repo, tag, ccrl, recipe, out = spec.split("|")
    wanted[ccrl] = key

text = (ladder / "ccrl-blitz.html").read_text(encoding="utf-8", errors="replace")
# The list header carries the date it was computed; record it, because a rating
# without a date is not reproducible.
m = re.search(r"Computed on ([A-Z][a-z]+ \d+, \d{4})", text)
listdate = m.group(1) if m else "unknown"

rows = {}
for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S):
    cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
             for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
    # rank | name | elo | +err | -err | score | avg opp | draws | games
    if len(cells) >= 5 and re.fullmatch(r"\d+", cells[0] or "") and re.fullmatch(r"\d{3,4}", cells[2] or ""):
        rows[cells[1]] = (int(cells[2]), cells[3], cells[4], cells[-1])

out = ["# key\tccrl_name\telo\terr\tgames\tlist_date"]
missing = []
for name, key in wanted.items():
    if name in rows:
        elo, plus, minus, games = rows[name]
        err = max(abs(int(plus.replace("+", ""))), abs(int(minus.replace("−", "").replace("-", ""))))
        out.append(f"{key}\t{name}\t{elo}\t{err}\t{games}\t{listdate}")
        print(f"  {key:<10} {elo:>5} ±{err:<3} ({games} games)  {name}")
    else:
        missing.append(name)
(ladder / "ratings.tsv").write_text("\n".join(out) + "\n")
if missing:
    print("  NOT FOUND in the list (no anchor for these):", ", ".join(missing))
print(f"\n  CCRL Blitz list computed {listdate}")
PY
}

cmd_list() {
  [ -f "$LADDER/ratings.tsv" ] || die "no ratings yet — run: tools/ladder.sh build"
  printf '%-12s %-32s %6s %5s   %s\n' key name elo err binary
  while IFS=$'\t' read -r key name elo err games date; do
    [ "${key:0:1}" = "#" ] && continue
    local mark="missing"
    [ -x "$LADDER/bin/$key" ] && mark="$LADDER/bin/$key"
    printf '%-12s %-32s %6s %5s   %s\n' "$key" "$name" "$elo" "$err" "$mark"
  done < "$LADDER/ratings.tsv"
}

# ---------------------------------------------------------------- gauntlet
cmd_run() {
  local games=200 tc="10+0.1" conc=5 only=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --games) games="$2"; shift 2 ;;
      --tc) tc="$2"; shift 2 ;;
      --concurrency|-c) conc="$2"; shift 2 ;;
      # An opponent 500 Elo below us scores 100% and tells us nothing except
      # how long the gauntlet took. --only skips them.
      --only) only=",$2,"; shift 2 ;;
      *) die "unknown option $1" ;;
    esac
  done
  [ -x "$BIN" ] || die "build the engine first: cargo build --release"
  [ -f "$LADDER/ratings.tsv" ] || die "no ladder — run: tools/ladder.sh build"

  local stamp results
  stamp="$(date +%Y%m%d-%H%M%S)"
  results="$LADDER/gauntlet-$stamp.tsv"
  # A rating whose eval nobody recorded is not reproducible. Print it and
  # store it, so a gauntlet on the wrong net is visible in its own output.
  say "eval $(basename "$CHESS_WDL")  sha $(sha256sum "$CHESS_WDL" | cut -c1-16)"
  {
    echo "# eval	$CHESS_WDL	$(sha256sum "$CHESS_WDL" | cut -c1-16)"
    echo "# opponent	ccrl_elo	ccrl_err	engine_errors	result_line"
  } > "$results"

  while IFS=$'\t' read -r key name elo err g date; do
    [ "${key:0:1}" = "#" ] && continue
    [ -x "$LADDER/bin/$key" ] || { warn "skipping $key (not built)"; continue; }
    [ -z "$only" ] || [[ "$only" == *",$key,"* ]] || continue
    say "vs $name  (CCRL $elo ±$err)"
    local line pgn errs
    pgn="$LADDER/gauntlet-$stamp-$key.pgn"
    line="$("$BIN" match --engine "$BIN" --opponent "$LADDER/bin/$key" \
              --games "$games" --tc "$tc" --concurrency "$conc" --quiet \
              --pgn "$pgn" \
            | grep '^RESULT' || true)"
    [ -n "$line" ] || { warn "no result against $key"; continue; }
    # A game the opponent lost to a protocol failure is a measurement of our
    # own parser, not of the opponent, and it always scores in our favour.
    # Count them from the PGN, which is the authoritative record, and carry
    # the count so the combine can refuse the anchor.
    errs="$(grep -c 'Termination "engine error' "$pgn" 2>/dev/null || true)"
    errs="${errs:-0}"
    echo "$line"
    [ "$errs" -eq 0 ] || warn "  $errs game(s) ended in an engine error — this anchor will be EXCLUDED"
    printf '%s\t%s\t%s\t%s\t%s\n' "$key" "$elo" "$err" "$errs" "$line" >> "$results"
  done < "$LADDER/ratings.tsv"

  say "combined rating estimate"
  # The combination lives in the engine (`chess elo --combine`, estimator in
  # sprt.rs, covered by cargo test) — not as a second implementation here.
  "$BIN" elo --combine "$results" --tc "$tc" --games "$games"
  echo
  echo "raw results: $results"
}

case "${1:-}" in
  build)   shift; cmd_build "$@" ;;
  ratings) shift; cmd_ratings "$@" ;;
  list)    shift; cmd_list "$@" ;;
  run)     shift; cmd_run "$@" ;;
  *) sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 1 ;;
esac
