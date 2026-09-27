#!/usr/bin/env bash
#
# The checkpoint gate.
#
# A version is worth keeping when it is (a) correct and (b) demonstrably not
# weaker than the last kept version. This script is the only thing allowed to
# decide that, because "it looks better" and "the bench went down" are both
# things that have shipped regressions in every engine ever written.
#
#   1. build                  release, with the repo's own rustflags
#   2. cargo test --release    perft to depth 4, zobrist, search, harness
#   3. chess perft-suite       full published depths            (skip: --fast)
#   4. chess bench             node-count fingerprint, recorded in the tag
#   5. SPRT vs the last checkpoint binary, both sides run as subprocesses
#   6. only if the SPRT accepts H1: commit, tag, append CHECKPOINTS.md, push
#
# Step 5 runs both engines through the *same* external UCI path on purpose. The
# in-process player and the subprocess player are different code paths, and any
# systematic difference between them would otherwise be silently added to every
# checkpoint decision.
#
# Usage:
#   tools/checkpoint.sh -m "singular extensions"      full gate
#   tools/checkpoint.sh -m "..." --fast               skip the full perft suite
#   tools/checkpoint.sh -m "..." --baseline           first checkpoint, no SPRT
#   tools/checkpoint.sh --restore-baseline            rebuild the reference binary
#   tools/checkpoint.sh -m "..." --net <file>         a different eval
#
# Both arms are given the SAME eval, by name. See the NET comment below: a gate
# that lets each side resolve its own net is a gate that compares two different
# engines and passes every time.
#
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BASELINE="$ROOT/.checkpoints/best"
BIN="$ROOT/target/release/chess"

MSG=""
# The eval the checkpoint plays with, named EXPLICITLY.
#
# Without this the gate silently compares two different engines. `qeval::net()`
# resolves to `quad.nnue` *beside the binary* and falls back to PeSTO when it
# is not there -- and `.checkpoints/best` is a bare copy in a bare directory,
# while `target/release/chess` sits next to whatever the last export wrote. So
# the challenger would play its net and the baseline would play PeSTO, every
# gate would pass, and none of them would mean anything. This is the same trap
# that cost `tools/wdlmatch.sh` a match against the wrong opponent.
#
# Nets are gitignored (819 MB of nnue/ is regenerable), so a tag CANNOT rebuild
# its own eval. The sha256 goes in the tag message instead: a future gate that
# quietly swapped the net is then visible rather than silent.
NET="${CHESS_NET:-}"
FAST=0
FIRST=0
ELO0=0
ELO1=10
TC="8+0.08"
CONCURRENCY=5
MAXGAMES=4000
DRYRUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    -m|--message)     MSG="$2"; shift 2 ;;
    --fast)           FAST=1; shift ;;
    --baseline)       FIRST=1; shift ;;
    --sprt)           ELO0="${2%,*}"; ELO1="${2#*,}"; shift 2 ;;
    --tc)             TC="$2"; shift 2 ;;
    --concurrency|-c) CONCURRENCY="$2"; shift 2 ;;
    --games)          MAXGAMES="$2"; shift 2 ;;
    --dry-run)        DRYRUN=1; shift ;;
    --net)            NET="$2"; shift 2 ;;
    --restore-baseline) RESTORE=1; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mGATE FAILED: %s\033[0m\n' "$*"; exit 1; }

# ---------------------------------------------------------------- baseline
# The reference binary is not in git (binaries do not belong there); it is
# rebuilt from the most recent checkpoint tag when missing.
restore_baseline() {
  local tag
  tag="$(git tag -l 'cp-*' | sort -V | tail -1)"
  [ -n "$tag" ] || fail "no checkpoint tag to rebuild a baseline from — run with --baseline once"
  say "rebuilding baseline from $tag"
  local wt="$ROOT/.checkpoints/wt"
  rm -rf "$wt"
  git worktree add --detach --quiet "$wt" "$tag"
  ( cd "$wt" && cargo build --release --quiet )
  mkdir -p "$ROOT/.checkpoints"
  cp "$wt/target/release/chess" "$BASELINE"
  git worktree remove --force "$wt"
  echo "baseline := $tag"
}

if [ "${RESTORE:-0}" = 1 ]; then restore_baseline; exit 0; fi
[ -n "$MSG" ] || fail "a checkpoint needs a message: -m \"what changed\""

# Unset means: gate with the net we deploy. `publish/net.sha256` is the one
# place that names it, it is in git, and tools/publish.sh already hash-checks
# it. A hardcoded default here goes stale in silence -- this one did, and
# cp-0006 was tagged naming an eval 58 Elo worse than the shipped one.
if [ -z "$NET" ]; then
  PIN="$ROOT/publish/net.sha256"
  [ -f "$PIN" ] || fail "no --net, no \$CHESS_NET, and no $PIN to fall back on."
  read -r PIN_SHA PIN_PATH < "$PIN"
  NET="$ROOT/$PIN_PATH"
  [ -f "$NET" ] || fail "the net pinned in publish/net.sha256 is not on this machine: $NET"
  [ "$(sha256sum "$NET" | cut -d" " -f1)" = "$PIN_SHA" ] \
    || fail "$NET does not match the sha pinned in publish/net.sha256."
fi

[ -f "$NET" ] || fail "no eval net at $NET — pass --net <file>, or CHESS_NET=. \
A gate with no named net is a gate that measures PeSTO."
NET_SHA="$(sha256sum "$NET" | cut -c1-16)"
EVAL_ARG="--wdl $NET"
echo "eval: $(basename "$NET")  sha256 $NET_SHA"

# ---------------------------------------------------------------- 1-4: correctness
say "build"
cargo build --release --quiet

say "cargo test --release"
cargo test --release --quiet 2>&1 | tail -20 || fail "tests"

if [ "$FAST" = 0 ]; then
  say "perft suite (full published depths)"
  "$BIN" perft-suite | tail -8 || fail "perft"
else
  echo "(skipping the full perft suite — --fast)"
fi

say "bench"
# `tail -1` was wrong: with a WDL net the bench prints accumulator stats and a
# move-type breakdown AFTER the node count, so the fingerprint recorded in the
# tag was "q promo 1192 0.31%". Match the line, do not count from the end.
BENCH_LINE="$("$BIN" bench $EVAL_ARG 2>/dev/null | grep -E '^[0-9]+ nodes [0-9]+ nps$' | tail -1)"
[ -n "$BENCH_LINE" ] || fail "could not read a node count out of \`chess bench\`"
BENCH_NODES="$(echo "$BENCH_LINE" | awk '{print $1}')"
BENCH_NPS="$(echo "$BENCH_LINE" | awk '{print $3}')"
echo "$BENCH_LINE"

# ---------------------------------------------------------------- 5: strength
ELO_SUMMARY="first checkpoint — no reference to compare against"
if [ "$FIRST" = 0 ]; then
  [ -x "$BASELINE" ] || restore_baseline
  say "SPRT vs baseline   H0: +${ELO0}   H1: +${ELO1}   tc ${TC}"
  set +e
  "$BIN" match \
      --engine "$BIN $EVAL_ARG" --opponent "$BASELINE $EVAL_ARG" \
      --name-a candidate --name-b baseline \
      --tc "$TC" --sprt "${ELO0},${ELO1}" \
      --games "$MAXGAMES" --concurrency "$CONCURRENCY" \
      --pgn "$ROOT/.checkpoints/last.pgn" \
      | tee "$ROOT/.checkpoints/last-match.txt"
  RC=${PIPESTATUS[0]}
  set -e
  ELO_SUMMARY="$(grep -E '^[0-9]+ games' "$ROOT/.checkpoints/last-match.txt" | tail -1)"
  case "$RC" in
    0) echo "SPRT accepted H1." ;;
    1) fail "SPRT accepted H0 — this build is not stronger than the baseline. Nothing committed, nothing pushed." ;;
    2) fail "SPRT inconclusive after $MAXGAMES games. Either the change is too small to resolve at this budget, or it is neutral. Nothing committed, nothing pushed." ;;
    *) fail "match runner error (exit $RC)" ;;
  esac
fi

# ---------------------------------------------------------------- 6: commit, tag, push
NEXT="$(printf 'cp-%04d' "$(( $(git tag -l 'cp-*' | wc -l) + 1 ))")"
say "checkpoint $NEXT"

if [ "$DRYRUN" = 1 ]; then
  echo "--dry-run: would tag $NEXT and push. Nothing written."
  exit 0
fi

DATE="$(date +%Y-%m-%d)"
# Before the first write, not after it: `pending` and the baseline binary both
# live here, and on a fresh clone the directory does not exist yet. Only the
# --baseline path ever hits that, which is why it survived this long.
mkdir -p "$ROOT/.checkpoints"
{
  echo "| $NEXT | $DATE | $MSG | $BENCH_NODES | $ELO_SUMMARY |"
} >> "$ROOT/.checkpoints/pending"

python3 - "$ROOT/CHECKPOINTS.md" "$ROOT/.checkpoints/pending" <<'PY'
import sys, pathlib
doc, pending = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
row = pending.read_text().strip().splitlines()[-1]
text = doc.read_text()
text = text.rstrip() + "\n" + row + "\n"
doc.write_text(text)
pending.unlink()
PY

git add -A
git commit -q -m "$NEXT: $MSG

bench $BENCH_NODES nodes, $BENCH_NPS nps
eval $(basename "$NET") sha256 $NET_SHA
$ELO_SUMMARY"
git tag -a "$NEXT" -m "$MSG

bench: $BENCH_NODES nodes @ $BENCH_NPS nps
eval: $(basename "$NET") sha256 $NET_SHA
strength: $ELO_SUMMARY"

mkdir -p "$ROOT/.checkpoints"
cp "$BIN" "$BASELINE"

git push --quiet origin HEAD
git push --quiet origin "$NEXT"
printf '\n\033[32mCheckpoint %s committed, tagged and pushed.\033[0m\n' "$NEXT"
echo "  $ELO_SUMMARY"
echo "  baseline for the next gate := this build"
