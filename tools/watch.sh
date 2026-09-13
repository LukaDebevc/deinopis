#!/usr/bin/env bash
#
# Live view of a running gauntlet.
#
# Reads the PGN files the match runner is writing and reports the standings so
# far. The statistics come from `chess elo`, which is the same estimator the
# match itself uses — a second implementation here would eventually disagree
# with the first and there would be no way to tell which was right.
#
# Usage:
#   tools/watch.sh              follow the newest gauntlet, refresh every 10s
#   tools/watch.sh -n 3         refresh every 3s
#   tools/watch.sh --once       print once and exit
#
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BIN="$ROOT/target/release/chess"

INTERVAL=10
ONCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    -n) INTERVAL="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

render() {
  local stamp
  stamp="$(ls -t "$ROOT"/.ladder/gauntlet-*-*.pgn 2>/dev/null | head -1 || true)"
  if [ -z "$stamp" ]; then echo "no gauntlet found — run tools/ladder.sh run"; return; fi
  stamp="$(basename "$stamp")"; stamp="${stamp#gauntlet-}"; stamp="${stamp%-*.pgn}"

  local running="stopped"
  pgrep -f "ladder.sh run" >/dev/null 2>&1 && running="running"

  printf '\033[1mgauntlet %s  (%s)\033[0m   %s\n\n' "$stamp" "$running" "$(date +%H:%M:%S)"
  printf '%-12s %6s %7s %8s %10s %12s  %s\n' opponent CCRL games score diff implied "95% interval"

  python3 - "$ROOT" "$stamp" "$BIN" <<'PY'
import subprocess, sys, pathlib, math, re
root, stamp, binary = sys.argv[1], sys.argv[2], sys.argv[3]
ladder = pathlib.Path(root) / ".ladder"

ratings = {}
rfile = ladder / "ratings.tsv"
if rfile.exists():
    for line in rfile.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        k, name, elo, err, games, date = line.split("\t")
        ratings[k] = (name, int(elo), int(err))

est, done = [], 0
order = list(ratings) or []
for pgn in sorted(ladder.glob(f"gauntlet-{stamp}-*.pgn")):
    key = pgn.name[len(f"gauntlet-{stamp}-"):-4]
    try:
        out = subprocess.run([binary, "elo", str(pgn)], capture_output=True, text=True, timeout=60).stdout
        line = next(l for l in out.splitlines() if l.startswith("RESULT"))
    except Exception:
        continue
    f = dict(kv.split("=") for kv in line.split()[1:])
    n, w, l, d = (int(f[k2]) for k2 in ("games", "w", "l", "d"))
    diff, lo, hi = (float(f[k2]) for k2 in ("elo", "lo", "hi"))
    done += n
    name, ccrl, cerr = ratings.get(key, (key, 0, 0))
    score = (w + d / 2) / n * 100
    if ccrl:
        sd = math.hypot(max(hi - diff, diff - lo) / 1.96, cerr / 1.96)
        rating = ccrl + diff
        est.append((rating, sd))
        interval = f"[{rating-1.96*sd:.0f}, {rating+1.96*sd:.0f}]"
        implied = f"{rating:.0f}"
    else:
        interval, implied = "", "?"
    print(f"{key:<12} {ccrl:>6} {n:>7} {score:>7.1f}% {diff:>+10.0f} {implied:>12}  {interval}")

if est:
    wsum = sum(1 / s**2 for _, s in est)
    mean = sum(r / s**2 for r, s in est) / wsum
    se = math.sqrt(1 / wsum)
    chi2 = sum(((r - mean) / s) ** 2 for r, s in est)
    dof = max(1, len(est) - 1)
    print(f"\n  so far: \033[1m{mean:.0f} ±{1.96*se:.0f}\033[0m on the CCRL Blitz scale"
          f"   ({done} games, {len(est)}/{len(ratings)} opponents, chi2/dof {chi2/dof:.1f})")
    if chi2 / dof > 2.5:
        print("  estimates disagree by more than their error bars — the single-number model is not fitting")
else:
    print(f"\n  {done} games, no completed opponent yet")
PY
}

if [ "$ONCE" = 1 ]; then render; exit 0; fi
while true; do
  clear
  render
  printf '\n  refreshing every %ss — ctrl-c to stop\n' "$INTERVAL"
  sleep "$INTERVAL"
done
