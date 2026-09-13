#!/usr/bin/env python3
"""Read the run log and show where things stand. Safe to run while the search
is going -- the log is append-only and every line is a finished candidate.

  python board.py                 # the board: every arm, ranked by how many
                                  # other arms beat it on BOTH val and flip2
  python board.py --front         # only the frontier, with each one's code
  python board.py --name c042     # one candidate: plan, code, numbers
  python board.py --failures      # what crashed or was screened out, and why
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import pareto


def load(path):
    arms, cands, eps, other = {}, {}, (0.000121, 0.0227), []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = r["kind"]
        if k == "noise":
            eps = (r["eps_val"], r["eps_flip"])
        elif k in ("baseline", "candidate"):
            m = {a: b for a, b in r.items()
                 if a not in ("t", "kind", "name", "plan", "code",
                              "origin", "front", "aux")}
            # Same gate the search applies on resume: a collapsed router scores
            # flip2 = 0 by construction and nothing can beat a zero, so showing
            # it here would put an arm on the printed frontier that the search
            # itself has already dropped.
            if m.get("eff", 1e9) < 4.0 or m.get("occ", 1e9) < 2:
                continue
            arms[r["name"]] = {a: b for a, b in r.items()
                               if a not in ("t", "kind", "name", "plan", "code",
                                            "origin", "front", "aux")}
            if k == "candidate":
                cands[r["name"]] = r
        else:
            other.append(r)
    return arms, cands, eps, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(Path(__file__).parent / "losslab_run.jsonl"))
    ap.add_argument("--front", action="store_true")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()

    arms, cands, (ev, ef), other = load(a.log)
    if not arms:
        print(f"no finished arms in {a.log} yet"); return

    if a.name:
        c = cands.get(a.name)
        if not c:
            print(f"{a.name} not found"); return
        print(f"=== {a.name} ({c.get('origin','')}) ===")
        print(f"val {c['val']:.6f}  flip2 {100*c['flip2']:.1f}%  "
              f"eff {c['eff']:.1f}  I(B;game) {c.get('I_game',0):.2f}")
        print(f"\nPLAN\n{c['plan']}\n\nCODE\n{c['code']}")
        return

    if a.failures:
        for r in other:
            if r["kind"] in ("crash", "ast_reject", "screened_out", "llm_error"):
                print(f"[{r['kind']}] {r.get('name','-')}: "
                      f"{str(r.get('err',''))[:200]}")
        return

    front = pareto.frontier(arms, ev, ef)
    if a.front:
        print(f"Pareto frontier: {len(front)} of {len(arms)} arms are beaten by "
              f"nothing on both axes at once\n")
        for n in front:
            m = arms[n]
            print(f"=== {n}  val {m['val']:.6f}  flip2 {100*m['flip2']:.1f}%  "
                  f"eff {m['eff']:.1f} ===")
            if n in cands:
                print(cands[n]["plan"][:400])
                print("```python\n" + cands[n]["code"] + "```")
            print()
        return

    from lab.config import RunConfig
    cfg = RunConfig()
    print(pareto.table(arms, ev, ef, ref="none", limit=a.limit,
                       band=(cfg.flip2_lo, cfg.flip2_hi)))
    print(f"\n{len(arms)} arms, {len(front)} on the frontier. "
          f"Frontier: {', '.join(front)}")
    print(f"`python board.py --front` for their code, "
          f"`python board.py --name NAME` for one.")


if __name__ == "__main__":
    main()
