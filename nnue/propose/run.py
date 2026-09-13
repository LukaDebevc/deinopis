"""The loop: hand the rule search to a model, keep only what measures.

    python3 run.py models                 # what the key can actually see
    python3 run.py probe                  # cache 8192 positions for validation
    python3 run.py propose --n 12         # one round of proposals
    python3 run.py score                  # measure everything unscored, one pass
    python3 run.py report
    python3 run.py loop --rounds 20 --score-every 5

Nothing here decides anything. The model proposes, the sandbox rejects what
cannot run, and `score` measures what survives against the twenty hand-written
baselines in `rulestats.POOL` -- in the same pass, on the same positions, so the
comparison is paired. A proposal that is not in the leaderboard has not beaten
anything; it has merely compiled.

The feedback channel is the point: last round's scores and last round's
rejection reasons both go into the next prompt. Without that the model restates
its first ten ideas forever.
"""
import argparse
import json
import os
import sys
import time

NNUE = "./nnue"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, NNUE)
os.chdir(NNUE)                      # power.py resolves its checkpoint relatively

import torch                                                     # noqa: E402
import client as C                                               # noqa: E402
import prims                                                     # noqa: E402
import sandbox as S                                              # noqa: E402

POOL = os.path.join(HERE, "pool.jsonl")
REJECTS = os.path.join(HERE, "rejects.jsonl")
SCORES = os.path.join(HERE, "scores.json")
PROBE = os.path.join(HERE, "probe.pt")
QUOTA = os.path.join(HERE, "quota.json")
DIM, RIDGE = 48, 1e3
W = {"quiet": 0.6897, "capture": 0.3019, "promo": 0.0113}   # legal-move mix


# ------------------------------------------------------------------- storage
def jload(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def jappend(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def probe_batch():
    if not os.path.exists(PROBE):
        raise SystemExit("no probe batch: run `python3 run.py probe` first")
    return torch.load(PROBE, weights_only=True)


# --------------------------------------------------------------- the prompt
def api_reference():
    """Generated from prims.Board, so the prompt cannot drift from the code."""
    import inspect
    lines = []
    for name, fn in inspect.getmembers(prims.Board, inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = str(inspect.signature(fn)).replace("(self, ", "(").replace("(self)", "()")
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        lines.append(f"  b.{name}{sig}\n      {doc}")
    return "\n".join(lines)


EXAMPLE = '''def rule(b):
    """Blocked centre x whether each side still has pawns on both wings."""
    own = b.pieces(0, 0)
    opp = b.pieces(1, 0)
    blocked = b.AND(b.shift(own, 0, 1), opp)
    jam = b.clamp(b.popcount(b.AND(blocked, b.mask("centrefiles"))), 0, 2)
    ours = b.any(b.AND(own, b.mask("kingside")))
    theirs = b.any(b.AND(opp, b.mask("queenside")))
    return b.combine([(jam, 3), (ours, 2), (theirs, 2)])'''


def build_prompt(n, scores, pool, rejects):
    brief = open(os.path.join(HERE, "brief.md")).read()
    ranked = sorted((s for s in scores.values() if "power" in s),
                    key=lambda s: -s["power"])
    board = "\n".join(
        f"  {s['name'][:34]:<34}{s['power']:>7.3f}%{s.get('refresh', float('nan')):>8.2f}%"
        f"{s.get('perplexity', 0):>8.0f}{s.get('cost', 0):>6}"
        for s in ranked[:18])
    tried = ", ".join(r["name"] for r in pool[-40:]) or "(nothing yet)"
    bad = {}
    for r in rejects[-40:]:
        bad.setdefault(r["reason"], []).append(r.get("detail", "")[:70])
    fails = "\n".join(f"  {k} x{len(v)}: {v[0]}" for k, v in bad.items()) or "  (none yet)"
    return f"""{brief}

## The API -- this is the whole vocabulary

{api_reference()}

Masks available to `b.mask`: {", ".join(sorted(prims.MASKS))}
Sides: 0 = the side to move, 1 = the opponent. Piece types: 0 pawn, 1 knight,
2 bishop, 3 rook, 4 queen, 5 king. Rank 0 is the mover's own back rank, so the
side to move always advances toward rank 7.

## A rule that scores well, for shape

```python
{EXAMPLE}
```

## Measured so far

  {'rule':<34}{'power':>8}{'refresh':>8}{'eff.bkt':>8}{'cost':>6}
{board or '  (nothing scored yet)'}

Already proposed (do not repeat the same distinction): {tried}

Last round's rejections:
{fails}

## Now

Propose {n} NEW rules. Each must draw a distinction none of the above draws.
Prefer one clear structural idea per rule over stacking four weak terms.

Reply with JSON only:
{{"rules": [{{"name": "short label", "code": "def rule(b):\\n    ..."}}]}}"""


def parse_reply(text):
    """Models drift between raw JSON, fenced JSON and fenced Python. Take any."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t.split("\n", 1)[1] if t[:10].strip() in ("json", "python", "py") else t
    try:
        obj = json.loads(t)
        rules = obj["rules"] if isinstance(obj, dict) else obj
        return [(r.get("name") or "unnamed", r["code"]) for r in rules]
    except Exception:
        pass
    out = []
    for blk in text.split("```"):
        b = blk.strip()
        if b.startswith("python"):
            b = b.split("\n", 1)[1]
        if b.startswith("def rule(b)"):
            doc = b.split('"""')[1].split("\n")[0].strip() if '"""' in b else "unnamed"
            out.append((doc[:48], b))
    return out


# ------------------------------------------------------------------ commands
def cmd_models(a):
    cl = C.Client(state_path=QUOTA)
    have = set(cl.list_models())
    print(f"{len(have)} models generate content with this key.\n")
    print(f"{'configured':<26}{'visible':>9}{'rpm':>6}{'rpd':>8}{'left today':>12}")
    for m, lim in C.MODELS.items():
        print(f"{m:<26}{'yes' if m in have else 'NO':>9}{lim['rpm']:>6}"
              f"{lim['rpd']:>8}{cl.remaining(m):>12}")
    unknown = sorted(h for h in have if h not in C.MODELS)
    print(f"\nvisible but not configured ({len(unknown)}):")
    for h in unknown:
        print(f"  {h}")


def cmd_probe(a):
    from loader import Batcher
    b = Batcher("data/all.data", "cpu",
                batch=a.probe_n, val_cap=a.probe_n, block=a.probe_n)
    feat = next(iter(b.val_batches()))[0]
    torch.save(feat, PROBE)
    print(f"probe: {feat.shape[0]} positions -> {PROBE}")


def cmd_propose(a):
    feat = probe_batch()
    pool, rejects = jload(POOL), jload(REJECTS)
    scores = json.load(open(SCORES)) if os.path.exists(SCORES) else {}
    known = {r["sig"]: r["name"] for r in pool}
    prompt = build_prompt(a.n, scores, pool, rejects)
    if a.dry_run:                       # inspect the prompt without spending quota
        print(prompt)
        return
    cl = C.Client(state_path=QUOTA)
    t0 = time.time()
    text = cl.generate(a.model, prompt, temperature=a.temperature)
    got = parse_reply(text)
    if not got:
        jappend(REJECTS, {"name": "-", "reason": "unparseable", "model": a.model,
                          "detail": text[:200]})
        print(f"{a.model}: reply parsed to zero rules")
        return
    kept = 0
    for name, code in got:
        try:
            rec = S.accept(code, feat, known=known)
        except S.Reject as e:
            jappend(REJECTS, {"name": name, "reason": e.reason, "detail": e.detail,
                              "model": a.model})
            print(f"  reject {name[:40]:<42} {e.reason}: {e.detail[:60]}")
            continue
        uniq = name if name not in {r["name"] for r in pool} else f"{name}~{len(pool)}"
        rec.update(name=uniq, code=code, model=a.model, round=len(pool))
        jappend(POOL, rec)
        known[rec["sig"]] = uniq
        pool.append(rec)
        kept += 1
        print(f"  keep   {uniq[:40]:<42} {rec['size']:>5} bkt {rec['perplexity']:>6.0f} eff"
              f" {rec['cost']:>4} ops  reads={rec['reads']}")
    print(f"{a.model}: {kept}/{len(got)} kept in {time.time() - t0:.0f}s"
          f"  ({cl.remaining(a.model)} requests left today)")


def _fams(names, pool):
    """name -> (feat -> index, size) for both baselines and candidates."""
    import rulestats as R
    out = {}
    for n in names:
        if n in R.POOL:
            fn = R.POOL[n]
            out[n] = ((lambda f, g=fn: g(f)[0]), fn(torch.zeros(1, 32, dtype=torch.long)
                                                    + prims.PAD)[1])
        else:
            rec = next(r for r in pool if r["name"] == n)
            f = S.compile_rule(rec["code"])
            out[n] = ((lambda x, g=f: g(prims.Board(x, max_cost=10 ** 6))[0]), rec["size"])
    return out


def cmd_score(a):
    import rulestats as R
    import power as P
    P.DEV = a.device
    pool = jload(POOL)
    scores = json.load(open(SCORES)) if os.path.exists(SCORES) else {}
    todo = [r["name"] for r in pool if r["name"] not in scores]
    todo += [n for n in R.POOL if n not in scores]          # baselines, once
    if not todo:
        print("nothing unscored")
        return
    # (size, 49, 49) of statistics per bucket, twice, so cap the buckets per pass.
    passes, cur, tot = [], [], 0
    sizes = _fams(todo, pool)
    for n in todo:
        s = sizes[n][1]
        if tot + s > a.budget and cur:
            passes.append(cur); cur, tot = [], 0
        cur.append(n); tot += s
    passes.append(cur)
    print(f"scoring {len(todo)} rules in {len(passes)} pass(es) on {a.device}, "
          f"{a.npos:,} positions each")
    for i, names in enumerate(passes):
        fams = {n: sizes[n] for n in names}
        acc, base_ss, seen = P.stats(fams, a.npos, DIM)
        for n in names:
            scores[n] = {"name": n, "power": round(P.score(*acc[n][:4], base_ss, seen, RIDGE), 4),
                         "n": seen, "size": fams[n][1]}
        print(f"  pass {i + 1}/{len(passes)}: {len(names)} rules, n = {seen:,}")
        json.dump(scores, open(SCORES, "w"), indent=1)
    _add_refresh(scores, todo, pool, a)
    json.dump(scores, open(SCORES, "w"), indent=1)
    cmd_report(a)


def _add_refresh(scores, todo, pool, a):
    """Per-move index change rate, from the engine's own legal moves."""
    try:
        import refresh as F
        feat, rows, frm, to, flag = F.load_moves(a.moves)
        g = F.apply_moves(feat, rows, frm, to, flag)
    except Exception as e:
        print(f"  refresh skipped ({type(e).__name__}: {e}) -- needs target/release/chess")
        return
    quiet, cap, pro = (flag == 0) | (flag == 1), (flag == 4) | (flag == 5), flag >= 8
    fams = _fams(todo, pool)
    for n in todo:
        fn, size = fams[n]
        base, idx = fn(feat), fn(g)
        ch = idx != base[rows]
        scores[n]["refresh"] = round(sum(
            w * ch[m].double().mean().item() * 100
            for m, w in ((quiet, W["quiet"]), (cap, W["capture"]), (pro, W["promo"]))), 3)
        c = torch.bincount(base, minlength=size).double()
        p = c / c.sum(); nz = p[p > 0]
        scores[n]["perplexity"] = round(float(torch.exp(-(nz * nz.log()).sum())), 1)
        rec = next((r for r in pool if r["name"] == n), None)
        if rec:
            scores[n]["cost"] = rec["cost"]
            scores[n]["reads"] = rec["reads"]


def cmd_report(a):
    scores = json.load(open(SCORES)) if os.path.exists(SCORES) else {}
    import rulestats as R
    rows = sorted(scores.values(), key=lambda s: -s.get("power", -1))
    hdr = (f"{'rule':<36}{'power':>9}{'refresh':>9}{'eff.bkt':>9}"
           f"{'cost':>6}{'power/refresh':>15}  where")
    print("\n" + hdr); print("-" * len(hdr))
    for s in rows:
        r = s.get("refresh")
        ratio = s["power"] / max(r, 1e-3) if r is not None else float("nan")
        print(f"{s['name'][:36]:<36}{s.get('power', float('nan')):>8.3f}%"
              f"{(r if r is not None else float('nan')):>8.2f}%"
              f"{s.get('perplexity', float('nan')):>9.0f}{s.get('cost', 0):>6}"
              f"{ratio:>15.2f}  {'hand' if s['name'] in R.POOL else 'model'}")


def cmd_loop(a):
    for i in range(a.rounds):
        print(f"\n=== round {i + 1}/{a.rounds}  [{a.model}] ===")
        try:
            cmd_propose(a)
        except C.Quota as e:
            print(f"quota: {e}")
            break
        if (i + 1) % a.score_every == 0:
            cmd_score(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["models", "probe", "propose", "score", "report", "loop"])
    p.add_argument("--model", default="gemini-3.1-flash-lite")
    p.add_argument("--n", type=int, default=12, help="proposals per round")
    p.add_argument("--probe-n", type=int, default=8192, help="positions in the validation probe")
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--score-every", type=int, default=5)
    p.add_argument("--npos", type=int, default=200_000)
    p.add_argument("--moves", type=int, default=20_000)
    p.add_argument("--budget", type=int, default=8_000, help="declared buckets per scoring pass")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
