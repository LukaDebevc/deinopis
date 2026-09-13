#!/usr/bin/env python3
"""Sweep the three-entropy family for the router, and print the curve.

    penalty = -A*H(across all positions) + B*H(within a position)
              + G*H(b_t, b_{t+2})

plus the router learning rate. Same protocol as the LLM search: nnue/train.py
through lab/evaluator.run_one, dev_steps x dev_seeds per arm, the noise margins
from lab/pareto, ranked by Pareto dominance on (val, flip2) with no
scalarisation. It shares the log format with losslab_run.jsonl but writes its
own file, so the LLM search's board is not polluted with grid points.

Why a grid at all, when there is a search running: the search buys one-off
functional forms and has put nothing in the target band; a dense sweep of a
family that is KNOWN to move flip2 (`ginfo` is the A==B, game-level corner of
it) buys the SHAPE of the val/flip2 trade-off -- whether there is a knee or a
straight line -- which no single candidate can tell you.

  # what it would run, with the cost, without running anything
  python sweep_entropy.py --stage axes --dry-run

  # stage 1: one term at a time, both pair modes  (the axis curves)
  python sweep_entropy.py --stage axes

  # stage 2: cross product in whatever region stage 1 says is interesting
  python sweep_entropy.py --alpha 0,3e-3 --beta 1e-3,3e-3 --gamma 3e-3,1e-2 \
      --pairmode cond --cross

  # the router LR axis, on the combos worth it
  python sweep_entropy.py --alpha 3e-3 --beta 3e-3 --gamma 3e-3 --cross \
      --lrmult 30,100,300

Resumable: an arm already in the log is skipped unless --redo.
"""

import argparse
import itertools
import json
import math
import random
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lab.config import RunConfig
from lab import evaluator, pareto
from lab.log import RunLog

TMPL = (HERE / "sweeps" / "entropy_family.tmpl.py").read_text()
ARM_DIR = HERE / "sweeps" / "arms"


def _fmt(x):
    """1e-3 -> '1e-3', 0.0 -> '0', 100.0 -> '100' -- short enough for an arm name."""
    if x == 0:
        return "0"
    s = f"{x:g}"
    return s


def arm_name(a, b, g, mode, lr, lrmult, mi=0.0, kl=0.0):
    n = f"e_a{_fmt(a)}_b{_fmt(b)}_g{_fmt(g)}"
    if g:
        n += f"_{mode[0]}"                       # j / c / m / a / k
    if mi:
        n += f"_M{_fmt(mi)}"
    if kl:
        n += f"_K{_fmt(kl)}"
    if lrmult != RunConfig.router_lrmult:
        n += f"_lm{_fmt(lrmult)}"
    if lr != RunConfig.lr:
        n += f"_lr{_fmt(lr)}"
    return n


def write_arm(name, a, b, g, mode, mi=0.0, kl=0.0, margin=0.0, mgap=1.0,
               persist=0.0, game=0.0, even=0.0, efloor=0.0):
    ARM_DIR.mkdir(parents=True, exist_ok=True)
    p = ARM_DIR / f"{name}.py"
    src = (TMPL.replace("__A__", repr(float(a)))
               .replace("__B__", repr(float(b)))
               .replace("__G__", repr(float(g)))
               .replace("__M__", repr(float(mi)))
               .replace("__K__", repr(float(kl)))
               .replace("__N__", repr(float(persist)))
               .replace("__EFLOOR__", repr(float(efloor)))
               .replace("__E__", repr(float(even)))
               .replace("__Q__", repr(float(game)))
               .replace("__P__", repr(float(margin)))
               .replace("__MGAP__", repr(float(mgap)))
               .replace("__PAIRMODE__", mode))
    p.write_text(src)
    # Compile it here, not 90 seconds into a train.py startup. A typo in the
    # template cost a whole 17-arm stage once -- every arm failed at the exec,
    # one minute apart, with the same traceback.
    ns = {}
    exec(compile(src, str(p), "exec"), ns)
    if not callable(ns.get("penalty")):
        raise SystemExit(f"{p} defines no penalty()")
    return str(p)


def axes_arms(alpha, beta, gamma, modes):
    """One term at a time: the three axis curves, and nothing crossed.

    Includes the all-zero arm FIRST as a plumbing check -- it must land on the
    `none` baseline (val 0.0258, flip2 42.2%). If it does not, the loss file is
    not being wired in the way this script thinks it is and no curve below it
    means anything.
    """
    out = [(0.0, 0.0, 0.0, "joint")]
    out += [(w, 0.0, 0.0, "joint") for w in alpha if w]
    out += [(0.0, w, 0.0, "joint") for w in beta if w]
    out += [(0.0, 0.0, w, m) for m in modes for w in gamma if w]
    return out


def cross_arms(alpha, beta, gamma, modes):
    seen, out = set(), []
    for a, b, g, m in itertools.product(alpha, beta, gamma, modes):
        if g == 0 and m != modes[0]:
            continue                            # the pair mode is dead at G=0
        k = (a, b, g, m if g else modes[0])
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def loguni(rng, lo, hi):
    return float(10.0 ** rng.uniform(math.log10(lo), math.log10(hi)))


def random_arms(n, rng, p_zero, ranges):
    """n independent draws over (A, B, M, K, lrmult), log-uniform in each.

    Log-uniform because every one of these knobs has been seen to do nothing at
    1e-4 and to collapse the router at 1e-2 -- the interesting structure is in
    the exponent, so uniform-in-value would put almost every sample in the
    region already known to be too strong.

    Each of the four penalty weights is set to exactly 0 with probability
    p_zero, so the sample contains pairs and triples as well as full
    four-term arms: without that, "is B doing anything" is unanswerable
    because every arm has some B.

    G/PAIRMODE is left out of the draw. The joint and cond forms are both
    minimised at the point mass (measured: collapse at every weight tried),
    `agree` likewise, and `cond` at A == G is algebraically the M term -- so
    the two pair forms worth spending samples on are M and K, and they now
    have their own weights.
    """
    out = []
    for _ in range(n):
        w = []
        for key in ("A", "B", "M", "K"):
            lo, hi = ranges[key]
            x = 0.0 if rng.random() < p_zero else loguni(rng, lo, hi)
            w.append(float(f"{x:.2g}"))          # 2 s.f.: a readable arm name,
                                                 # and the 3rd digit is far
                                                 # below the noise floor anyway
        lm = loguni(rng, *ranges["lrmult"])
        a, b, mi, kl = w
        # An arm with no penalty at all is the `none` baseline with a different
        # LR; keep it only if the LR actually moved.
        out.append((a, b, 0.0, "mi", float(f"{lm:.3g}"), mi, kl))
    return out


def load_done(path):
    """{arm: metrics} for arms already finished, so a re-run continues."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "arm" and r.get("metrics"):
                done[r["name"]] = r["metrics"]
    return done


def load_baselines(path):
    """(baselines, noise replicates) read off the LLM search's own log.

    Reusing them rather than re-measuring: same code, same steps, same seeds,
    and they cost 14 minutes of GPU the search has already spent. Baseline
    records store their metrics FLAT; the noise record stores the raw per-seed
    `vals`/`flip2s`, which is what the margins have to be recomputed from --
    the `eps_*` in that record were sized for the search's seed count, not
    necessarily this sweep's.
    """
    skip = {"t", "kind", "name", "aux", "plan", "code", "origin", "front"}
    out, vals, flips = {}, [], []
    if not os.path.exists(path):
        return out, vals, flips
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "baseline":
                out[r.get("name", "?")] = {k: v for k, v in r.items()
                                           if k not in skip}
            elif r.get("kind") == "noise" and r.get("vals"):
                vals, flips = r["vals"], r["flip2s"]      # the latest wins
    return out, vals, flips


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", default="0,1e-3,3e-3,1e-2,3e-2",
                    help="weights on -H(across all positions)")
    ap.add_argument("--beta", default="0,1e-3,3e-3,1e-2,3e-2",
                    help="weights on +H(within a position)")
    ap.add_argument("--gamma", default="0,1e-3,3e-3,1e-2,3e-2",
                    help="weights on +H(b_t, b_t+2)")
    ap.add_argument("--pairmode", default="joint,cond",
                    help="joint = H(b_t,b_t+2) as written; cond = that minus "
                         "the marginal, so it does not fight alpha")
    ap.add_argument("--lrmult", default=str(RunConfig.router_lrmult),
                    help="router LR multiplier (the router's own LR knob)")
    ap.add_argument("--lr", default=str(RunConfig.lr),
                    help="global LR -- moves the whole net, not just the router")
    ap.add_argument("--stage", choices=("axes", "cross", "random"),
                    default="axes")
    ap.add_argument("--n", type=int, default=60,
                    help="random stage: how many arms to draw")
    ap.add_argument("--rseed", type=int, default=0,
                    help="random stage: the draw is reproducible from this, so "
                         "a re-run continues the same sample instead of a new one")
    ap.add_argument("--p-zero", type=float, default=0.2,
                    help="random stage: chance each weight is exactly 0")
    ap.add_argument("--range-a", default="1e-4,1e-1")
    ap.add_argument("--range-b", default="1e-4,1e-1")
    ap.add_argument("--range-m", default="1e-4,3e-1")
    ap.add_argument("--range-k", default="1e-5,5e-3")
    ap.add_argument("--range-lrmult", default="10,1000")
    ap.add_argument("--cross", action="store_true", help="same as --stage cross")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=RunConfig.dev_steps)
    ap.add_argument("--log", default=str(HERE / "sweep_entropy.jsonl"))
    ap.add_argument("--search-log", default=str(HERE / "losslab_run.jsonl"),
                    help="where to read the baselines and noise floor from")
    ap.add_argument("--redo", action="store_true", help="re-run finished arms")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="stop starting new arms after this much wall clock")
    args = ap.parse_args()

    nums = lambda s: [float(x) for x in s.split(",") if x != ""]
    alpha, beta, gamma = nums(args.alpha), nums(args.beta), nums(args.gamma)
    modes = [m.strip() for m in args.pairmode.split(",") if m.strip()]
    lrmults, lrs = nums(args.lrmult), nums(args.lr)
    seeds = [int(x) for x in args.seeds.split(",")]
    stage = "cross" if args.cross else args.stage

    if stage == "random":
        rng = random.Random(args.rseed)
        ranges = {"A": nums(args.range_a), "B": nums(args.range_b),
                  "M": nums(args.range_m), "K": nums(args.range_k),
                  "lrmult": nums(args.range_lrmult)}
        plan = random_arms(args.n, rng, args.p_zero, ranges)
        plan = [(a, b, g, m, lrs[0], lm, mi, kl)
                for (a, b, g, m, lm, mi, kl) in plan]
    else:
        combos = (cross_arms if stage == "cross" else axes_arms)(
            alpha, beta, gamma, modes)
        plan = [(a, b, g, m, lr, lm, 0.0, 0.0)
                for (a, b, g, m) in combos for lm in lrmults for lr in lrs]

    done = {} if args.redo else load_done(args.log)
    todo = [p for p in plan if arm_name(*p) not in done]
    per_arm_s = 60.0 * args.steps / 1526.0 * len(seeds)
    print(f"stage {stage}: {len(plan)} arms, {len(done)} already in the log, "
          f"{len(todo)} to run")
    print(f"cost: {args.steps} steps x {len(seeds)} seeds ~ {per_arm_s/60:.1f} "
          f"min/arm -> {len(todo) * per_arm_s / 3600:.1f} h "
          f"(and the GPU is shared -- twice that if the LLM search is running)")
    for p in plan:
        n = arm_name(*p)
        print(f"  {'done' if n in done else '    '} {n}")
    if args.dry_run:
        return

    cfg = RunConfig()
    cfg.dev_steps = args.steps
    cfg.dev_seeds = tuple(seeds)
    os.makedirs(cfg.workdir, exist_ok=True)
    log = RunLog(args.log)
    base, vals, flips = load_baselines(args.search_log)
    reps = [{"val": v, "flip2": f} for v, f in zip(vals, flips)]
    if len(reps) > 1:
        eps_val, eps_flip = pareto.noise_margins(reps, n_seeds=len(seeds))
        print(f"noise floor from {len(reps)} replicates of `none` in "
              f"{os.path.basename(args.search_log)}: val +-{eps_val:.6f}, "
              f"flip2 +-{100*eps_flip:.2f} pts (for a {len(seeds)}-seed mean)")
    else:
        eps_val, eps_flip = cfg.eps_val, cfg.eps_flip
        print(f"NO noise replicates found in {args.search_log} -- using the "
              f"fallback margins, which are a guess, not a measurement")

    arms = dict(base)
    arms.update({n: m for n, m in done.items()
                 if m.get("eff", 99) >= cfg.min_eff})
    collapsed = {n: m for n, m in done.items()
                 if m.get("eff", 99) < cfg.min_eff}
    zero_arm = arm_name(0.0, 0.0, 0.0, modes[0], lrs[0], lrmults[0])
    t0 = time.time()
    for i, (a, b, g, m, lr, lm, mi, kl) in enumerate(todo, 1):
        if args.hours and time.time() - t0 > args.hours * 3600:
            print(f"\nout of time after {i-1} arms")
            break
        name = arm_name(a, b, g, m, lr, lm, mi, kl)
        path = write_arm(name, a, b, g, m, mi, kl)
        cfg.lr, cfg.router_lrmult = lr, lm
        print(f"\n--- {i}/{len(todo)} {name} ---", flush=True)
        runs, err = [], ""
        for s in seeds:
            met, e = evaluator.run_one(cfg, steps=args.steps, seed=s,
                                       aux="custom=1", loss_file=path)
            if met is None:
                err = e
                break
            print(f"  seed {s}: val {met['val']:.6f}  flip2 "
                  f"{100*met['flip2']:.1f}%  eff {met['eff']:.1f}", flush=True)
            runs.append(met)
        if err:
            print(f"  FAILED: {err[:400]}", flush=True)
            log.write("arm", name=name, A=a, B=b, G=g, pairmode=m, M=mi,
                      K=kl, lr=lr, lrmult=lm, steps=args.steps, seeds=seeds,
                      error=err)
            continue
        met = evaluator.average(runs)
        met.update({"A": a, "B": b, "G": g, "pairmode": m, "M": mi, "K": kl,
                    "lr": lr, "lrmult": lm})
        dead = met["eff"] < cfg.min_eff
        log.write("arm", name=name, A=a, B=b, G=g, pairmode=m, M=mi, K=kl,
                  lr=lr, lrmult=lm, steps=args.steps, seeds=seeds, metrics=met,
                  per_seed=runs, collapsed=dead)
        print(f"  MEAN val {met['val']:.6f}  flip2 {100*met['flip2']:.1f}%  "
              f"eff {met['eff']:.1f}  (spread {100*met['flip2_spread']:.1f} pts)",
              flush=True)
        if dead:
            # A router that stopped routing scores flip2 = 0, which nothing can
            # beat, so leaving it in the ranking would hand it the frontier and
            # bend the whole curve around a model that is the architecture
            # switched off. It stays in the log and in the list below, because
            # WHERE the weights kill the router is one of the things a sweep is
            # for -- it just does not get to be an option.
            collapsed[name] = met
            print(f"  COLLAPSED (eff {met['eff']:.1f} < {cfg.min_eff}) -- "
                  f"kept in the log, left out of the ranking", flush=True)
            continue
        if name == zero_arm and "none" in arms:
            dv = 100 * (met["val"] / arms["none"]["val"] - 1)
            dfl = 100 * (met["flip2"] - arms["none"]["flip2"])
            ok = abs(dv) < 100 * eps_val / arms["none"]["val"] * 1.0 and \
                abs(dfl) < 100 * eps_flip
            print(f"  plumbing check vs `none`: dval {dv:+.2f}%, dflip2 "
                  f"{dfl:+.1f} pts -- {'within' if ok else 'OUTSIDE'} the "
                  f"noise margins", flush=True)
        arms[name] = met
        print(pareto.table(arms, eps_val, eps_flip, ref="none", limit=60,
                           band=(cfg.flip2_lo, cfg.flip2_hi)), flush=True)

    print("\n" + pareto.table(arms, eps_val, eps_flip, ref="none", limit=100,
                              band=(cfg.flip2_lo, cfg.flip2_hi)))
    if collapsed:
        print(f"\ncollapsed -- router stopped routing (eff < {cfg.min_eff}), "
              f"not ranked:")
        for n, mm in sorted(collapsed.items(), key=lambda kv: -kv[1]["eff"]):
            print(f"  {n:<34} val {mm['val']:.6f}  flip2 "
                  f"{100*mm['flip2']:.1f}%  eff {mm['eff']:.2f}")
    print(f"\nlog: {args.log}\narm sources: {ARM_DIR}")


if __name__ == "__main__":
    main()
