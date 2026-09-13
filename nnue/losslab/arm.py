#!/usr/bin/env python3
"""Run ONE arm of the entropy family, print where it lands, keep the board.

    penalty = -alpha   * H(bucket marginal over all tape positions) # inflate
              +beta    * H(bucket | one position)                   # decide
              -mi      * I(b_t ; b_t+2)                             # align
              +kl      * KL(p_t+2 || p_t)                           # align
              +even    * KL(uniform || bucket marginal)             # stay open
              -game    * I(bucket ; GAME)                           # game types
              -persist * log p_t+2[argmax p_t]                      # don't move

and --lrmult is the router's own LR multiplier (default 100), which is not a
penalty at all but is the other knob that moves flip2.

Usage, from the repo root, with the venv python:

    python arm.py --alpha 1 --beta 0.3 --mi 3 --kl 0.1
    python arm.py --alpha 3 --beta 1 --mi 10 --kl 0.3 --lrmult 300 --seeds 0,1,2
    python arm.py --game 1 --persist 1 --bits 12 --batch 16384 --pretrain 2000 \
                  --merge 512,128,64,32,16
    python arm.py --board                # just print the board, run nothing
    python arm.py --alpha 1 --dry-run    # scale report only, no GPU

Every run appends to sweeps/manual.jsonl and is ranked against every arm ever
measured (the LLM search's log, the grid sweeps, the random sweeps), by Pareto
dominance on (val, flip2) with the measured noise margins. Re-running the same
weights re-measures rather than skipping, so a repeat is a free seed.

TWO-STAGE. --pretrain N fits the ROUTER ALONE to the penalty for N tape steps,
then FREEZES it and fits the tables to it. Nothing else changes. Use it when
the penalty and the loss are pulling against each other, which they always are:
jointly, what comes out is a truce set by the weight; frozen, the rule's
stability is decided in stage 1 and stage 2 cannot trade it away. With
--pretrain the penalty weights are the ONLY thing acting in stage 1, so only
their RATIO matters -- pin one at 1 and move the other.

MERGE. --merge 512,128,64,32,16 k-means the learned bucket tables down that
ladder after training and re-measures val/flip2/eff at every rung, from ONE
trained model. Two buckets that end up sharing a table stop counting as a flip
(nothing to re-gather), so this is the cheap way to ask "how many tables does
this router actually need". Measured on a loss-trained 4096-bucket router it is
a bad trade: 4096 -> 64 cost +5.2% val and bought 6 points of flip2.

SCALE. The three entropy terms are bounded by log(nb) nats, so at
weight w the term contributes at most 4.16 * w, against a main loss of about
0.0258. They are equal at w = 0.006; at w = 1 the penalty is ~160x the main
loss, at w = 100 it is ~16000x. KL is unbounded, so it has no such ceiling and
runs hotter than its weight suggests. --dry-run prints this for your weights.
"""

import argparse, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lab.config import RunConfig
from lab import evaluator, pareto
from lab.log import RunLog
from sweep_entropy import write_arm, _fmt, load_baselines

VAR_TMPL = (HERE / "sweeps" / "var_family.tmpl.py").read_text()
ARM_DIR = HERE / "sweeps" / "arms"


def write_var_arm(name, v, d, r, l, w, x):
    """The bit-variance family. Same interface as write_arm: a penalty file."""
    ARM_DIR.mkdir(parents=True, exist_ok=True)
    src = VAR_TMPL
    for key, val in (("V", v), ("D", d), ("R", r), ("L", l), ("W", w),
                     ("X", x)):
        src = src.replace(f"__{key}__", repr(float(val)))
    path = ARM_DIR / f"{name}.py"
    path.write_text(src)
    ns = {}
    exec(compile(src, str(path), "exec"), ns)     # a typo costs a second here,
    if not callable(ns.get("penalty")):           # not a minute of GPU
        raise SystemExit(f"{path} defines no penalty()")
    return str(path)

LOGS = ["losslab_run.jsonl", "sweep_entropy.jsonl", "sweeps/probe.jsonl",
        "sweeps/random.jsonl", "sweeps/wide.jsonl", "sweeps/manual.jsonl"]
LOG64 = math.log(64.0)


def board(cfg, seeds):
    """Every measured arm, from every log, ranked together."""
    base, vals, flips = load_baselines(str(HERE / "losslab_run.jsonl"))
    arms, collapsed = dict(base), {}
    for name in LOGS:
        p = HERE / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") != "arm" or not r.get("metrics"):
                continue
            m = r["metrics"]
            (collapsed if m.get("eff", 99) < cfg.min_eff else arms)[r["name"]] = m
    reps = [{"val": v, "flip2": f} for v, f in zip(vals, flips)]
    if len(reps) > 1:
        ev, ef = pareto.noise_margins(reps, n_seeds=len(seeds))
    else:
        ev, ef = cfg.eps_val, cfg.eps_flip
    return arms, collapsed, ev, ef


def scale_report(a, b, mi, kl, game=0.0, persist=0.0, even=0.0, efloor=0.01,
                 nb=64, val=0.0258, pretrain=0):
    cap0 = math.log(nb)
    if pretrain:
        print(f"\nscale check: --pretrain {pretrain} means the penalty is the "
              f"ONLY loss in stage 1, so absolute weights do nothing and only "
              f"their ratio matters. Entropy terms cap at log {nb} = "
              f"{cap0:.2f} nats.")
    else:
        print(f"\nscale check (main loss ~ {val:.4f}; entropy terms cap at "
              f"log {nb} = {cap0:.2f} nats):")
    for nm, w, cap in (("alpha", a, cap0), ("beta", b, cap0),
                       ("mi", mi, cap0), ("game", game, cap0),
                       ("persist", persist, None), ("kl", kl, None),
                       ("even", even, (math.log(1.0 / efloor) if efloor
                                       else None))):
        if not w:
            print(f"  {nm:<8} 0        -- off")
        elif cap:
            print(f"  {nm:<8} {w:<9.4g} max term {w*cap:9.4g} = "
                  f"{w*cap/val:8.1f}x the main loss")
        else:
            print(f"  {nm:<8} {w:<9.4g} unbounded -- a log of a probability "
                  f"has no ceiling; it blows up where the mass goes to zero")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="weight on -H(marginal over all positions): inflates "
                         "the number of buckets used")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="weight on +H(within one position): makes each "
                         "position's choice decisive")
    ap.add_argument("--mi", type=float, default=0.0,
                    help="weight on -I(b_t; b_t+2): two-ply predictability. "
                         "Collapse-proof (I = 0 at the point mass)")
    ap.add_argument("--kl", type=float, default=0.0,
                    help="weight on +KL(p_t+2 || p_t): two-ply agreement. "
                         "Strong, cheap on val, collapse-prone alone")
    # --- the bit-variance family: 6 bits, not the 64-bucket simplex ---------
    ap.add_argument("--var", type=float, default=0.0,
                    help="reward mean_j Var_t(sigmoid z_j): use the whole span "
                         "of each bit (max 0.25 per bit)")
    ap.add_argument("--dmove", type=float, default=0.0,
                    help="charge mean_j E[(s_j(t+2)-s_j(t))^2]: don't flip")
    ap.add_argument("--ratio", type=float, default=0.0,
                    help="charge move-to-move variance OVER across-position "
                         "variance, per bit. Scale-free, needs no balancing")
    ap.add_argument("--logdet", type=float, default=0.0,
                    help="reward logdet Cov(s): span AND decorrelation in one "
                         "term. Ceiling 6*log(0.25) = -8.32 for independent "
                         "bits at full span; -inf if bits duplicate")
    ap.add_argument("--ldmove", type=float, default=0.0,
                    help="charge logdet Cov(s(t+2)-s(t)): the VOLUME of the "
                         "two-ply step, not just its size. At --logdet == "
                         "--ldmove it is a scale-free volume ratio")
    ap.add_argument("--decorr", type=float, default=0.0,
                    help="charge mean off-diagonal corr(s_i,s_j)^2")
    ap.add_argument("--even", type=float, default=0.0,
                    help="weight on KL(uniform || marginal) = -mean_b log m_b "
                         "- log nb. Anti-collapse with the log on the EMPTY "
                         "side: unlike --alpha, which weights each bucket by "
                         "its own mass and so goes quiet exactly where a "
                         "bucket is dying, this one holds full strength down "
                         "to m ~ 1e-11 (measured; --alpha is 1000x weaker at "
                         "m ~ 1e-6). Prevents collapse; cannot cure a "
                         "saturated one")
    ap.add_argument("--even-floor", type=float, default=0.0,
                    help="floor inside that log, in units of 1/nb. Default OFF "
                         "and you almost certainly want it off: the gradient "
                         "w.r.t. the LOGITS is ~(1/nb)(p/m), which is O(1/nb) "
                         "at any scale because dm/dz ~ m cancels the 1/m. A "
                         "floor caps d/dm at 1/floor and then multiplies by a "
                         "vanishing dm/dz, which is what kills the term")
    ap.add_argument("--game", type=float, default=0.0,
                    help="weight on -I(bucket; GAME): spread across games, "
                         "agree within one. The game-type criterion. "
                         "Collapse-proof (I = 0 at the point mass)")
    ap.add_argument("--persist", type=float, default=0.0,
                    help="weight on -log p_t+2[argmax p_t], the LOG-LOSS form "
                         "of persistence. Hard current bucket on the left of "
                         "the KL, unlike --kl. Strong; collapse-prone alone")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="weight on relu(MGAP - distance to a flip). In --flat "
                         "that distance is the top-1/top-2 logit gap (ONE "
                         "boundary); in bits mode it is min_j |z_j| (a min "
                         "over six)")
    ap.add_argument("--mgap", type=float, default=1.0,
                    help="how wide a margin --margin demands, in logit units")
    ap.add_argument("--flat", action="store_true",
                    help="softmax over 64 buckets + argmax instead of 6 sign "
                         "bits: a (rows,64) matrix, an unconstrained partition, "
                         "and one decision boundary per position")
    ap.add_argument("--bits", type=int, default=RunConfig.router_bits,
                    help="2**bits buckets (default 6 = 64). At 12 a (batch, nb) "
                         "score matrix is 1 GB, so drop --batch to 16384")
    ap.add_argument("--batch", type=int, default=RunConfig.batch)
    ap.add_argument("--norm", default="none",
                    choices=["none", "center", "bn", "white", "stat"],
                    help="normalise the router logits before the threshold; "
                         "free at export either way. center: subtract the mean "
                         "so every bit splits 50/50 and none can go constant. "
                         "white: ZCA on top of that, so the bits decorrelate "
                         "and all 2^k buckets come out about equally likely -- "
                         "even buckets for nothing, at the risk of promoting a "
                         "flat direction to a bit that flips every move. bn: "
                         "full BatchNorm, which can only differ from center in "
                         "its gradient, since sign(z/sigma) = sign(z)")
    ap.add_argument("--eigfloor", type=float, default=1e-2,
                    help="--norm white only: eigenvalue floor, relative to the "
                         "mean eigenvalue, before Cov^-1/2. 0 = exact "
                         "whitening (even buckets, flat directions promoted to "
                         "flipping bits). It does NOT reach center at 1: the "
                         "floor only lifts the directions BELOW it, so center "
                         "needs a floor above the largest eigenvalue. 300-step "
                         "probe: eff 3084 / 3092 / 1389 at 0 / 0.1 / 1 against "
                         "center's 390.")
    ap.add_argument("--pretrain", type=int, default=0,
                    help="two-stage: fit the ROUTER ALONE to the penalty for "
                         "this many tape steps, then freeze it and fit the "
                         "tables to it. 0 = joint, as every earlier arm")
    ap.add_argument("--pretrain-lr", type=float, default=1e-2)
    ap.add_argument("--merge", default="",
                    help="after training, k-means the tables down this ladder "
                         "(e.g. 512,128,64,32,16) and re-measure at each rung")
    ap.add_argument("--lrmult", type=float, default=RunConfig.router_lrmult,
                    help="router LR multiplier (default 100)")
    ap.add_argument("--rank", type=int, default=0,
                    help="factorise the router matrix as (rows,r) @ (r,6). "
                         "r >= 6 is the same rule, a different gradient path; "
                         "r < 6 is a smaller family (<= sum_{i<=r} C(6,i) "
                         "buckets reachable)")
    ap.add_argument("--wd", type=float, default=0.0,
                    help="weight decay on the router only. With --rank this is "
                         "nuclear norm on the product, i.e. low rank")
    ap.add_argument("--lr", type=float, default=RunConfig.lr,
                    help="global LR -- moves the whole net, not just the router")
    ap.add_argument("--seeds", default="0",
                    help="comma list. 1 seed screens, 3 seeds is what a number "
                         "has to survive before it means anything")
    ap.add_argument("--steps", type=int, default=RunConfig.dev_steps)
    ap.add_argument("--name", default="", help="override the arm name")
    ap.add_argument("--loss-file", default="",
                    help="run a hand-written penalty file as-is, ignoring all "
                         "the weight flags. Needs --name")
    ap.add_argument("--log", default=str(HERE / "sweeps" / "manual.jsonl"))
    ap.add_argument("--board", action="store_true",
                    help="print the board and exit -- no GPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the scale report and exit -- no GPU")
    ap.add_argument("--no-table", action="store_true")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cfg = RunConfig()
    cfg.dev_steps, cfg.dev_seeds = args.steps, tuple(seeds)
    cfg.lr, cfg.router_lrmult = args.lr, args.lrmult
    cfg.router_rank, cfg.router_wd = args.rank, args.wd
    cfg.router_flat = bool(args.flat)
    cfg.router_bits, cfg.batch = args.bits, args.batch
    cfg.router_norm = args.norm
    cfg.router_eigfloor = args.eigfloor
    cfg.router_pretrain, cfg.router_pretrain_lr = args.pretrain, args.pretrain_lr
    cfg.router_merge = args.merge
    os.makedirs(cfg.workdir, exist_ok=True)

    arms, collapsed, ev, ef = board(cfg, seeds)
    if args.board:
        print(pareto.table(arms, ev, ef, ref="none", limit=200,
                           band=(cfg.flip2_lo, cfg.flip2_hi)))
        print(f"\n{len(collapsed)} collapsed arms (eff < {cfg.min_eff}) not "
              f"shown; margins are for a {len(seeds)}-seed mean")
        return

    if args.loss_file:
        if not args.name:
            raise SystemExit("--loss-file needs --name")
        args.dry_run = False
    varfam = any((args.var, args.dmove, args.ratio, args.logdet,
                  args.ldmove, args.decorr))
    entfam = any((args.alpha, args.beta, args.mi, args.kl, args.margin,
                  args.game, args.persist, args.even))
    if args.pretrain and not entfam and not args.loss_file:
        raise SystemExit("--pretrain with no penalty would train the router on "
                         "nothing for {} steps and then freeze it at its "
                         "random init. Give it --game/--persist/--alpha/..."
                         .format(args.pretrain))
    if args.bits >= 10 and args.batch > 32768:
        raise SystemExit(f"--bits {args.bits} is {1 << args.bits} buckets; a "
                         f"(batch, nb) score matrix at batch {args.batch} is "
                         f"{4 * args.batch * (1 << args.bits) / 2**30:.1f} GB "
                         f"and the card is 7.6. Use --batch 16384 (and ~4x the "
                         f"--steps to see the same positions).")
    if varfam and args.flat:
        raise SystemExit("the bit-variance family reads 6 soft bits; in --flat "
                         "mode tz is 64 bucket logits and sigmoid(tz) means "
                         "nothing. Use --alpha/--beta/--mi/--kl/--margin")
    if varfam and entfam:
        raise SystemExit("the two families are separate penalty files -- run "
                         "one at a time (entropy: --alpha/--beta/--mi/--kl; "
                         "bit variance: --var/--dmove/--ratio/--logdet/"
                         "--ldmove/--decorr)")
    if not varfam:
        scale_report(args.alpha, args.beta, args.mi, args.kl,
                     game=args.game, persist=args.persist,
                     even=args.even, efloor=args.even_floor,
                     nb=1 << args.bits, pretrain=args.pretrain)
        if args.even and args.even_floor:
            print(f"  even     FLOORED at {args.even_floor:g}/nb: the term "
                  f"stops pushing on any bucket below that, which is most of "
                  f"the reason to use it. Measured: at m ~ 1e-6 a floor of "
                  f"0.01/nb costs 100x the push-up gradient.")
    if args.dry_run:
        return

    suffix = ("" if args.lr == RunConfig.lr else "_lr" + _fmt(args.lr))
    if args.rank:
        suffix += "_r" + str(args.rank)
    if args.wd:
        suffix += "_wd" + _fmt(args.wd)
    if args.flat:
        suffix += "_flat"
    if args.bits != RunConfig.router_bits:
        suffix += "_nb" + str(1 << args.bits)
    if args.batch != RunConfig.batch:
        suffix += "_bs" + str(args.batch)
    if args.norm != "none":
        suffix += "_" + args.norm
        if args.norm == "white" and args.eigfloor != 1e-2:
            suffix += f"_ef{args.eigfloor:g}"
    if args.pretrain:
        suffix += "_pre" + str(args.pretrain)
    if args.loss_file:
        name, path = args.name + suffix, str(Path(args.loss_file).resolve())
    elif varfam:
        name = args.name or ("v_V" + _fmt(args.var) + "_D" + _fmt(args.dmove)
                             + "_R" + _fmt(args.ratio) + "_L" + _fmt(args.logdet)
                             + "_W" + _fmt(args.ldmove) + "_X" + _fmt(args.decorr)
                             + "_lm" + _fmt(args.lrmult) + suffix)
        path = write_var_arm(name, args.var, args.dmove, args.ratio,
                             args.logdet, args.ldmove, args.decorr)
    else:
        name = args.name or ("m_a" + _fmt(args.alpha) + "_b" + _fmt(args.beta)
                             + "_M" + _fmt(args.mi) + "_K" + _fmt(args.kl)
                             + ("_Q" + _fmt(args.game) if args.game else "")
                             + ("_N" + _fmt(args.persist) if args.persist else "")
                             + ("_E" + _fmt(args.even) if args.even else "")
                             + ("_P" + _fmt(args.margin) + "g"
                                + _fmt(args.mgap) if args.margin else "")
                             + "_lm" + _fmt(args.lrmult) + suffix)
        # G/PAIRMODE are the older joint/cond/agree pair forms; unused here.
        path = write_arm(name, args.alpha, args.beta, 0.0, "mi",
                         mi=args.mi, kl=args.kl, margin=args.margin,
                         mgap=args.mgap, persist=args.persist, game=args.game,
                         even=args.even, efloor=args.even_floor)
    print(f"\narm {name}\n  penalty {path}\n  {args.steps} steps x "
          f"{len(seeds)} seed(s) on cuda, ~{60*args.steps/1526*len(seeds):.0f} s"
          f"\n", flush=True)

    knobs = {"A": args.alpha, "B": args.beta, "M": args.mi, "K": args.kl,
             "Q": args.game, "N": args.persist,
             "E": args.even, "EFLOOR": args.even_floor,
             "bits": args.bits, "batch": args.batch,
             "pretrain": args.pretrain, "merge": args.merge,
             "norm": args.norm,
             "eigfloor": args.eigfloor,
             "P": args.margin, "MGAP": args.mgap, "flat": bool(args.flat),
             "V": args.var, "D": args.dmove, "R": args.ratio,
             "L": args.logdet, "W": args.ldmove, "X": args.decorr,
             "rank": args.rank, "wd": args.wd}
    log, runs, err = RunLog(args.log), [], ""
    t0 = time.time()
    for s in seeds:
        met, e = evaluator.run_one(cfg, steps=args.steps, seed=s,
                                   aux="custom=1", loss_file=path)
        if met is None:
            err = e
            break
        print(f"  seed {s}: val {met['val']:.6f}  flip2 {100*met['flip2']:5.1f}%"
              f"  eff {met['eff']:5.1f}  occ {met.get('occ', 0):.0f}", flush=True)
        runs.append(met)
    if err:
        print(f"\nFAILED: {err[:800]}")
        log.write("arm", name=name, G=0.0, pairmode="mi", lr=args.lr,
                  lrmult=args.lrmult, steps=args.steps, seeds=seeds,
                  error=err, **knobs)
        raise SystemExit(1)

    met = evaluator.average(runs)
    met.update(dict(knobs, G=0.0, pairmode="mi", lr=args.lr,
                    lrmult=args.lrmult))
    dead = met["eff"] < cfg.min_eff
    log.write("arm", name=name, G=0.0, pairmode="mi", lr=args.lr,
              lrmult=args.lrmult, steps=args.steps, seeds=seeds, metrics=met,
              per_seed=runs, collapsed=dead, **knobs)

    ref = arms.get("none", {"val": 0.025805, "flip2": 0.422})
    print(f"\n  MEAN val {met['val']:.6f} ({100*(met['val']/ref['val']-1):+.2f}%"
          f" vs none)  flip2 {100*met['flip2']:.1f}% "
          f"({100*(met['flip2']-ref['flip2']):+.1f} pts)  eff {met['eff']:.1f}"
          f"  occ {met.get('occ', 0):.0f}  [{time.time()-t0:.0f} s]")
    if len(runs) > 1:
        print(f"  seed spread: val {met.get('val_spread', 0):.6f}, flip2 "
              f"{100*met.get('flip2_spread', 0):.1f} pts")
    if dead:
        print(f"  COLLAPSED (eff {met['eff']:.2f} < {cfg.min_eff}): the router "
              f"stopped routing, so flip2 = 0 means nothing. Logged, not ranked.")
        return
    arms[name] = met
    if not args.no_table:
        print("\n" + pareto.table(arms, ev, ef, ref="none", limit=40,
                                  band=(cfg.flip2_lo, cfg.flip2_hi)))


if __name__ == "__main__":
    main()
