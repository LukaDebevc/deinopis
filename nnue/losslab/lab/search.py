"""The controller: baselines -> noise floor -> {propose, screen, evaluate} loop.

Shape of one candidate's life:

  1. AST gate          rejects forbidden code without running it
  2. KNOBS expansion   a literal grid becomes up to `max_variants` arms
  3. SCREEN            300 steps, 1 seed (~15 s). Dropped only if CLEARLY
                       dominated by the no-penalty baseline -- the cheap
                       ranking is a proxy and has not been shown to preserve
                       the 1526-step order, so the gate is deliberately loose.
  4. DEV               1526 steps x dev_seeds (~1 min each), the LEDGER 046
                       setting, averaged over seeds
  5. RANK              domination count on (val, flip2). Nothing is scalarised.

Everything is appended to a JSONL as it happens, so the run is resumable and
so a killed search loses at most one candidate.
"""

import itertools
import os
import queue
import random
import statistics
import threading
import time
import traceback

from . import evaluator, pareto, prompts
from .log import RunLog
from .sandbox import apply_knobs, check_ast, extract_knobs


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(cfg.workdir, exist_ok=True)
        os.makedirs(cfg.cand_dir, exist_ok=True)
        self.log = RunLog(cfg.log_path)
        self.arms = {}          # name -> metrics (baselines + candidates)
        self.cands = {}         # name -> {code, plan, val, flip2, ...}
        self.eps_val = cfg.eps_val
        self.eps_flip = cfg.eps_flip
        self.t0 = time.time()
        self.n_done = 0
        self.llm = None
        # Candidate names are cNNN from a counter that restarts at 1 with the
        # process. Across a resume that reuses names already in the log: the
        # board is a dict keyed by name, so a second scored arm called c004
        # silently replaces the first. The archived run got away with it (10
        # names reused, but never two SCORED arms), which is luck and not a
        # design. `load` pushes this past the highest index in the log.
        self.name_offset = 0
        self.llm = None
        prompts.configure(cfg.flip2_lo, cfg.flip2_hi)

    # ---------------------------------------------------------------- utils
    def say(self, *a):
        if not self.cfg.quiet:
            print(*a, flush=True)

    def out_of_time(self):
        return (self.cfg.hours > 0
                and time.time() - self.t0 > self.cfg.hours * 3600)

    def band(self):
        return (self.cfg.flip2_lo, self.cfg.flip2_hi)

    def board(self, limit=40):
        return pareto.table(self.arms, self.eps_val, self.eps_flip,
                            ref="none", limit=limit, band=self.band())

    # ------------------------------------------------------------ baselines
    def run_baselines(self):
        """The hand-written terms, at the same settings a candidate gets.

        They are not just context for the model: they are the anchors that make
        a domination count mean something. Without `none` on the board, a
        candidate that is merely bad has nothing to lose to.
        """
        self.say("\n=== baselines ===")
        for name, aux in evaluator.BASELINE_AUX.items():
            runs = []
            for seed in self.cfg.dev_seeds:
                m, err = evaluator.run_one(
                    self.cfg, steps=self.cfg.dev_steps, seed=seed, aux=aux)
                if err:
                    self.say(f"  baseline {name} seed {seed} FAILED: {err[:200]}")
                    continue
                runs.append(m)
            if not runs:
                continue
            self.arms[name] = evaluator.average(runs)
            a = self.arms[name]
            self.say(f"  {name:<22} val {a['val']:.6f}  flip2 {100*a['flip2']:.1f}%"
                     f"  eff {a['eff']:.1f}")
            self.log.write("baseline", name=name, aux=aux, **a)

    def measure_noise(self):
        """How far apart are two runs of the SAME loss?

        This sets the margins in the dominance test. It is the first thing the
        search does because without it every ranking below is unfalsifiable:
        a 2.4-point flip2 gap was observed between two identical losses on a
        300-step screen, which is larger than several of the effects being
        looked for.
        """
        self.say("\n=== noise floor: the same loss, different seeds ===")
        reps = []
        for seed in self.cfg.noise_seeds:
            m, err = evaluator.run_one(self.cfg, steps=self.cfg.dev_steps,
                                       seed=seed, aux="")
            if err:
                self.say(f"  seed {seed} FAILED: {err[:200]}")
                continue
            reps.append(m)
            self.say(f"  seed {seed}: val {m['val']:.6f}  "
                     f"flip2 {100*m['flip2']:.1f}%")
        if len(reps) >= 2:
            self.eps_val, self.eps_flip = pareto.noise_margins(
                reps, n_seeds=len(self.cfg.dev_seeds))
        self.say(f"  single-run sd: val "
                 f"{statistics.stdev([r['val'] for r in reps]):.6f}  flip2 "
                 f"{100*statistics.stdev([r['flip2'] for r in reps]):.2f} points"
                 if len(reps) >= 2 else "  (too few replicates)")
        self.say(f"  margins for a {len(self.cfg.dev_seeds)}-seed mean: "
                 f"val +-{self.eps_val:.6f}  "
                 f"flip2 +-{100*self.eps_flip:.2f} points")
        self.log.write("noise", n=len(reps), eps_val=self.eps_val,
                       eps_flip=self.eps_flip,
                       vals=[r["val"] for r in reps],
                       flip2s=[r["flip2"] for r in reps])

    # ------------------------------------------------------------ evaluate
    def _write_code(self, name, code):
        path = os.path.join(self.cfg.cand_dir, name + ".py")
        with open(path, "w") as f:
            f.write(code)
        return path

    def collapsed(self, m, stage="dev"):
        """Did the router stop routing?

        If it did, flip2 = 0 is true by construction -- there are no bucket
        changes left to count -- and nothing can dominate a zero, so the arm
        would hold a frontier slot for ever and be handed to every later refine
        and synthesis as a thing that worked. It is not a point on the
        trade-off curve; it is the 64-bucket architecture switched off, and
        `none` already covers "no penalty" with a router that still routes.
        """
        floor = (self.cfg.screen_min_eff if stage == "screen"
                 else self.cfg.min_eff)
        return m.get("eff", 1e9) < floor or m.get("occ", 1e9) < 2

    def screen(self, name, code, plan):
        """screen_steps at one seed. -> (metrics | None, error).

        `None` with an empty error means rejected here -- the caller must not
        spend dev time on it, but nothing crashed.
        """
        path = self._write_code(name, code)
        m, err = evaluator.run_one(self.cfg, steps=self.cfg.screen_steps,
                                   seed=self.cfg.screen_seeds[0],
                                   aux="custom=1", loss_file=path,
                                   val_cap=self.cfg.screen_val_cap)
        if err:
            return None, err
        if self.collapsed(m, stage="screen"):
            self.say(f"  {name}: dropped -- the router collapsed "
                     f"(eff {m['eff']:.1f}, {m['occ']:.0f} buckets used). "
                     f"flip2 {100*m['flip2']:.1f}% is not a result.")
            self.log.write("collapsed", name=name, stage="screen", plan=plan,
                           code=code, **m)
            return None, ""
        base = self.arms.get("none")
        if base is not None:
            sl = self.cfg.screen_slack
            if (m["val"] > base["val"] + sl * self.eps_val
                    and m["flip2"] > base["flip2"] + sl * self.eps_flip):
                self.say(f"  {name}: screened out (val {m['val']:.6f}, "
                         f"flip2 {100*m['flip2']:.1f}% -- worse on both)")
                self.log.write("screened_out", name=name, plan=plan,
                               code=code, **m)
                return None, ""
        return m, ""

    def dev(self, name, code, plan, origin):
        """dev_seeds x dev_steps, then rank and record. -> (metrics, error)."""
        path = self._write_code(name, code)
        runs = []
        for seed in self.cfg.dev_seeds:
            d, err = evaluator.run_one(self.cfg, steps=self.cfg.dev_steps,
                                       seed=seed, aux="custom=1",
                                       loss_file=path)
            if err:
                return None, err
            runs.append(d)
        m = evaluator.average(runs)
        if self.collapsed(m):
            self.say(f"  {name}: dropped -- the router collapsed "
                     f"(eff {m['eff']:.1f}, {m['occ']:.0f} buckets used). "
                     f"flip2 {100*m['flip2']:.1f}% is not a result.")
            self.log.write("collapsed", name=name, stage="dev", plan=plan,
                           code=code, **m)
            return None, ""
        return self._record(name, code, plan, origin, m), ""

    def _record(self, name, code, plan, origin, m):
        new_front = pareto.is_new_front_point(self.arms, m,
                                              self.eps_val, self.eps_flip)
        self.arms[name] = m
        self.cands[name] = {"name": name, "code": code, "plan": plan,
                            "origin": origin, **m}
        self.log.write("candidate", name=name, plan=plan, code=code,
                       origin=origin, front=new_front, **m)
        mark = "  <-- FRONT" if new_front else ""
        self.say(f"  {name:<22} val {m['val']:.6f}  flip2 {100*m['flip2']:.1f}%"
                 f"  eff {m['eff']:.1f}{mark}")
        return m

    def evaluate(self, name, code, plan):
        """screen -> dev, for one arm. Returns (metrics | None, error)."""
        m, err = self.screen(name, code, plan)
        if err or m is None:
            return None, err
        path = self._write_code(name, code)
        runs = []
        for seed in self.cfg.dev_seeds:
            d, err = evaluator.run_one(self.cfg, steps=self.cfg.dev_steps,
                                       seed=seed, aux="custom=1",
                                       loss_file=path)
            if err:
                return None, err
            runs.append(d)
        m = evaluator.average(runs)
        if self.collapsed(m):
            self.log.write("collapsed", name=name, stage="dev", plan=plan,
                           code=code, **m)
            return None, ""
        return m, ""

    def submit(self, name, code, plan, origin):
        m, err = self.evaluate(name, code, plan)
        self.n_done += 1
        if err:
            return None, err
        if m is None:
            return None, ""
        return self._record(name, code, plan, origin, m), ""

    # ------------------------------------------------------------- proposal
    def ledger_text(self, limit=14):
        """Short history for the prompt: what was tried and where it landed.

        Sorted by distance to the target band, NOT by val. Sorting by val
        (what this did) filled all 14 slots with arms at flip2 32-42%, because
        cheap-and-unstable is where the lowest val lives -- so every example
        the model saw of "work done here" was in the region the prompt tells it
        to stop proposing. The high-flip2 arms are not hidden, they are
        summarised in one line, which is all their information is worth: the
        model needs to know that end is explored, not to read fourteen of them.
        """
        if not self.cands:
            return "(nothing proposed yet this run)"
        rows = sorted(self.cands.values(),
                      key=lambda c: (pareto.band_distance(c["flip2"], *self.band()),
                                     c["val"]))
        out = ["The closest things to the target band that have been tried "
               "this run, nearest first:"]
        for c in rows[:limit]:
            out.append(f"- {c['name']}: val {c['val']:.6f}, flip2 "
                       f"{100*c['flip2']:.1f}%, eff {c['eff']:.1f} :: "
                       f"{c['plan'][:220]}")
        far = [c for c in rows[limit:]]
        if far:
            n25 = sum(1 for c in far if c["flip2"] >= 0.25)
            out.append(f"\n(plus {len(far)} more not shown, {n25} of them at "
                       f"flip2 >= 25%. That end of the plane is thoroughly "
                       f"explored and nothing there is wanted.)")
        return "\n".join(out)

    def steer_pool(self):
        """Front members worth BUILDING ON.

        Ranking is untouched: a low-eff arm keeps its place on the board and
        its frontier slot. It is only excluded as a refine/synthesis seed,
        because asking the model to deepen an arm that already uses 10 of 64
        buckets is asking it to walk the rest of the way into the gate.
        """
        front = pareto.frontier(self.arms, self.eps_val, self.eps_flip)
        pool = [self.cands[n] for n in front if n in self.cands
                and self.cands[n].get("eff", 0) >= self.cfg.steer_min_eff]
        if not pool:      # nothing clean on the front yet -- fall back
            pool = [self.cands[n] for n in front if n in self.cands]
        return pool

    def seed_front(self):
        """Submit the hand-written terms through the candidate interface.

        Two things at once. It makes them refinable -- refine and synthesis
        draw only from `cands`, so a term that exists only as a baseline can
        never be built on, and the two arms nearest the flip2 target are both
        baselines. And it checks the interface: a seed that misses its baseline
        by more than the noise margin means `ctx` cannot express that term, and
        every candidate ranked against it is then suspect.
        """
        from .seeds import SEEDS
        self.say("\n=== seeding the front with the hand-written terms ===")
        for name, code, plan, base in SEEDS:
            if name in self.cands:
                self.say(f"  {name}: already in the log, skipping")
                continue
            m, err = self.submit(name, code, plan, "seed")
            if err:
                self.say(f"  {name} FAILED: {self._one_line(err)}")
                continue
            if m is None:
                continue
            # `base` is either a baseline arm measured in THIS run, or a
            # (val, flip2) pair recorded in an earlier one. The second form is
            # the weaker check -- it compares across runs rather than within
            # one -- but it is the only check available for an arm carried over
            # from an archived board, and a seed that misses it by several
            # margins means the transcription is wrong.
            if isinstance(base, str):
                if base not in self.arms:
                    continue
                b, label = self.arms[base], base
            else:
                b = {"val": base[0], "flip2": base[1]}
                label = f"archived {base[0]:.6f}/{100*base[1]:.1f}%"
            dv, df = m["val"] - b["val"], m["flip2"] - b["flip2"]
            ok = abs(dv) <= self.eps_val and abs(df) <= self.eps_flip
            self.say(f"      vs {label}: dval {dv:+.6f} (margin "
                     f"{self.eps_val:.6f}), dflip2 {100*df:+.1f} points "
                     f"(margin {100*self.eps_flip:.1f}) -- "
                     f"{'MATCHES' if ok else 'DOES NOT MATCH'}")
            self.log.write("seed_check", name=name, baseline=label, dval=dv,
                           dflip2=df, matches=bool(ok))

    def pick_refine(self):
        """Which front member to deepen.

        Uniform over the front spends most proposals where most of the front
        IS, and most of the front sits at flip2 27-42% -- away from the target.
        Draw with a triangular weight toward the low-flip2 end instead: the
        closest arm is picked len(pool) times as often as the furthest, and
        every member stays reachable so the search can still be surprised.
        """
        pool = self.steer_pool()
        if not pool:
            return None
        # Distance to the BAND, not simply the lowest flip2: an arm already
        # inside it should be refined to make it CHEAPER, not pushed further
        # down an axis we have stopped caring about. Ties inside the band --
        # every member is at distance 0 -- are broken by val, so the cheapest
        # in-band arm is the one that gets deepened.
        pool.sort(key=lambda c: (pareto.band_distance(c["flip2"], *self.band()),
                                 c["val"]))
        w = [self.cfg.refine_decay ** i for i in range(len(pool))]
        return random.choices(pool, weights=w, k=1)[0]

    def pick_synth(self):
        """Two front members to cross.

        Sorting by val and taking both ends (what this used to do) picks along
        the axis we are NOT targeting, and usually pairs an arm with a more
        heavily weighted version of itself. The cross worth making is "the arm
        that gets closest to the flip2 target" against "the arm that costs the
        least val": two different mechanisms, each holding the half the other
        lacks.
        """
        pool = self.steer_pool()
        if len(pool) < 2:
            return None
        lo_flip = min(pool, key=lambda c: (
            pareto.band_distance(c["flip2"], *self.band()), c["val"]))
        lo_val = min(pool, key=lambda c: c["val"])
        if lo_flip["name"] == lo_val["name"]:
            rest = [c for c in pool if c["name"] != lo_flip["name"]]
            return lo_flip, min(rest, key=lambda c: c["flip2"])
        return lo_flip, lo_val

    def propose(self, step):
        """-> (plan, code, origin). One LLM round."""
        if step and step % self.cfg.synth_every == 0:
            pair = self.pick_synth()
            if pair:
                p, c, _ = self.llm.synthesize(*pair)
                return p, c, f"synth({pair[0]['name']},{pair[1]['name']})"
        if step and step % self.cfg.refine_every == 0:
            t = self.pick_refine()
            if t:
                detail = (f"val {t['val']:.6f} (no-penalty baseline "
                          f"{self.arms['none']['val']:.6f}), flip2 "
                          f"{100*t['flip2']:.1f}% (baseline "
                          f"{100*self.arms['none']['flip2']:.1f}%), effective "
                          f"buckets used {t['eff']:.1f} of 64, I(B;game) "
                          f"{t.get('I_game', 0):.2f} bits, seed-to-seed spread "
                          f"val {t.get('val_spread', 0):.6f} flip2 "
                          f"{100*t.get('flip2_spread', 0):.1f} points. "
                          f"Target band flip2 "
                          f"{100*self.cfg.flip2_lo:.0f}-"
                          f"{100*self.cfg.flip2_hi:.0f}%, so this arm is "
                          + ("INSIDE the band -- the fix worth making is one "
                             "that keeps flip2 there for LESS val, not one "
                             "that pushes flip2 lower."
                             if pareto.band_distance(t["flip2"], *self.band()) == 0
                             else f"{100*pareto.band_distance(t['flip2'], *self.band()):.1f} "
                                  f"points OUTSIDE it."))
                p, c, _ = self.llm.refine(t["name"], t["plan"], t["code"], detail)
                return p, c, f"refine({t['name']})"
        p, c, _ = self.llm.explore(self.board(), self.ledger_text(),
                                   prompts.idea_nudge(random))
        return p, c, "explore"

    # ----------------------------------------------------------------- loop
    def run(self):
        cfg = self.cfg
        if cfg.resume:
            self.load()
        if "none" not in self.arms:
            self.measure_noise()
            self.run_baselines()
        self.say("\n" + self.board())
        if cfg.seed_front:
            self.seed_front()
            self.say("\n" + self.board())

        if not (cfg.google_api_key or cfg.openrouter_api_key):
            self.say("\nNo GOOGLE_API_KEY / OPENROUTER_API_KEY -- baselines "
                     "and noise floor only. Export a key to run the search.")
            return
        from .llm import LLMClient
        self.llm = LLMClient(google_api_key=cfg.google_api_key,
                             openrouter_api_key=cfg.openrouter_api_key,
                             google_models=cfg.google_models,
                             cooldown_s=cfg.llm_cooldown_s)

        q = queue.Queue(maxsize=cfg.buffer_size)
        stop = threading.Event()

        def producer():
            """LLM calls are slow and the GPU should not wait for them."""
            for step in itertools.count():
                if stop.is_set():
                    break
                try:
                    plan, code, origin = self.propose(step)
                except Exception as e:
                    self.log.write("llm_error", err=repr(e))
                    time.sleep(5)
                    continue
                q.put((plan, code, origin))

        threads = [threading.Thread(target=producer, daemon=True)
                   for _ in range(max(1, cfg.n_explore_parallel))]
        for t in threads:
            t.start()

        seen = 0
        try:
            while seen < cfg.max_candidates and not self.out_of_time():
                try:
                    plan, code, origin = q.get(timeout=60)
                except queue.Empty:
                    continue
                seen += 1
                self.say(f"\n--- candidate {seen}/{cfg.max_candidates} "
                         f"[{origin}] ({(time.time()-self.t0)/3600:.1f} h) ---")
                self.say("  " + plan.replace("\n", "\n  ")[:600])
                self.handle(seen, plan, code, origin)
                if seen % 10 == 0:
                    self.say("\n" + self.board())
        except KeyboardInterrupt:
            self.say("\ninterrupted -- every candidate is already in the log")
        finally:
            stop.set()
        self.say("\n=== final board ===\n" + self.board(limit=60))
        self.report_front()

    @staticmethod
    def _one_line(err):
        return (err.splitlines()[-1][:160] if err.strip() else err[:160])

    def try_fix(self, name, plan, body, err, origin):
        """One repair round on a candidate that would not run. -> (m, err)."""
        for _ in range(self.cfg.fix_attempts):
            try:
                p2, c2, _ = self.llm.fix(plan, body, err[-1500:])
            except Exception as e:
                self.log.write("llm_error", err=repr(e))
                return None, err
            gate = check_ast(c2)
            if gate:
                self.log.write("ast_reject", name=name + "f", plan=p2,
                               code=c2, err=gate)
                return None, err
            m, err2 = self.screen(name + "f", c2, p2)
            if err2:
                # The repair did not run either. Log it -- an unlogged failure
                # is one the next run cannot learn from, and a fix that repeats
                # the original mistake is exactly what we want to see.
                self.say(f"  {name}f crashed too: {self._one_line(err2)}")
                self.log.write("crash", name=name + "f", plan=p2, code=c2,
                               err=err2)
                return None, err2
            if m is None:
                return None, ""
            return self.dev(name + "f", c2, p2, origin + "+fix")
        return None, err

    def handle(self, seen, plan, code, origin):
        cfg = self.cfg
        self.n_done += 1
        err = check_ast(code)
        if err:
            self.say(f"  rejected by AST gate: {err}")
            self.log.write("ast_reject", plan=plan, code=code, err=err)
            return
        knobs = extract_knobs(code)
        variants = [{}]
        if knobs:
            keys = list(knobs)
            grid = list(itertools.product(*(knobs[k] for k in keys)))
            variants = [dict(zip(keys, g)) for g in grid[:cfg.max_variants]]
            self.say(f"  KNOBS -> {len(variants)} variants: {keys}")

        # -- screen every variant: one seed, screen_steps, ~a tenth of a dev --
        survivors = []                       # [(name, body, screen metrics)]
        for i, choice in enumerate(variants):
            tag = ("-" + "_".join(f"{k}{v:g}" for k, v in choice.items())
                   if choice else "")
            name = f"c{self.name_offset + seen:03d}{tag}"
            body = apply_knobs(code, choice)
            m, err = self.screen(name, body, plan)
            if err:
                # Every variant shares the mechanism and differs only in a
                # constant, so if the first one cannot run none of them can.
                # Ask for one repair and drop the family either way.
                if i == 0:
                    self.say(f"  {name} crashed: {self._one_line(err)}")
                    self.log.write("crash", name=name, plan=plan, code=body,
                                   err=err)
                    self.try_fix(name, plan, body, err, origin)
                return
            if m is not None:
                survivors.append((name, body, m))
        if not survivors:
            return

        # -- spend the dev budget on the most promising of them --------------
        # The screen is one seed at screen_steps and has NOT been shown to
        # preserve the dev order, so it is used ONLY to allocate dev time
        # inside this one parametric family -- never to rank an arm or to put
        # anything on the board. Every arm that reaches the frontier still got
        # the full dev_seeds x dev_steps treatment. Ordering is by the same
        # domination count used everywhere else, so no exchange rate between
        # val and flip2 is invented here either.
        if len(survivors) > cfg.dev_top_k:
            fam = {n: m for n, _, m in survivors}
            r = pareto.rank_all(fam, self.eps_val, self.eps_flip)
            survivors.sort(key=lambda t: (r[t[0]], t[2]["val"]))
            dropped = [n for n, _, _ in survivors[cfg.dev_top_k:]]
            survivors = survivors[:cfg.dev_top_k]
            self.say(f"  dev budget -> {', '.join(n for n, _, _ in survivors)}"
                     f"  (screened but not developed: {', '.join(dropped)})")
        for name, body, _ in survivors:
            self.dev(name, body, plan, origin)

    def report_front(self):
        front = pareto.frontier(self.arms, self.eps_val, self.eps_flip)
        self.say(f"\nPareto frontier -- {len(front)} arms nothing beats on both "
                 f"axes at once:")
        for n in front:
            a = self.arms[n]
            self.say(f"  {n:<22} val {a['val']:.6f}  flip2 {100*a['flip2']:.1f}%"
                     f"  eff {a['eff']:.1f}")
            if n in self.cands:
                self.say(f"      {self.cands[n]['plan'][:300]}")
        self.say(f"\nfull log: {self.cfg.log_path}"
                 f"\ncandidate sources: {self.cfg.cand_dir}")

    def load(self):
        """Rebuild the board from the JSONL so a killed run continues."""
        import json
        if not os.path.exists(self.cfg.log_path):
            return
        n = 0
        with open(self.cfg.log_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                nm = r.get("name", "")
                if nm.startswith("c") and nm[1:4].isdigit():
                    self.name_offset = max(self.name_offset, int(nm[1:4]))
                if r["kind"] == "noise":
                    self.eps_val, self.eps_flip = r["eps_val"], r["eps_flip"]
                elif r["kind"] in ("baseline", "candidate"):
                    m = {k: v for k, v in r.items()
                         if k not in ("t", "kind", "name", "plan", "code",
                                      "origin", "front", "aux")}
                    if r["kind"] == "candidate" and self.collapsed(m):
                        # Logged before the collapse gate existed. Reloading it
                        # would hand a permanent frontier slot back to a router
                        # that does not route.
                        self.say(f"  dropped on resume: {r['name']} -- router "
                                 f"collapsed (eff {m.get('eff', 0):.1f})")
                        continue
                    self.arms[r["name"]] = m
                    if r["kind"] == "candidate":
                        self.cands[r["name"]] = {"name": r["name"],
                                                 "code": r["code"],
                                                 "plan": r["plan"],
                                                 "origin": r.get("origin", ""),
                                                 **m}
                    n += 1
        self.say(f"resumed: {n} arms from {self.cfg.log_path}")
