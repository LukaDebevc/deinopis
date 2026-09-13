#!/usr/bin/env python3
"""LLM-driven search for a router auxiliary loss, ranked by Pareto dominance
on (validation loss, flip2).

  python run.py --baselines-only      # noise floor + the hand-written terms
  python run.py --hours 24            # let it search for a day
  python run.py --resume --hours 12   # continue an existing log
  python run.py --smoke               # one hand-written candidate, end to end

Set GOOGLE_API_KEY (preferred) or OPENROUTER_API_KEY to enable the LLM loop.
Without one it measures the baselines, prints the board, and stops -- which is
still the useful half, because that is where the noise margins come from.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab.config import RunConfig


def main():
    ap = argparse.ArgumentParser(description="router loss search")
    ap.add_argument("--candidates", type=int, default=None)
    ap.add_argument("--hours", type=float, default=0.0,
                    help="stop after this much wall clock (0 = only the "
                         "candidate budget stops it)")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="baselines off; run one known-good candidate through "
                         "the whole path to check the plumbing")
    ap.add_argument("--quick", action="store_true",
                    help="short arms and one seed -- for checking the loop, "
                         "NOT for a number anyone quotes")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed-front", action="store_true",
                    help="submit the hand-written terms through the candidate "
                         "interface first, so they can be refined -- and so a "
                         "mismatch against their baselines is caught early")
    ap.add_argument("--log", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = RunConfig()
    if args.log:
        cfg.log_path = args.log
    if args.device:
        cfg.device = args.device
    if args.resume:
        cfg.resume = True
    if args.seed_front:
        cfg.seed_front = True
    if args.hours:
        cfg.hours = args.hours
    if args.candidates is not None:
        cfg.max_candidates = args.candidates
    if args.baselines_only:
        cfg.max_candidates = 0
    if args.quick:
        cfg.dev_steps = 300
        cfg.dev_seeds = (0,)
        cfg.noise_seeds = (0, 1)
        cfg.dev_val_cap = 500_000

    from lab.search import Controller
    c = Controller(cfg)

    if args.smoke:
        # `tv` written through the candidate interface. It must land near the
        # tv=3e-3 baseline; if it does not, the plumbing is wrong and no
        # ranking downstream means anything.
        code = ('def penalty(ctx):\n'
                '    tp, pi = ctx["tp"], ctx["pair"]\n'
                '    d = tp[pi + 2] - tp[pi]\n'
                '    return 3e-3 * d.pow(2).sum(1).mean()\n')
        c.measure_noise()
        c.run_baselines()
        c.submit("smoke_tv", code, "tv=3e-3 through the candidate path", "smoke")
        print("\n" + c.board())
        return

    c.run()


if __name__ == "__main__":
    main()
