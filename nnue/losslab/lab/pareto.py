"""Ranking by Pareto dominance on (val loss, flip2) -- and nothing else.

There is no scalarisation anywhere in this file on purpose. A weighted sum of
val and flip2 would silently pick an exchange rate between "eval accuracy" and
"accumulator refreshes", and that rate is exactly the thing the search is
supposed to inform rather than assume. So the only question asked of a
candidate is: **how many other arms beat it on BOTH numbers at once?**

    dominates(A, B)  <=>  A.val < B.val - eps_val  and  A.flip2 < B.flip2 - eps_flip

`rank` = that count. rank 0 = on the frontier: nothing beats it outright, so it
is a real option and the choice between it and the other rank-0 arms is a
judgement about what a refresh costs, not a measurement.

The margins are not decoration. Two runs of the SAME loss differing only in
seed came out 2.4 points of flip2 apart on a 300-step screen. Without eps the
front would be full of arms that are only ahead by an RNG draw, and the search
would spend the day chasing its own noise. `noise_margins` sizes eps from
replicates of the no-penalty baseline, so "better" means "further apart than
the same loss run twice".
"""

import math
import statistics


def noise_margins(reps, n_seeds=1, k=2.0):
    """(eps_val, eps_flip) for comparing two arms, each an average of `n_seeds`.

    The replicates give the SINGLE-RUN standard deviation s. The arms actually
    being compared are means of n runs, so each has standard error s/sqrt(n),
    and their difference has sqrt(2) times that. Using the raw single-run s as
    the margin (the obvious mistake) makes the test far too strict at n>1 and
    hides real effects; using zero makes every seed look like a discovery.

        eps = k * s * sqrt(2 / n)

    k = 2, so "better" means about two standard errors apart. Falls back to a
    wide margin on fewer than two replicates -- an unmeasured noise floor
    should make the search more conservative, not less.
    """
    vals = [r["val"] for r in reps]
    flips = [r["flip2"] for r in reps]
    if len(vals) < 2:
        return 0.0005, 0.02
    scale = k * math.sqrt(2.0 / max(1, n_seeds))
    return scale * statistics.stdev(vals), scale * statistics.stdev(flips)


def band_distance(flip2, lo, hi):
    """How far outside the target band an arm is. 0 = inside it.

    Used ONLY to steer where proposals are spent -- which arm gets refined,
    which pair gets crossed. It never enters `dominates`, because turning "how
    close to the band" into part of the score would be the scalarisation this
    file exists to avoid.
    """
    if flip2 < lo:
        return lo - flip2
    if flip2 > hi:
        return flip2 - hi
    return 0.0


def dominates(a, b, eps_val, eps_flip):
    """Strictly better on BOTH axes, by more than the noise."""
    return (a["val"] < b["val"] - eps_val
            and a["flip2"] < b["flip2"] - eps_flip)


def rank_all(arms, eps_val, eps_flip):
    """{name: n_dominators}. 0 = on the Pareto frontier."""
    out = {}
    for name, a in arms.items():
        out[name] = sum(1 for other, b in arms.items()
                        if other != name and dominates(b, a, eps_val, eps_flip))
    return out


def frontier(arms, eps_val, eps_flip):
    """Front members, ordered by val (so the table reads as a trade-off curve)."""
    r = rank_all(arms, eps_val, eps_flip)
    return sorted((n for n, c in r.items() if c == 0),
                  key=lambda n: arms[n]["val"])


def is_new_front_point(arms, cand, eps_val, eps_flip):
    """Does `cand` reach a corner of the plane no existing arm reaches?"""
    return not any(dominates(a, cand, eps_val, eps_flip) for a in arms.values())


def table(arms, eps_val, eps_flip, ref=None, limit=40, band=None):
    """The leaderboard the LLM is shown. Front first, then by how badly beaten.

    `ref` names the arm the % column is relative to (normally the no-penalty
    router), because an absolute loss of 0.0256 means nothing to a reader who
    has not seen the other numbers.
    """
    r = rank_all(arms, eps_val, eps_flip)
    # Frontier first, then by how badly beaten -- but WITHIN a rank, order by
    # distance to the target band rather than by val. Ordering by val puts the
    # cheap-and-unstable corner at the top, which is the corner `none` already
    # occupies; with 27 arms on the frontier that pushed every arm near the
    # target off the bottom of a truncated table. Ranking is unaffected: this
    # is the order rows are PRINTED in, not the score.
    if band:
        order = sorted(arms, key=lambda n: (r[n],
                                            band_distance(arms[n]["flip2"], *band),
                                            arms[n]["val"]))
    else:
        order = sorted(arms, key=lambda n: (r[n], arms[n]["val"]))
    rv = arms[ref]["val"] if ref in arms else None
    rf = arms[ref]["flip2"] if ref in arms else None
    head = (f"{'arm':<26} {'val':>9} {'dval%':>7} {'flip2':>7} {'dflip2':>7} "
            f"{'eff':>5} {'I(B;g)':>7} {'beaten by':>9} {'band':>5}")
    lines = [head, "-" * len(head)]
    for n in order[:limit]:
        a = arms[n]
        dv = f"{100 * (a['val'] / rv - 1):+.2f}" if rv else "     -"
        df = f"{100 * (a['flip2'] - rf):+.1f}" if rf is not None else "    -"
        inb = ("  yes" if band and band_distance(a["flip2"], *band) == 0
               else "    -")
        lines.append(f"{n[:26]:<26} {a['val']:>9.6f} {dv:>7} "
                     f"{100 * a['flip2']:>6.1f}% {df:>7} "
                     f"{a.get('eff', 0):>5.1f} {a.get('I_game', 0):>7.2f} "
                     f"{r[n]:>9} {inb:>5}")
    lines.append(f"(margins: val +-{eps_val:.6f}, flip2 +-{100 * eps_flip:.2f} "
                 f"points -- 2 sd of the same loss run at different seeds. "
                 f"'beaten by' 0 = on the frontier.)")
    if band:
        n_in = sum(1 for a in arms.values()
                   if band_distance(a["flip2"], *band) == 0)
        lines.append(f"(band = the target region, flip2 {100*band[0]:.0f}-"
                     f"{100*band[1]:.0f}%. {n_in} of {len(arms)} arms are in "
                     f"it. Ranking does NOT use the band -- it only decides "
                     f"where the next proposals are spent.)")
    return "\n".join(lines)
