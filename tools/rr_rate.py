"""Turn a round-robin crosstable into one rating per net, with an interval.

`chess elo` cannot do this -- it anchors one engine against one opponent, and
LEDGER 004's multi-opponent number was combined by hand (STATE.md). This is
that combination, written down.

The fit is weighted least squares on the pairwise Elo differences rather than a
maximum-likelihood Bradley-Terry, for one reason: the weights then come from
the ACTUAL variance of the game scores, not from the binomial variance a
win/loss model assumes. These matches are 55-60% draws, and a draw-heavy
pairing carries far more information per game than a coin flip does. Using the
binomial would inflate every interval by roughly 1/sqrt(1 - draw rate) and make
a real difference look unresolved.

Anchoring: ratings are identified only up to a constant, so one net is pinned
at 0 and every other rating is read relative to it. Its own interval is 0 by
construction -- that is the anchor, not a claim that it is measured perfectly.

Input is TSV: name_a, name_b, wins_a, draws, losses_a, games.
"""

import argparse
import math

import numpy as np

C = math.log(10.0) / 400.0     # d(logistic p) / d(Elo) = C * p * (1 - p)


def pairing_elo(w, d, l):
    """Elo difference and its standard error, from one pairing's W/D/L.

    Score variance is measured, not assumed: a game scores 1, 0.5 or 0, so
    var = E[s^2] - E[s]^2 over the three observed counts.
    """
    n = w + d + l
    f = (w + 0.5 * d) / n
    if f <= 0.0 or f >= 1.0:
        return None, None, f, n           # unresolvable; caller drops it
    ex2 = (w * 1.0 + d * 0.25) / n
    var = max(ex2 - f * f, 1e-12) / n     # variance of the MEAN score
    elo = -400.0 * math.log10(1.0 / f - 1.0)
    # delta method: d(elo)/d(f) = 400 / (ln10 * f * (1-f))
    se = math.sqrt(var) * 400.0 / (math.log(10.0) * f * (1.0 - f))
    return elo, se, f, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--anchor", default=None, help="net pinned at 0")
    a = ap.parse_args()

    rows = []
    for line in open(a.results):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        rows.append((p[0], p[1], int(p[2]), int(p[3]), int(p[4])))
    if not rows:
        raise SystemExit("no pairings in %s" % a.results)

    names = []
    for x, y, *_ in rows:
        for n in (x, y):
            if n not in names:
                names.append(n)
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    anchor = a.anchor if a.anchor in idx else names[0]

    # --- crosstable ---------------------------------------------------------
    print("pairings")
    print("%-14s %-14s %6s %6s %6s %7s %9s %8s" %
          ("net", "vs", "+", "=", "-", "score", "elo", "+-"))
    used = []
    for x, y, w, d, l in rows:
        elo, se, f, n = pairing_elo(w, d, l)
        if elo is None:
            print("%-14s %-14s %6d %6d %6d %6.1f%%   UNRESOLVED (0%% or 100%%)"
                  % (x, y, w, d, l, 100 * f))
            continue
        print("%-14s %-14s %6d %6d %6d %6.1f%% %+9.1f %8.1f"
              % (x, y, w, d, l, 100 * f, elo, se))
        used.append((idx[x], idx[y], elo, se))
    if not used:
        raise SystemExit("no usable pairings")

    # --- weighted least squares on r_i - r_j = elo_ij ------------------------
    A = np.zeros((k, k))
    b = np.zeros(k)
    for i, j, elo, se in used:
        wt = 1.0 / (se * se)
        A[i, i] += wt; A[j, j] += wt
        A[i, j] -= wt; A[j, i] -= wt
        b[i] += wt * elo; b[j] -= wt * elo

    # Pin the anchor by deleting its row/column, then re-insert a zero.
    p = idx[anchor]
    keep = [i for i in range(k) if i != p]
    As = A[np.ix_(keep, keep)]
    r = np.zeros(k)
    cov = np.zeros((k, k))
    sol = np.linalg.solve(As, b[keep])
    inv = np.linalg.inv(As)
    for m, i in enumerate(keep):
        r[i] = sol[m]
        for m2, j in enumerate(keep):
            cov[i, j] = inv[m, m2]

    # --- goodness of fit ----------------------------------------------------
    # A round robin has more pairings than free ratings, so the model can be
    # WRONG and say so. chi2/dof far above 1 means one rating per net does not
    # describe these games -- non-transitivity, or an arm whose strength
    # depends on the opponent. Read it before reading the ratings.
    dof = len(used) - (k - 1)
    chi2 = sum(((r[i] - r[j] - elo) / se) ** 2 for i, j, elo, se in used)

    order = sorted(range(k), key=lambda i: -r[i])
    print()
    print("ratings (anchor %s = 0, +- is one standard error)" % anchor)
    print("%-14s %9s %8s   %s" % ("net", "elo", "+-", "95% interval"))
    for i in order:
        se = math.sqrt(max(cov[i, i], 0.0))
        if i == idx[anchor]:
            print("%-14s %9.1f %8s   %s" % (names[i], 0.0, "-", "(anchor)"))
        else:
            print("%-14s %+9.1f %8.1f   [%+.0f, %+.0f]"
                  % (names[i], r[i], se, r[i] - 1.96 * se, r[i] + 1.96 * se))
    print()
    if dof > 0:
        print("fit: chi2 %.1f on %d dof (%.2f per dof)%s"
              % (chi2, dof, chi2 / dof,
                 "  -- one rating per net does NOT fit these games"
                 if chi2 / dof > 2.5 else ""))
    else:
        print("fit: %d pairings for %d free ratings, nothing left over to test"
              % (len(used), k - 1))


if __name__ == "__main__":
    main()
