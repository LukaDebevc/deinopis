**025 · Absolute rating after the quadratic eval: about 2969 on CCRL Blitz** ·
2026-08-27
`tools/ladder.sh run --games 200 --tc 10+0.1 --concurrency 6`, 894M-position
net, four new anchors added to reach the range the eval put us in.

| opponent | CCRL | score | diff | implied |
|---|---|---|---|---|
| Tantabus 2.0.0 | 2555 | 83.5% | +281.7 | 2837 |
| Blunder 8.5.5 | 2664 | 87.0% | +330.2 | 2994 |
| byte-knight 4.0.0 | 2859 | 67.0% | +123.0 | 2982 |
| 4ku 5.1 | 3057 | 41.0% | -63.2 | 2994 |
| Inanis 1.6.0 | 3087 | 36.5% | -96.2 | 2991 |
| Simbelmyne 1.10.0 | 3238 | 18.75% | -254.7 | 2983 |

**Result: about 2971 +/- 43 on the CCRL Blitz scale** (6 anchors, chi2/dof =
4.05, error scaled x2.01), against 2554 +/- 56 for the pre-eval build.
**Five anchors spanning 683 Elo agree within 12 points** (2982-2994), and they
cross from 87% to 18.75% score -- so this is not the ladder saturating, which a
one-sided ladder could not have shown. Simbelmyne at ~250 Elo above us is where
a logistic model would break down if it were going to; it lands dead centre.
(Simbelmyne's row is the re-run after the LEDGER 024 parser fix. Before it, the
same opponent implied **3217** -- a 234-Elo error from one bug, which would have
moved the combination to 3017 while inflating chi2, so the contradiction would
have read as ordinary anchor disagreement.) Tantabus sits 145 Elo low and is the
sole source of the chi2 -- five anchors now bracket it on both sides and agree
with each other. It is kept: dropping an anchor for disagreeing is how a rating
gets manufactured, and the x2.01 scaling already prices it in.
**Do not quote 2989.** Excluding Tantabus gives 2989 +/- 23 at chi2/dof =
**0.05**, which is four estimates agreeing *far better* than their error bars
-- a sign the combine's error model is conservative (it symmetrises each match
interval on its wider side, then adds CCRL's error in quadrature), not a sign
of precision.
**Conditions, as required:** 10+0.1, 64 MB hash, concurrency 6, 200 games per
opponent, colour-reversed pairs, built-in book, no tablebases. On top of the
interval sits a systematic offset from CCRL's own conditions. It is a scale,
not a certificate.
**New ladder anchors** (exact CCRL-rated tags, each verified by its UCI id
string): byte-knight 4.0.0 (2859), 4ku 5.1 (3057), Inanis 1.6.0 (3087),
Simbelmyne 1.10.0 (3238). OliThink 5.10.5 and DiscoCheck 5.2.1 were rejected --
no matching tag and no repository respectively, and an untagged build has no
published rating, which is the same as having no anchor.


---
