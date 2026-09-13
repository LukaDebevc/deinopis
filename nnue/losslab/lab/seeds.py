"""The hand-written terms, written through the candidate interface.

Two jobs. First, a check: if `ginfo` expressed here does not reproduce the
`ginfo=1e-3` baseline, then the ctx a candidate sees cannot express the best
arm on the board and no ranking below it means anything. Second, and the
reason this exists at all: refine and synthesis draw only from `cands`, so a
term that lives only as a BASELINE can never be built on. The two arms that
reach the low-flip2 region are both baselines, which left the search unable to
improve on its own best result.

Transcribed from train.py `_aux_terms`; the weights match BASELINE_AUX.
"""

# -I(B;game): bucket usage spread ACROSS games, concentrated WITHIN one.
GINFO = '''def penalty(ctx):
    """-I(B;game): entropy of bucket usage within a game, minus across the pool.
    Low within a game and high across games means the bucket tracks something
    slow and game-level rather than the current position."""
    torch = ctx["torch"]
    tp, gid, cnt = ctx["tp"], ctx["gid"], ctx["cnt"]
    mall = tp.mean(0)
    h_pool = -(mall * (mall + 1e-9).log()).sum()
    sums = torch.zeros(cnt.numel(), tp.shape[1],
                       device=tp.device, dtype=tp.dtype)
    sums.index_add_(0, gid, tp)
    c = cnt.to(tp.dtype)
    mg = sums / c.unsqueeze(1)
    hg = -(mg * (mg + 1e-9).log()).sum(1)
    h_within = (hg * c).sum() / c.sum()
    return W * (h_within - h_pool)


W = 1e-3
'''

# Squared change in the bucket posterior across a same-side two-ply move.
TV = '''def penalty(ctx):
    """Squared movement of the bucket posterior over a same-side move pair."""
    tp, pi = ctx["tp"], ctx["pair"]
    d = tp[pi + 2] - tp[pi]
    return W * d.pow(2).sum(1).mean()


W = 3e-3
'''

# The best arm on the board: both terms at the weights that reached flip2 16.7%.
GINFO_TV = '''def penalty(ctx):
    """ginfo=3e-3 + tv=3e-2 -- the best low-flip2 arm measured so far.
    flip2 16.7% for +3.08% val, and it holds eff at 51 of 64 buckets."""
    torch = ctx["torch"]
    tp, gid, cnt, pi = ctx["tp"], ctx["gid"], ctx["cnt"], ctx["pair"]
    mall = tp.mean(0)
    h_pool = -(mall * (mall + 1e-9).log()).sum()
    sums = torch.zeros(cnt.numel(), tp.shape[1],
                       device=tp.device, dtype=tp.dtype)
    sums.index_add_(0, gid, tp)
    c = cnt.to(tp.dtype)
    mg = sums / c.unsqueeze(1)
    hg = -(mg * (mg + 1e-9).log()).sum(1)
    h_within = (hg * c).sum() / c.sum()
    d = tp[pi + 2] - tp[pi]
    return WG * (h_within - h_pool) + WT * d.pow(2).sum(1).mean()


WG = 3e-3
WT = 3e-2
'''

# --- carried over from the archived run (losslab_run.jsonl.*.archive) --------
# The board was cleared on 2026-09-02: the ledger and the table were both
# dominated by ~30 arms at flip2 30-42%, a region the search has stopped
# wanting. These two are the arms from that run worth keeping -- both reach
# under 15% flip2, which nothing else has. They come back as CANDIDATES so they
# can be refined, and their recorded numbers are the reproduction check.

# ginfo+tv with H(B|game) computed separately for even and odd plies. Same
# weights as the ginfo=3e-3,tv=3e-2 baseline, so the split is the only change
# -- and it moved flip2 16.7% -> 13.7% (3.0 points, margin 1.85) for +1.07%
# more val, holding eff at 49.7. The best arm the search has produced.
SPLIT_GINFO_TV = r'''def penalty(ctx):
    """
    Fixed ginfo+tv: Splits game-level entropy by side to avoid 
    penalising the side-to-move board mirror.
    """
    torch = ctx["torch"]
    tp, gid, cnt, pi = ctx["tp"], ctx["gid"], ctx["cnt"], ctx["pair"]

    # Global pool entropy (H(B)) remains calculated on the whole population
    mall = tp.mean(0)
    h_pool = -(mall * (mall + 1e-9).log()).sum()

    def get_h_within(t_slice, g_slice):
        # Use local logic to calculate conditional entropy H(B|game)
        # We determine current unique games and counts in the slice
        u_gid = g_slice.unique()
        num_g = u_gid.numel()

        # Sum posteriors per game
        sums = torch.zeros(num_g, t_slice.shape[1], device=t_slice.device, dtype=t_slice.dtype)
        # Map global gid to local 0..G-1 index for index_add
        # Faster: use a mapping if G is large, but we can just use scatter/index_add
        # since we are operating on slices. 
        # The simplest way to preserve gradient and use gid is a loop or group-mean.
        # Given T=8192, a small loop or index_add on reshaped ids is fine.

        # Create local indices for the split gid
        # To handle splitting, we create a mapping from old gid to new unique ids
        sorted_gid, indices = torch.sort(g_slice)
        # Use a simple count to group. Since tp was game-sequential, 
        # we can calculate counts.
        diff = torch.cat([torch.tensor([1], device=t_slice.device), sorted_gid[1:] != sorted_gid[:-1]])
        local_gid = torch.cumsum(diff, 0) - 1

        sums.index_add_(0, local_gid, t_slice[indices])

        # Game counts
        g_cnt = torch.bincount(local_gid).to(t_slice.dtype)
        mg = sums / g_cnt.unsqueeze(1)
        hg = -(mg * (mg + 1e-9).log()).sum(1)

        return (hg * g_cnt).sum() / g_cnt.sum()

    # Split tape by player (even/odd plies)
    h_white = get_h_within(tp[0::2], gid[0::2])
    h_black = get_h_within(tp[1::2], gid[1::2])
    h_within_split = (h_white + h_black) / 2

    # tv penalty: squared L2 distance on posterior change (already same-side)
    d = tp[pi + 2] - tp[pi]
    tv = d.pow(2).sum(1).mean()

    return WG * (h_within_split - h_pool) + WT * tv

WG = 3e-3
WT = 3e-2'''

# Per-BIT mutual information with the game, each bit weighted by how decisive
# it is. Reaches 14.8% flip2 but at eff 12.5 -- it buys stability by shrinking
# the router, the failure mode the brief warns about. Kept as the clearest
# measured EXAMPLE of that trade, not as a good arm: steer_min_eff keeps it out
# of the refine pool.
PERBIT_GINFO = r'''KNOBS = {"W": [1e-3, 3e-3, 1e-2]}

def penalty(ctx):
    torch = ctx['torch']
    F = ctx['F']
    tz = ctx['tz']       # (T, 6)
    gid = ctx['gid']     # (T,)
    cnt = ctx['cnt']     # (G,)

    # Soft probabilities for each bit
    p = torch.sigmoid(tz) # (T, 6)

    # Marginal probability of each bit being 1 across the whole tape
    p_marginal = p.mean(dim=0) # (6,)

    # Probability of each bit being 1 per game (tape run)
    # Use index_add to avoid explicit loops over games
    G = cnt.shape[0]
    game_sums = torch.zeros(G, 6, device=tz.device).index_add_(0, gid, p)
    p_game = game_sums / cnt.unsqueeze(1) # (G, 6)

    def binary_entropy(prob):
        # Numerical stability for log(0)
        prob = torch.clamp(prob, 1e-6, 1.0 - 1e-6)
        return -prob * torch.log(prob) - (1.0 - prob) * torch.log(1.0 - prob)

    # H(bit_j) : Overall uncertainty of bit j
    h_marginal = binary_entropy(p_marginal) # (6,)

    # H(bit_j | game) : Average uncertainty of bit j given the game context
    # we take the mean over games G
    h_conditional = binary_entropy(p_game).mean(dim=0) # (6,)

    # I(bit_j; game) = H(bit_j) - H(bit_j | game)
    # Higher mutual information means the bit is a stable identifier for the game.
    mutual_info = h_marginal - h_conditional # (6,)

    # Gating: prioritize stability for bits that are "decisive" (far from the sign-flip boundary)
    # Weights are based on the magnitude of the logits.
    weights = torch.sigmoid(tz.abs()).mean(dim=0) # (6,)

    # We want to maximize mutual information (minimize -I) 
    # gated by the decisiveness of the bit.
    # If the router is collapsed (z=0 or constant), mutual_info -> 0.
    term = (weights * (-mutual_info)).sum()

    return W * term

# --- knob binding, appended by the search ---
W = 0.003'''

# (name, code, plan, what it must reproduce -- a baseline arm name, or a
#  (val, flip2) pair measured in an earlier run)
SEEDS = [
    ("seed_ginfo_tv", GINFO_TV,
     "ginfo=3e-3 + tv=3e-2 through the candidate interface. The best low-flip2 "
     "arm on the board: flip2 16.7% at +3.08% val, holding eff 51 of 64. "
     "Must reproduce the `ginfo=3e-3,tv=3e-2` baseline.",
     "ginfo=3e-3,tv=3e-2"),
    ("seed_split_ginfo_tv", SPLIT_GINFO_TV,
     "ginfo=3e-3 + tv=3e-2, but the within-game bucket entropy is computed "
     "separately for even and odd plies and averaged, so the term does not "
     "charge for the side-to-move board mirror. Measured 13.7% flip2 at "
     "+4.15% val, eff 49.7 -- the lowest flip2 of any arm that still routes.",
     (0.026876, 0.1371)),
    ("seed_perbit_ginfo", PERBIT_GINFO,
     "Per-bit I(bit_j; game), each bit weighted by sigmoid(|z_j|) so decisive "
     "bits are pushed hardest. Measured 14.8% flip2 at +5.45% val but eff "
     "12.5: it reaches the low-flip2 region by shrinking the router, the "
     "failure mode the brief warns about. On the board as evidence for that "
     "warning, not as a thing to build on.",
     (0.027213, 0.1482)),
    ("seed_ginfo", GINFO,
     "-I(B;game) at 1e-3 through the candidate interface. Must reproduce the "
     "`ginfo=1e-3` baseline: flip2 24.3% at +2.49% val, eff 53.",
     "ginfo=1e-3"),
    ("seed_tv", TV,
     "Posterior movement over a move pair at 3e-3, through the candidate "
     "interface. Must reproduce the `tv=3e-3` baseline: the only free win so "
     "far, -0.55% val AND flip2 38.4%.",
     "tv=3e-3"),
]
