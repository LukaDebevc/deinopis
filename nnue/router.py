"""The learned bucket rule, the game tape, and everything that reports on them.

MODULE STATE: TAPE_NCHUNK, TAPE_CLEN, TAPE_STRIDE, TAPE_TRAIN_HI and
CUSTOM_LOSS are set from the CLI by train.main(), which assigns them as
ATTRIBUTES of this module (`router.TAPE_STRIDE = ...`). A `global` statement
in train.py would bind train.py's own name and leave these at their
defaults -- silently, with every run still finishing."""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import NFEAT, PAD   # noqa: E402


from common import bag, loss_fn   # noqa: E402
from features import Extras   # noqa: E402
from movegraph import graph   # noqa: E402
from loader import GpuUnpacker   # noqa: E402


def parse_aux(spec):
    """'info=0.01,margin=0.02:1.5,swing=1e-4' -> {name: (weight, param)}."""
    out = {}
    for part in filter(None, (x.strip() for x in spec.split(","))):
        name, _, rest = part.partition("=")
        w, _, p = rest.partition(":")
        out[name] = (float(w), float(p) if p else None)
    return out



TAPE_NCHUNK, TAPE_CLEN = 128, 64      # overridden by --tape
TAPE_STRIDE = 0                       # overridden by --tape-stride; 0 = consecutive
TAPE_TRAIN_HI = 0.8                   # the flip2 probe reads the rest
CUSTOM_LOSS = None                    # set by --router-loss-file


class _Ctx(dict):
    """Candidate ctx: `ctx["tz"]` and `ctx.tz` both work.

    Proposals reach for attribute access about as often as indexing, and an
    AttributeError raised at the first training step costs a whole arm.
    """

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


class GameTape:
    """Contiguous runs of ONE game, for the two losses that need real time.

    `swing` prices a bucket change on an empty-board move graph: no legality,
    no captures, every piece equally likely, no notion of a game. The corpus
    has the real thing. tono_games.data is UNFILTERED and stored in game
    order -- byte 29 bit 1 marks a game start, bytes 30-31 an LE u16 ply -- so
    a contiguous slice of records IS a run of consecutive plies and the game
    boundary is read off rather than inferred.

    Two quantities become available that a shuffled batch cannot express:

      * H(B | one game), hence I(B;game) = H(B) - E_g H(B|g).  That is "spread
        out across games, constant inside one" written as a single number.
        `bal` was the previous stand-in and it is the WRONG target: it pushes
        the marginal to uniform over all 64 buckets (measured: eff 46.6 -> 63.5)
        and costs 2.1-2.9% val, because uniform usage is not conditional
        structure.  king6, the best rule on record, uses eff 20.6 buckets.

      * p(b | x_t) against p(b | x_{t+2}) on the SAME side to move -- the
        actual distribution shift one real move causes, on real moves.

    The tape is UNSUPERVISED: it reads features only, never score or result,
    so it cannot leak a label even where it overlaps the training corpus's
    validation tail.
    """

    PATH = "data/tono_games.data"

    def __init__(self, device, nchunk=128, clen=64, path=None, lo=0.0, hi=1.0,
                 stride=0):
        """`lo`/`hi` cut the record range to a fraction of the file.

        A penalty that TRAINS on this tape and is then SCORED on it is scored on
        its own training set: flip2 could fall because the router memorised
        these games. Training reads [0, 0.8) and the flip2 probe reads [0.8, 1),
        so the number that ranks a candidate comes from games its loss never
        touched. Records are in game order, so the split is by game, not by ply.
        """
        self.path = path or self.PATH
        mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        self.rec = mm.reshape(-1, 32)
        self.n = self.rec.shape[0]
        self.lo = int(lo * self.n)
        self.hi = int(hi * self.n) - clen - 2
        assert self.hi > self.lo, "tape range too small for this chunk length"
        self.nchunk, self.clen, self.device = nchunk, clen, device
        self.stride = int(stride or 0)
        if self.stride:
            # A strided chunk still fetches `clen` records but spans
            # stride*clen/4 plies, so `hi` has to leave room for the span.
            self.hi = int(hi * self.n) - self.stride * (clen // 4) - 4
        self.unpack = GpuUnpacker(device)
        probe = np.array(self.rec[:4096])
        assert (probe[:, 29] != 0).any(), (
            f"{self.path} carries no game flags -- a filtered corpus like "
            "all.data has bytes 29..31 identically zero and cannot be a tape")

    def _offsets(self):
        """Record offsets within one chunk, relative to its start.

        stride 0 -- `clen` CONSECUTIVE plies. Maximum pair density (clen-2
        pairs) and maximum autocorrelation: every pair in a chunk comes from
        one 64-ply window of one game, so the effective sample size behind
        both the pair term and any batch statistic is nearer `nchunk` than
        `nchunk * clen`.

        stride s > 0 -- QUADRUPLES [a, a+1, a+2, a+3] starting every `s` plies
        from a random offset. Two things this buys:

          * The `pi+2` contract is unchanged. Slots 4j and 4j+2 are the same
            side to move two real plies apart, and so are 4j+1 and 4j+3, so a
            candidate written against `tp[pair + 2]` needs no edit and gets
            clen/2 pairs instead of clen-2 -- half the pairs, spread over
            s*clen/4 plies of game rather than 64.
          * The random per-chunk offset in [0, s) matters because chunk starts
            are drawn uniformly but the OFFSET pattern would otherwise be a
            fixed lattice: over many steps a fixed lattice keeps landing on
            the same plies of the same games and systematically misses the
            rest. `randint(0, s)` per chunk makes the sampled ply uniform.

        The middle ply of each quadruple is not waste: it is the other side to
        move, and every marginal term (H across, H within, I(B;game)) reads
        the whole tensor.
        """
        if not self.stride:
            return np.arange(self.clen)[None, :].repeat(self.nchunk, 0)
        nq = self.clen // 4
        base = (np.arange(nq) * self.stride)[None, :, None] + np.arange(4)[None, None, :]
        off = np.random.randint(0, self.stride, (self.nchunk, 1, 1))
        return (base + off).reshape(self.nchunk, -1)

    def sample(self):
        """(feat, gid, pair starts, group sizes) for nchunk runs of clen plies."""
        st = np.random.randint(self.lo, self.hi, self.nchunk)
        idx = (st[:, None] + self._offsets()).ravel()
        raw = np.array(self.rec[idx])
        ply = raw[:, 30].astype(np.int64) | (raw[:, 31].astype(np.int64) << 8)
        # a group breaks at every window start and at every game start inside one
        newg = (raw[:, 29] & 2) != 0
        newg[::self.clen] = True
        gid = np.cumsum(newg) - 1
        # same side to move, two real plies apart, same group
        # A pair is valid only if both records are the same game AND the ply
        # counters really are two apart. Under striding that check is what
        # rejects the pair that straddles a quadruple boundary (slots 4j+2 and
        # 4j+4 are `stride` plies apart, not 2) -- so no extra bookkeeping is
        # needed, and a game boundary inside a chunk is still caught by `gid`.
        ok = (gid[2:] == gid[:-2]) & (ply[2:] == ply[:-2] + 2)
        feat, _, _ = self.unpack(torch.from_numpy(raw).to(self.device))
        d = self.device
        return (feat,
                torch.from_numpy(gid).to(d),
                torch.from_numpy(np.flatnonzero(ok)).to(d),
                torch.from_numpy(np.bincount(gid)).to(d))


# ---------------------------------------------------------------- learned router

class Router(nn.Module):
    """A bucket rule with a gradient: b(x) = hard threshold of W x.

    Every rule tried so far -- magic hash, king square, the annealed
    `popcount(pawns & mask_j) > t_j` predicates -- is a FIXED function chosen by
    an outer search against a proxy, then trained around.  This is the same
    family with the search replaced by gradient descent on the loss we actually
    care about, because an annealed rule bit

        popcount(our_pawns & mask_j) > t_j     mask_j in {0,1}^64, t_j integer

    is exactly `sum_i W_ji x_i > t_j` with W restricted to 0/1 on the pawn rows.
    Letting W be real-valued and fitting it by backprop strictly contains that
    family, so `mode=bits, input=pawns` is the like-for-like arm: same
    information, same shape of rule, different way of choosing it.  With
    `input=all` the router can also read pieces and kings, and can represent
    `king6` exactly (bit j = bit j of the king square is linear in the one-hot
    king features), so an arm that loses to king6 is losing on optimisation,
    not on capacity.

    Two ways to turn W x into a bucket:
      bits   nb = 2**k independent sigmoids -> a factorised distribution,
             p(b) = prod_j s_j^{b_j} (1-s_j)^{1-b_j}.  Luka's shape.  k=6
             gives the 64 buckets every other arm has.
      flat   one softmax over nb logits.  More freedom per parameter, no
             factorised structure, k*nb weights instead of k*log2(nb).
      grouped  G groups of H-way softmaxes (e.g. 4x8: 32 logits, 4096 raw
             states).  One argmax per group, the raw code is the mixed-radix
             combine, and a fitted LUT (merge assign + refit) takes it back
             down to the deployed table count.  Same logit cost as flat32
             (one k-wide accumulator with k = G*H) plus one LUT lookup; the
             factorised structure sits between bits and flat.

    and three ways to train through it:
      soft   exact mixture sum_b p_b * delta_b.  No sampling: with 64 buckets
             the expectation is a matmul, and the exact value has strictly less
             variance than any sample of it.  Cheap, but it trains an ENSEMBLE
             of 64 tables and deploys ONE, so it is the arm most likely to
             flatter itself.
      st     hard argmax forward, soft gradient (straight-through).  Trains
             the thing that gets deployed.
      sample b ~ p, straight-through.  Luka's "pick the bucket randomly": the
             sampling noise is a regulariser that stops one bucket eating the
             batch, at the cost of a noisier gradient.

    In eval() ALL modes route hard.  The val loss printed for a router arm is
    therefore the loss of a deterministic 64-bucket rule and is directly
    comparable to magic / king6 / annealed numbers.  `soft_gap()` reports what
    the soft mixture would have scored, which is the number NOT to quote.

    Cost in the engine, for later: W x is a sum over the pieces on the board,
    so it is incremental exactly like the accumulator (a piece move is two
    adds), and a bucket change costs the same PSQT-table re-gather any other
    rule costs.  It is not more expensive than a magic; it is one k-wide
    accumulator bigger.
    """

    def __init__(self, rows, nbits=6, mode="bits", train_mode="st",
                 inp="all", std=0.05, gain=1.0, rank=0, sym=False,
                 groups=0, ways=0):
        super().__init__()
        assert mode in ("bits", "flat", "grouped")
        assert train_mode in ("soft", "st", "sample")
        self.mode, self.train_mode, self.gain = mode, train_mode, gain
        self.groups, self.ways = int(groups or 0), int(ways or 0)
        if mode == "grouped":
            assert self.groups >= 2 and self.ways >= 2, \
                "grouped needs groups x ways, e.g. 4x8"
            self.nb = self.ways ** self.groups
            k = self.groups * self.ways
        else:
            self.nb = 1 << nbits if mode == "bits" else nbits
            k = nbits if mode == "bits" else self.nb
        self.rank = int(rank or 0)
        # Chair-symmetric rule: W[their p, sq] = W[our p, sq^56], so the logits
        # are invariant under swapping the side to move (which also mirrors
        # ranks). A rule with this symmetry cannot flip on the stm mirror that
        # dominates flip1 (046: 98.4% -> 12.5% was the loss discovering it
        # unasked). Only the own-side half (384 rows) is stored; the other half
        # is the rank-mirrored copy. Folds out at export to exactly today's
        # matrix at exactly today's cost. Extras rows are not supported with
        # sym (router arms run extras=()).
        self.sym = bool(sym)
        assert not (self.sym and self.rank), "sym x rank not parameterised"
        if self.sym:
            assert rows == NFEAT + 1, "sym needs exactly the 768+PAD rows"
        prow = 384 if self.sym else rows
        if self.rank:
            # Split `std` between the factors so the PRODUCT starts at the same
            # scale as the unfactorised matrix: sd(wa @ wb) = sqrt(r)*sa*sb.
            sc = (std / math.sqrt(self.rank)) ** 0.5
            self.wa = nn.Parameter(torch.randn(rows, self.rank) * sc)
            self.wb = nn.Parameter(torch.randn(self.rank, k) * sc)
        else:
            self.w = nn.Parameter(torch.randn(prow, k) * std)
        self.bias = nn.Parameter(torch.zeros(k))
        # Which feature rows the router is allowed to read. `pawns` is rows
        # 0..63 -- side 0, piece type 0 -- which is precisely the `our pawns`
        # bitboard the annealed rules and the magic hashes see, so that arm
        # differs from them in ONE thing: how the rule was chosen.
        m = torch.zeros(rows, 1)
        if inp == "pawns":
            m[:64] = 1.0
            if self.sym:
                # The tied counterpart lives on rows 384..447 (their pawns);
                # without this the mask would break the symmetry it is for.
                m[384:448] = 1.0
        else:
            m[:NFEAT] = 1.0
        m[PAD] = 0.0
        self.register_buffer("rowmask", m)
        self.inp = inp
        self.bn = None                    # set by make_bn(), see below
        if mode == "bits":
            codes = torch.tensor(
                [[(b >> j) & 1 for j in range(nbits)] for b in range(self.nb)],
                dtype=torch.float32)
            self.register_buffer("codes", codes)
            self.register_buffer("pow2", (2 ** torch.arange(nbits)).long())
        if mode == "grouped":
            self.register_buffer(
                "strides",
                (self.ways ** torch.arange(self.groups)).long())

    def make_norm(self, k, mode, eigfloor=1e-2):
        """Attach a logit normaliser. See LogitNorm for what each mode buys."""
        self.bn = (nn.BatchNorm1d(k) if mode == "bn"
                   else LogitNorm(k, mode, eigfloor=eigfloor))

    def wmat(self):
        """The (rows, k) rule matrix, however it happens to be parameterised.

        With rank r > 0 the same matrix is stored as wa (rows, r) @ wb (r, k).
        For r >= k this is the SAME function class -- a bare (rows, k) matrix
        already has rank at most k, and a product with r >= k can be any such
        matrix -- so the change is purely one of parameterisation and the
        DEPLOYED rule is unchanged: multiply the factors out at export and the
        engine sees exactly the matrix it sees today, at exactly the cost it
        pays today.

        What changes is the gradient. A row of the unfactorised matrix only
        gets a gradient when its feature is on the board, which for a 768-row
        sparse input means about 32 rows per position. In the factorised form
        wb (r*k numbers) gets a gradient from EVERY position, so the rule can
        be rotated and rescaled as a whole while the sparse factor fills in.

        For r < k the family is strictly smaller: the k logits are then k
        linear functions of r underlying numbers, so the reachable buckets are
        the cells of an arrangement of k hyperplanes in r dimensions, at most
        sum_{i<=r} C(k, i) of them -- for k=6 that is 42 of 64 at r=3, 22 at
        r=2, 7 at r=1. Low rank costs buckets slowly, not suddenly.

        With sym the stored half is expanded: rows 0..383 as stored,
        384..767 the rank-mirrored copy, PAD zero. The deployed rule is an
        ordinary (rows, k) matrix.
        """
        base = (self.wa @ self.wb) if self.rank else self.w
        if not self.sym:
            return base
        w0 = base.reshape(6, 64, -1)
        w1 = w0[:, (torch.arange(64) ^ 56).to(base.device), :]
        out = torch.cat([w0.reshape(384, -1), w1.reshape(384, -1),
                         base.new_zeros(1, base.shape[-1])], 0)
        return out

    def logits(self, ext):
        w = self.wmat() * self.rowmask
        z = torch.nn.functional.embedding_bag(
            ext, w, mode="sum") * self.gain + self.bias
        return self.bn(z) if self.bn is not None else z

    def _joint(self, pg):
        """(B, H**G) joint from per-group (B, G, H) softmaxes, least-significant
        group first (matches `strides`). Built group by group so the peak is
        the final table, not G copies of it."""
        j = pg[:, 0, :]
        for g in range(1, self.groups):
            j = (j.unsqueeze(-1) * pg[:, g, :].unsqueeze(1)
                 ).reshape(j.shape[0], -1)
        return j

    def probs_and_hard(self, ext):
        """(soft distribution over buckets, hard bucket index)."""
        z = self.logits(ext)
        self._last_z = z
        if self.mode == "grouped":
            pg = torch.softmax(
                z.reshape(z.shape[0], self.groups, self.ways), dim=-1)
            idx = pg.argmax(-1)
            hard = (idx.to(torch.long) * self.strides).sum(1)
            p = self._joint(pg)
            return p, hard
        if self.mode == "bits":
            # log p(b) = sum_j [ b_j log s_j + (1-b_j) log(1-s_j) ], one matmul.
            ls, ls1 = nn.functional.logsigmoid(z), nn.functional.logsigmoid(-z)
            logp = ls @ self.codes.t() + ls1 @ (1.0 - self.codes).t()
            p = logp.exp()
            hard = ((z > 0).long() * self.pow2).sum(1)
        else:
            p = torch.softmax(z, dim=-1)
            hard = z.argmax(1)
        return p, hard

    def weights(self, ext):
        """(B, nb) mixing weights to apply to the per-bucket deltas."""
        p, hard = self.probs_and_hard(ext)
        self._last_hard = hard
        self._last_p = p
        if not self.training:
            return nn.functional.one_hot(hard, self.nb).to(p.dtype)
        if self.train_mode == "soft":
            return p
        if self.train_mode == "sample":
            if self.mode == "grouped":
                pg = torch.softmax(
                    self._last_z.reshape(-1, self.groups, self.ways), dim=-1)
                idx = torch.stack(
                    [torch.multinomial(pg[:, g, :], 1).squeeze(1)
                     for g in range(self.groups)], 1).long()
                idx = (idx * self.strides).sum(1)
            elif self.mode == "bits":
                bits = torch.bernoulli(torch.sigmoid(self.logits(ext)))
                idx = (bits.long() * self.pow2).sum(1)
            else:
                idx = torch.multinomial(p, 1).squeeze(1)
        else:
            idx = hard
        oh = nn.functional.one_hot(idx, self.nb).to(p.dtype)
        return oh + (p - p.detach())     # straight-through

    @torch.no_grad()
    def rule_table(self):
        """The deployable rule: masked weights and bias, nothing else."""
        return (self.wmat() * self.rowmask).cpu(), self.bias.detach().cpu()


class LogitNorm(nn.Module):
    """Centre -- and optionally orthogonalise -- the router logits.

    FREE IN THE ENGINE. Both modes are affine maps on the logits, so at export
    z' = W(Ax - mu) = (WA)x - W mu is one linear rule at exactly today's cost.

    Why centring is the whole of what BatchNorm does here: the hard bucket is
    sign(z_j) per bit, and sign(z_j / sigma_j) = sign(z_j) for any positive
    sigma. Dividing by the standard deviation CANNOT change the partition -- it
    only reparameterises the gradient. Subtracting the mean can, and does
    exactly the wanted thing: it puts each bit's threshold at its own median-ish
    point, so every bit splits the batch about 50/50 and no bit can go constant.

    Centring alone does not give even BUCKETS, though. Twelve bits that are all
    copies of one direction are each 50/50 and still name 2 patterns. Even
    buckets need the bits to be independent, and for a near-Gaussian logit that
    is decorrelation:

      white:  z' = Cov^{-1/2} (z - mu)      (ZCA -- the symmetric inverse
                                             square root, which is the whitening
                                             that rotates least)

    at which point 2^k sign patterns are all about equally likely and the even
    partition is had for nothing. The catch is that Cov^{-1/2} divides each
    direction by its own standard deviation, INCLUDING the near-degenerate ones
    a trained router has left flat. It promotes noise to a full bit, and a noise
    bit flips on nearly every move -- so evenness bought this way can cost flip2
    exactly where the buckets were empty for a reason. `eigfloor` is the knob
    that limits how far a flat direction can be amplified.
    """

    def __init__(self, k, mode="center", momentum=0.05, eigfloor=1e-2):
        super().__init__()
        # "stat" is the CONTROL: it tracks the mean and covariance and returns
        # the logits untouched, so an arm that is bit-for-bit `none` still
        # reports the spectrum whitening would have been applied to.
        assert mode in ("center", "white", "stat")
        self.mode, self.momentum, self.eigfloor = mode, momentum, eigfloor
        self.register_buffer("rmean", torch.zeros(k))
        self.register_buffer("rcov", torch.eye(k))
        # One global scale, learnable. It cannot change the partition (a
        # positive scalar leaves every sign alone) but it sets the temperature
        # the straight-through soft path sees, so the net gets to pick it.
        self.scale = nn.Parameter(torch.ones(1))

    def _wmat(self, c):
        k = c.shape[0]
        # Floor the eigenvalues RELATIVE to the mean one: an absolute floor
        # means something different at every logit scale.
        ev, evec = torch.linalg.eigh(c)
        ev = ev.clamp(min=self.eigfloor * ev.mean().clamp(min=1e-12))
        return (evec * ev.rsqrt()) @ evec.t()          # ZCA, symmetric

    def forward(self, z):
        if self.training:
            mu = z.mean(0)
            with torch.no_grad():
                self.rmean.mul_(1 - self.momentum).add_(self.momentum * mu.detach())
        else:
            mu = self.rmean
        zc = z - mu
        # Track the covariance in EVERY mode. It is k x k with k = 12, so it
        # costs nothing, and without it only whitened arms can report a
        # spectrum -- which is the one arm whose spectrum is endogenous, since
        # the router trained under whitening has already been pushed towards
        # isotropy. The centred and unnormalised arms are the ones that say
        # what whitening had to correct.
        if self.training:
            n = max(zc.shape[0] - 1, 1)
            c = (zc.t() @ zc) / n
            with torch.no_grad():
                self.rcov.mul_(1 - self.momentum).add_(self.momentum * c.detach())
        else:
            c = self.rcov
        if self.mode == "stat":
            return z
        if self.mode == "center":
            return zc * self.scale
        return (zc @ self._wmat(c).t()) * self.scale


class RoutedPSQTBucket(nn.Module):
    """`PurePSQTBucket` with the bucket function learned instead of chosen.

    Identical model otherwise: out = psqt(x) + delta_{b(x)}(x) + bias, where
    delta_b is a per-bucket PSQT table summed over the active features.  The
    delta table is stored transposed -- one row per FEATURE, one column per
    BUCKET -- which is the same nb * rows parameters as `PurePSQTBucket` but
    lets a single EmbeddingBag produce all nb bucket deltas at once, so the
    soft mixture costs one (B, nb) matvec instead of nb gathers.

    `balance` is an optional penalty on the router collapsing: -H(mean p) pulls
    the marginal bucket distribution towards uniform.  Default 0 -- collapse is
    something to MEASURE first, not to prevent before knowing it happens.
    """

    def __init__(self, scale=1.0, nbits=6, mode="bits", train_mode="st",
                 inp="all", extras=(), balance=0.0, rstd=0.05, gain=1.0,
                 rlr_mult=1.0, freeze=False, aux=None, rank=0, rwd=0.0,
                 scalar=False, bn="none", eigfloor=1e-2, sym=False,
                 groups=0, ways=0):
        super().__init__()
        self.scale = scale
        self.rlr_mult = rlr_mult
        self.ex = Extras(extras)
        self.psqt = bag(1, 0.0, rows=self.ex.rows)
        self.bias = nn.Parameter(torch.zeros(1))
        self.router = Router(self.ex.rows, nbits, mode, train_mode, inp,
                             std=rstd, gain=gain, rank=rank, sym=sym,
                             groups=groups, ways=ways)
        if bn and bn != "none":
            # Logit dim, not bucket count: for grouped that is G*H (32),
            # not H**G (4096) -- whitening a 4096-covariance per step
            # would cost more than the training.
            self.router.make_norm(int(self.router.bias.numel()),
                                  bn, eigfloor=eigfloor)
        # Read by train() to give the router group its own weight decay. On a
        # FACTORISED matrix, L2 on both factors is nuclear-norm regularisation
        # on their product, i.e. it pulls the rule towards LOW RANK rather
        # than merely towards small -- which is the reason to combine the two.
        self.router_wd = float(rwd)
        self.nb = self.router.nb
        # scalar: the adapter is ONE number per bucket instead of a PSQT delta
        # table per bucket -- nb parameters instead of nb * rows. Same router,
        # same rule, same soft mixture; only what a bucket is worth changes.
        self.scalar = bool(scalar)
        self.cvec = nn.Parameter(torch.zeros(self.nb)) if self.scalar else None
        self.delta = (None if self.scalar else
                      nn.EmbeddingBag(self.ex.rows, self.nb, mode="sum",
                                      padding_idx=PAD))
        if self.delta is not None:
            nn.init.zeros_(self.delta.weight)
        self.balance = balance
        # Extra terms on the ROUTER only. Every one of them makes the fit worse
        # on purpose: the engine pays for a bucket change with a table
        # re-gather, so the quantity being bought is the flip rate, and this is
        # where the val-loss headroom gets spent.
        #
        #   info    -I(B;X) = H(E p) - E H(p).  The differentiable form of the
        #           I(B;G) criterion the annealer selects finalists on: routing
        #           should be confident per position AND spread over buckets.
        #   bal     the balance half of that on its own (anti-collapse).
        #   margin  relu(m - |z|).  Keeps every logit away from its threshold,
        #           so a small perturbation cannot flip a bit.
        #   swing   mean over the MOVE GRAPH of (w[to] - w[from])^2.  A quiet
        #           move changes the logit by exactly w[to] - w[from], so this
        #           is the flip mechanism itself, priced directly. A variance
        #           over squares would be wrong: it also charges the rule for
        #           telling apart squares no single move connects.
        #   l1      mean |W|.  Fewer squares read -> fewer moves that can flip
        #           a bit, and a rule somebody can look at.
        self.aux = dict(aux or {})
        self.tape = None                      # built lazily, needs the device
        self.tape_nchunk, self.tape_clen = TAPE_NCHUNK, TAPE_CLEN
        self.tape_stride = TAPE_STRIDE
        if self.balance:
            self.aux.setdefault("bal", (self.balance, None))
        if "swing" in self.aux:
            e, _, _ = graph()
            self.register_buffer("edges", e)
        self.force_base = False
        self._aux = None
        if freeze:
            # The control that `rand64` is not. rand64 hashes the position, so
            # two positions one move apart land in unrelated buckets; a random
            # LINEAR THRESHOLD rule has the router's inductive bias -- it is a
            # smooth function of the piece set, so nearby positions usually
            # share a bucket -- with nothing learned. It separates "a rule of
            # this shape helps" from "learning the rule helps", which rand64
            # cannot.
            for p in self.router.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        if self.delta is not None:
            nn.init.zeros_(self.delta.weight)
        else:
            self.cvec.zero_()

    @torch.no_grad()
    def release_buckets(self):
        self.force_base = False

    def pretrain_router(self, steps, lr=1e-2, log_every=0, opt_name="adamw"):
        """Fit the ROUTER ALONE to the game-type objective, then freeze it.

        The reason to do this rather than train everything together: the two
        gradients fight. The prediction loss wants the router to separate
        positions that want different tables, which it can always do a little
        better by splitting on something that changes move to move; the
        game-type penalty wants the opposite. Trained jointly, whatever comes
        out is a truce between them at one particular penalty weight, and the
        weight is doing the deciding.

        Trained separately there is no truce to strike. Stage 1 sees only the
        game-type objective -- 768 x k router parameters, tape forwards, no
        prediction loss and no delta table in the graph at all. Stage 2 freezes
        that rule and fits the tables to it, which is exactly the problem the
        engine has: the bucket function is given, make the best of it. The
        stability of the deployed rule is then a property of stage 1 alone and
        cannot be traded away by stage 2.

        What it gives up: the router can no longer discover a split that the
        loss wanted but the game-type objective is indifferent to. That is the
        thing to measure, by comparing against the joint arm.
        """
        dev = next(self.parameters()).device
        if self.tape is None:
            self.tape = GameTape(dev, self.tape_nchunk, self.tape_clen,
                                 hi=TAPE_TRAIN_HI, stride=self.tape_stride)
        rp = list(self.router.parameters())
        opt = (torch.optim.AdamW(rp, lr=lr, weight_decay=self.router_wd)
               if opt_name == "adamw" else torch.optim.SGD(rp, lr=lr))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
        was = self.training
        self.train()
        t0 = time.time()
        for i in range(1, steps + 1):
            tf, gid, pi, cnt = self.tape.sample()
            # The "batch" for the per-position terms is the tape itself: there
            # is no shuffled batch in this stage and no prediction loss to
            # attach one to.
            pp, hh = self.router.probs_and_hard(self.ex(tf))
            self.router._last_p, self.router._last_hard = pp, hh
            loss = self._aux_terms()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if log_every and (i % log_every == 0 or i == steps or i == 1):
                with torch.no_grad():
                    tf, gid, pi, cnt = self.tape.sample()
                    tp, th = self.router.probs_and_hard(self.ex(tf))
                    fl = float((th[pi + 2] != th[pi]).float().mean())
                    m = torch.bincount(th, minlength=self.nb).float()
                    m = m / m.sum()
                    hb = float(-(m[m > 0] * m[m > 0].log()).sum())
                    print(f"  pretrain step {i:6d}  penalty {float(loss):+.4f}"
                          f"  tape flip2 {100 * fl:.1f}%  occ {int((m > 0).sum())}"
                          f"  eff {math.exp(hb):.1f}"
                          f"  {time.time() - t0:.0f}s", flush=True)
        for q in self.router.parameters():
            q.requires_grad_(False)
        # Stage 2 must not pay for the penalty it can no longer act on: with
        # the router frozen every aux term has zero gradient and costs a tape
        # forward per step.
        self.aux = {}
        if not was:
            self.eval()

    def _aux_terms(self):
        """Sum of the requested router penalties, using the last forward's z/p."""
        tot, z, p = 0.0, self.router._last_z, self.router._last_p
        if "bal" in self.aux or "info" in self.aux:
            m = p.mean(0)
            hb = -(m * (m + 1e-9).log()).sum()            # H(E p), want it BIG
        if "bal" in self.aux:
            tot = tot + self.aux["bal"][0] * (math.log(self.nb) - hb)
        if "info" in self.aux:
            hc = -(p * (p + 1e-9).log()).sum(1).mean()    # E H(p), want it SMALL
            tot = tot + self.aux["info"][0] * (hc - hb)   # = -I(B;X)
        if "margin" in self.aux:
            lam, m = self.aux["margin"]
            m = 1.0 if m is None else m
            tot = tot + lam * torch.relu(m - z.abs()).mean()
        if "swing" in self.aux:
            w = self.router.wmat() * self.router.rowmask
            d = w[self.edges[:, 1]] - w[self.edges[:, 0]]
            tot = tot + self.aux["swing"][0] * (self.router.gain ** 2) * d.pow(2).mean()
        if "l1" in self.aux:
            tot = tot + self.aux["l1"][0] * (self.router.wmat()
                                             * self.router.rowmask).abs().mean()
        if "ginfo" in self.aux or "tv" in self.aux or "custom" in self.aux:
            # A SECOND forward, on real game tape rather than the shuffled
            # batch. probs_and_hard() writes _last_z/_last_p, which the stats
            # printer and the terms above read, so it is put back.
            if self.tape is None:
                self.tape = GameTape(z.device, self.tape_nchunk,
                                     self.tape_clen, hi=TAPE_TRAIN_HI,
                                     stride=self.tape_stride)
            tf, gid, pi, cnt = self.tape.sample()
            tp, thard = self.router.probs_and_hard(self.ex(tf))
            tz = self.router._last_z                 # tape logits, pre-restore
            self.router._last_z, self.router._last_p = z, p
            if "ginfo" in self.aux:
                mall = tp.mean(0)
                h_pool = -(mall * (mall + 1e-9).log()).sum()
                sums = torch.zeros(cnt.numel(), self.nb,
                                   device=tp.device, dtype=tp.dtype)
                sums.index_add_(0, gid, tp)
                c = cnt.to(tp.dtype)
                mg = sums / c.unsqueeze(1)
                hg = -(mg * (mg + 1e-9).log()).sum(1)
                h_within = (hg * c).sum() / c.sum()
                # -I(B;game): high across games, low within one, as one number
                tot = tot + self.aux["ginfo"][0] * (h_within - h_pool)
            if "tv" in self.aux:
                # the real move, not the move-graph surrogate: same side to
                # move, two plies apart, squared change in the bucket posterior
                d = tp[pi + 2] - tp[pi]
                tot = tot + self.aux["tv"][0] * d.pow(2).sum(1).mean()
            if "custom" in self.aux:
                # A penalty supplied at run time by --router-loss-file. It sees
                # exactly what the hand-written terms above see and nothing
                # more, so a candidate can reproduce any of them and the
                # comparison is like-for-like. Cost: the same one extra forward.
                tot = tot + self.aux["custom"][0] * CUSTOM_LOSS(_Ctx({
                    "p": p, "z": z,
                    "tp": tp, "tz": tz, "thard": thard,
                    "gid": gid, "cnt": cnt, "pair": pi,
                    "nb": self.nb, "mode": self.router.mode,
                    "w": self.router.wmat() * self.router.rowmask,
                    "gain": self.router.gain,
                    "torch": torch, "F": nn.functional, "math": math,
                    "np": np,
                }))
        return tot

    def forward(self, feat):
        ext = self.ex(feat)
        scores = None if self.scalar else self.delta(ext)   # (B, nb) deltas
        if self.force_base:
            d = scores[:, 0]
            self._aux = None
        elif self.scalar:
            p, hard = self.router.probs_and_hard(ext)
            self.router._last_hard, self.router._last_p = hard, p
            c = self.cvec
            if not self.training:
                d = c[hard]
            elif self.router.train_mode == "soft":
                d = p @ c
            else:
                if self.router.train_mode == "sample":
                    idx = torch.multinomial(p, 1).squeeze(1)
                else:
                    idx = hard
                # Straight-through on the SCALAR, never a (B, nb) one-hot.
                # c is detached in the soft half so the gradient reaching c is
                # the one-hot alone -- exactly what (oh + p - p.detach()) @ c
                # gives, and exactly what the table path below does. Leaving c
                # attached here adds a second, soft gradient path into c; that
                # is a different estimator, not this one.
                sm = p @ c.detach()
                d = c[idx] + (sm - sm.detach())
            self._aux = self._aux_terms() if (self.aux and self.training) else None
        else:
            w = self.router.weights(ext)
            d = (w * scores).sum(-1)
            self._aux = self._aux_terms() if (self.aux and self.training) else None
        return (self.psqt(ext).squeeze(-1) + d + self.bias) * self.scale


@torch.no_grad()
def flip_proxy(model, feat, gen, remap=None):
    """Fraction of positions whose bucket changes when ONE piece makes one move.

    A PROXY, and named one: destinations are taken from the empty-board move
    graph, so it ignores legality, ignores that a capture also removes the
    captured piece, has no castling, and weights every piece equally instead of
    using the mix the search actually plays. `nnue/treeprice.py` is the
    instrument for the real number -- this one exists so a stability penalty
    can be steered inside the training loop without a round trip through the
    engine, and every number it gives must be confirmed there.
    """
    _, dest, deg = graph()
    dest, deg = dest.to(feat.device), deg.to(feat.device)
    idx = feat.long().clamp(min=0, max=NFEAT)
    ok = (feat != PAD) & (deg[idx] > 0)
    if not ok.any():
        return float("nan")
    slot = torch.multinomial(ok.float(), 1, generator=gen)          # (B,1)
    row = idx.gather(1, slot)                                       # (B,1)
    e = (torch.rand(row.shape, device=feat.device, generator=gen)
         * deg[row].float()).long().clamp(max=dest.shape[1] - 1)
    new = dest[row.squeeze(1), e.squeeze(1)].unsqueeze(1)
    moved = feat.clone()
    moved.scatter_(1, slot, new.to(feat.dtype))
    _, h0 = model.router.probs_and_hard(model.ex(feat))
    _, h1 = model.router.probs_and_hard(model.ex(moved))
    if remap is not None:                 # after a merge, a move between two
        h0, h1 = remap[h0], remap[h1]     # buckets sharing a table is no flip
    return float((h0 != h1).float().mean())


@torch.no_grad()
def router_report(model, b, K, lam, remap=None, tag="router", nstat=None):
    """What the router actually learned, and what the soft mixture was worth.

    The gap matters: a `soft` arm optimises an ensemble of nb tables and then
    has to deploy one of them.  If hard >> soft the arm has been scoring itself
    on a model it cannot ship.
    """
    if not isinstance(model, RoutedPSQTBucket):
        return
    was = model.training
    model.eval()
    hard_tot = soft_tot = n = 0.0
    # remap sends the raw bucket id to the id the ENGINE would see -- after a
    # merge, several routed buckets share one table, and a "flip" between two
    # of them costs nothing because there is nothing to re-gather. The val loss
    # needs no remap: the merged tables are already written into delta.
    nstat = (model.nb if remap is None
             else (int(nstat) if nstat else int(remap.max()) + 1))
    rm = None if remap is None else remap.to(b.device)
    counts = torch.zeros(nstat, device=b.device)
    ent = 0.0
    for feat, s, z in b.val_batches():
        ext = model.ex(feat)
        p, hard = model.router.probs_and_hard(ext)
        base = model.psqt(ext).squeeze(-1) + model.bias
        oh = nn.functional.one_hot(hard, model.nb).to(p.dtype)
        if getattr(model, "scalar", False):
            # One number per bucket: the hard read is a gather and the soft
            # read is a matvec. Never form a (B, nb) score matrix -- at
            # nb = 4096 and batch 65536 that alone is 1 GB.
            c = model.cvec.to(p.dtype)
            hp = (base + c[hard]) * model.scale
            sp = (base + p @ c) * model.scale
        else:
            scores = model.delta(ext)
            hp = (base + (oh * scores).sum(-1)) * model.scale
            sp = (base + (p * scores).sum(-1)) * model.scale
        m = len(feat)
        hard_tot += loss_fn(hp, s, z, K, lam).item() * m
        soft_tot += loss_fn(sp, s, z, K, lam).item() * m
        counts += (oh.sum(0) if rm is None
                   else torch.zeros_like(counts).index_add_(0, rm, oh.sum(0)))
        ent += float(-(p * (p + 1e-9).log()).sum(1).sum())
        n += m
    q = counts / counts.sum()
    occ = int((counts > 0).sum())
    hb = float(-(q[q > 0] * q[q > 0].log()).sum())
    print(f"  {tag}: hard val {hard_tot / n:.6f}   soft val {soft_tot / n:.6f}"
          f"   gap {100 * (hard_tot - soft_tot) / soft_tot:+.2f}%", flush=True)
    print(f"  {tag}: {occ}/{nstat} buckets used, "
          f"H(bucket) {hb:.3f} / {math.log(nstat):.3f} nats, "
          f"eff {math.exp(hb):.1f} buckets, "
          f"mean per-position H(p) {ent / n:.3f} nats "
          f"(0 = already hard)", flush=True)
    gen = torch.Generator(device=b.device).manual_seed(7)
    feat = next(iter(b.val_batches()))[0]
    fp = flip_proxy(model, feat, gen, remap=rm)
    w = (model.router.wmat() * model.router.rowmask)
    e, _, _ = graph()
    e = e.to(w.device)
    sw = float((w[e[:, 1]] - w[e[:, 0]]).pow(2).mean().sqrt()) * model.router.gain
    mz = float(model.router._last_z.abs().mean())
    print(f"  {tag}: one-move flip PROXY {100 * fp:.1f}%  "
          f"(rms logit swing per move {sw:.3f} vs mean |logit| {mz:.3f}) "
          f"-- confirm with treeprice.py", flush=True)
    # The quantity `ginfo`/`tv` are aimed at, measured on HARD buckets over
    # real games -- the training terms use soft p, so this is not a restatement
    # of the loss. 8 windows of 256 plies: long runs, because H(B|game) from a
    # few dozen samples is biased DOWN and would inflate I.  64x512 and not
    # 8x256: at ~2k pairs this probe put `none` at flip2 30.9% where the n=1.5M
    # measurement in routergames.py says 41.2% -- wrong by more than the effect
    # being looked for.  Plies inside a game are correlated, so the effective n
    # is far below the record count and a small tape lies.
    tape = getattr(model, "_stats_tape", None)
    if tape is None:
        # HELD OUT: lo=TAPE_TRAIN_HI, so these are games no penalty trained on.
        # Scoring flip2 on the training tape would let a candidate win by
        # memorising those games rather than by finding a stable rule.
        tape = model._stats_tape = GameTape(b.device, 128, 512,
                                            lo=TAPE_TRAIN_HI)
    tf, tgid, tpi, tcnt = tape.sample()
    _, thard = model.router.probs_and_hard(model.ex(tf))
    if rm is not None:
        thard = rm[thard]
    oh = nn.functional.one_hot(thard, nstat).float()
    ln2 = math.log(2)
    pm = oh.mean(0)
    nrec = oh.shape[0]
    hp = float(-(pm[pm > 0] * pm[pm > 0].log()).sum()) / ln2
    hp += (int((pm > 0).sum()) - 1) / (2 * nrec * ln2)          # Miller-Madow
    sums = torch.zeros(tcnt.numel(), nstat, device=oh.device)
    sums.index_add_(0, tgid, oh)
    hw, wt = 0.0, 0.0
    for gi in range(tcnt.numel()):
        ng = int(tcnt[gi])
        if ng < 8:
            continue
        q2 = sums[gi] / ng
        h = float(-(q2[q2 > 0] * q2[q2 > 0].log()).sum()) / ln2
        h += (int((q2 > 0).sum()) - 1) / (2 * ng * ln2)
        hw += h * ng
        wt += ng
    hw /= max(wt, 1.0)
    fl2 = float((thard[tpi + 2] != thard[tpi]).float().mean())
    print(f"  {tag}: on real games  flip2 {100 * fl2:.1f}%  "
          f"H across {hp:.2f} - H within {hw:.2f} = I(B;game) {hp - hw:.2f} bits",
          flush=True)
    if was:
        model.train()
    return {"hard": hard_tot / n, "soft": soft_tot / n, "occ": occ, "H": hb,
            "nstat": nstat,
            "eff": math.exp(hb), "h_pool": hp, "h_within": hw,
            "flip2": fl2, "I_game": hp - hw,
            "flip_proxy": fp, "swing": sw, "mean_abs_logit": mz,
            "counts": counts.cpu()}


@torch.no_grad()
def _feature_freq(model, b, cap=20):
    """How often each feature row is on the board, over a slice of val.

    Two bucket tables differ, as far as the LOSS is concerned, only on rows the
    board actually lights up: a delta on a row that never fires is free to be
    anything. So the distance that decides which buckets may be merged is
    weighted by feature frequency, not plain Euclidean on the raw columns.
    """
    f = torch.zeros(model.ex.rows, device=b.device)
    n = 0
    for i, (feat, s_, z_) in enumerate(b.val_batches()):
        ext = model.ex(feat)
        f.index_add_(0, ext.reshape(-1),
                     torch.ones(ext.numel(), device=f.device))
        n += len(feat)
        if i + 1 >= cap:
            break
    f[PAD] = 0.0
    return f / max(n, 1)


@torch.no_grad()
def _wkmeans(x, w, k, iters=30, seed=0):
    """Weighted k-means. Returns (centroids (k, d), label per input point)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    if k >= n:
        return x.clone(), torch.arange(n, device=x.device)
    # k-means++ seeding, sampling proportional to weight * distance^2 so that
    # a bucket the router never uses cannot claim a centroid.
    idx = [int(torch.multinomial(w.cpu().clamp(min=1e-12), 1, generator=g))]
    d2 = ((x - x[idx[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        pr = (w * d2).cpu().clamp(min=0)
        if float(pr.sum()) <= 0:
            pr = torch.ones(n)
        j = int(torch.multinomial(pr, 1, generator=g))
        idx.append(j)
        d2 = torch.minimum(d2, ((x - x[j]) ** 2).sum(1))
    c = x[torch.tensor(idx, device=x.device)].clone()
    lab = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iters):
        dist = torch.cdist(x, c)
        new = dist.argmin(1)
        if torch.equal(new, lab) and _ > 0:
            break
        lab = new
        num = torch.zeros_like(c).index_add_(0, lab, x * w.unsqueeze(1))
        den = torch.zeros(k, device=x.device).index_add_(0, lab, w)
        live = den > 0
        c[live] = num[live] / den[live].unsqueeze(1)
        # An empty cluster is re-seeded on the heaviest point furthest from its
        # own centroid, so a level never silently returns fewer than k tables.
        if (~live).any():
            far = (w * ((x - c[lab]) ** 2).sum(1))
            for ci in torch.nonzero(~live).flatten().tolist():
                j = int(far.argmax())
                c[ci] = x[j]
                far[j] = -1.0
    return c, lab


@torch.no_grad()
def merge_ladder(model, b, K, lam, levels, counts, seed=0):
    """Halve the bucket table by k-means and re-measure at every level.

    The router is left EXACTLY as trained -- 2**k logits, same rule, same
    argmax. What shrinks is how many distinct PSQT deltas those buckets point
    at. Two buckets in one cluster become the same table, so a move between
    them stops being a flip and stops costing an engine re-gather, and the val
    loss pays whatever that merge was worth. That is the whole tradeoff, on one
    curve, from ONE trained model -- no confound from retraining each size.

    Merging is nested: each level clusters the PREVIOUS level's centroids, so
    the 64-bucket table is a refinement of the 32-bucket one and the curve is a
    single hierarchy rather than eight unrelated fits.
    """
    W0 = model.delta.weight.detach().clone()          # (rows, nb)
    freq = _feature_freq(model, b)
    # clamped, not zeroed: a row that never fires must not be able to divide
    # by zero on the way back out, and at 1e-8 it carries no weight anyway.
    sc = freq.clamp(min=1e-8).sqrt().unsqueeze(0)     # loss-weighted metric
    cent = (W0.t() * sc).clone()                      # (nb, rows), scaled
    wts = counts.to(W0.device).clone().float()
    assign = torch.arange(model.nb, device=W0.device)
    out = []
    for k in levels:
        cent, lab = _wkmeans(cent, wts, k, seed=seed)
        assign = lab[assign]
        wts = torch.zeros(cent.shape[0], device=W0.device).index_add_(
            0, lab, wts)
        model.delta.weight.copy_(((cent / sc)[assign]).t())
        r = router_report(model, b, K, lam, remap=assign, tag=f"merge{k}",
                          nstat=cent.shape[0])
        r["k"] = k
        # The super-bucket id per raw bucket: what treeprice_ckpt needs to
        # score stability the way the engine would see it (a move between two
        # raw buckets sharing a table is no flip). router-json keeps it.
        r["assign"] = assign.detach().cpu().tolist()
        out.append(r)
        print(flush=True)
    model.delta.weight.copy_(W0)
    return out


