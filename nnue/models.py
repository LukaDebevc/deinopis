"""Every model arm except the learned-router one, which lives in router.py."""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import NFEAT, PAD   # noqa: E402


from common import bag   # noqa: E402
from features import Extras, flip_feat, side_split   # noqa: E402
from buckets import FAMILIES, b_king6, b_king_magic, b_magic6   # noqa: E402
from features import _parts   # noqa: E402


class DeepLinear(nn.Module):
    """768 -> ... -> 1 with NO activation anywhere, output multiplied by `scale`.

    Every width list here describes the *same* function class -- a composition
    of linear maps is a single 768-vector of weights -- so any difference
    between arms is optimisation trajectory, never capacity.

    `scale` exists to separate two explanations of why the factored arms win.
    The output lives in centipawns, so weights must travel from a ~0.02 init to
    O(100). Adam's step is ~lr regardless of gradient magnitude, so a single
    weight grows *additively* while a product of factors grows
    *multiplicatively*. Init std is divided by `scale` so every arm starts from
    the same distribution over outputs, leaving only the parameterisation's
    scale as the difference.
    """

    def __init__(self, widths=(), scale=1.0, std=0.02):
        super().__init__()
        self.scale = scale
        h = widths[0] if widths else 1
        self.bag = bag(h, std / scale)
        layers = []
        for a, b in zip(widths, widths[1:]):
            layers.append(nn.Linear(a, b, bias=False))
        if widths:
            layers.append(nn.Linear(widths[-1], 1))
        self.head = nn.Sequential(*layers) if layers else None
        self.bias = nn.Parameter(torch.zeros(1)) if not widths else None

    def forward(self, feat):
        x = self.bag(feat)
        x = self.head(x) if self.head is not None else x + self.bias
        return x.squeeze(-1) * self.scale

    @torch.no_grad()
    def collapse(self):
        """The end-to-end (weights, bias) in centipawns.

        A composition of linear maps is one linear map, so however many hidden
        widths were used for the optimisation, the trained model *is* a single
        768-vector. Multiplying the factors out recovers it, which is what makes
        a deep-linear PSQT free to use: the training-time parameterisation costs
        nothing at inference."""
        w = self.bag.weight.clone()                     # (769, h)
        b = torch.zeros((), device=w.device)
        if self.head is not None:
            for layer in self.head:
                w = w @ layer.weight.t()
                if layer.bias is not None:
                    b = b + layer.bias.squeeze()
        else:
            b = self.bias.squeeze()
        return (w.squeeze(-1) * self.scale)[:NFEAT], b * self.scale


class Quadratic(nn.Module):
    """eval = sum_k lam_k (v_k . x)^2 + b.x + c

    A symmetric W factors as W = sum_k lam_k v_k v_k^T, so at r = 768 this IS
    the full 768^2 -> 1 model, and below that a rank-r approximation. Unlike the
    deep-linear case, r is a real capacity constraint -- the square is what
    makes it one.

    The linear term is kept separate because at inference it needs no
    accumulator at all (32 gathers per node), which is what will make it cheap
    to condition on material buckets later. Only `v` needs incremental update,
    and it must stay bucket-independent or every capture rebuilds it.
    """

    def __init__(self, rank=64, scale=1.0, std=0.02, psqt=True):
        super().__init__()
        self.scale = scale
        self.v = bag(rank, std)
        self.lam = nn.Parameter(torch.randn(rank) * 0.01 / scale)
        self.psqt = bag(1, 0.0) if psqt else None
        self.bias = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        """Start the linear term at a trained PSQT, quadratic part at zero.

        For binary x, x_i^2 = x_i, so the diagonal of W *is* a PSQT -- the
        quadratic form already contains every linear function. But in the
        rank-r factorisation W = sum_k lam_k v_k v_k^T the diagonal cannot be
        set independently of the off-diagonal, which is exactly why the linear
        term is a separate parameter here. That separation is what gives this
        warm start somewhere to live: the model starts *at* the linear optimum
        and only has to learn the off-diagonal from there."""
        if self.psqt is None:
            raise ValueError("model has no linear term to initialise")
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        self.lam.zero_()          # quadratic part contributes nothing at step 0

    def forward(self, feat):
        z = self.v(feat)
        out = (z * z * self.lam).sum(-1) + self.bias
        if self.psqt is not None:
            out = out + self.psqt(feat).squeeze(-1)
        return out * self.scale




class Bucketed(nn.Module):
    """eval = psqt(x) + c + < act(V x), sum_f R_f[b_f(x)] >

    With act = square and one bucket this is *exactly* `Quadratic`: the reader
    row plays the role of `lam`, and W = V^T diag(R) V. So the baseline is
    nested inside the family, and any win is new capacity rather than a
    different model that happens to score better.

    With several families the reads are summed, not crossed: crossing
    bishops x kings x queens would be 590k buckets, while summing costs the
    sum of their sizes and still lets each family shift the output on its own.
    """

    ACTS = {
        "square": lambda z: z * z,
        "id": lambda z: z,
        "crelu": lambda z: torch.clamp(z, 0.0, 1.0),
    }

    def __init__(self, rank=512, scale=1.0, std=0.02, families=("none",),
                 act="square", pre=False, extras=(), pfams=(), vbuck=0):
        super().__init__()
        self.scale = scale
        self.act = self.ACTS[act]
        self.act_name = act
        self.families = list(families)
        self.pre = pre
        self.ex = Extras(extras)
        self.v = bag(rank, std, rows=self.ex.rows)
        self.psqt = bag(1, 0.0, rows=self.ex.rows)
        self.bias = nn.Parameter(torch.zeros(1))
        # A PSQT chosen by a bucket. The linear term is the one place in the
        # model that costs nothing at inference -- 32 gathers, no accumulator
        # (LEDGER 019) -- so conditioning it changes the table size and nothing
        # else: still 32 gathers, from a 576 x 769 table instead of a 769 one.
        # That makes it the cheapest degree-3 term available.
        self.pfams = list(pfams)
        self.pq = nn.ModuleList(
            [nn.Embedding(FAMILIES[f][1] * self.ex.rows, 1) for f in self.pfams])
        for m in self.pq:
            nn.init.zeros_(m.weight)
        # `vbuck` k: the last k accumulator entries are chosen by the rule, the
        # first `rank` are shared. This is the ONLY conditioning in this class
        # that sits in front of the accumulator, and it is the one the refresh
        # budget is about: on a bucket change those k entries must be rebuilt
        # from scratch, the other `rank` stay incremental. A full bucketed
        # feature transformer (HalfKP's design) is the k = rank case and does
        # not fit in memory here; this asks whether a slice of one is worth
        # anything, which is the question that decides the budget.
        self.vb = vbuck
        self.width = rank + vbuck
        if vbuck:
            assert len(self.families) == 1 and self.families[0] != "none", \
                "vbuck conditions the accumulator on exactly one rule"
            self.vbtab = nn.Embedding(FAMILIES[self.families[0]][1] * self.ex.rows,
                                      vbuck)
            nn.init.normal_(self.vbtab.weight, std=std)
        self.readers = nn.ModuleList(
            [nn.Embedding(FAMILIES[f][1], self.width) for f in self.families])
        for r in self.readers:
            nn.init.normal_(r.weight, std=0.01 / max(scale, 1.0))
        # Per-bucket shift *before* the activation. Still not in the
        # accumulator: V x is computed bucket-free and incrementally as before,
        # and the shift is r adds applied to the copy that is read out. What it
        # buys is a bucket-dependent operating point for the nonlinearity --
        # for crelu, where the units clip; for the square, (a+B)^2 contributes
        # a per-bucket *linear* term in the accumulator that the read alone
        # cannot express.
        self.shifts = nn.ModuleList(
            [nn.Embedding(FAMILIES[f][1], self.width)
             for f in self.families]) if pre else None
        if pre:
            for b in self.shifts:
                nn.init.zeros_(b.weight)
        self.force_base = False

    @torch.no_grad()
    def release_buckets(self):
        """Copy bucket 0 into every bucket, then let them specialise.

        The copy is what makes a curriculum out of a slow start: without it the
        buckets that were never routed to would be released still sitting at
        their initialisation, having learned nothing from the pooled data."""
        for m in list(self.readers) + list(self.shifts or []):
            m.weight.copy_(m.weight[:1].clone().expand_as(m.weight))
        for m in ([self.vbtab] if self.vb else []) + list(self.pq):
            # A bucketed table is indexed b * rows + feature, so bucket 0 is the
            # first `rows` entries and broadcasting means tiling them.
            row0 = m.weight[:self.ex.rows].clone()
            m.weight.copy_(row0.repeat(m.weight.shape[0] // self.ex.rows, 1))
        self.force_base = False

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        for r in self.readers:
            r.weight.zero_()      # the bucketed part contributes nothing at step 0
        if self.shifts is not None:
            for b in self.shifts:
                b.weight.zero_()
        for m in self.pq:
            m.weight.zero_()      # and neither does the bucketed PSQT

    def _bucket_psqt(self, ext, feat):
        """sum_f psqt_f[b_f(x)][i] over the active features i.

        Gathered with a mask rather than an EmbeddingBag padding row, because
        the padding row moves with the bucket; masking gives it zero gradient
        for free."""
        if not self.pq:
            return 0.0
        live = (ext != PAD).float()
        tot = 0.0
        for name, m in zip(self.pfams, self.pq):
            i = FAMILIES[name][0](feat)
            if self.force_base:
                i = torch.zeros_like(i)
            b = i.unsqueeze(1) * self.ex.rows
            tot = tot + (m(b + ext).squeeze(-1) * live).sum(1)
        return tot

    def forward(self, feat):
        idx = [FAMILIES[name][0](feat) for name in self.families]
        if self.force_base:
            idx = [torch.zeros_like(i) for i in idx]
        ext = self.ex(feat)
        a = self.v(ext)
        if self.vb:
            # Masked gather rather than a padding row, for the same reason as
            # `_bucket_psqt`: the padding row moves with the bucket.
            live = (ext != PAD).float().unsqueeze(-1)
            b = idx[0].unsqueeze(1) * self.ex.rows
            a = torch.cat([a, (self.vbtab(b + ext) * live).sum(1)], dim=-1)
        if self.shifts is not None:
            for i, shift in zip(idx, self.shifts):
                a = a + shift(i)
        z = self.act(a)
        w = None
        for i, reader in zip(idx, self.readers):
            r = reader(i)
            w = r if w is None else w + r
        out = ((z * w).sum(-1) + self.bias + self.psqt(ext).squeeze(-1)
               + self._bucket_psqt(ext, feat))
        return out * self.scale


class PurePSQTBucket(nn.Module):
    """psqr + bucketed psqr deltas only, no quadratic. For the 4096 adapter study.

    Model: out = psqt(x) + sum_f pq_f[b_f(x)](x) + bias
    where pq_f is a table (nb * rows, 1) and is summed over active features.
    This is exactly Bucketed's psqt term in isolation, which is what 'psqr + 4096 x psqr'
    means: base PSQT plus per-bucket PSQT delta, learning only relative difference.
    No V, no readers, no act. 4096*768 ~3.1M params, cheap (32 gathers per eval).
    """
    def __init__(self, scale=1.0, pfams=("king_magic",), extras=()):
        super().__init__()
        self.scale = scale
        self.ex = Extras(extras)
        self.pfams = list(pfams)
        self.psqt = bag(1, 0.0, rows=self.ex.rows)
        self.bias = nn.Parameter(torch.zeros(1))
        self.pq = nn.ModuleList(
            [nn.Embedding(FAMILIES[f][1] * self.ex.rows, 1) for f in self.pfams])
        for m in self.pq:
            nn.init.zeros_(m.weight)
        self.force_base = False

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        for m in self.pq:
            m.weight.zero_()

    @torch.no_grad()
    def release_buckets(self):
        for m in self.pq:
            row0 = m.weight[:self.ex.rows].clone()
            m.weight.copy_(row0.repeat(m.weight.shape[0] // self.ex.rows, 1))
        self.force_base = False

    def _bucket_psqt(self, ext, feat):
        if not self.pq:
            return 0.0
        live = (ext != PAD).float()
        tot = 0.0
        for name, m in zip(self.pfams, self.pq):
            i = FAMILIES[name][0](feat)
            if self.force_base:
                i = torch.zeros_like(i)
            b = i.unsqueeze(1) * self.ex.rows
            tot = tot + (m(b + ext).squeeze(-1) * live).sum(1)
        return tot

    def forward(self, feat):
        ext = self.ex(feat)
        out = self.psqt(ext).squeeze(-1) + self._bucket_psqt(ext, feat) + self.bias
        return out * self.scale


class ScalarBucket(nn.Module):
    """psqt(x) + c[b(x)] + bias -- ONE number per bucket, not a table.

    Every bucketed arm so far spends nb * rows parameters: a whole PSQT delta
    table per bucket, so 64 buckets cost 49k weights. This spends nb, so 2^16
    buckets cost 65k weights and the engine ships a flat array of int16 with no
    gather over features at all -- one index, one add. The question it answers
    is whether the capacity a bucketed PSQT buys is per-(feature, bucket) or
    merely a per-bucket OFFSET.

    `b_rand` is the severity control that makes the rest readable: a hash of
    the position carries no information about the position beyond its identity,
    so on HELD-OUT positions its offsets are fitted from unrelated positions
    and average to nothing. A random hash that gains is memorising, not
    routing, and the size of that gain is the ceiling for reading anything into
    the structured families at the same nb.
    """

    def __init__(self, scale=1.0, fam="rand64", extras=()):
        super().__init__()
        self.scale = scale
        self.ex = Extras(extras)
        self.fam = fam
        self.idx, self.nb = FAMILIES[fam]
        self.psqt = bag(1, 0.0, rows=self.ex.rows)
        self.bias = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.zeros(self.nb))
        self.force_base = False

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        self.c.zero_()

    @torch.no_grad()
    def release_buckets(self):
        self.c.fill_(float(self.c[0]))
        self.force_base = False

    def forward(self, feat):
        ext = self.ex(feat)
        d = 0.0 if (self.force_base or self.nb == 1) else self.c[self.idx(feat)]
        return (self.psqt(ext).squeeze(-1) + d + self.bias) * self.scale




class ReLUNet(nn.Module):
    """Control arm for the degree-2 ceiling test: matched parameter budget, but
    an activation that composes. A quadratic form cannot express any three-way
    interaction, and king safety and passed pawns are genuinely three-way. If
    this is far ahead at matched parameters, that ceiling binds and we want to
    know before building any incremental-update machinery."""

    def __init__(self, hidden=64, scale=1.0, std=0.02):
        super().__init__()
        self.scale = scale
        self.bag = bag(hidden, std)
        self.out = nn.Linear(hidden, 1)

    def forward(self, feat):
        z = torch.clamp(self.bag(feat), 0.0, 1.0)
        return self.out(z).squeeze(-1) * self.scale




class SideQuadratic(nn.Module):
    """eval = psqt(x) + c + <a_u^2, R_u> + <a_t^2, R_t> + <a_u a_t, R_x>

    a_u = V x_u sums the accumulator over our pieces only, a_t over theirs. The
    three readers are exactly the three blocks of W: ours x ours, theirs x
    theirs, and the cross block ours x theirs.

    The deployed model is the arm `tied`: one accumulator a_u + a_t and one
    reader, which expands to a_u^2 + 2 a_u a_t + a_t^2 and so forces all three
    blocks to share one coefficient per rank. `all` unties them for 2r extra
    parameters; the rest delete blocks outright. Together they answer which
    side's pieces actually have to multiply which.
    """

    BLOCKS = {"tied": (), "all": ("uu", "tt", "ut"), "same": ("uu", "tt"),
              "cross": ("ut",), "us": ("uu",), "them": ("tt",)}

    def __init__(self, rank=512, scale=1.0, std=0.02, blocks="tied"):
        super().__init__()
        self.scale, self.blocks = scale, blocks
        self.parts = self.BLOCKS[blocks]
        self.v = bag(rank, std)
        self.psqt = bag(1, 0.0)
        self.bias = nn.Parameter(torch.zeros(1))
        n = 1 if blocks == "tied" else len(self.parts)
        self.r = nn.Parameter(torch.randn(n, rank) * 0.01 / max(scale, 1.0))

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        self.r.zero_()

    def forward(self, feat):
        out = self.bias + self.psqt(feat).squeeze(-1)
        if self.blocks == "tied":
            a = self.v(feat)
            out = out + (a * a * self.r[0]).sum(-1)
        else:
            fu, ft = side_split(feat)
            au, at = self.v(fu), self.v(ft)
            term = {"uu": au * au, "tt": at * at, "ut": au * at}
            for i, part in enumerate(self.parts):
                out = out + (term[part] * self.r[i]).sum(-1)
        return out * self.scale


class Perspective(nn.Module):
    """Two accumulators from ONE table: a1 = V x, a2 = V flip(x).

    flip is a permutation of the feature indices, so a2 is what a second
    accumulator with the table V.flip would compute. This is a width-2r model
    whose second half is *tied* to a permuted copy of the first: the parameters
    of rank r, the accumulator width of 2r, and the colour-swap symmetry of
    chess asserted as a hard constraint on the table.

    The honest comparison needs both controls: an untied model of the same
    parameter count (rank r) and one of the same width (rank 2r). The tie is
    worth having only if it beats the first without losing to the second.
    """

    def __init__(self, rank=256, scale=1.0, std=0.02, cross=True):
        super().__init__()
        self.scale, self.cross = scale, cross
        self.v = bag(rank, std)
        self.psqt = bag(1, 0.0)
        self.bias = nn.Parameter(torch.zeros(1))
        self.r = nn.Parameter(torch.randn(3 if cross else 2, rank)
                              * 0.01 / max(scale, 1.0))

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        self.r.zero_()

    def forward(self, feat):
        a1 = self.v(feat)
        a2 = self.v(flip_feat(feat))
        out = (a1 * a1 * self.r[0]).sum(-1) + (a2 * a2 * self.r[1]).sum(-1)
        if self.cross:
            out = out + (a1 * a2 * self.r[2]).sum(-1)
        return (out + self.bias + self.psqt(feat).squeeze(-1)) * self.scale


class PerspSplit(nn.Module):
    """A fixed accumulator width, part of it tied across the two perspectives.

        a_free = V_f x                       width `free`
        a_us   = V_t x                       width `tied`
        a_them = V_t flip(x)                 width `tied`
        total accumulator width = free + 2 * tied

    Why the split only means anything when the halves share V_t: flip is a
    permutation P of the feature indices, so an untied pair [V1 x ; V2 P x] is
    [V1 ; V2 P] x -- one unconstrained accumulator of the summed width, with the
    permutation absorbed into the second table. Untied, "perspective" is just a
    name for a plain accumulator, and every allocation between the two halves is
    the same model. Sharing V_t is what turns the split into a constraint: the
    colour-swap symmetry of chess, asserted on the weights, at half the table
    for the same width.

    So this sweep is over how much of a fixed width budget is spent under that
    constraint. Note what is *not* held fixed: the tied half gets a cross reader
    (a_us . a_them) for free, because both vectors are already in registers, so
    the tied end has 3 readers per rank against the free end's 1. That is a real
    property of the design and not a confound -- the extra cost is one length-r
    multiply-accumulate on the read-out, not on the accumulator -- but it means
    the ends differ in quadratic terms as well as in weight sharing.
    """

    def __init__(self, free=512, tied=256, scale=1.0, std=0.02):
        super().__init__()
        self.scale, self.free, self.tied = scale, free, tied
        self.vf = bag(free, std) if free else None
        self.vt = bag(tied, std) if tied else None
        self.psqt = bag(1, 0.0)
        self.bias = nn.Parameter(torch.zeros(1))
        n = (1 if free else 0) + (3 if tied else 0)
        self.r = nn.Parameter(torch.zeros(n, max(free, tied)))
        with torch.no_grad():
            self.r.normal_(std=0.01 / max(scale, 1.0))

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        self.r.zero_()

    def forward(self, feat):
        out = self.bias + self.psqt(feat).squeeze(-1)
        i = 0
        if self.vf is not None:
            a = self.vf(feat)
            out = out + (a * a * self.r[i, :self.free]).sum(-1)
            i += 1
        if self.vt is not None:
            u = self.vt(feat)
            t = self.vt(flip_feat(feat))
            out = (out + (u * u * self.r[i, :self.tied]).sum(-1)
                       + (t * t * self.r[i + 1, :self.tied]).sum(-1)
                       + (u * t * self.r[i + 2, :self.tied]).sum(-1))
        return out * self.scale


class Antisym(nn.Module):
    """f(x) = (g(x) - g(flip(x))) / 2 for any inner model g.

    Asserts that a position is worth exactly minus the same position with the
    other side to move. That is false by one tempo, so the arm prices the tempo:
    if the constraint costs nothing, the model was spending no capacity on the
    side-to-move asymmetry and half its table is redundant. For a quadratic form
    the constraint is free at inference -- antisymmetrising W is a
    precomputation -- so a null result here is a 2x table saving.

    Note two structural consequences. The constant cancels, so an antisymmetric
    model cannot have a bias at all; and the warm start survives only in its
    antisymmetric part, so this arm does not start exactly at the linear optimum.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.inner.init_psqt(weights, bias)

    def forward(self, feat):
        return 0.5 * (self.inner(feat) - self.inner(flip_feat(feat)))




# ---------------------------------------------------------------- deep head

class Deep(nn.Module):
    """acc -> act -> 256->h -> crelu -> bucketed h->h -> crelu -> bucketed h->1

    The shape under test: two perspectives into a 2 x width accumulator, one
    layer everybody shares, then two small layers the bucket chooses. Buckets go
    late and narrow on purpose. A bucket that selected the 768 x 128 table would
    make every bucket relearn the representation from its own slice of the data;
    a bucketed 32x32 costs 1k parameters per bucket and inherits everything
    upstream, which is the same argument LEDGER 023 made for read-only buckets.

    `warm_buckets` implements Luka's curriculum: train with every position
    routed to bucket 0, then copy bucket 0 into every bucket and release. That
    is what makes it a curriculum rather than a slow start -- without the copy,
    the untouched buckets would be released still at their initialisation.
    """

    ACTS = {"crelu": lambda z: torch.clamp(z, 0.0, 1.0),
            "square": lambda z: z * z,
            "sqrclip": lambda z: torch.clamp(z, 0.0, 1.0) ** 2}

    def __init__(self, width=128, hidden=32, scale=1.0, std=0.02,
                 families=("none",), act1="crelu", persp=True, extras=()):
        super().__init__()
        self.scale, self.persp = scale, persp
        self.act1 = self.ACTS.get(act1, lambda z: z)
        self.act1_name = act1
        self.ex = Extras(extras)
        self.families = list(families)
        self.nb = [FAMILIES[f][1] for f in self.families]
        self.v = bag(width, std, rows=self.ex.rows)
        self.psqt = bag(1, 0.0, rows=self.ex.rows)
        self.bias = nn.Parameter(torch.zeros(1))
        # A pairwise-multiply read is not elementwise, so it cannot live in
        # ACTS: it consumes two accumulator entries per output. The accumulator
        # itself is unchanged -- same table, same incremental update cost -- so
        # comparing it against crelu at equal `width` holds the expensive half
        # of the model fixed and varies only how the read combines it. This is
        # the cheap stand-in for a quadratic feature transformer: a_i * a_j is a
        # rank-1 quadratic form in x, at linear-accumulator cost.
        self.pair = act1 in ("pairmul", "pairclip")
        self.w = width
        din = width * (2 if persp else 1)
        if self.pair:
            assert width % 2 == 0
            din //= 2
        self.l1 = nn.Linear(din, hidden)
        self.h = hidden
        self.l2 = nn.ModuleList([nn.Embedding(n, hidden * hidden + hidden)
                                 for n in self.nb])
        self.l3 = nn.ModuleList([nn.Embedding(n, hidden + 1) for n in self.nb])
        for m in list(self.l2) + list(self.l3):
            nn.init.normal_(m.weight, std=0.05)
        self.force_base = False

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        """Start at the trained PSQT with the network contributing nothing.

        Zeroing the last layer is what makes this exact: whatever the hidden
        units do, the output reads zero, so the model is the linear one at step
        0 and the network only has to earn what it adds."""
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        for m in self.l3:
            m.weight.zero_()

    @torch.no_grad()
    def release_buckets(self):
        """Broadcast bucket 0 into every bucket, then let them specialise."""
        for m in list(self.l2) + list(self.l3):
            m.weight.copy_(m.weight[:1].clone().expand_as(m.weight))
        self.force_base = False

    def forward(self, feat):
        idx = [FAMILIES[n][0](feat) for n in self.families]
        if self.force_base:
            idx = [torch.zeros_like(i) for i in idx]
        ext = self.ex(feat)
        a = self.v(ext)
        if self.persp:
            a = torch.cat([a, self.v(self.ex(flip_feat(feat)))], dim=-1)
        if self.pair:
            half = self.w // 2
            parts = list(a.split(self.w, dim=-1))       # one per perspective
            if self.act1_name == "pairclip":
                parts = [torch.clamp(p, 0.0, 1.0) for p in parts]
            a = torch.cat([p[..., :half] * p[..., half:] for p in parts], dim=-1)
        else:
            a = self.act1(a)
        z = torch.clamp(self.l1(a), 0.0, 1.0)
        h = self.h
        if self.l2 and (self.force_base or all(n == 1 for n in self.nb)):
            # Every sample would gather the same row, and the gather is an
            # [n, h, h] tensor -- 4.3 GB at h=128, n=65536. Skip it: the head
            # is a plain shared linear layer. Exactly equal, not an approximation.
            p = sum(m2.weight[0] for m2 in self.l2)     # [h*h + h]
            q = sum(m3.weight[0] for m3 in self.l3)     # [h + 1]
            z = torch.clamp(nn.functional.linear(z, p[:h * h].view(h, h),
                                                 p[h * h:]), 0.0, 1.0)
            out = (z * q[:h]).sum(-1) + q[h]
        else:
            w2 = b2 = w3 = b3 = None
            for m2, m3, i in zip(self.l2, self.l3, idx):
                p, q = m2(i), m3(i)
                w2 = p[:, :h * h] if w2 is None else w2 + p[:, :h * h]
                b2 = p[:, h * h:] if b2 is None else b2 + p[:, h * h:]
                w3 = q[:, :h] if w3 is None else w3 + q[:, :h]
                b3 = q[:, h:] if b3 is None else b3 + q[:, h:]
            z = torch.clamp(torch.einsum("nh,nkh->nk", z, w2.view(-1, h, h)) + b2, 0.0, 1.0)
            out = (z * w3).sum(-1) + b3.squeeze(-1)
        return (out + self.bias + self.psqt(ext).squeeze(-1)) * self.scale


class CoreLora(nn.Module):
    """psqt(x) + <crelu(a), R[b]> + <crelu(A a_core + L[b] a_lora), U[b]>

    Luka's design. One accumulator, three things each bucket owns, and a
    deliberate split of the accumulator between what every bucket shares and
    what each bucket adapts.

    The accumulator `a` is the usual 768 -> width table, run for both
    perspectives and concatenated, then CReLU'd. Nothing about it depends on
    the bucket, so a bucket change costs a different reader row and no refresh
    at all -- the whole point of LEDGER 023/036. The bucket then owns:

      1. a direct read of the entire activated accumulator, `R[b]` (2*width -> 1);
      2. `L[b]`, its own map from the SECOND half of each perspective into the
         same 16 hidden units the shared core `A` writes into;
      3. `U[b]`, the read of those 16 units.

    `A` is shared and sees the FIRST half of each perspective. So half of every
    accumulator is bucket-agnostic representation and half is what the bucket
    is allowed to reinterpret, and they meet by addition in 16 units before a
    single CReLU. That is the "core + bucketed adapter" shape: the core cannot
    be broken by a starved bucket, and a starved bucket falls back to the core
    plus whatever the curriculum left in it.

    The PSQT term is a plain linear pass-through straight to the answer, warm
    started from a trained linear model, with every bucketed output weight
    zeroed. At step 0 the model IS that linear model, so the network only ever
    has to earn what it adds on top.

    Measured caveat, worth keeping in view when reading the result: the 576
    material buckets have an effective count of 24.4 on this corpus (top two
    hold 25% of positions, half the buckets see under 46k positions in the
    whole 892M). The starved buckets keep their curriculum value; this run
    tests roughly 30 live buckets, not 576.
    """

    def __init__(self, width=128, hidden=16, core_frac=0.5, scale=1.0, std=0.02,
                 family="material", extras=()):
        super().__init__()
        self.scale = scale
        self.ex = Extras(extras)
        self.family = family
        self.bfn, self.nb = FAMILIES[family]
        self.w = width
        self.h = hidden
        self.wc = int(round(width * core_frac))     # dims per perspective -> core
        self.wl = width - self.wc                   # dims per perspective -> lora
        assert self.wc > 0 and self.wl > 0

        self.v = bag(width, std, rows=self.ex.rows)
        self.psqt = bag(1, 0.0, rows=self.ex.rows)
        self.bias = nn.Parameter(torch.zeros(1))

        self.core = nn.Linear(2 * self.wc, hidden)            # shared
        self.lora = nn.Embedding(self.nb, hidden * 2 * self.wl)   # per bucket
        self.read = nn.Embedding(self.nb, 2 * width + 1)          # acc -> answer
        self.out = nn.Embedding(self.nb, hidden + 1)              # 16 -> answer
        nn.init.normal_(self.lora.weight, std=0.05)
        nn.init.zeros_(self.read.weight)
        nn.init.zeros_(self.out.weight)
        self.force_base = False

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        """Start exactly at the trained linear model: psqt loaded, heads zero."""
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        self.read.weight.zero_()
        self.out.weight.zero_()

    @torch.no_grad()
    def release_buckets(self):
        """Broadcast bucket 0 into every bucket, then let them specialise."""
        for m in (self.lora, self.read, self.out):
            m.weight.copy_(m.weight[:1].clone().expand_as(m.weight))
        self.force_base = False

    def forward(self, feat):
        ext = self.ex(feat)
        a = self.v(ext)
        a = torch.cat([a, self.v(self.ex(flip_feat(feat)))], dim=-1)
        a = torch.clamp(a, 0.0, 1.0)                       # crelu, [n, 2w]

        w, wc, h = self.w, self.wc, self.h
        core_in = torch.cat([a[:, :wc], a[:, w:w + wc]], dim=-1)
        lora_in = torch.cat([a[:, wc:w], a[:, w + wc:]], dim=-1)
        z = self.core(core_in)

        if self.force_base or self.nb == 1:
            # Every row gathers bucket 0: do it as plain shared layers. The
            # gather is an [n, h, 2*wl] tensor -- 537 MB at n=65536, h=16,
            # wl=64 -- and it is exactly equal to this, not an approximation.
            lw = self.lora.weight[0].view(h, 2 * self.wl)
            z = torch.clamp(z + lora_in @ lw.t(), 0.0, 1.0)
            rd, ou = self.read.weight[0], self.out.weight[0]
            head = a @ rd[:-1] + rd[-1] + z @ ou[:-1] + ou[-1]
        else:
            idx = self.bfn(feat)
            lw = self.lora(idx).view(-1, h, 2 * self.wl)
            z = torch.clamp(z + torch.einsum("nd,nhd->nh", lora_in, lw), 0.0, 1.0)
            rd, ou = self.read(idx), self.out(idx)
            head = ((a * rd[:, :-1]).sum(-1) + rd[:, -1]
                    + (z * ou[:, :-1]).sum(-1) + ou[:, -1])

        return (head + self.bias + self.psqt(ext).squeeze(-1)) * self.scale


class HalfKAvsTonoMLP(nn.Module):
    """inn -> 256 -> 256 -> 1, persp (both sides), shared table.

    Two variants on the same backbone to isolate the feature family:

      halfka : 768 psqr + 64*2 king squares (king_us, king_them), i.e. the
               classic HalfKP/HalfKA lookup. Each side's king square is a
               64-state index; the two embeddings are summed with the psqr
               bag before the MLP. This is the (king position, psqr) bar.

      tono   : 768 psqr + 12 latch flags (turn-on but not off) per side*,
               i.e. 6 per side: pawn<8, knight<2, bishop<2, rook<2,
               queen<1, king_moved. Flags are computed from the position's
               piece counts / king squares vs start (4/60). Material flags
               are monotonic by construction (counts only decrease); king
               flag is ORed over the game via the batch's game_start extra,
               but current vs start is already a 99% latch (king rarely
               returns). Each active flag adds its embedding - exactly the
               "from each side just like halfka/halfkp" persp structure.

    Backbone is identical: EmbeddingBag(768+extra, 256) summed -> ReLU(256)
    -> ReLU(256) -> 1, scaled by K. No buckets, no Deep tricks, so the only
    difference between arms is the extra family.
    * 6*2=12 flags -> 12 embeddings, summed when active.
    """

    def __init__(self, mode="halfka", scale=1.0, std=0.02, hidden=256):
        super().__init__()
        assert mode in ("halfka", "tono", "king_magic", "king_magic32", "king6", "magic6", "km32")
        self.mode = mode
        self.scale = scale
        self.hidden = hidden
        # psqr bag + extra rows (king 128 or tono 12) share the table for
        # simplicity, but keep separate embeddings for the extra to make the
        # contribution isolatable.
        self.bag = bag(hidden, std)  # 768+1 rows, standard
        if mode == "halfka":
            self.extra = nn.Embedding(128, hidden)  # 64 us + 64 them
            nn.init.normal_(self.extra.weight, std=std)
        elif mode == "king_magic":
            self.extra = nn.Embedding(4096, hidden)
            nn.init.normal_(self.extra.weight, std=std)
        elif mode in ("king_magic32", "km32"):
            self.extra = nn.Embedding(32, hidden)
            nn.init.normal_(self.extra.weight, std=std)
            # load clustering if exists
            self._km_group = None
            km_path = "./nnue/kmeans32.npz"
            if os.path.exists(km_path):
                try:
                    d = np.load(km_path)
                    self._km_group = torch.from_numpy(d["group"].astype(np.int64))
                    print(f"  km32: loaded {km_path} 4096->32", flush=True)
                except Exception as e:
                    print(f"  km32: failed to load {km_path}: {e}", flush=True)
        elif mode == "king6":
            self.extra = nn.Embedding(64, hidden)
            nn.init.normal_(self.extra.weight, std=std)
        elif mode == "magic6":
            self.extra = nn.Embedding(64, hidden)
            nn.init.normal_(self.extra.weight, std=std)
        else:  # tono
            self.extra = nn.Embedding(12, hidden)   # 6 us + 6 them
            nn.init.normal_(self.extra.weight, std=std)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.psqt = bag(1, 0.0)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        # keep fc's small at start so model starts near psqt
        nn.init.normal_(self.fc1.weight, std=0.02); nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02); nn.init.zeros_(self.fc2.bias)

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def _king_indices(self, feat):
        valid, side, pt, sq = _parts(feat)
        is_k = valid & (pt == 5)
        us = (sq * (is_k & (side == 0))).sum(1)   # 0..63, 0 if missing (never)
        them = (sq * (is_k & (side == 1))).sum(1)
        return us.clamp(0, 63), them.clamp(0, 63)

    def _tono_flags(self, feat):
        # counts per side per type
        valid, side, pt, sq = _parts(feat)
        # _present expects (feat, mask, key, width)
        # build per-side counts for pawn,knight,bishop,rook,queen
        flags = []
        for s in (0, 1):  # us, them
            is_side = valid & (side == s)
            # pawn
            pawn_cnt = (is_side & (pt == 0)).sum(1)
            knight_cnt = (is_side & (pt == 1)).sum(1)
            bishop_cnt = (is_side & (pt == 2)).sum(1)
            rook_cnt = (is_side & (pt == 3)).sum(1)
            queen_cnt = (is_side & (pt == 4)).sum(1)
            flags.append((pawn_cnt < 8).long())
            flags.append((knight_cnt < 2).long())
            flags.append((bishop_cnt < 2).long())
            flags.append((rook_cnt < 2).long())
            flags.append((queen_cnt < 1).long())
            # king moved: us 4 vs them 60 in stm canonical start
            is_k = valid & (side == s) & (pt == 5)
            ksq = (sq * is_k).sum(1)
            start = 4 if s == 0 else 60
            flags.append((ksq != start).long())
        # 12 flags, order: us pawn,knight,bishop,rook,queen,king, them ...
        return torch.stack(flags, dim=1)  # (N,12)

    def forward(self, feat):
        x = self.bag(feat)  # (N, hidden)
        if self.mode == "halfka":
            us, them = self._king_indices(feat)
            x = x + self.extra(us) + self.extra(them + 64)
        elif self.mode == "king_magic":
            idx = b_king_magic(feat)
            x = x + self.extra(idx)
        elif self.mode in ("king_magic32", "km32"):
            idx4096 = b_king_magic(feat)
            if self._km_group is not None:
                grp = self._km_group.to(feat.device)[idx4096]
            else:
                # fallback: simple modulo if cluster not yet trained
                grp = idx4096 % 32
            x = x + self.extra(grp)
        elif self.mode == "king6":
            idx = b_king6(feat)
            x = x + self.extra(idx)
        elif self.mode == "magic6":
            idx = b_magic6(feat)
            x = x + self.extra(idx)
        else:
            flags = self._tono_flags(feat)  # (N,12)
            # sum embeddings of active flags
            # extra weight (12, hidden)
            w = self.extra.weight  # (12, hidden)
            # flags float (N,12) @ (12, hidden) -> (N, hidden)
            add = flags.float() @ w
            x = x + add
        # psqt skip
        psqt = self.psqt(feat).squeeze(-1)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        out = self.out(h).squeeze(-1)
        return (out + psqt + self.bias) * self.scale


class RulePieceMLP(nn.Module):
    """(rule, piece) joint embedding: rule selects piece table, then sum over pieces.

    User's request: pairs (rule, piece) vs (king, piece). This is 32×(768×256) vs 64×(768×256)
    then 256→256→1. Very different from HalfKAvsTonoMLP which did bag+extra additive.

    Architecture:
      For halfka: rule = our king square 0..63 (b_king6), table 64*768×256
      For km32:   rule = km32 cluster 0..31, table 32*768×256
      Accumulator = sum_{pieces on board} emb[rule*768 + piece]
      Then MLP: ReLU(acc →256) → ReLU(256→256) → 1, plus psqt skip.

    This matches the user's 32×(768×256) description. Halfka's 64 us + 64 them question:
    HalfKA classically has 64 us king positions; the '64 them' additive version was the
    previous HalfKAvsTonoMLP's 128 extra (us+them separate). Here joint uses 64 us only for
    direct comparison; we also support 128 if needed (king_us+king_them combined would be 4096).
    """

    def __init__(self, mode="halfka", scale=1.0, std=0.02, hidden=256):
        super().__init__()
        assert mode in ("halfka", "km32", "king6", "magic6")
        self.mode = mode
        self.scale = scale
        self.hidden = hidden
        if mode == "halfka":
            self.num_rules = 64  # our king only, 6 bits
            self.rule_fn = b_king6
        elif mode == "km32":
            self.num_rules = 32
            self.rule_fn = None  # special: needs b_king_magic + cluster
            km_path = "./nnue/kmeans32.npz"
            self._km_group = None
            if os.path.exists(km_path):
                try:
                    d = np.load(km_path)
                    self._km_group = torch.from_numpy(d["group"].astype(np.int64))
                    print(f"  RulePiece km32: loaded {km_path}", flush=True)
                except Exception as e:
                    print(f"  RulePiece km32 load failed {e}", flush=True)
            else:
                print(f"  RulePiece km32: WARNING no {km_path}, fallback to modulo", flush=True)
        elif mode == "king6":
            self.num_rules = 64
            self.rule_fn = b_king6
        elif mode == "magic6":
            self.num_rules = 64
            self.rule_fn = b_magic6

        # Joint table: num_rules * NFEAT entries, each hidden dim, plus one padding entry
        # Padding idx = num_rules * NFEAT, for PAD features (768) we map to zero vector
        self.emb = nn.Embedding(self.num_rules * NFEAT + 1, hidden, padding_idx=self.num_rules * NFEAT)
        nn.init.normal_(self.emb.weight, std=std)
        with torch.no_grad():
            self.emb.weight[self.num_rules * NFEAT].zero_()

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.psqt = bag(1, 0.0)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.fc1.weight, std=0.02); nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02); nn.init.zeros_(self.fc2.bias)

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, feat):
        # rule per position (N,)
        if self.mode == "km32":
            idx4096 = b_king_magic(feat)
            if self._km_group is not None:
                rule = self._km_group.to(feat.device)[idx4096]
            else:
                rule = idx4096 % 32
        else:
            rule = self.rule_fn(feat)  # (N,)

        # feat (N,32) contains 0..767 + PAD 768
        # Build joint indices: rule* NFEAT + feat, with PAD mapped to padding_idx
        # feat where PAD -> we want padding_idx, else rule*NFEAT+feat
        N = feat.shape[0]
        # Clamp feat for index calc but mask PAD
        is_valid = feat != PAD
        f_clamped = feat.clamp(max=NFEAT-1)
        joint = rule.unsqueeze(1) * NFEAT + f_clamped  # (N,32)
        # For invalid (PAD), set to padding_idx
        pad_idx = self.num_rules * NFEAT
        joint = torch.where(is_valid, joint, torch.full_like(joint, pad_idx))

        # Gather and sum over pieces: emb(joint) -> (N,32,hidden) -> sum over 32
        # Use embedding lookup: flat then reshape
        emb = self.emb(joint)  # (N,32,hidden)
        acc = emb.sum(dim=1)  # (N,hidden)  sum over pieces, pad contributes 0

        psqt = self.psqt(feat).squeeze(-1)
        h = torch.relu(self.fc1(acc))
        h = torch.relu(self.fc2(h))
        out = self.out(h).squeeze(-1)
        return (out + psqt + self.bias) * self.scale


class TreeHalving(nn.Module):
    """psqr -> acc with relu, then halving tree left+relu(a*left+b*right).

    `acc = relu(V x)`  (V is 768->W via bag). Then for cur size S:
        left = cur[:,:S//2], right = cur[:,S//2:]
        cur = left + relu(a*left + b*right)  # a,b are (S//2,) learned
    Halve until 1. So 256 ->128->64->32->16->8->4->2->1 (8 levels).
    Residual `left +` keeps gradient flow; `a,b` are elementwise scales
    per position in the half. No buckets, no extra families - pure psqr
    to test the combining rule itself. Comparable to HalfKAvsTonoMLP:
    same bag, same psqt warm start, same loss/scale/K, same 768 input,
    same data/batches if run with same seed/steps.
    """

    def __init__(self, acc_size=256, scale=1.0, std=0.02):
        super().__init__()
        assert acc_size > 1 and (acc_size & (acc_size - 1)) == 0, "acc_size must be power of 2"
        self.scale = scale
        self.acc_size = acc_size
        self.bag = bag(acc_size, std)
        self.psqt = bag(1, 0.0)
        self.bias = nn.Parameter(torch.zeros(1))
        self.a_params = nn.ParameterList()
        self.b_params = nn.ParameterList()
        sz = acc_size
        while sz > 1:
            half = sz // 2
            a = nn.Parameter(torch.randn(half) * 0.02)
            b = nn.Parameter(torch.randn(half) * 0.02)
            self.a_params.append(a)
            self.b_params.append(b)
            sz = half

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)

    def forward(self, feat):
        x = torch.relu(self.bag(feat))  # relu on acc, as requested
        cur = x
        for a, b in zip(self.a_params, self.b_params):
            half = cur.shape[1] // 2
            left = cur[:, :half]
            right = cur[:, half:half * 2]
            # a,b are (half,) broadcast to (N,half)
            cur = left + torch.relu(a * left + b * right)
        psqt = self.psqt(feat).squeeze(-1)
        return (cur.squeeze(-1) + psqt + self.bias) * self.scale


class TreeHalvingHeads(nn.Module):
    """Same halving tree but each level's output goes to prediction.

    psqr -> relu(acc) -> level0 head, then for each halving step
    cur = left+relu(a*left+b*right) and head(cur) is added. Final
    is sum over all heads + psqt + bias. So early levels contribute
    directly, not only through the tree's bottleneck. Comparable to
    TreeHalving: same bag, same a/b tree, same data/steps.
    """

    def __init__(self, acc_size=256, scale=1.0, std=0.02):
        super().__init__()
        assert acc_size > 1 and (acc_size & (acc_size - 1)) == 0, "acc_size must be power of 2"
        self.scale = scale
        self.acc_size = acc_size
        self.bag = bag(acc_size, std)
        self.psqt = bag(1, 0.0)
        self.bias = nn.Parameter(torch.zeros(1))
        self.a_params = nn.ParameterList()
        self.b_params = nn.ParameterList()
        self.heads = nn.ModuleList()
        sz = acc_size
        while sz >= 1:
            # head for current size
            h = nn.Linear(sz, 1)
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
            self.heads.append(h)
            if sz == 1:
                break
            half = sz // 2
            a = nn.Parameter(torch.randn(half) * 0.02)
            b = nn.Parameter(torch.randn(half) * 0.02)
            self.a_params.append(a)
            self.b_params.append(b)
            sz = half

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)

    def forward(self, feat):
        x = torch.relu(self.bag(feat))
        psqt = self.psqt(feat).squeeze(-1)
        out = self.heads[0](x).squeeze(-1)
        cur = x
        for i, (a, b) in enumerate(zip(self.a_params, self.b_params)):
            half = cur.shape[1] // 2
            left = cur[:, :half]
            right = cur[:, half:half * 2]
            cur = left + torch.relu(a * left + b * right)
            out = out + self.heads[i + 1](cur).squeeze(-1)
        return (out + psqt + self.bias) * self.scale


class PairAttn(nn.Module):
    """psqt(x) + sum_i <e_i, [act]( sum_j crelu(q_i * k_j) * v_j )>

    Luka's design. Every (piece, square) feature carries four d-vectors --
    query, key, value, exit -- and the eval is a sum over ORDERED PAIRS of the
    pieces on the board rather than over the pieces themselves.

    What it is, algebraically. Drop the crelu and the whole thing factorises:

        sum_c (sum_i q_ic e_ic) (sum_j k_jc v_jc)

    which is a rank-d quadratic form in x, i.e. exactly the deployed eval with
    r = d. That is the thing to keep in view -- LEDGER 036 measured r=512 as
    saturated and r=64 as far from it, so at d=8 the LINEAR part of this model
    is 64x poorer than what is already deployed. **Everything this arm can win
    has to come from the clamp**, which gates a pair off when q_i . k_j goes
    negative in a channel and saturates it when it goes above 1. That is a
    genuine three-way interaction (i, j, and which channel survives), which no
    quadratic form of any rank can express, and `sides` said cross-colour pairs
    carry 49% of the quadratic gain -- so pairs are the right object.

    `second_act="crelu"` clamps the per-piece sum s_i before the exit read,
    which makes the piece's total incoming interaction saturate: one piece
    attacked by five things stops counting the fifth.

    Cost, if it ever ships: the pair sum IS incrementally maintainable, despite
    the nonlinearity sitting between i and j -- keep s_i per piece, and a move
    touches one row and one column of the pair matrix, so the update is O(P d)
    with P ~ 16.6 pieces at a tree node. See library/011.

    The i == j term is left in. It is a per-feature scalar, i.e. a second PSQT,
    and the model is free to drive it to zero.
    """

    def __init__(self, d=8, scale=1.0, qk_std=0.7, ve_std=0.05,
                 second_act="none", chunk=0, ckpt=False, compile=True):
        super().__init__()
        self.scale, self.d = scale, d
        self.second = second_act
        self.chunk = chunk if chunk else d
        self.ckpt = ckpt
        # Measured, not assumed: the pair block is four elementwise passes over
        # an [n, 32, 32, d] tensor -- 1.07 GB at n=8192, d=32 -- so it is bound
        # by writing and re-reading that tensor, not by the arithmetic (it runs
        # at 0.08% of this card's fp32 peak). Fusing the chain into one kernel
        # is worth 5.2x at d=8 (253k -> 1.32M pos/s) and 5.0x at d=32 (77k ->
        # 383k), and cuts peak memory 3.4x. bf16 autocast on top buys nothing.
        if compile:
            self._block = torch.compile(self._block)
        # One table, four slices: the engine would gather 4d contiguous floats
        # per piece, so keeping them in one row is the honest layout.
        self.tab = nn.Embedding(NFEAT + 1, 4 * d, padding_idx=PAD)
        with torch.no_grad():
            w = self.tab.weight
            nn.init.normal_(w[:, :2 * d], std=qk_std)   # q, k
            nn.init.normal_(w[:, 2 * d:], std=ve_std)   # v, e
            w[PAD].zero_()
        self.psqt = bag(1, 0.0)
        self.bias = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def init_psqt(self, weights, bias):
        """Start exactly at the trained linear model: psqt loaded, exit zero.

        Zeroing `e` is what makes the pair term contribute nothing at step 0,
        the same contract Deep.init_psqt has with its last layer. q, k and v
        get no gradient on the first step because of it and start moving on the
        second, which is the same one-step delay Deep already pays."""
        self.psqt.weight.zero_()
        self.psqt.weight[:NFEAT, 0] = weights / self.scale
        self.psqt.weight[PAD].zero_()
        self.bias.copy_(bias.reshape(1) / self.scale)
        self.tab.weight[:, 3 * self.d:].zero_()

    def _block(self, q, k, v, e):
        """One slice of channels, start to finish. Elementwise in c, so
        chunking over c is exact rather than an approximation."""
        g = torch.clamp(q.unsqueeze(2) * k.unsqueeze(1), 0.0, 1.0)   # [n,P,P,c]
        s = (g * v.unsqueeze(1)).sum(2)                              # [n,P,c]
        if self.second == "crelu":
            s = torch.clamp(s, 0.0, 1.0)
        return (s * e).sum((1, 2))

    def forward(self, feat):
        q, k, v, e = self.tab(feat).split(self.d, dim=-1)
        tot = None
        for c in range(0, self.d, self.chunk):
            sl = slice(c, min(c + self.chunk, self.d))
            args = (q[..., sl], k[..., sl], v[..., sl], e[..., sl])
            part = (torch.utils.checkpoint.checkpoint(self._block, *args,
                                                      use_reentrant=False)
                    if self.ckpt and self.training else self._block(*args))
            tot = part if tot is None else tot + part
        return (tot + self.bias + self.psqt(feat).squeeze(-1)) * self.scale



