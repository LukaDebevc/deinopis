"""Batching, with the record -> feature unpack done on the GPU.

Two things dominate throughput and both are solved here.

**The unpack moves to the GPU.** Turning a 32-byte record into its ~17 active
feature indices costs ~13 ms/batch in numpy, which was 30% of a CPU step and
would have been ~60% of a GPU one. Done in torch it is a handful of elementwise
kernels. The CPU is then only doing a memcpy of 32 bytes per position.

**No `nonzero`, so no device sync.** The obvious way to find which squares are
occupied is `bits.nonzero()`, but that forces a device-to-host sync every batch
because the output size is data-dependent, which serialises the pipeline. A
stable descending argsort over the 64 occupancy bits puts the occupied squares
first, in ascending square order, with a fixed output shape and no sync. Every
position then has exactly 32 feature slots, unused ones pointing at a frozen
padding row.

Records are stored in game order, so a sequential batch would be ~50 positions
from one game sharing a single result label. Batches are drawn from a
contiguous BLOCK, which spans tens of thousands of games: that decorrelates as
well as a global shuffle while every disk read stays sequential.

Two ways of picking the block, and they are not equivalent:

`order="block"` picks a random start every time. Blocks overlap and gaps are
never visited, so at B draws over N records the fraction of the corpus the run
actually sees is 1 - exp(-B/N) -- 44% at B=524M over N=892M, with a quarter of
the stream being records already seen. That was the old default and it is why
a "one epoch" run was neither one epoch nor unique.

`order="seq"` cuts the corpus into fixed non-overlapping blocks, shuffles the
block ORDER, and visits each block exactly once per pass. Every record is used
exactly once. `stride` interleaves on top: pass e reads records congruent to
offset e (mod stride), so a pass sees positions `stride` plies apart and after
`stride` passes every record has been used exactly once, still with no repeats.
Ask for more batches than the corpus holds and it raises, unless the caller
says repeats are allowed.

**The block is permuted on the GPU, not in numpy.** `np.random.shuffle` on an
8M-row byte array takes ~5.7 s -- Fisher-Yates row swaps over a quarter of a
gigabyte have no locality -- which was ~47 ms per batch against ~17 ms of
compute, i.e. the whole pipeline was a shuffle with some training attached. A
`randperm` and a gather on the GPU costs well under a millisecond. The reader
thread now only reads bytes and pins them, and overlaps compute entirely.

The validation split is a contiguous TAIL. Games are contiguous, so a tail
split is a game-level split; a random position split would leak, because every
position in a game carries the same result label.
"""

import mmap
import os
import queue
import threading

import numpy as np
import torch

REC_BYTES = 32
_PAGE = os.sysconf("SC_PAGESIZE")
NFEAT = 768          # 384 ours + 384 theirs
PAD = NFEAT          # frozen embedding row for empty piece slots
MAX_PIECES = 32


class GpuUnpacker:
    """Record bytes on the GPU -> (features, score, result)."""

    def __init__(self, device):
        self.device = device
        self.bit = torch.arange(8, device=device, dtype=torch.int16)
        self.slot = torch.arange(MAX_PIECES, device=device)

    def __call__(self, raw):
        """raw: (N, 32) uint8 on device. Returns feat (N,32) int64, score, result."""
        n = raw.shape[0]

        # --- which squares are occupied, in ascending order ---
        occ = raw[:, :8].to(torch.int16)                       # byte b holds squares 8b..8b+7
        bits = ((occ.unsqueeze(-1) >> self.bit) & 1).reshape(n, 64)
        counts = bits.sum(1, dtype=torch.long)                 # pieces on the board
        # Stable + descending: the 1s come first, and among equal values the
        # original (ascending square) order is preserved. Fixed output shape.
        order = torch.argsort(bits, dim=1, descending=True, stable=True)
        sq = order[:, :MAX_PIECES]                             # (N,32), valid where slot < count

        # --- which piece sits in each occupancy slot ---
        # pcs packs one 4-bit code per occupied square, low nibble first.
        pcs = raw[:, 8:24]
        nib = torch.empty((n, MAX_PIECES), dtype=torch.uint8, device=self.device)
        nib[:, 0::2] = pcs & 0x0F
        nib[:, 1::2] = pcs >> 4
        nib = nib.long()

        # code = colour<<3 | piece type, colour 0 = the side to move.
        feat = (nib >> 3) * 384 + (nib & 7) * 64 + sq
        valid = self.slot.unsqueeze(0) < counts.unsqueeze(1)
        feat = torch.where(valid, feat, torch.full_like(feat, PAD))

        # --- labels: little-endian i16 score, then 0/1/2 result ---
        score = raw[:, 24].int() | (raw[:, 25].int() << 8)
        score = torch.where(score >= 32768, score - 65536, score).float()
        result = raw[:, 26].float() * 0.5
        return feat, score, result


class Batcher:
    def __init__(self, path, device, batch=65536, block=8_000_000,
                 val_frac=0.02, val_cap=1_000_000, seed=0, prefetch=2,
                 order="seq", stride=1, allow_repeat=False, data_frac=1.0,
                 val_stride=1, val_file=None, pool_end=0, labels=None):
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        assert len(mm) % REC_BYTES == 0, "file is not a whole number of records"
        self.mm = mm.reshape(-1, REC_BYTES)
        self._map = mm._mmap
        self._nbytes = len(mm)
        self._fd = os.open(path, os.O_RDONLY)
        n = len(self.mm)

        # An external per-record target, in lockstep with the records. The
        # bytes are CONCATENATED onto each record before anything shuffles, so
        # a label cannot drift from its position: there is no second index to
        # get wrong. `(N,3) float16` is `(N,6) uint8` viewed differently.
        self.lab = None
        if labels:
            lab = np.load(labels, mmap_mode="r")
            assert lab.shape == (n, 3), f"labels {lab.shape} != ({n}, 3)"
            assert lab.dtype == np.float16, f"labels are {lab.dtype}, want float16"
            self.lab = lab.view(np.uint8).reshape(n, 6)
            assert not val_file, "--val-file and --labels: the frozen val set " \
                                 "has no labels, so the two cannot be combined"

        self.device = device
        self.unpack = GpuUnpacker(device)
        self.batch = batch
        self.n_val = min(int(n * val_frac), val_cap)
        # `data_frac` shrinks the POOL without touching the validation tail, so
        # a run can be given fewer unique positions at the same step budget and
        # the two are still scored on the same held-out set. That is the only
        # way to price fresh data against repeated data.
        self.n_all = n
        # The val set is a tail, and `val_stride` spreads it over `val_stride`
        # times as many records -- so the same number of positions comes from
        # `val_stride` times as many GAMES. It is still a game-level split,
        # because the whole span is withheld from the pool. This matters for
        # any label that is a property of the game rather than of the position:
        # a 1M-position contiguous tail is only ~20k independent outcomes.
        self.val_stride = max(1, int(val_stride))
        # With an external `val_file` nothing is held back from the pool: the
        # held-out records live in a file of their own, so the span is zero and
        # the whole of `--data` is trainable.
        self.val_span = 0 if val_file else self.n_val * self.val_stride
        self.n_train = int((n - self.val_span) * data_frac)
        # `pool_end` caps the pool at an explicit record index, and is how a
        # frozen val set stays honest. `val_file` sets `val_span` to 0, so
        # without this the pool would run over the very records the val file
        # was frozen FROM and we would be training on the held-out set. The
        # caller passes the number `freeze_val.py` prints.
        if pool_end:
            assert pool_end <= n, f"pool_end {pool_end} > {n} records"
            self.n_train = min(self.n_train, int(pool_end * data_frac))
        self.order = order
        self.stride = max(1, int(stride))
        self.allow_repeat = allow_repeat
        # In seq mode `block` counts POOL entries, so it spans block*stride
        # records of the file. Round it to whole batches: the leftover of a
        # chunk is carried into the next one, but keeping it aligned means
        # there is usually nothing to carry.
        self.block = max(batch, (min(block, self.n_train // self.stride)
                                 // batch) * batch)
        self.seed = seed
        self.prefetch = prefetch
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        # Every record is reachable exactly once across `stride` passes, so the
        # unique budget does not depend on the stride -- only the ORDER does.
        self.n_unique = self.n_train

        # The validation tail is small and reused every eval: keep it resident.
        # The LAST n_val records of the FILE, never `mm[n_train:]` -- with
        # data_frac < 1 the pool no longer reaches the tail, and that slice
        # would quietly make the val set hundreds of millions of records.
        # `val_file` PINS the validation set to a file of its own.
        #
        # Without it the val set is the tail of `--data`, so APPENDING a month
        # to the pool silently replaces the held-out set and every number ever
        # recorded stops being comparable to anything measured afterwards --
        # no error, no warning, the numbers just quietly change meaning. Since
        # the pool is meant to grow (LEDGER 021: buy distinct games, not extra
        # passes), the val set has to stop being a function of it.
        #
        # `tools/freeze_val.py` writes the file, and writes exactly the records
        # this same expression would have selected, so a frozen run is directly
        # comparable to every run that came before the freeze.
        if val_file:
            vm = np.memmap(val_file, dtype=np.uint8, mode="r")
            assert len(vm) % REC_BYTES == 0, "val file is not whole records"
            self.val = torch.from_numpy(np.array(vm.reshape(-1, REC_BYTES))).to(device)
            self.n_val = len(self.val)
        else:
            self.val = torch.from_numpy(
                self._rows(self.n_all - self.val_span, self.n_all, self.val_stride)
            ).to(device)

    def _rows(self, lo, hi, step=1):
        """Records [lo, hi, step), with the external target appended if there
        is one. Every reader goes through here, so labelled and unlabelled runs
        differ in exactly one place."""
        r = self.mm[lo:hi:step]
        if self.lab is None:
            return np.array(r)
        return np.concatenate([r, self.lab[lo:hi:step]], axis=1)

    def unpack_row(self, raw):
        """`(N, 32)` -> (feat, score, result); `(N, 38)` adds the external
        `(L, D, W)` target as a fourth element."""
        if raw.shape[1] == REC_BYTES:
            return self.unpack(raw)
        t = raw[:, REC_BYTES:].contiguous().view(torch.float16).float()
        return self.unpack(raw[:, :REC_BYTES].contiguous()) + (t,)

    @property
    def host_mb(self):
        """Pinned host memory the reader can hold at once, in MB.

        This machine has 15.4 GB total with ~10 GB of desktop resident, so the
        trainer has ~5 GB to live in and pinned pages cannot be swapped. Worth
        printing rather than discovering when the desktop locks up."""
        w = REC_BYTES + (6 if self.lab is not None else 0)
        return (self.prefetch + 1) * self.block * self.stride * w / 2 ** 20

    def __repr__(self):
        return (f"Batcher(train={self.n_train:,}, val={self.n_val:,}, "
                f"batch={self.batch:,}, block={self.block:,}, "
                f"order={self.order}, stride={self.stride}, "
                f"val_stride={self.val_stride}, "
                f"max_steps={self.max_steps:,}, host<={self.host_mb:,.0f}MB)")

    @property
    def max_steps(self):
        """Batches available before any record would be shown twice."""
        return self.n_unique // self.batch

    def reset(self):
        """Re-seed so every arm of an experiment sees identical batches."""
        self.rng = np.random.default_rng(self.seed)
        self.gen = torch.Generator(device=self.device).manual_seed(self.seed)

    def _drop(self, lo, hi):
        """Release the page cache holding records [lo, hi).

        The block is copied into pinned memory the instant it is read and is
        never looked at again -- random offsets over a file far larger than RAM
        mean there is no reuse to preserve. Left alone the kernel keeps every
        one of those pages as warm file cache and pays for it by swapping out
        whatever else is on the machine: on a 27 GB corpus this reader held
        6.5 GB of single-use cache while 8 GB of desktop anon sat in swap.

        Both calls are needed. madvise unmaps the range from our page table;
        fadvise then evicts it, which it silently declines to do for pages that
        are still mapped anywhere. Measured: fadvise alone drops 0 of 512 MB,
        the pair drops all of it."""
        a = (lo * REC_BYTES) // _PAGE * _PAGE
        b = min(-(-(hi * REC_BYTES) // _PAGE) * _PAGE, self._nbytes)
        if b <= a:
            return
        try:
            self._map.madvise(mmap.MADV_DONTNEED, a, b - a)
            os.posix_fadvise(self._fd, a, b - a, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass

    def _chunks(self, epoch):
        """Non-overlapping (lo, hi, step) file ranges covering one whole pass.

        Pass `epoch` reads the records congruent to `off` modulo the stride,
        where `off` walks a fixed permutation of [0, stride) so consecutive
        passes do not repeat an interleave. The chunk ORDER is shuffled; the
        chunks themselves tile the pass with no overlap and no gaps, which is
        what makes every record appear exactly once.
        """
        s = self.stride
        off = int(np.random.default_rng(self.seed).permutation(s)[epoch % s])
        m = (self.n_train - off + s - 1) // s          # pool entries this pass
        starts = np.arange(0, m, self.block)
        np.random.default_rng((self.seed, epoch)).shuffle(starts)
        for c in starts:
            k = min(self.block, m - c)                 # pool entries in chunk
            lo = off + int(c) * s
            yield lo, lo + (k - 1) * s + 1, s

    def _block_reader(self, q, stop):
        """Read raw blocks off the memmap into pinned memory, ahead of the GPU.

        No shuffling here on purpose -- see the module docstring. Pinned so the
        host-to-device copy can be async and overlap the previous block's
        training."""
        epoch = 0
        while not stop.is_set():
            if self.order == "seq":
                ranges = self._chunks(epoch)
            else:
                lo = int(self.rng.integers(0, max(1, self.n_train - self.block)))
                ranges = [(lo, lo + self.block, 1)]
            for lo, hi, s in ranges:
                if stop.is_set():
                    return
                blk = torch.from_numpy(self._rows(lo, hi, s)).pin_memory()
                self._drop(lo, hi)
                while not stop.is_set():
                    try:
                        q.put(blk, timeout=1.0)
                        break
                    except queue.Full:
                        continue
            epoch += 1

    def train_batches(self, n_batches):
        """Yield exactly `n_batches` batches.

        In seq mode this raises rather than quietly starting a second lap over
        data the run has already seen -- a repeat is a thing to opt into, not
        to discover afterwards in a loss curve."""
        if (self.order == "seq" and not self.allow_repeat
                and n_batches > self.max_steps):
            raise ValueError(
                f"{n_batches:,} batches x {self.batch:,} = "
                f"{n_batches * self.batch:,} positions, but only "
                f"{self.n_unique:,} unique training records exist "
                f"({self.max_steps:,} batches). Cut the steps, cut the batch, "
                f"add data, or pass allow_repeat=True.")
        q = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        th = threading.Thread(target=self._block_reader, args=(q, stop), daemon=True)
        th.start()
        try:
            emitted, carry = 0, None
            while emitted < n_batches:
                blk = q.get().to(self.device, non_blocking=True)
                perm = torch.randperm(len(blk), device=self.device, generator=self.gen)
                blk = blk[perm]
                # Carry a chunk's remainder into the next chunk rather than
                # dropping it, so "exactly once" stays exactly true.
                if carry is not None:
                    blk = torch.cat([carry, blk])
                    carry = None
                i = 0
                while i + self.batch <= len(blk):
                    yield self.unpack_row(blk[i:i + self.batch])
                    i += self.batch
                    emitted += 1
                    if emitted >= n_batches:
                        return
                if i < len(blk):
                    carry = blk[i:].clone()
        finally:
            stop.set()

    def val_batches(self):
        for i in range(0, len(self.val), self.batch):
            yield self.unpack_row(self.val[i:i + self.batch])
