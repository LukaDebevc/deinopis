"""Build lc0's 112 input planes from our 32-byte records, on the GPU.

The square decode is deliberately the same argsort-over-occupancy trick as
`loader.GpuUnpacker`: same primitives, same ordering, so the planes the teacher
sees and the features the student sees cannot drift apart. See `loader.py` for
why it is an argsort and not a `nonzero`.

Two conventions line up for free and are the reason this is short:

  * both formats are side-to-move relative, with "our" pieces at the bottom --
    our records mirror the board for Black, and so does lc0's encoder;
  * the piece code in the low three bits is P N B R Q K, which is exactly
    lc0's plane order within a history step.

What the records cannot supply is castling, the rule-50 count and which colour
is actually to move; those come from the `--aux` side file that
`nnue/extract` writes. History is filled by repeating the current position,
which is lc0's own behaviour when it is given a position with no history.
"""
import torch

MAX_PIECES = 32
HIST = 8          # history steps
PPB = 13          # planes per board


class PlaneBuilder:
    def __init__(self, device):
        self.device = device
        self.bit = torch.arange(8, device=device, dtype=torch.int16)
        self.slot = torch.arange(MAX_PIECES, device=device)

    def __call__(self, raw, aux, out=None):
        """raw (N,32) uint8, aux (N,) int32 -> (N,112,8,8) float32."""
        n = raw.shape[0]
        occ = raw[:, :8].to(torch.int16)
        bits = ((occ.unsqueeze(-1) >> self.bit) & 1).reshape(n, 64)
        counts = bits.sum(1, dtype=torch.long)
        order = torch.argsort(bits, dim=1, descending=True, stable=True)
        sq = order[:, :MAX_PIECES]

        pcs = raw[:, 8:24]
        nib = torch.empty((n, MAX_PIECES), dtype=torch.uint8, device=self.device)
        nib[:, 0::2] = pcs & 0x0F
        nib[:, 1::2] = pcs >> 4
        nib = nib.long()
        # colour<<3 | type, colour 0 = the side to move -> plane 0..11.
        plane = (nib >> 3) * 6 + (nib & 7)
        valid = self.slot.unsqueeze(0) < counts.unsqueeze(1)

        x = torch.zeros((n, 112, 64), dtype=torch.float32, device=self.device) if out is None else out.zero_()
        # Scatter this position into history step 0, then copy it into the rest.
        idx = plane * 64 + sq
        x[:, :PPB].reshape(n, -1).scatter_(1, idx, valid.float())
        for i in range(1, HIST):
            x[:, i * PPB:(i + 1) * PPB] = x[:, :PPB]

        a = aux.long()
        x[:, 105] = (a & 1).unsqueeze(-1).float()          # we can castle kingside
        x[:, 104] = ((a >> 1) & 1).unsqueeze(-1).float()   # ... queenside
        x[:, 107] = ((a >> 2) & 1).unsqueeze(-1).float()
        x[:, 106] = ((a >> 3) & 1).unsqueeze(-1).float()
        x[:, 108] = ((a >> 4) & 1).unsqueeze(-1).float()   # we are black
        x[:, 109] = ((a >> 5) & 127).unsqueeze(-1).float()  # rule-50 ply, raw
        x[:, 111] = 1.0
        return x.reshape(n, 112, 8, 8)
