"""Cross-check the engine's WDL forward pass against the trained model.

Four things are on trial and every one of them fails silently in real play:

  1. the feature convention, which the engine re-implements rather than reads;
  2. the FIVE bucket indices -- our king, their king mirrored, our material,
     their material, and the side-symmetric material x pawns bucket. A wrong
     bucket reads a plausible row of a trained table and the eval stays
     sensible;
  3. the three logits, compared BEFORE they collapse to cp. A swapped W and L
     in the psqt passthrough moves the distribution and can leave the rounded
     centipawn untouched, so comparing cp alone would not catch it;
  4. the file format and the base+delta fold in `export_wdl.py`.

The model side is built from the CHECKPOINT, not from the exported file, so the
exporter is on trial too. Everything is f32 on both sides, so the tolerance is
float noise, not a quantisation budget: anything bigger is a bug.

    chess fendump 2000 > /tmp/fens.txt
    CHESS_WDL=nets/combo.nnue CHESS_CKPT=ckpt-wdlnet/combo.pt \
        python3 verify_wdl.py /tmp/fens.txt
"""
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from wdlnet import WdlNet, king_sqs, mat_buckets   # noqa: E402
from wdl import b_sym                              # noqa: E402
from loader import PAD                             # noqa: E402

ENGINE = os.environ.get("CHESS_BIN", "./target/release/chess")
NET = os.environ["CHESS_WDL"]
CKPT = os.environ["CHESS_CKPT"]

fens = [l.strip() for l in open(sys.argv[1]) if l.strip()]
# CHESS_SMALL=1: check a dual file's quiescence tail (`evalfen --small`)
# against the tail's own checkpoint, the way the big tail is checked.
cmd = [ENGINE, "evalfen", "--wdl", NET]
if os.environ.get("CHESS_SMALL"):
    cmd.append("--small")
out = subprocess.run(cmd, input="\n".join(fens),
                     capture_output=True, text=True)
lines = [l for l in out.stdout.splitlines() if "|" in l]
assert len(lines) == len(fens), f"{len(lines)} results for {len(fens)}\n{out.stderr}"
assert "wdl eval loaded" in out.stderr, out.stderr

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = WdlNet(**ck["cfg"])
model.load_state_dict(ck["state"])
model.eval()
K = float(os.environ.get("CHESS_K", 0)) or ck.get("k_cp") or 288.5

ORDER = "PNBRQK"


def features_from_fen(fen):
    """Independent implementation, from the FEN and nothing else."""
    board, stm = fen.split()[0], fen.split()[1]
    feats = []
    for r, row in enumerate(board.split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            sq = (7 - r) * 8 + f
            if stm == "b":
                sq ^= 56
            side = 0 if (("w" if ch.isupper() else "b") == stm) else 384
            feats.append(side + 64 * ORDER.index(ch.upper()) + sq)
            f += 1
    return sorted(feats)


def cp_of(logits):
    """`cp = K * logit(W + D/2)`, the readout in `wdl.py` and in wdleval.rs."""
    m = float(np.max(logits))
    e = np.exp(np.asarray(logits, dtype=np.float64) - m)
    return K * np.log((e[2] + e[1] / 2) / (e[0] + e[1] / 2))


feat_bad = bucket_bad = 0
d_cp, d_lg = [], []
with torch.no_grad():
    for fen, line in zip(fens, lines):
        parts = line.split("|")
        assert len(parts) == 5, f"engine printed {len(parts)} fields: {line[:80]}"
        cp_rs = int(parts[0])
        fs = sorted(int(x) for x in parts[1].split(","))
        bks_rs = [int(x) for x in parts[3].split(",")]
        lg_rs = np.array([float(x) for x in parts[4].split(",")])

        py = features_from_fen(fen)
        if fs != py:
            feat_bad += 1
            if feat_bad <= 3:
                print(f"FEATURE MISMATCH {fen}\n  engine {fs}\n  python {py}")
            continue
        row = torch.tensor(py + [PAD] * (32 - len(py))).unsqueeze(0)

        ku, kt = king_sqs(row)
        mu, mt = mat_buckets(row)
        bks_py = [int(ku), int(kt), int(mu), int(mt), int(b_sym(row))]
        if bks_rs != bks_py:
            bucket_bad += 1
            if bucket_bad <= 3:
                print(f"BUCKET MISMATCH {fen}\n  engine {bks_rs}\n  python {bks_py}")

        lg_py = model(row)[0].numpy().astype(np.float64)
        d_lg.append(np.abs(lg_rs - lg_py).max())
        d_cp.append(cp_rs - cp_of(lg_py))

d_cp, d_lg = np.array(d_cp), np.array(d_lg)
print(f"positions            {len(fens)}")
print(f"feature mismatches   {feat_bad}   <- must be 0")
print(f"bucket mismatches    {bucket_bad}   <- must be 0")
print(f"logit  |engine-model|: mean {d_lg.mean():.2e}  max {d_lg.max():.2e}")
print(f"engine cp - model cp : mean {d_cp.mean():+.4f}  max|.| {np.abs(d_cp).max():.4f} cp")
print("                     (both sides are f32 and the engine rounds to whole")
print("                      centipawns, so anything over ~0.5 cp is a bug)")
ok = feat_bad == 0 and bucket_bad == 0 and np.abs(d_cp).max() < 0.6
print("ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
