"""Cross-check the engine's `Deep` forward pass against the trained model.

Three things are on trial and every one of them fails silently in real play:
  1. the feature convention, which the engine re-implements rather than reads;
  2. the DERIVED extra features -- `pawnfile` is "furthest-advanced pawn per
     file, ours crossed with theirs", so an off-by-one in the rank or a swapped
     ours/theirs gives a perfectly plausible eval from the wrong row;
  3. the bucket index, the weight layout and the file format.

The model side is built from the CHECKPOINT, not from the exported file, so the
exporter is on trial too. Everything is f32 on both sides, so the tolerance is
float noise (~1e-3 cp), not a quantisation budget: anything bigger is a bug.
"""
import os, subprocess, sys
import numpy as np, torch

sys.path.insert(0, '.')
import train as T

ENGINE = os.environ.get("CHESS_BIN", "./target/release/chess")
NET = os.environ["CHESS_DEEP"]
CKPT = os.environ["CHESS_CKPT"]
EXTRAS = tuple(x for x in os.environ.get("CHESS_EXTRAS", "").split("+") if x)
FAMS = tuple(x for x in os.environ.get("CHESS_FAMS", "").split(",") if x)

fens = [l.strip() for l in open(sys.argv[1]) if l.strip()]
out = subprocess.run([ENGINE, "evalfen", "--deep", NET], input="\n".join(fens),
                     capture_output=True, text=True)
lines = [l for l in out.stdout.splitlines() if "|" in l]
assert len(lines) == len(fens), f"{len(lines)} results for {len(fens)}\n{out.stderr}"
assert "deep eval loaded" in out.stderr, out.stderr

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
w = ck["state"]["v.weight"]
model = T.Deep(w.shape[1], ck["state"]["l1.weight"].shape[0], ck["scale"],
               families=FAMS or ("none",), act1="crelu", extras=EXTRAS)
model.load_state_dict(ck["state"])
model.eval()

ORDER = "PNBRQK"


def features_from_fen(fen):
    """Independent implementation, from the FEN and nothing else."""
    board, stm = fen.split()[0], fen.split()[1]
    feats = []
    for r, row in enumerate(board.split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch); continue
            sq = (7 - r) * 8 + f
            if stm == "b":
                sq ^= 56
            side = 0 if (("w" if ch.isupper() else "b") == stm) else 384
            feats.append(side + 64 * ORDER.index(ch.upper()) + sq)
            f += 1
    return sorted(feats)


feat_bad, errs = 0, []
with torch.no_grad():
    for fen, line in zip(fens, lines):
        cp_rs, fs = line.split("|")[:2]
        fs = sorted(int(x) for x in fs.split(","))
        py = features_from_fen(fen)
        if fs != py:
            feat_bad += 1
            if feat_bad <= 3:
                print(f"FEATURE MISMATCH {fen}\n  engine {fs}\n  python {py}")
            continue
        row = torch.tensor(py + [T.PAD] * (32 - len(py))).unsqueeze(0)
        errs.append(int(cp_rs) - float(model(row)[0]))

errs = np.array(errs)
print(f"positions            {len(fens)}")
print(f"feature mismatches   {feat_bad}   <- must be 0")
print(f"engine cp - model cp: mean {errs.mean():+.4f}  max|.| {np.abs(errs).max():.4f} cp")
print(f"                     (both sides are f32; anything over ~0.5 cp is a bug,")
print(f"                      and the engine rounds to whole centipawns)")
