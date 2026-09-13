"""Cross-check the engine's quadratic eval against the trained model.

Two independent things get tested, and both can fail silently in real play:
  1. the FEN -> 768 feature convention (side-to-move mirror, colour swap,
     piece order, square numbering) -- derived here from the FEN directly, not
     copied from the engine's output;
  2. the collapse to one matrix, the i16 quantisation and the engine's integer
     arithmetic -- by evaluating the same feature set with float torch;
  3. for a version-4 net, the BUCKET INDEX. That is the one part of the eval
     the engine re-implements rather than reads from the file, so a mismatch is
     silent: the eval stays plausible and every game is played from the wrong
     table. Set CHESS_PFAMS to the families the checkpoint was trained with.
"""
import subprocess, sys
import numpy as np, torch
sys.path.insert(0, '.')
from export import collapse

import os
ENGINE = os.environ.get("CHESS_BIN", "./target/release/chess")
# Both must describe the same model: NET is what the engine loads, CKPT is what
# it is checked against. Overridable so a new run can be verified without
# editing the script (and silently verifying the previous net).
NET = os.environ.get("CHESS_QUAD", "data/quad.bin")
CKPT = os.environ.get("CHESS_CKPT", "ckpt/quadratic_r=768.pt")
# Bucketed-PSQT families in CKPT, comma separated, in the order Bucketed built
# them. Empty for a plain version-3 net.
PFAMS = [x for x in os.environ.get("CHESS_PFAMS", "").split(",") if x]
ORDER = "PNBRQK"

def features_from_fen(fen):
    """Independent implementation: piece placement -> stm-relative indices."""
    board, stm = fen.split()[0], fen.split()[1]
    feats = []
    for r, row in enumerate(board.split("/")):        # rank 8 first
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch); continue
            sq = (7 - r) * 8 + f                      # A1 = 0
            colour = "w" if ch.isupper() else "b"
            pt = ORDER.index(ch.upper())
            if stm == "b":
                sq ^= 56                              # mirror ranks
            side = 0 if colour == stm else 384
            feats.append(side + 64 * pt + sq)
            f += 1
    return sorted(feats)

fens = [l.strip() for l in open(sys.argv[1]) if l.strip()]
out = subprocess.run([ENGINE, "evalfen", "--quad", NET], input="\n".join(fens),
                     capture_output=True, text=True)
lines = [l for l in out.stdout.splitlines() if "|" in l]
assert len(lines) == len(fens), f"{len(lines)} results for {len(fens)} fens\n{out.stderr}"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
M, bias = collapse(ck)

# The bucketed PSQT, straight out of the checkpoint -- deliberately NOT via the
# exported file, so the file format is on trial too.
import rulestats as R
RULE = {"material": R.r_material, "count": R.r_count}
pq = []
for k, name in enumerate(PFAMS):
    w = ck["state"][f"pq.{k}.weight"].double().reshape(-1) * ck["scale"]
    n_b = {"material": 576, "count": 8}[name]
    pq.append((RULE[name], w.reshape(n_b, -1)))

feat_bad, bucket_bad, errs = 0, 0, []
for fen, line in zip(fens, lines):
    cp_rs, fs = line.split("|")[:2]
    fs = sorted(int(x) for x in fs.split(","))
    py = features_from_fen(fen)
    if fs != py:
        feat_bad += 1
        if feat_bad <= 3:
            print(f"FEATURE MISMATCH {fen}\n  engine {fs}\n  python {py}")
        continue
    idx = torch.tensor(py)
    exact = M[idx][:, idx].sum().item() + bias
    if pq:
        row = torch.tensor(py + [768] * (32 - len(py))).unsqueeze(0)
        want = [int(rule(row)[0][0]) for rule, _ in pq]
        # `chess evalfen` always prints material,count in that fixed order, so
        # CHESS_PFAMS has to be given in that order too.
        got = [int(x) for x in line.split("|")[2].split(",")][:len(want)]
        if got != want:
            bucket_bad += 1
            if bucket_bad <= 3:
                print(f"BUCKET MISMATCH {fen}  engine {got} python {want}")
            continue
        for b, (_, tab) in zip(want, pq):
            exact += tab[b, idx].sum().item()
    errs.append(int(cp_rs) - exact)

errs = np.array(errs)
print(f"positions            {len(fens)}")
print(f"feature mismatches   {feat_bad}   <- must be 0")
if PFAMS:
    print(f"bucket mismatches    {bucket_bad}   <- must be 0  ({'+'.join(PFAMS)})")
print(f"engine cp - model cp: mean {errs.mean():+.3f}  max|.| {np.abs(errs).max():.3f} cp")
print(f"                     (quantisation budget was ~0.14 cp mean, 0.58 max)")
