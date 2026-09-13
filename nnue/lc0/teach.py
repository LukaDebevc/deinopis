"""Label binpack FENs with maia-1900's full policy (library/014 Phase 2).

Streams FENs in shards; per position stores every LEGAL move's (policy_idx,
prob) with prob renormalized over legal moves (the ~0.8% illegal mass is
dropped, nothing else is filtered). Output: one .npz per shard with
  idx  u16[total_legal]   policy indices (1858 space)
  prob f16[total_legal]   renormalized probabilities
  ptr  i64[n_pos+1]       row offsets into idx/prob
Shard order = FEN-file order, so training can re-read positions from the dump.

Usage:
  nnue/.venv/bin/python nnue/lc0/teach.py --fens <dump> --outdir <dir> [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess  # noqa: E402
from nnue.lc0.encode import planes_from_fen  # noqa: E402
from nnue.lc0.policy import LC0Policy, uci_to_idx  # noqa: E402

SHARD = 200_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fens", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--net", default="nnue/lc0/maia-1900.pb.gz")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=SHARD)
    a = ap.parse_args()

    dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = True
    model = LC0Policy(a.net).to(dev).half().eval().to(memory_format=torch.channels_last)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def fen_stream():
        n = 0
        with open(a.fens) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield line
                n += 1
                if a.limit and n >= a.limit:
                    return

    stream = fen_stream()
    shard_no = 0
    total = 0
    t0 = time.time()
    done = True
    while done:
        fens = []
        try:
            for _ in range(a.shard):
                fens.append(next(stream))
        except StopIteration:
            done = False
        if not fens:
            break
        idx_all, prob_all, ptr = [], [], [0]
        n_legal_sum = 0
        for s in range(0, len(fens), a.batch):
            batch = fens[s:s + a.batch]
            x = np.stack([planes_from_fen(f) for f in batch])
            with torch.no_grad():
                logits = model(torch.from_numpy(x).to(dev).half()
                               .contiguous(memory_format=torch.channels_last))
                probs = torch.softmax(logits.float(), -1).cpu().numpy()
            for fen, pr in zip(batch, probs):
                b = chess.Board(fen)
                black = b.turn == chess.BLACK
                moves = list(b.legal_moves)
                ii = np.array([uci_to_idx(m.uci(), black) for m in moves],
                              dtype=np.uint16)
                pp = pr[ii].astype(np.float64)
                pp /= pp.sum()
                idx_all.append(ii)
                prob_all.append(pp.astype(np.float16))
                n_legal_sum += len(moves)
                ptr.append(n_legal_sum)
        np.savez_compressed(
            outdir / f"shard-{shard_no:04d}.npz",
            idx=np.concatenate(idx_all), prob=np.concatenate(prob_all),
            ptr=np.array(ptr, dtype=np.int64), n=len(fens))
        total += len(fens)
        r = total / (time.time() - t0)
        print(f"shard {shard_no}: {total:,} positions, {r:,.0f} pos/s",
              flush=True)
        shard_no += 1
    print(f"done {total:,} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
