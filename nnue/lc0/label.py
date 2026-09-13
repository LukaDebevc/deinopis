"""Relabel our training records with an lc0 net's WDL.

    python3 -m nnue.lc0.label --data all.data --aux all.aux \
        --net 512x15-t79_9-swa-2016000.pb.gz --out all.lc0 [--limit N] [--score]

Writes three float16 per record -- (L, D, W), in the order `wdlnet.target_dist`
uses, so the file drops straight into the trainer as a target.

`--score` is the part worth running first. It reports `ce_out` -- cross entropy
of the teacher's own distribution against the game outcome -- for the lc0 net
and for the incumbent `TeacherWDL` sigmoid, on the same positions. That is the
same number `wdlarms.evaluate` reports and it is teacher-independent, so it
says whether this teacher knows anything the old one did not BEFORE anyone
spends a GPU-day training a student on it. The incumbent scores 0.64714 on the
frozen val set.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnue.lc0.net import LC0Value          # noqa: E402
from nnue.lc0.planes import PlaneBuilder   # noqa: E402

REC = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole file")
    ap.add_argument("--offset", type=int, default=0, help="first record")
    # 4096 + channels_last measured 15,064 pos/s against 12,709 for 1024 in
    # NCHW on an A100 (512x15 net, fp16). An 8x8 board is a small spatial
    # extent, so the win is filling the tensor cores, not locality.
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score", action="store_true")
    # The refit cubic of LEDGER 048, i.e. exactly the teacher every arm in the
    # 10B sweep trained against. Defaults copied from wdlarms.py; Stockfish's
    # shipped constants are 1.84x worse in NLL on this corpus.
    ap.add_argument("--wdl-a", default="[219.5003,-471.8433,164.5011,319.7684]")
    ap.add_argument("--wdl-b", default="[472.3219,-1141.005,998.184,-121.0777]")
    a = ap.parse_args()

    dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = True
    model = LC0Value(a.net).to(dev).half().eval().to(memory_format=torch.channels_last)

    data = np.memmap(a.data, dtype=np.uint8, mode="r").reshape(-1, REC)
    aux = np.memmap(a.aux, dtype="<u2", mode="r")
    assert len(aux) == len(data), f"aux has {len(aux)} rows, data has {len(data)}"
    lo = a.offset
    hi = len(data) if a.limit <= 0 else min(len(data), lo + a.limit)
    n = hi - lo
    print(f"{n:,} records from {a.data} [{lo}:{hi}]", flush=True)

    out = None
    if a.out:
        out = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float16, shape=(n, 3))

    pb = PlaneBuilder(dev)
    ce_lc0 = ce_tea = 0.0
    teacher = None
    if a.score:
        import json
        from nnue.loader import GpuUnpacker
        from nnue.wdl import TeacherWDL
        unpack = GpuUnpacker(dev)
        teacher = TeacherWDL(torch.tensor(json.loads(a.wdl_a), device=dev),
                             torch.tensor(json.loads(a.wdl_b), device=dev))

    t0 = time.time()
    done = 0
    for s in range(lo, hi, a.batch):
        e = min(s + a.batch, hi)
        raw = torch.from_numpy(np.array(data[s:e])).to(dev)
        ax = torch.from_numpy(np.array(aux[s:e]).astype(np.int32)).to(dev)
        x = pb(raw, ax).half().contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            wdl = torch.softmax(model(x).float(), -1)      # (W, D, L)
        ldw = wdl.flip(-1)                                  # -> (L, D, W)
        if out is not None:
            out[s - lo:e - lo] = ldw.cpu().numpy().astype(np.float16)
        if a.score:
            cls = raw[:, 26].long()                         # 0 = loss, 1 = draw, 2 = win
            ce_lc0 += -ldw.clamp_min(1e-9).log().gather(1, cls[:, None]).sum().item()
            feat, score, _ = unpack(raw)
            tW, tD, tL = teacher(score, feat)
            t = torch.stack([tL, tD, tW], 1).clamp_min(1e-9)
            ce_tea += -t.log().gather(1, cls[:, None]).sum().item()
        done = e - lo
        if done % (a.batch * 200) == 0:
            r = done / (time.time() - t0)
            print(f"  {done:>12,} / {n:,}   {r:,.0f} pos/s   eta {(n-done)/r/60:.1f} min",
                  flush=True)

    dt = time.time() - t0
    print(f"done {done:,} in {dt:.0f}s = {done/dt:,.0f} pos/s", flush=True)
    if a.score:
        print(f"ce_out  lc0 {ce_lc0/done:.5f}   TeacherWDL {ce_tea/done:.5f}   "
              f"(lower is better; the frozen-val incumbent is 0.64714)", flush=True)
    if out is not None:
        out.flush()


if __name__ == "__main__":
    main()
