"""Is the lc0 teacher better, or just differently shaped?

`ce_out` alone cannot answer that. It moves for two reasons at once:

  accuracy  a Stockfish search is a better outcome predictor than one lc0
            forward pass, which favours the incumbent;
  shape     the incumbent's D is a deterministic function of (score, material)
            and cannot say "this ending is drawn", which favours lc0.

So this splits them.

  onll        binary NLL of E = W + D/2 against the outcome. Depends only on
              the scalar each teacher implies -- pure accuracy, no shape.
  ce_out(K)   lc0's own E pushed back through the incumbent's sigmoid, with K
              fitted on this data. Same accuracy as lc0, same shape as the
              incumbent, so lc0-raw minus this is the value of the shape alone.
  hybrid      the incumbent's scalar wearing lc0's D. Given E and D the rest
              is forced: W = E - D/2, L = 1 - E - D/2. If the shape is worth
              something and the scalar is not, this is the teacher to train on.
  draw table  empirical draw rate against each teacher's predicted D, binned by
              E. This is the dark-knowledge claim stated directly: if lc0 knows
              something about drawishness that a sigmoid of the score cannot,
              it shows up here or nowhere.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnue.lc0.net import LC0Value          # noqa: E402
from nnue.lc0.planes import PlaneBuilder   # noqa: E402
from nnue.loader import GpuUnpacker        # noqa: E402
from nnue.wdl import TeacherWDL            # noqa: E402

REC = 32
KS = torch.arange(80.0, 620.0, 20.0)       # candidate scales for lc0's logit
NB = 10                                    # E bins for the draw table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=4096)
    # lc0 over-predicts D by 0.01-0.02 in nearly every bin of the draw table,
    # which is a calibration bias and would otherwise be charged to "shape".
    # --fit scans a (temperature, draw-logit bias) grid; pass the winner back in
    # via --temp/--dbias AND EVALUATE ON A DIFFERENT SLICE.
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--dbias", type=float, default=0.0)
    ap.add_argument("--wdl-a", default="[219.5003,-471.8433,164.5011,319.7684]")
    ap.add_argument("--wdl-b", default="[472.3219,-1141.005,998.184,-121.0777]")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = LC0Value(a.net).to(dev).half().eval().to(memory_format=torch.channels_last)
    teacher = TeacherWDL(torch.tensor(json.loads(a.wdl_a), device=dev),
                         torch.tensor(json.loads(a.wdl_b), device=dev))
    pb, unpack = PlaneBuilder(dev), GpuUnpacker(dev)
    ks = KS.to(dev)

    data = np.memmap(a.data, dtype=np.uint8, mode="r").reshape(-1, REC)
    aux = np.memmap(a.aux, dtype="<u2", mode="r")
    lo, hi = a.offset, min(len(data), a.offset + a.limit)

    n = 0
    names = ("lc0", "teacher", "hybrid", "reverse")
    ce = {k: 0.0 for k in names}
    onll = {k: 0.0 for k in names}
    ce_k = torch.zeros(len(ks), device=dev)
    TS = torch.arange(0.6, 1.85, 0.05, device=dev)
    DB = torch.arange(-0.6, 0.65, 0.05, device=dev)
    fit_ce = torch.zeros(len(TS), len(DB), device=dev)
    # draw table: per E bin, count, empirical draws, sum of each teacher's D
    tab = torch.zeros(NB, 4, device=dev, dtype=torch.float64)

    for s in range(lo, hi, a.batch):
        e = min(s + a.batch, hi)
        raw = torch.from_numpy(np.array(data[s:e])).to(dev)
        ax = torch.from_numpy(np.array(aux[s:e]).astype(np.int32)).to(dev)
        x = pb(raw, ax).half().contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            lg = model(x).float()                            # (W, D, L) logits
        lg = lg / a.temp
        lg[:, 1] += a.dbias
        lc0 = torch.softmax(lg, -1).flip(-1).clamp_min(1e-9)  # (L, D, W)

        feat, score, _ = unpack(raw)
        tW, tD, tL = teacher(score, feat)
        tea = torch.stack([tL, tD, tW], 1).clamp_min(1e-9)

        cls = raw[:, 26].long()                              # 0 L, 1 D, 2 W
        z = cls.float() / 2.0
        m = len(cls)

        def recombine(E, D):
            """The distribution with this scalar and this draw mass.
            D cannot exceed 2*min(E, 1-E) or a wing goes negative."""
            D = torch.minimum(D, 2 * torch.minimum(E, 1 - E) * 0.999)
            return torch.stack([1 - E - D / 2, D, E - D / 2], 1).clamp_min(1e-9)

        Et = (tea[:, 2] + tea[:, 1] / 2).clamp(1e-6, 1 - 1e-6)
        El0 = (lc0[:, 2] + lc0[:, 1] / 2).clamp(1e-6, 1 - 1e-6)
        hyb = recombine(Et, lc0[:, 1])       # Stockfish's scalar, lc0's D
        rev = recombine(El0, tea[:, 1])      # lc0's scalar, the sigmoid's D

        for name, p in (("lc0", lc0), ("teacher", tea), ("hybrid", hyb), ("reverse", rev)):
            ce[name] += -p.log().gather(1, cls[:, None]).sum().item()
            E = (p[:, 2] + p[:, 1] / 2).clamp(1e-6, 1 - 1e-6)
            onll[name] += -(z * E.log() + (1 - z) * (1 - E).log()).sum().item()

        # lc0's accuracy wearing the incumbent's shape
        El = (lc0[:, 2] + lc0[:, 1] / 2).clamp(1e-6, 1 - 1e-6)
        logit = torch.log(El / (1 - El))
        for i, k in enumerate(ks):
            W2, D2, L2 = teacher(k * logit, feat)
            p2 = torch.stack([L2, D2, W2], 1).clamp_min(1e-9)
            ce_k[i] += -p2.log().gather(1, cls[:, None]).sum()

        if a.fit:
            cal = (lg[None, None] / TS[:, None, None, None])
            cal = cal + DB[None, :, None, None] * torch.tensor(
                [0.0, 1.0, 0.0], device=dev)
            q = torch.log_softmax(cal, -1).flip(-1)
            fit_ce -= q.gather(3, cls[None, None, :, None].expand(
                len(TS), len(DB), -1, 1)).sum((2, 3))

        b = (El * NB).long().clamp(0, NB - 1)
        tab.index_add_(0, b, torch.stack(
            [torch.ones(m, device=dev), (cls == 1).float(), lc0[:, 1], tea[:, 1]], 1).double())
        n += m

    print(f"n = {n:,}   {a.data} [{lo}:{hi}]")
    if a.fit:
        f = fit_ce / n
        i, j = divmod(int(f.argmin()), f.shape[1])
        print(f"  best calibration: --temp {TS[i]:.2f} --dbias {DB[j]:.2f}  "
              f"-> lc0 ce_out {f[i, j]:.5f} (was {f[TS.sub(1).abs().argmin(), DB.abs().argmin()]:.5f})")
    print(f"  {'teacher':<28} {'ce_out':>9} {'onll':>9}")
    for k in names:
        print(f"  {k:<28} {ce[k]/n:>9.5f} {onll[k]/n:>9.5f}")
    best = int(ce_k.argmin())
    print(f"  {'lc0 E -> sigmoid, K=' + str(int(KS[best])):<28} {ce_k[best].item()/n:>9.5f}")
    print(f"\n  scalar: lc0's E costs {(ce_k[best].item()/n - ce['teacher']/n):+.5f} "
          f"against Stockfish's, both wearing the sigmoid")
    print(f"  shape : lc0's D buys  {(ce_k[best].item()/n - ce['lc0']/n):+.5f} "
          f"over a sigmoid of the same E   (positive = lc0's shape is better)")
    print("\n  draw calibration, binned by lc0's E")
    print(f"  {'E bin':>10} {'n':>10} {'actual D':>9} {'lc0 D':>9} {'teacher D':>10}")
    t = tab.cpu().numpy()
    for i in range(NB):
        if t[i, 0] < 100:
            continue
        print(f"  {i/NB:.1f}-{(i+1)/NB:.1f} {t[i,0]:>10,.0f} {t[i,1]/t[i,0]:>9.3f} "
              f"{t[i,2]/t[i,0]:>9.3f} {t[i,3]/t[i,0]:>10.3f}")


if __name__ == "__main__":
    main()
