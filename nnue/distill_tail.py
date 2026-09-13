"""Distill a small tail from a trained WDL net, sharing its frozen accumulator.

Rung 3 of the dual-net plan (LEDGER 077/078): quiescence runs a cheap tail
while the main search keeps the full net. The noise probe priced disagreement
at σ=25 ≈ −12 Elo against a 1.29x speedup worth ~+25–30, so a tail that agrees
with the teacher much better than white noise is worth training.

Setup: the student keeps the teacher's ft table + ft_bias EXACTLY (copied,
frozen — one accumulator, two tails) and trains everything downstream on KL
to the teacher's own logits. No labels needed: agreement is the target, not
absolute quality, so this runs on the local binpack. psqt is trained, not
frozen — it is tail arithmetic (a gather + add per eval), not the accumulator.

Loss is KL; the reported numbers are cp disagreement vs the teacher (RMSE /
MAE / p99, exactly as `quant.measure` computes it) plus ce_out against the
game outcome for reference. The judge is the real-dual SPRT, not any of these.

    nnue/.venv/bin/python nnue/distill_tail.py \
        --teacher nnue/runs/b1-20260909/m1-b1.pt \
        --data data/all.data \
        --val-file data/val_frozen.data \
        --pool-end 853500985 --steps 3000 --out nnue/runs/distill/tail-b8.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from loader import Batcher                      # noqa: E402
from wdlnet import WdlNet, ce, expected, target_dist  # noqa: E402
from export_wdl import check as export_check    # noqa: E402


def cp_of_logits(logits, k):
    E = expected(logits).clamp(1e-6, 1 - 1e-6)
    return k * torch.log(E / (1 - E))


@torch.no_grad()
def evaluate(student, teacher, cache, k):
    """(KL student||teacher, cp rmse, cp p99, cp mae, student ce_out)."""
    student.eval()
    n = 0
    kl = out = 0.0
    de = []
    for feat, s, z, tcp in cache:
        lg = student(feat)
        tgt = F.softmax(teacher(feat), -1)
        kl += ce(lg, tgt).item() * len(feat)
        out += ce(lg, target_dist(s, z, None, feat, 0.0)).item() * len(feat)
        de.append(cp_of_logits(lg, k) - tcp)
        n += len(feat)
    student.train()
    d = torch.cat(de)
    return (kl / n, d.pow(2).mean().sqrt().item(),
            d.abs().quantile(0.99).item(), d.abs().mean().item(), out / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", default="data/all.data")
    ap.add_argument("--val-file", dest="val_file",
                    default="data/val_frozen.data")
    ap.add_argument("--pool-end", dest="pool_end", type=int, default=853_500_985)
    ap.add_argument("--steps", type=int, default=3000,
                    help="3000 x 16384 ≈ 50M positions for the feel-run")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--block", type=int, default=1_000_000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--bneck", type=int, default=8,
                    help="student waist; teacher is 16 (halves down/up reads)")
    ap.add_argument("--hidden", type=int, default=32,
                    help="kept at the teacher's width for the feel-run")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mse-lambda", dest="mse_lambda", type=float, default=0.0,
                    help="fat-tail loss weight (LEDGER 081 item 1): loss = KL + "
                         "lambda * mean((cp_s - cp_t)^2) in cp units. rmse 70 "
                         "-> MSE ~4900, so 1e-4 prices it ~0.5 vs KL ~0.73. "
                         "0 = pure KL (079/080).")
    ap.add_argument("--wd", type=float, default=0.0,
                    help="0: fidelity objective, nothing to regularise toward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm-start", action="store_true",
                    help="copy every same-shaped teacher tensor as init (l2, "
                         "head, skiph, psqt when hidden matches; the waist "
                         "layers differ in shape and stay fresh). Copied "
                         "weights stay trainable — this is init, not freezing.")
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--val-cap", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tck = torch.load(args.teacher, map_location=dev, weights_only=False)
    k = tck.get("k_cp") or 288.5
    teacher = WdlNet(**tck["cfg"]).to(dev)
    teacher.load_state_dict(tck["state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    cfg = dict(tck["cfg"])
    cfg["bneck"] = args.bneck
    cfg["hidden"] = args.hidden
    export_check(cfg)  # fail now, not after a run, if the engine can't read this
    student = WdlNet(**cfg).to(dev)
    with torch.no_grad():
        student.ft.weight.copy_(teacher.ft.weight)
        student.ft_bias.copy_(teacher.ft_bias)
    student.ft.weight.requires_grad_(False)
    student.ft_bias.requires_grad_(False)
    if args.warm_start:
        tpar = dict(teacher.named_parameters())
        n_copied = 0
        with torch.no_grad():
            for name, p in student.named_parameters():
                t = tpar.get(name)
                if t is not None and t.shape == p.shape and p.requires_grad:
                    p.copy_(t)
                    n_copied += 1
        print(f"warm-start: copied {n_copied} same-shaped tensors from teacher",
              flush=True)
    n_frozen = sum(p.numel() for p in student.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"teacher {args.teacher}  bneck {tck['cfg']['bneck']} -> {args.bneck}, "
          f"hidden {args.hidden}, K {k}", flush=True)
    print(f"student params: {n_train:,} trainable, {n_frozen:,} frozen "
          f"(ft + ft_bias)", flush=True)

    b = Batcher(args.data, dev, batch=args.batch, block=args.block,
                seed=args.seed, order="seq", stride=args.stride,
                val_file=args.val_file, pool_end=args.pool_end, val_cap=1)
    print(b, flush=True)

    # Frozen val cache with the teacher's cp precomputed once.
    vb = Batcher(args.val_file, dev, batch=args.batch, val_file=args.val_file,
                 pool_end=1, val_cap=args.val_cap, val_frac=1.0)
    cache = []
    with torch.no_grad():
        for feat, s, z in vb.val_batches():
            if sum(len(f) for f, _, _, _ in cache) >= args.val_cap:
                break
            cache.append((feat, s, z, cp_of_logits(teacher(feat), k)))
    nval = sum(len(f) for f, _, _, _ in cache)
    t_ce = sum(ce(teacher(f), target_dist(s, z, None, f, 0.0)).item() * len(f)
               for f, s, z, _ in cache) / nval
    print(f"val {nval:,} positions  teacher ce_out {t_ce:.6f}", flush=True)

    train = [q for q in student.parameters() if q.requires_grad]
    decay = [q for q in train if q.dim() > 1]
    plain = [q for q in train if q.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": plain, "weight_decay": 0.0}], lr=args.lr)
    warm = max(1, int(args.steps * 0.02))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.02, 1.0, warm),
              torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps - warm)],
        milestones=[warm])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    curve = open(os.path.splitext(args.out)[0] + ".csv", "w", buffering=1)
    curve.write("step,positions,train_kl,train_mse,val_kl,cp_rmse,cp_p99,cp_mae,ce_out,lr,seconds\n")

    def save():
        torch.save({"cfg": cfg, "state": student.state_dict(), "seed": args.seed,
                    "arm": f"tail-b{args.bneck}", "k_cp": k,
                    "distill_from": args.teacher,
                    "frozen": ["ft.weight", "ft_bias"]}, args.out + ".tmp")
        os.replace(args.out + ".tmp", args.out)

    t0 = time.time()
    end = None
    try:
        for i, (feat, _s, _z) in enumerate(b.train_batches(args.steps), 1):
            with torch.no_grad():
                lg_t = teacher(feat)
                tgt = F.softmax(lg_t, -1)
            lg_s = student(feat)
            loss = ce(lg_s, tgt)
            mse_v = 0.0
            if args.mse_lambda > 0:
                # cp-space MSE targets the fat tails (p99) that grow the
                # tree via stand-pat cutoff errors, not the mean (081).
                mse_v = (cp_of_logits(lg_s, k) - cp_of_logits(lg_t, k)).pow(2).mean()
                loss = loss + args.mse_lambda * mse_v
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if i % args.every == 0 or i == args.steps:
                kl, rmse, p99, mae, ce_out = evaluate(student, teacher, cache, k)
                dt = time.time() - t0
                end = (kl, rmse, p99, mae, ce_out)
                curve.write(f"{i},{i * b.batch},{loss.item():.6f},"
                            f"{mse_v.item() if args.mse_lambda > 0 else 0.0:.2f},"
                            f"{kl:.6f},"
                            f"{rmse:.2f},{p99:.2f},{mae:.2f},{ce_out:.6f},"
                            f"{opt.param_groups[0]['lr']:.3e},{dt:.1f}\n")
                save()
                print(f"  step {i:>5}/{args.steps}  train-kl {loss.item():.5f}  "
                      f"val-kl {kl:.5f}  cp rmse {rmse:7.2f} mae {mae:6.2f} "
                      f"p99 {p99:7.2f}  ce_out {ce_out:.5f} (t {t_ce:.5f})  "
                      f"{i * b.batch / dt:,.0f} pos/s", flush=True)
    finally:
        curve.close()
        save()
    print(f"wrote {args.out}  " + (
        f"val-kl {end[0]:.5f} cp rmse {end[1]:.2f} mae {end[3]:.2f} p99 {end[2]:.2f}"
        if end else "no val point reached"))


if __name__ == "__main__":
    main()
