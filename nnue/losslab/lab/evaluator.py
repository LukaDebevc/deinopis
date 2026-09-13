"""Train one router arm per (candidate, seed) by running nnue/train.py.

Why a subprocess rather than importing train.py: a candidate loss that leaks
CUDA memory, spins, or corrupts autograd state takes down one process instead
of the search. The timeout in the parent is the backstop. It also means the
numbers here come from the SAME code path as every LEDGER entry -- the search
cannot quietly diverge from the way the engine is really trained.
"""

import json
import os
import subprocess
import tempfile

BASELINE_AUX = {
    "none": "",
    "tv=3e-3": "tv=3e-3",
    "tv=1e-2": "tv=1e-2",
    "ginfo=1e-3": "ginfo=1e-3",
    "ginfo=3e-3,tv=3e-2": "ginfo=3e-3,tv=3e-2",
}


def _cmd(cfg, steps, seed, aux, val_cap, out_json, loss_file="", only="st all"):
    flat = bool(getattr(cfg, "router_flat", False))
    if flat and only == "st all":
        # "st all" is a substring of BOTH arm names, so it would match two arms
        # and train_py takes the first. Name the flat arm explicitly.
        only = f"flat{1 << cfg.router_bits}"
    c = [cfg.python, cfg.train_py,
         "--data", cfg.data,
         "--experiment", "router_arms",
         "--steps", str(steps), "--batch", str(cfg.batch),
         "--lr", str(cfg.lr), "--device", cfg.device, "--seed", str(seed),
         "--log-every", str(steps), "--val-cap", str(val_cap),
         "--allow-repeat",
         "--router-modes", cfg.router_mode,
         "--router-inputs", cfg.router_input,
         "--router-bits", str(cfg.router_bits),
         "--router-gain", "1",
         "--router-lrmult", str(cfg.router_lrmult),
         "--tape", cfg.tape,
         "--tape-train-hi", str(cfg.tape_train_hi),
         "--router-aux", aux,
         "--router-json", out_json,
         "--only", only]
    if getattr(cfg, "router_rank", 0):
        c += ["--router-rank", str(int(cfg.router_rank))]
    if getattr(cfg, "router_wd", 0.0):
        c += ["--router-wd", str(cfg.router_wd)]
    if flat:
        c += ["--router-flat"]
    if getattr(cfg, "router_norm", "none") != "none":
        c += ["--router-norm", cfg.router_norm]
        if cfg.router_norm in ("white", "stat"):
            c += ["--router-eigfloor", str(getattr(cfg, "router_eigfloor", 1e-2))]
    if getattr(cfg, "router_pretrain", 0):
        c += ["--router-pretrain", str(int(cfg.router_pretrain)),
              "--router-pretrain-lr", str(cfg.router_pretrain_lr)]
    if getattr(cfg, "router_merge", ""):
        c += ["--router-merge", cfg.router_merge]
    if loss_file:
        c += ["--router-loss-file", loss_file]
    if cfg.init_psqt and os.path.exists(cfg.init_psqt):
        c += ["--init-psqt", cfg.init_psqt]
    return c


def run_one(cfg, *, steps, seed, aux, loss_file="", val_cap=None, only="st all"):
    """-> (metrics dict | None, error string). One arm, one seed."""
    val_cap = val_cap or cfg.dev_val_cap
    fd, out_json = tempfile.mkstemp(suffix=".json", dir=cfg.workdir)
    os.close(fd)
    cmd = _cmd(cfg, steps, seed, aux, val_cap, out_json, loss_file, only)
    try:
        pr = subprocess.run(cmd, capture_output=True, timeout=cfg.timeout,
                            cwd=os.path.dirname(cfg.train_py))
    except subprocess.TimeoutExpired:
        os.unlink(out_json)
        return None, f"timeout after {cfg.timeout}s"
    try:
        with open(out_json) as f:
            d = json.load(f)
    except Exception:
        err = (pr.stderr or b"").decode()[-1800:]
        os.unlink(out_json) if os.path.exists(out_json) else None
        return None, err.strip() or "train.py produced no JSON"
    os.unlink(out_json)
    if not d:
        return None, "train.py matched no arm"
    m = next(iter(d.values()))
    # The whitened router's logit covariance spectrum, eigenvalue / mean,
    # descending. train.py's stdout is captured, so this is the only place it
    # can surface; `average` drops it, since it is a vector per seed and not a
    # ranking field. It says what --router-eigfloor has to work with: the floor
    # lifts everything below f, and leaves everything above it whitened.
    sp = m.pop("spectrum", None)
    if sp:
        head = " ".join(f"{v:.3g}" for v in sp[:6])
        tail = " ".join(f"{v:.3g}" for v in sp[-4:])
        print(f"    spectrum seed {seed}: max/mean {sp[0]:.3g} "
              f"min/mean {sp[-1]:.3g}  {sum(1 for v in sp if v < 1)}/{len(sp)} "
              f"below mean\n      top {head} ... bottom {tail}", flush=True)
    # A candidate that drove the loss to NaN reports a NaN val, not a crash.
    for k in ("val", "flip2"):
        v = m.get(k)
        if v is None or v != v:
            return None, f"{k} is {v} -- the penalty destabilised training"
    return m, ""


def average(runs):
    """Mean over seeds of the fields that rank an arm; min/max kept for spread."""
    keys = ("val", "hard", "soft", "flip2", "I_game", "eff", "occ")
    out = {k: sum(r[k] for r in runs) / len(runs)
           for k in keys if all(k in r for r in runs)}
    out["n_seeds"] = len(runs)
    out["val_spread"] = max(r["val"] for r in runs) - min(r["val"] for r in runs)
    out["flip2_spread"] = (max(r["flip2"] for r in runs)
                           - min(r["flip2"] for r in runs))
    return out
