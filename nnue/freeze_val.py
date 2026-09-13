"""Write the validation tail of a pool to a file of its own.

The point is comparability. `loader.Batcher` builds its val set as the tail of
`--data`, so appending a month to the pool silently replaces the held-out set:
nothing errors, and every number ever recorded quietly stops meaning the same
thing. The pool is meant to grow -- LEDGER 021's conclusion is to buy distinct
games rather than extra passes -- so the val set has to stop being a function
of it.

This writes exactly the records `Batcher` would have selected:

    mm[n_all - n_val * val_stride :: val_stride]

so a run using `--val-file` is directly comparable to every run measured
before the freeze. Verify that with `--check`, which recomputes the selection
and compares it byte for byte.

    python3 tools/freeze_val.py --data /path/all.data --out val_frozen.data
    python3 tools/freeze_val.py --data /path/all.data --out val_frozen.data --check
"""

import argparse
import numpy as np

REC_BYTES = 32


def select(path, val_cap, val_frac, val_stride):
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    assert len(mm) % REC_BYTES == 0, "file is not a whole number of records"
    mm = mm.reshape(-1, REC_BYTES)
    n = len(mm)
    n_val = min(int(n * val_frac), val_cap)
    span = n_val * val_stride
    return mm, mm[n - span::val_stride], n, n_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-cap", type=int, default=1_000_000)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-stride", type=int, default=40)
    ap.add_argument("--check", action="store_true",
                    help="compare an existing --out against the live selection")
    a = ap.parse_args()

    mm, val, n, n_val = select(a.data, a.val_cap, a.val_frac, a.val_stride)
    print(f"pool        {n:,} records ({n * REC_BYTES / 1e9:.2f} GB)")
    print(f"val span    last {n_val * a.val_stride:,} records, every {a.val_stride}")
    print(f"val         {len(val):,} records ({len(val) * REC_BYTES / 1e6:.1f} MB)")
    print(f"pool after  {n - n_val * a.val_stride:,} trainable records")
    print(f"\n  use:  --val-file {a.out} --pool-end {n - n_val * a.val_stride}")
    print("  (--pool-end is REQUIRED: --val-file frees the tail back into the\n"
          "   pool, so without it the run trains on its own validation set.)")

    if a.check:
        got = np.memmap(a.out, dtype=np.uint8, mode="r").reshape(-1, REC_BYTES)
        same = got.shape == val.shape and bool((np.asarray(got) == np.asarray(val)).all())
        print(f"\n{a.out}: {'MATCHES' if same else 'DIFFERS FROM'} the live selection")
        raise SystemExit(0 if same else 1)

    with open(a.out, "wb") as f:
        f.write(np.ascontiguousarray(val).tobytes())
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
