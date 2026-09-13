"""Run configuration for the router-loss search.

The two numbers that decide everything are `screen_steps` and `dev_steps`.
Measured on this box (RTX 4060 Ti, batch 65536, one router arm with a game
tape): 300 steps = 15 s, 1526 steps = 60 s.  So a candidate costs

    screen (1 seed, 300 steps)      ~15 s
    dev    (dev_seeds x 1526 steps) ~60 s x seeds

and a day of searching is roughly 500 fully-evaluated candidates.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os

CHESS = Path(__file__).resolve().parents[3]          # .
LAB = Path(__file__).resolve().parents[1]            # .../nnue/losslab


def _key(env, fname):
    """Environment first, then a one-line file next to run.py.

    The file is the convenient form (`echo 'AIza...' > nnue/losslab/.key`) and
    survives a new shell; the env var wins so a one-off override is still easy.
    Both are gitignored.
    """
    v = os.environ.get(env, "").strip()
    if v:
        return v
    f = LAB / fname
    try:
        return f.read_text().strip()
    except OSError:
        return ""


@dataclass
class RunConfig:
    # -- what to train ----------------------------------------------------
    python: str = "python3"
    train_py: str = str(CHESS / "nnue" / "train.py")
    data: str = "data/all.data"
    init_psqt: str = str(CHESS / "nnue" / "ckpt-big" / "psqt_768-_64-_16-_1.pt")
    batch: int = 65536
    lr: float = 1e-3
    router_bits: int = 6
    router_mode: str = "st"
    router_input: str = "all"
    router_lrmult: float = 100.0
    # Factorise the router matrix as (rows, r) @ (r, k). 0 = off, which is
    # every arm measured so far. r >= k is the same function class and the same
    # deployed rule -- only the gradient path differs.
    router_rank: int = 0
    # Weight decay on the router parameters only. With router_rank > 0 this is
    # nuclear-norm regularisation on the product, so the rule goes low-rank.
    router_wd: float = 0.0
    # Softmax over nb buckets instead of nb sign bits: one (rows, 64) matrix
    # and argmax, rather than 6 hyperplanes whose sign pattern names the
    # bucket. 64x the router parameters, an unconstrained partition instead of
    # a product of half-spaces, and ONE decision boundary per position (top-1
    # vs top-2) instead of six.
    router_flat: bool = False
    # Two-stage: fit the ROUTER ALONE to its penalty for this many tape steps,
    # then FREEZE it and fit the tables to it. 0 = joint training, which is
    # every arm measured before 2026-09-03. The point is that the two gradients
    # fight -- the prediction loss always gains a little by splitting on
    # something that changes move to move, the penalty wants the opposite -- so
    # trained jointly the result is a truce set by the penalty weight. Frozen,
    # the deployed rule's stability is a property of stage 1 and stage 2 cannot
    # trade it away.
    # none | center | bn | white. Normalise the router logits before the
    # threshold; all of it folds into the rule matrix at export, so the engine
    # pays nothing. center gives every bit a 50/50 split (the only part of
    # BatchNorm that can move the partition, as sign(z/sigma) = sign(z));
    # white additionally decorrelates them, which is what makes all 2^k
    # buckets about equally likely.
    router_norm: str = "none"
    # white only: eigenvalue floor relative to the mean eigenvalue, before
    # Cov^-1/2. The dial between the two ends: 0 whitens exactly (even buckets,
    # flat directions promoted to bits that flip every move), >=1 clamps every
    # eigenvalue to the mean and so degenerates to center.
    router_eigfloor: float = 1e-2
    router_pretrain: int = 0
    router_pretrain_lr: float = 1e-2
    # After training, k-means the learned bucket tables down this ladder and
    # re-measure at every rung. The router is untouched; only how many distinct
    # tables its buckets point at shrinks, so two buckets sharing a table stop
    # counting as a flip. Nested: each rung clusters the previous rung's
    # centroids.
    router_merge: str = ""
    tape: str = "128x64"
    # The penalty trains on the first 80% of the game tape; flip2 is scored on
    # the last 20%. A candidate cannot win by memorising the games it saw.
    tape_train_hi: float = 0.8

    # -- staging ----------------------------------------------------------
    screen_steps: int = 300
    screen_seeds: tuple = (0,)
    screen_val_cap: int = 500_000
    dev_steps: int = 1526                # 100M positions, the LEDGER 046 setting
    # 3, not 2: the noise floor came out at 1.13 points of flip2 for a single
    # run, so a 2-seed mean can only resolve a 2.3-point flip2 change and `tv`
    # only moves it by 2.6. A third seed costs a minute and buys the resolution
    # the frontier needs to not be an artefact.
    dev_seeds: tuple = (0, 1, 2)
    dev_val_cap: int = 1_000_000
    # Replicates of the `none` baseline used to size the noise bars. A margin
    # smaller than this is not a Pareto improvement, it is a different RNG draw.
    noise_seeds: tuple = (0, 1, 2, 3)
    # Fallback margins if the noise stage is skipped (relative on val, absolute
    # on flip2). Overwritten by the measured spread.
    eps_val: float = 0.0002
    eps_flip: float = 0.01
    # A screened candidate is dropped only if it is CLEARLY dominated by the
    # no-penalty baseline: worse on both by this many noise widths. Loose on
    # purpose -- the 300-step ranking is a cheap proxy for the 1526-step one and
    # has not been shown to preserve order.
    screen_slack: float = 3.0
    # A router that stopped routing scores flip2 = 0, which nothing can beat, so
    # it would sit on the frontier for ever and seed every later refine and
    # synthesis. That is not a cheap-refresh option, it is the architecture
    # switched off -- `none` already covers "no penalty" with a router that
    # works. Measured: every legitimate arm so far sits at eff 37-53 of 64, the
    # collapse at eff 1.0, so the gate is nowhere near a real arm.
    min_eff: float = 4.0
    # The SCREEN runs 300 steps against dev's 1526, and the router has not
    # finished spreading out that early, so eff read at the screen is not on
    # the same scale as eff read at dev. Measured on one arm carried over from
    # the archive -- identical code -- screen eff 3.96 against a 3-seed dev eff
    # of 12.5, with 63 of 64 buckets occupied at the screen. Calling that "the
    # router stopped routing" is wrong, and the arm was dropped for it. The
    # screen gate exists only to catch the extreme case cheaply; the real gate
    # is at dev, where the number means what the name says.
    screen_min_eff: float = 2.0
    # Variants of one KNOBS family that get the full dev treatment. The screen
    # is 1 seed at screen_steps and has NOT been shown to preserve the dev
    # order, so it only ALLOCATES dev time inside a family -- it never ranks an
    # arm. Everything on the board still got dev_seeds x dev_steps.
    dev_top_k: int = 2
    # The region we actually want, as a BAND rather than a point. Everything
    # measured so far sits at flip2 16-42%; king6, the best hand rule, is 18%.
    # So the band is below every arm on the board and the ordering inside it is
    # "as low as it will go while eff stays up". The low edge is not a floor to
    # stop at, it is there so an arm that reaches 2% is treated as ON target and
    # refined to get CHEAPER, instead of being pushed further down an axis that
    # has stopped paying.
    #
    # This steers WHERE proposals are spent -- which front member gets refined,
    # which pair gets crossed, what the model is told to aim at. It never
    # touches `dominates`: an arm is still scored against every other arm with
    # no exchange rate between val and flip2, because that rate is the thing
    # the search is meant to inform. Favouring the band in the RANKING would be
    # exactly the scalarisation this design refuses.
    flip2_lo: float = 0.02
    flip2_hi: float = 0.10
    # How sharply the refine draw favours the arms nearest the band. Weight
    # for the i-th nearest is refine_decay**i, so at 0.6 with an 11-member
    # front the two closest arms take ~64% of refines and the 40%-flip2 tail
    # keeps a couple of percent -- reachable, so the search can still be
    # surprised, but no longer where most of the budget goes. A linear ramp
    # (what this was) left 16% of refines on arms the prompt now calls
    # uninteresting.
    refine_decay: float = 0.6
    # Arms below this are still ranked and still shown -- they are just not used
    # as refine or synthesis SEEDS, because building on an arm that is already
    # most of the way to a collapsed router walks further into the rejection
    # gate. Sited in the gap between the measured arms: 53, 51, 49, 47, 44, 40,
    # 37 | 25, 24, 17, 10.
    steer_min_eff: float = 30.0
    # Submit the hand-written terms through the candidate interface at startup,
    # so the best arms on the board can be refined instead of only admired.
    seed_front: bool = False

    # -- search loop ------------------------------------------------------
    max_candidates: int = 400
    fix_attempts: int = 1
    refine_every: int = 3                # deepen a front member
    synth_every: int = 5                 # cross two front members
    max_variants: int = 4                # expand a literal KNOBS grid
    n_explore_parallel: int = 6
    buffer_size: int = 12
    hours: float = 0.0                   # >0 = stop after this much wall clock

    # -- plumbing ---------------------------------------------------------
    device: str = "cuda"
    timeout: float = 1800.0
    workdir: str = str(LAB / "results")
    cand_dir: str = str(LAB / "candidates")
    log_path: str = str(LAB / "losslab_run.jsonl")
    resume: bool = False
    quiet: bool = False

    # -- LLM --------------------------------------------------------------
    google_api_key: str = field(
        default_factory=lambda: _key("GOOGLE_API_KEY", ".key"))
    openrouter_api_key: str = field(
        default_factory=lambda: _key("OPENROUTER_API_KEY", ".key-openrouter"))
    google_models: tuple = ("gemini-flash-latest", "gemma-4-31b-it",
                            "gemma-4-26b-a4b-it")
    llm_cooldown_s: float = 25.0
