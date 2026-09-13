"""Zero penalty, but the tape forward still happens.

The control for one question: does forwarding the game tape through the router
in TRAINING mode change the deployed rule, independently of any gradient? It
can, because LogitNorm/BatchNorm update their running mean and covariance on
every training forward, and the tape is a different distribution from the
shuffled batch -- unfiltered tono_games against filtered all.data, and 128
games' worth of 64 consecutive plies rather than 8192 independent positions.
Those running buffers are what eval() routes with.

Weight is set to 0 by the caller, so the gradient is identically zero and this
arm differs from `none` in the BN statistics and nothing else.
"""


def penalty(ctx):
    return ctx["tp"].sum() * 0.0
