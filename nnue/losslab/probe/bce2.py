"""Luka's consecutive-move stability loss, written per BIT on the logits.

    L = mean_t mean_j  -[ b_{t,j} log s(z_{t+2,j}) + (1 - b_{t,j}) log(1 - s(z_{t+2,j} )) ]
    b_{t,j} = 1[z_{t,j} > 0]                                    (detached)

t and t+2 are the SAME side to move, which is why the pair is two plies and not
one: the binpack records are stm-canonical, so a one-ply pair would compare a
board with its own mirror and measure the flip the perspective swap causes
rather than the flip the move causes.

Relation to what is already on the board: in `bits` mode the bucket posterior
factorises, p(b) = prod_j s_j^{b_j} (1-s_j)^{1-b_j}, so

    -log p_{t+2}[ argmax_b p_t(b) ]  =  sum_j BCE(b_{t,j}, z_{t+2,j})

which is EXACTLY the `N` term of sweep_entropy's family, up to the 1/k. So this
is not a new penalty at 6 bits; it is N written in the form that makes the
per-bit structure visible, and it is the form that still means something in
`flat` mode where the two differ.
"""


def penalty(ctx):
    z, pi = ctx["tz"], ctx["pair"]
    a, b = z[pi], z[pi + 2]
    tgt = (a.detach() > 0).to(b.dtype)
    return ctx["F"].binary_cross_entropy_with_logits(b, tgt)
