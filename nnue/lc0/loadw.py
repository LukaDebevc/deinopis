"""Decode an lc0 .pb.gz into numpy arrays.

Every tensor in the file is a `Weights.Layer`: LINEAR16, i.e. uint16 values
rescaled onto [min_val, max_val]. `dims` is usually absent, so shapes come from
the architecture, not the file — which is why `net.py` asserts every count.
"""
import numpy as np
import struct
from .pbwire import load, walk

_F = lambda b: struct.unpack('<f', b)[0]


def layer(buf):
    d = walk(buf)
    p = np.frombuffer(d[3][0], dtype='<u2').astype(np.float32)
    mn, mx = _F(d[1][0]), _F(d[2][0])
    return mn + p * ((mx - mn) / 65535.0)


def convblock(buf):
    d = walk(buf)
    # 1=weights 2=biases 3=bn_means 4=bn_stddivs 5=bn_gammas 6=bn_betas
    return {n: layer(d[f][0]) for f, n in
            ((1, 'w'), (2, 'b'), (3, 'mean'), (4, 'std'), (5, 'gamma'), (6, 'beta'))
            if f in d}


def read(path):
    """-> (network_format dict, weights dict-of-raw-message-bytes)"""
    net = load(path)
    nf = walk(walk(net[4][0])[2][0])
    return {k: v[0] for k, v in nf.items()}, walk(net[10][0])
