"""Minimal protobuf wire-format reader — enough to open an lc0 .pb.gz.

No protoc, no `protobuf` runtime. lc0's `net.proto` is proto2 and we only ever
read, so a generic wire walk plus the field numbers from that file is the whole
job. Nested messages stay as raw bytes until someone asks for them, which keeps
a 37 MB net cheap to inspect.

Returned shape: {field_number: [value, ...]} — always a list, because proto2
`repeated` and `optional` look identical on the wire.
"""
import gzip, struct


def walk(buf):
    """Parse one message body into {field_number: [raw value, ...]}."""
    out, i, n = {}, 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fld, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4        # fixed32: caller decodes
        else:
            raise ValueError(f'wire type {wt} at {i}')
        out.setdefault(fld, []).append(v)
    return out


def _varint(buf, i):
    r = s = 0
    while True:
        b = buf[i]; i += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, i
        s += 7


def load(path):
    with gzip.open(path, 'rb') as f:
        return walk(f.read())
