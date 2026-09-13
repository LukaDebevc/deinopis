# `nnue/lc0` — relabelling the corpus with a Leela net

## What this is

Our training records carry **one Stockfish centipawn score and one game
outcome** per position, and the "WDL teacher" they are turned into is a fixed
sigmoid of that scalar (LEDGER 048). So the three-way target contains no
information the scalar did not, and `D` in particular is a function of
(score, material) alone — nothing in the label can say *this* endgame is drawn.

This directory computes a second opinion: the WDL a real Leela net gives the
same positions. The question it exists to answer is whether that extra
structure is worth anything to the student, and a negative answer is a useful
answer.

The teacher is `512x15-t79_9-swa-2016000`, from lczero.org's `more_nets`. It is
**a net from run T79, which is the run that generated these very games** — the
corpus is `linrock/test79`, Leela self-play that was then rescored by
Stockfish. So this is close to asking what would have happened had Leela's own
labels been kept.

## What is live

| file | what it does |
|---|---|
| `pbwire.py` | 60-line protobuf wire reader; no `protoc`, no runtime dep |
| `loadw.py` | `.pb.gz` -> numpy; LINEAR16 dequantisation |
| `net.py` | the SE-resnet value head in PyTorch |
| `planes.py` | our 32-byte records + the `--aux` side file -> lc0's 112 planes, on the GPU |
| `encode.py` | FEN -> the same 112 planes. Verification only |
| `label.py` | the batch pass: writes `(L, D, W)` float16 per record, and scores teachers |

## Two things that were checked rather than assumed

**The model.** Everything in `net.py` is transcribed from lc0's own source, not
from memory — in particular `bn_stddivs` holds the **variance**, despite the
name, and the SE unit folds conv2's bias into both of its outputs. The built
net gives startpos W .367 / D .433 / L .201, mirrored queen odds .995 and .988
the other way, and **bare kings D .999**. That last row is the whole argument in
one number: our sigmoid teacher cannot express it.

**The encoder.** `planes.py` (Rust bit-packing plus a GPU argsort decode) and
`encode.py` (Python FEN-string parsing) share no code. On 4000 real records
they produce **bit-identical planes**, which is what makes the black mirror,
the castling side-assignment, the rule-50 plane and the square order trusted
rather than hoped for.

## Known approximations

* **No history.** The 112-plane format wants 8 plies; our records are single
  positions, so all 8 steps repeat the current one. That is lc0's own behaviour
  when handed a bare FEN (`FillEmptyHistory::FEN_ONLY`), but it is off the
  distribution the net was trained on and its cost is not yet measured.
* **No en passant.** The classical format carries it only through history.
  `--aux` keeps a bit saying an en-passant capture was available, so the cost
  can be measured later without re-extracting.

## Cost

15,064 pos/s on one A100 (fp16, batch 4096, channels-last), i.e. **~55 min per
50M positions** and ~16 h for the whole 893.5M corpus. The GPU is the
bottleneck by a wide margin: `nnue/extract` decodes the binpack at 4.4M
entries/s on one core.
