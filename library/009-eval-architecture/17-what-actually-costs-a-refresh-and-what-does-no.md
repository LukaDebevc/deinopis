## What actually costs a refresh (and what does not)

Worth stating plainly, because a whole day went into pricing rules and the
models we have trained do not pay that price. `Bucketed` is

    eval = psqt(x) + < act(V x), R[b(x)] >

and **`a = Vx` does not depend on the bucket.** So:

| conditioning | what a bucket change costs |
|---|---|
| `readers` — `R[b]` | nothing. Read a different row. |
| `shifts` (`pre=True`) — `a + shift[b]` | one 512-wide vector swap, `a − shift[old] + shift[new]`. ~10% of what an ordinary piece move already costs. |
| `pq` — bucketed PSQT | nothing. The linear term has no accumulator; it is 32 gathers either way. |
| `vbuck` — bucket-indexed FEATURE table | **a real refresh.** Those entries must be rebuilt from all ~32 features. |

Only the last one pays, and it is the only one never trained. Every price in
LEDGER 031/034 — HalfKP 40.34%/ply, material 24^2 22.55%/ply — is a price for
`vbuck`-style conditioning. It bears on `merged`, on any covering-with-hysteresis
scheme, and on HalfKP itself; it does not bear on anything in the study's
results tables.

Which reorders the work. Read-side conditioning is worth **−13.4%** and is
free; king routing costs the most and saturates at **−5.8%**. The cheap thing
is also the better thing, on our data. `vbuck` is the experiment that says
whether accumulator-side conditioning is worth paying for at all, and until it
reports, designing cheaper rules for it is optimising a term of unknown sign.
