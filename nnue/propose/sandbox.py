"""Accept or reject a generated rule, and give every reason a name.

A rejection is a result: the reasons are counted per round and fed back into the
next prompt, which is the only thing that stops a model re-proposing the same
broken idea twenty times.

Nothing here trusts the generated text. The source is parsed and walked before
it runs: no imports, no dunders, no builtins beyond a handful of pure ones. The
rule can therefore only reach the board through `prims.Board`, and the Board is
where the cost accounting and the read tracking live.
"""
import ast
import builtins
import hashlib

import torch

import prims

ALLOWED_CALLS = {"range", "len", "min", "max", "abs", "int", "sum", "list", "enumerate", "zip"}
BANNED_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda,
                ast.While, ast.Try, ast.With, ast.AsyncFunctionDef, ast.Await, ast.Yield)
MAX_SIZE = 4096          # a bucket table is r floats wide; beyond this is silly
MAX_COST = 200           # bitboard ops per recompute
MIN_PERP = 2.0           # a rule that always answers the same thing is not a rule
MAX_SHARE = 0.90         # nor is one where 90% of positions land in one bucket


class Reject(Exception):
    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def check_source(src):
    """Static gate. Runs before anything is executed."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise Reject("syntax", str(e))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != len(fns) or len(fns) != 1 or fns[0].name != "rule":
        raise Reject("shape", "the source must be exactly one `def rule(b):`")
    if [a.arg for a in fns[0].args.args] != ["b"]:
        raise Reject("shape", "rule must take exactly one argument, `b`")
    for node in ast.walk(tree):
        if isinstance(node, BANNED_NODES):
            raise Reject("banned", type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise Reject("banned", node.id)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise Reject("banned", node.attr)
            if not (isinstance(node.value, ast.Name) and node.value.id == "b"):
                raise Reject("banned", f"attribute on something other than b: .{node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_CALLS:
                raise Reject("banned", f"call to {node.func.id}")
    return tree


def compile_rule(src):
    check_source(src)
    env = {"__builtins__": {k: getattr(builtins, k) for k in ALLOWED_CALLS}}
    try:
        exec(compile(src, "<rule>", "exec"), env)
    except Exception as e:
        raise Reject("compile", f"{type(e).__name__}: {e}")
    return env["rule"]


def run(fn, feat, max_cost=MAX_COST):
    """Execute a rule on one batch. Returns (index, size, cost, reads)."""
    b = prims.Board(feat, max_cost=max_cost)
    try:
        out = fn(b)
    except prims.Budget as e:
        raise Reject("too_expensive", str(e))
    except Reject:
        raise
    except Exception as e:
        raise Reject("runtime", f"{type(e).__name__}: {e}")
    if not (isinstance(out, tuple) and len(out) == 2):
        raise Reject("contract", "rule must return (index, size) -- use b.combine([...])")
    idx, size = out
    if not torch.is_tensor(idx) or idx.shape != (feat.shape[0],):
        raise Reject("contract", f"index must be one integer per position, got {type(idx)}")
    idx = idx.long()
    size = int(size)
    if size < 2:
        raise Reject("degenerate", "declared size < 2")
    if size > MAX_SIZE:
        raise Reject("too_big", f"declared {size} buckets, ceiling is {MAX_SIZE}")
    if int(idx.min()) < 0 or int(idx.max()) >= size:
        raise Reject("out_of_range", f"index in [{int(idx.min())}, {int(idx.max())}] for size {size}")
    return idx, size, b.cost, sorted(b.reads)


def balance(idx, size):
    """Perplexity of the visit distribution, and the largest bucket's share."""
    c = torch.bincount(idx, minlength=size).double()
    p = c / c.sum()
    nz = p[p > 0]
    return float(torch.exp(-(nz * nz.log()).sum())), float(p.max())


def signature(idx):
    """A hash of the PARTITION, with bucket labels canonicalised.

    Two rules that split the probe positions the same way are the same rule, no
    matter what arithmetic produced the labels or what the proposer called it.
    This is what stops the pool filling up with restatements."""
    seen = {}
    lab = []
    for v in idx.tolist():
        if v not in seen:
            seen[v] = len(seen)
        lab.append(seen[v])
    return hashlib.sha1(bytes(str(lab), "utf8")).hexdigest()[:16]


def accept(src, feat, known=(), max_cost=MAX_COST):
    """The whole gate. Raises Reject, or returns a record ready to be scored."""
    fn = compile_rule(src)
    idx, size, cost, reads = run(fn, feat, max_cost)
    again = run(fn, feat[:256], max_cost)[0]
    if not torch.equal(again, idx[:256]):
        raise Reject("nondeterministic", "two runs disagreed")
    perp, share = balance(idx, size)
    if perp < MIN_PERP:
        raise Reject("degenerate", f"effective buckets {perp:.2f} < {MIN_PERP}")
    if share > MAX_SHARE:
        raise Reject("degenerate", f"largest bucket holds {share:.1%}")
    sig = signature(idx[:4096])
    if sig in known:
        raise Reject("duplicate", f"same partition as {known[sig]}"
                     if isinstance(known, dict) else "already in the pool")
    return {"size": size, "cost": cost, "reads": reads,
            "perplexity": round(perp, 1), "top_share": round(share, 4), "sig": sig}
