"""AST gate for a proposed loss. Ported from LAB/new_optim/optim_search.

The candidate is a few lines of tensor algebra, so the whitelist is tighter
than the optimizer search needed: torch, math, numpy and nothing else. The
file is then executed inside train.py's own process, so this gate plus the
subprocess boundary are the only things between a proposal and the box.
"""

import ast

ALLOWED_IMPORTS = {"torch", "math", "numpy", "np", "typing", "dataclasses"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "input",
                   "getattr", "setattr", "globals", "locals", "vars"}


def check_ast(code):
    """-> error string, or None if the candidate is allowed to run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "penalty"), None)
    if fn is None:
        return "no top-level `def penalty(ctx):`"
    # A body with no `return <expr>` reaches train.py as `weight * None` and
    # dies at the first step -- a full startup spent on a stub. Seen in the
    # wild: a proposal whose entire body was `...`.
    if not any(isinstance(n, ast.Return) and n.value is not None
               for n in ast.walk(fn)):
        return "`penalty` never returns a value"
    # KNOBS is a declaration TO the harness, which binds each knob as a module
    # global before the file runs. Reading it inside the function rebinds the
    # name to the whole list and silently defeats the sweep -- every variant
    # then computes the same thing.
    if any(isinstance(n, ast.Name) and n.id == "KNOBS" for n in ast.walk(fn)):
        return "`penalty` reads KNOBS itself -- read each knob as a module global"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"Forbidden import: {a.name!r}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top and top not in ALLOWED_IMPORTS:
                return f"Forbidden import: from {node.module!r}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return f"Forbidden call: {node.func.id}()"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return f"Forbidden dunder access: .{node.attr}"
    return None


def extract_knobs(code):
    """A literal top-level `KNOBS = {name: [values]}` turns one proposal into a
    small ablation the GPU sweeps -- costs compute, not tokens. literal_eval,
    so nothing in the candidate runs here."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "KNOBS"
                   for t in node.targets):
            continue
        try:
            val = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError):
            return {}
        if not isinstance(val, dict):
            return {}
        return {k: list(v) for k, v in val.items()
                if isinstance(k, str) and isinstance(v, (list, tuple)) and v}
    return {}


def apply_knobs(code, choice):
    """Bind one point of the KNOBS grid by appending literal assignments.

    The candidate reads its knobs as module globals (`W = 1e-3` at top level),
    so re-assigning them after the def is enough -- Python resolves a global at
    call time, not at def time.
    """
    if not choice:
        return code
    lines = [code, "", "# --- knob binding, appended by the search ---"]
    lines += [f"{k} = {v!r}" for k, v in choice.items()]
    return "\n".join(lines)
