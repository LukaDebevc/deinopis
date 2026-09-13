"""LLM client for the optimizer search — chain-prompting ported from LAB-1.

EXPLORE and SYNTHESIS are TWO calls: a PLAN call at high temperature (for
novelty) followed by a CODE call at low temperature (for faithful, correct
implementation). REFINE and FIX are single diagnosis-structured calls.

Defaults to Google AI Studio (OpenAI-compatible) with gemma-4-31b-it, keyed by
GOOGLE_API_KEY; falls back to OpenRouter if only OPENROUTER_API_KEY is set.
"""

import os
import re
import textwrap
import threading
import time
from typing import Optional

try:
    import requests
except ImportError:                       # pragma: no cover
    requests = None

from . import prompts

GOOGLE_URL     = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PLAN_TEMPERATURE   = 2    # creative: novel mechanisms
CODE_TEMPERATURE   = 1    # faithful: correct implementation of the plan
REFINE_TEMPERATURE = 2
FIX_TEMPERATURE    = 1
MAX_RETRIES        = 6    # passes over the whole model chain before giving up
RETRY_BASE_WAIT    = 2.0


class _RateLimited(Exception):
    """A model is rate-limited (429 / RESOURCE_EXHAUSTED / 503) — try the next."""

_PLAN_RE = re.compile(r"PLAN:\s*(.*?)(?=```|$)", re.DOTALL | re.IGNORECASE)
# Match the fence + optional language tag up to the newline, then capture the
# body. We deliberately stop at the newline (not \s*) so the first code line
# keeps its indentation; textwrap.dedent then strips the *common* leading
# whitespace uniformly. Gemma often nests the block under a markdown bullet,
# which left every line but the first indented → "unexpected indent (line 2)".
_CODE_RE = re.compile(r"```[ \t]*\w*[ \t]*\r?\n(.*?)```", re.DOTALL)


def _extract_plan(text: str) -> str:
    m = _PLAN_RE.search(text)
    return (m.group(1).strip() if m else text.strip())[:1500]


def _extract_code(text: str) -> Optional[str]:
    blocks = [textwrap.dedent(b).strip() for b in _CODE_RE.findall(text)]
    for b in blocks:
        if "def penalty" in b:
            return b
    return blocks[-1] if blocks else None


class LLMClient:
    """Thread-safe client with an ordered model-fallback chain.

    Each request walks an ordered list of endpoints, SKIPPING any model that is
    currently in rate-limit cooldown, and returns the first success. On a
    rate-limit the offending model is put on a short cooldown so concurrent
    callers immediately route around it (keeping the candidate pipeline — and
    thus the GPU — fed) instead of all hammering the same throttled model.
    """

    def __init__(self, google_api_key="", openrouter_api_key="",
                 google_models=("gemini-flash-latest", "gemma-4-31b-it", "gemma-4-26b-a4b-it"),
                 openrouter_model="meta-llama/llama-3.3-70b-instruct:free",
                 cooldown_s=25.0):
        gkey = google_api_key or os.environ.get("GOOGLE_API_KEY", "")
        okey = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        # endpoints = ordered [(model, url, key)]; tried in order, cooldown-aware.
        self._endpoints: list[tuple[str, str, str]] = []
        if gkey:
            for m in (google_models or ()):
                self._endpoints.append((m, GOOGLE_URL, gkey))
        if okey:
            self._endpoints.append((openrouter_model, OPENROUTER_URL, okey))
        if not self._endpoints:
            raise RuntimeError("No API key: set GOOGLE_API_KEY (preferred) or OPENROUTER_API_KEY")
        self._cooldown_s = cooldown_s
        self._cooldown_until: dict[str, float] = {}   # model -> ts to skip until
        self._lock = threading.Lock()
        self.last_raw = ""
        print("[llm] model chain: " + " → ".join(m for m, _, _ in self._endpoints))

    def _in_cooldown(self, model: str) -> bool:
        with self._lock:
            return time.time() < self._cooldown_until.get(model, 0.0)

    def _set_cooldown(self, model: str):
        with self._lock:
            self._cooldown_until[model] = time.time() + self._cooldown_s

    def _post(self, model, url, key, user_prompt, temperature) -> str:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"model": model,
                "messages": [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                             {"role": "user", "content": user_prompt}],
                "temperature": temperature}
        r = requests.post(url, json=body, headers=headers, timeout=240)
        if r.status_code in (429, 503) or "RESOURCE_EXHAUSTED" in (r.text or ""):
            raise _RateLimited(f"{model}: {r.status_code}")
        if not r.ok:
            raise RuntimeError(f"{model} {r.status_code} {r.reason}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]

    def _call(self, user_prompt, temperature) -> str:
        if requests is None:
            raise RuntimeError("requests library not installed")
        wait = RETRY_BASE_WAIT
        last_err: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            tried_any = False
            for model, url, key in self._endpoints:
                if self._in_cooldown(model):
                    continue
                tried_any = True
                try:
                    return self._post(model, url, key, user_prompt, temperature)
                except _RateLimited as e:
                    self._set_cooldown(model); last_err = e
                except Exception as e:                       # network/5xx/parse
                    last_err = e
            # whole chain was unavailable this pass (all limited or all erroring)
            if attempt < MAX_RETRIES - 1:
                # if nothing was even tried, every model is cooling down → wait it
                # out; otherwise back off before retrying the chain.
                time.sleep(wait if tried_any else min(self._cooldown_s, wait))
                wait *= 2
        raise last_err or RuntimeError("all LLM endpoints failed")

    def _plan_then_code(self, plan_prompt: str) -> tuple[str, str, str]:
        """Two-call: PLAN (high temp) → CODE (low temp). Returns (plan, code, raw);
        `raw` is returned (not just stashed) so concurrent calls don't race."""
        plan_resp = self._call(plan_prompt, PLAN_TEMPERATURE)
        plan = _extract_plan(plan_resp)
        code_resp = self._call(prompts.code_prompt(plan), CODE_TEMPERATURE)
        raw = f"=== PLAN CALL ===\n{plan_resp}\n\n=== CODE CALL ===\n{code_resp}"
        self.last_raw = raw
        code = _extract_code(code_resp if "```" in code_resp else code_resp + "\n```")
        if code is None:
            raise ValueError(f"could not parse code from:\n{code_resp[:400]}")
        return plan, code, raw

    def explore(self, report, ledger, nudge="") -> tuple[str, str, str]:
        return self._plan_then_code(
            prompts.explore_plan_prompt(report, ledger, nudge))

    def synthesize(self, branch_a, branch_b) -> tuple[str, str, str]:
        return self._plan_then_code(prompts.synthesis_plan_prompt(branch_a, branch_b))

    def _single(self, user_prompt, temperature) -> tuple[str, str, str]:
        resp = self._call(user_prompt, temperature)
        self.last_raw = resp
        plan = _extract_plan(resp)
        code = _extract_code(resp if "```" in resp else resp + "\n```")
        if code is None:
            raise ValueError(f"could not parse code from:\n{resp[:400]}")
        return plan, code, resp

    def refine(self, name, plan, code, detail) -> tuple[str, str, str]:
        return self._single(prompts.refine_prompt(name, plan, code, detail), REFINE_TEMPERATURE)

    def fix(self, plan, code, error) -> tuple[str, str, str]:
        return self._single(prompts.fix_prompt(plan, code, error), FIX_TEMPERATURE)
