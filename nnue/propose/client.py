"""Talk to the Gemini API over plain HTTP, and respect the free-tier quotas.

The free tier is not one budget, it is three per model -- requests per minute,
tokens per minute, requests per day -- and they differ by an order of magnitude
between models. The good text models allow tens of requests a day; the Gemma
models allow fourteen thousand but only 16k tokens a minute, which caps the
prompt, not the count. So the loop rotates: a few expensive proposals from a
strong model, bulk variation from a cheap one.

Quotas here are DECLARED, from the console, not discovered. `python3 run.py
models` asks the API which model ids the key can actually see; if an id below is
wrong the loop says so on the first call rather than silently falling back.
"""
import json
import os
import time
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta"

# name -> requests/min, tokens/min, requests/day. Edit to match your console.
MODELS = {
    "gemini-3.5-flash":      dict(rpm=5,  tpm=250_000, rpd=20,     json=True),
    "gemini-3.1-flash-lite": dict(rpm=15, tpm=250_000, rpd=500,    json=True),
    "gemma-4-31b-it":        dict(rpm=30, tpm=16_000,  rpd=14_400, json=False),
    "gemma-4-26b-it":        dict(rpm=30, tpm=16_000,  rpd=14_400, json=False),
}


class Quota(Exception):
    pass


class Client:
    def __init__(self, key=None, state_path=None):
        self.key = key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            kf = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".key")
            if os.path.exists(kf):
                self.key = open(kf).read().strip()
        if not self.key:
            raise SystemExit(
                "no API key. Either\n"
                "  echo 'AIza...' > nnue/propose/.key   (gitignored)\n"
                "or export GEMINI_API_KEY=AIza...\n"
                "Get one at https://aistudio.google.com/apikey")
        self.state_path = state_path
        self.state = {}
        if state_path and os.path.exists(state_path):
            self.state = json.load(open(state_path))

    # ------------------------------------------------------------- bookkeeping
    def _bucket(self, model):
        day = time.strftime("%Y-%m-%d")
        s = self.state.setdefault(model, {"day": day, "count": 0, "last": 0.0})
        if s["day"] != day:
            s.update(day=day, count=0)
        return s

    def _save(self):
        if self.state_path:
            json.dump(self.state, open(self.state_path, "w"), indent=1)

    def remaining(self, model):
        return MODELS[model]["rpd"] - self._bucket(model)["count"]

    def _wait(self, model):
        s = self._bucket(model)
        lim = MODELS[model]
        if s["count"] >= lim["rpd"]:
            raise Quota(f"{model}: {lim['rpd']} requests/day used up")
        gap = 60.0 / lim["rpm"]
        sleep = gap - (time.time() - s["last"])
        if sleep > 0:
            time.sleep(sleep)

    # ------------------------------------------------------------------ calls
    def _post(self, path, body, tries=4):
        req = urllib.request.Request(
            f"{BASE}/{path}?key={self.key}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        for i in range(tries):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:300]
                if e.code == 429 and i < tries - 1:
                    time.sleep(20 * (i + 1))
                    continue
                if e.code == 404:
                    raise SystemExit(
                        f"model id rejected by the API: {path}\n{msg}\n"
                        "run `python3 run.py models` and fix MODELS in client.py")
                raise Quota(f"HTTP {e.code}: {msg}")
            except (urllib.error.URLError, TimeoutError) as e:
                if i == tries - 1:
                    raise
                time.sleep(5 * (i + 1))

    def list_models(self):
        with urllib.request.urlopen(f"{BASE}/models?key={self.key}&pageSize=200", timeout=60) as r:
            data = json.load(r)
        return [m["name"].split("/", 1)[1] for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])]

    def generate(self, model, prompt, temperature=1.0, max_tokens=4096):
        """One call. Returns the raw text; parsing is the caller's problem."""
        self._wait(model)
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature,
                                     "maxOutputTokens": max_tokens}}
        if MODELS[model]["json"]:
            body["generationConfig"]["responseMimeType"] = "application/json"
        out = self._post(f"models/{model}:generateContent", body)
        s = self._bucket(model)
        s["count"] += 1
        s["last"] = time.time()
        self._save()
        try:
            parts = out["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            fin = (out.get("candidates") or [{}])[0].get("finishReason", "?")
            raise Quota(f"empty response (finishReason={fin})")
