"""Append-only JSONL run log (thread-safe: producer + consumer both write)."""

import json
import threading
import time


class RunLog:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "a", buffering=1)
        self._lock = threading.Lock()

    def write(self, kind: str, **fields):
        rec = {"t": round(time.time(), 3), "kind": kind, **fields}
        line = json.dumps(rec, default=str) + "\n"
        with self._lock:
            self._f.write(line)

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
