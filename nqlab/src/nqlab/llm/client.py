"""Ollama client. Models are detected at startup via GET /api/tags (never hardcoded).
If Ollama is unreachable every LLM feature is skipped with a warning; nothing else in
nqlab depends on it. Every call is logged (prompt, response, model) to research_log/llm/."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import httpx

from nqlab.log import get_logger

log = get_logger("llm")


class LLMUnavailable(RuntimeError):
    pass


def _size_b(m: dict[str, Any]) -> float | None:
    ps = (m.get("details") or {}).get("parameter_size") or ""
    mm = re.search(r"([\d.]+)\s*[bB]", ps) or re.search(r"[:\-_](\d+(?:\.\d+)?)b", m.get("name", ""), re.I)
    return float(mm.group(1)) if mm else None


def choose_model(models: list[dict[str, Any]], lo: float = 7, hi: float = 14) -> str | None:
    """Prefer an instruct/chat model in the 7B-14B range; then anything in range; then the
    smallest model above the range; then whatever exists. Embedding models are skipped."""
    cands = [m for m in models if "embed" not in m.get("name", "").lower()]
    if not cands:
        return None

    def score(m):
        n = m.get("name", "").lower()
        s = _size_b(m)
        in_range = s is not None and lo <= s <= hi
        instruct = any(k in n for k in ("instruct", "chat", "-it", ":it", "qwen", "llama3", "mistral", "gemma"))
        return (not in_range, not instruct, abs((s or 100) - (lo + hi) / 2))

    return sorted(cands, key=score)[0]["name"]


class OllamaClient:
    def __init__(self, settings, base_url: str | None = None, transport: httpx.BaseTransport | None = None):
        cfg = settings.section("llm")
        self.settings = settings
        self.enabled = bool(cfg.get("enabled", True))
        self.base_url = (base_url or cfg.get("base_url", "http://localhost:11434")).rstrip("/")
        self.timeout = float(cfg.get("timeout_s", 120))
        self.lo, self.hi = cfg.get("preferred_size_b", [7, 14])
        self._transport = transport
        self.model: str | None = None
        self.log_dir = settings.path("research_log") / "llm"

    def _http(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self._transport)

    def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            with self._http() as c:
                r = c.get("/api/tags", timeout=5)
                r.raise_for_status()
                self.model = choose_model(r.json().get("models", []), self.lo, self.hi)
            if not self.model:
                log.warning("Ollama is running but has no usable models; LLM features skipped")
                return False
            return True
        except Exception as exc:
            log.warning("Ollama unreachable at %s (%s); LLM features skipped, everything else runs", self.base_url, exc)
            return False

    def generate(self, role: str, prompt: str, system: str = "", json_mode: bool = False) -> str:
        if self.model is None and not self.available():
            raise LLMUnavailable("Ollama is not available")
        body: dict[str, Any] = {"model": self.model, "prompt": prompt, "system": system, "stream": False,
                                "options": {"temperature": 0.2}}
        if json_mode:
            body["format"] = "json"
        with self._http() as c:
            r = c.post("/api/generate", json=body)
            r.raise_for_status()
            text = r.json().get("response", "")
        self._log(role, system, prompt, text)
        return text

    def _log(self, role: str, system: str, prompt: str, response: str) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now(dt.timezone.utc)
        p = self.log_dir / f"{ts.strftime('%Y%m%dT%H%M%S%f')}_{role}.json"
        p.write_text(json.dumps({"ts_utc": ts.isoformat(), "role": role, "model": self.model, "system": system,
                                 "prompt": prompt, "response": response}, indent=2), encoding="utf-8")
        return p
