"""Hard guardrails on LLM output (assertions, not comments).

* The LLM never computes a metric, ranks strategies or chooses a parameter value. Any
  number in its output that does not appear in its input is removed.
* LLM-proposed variants may only pick values that already exist in the family's declared
  parameter space; anything else is discarded.
"""

from __future__ import annotations

import re
from typing import Any

# a number may end a sentence ("3.1."), so only a following word char or ".digit" blocks it
NUM = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?(?!\w)(?!\.\d)")
REMOVED = "[number removed: not computed by code]"


def _norm(tok: str) -> str:
    return tok.replace("$", "").replace(",", "").replace("%", "").lstrip("+")


def allowed_numbers(*sources: str) -> set[str]:
    out = set()
    for s in sources:
        for m in NUM.findall(s or ""):
            n = _norm(m)
            out.add(n)
            try:
                f = float(n)
                out.add(f"{f:g}")
                if f == int(f):
                    out.add(str(int(f)))
            except ValueError:
                pass
    return out


def strip_unsourced_numbers(text: str, *sources: str) -> tuple[str, int]:
    """Replace every number in ``text`` that does not occur in ``sources``."""
    ok = allowed_numbers(*sources)
    removed = 0

    def sub(m):
        nonlocal removed
        n = _norm(m.group(0))
        try:
            g = f"{float(n):g}"
        except ValueError:
            g = n
        if n in ok or g in ok:
            return m.group(0)
        removed += 1
        return REMOVED

    out = NUM.sub(sub, text)
    assert all(_norm(t) in ok or f"{float(_norm(t)):g}" in ok for t in NUM.findall(out)), "guardrail failure"
    return out, removed


def filter_variant(variant: dict[str, Any], space: dict[str, list[Any]]) -> dict[str, Any] | None:
    """Keep only keys of the declared space whose values are lists of declared values.
    Returns the restricted sub-space (a variant is a FAMILY, not a point), or None."""
    out: dict[str, list[Any]] = {}
    for k, vals in (variant.get("param_space") or {}).items():
        if k not in space:
            continue
        vals = vals if isinstance(vals, list) else [vals]
        keep = [v for v in vals if v in space[k]]
        if keep:
            out[k] = keep
    for k in space:
        out.setdefault(k, list(space[k]))       # unspecified dims keep the full declared grid
    assert all(set(map(repr, out[k])) <= set(map(repr, space[k])) for k in out), "variant escaped the declared space"
    return out


def assert_no_ranking(text: str) -> None:
    """Morning summaries narrate; they never recommend which strategy to trade."""
    bad = re.search(r"\b(you should trade|go live with|recommend(?:ed)? (?:trading|deploying)|the best strategy is)\b",
                    text, re.I)
    if bad:
        raise AssertionError(f"LLM attempted to select a strategy: {bad.group(0)!r}")


_SELECT = re.compile(r"[^.!?\n]*\b(you should trade|go live with|recommend(?:ed)? (?:trading|deploying)|the best strategy is|"
                     r"increase (?:the )?size|switch to)\b[^.!?\n]*[.!?]?", re.I)


def scrub_selection(text: str) -> tuple[str, int]:
    """Remove sentences in which the LLM tries to pick a strategy or a size."""
    n = len(_SELECT.findall(text))
    return _SELECT.sub("[removed: the LLM does not select strategies]", text), n
