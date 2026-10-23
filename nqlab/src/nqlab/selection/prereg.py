"""Stage 0: pre-registration. A strategy FAMILY is registered with its full parameter
grid and pass criteria declared up front, hashed, and locked. Any later edit is
detected and refused: a tweak is a new family (which joins the trial count)."""

from __future__ import annotations

import datetime as dt
import os
import stat
from pathlib import Path
from typing import Any

import yaml

from nqlab.hashing import stable_hash
from nqlab.strategies.registry import get_strategy

LOCK_KEYS = ("family_id", "strategy", "hypothesis", "param_space", "fixed", "categorical", "fill_tier",
             "criteria", "origin", "spec_hash", "parent_family", "llm_variant_id")


from nqlab.errors import Refusal


class PreregError(Refusal):
    pass


def prereg_hash(doc: dict[str, Any]) -> str:
    return stable_hash({k: doc.get(k) for k in LOCK_KEYS}, n=32)


def default_criteria(settings) -> dict[str, Any]:
    g = settings.section("validation")["gates"]
    return {k: g[k] for k in ("mcpt_p_max", "dsr_min", "pbo_max", "wfe_min", "min_trades", "min_sessions",
                              "family_t_stat")}


def make_prereg(settings, family_id: str, strategy: str, hypothesis: str, param_space: dict | None = None,
                fixed: dict | None = None, categorical: list | None = None, origin: str = "human",
                spec_hash: str | None = None, parent_family: str | None = None,
                llm_variant_id: str | None = None) -> dict[str, Any]:
    strat = get_strategy(strategy)
    space = param_space if param_space is not None else strat.param_space
    # dimensions of the strategy's declared grid that the caller did not search are pinned
    # to the strategy default and written into the pre-registration (never implicit)
    fx = {k: v for k, v in strat.defaults().items() if k not in space}
    fx.update(fixed or {})
    if not hypothesis or len(hypothesis.strip()) < 10:
        raise PreregError("state the hypothesis in a sentence before testing it")
    doc = {
        "family_id": family_id, "strategy": strategy, "hypothesis": hypothesis.strip(),
        "param_space": {k: list(v) for k, v in space.items()}, "fixed": fx,
        "categorical": sorted(categorical if categorical is not None else strat.categorical),
        "fill_tier": int(settings.section("fill").get("default_tier", 2)),
        "criteria": default_criteria(settings), "origin": origin, "spec_hash": spec_hash,
        "parent_family": parent_family, "llm_variant_id": llm_variant_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_configs": int(len(strat.grid(space))) if space else 1,
    }
    if doc["fill_tier"] != 2:
        raise PreregError("the headline fill tier must be 2 (trade-through)")
    doc["hash"] = prereg_hash(doc)
    return doc


def prereg_path(settings, family_id: str) -> Path:
    return settings.path("research_log") / f"{family_id}.yaml"


def write_prereg(settings, doc: dict[str, Any]) -> Path:
    p = prereg_path(settings, doc["family_id"])
    if p.exists():
        raise PreregError(f"{p.name} already exists: families are never re-registered. Use a new family id.")
    p.parent.mkdir(parents=True, exist_ok=True)
    header = ("# PRE-REGISTRATION (locked). Editing this file invalidates the family.\n"
              "# A change of plan is a NEW family with a new id; it joins the trial count.\n")
    p.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)   # read-only (works on Windows too)
    return p


def load_prereg(settings, family_id: str, expected_hash: str | None = None) -> dict[str, Any]:
    p = prereg_path(settings, family_id)
    if not p.exists():
        raise PreregError(f"family {family_id!r} is not pre-registered (nqlab preregister ...)")
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    h = prereg_hash(doc)
    if h != doc.get("hash") or (expected_hash and h != expected_hash):
        raise PreregError(f"pre-registration for {family_id!r} was modified after it was locked. "
                          "That family is void; register a new one.")
    return doc
