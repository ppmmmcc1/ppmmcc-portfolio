"""Settings loading and project-root discovery.

The project root is the folder containing ``config/settings.yaml``. It is taken from
``NQLAB_ROOT`` when set (tests point this at a temp copy), otherwise found by walking
up from the current directory.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class ContractSpec(BaseModel):
    tick_size: float
    tick_value: float
    point_value: float


class Settings(BaseModel):
    model_config = ConfigDict(extra="allow")

    root: Path
    raw: dict[str, Any]

    # ---- convenience accessors -------------------------------------------------
    def path(self, key: str) -> Path:
        return (self.root / self.raw["paths"][key]).resolve()

    def contract(self, symbol: str) -> ContractSpec:
        root = symbol_root(symbol)
        return ContractSpec(**self.raw["contracts"][root])

    def round_turn_cost(self, symbol: str) -> float:
        return float(self.raw["costs"]["round_turn"][symbol_root(symbol)])

    def section(self, key: str) -> dict[str, Any]:
        return self.raw[key]

    @property
    def topstep_rules(self) -> dict[str, Any]:
        path = (self.root / self.raw["topstep"]["rules_file"]).resolve()
        return load_yaml(path)


def symbol_root(symbol: str) -> str:
    """'MNQZ25' -> 'MNQ', 'NQ_CONT' -> 'NQ', 'NQ' -> 'NQ'."""
    s = symbol.upper()
    if s.startswith("MNQ"):
        return "MNQ"
    if s.startswith("NQ"):
        return "NQ"
    raise ValueError(f"unknown symbol root for {symbol!r}")


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def find_root(start: Path | None = None) -> Path:
    env = os.environ.get("NQLAB_ROOT")
    if env:
        p = Path(env).resolve()
        if not (p / "config" / "settings.yaml").is_file():
            raise FileNotFoundError(f"NQLAB_ROOT={p} has no config/settings.yaml")
        return p
    here = (start or Path.cwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "config" / "settings.yaml").is_file():
            return cand
    raise FileNotFoundError(
        "could not find config/settings.yaml; run from the nqlab folder or set NQLAB_ROOT"
    )


@lru_cache(maxsize=8)
def _load(root: Path) -> Settings:
    raw = load_yaml(root / "config" / "settings.yaml")
    return Settings(root=root, raw=raw)


def get_settings(root: Path | None = None) -> Settings:
    return _load(Path(root).resolve() if root else find_root())


def clear_settings_cache() -> None:
    _load.cache_clear()
