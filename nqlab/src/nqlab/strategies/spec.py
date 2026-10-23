"""Strategy spec files (specs/*.yaml): load, flatten, find undecided fields, grid dims."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml

from nqlab.hashing import stable_hash

UNDECIDED = "REQUIRED_UNDECIDED"
META_KEYS = ("spec_version", "strategy", "variant")


class SpecIncomplete(ValueError):  # also raised for runs; see cli._guard
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(
            "spec has undecided fields; nqlab refuses to run it until they are filled:\n  - "
            + "\n  - ".join(missing)
        )


def load_spec(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def undecided(spec: dict[str, Any]) -> list[str]:
    return sorted(k for k, v in flatten(spec).items() if v == UNDECIDED
                  or (isinstance(v, list) and UNDECIDED in v))


def spec_hash(spec: dict[str, Any]) -> str:
    return stable_hash(spec)


# leaves whose VALUE is naturally a list (not a grid dimension)
LIST_VALUED = {"volume_profile.long_levels", "volume_profile.short_levels", "trigger.timeframes"}


def grid_dimensions(spec: dict[str, Any]) -> dict[str, list[Any]]:
    """Leaves given as lists are grid dimensions (except list-valued leaves, which are
    dimensions only when given as a list of lists)."""
    dims = {}
    for k, v in flatten(spec).items():
        if k in META_KEYS:
            continue
        if isinstance(v, list):
            if k in LIST_VALUED:
                if v and all(isinstance(x, list) for x in v):
                    dims[k] = v
            else:
                dims[k] = v
    return dims


def expand(spec: dict[str, Any]) -> list[dict[str, Any]]:
    flat = {k: v for k, v in flatten(spec).items()}
    dims = grid_dimensions(spec)
    if not dims:
        return [flat]
    keys = list(dims)
    out = []
    for combo in itertools.product(*(dims[k] for k in keys)):
        p = dict(flat)
        p.update(dict(zip(keys, combo)))
        out.append(p)
    return out
