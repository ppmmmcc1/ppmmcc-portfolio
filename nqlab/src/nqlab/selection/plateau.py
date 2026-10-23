"""Plateau selection. Configs never compete individually: a config's score is the MEDIAN
Sharpe of its parameter neighborhood (itself plus every config at most one grid step
away on each ordered dimension; categorical dimensions must match). A lone spike
surrounded by losers scores like its losers.

The expected maximum Sharpe of N worthless configs grows like sqrt(2 ln N); selecting the
peak would systematically pick noise."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Grid:
    keys: list[str]
    values: dict[str, list[Any]]
    categorical: set[str]
    configs: list[dict[str, Any]]
    coords: np.ndarray            # (n_configs, n_dims) value indices
    neighbors: list[np.ndarray]   # per config: indices of its neighborhood (incl. itself)

    @property
    def n(self) -> int:
        return len(self.configs)

    @classmethod
    def build(cls, space: dict[str, list[Any]], fixed: dict[str, Any] | None = None,
              categorical: tuple[str, ...] | set[str] = ()) -> "Grid":
        keys = list(space)
        cat = set(categorical)
        combos = list(itertools.product(*(range(len(space[k])) for k in keys))) or [()]
        coords = np.array(combos, dtype=np.int64).reshape(len(combos), len(keys))
        configs = [{**(fixed or {}), **{k: space[k][c[j]] for j, k in enumerate(keys)}} for c in combos]
        nbrs = []
        for c in coords:
            if len(keys) == 0:
                nbrs.append(np.array([0]))
                continue
            d = np.abs(coords - c)
            ok = np.ones(len(coords), bool)
            for j, k in enumerate(keys):
                ok &= (d[:, j] == 0) if k in cat else (d[:, j] <= 1)
            nbrs.append(np.nonzero(ok)[0])
        return cls(keys, dict(space), cat, configs, coords, nbrs)


def config_sharpes(matrix: np.ndarray) -> np.ndarray:
    """Per-config daily Sharpe; a config that never trades scores 0 (no edge), not NaN."""
    m = np.asarray(matrix, float)
    if m.shape[0] < 2:
        return np.zeros(m.shape[1])
    mu = m.mean(axis=0)
    sd = m.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(sd > 0, mu / sd, 0.0)
    return s


def plateau_scores(sharpes: np.ndarray, grid: Grid) -> np.ndarray:
    return np.array([float(np.median(sharpes[nb])) for nb in grid.neighbors])


def select(matrix: np.ndarray, grid: Grid) -> tuple[int, float]:
    """Index of the config with the best plateau score (ties -> first) and that score."""
    sc = plateau_scores(config_sharpes(matrix), grid)
    i = int(np.argmax(sc))
    return i, float(sc[i])


def heatmap(sharpes: np.ndarray, grid: Grid, x: str, y: str) -> tuple[list, list, np.ndarray, np.ndarray]:
    """2-D slice for the report: Sharpe and plateau score over (x, y), taking the median
    across all other dimensions."""
    xs, ys = grid.values[x], grid.values[y]
    ix, iy = grid.keys.index(x), grid.keys.index(y)
    pl_ = plateau_scores(sharpes, grid)
    raw = np.full((len(ys), len(xs)), np.nan)
    pla = np.full((len(ys), len(xs)), np.nan)
    for a in range(len(xs)):
        for b in range(len(ys)):
            m = (grid.coords[:, ix] == a) & (grid.coords[:, iy] == b)
            raw[b, a] = np.median(sharpes[m])
            pla[b, a] = np.median(pl_[m])
    return xs, ys, raw, pla
