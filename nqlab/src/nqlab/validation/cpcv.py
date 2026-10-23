"""Combinatorial Purged Cross-Validation (Lopez de Prado). N contiguous groups of
sessions; every combination of k groups is a test set, the rest (minus an embargo
around each test group) trains the plateau selection rule. Yields a DISTRIBUTION of
out-of-sample paths, not one."""

from __future__ import annotations

import itertools
from math import comb
from typing import Any

import numpy as np

from nqlab.selection.plateau import Grid, config_sharpes, select


def cpcv(pnl: np.ndarray, grid: Grid, n_groups: int = 10, k_test: int = 2, embargo: int = 1) -> dict[str, Any]:
    T = pnl.shape[0]
    if T < n_groups * 5:
        return {"paths": [], "path_sharpes": np.array([]), "note": "too few sessions for CPCV"}
    bounds = np.linspace(0, T, n_groups + 1).astype(int)
    groups = [np.arange(bounds[g], bounds[g + 1]) for g in range(n_groups)]
    combos = list(itertools.combinations(range(n_groups), k_test))
    # per group, the OOS series produced by each combo that tests it (in combo order)
    per_group: dict[int, list[np.ndarray]] = {g: [] for g in range(n_groups)}
    split_sharpes = []
    for combo in combos:
        test_idx = np.concatenate([groups[g] for g in combo])
        banned = np.zeros(T, bool)
        for g in combo:
            lo = max(0, bounds[g] - embargo)
            hi = min(T, bounds[g + 1] + embargo)
            banned[lo:hi] = True          # purge + embargo on both sides
        train_idx = np.nonzero(~banned)[0]
        sel, _ = select(pnl[train_idx], grid)
        for g in combo:
            per_group[g].append(pnl[groups[g], sel])
        split_sharpes.append(float(config_sharpes(pnl[test_idx, sel][:, None])[0]))
    n_paths = comb(n_groups - 1, k_test - 1)
    paths = [np.concatenate([per_group[g][p] for g in range(n_groups)]) for p in range(n_paths)]
    ps = np.array([config_sharpes(p[:, None])[0] for p in paths])
    return {"paths": paths, "path_sharpes": ps, "split_sharpes": np.array(split_sharpes),
            "n_paths": n_paths, "n_splits": len(combos),
            "frac_paths_positive": float((ps > 0).mean()) if len(ps) else float("nan")}
