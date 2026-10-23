"""Walk-forward of the SELECTION RULE ("pick the best plateau from the last 120
sessions"), not of a fixed parameter set: this is what the operator will do live."""

from __future__ import annotations

from typing import Any

import numpy as np

from nqlab.selection.plateau import Grid, config_sharpes, select


def walk_forward(pnl: np.ndarray, grid: Grid, train: int = 120, test: int = 20, step: int = 20,
                 anchored: bool = False) -> dict[str, Any]:
    T = pnl.shape[0]
    windows = []
    oos = []
    start = 0
    while True:
        tr0 = 0 if anchored else start
        tr1 = start + train
        te1 = tr1 + test
        if te1 > T:
            break
        idx, score = select(pnl[tr0:tr1], grid)
        is_mean = float(pnl[tr0:tr1, idx].mean())
        seg = pnl[tr1:te1, idx]
        windows.append({"train": (tr0, tr1), "test": (tr1, te1), "selected": idx, "plateau_score": score,
                        "is_mean": is_mean, "oos_mean": float(seg.mean()), "oos_sum": float(seg.sum()),
                        "params": grid.configs[idx]})
        oos.append(seg)
        start += step
    if not windows:
        return {"windows": [], "oos": np.array([]), "wfe": float("nan"), "n_windows": 0,
                "note": f"need at least {train + test} sessions, have {T}"}
    oos_arr = np.concatenate(oos)
    is_mean = np.mean([w["is_mean"] for w in windows])
    oos_mean = float(oos_arr.mean())
    wfe = oos_mean / is_mean if is_mean > 0 else float("nan")
    sel = np.array([w["selected"] for w in windows])
    changes = int((np.diff(sel) != 0).sum())
    # normalized jump size per dimension (0 = identical, 1 = across the whole axis)
    jumps = {}
    for j, k in enumerate(grid.keys):
        if len(grid.values[k]) > 1:
            c = grid.coords[sel, j].astype(float)
            jumps[k] = float(np.abs(np.diff(c)).mean() / (len(grid.values[k]) - 1)) if len(c) > 1 else 0.0
    unstable = changes > 0.5 * max(len(windows) - 1, 1) and any(v > 0.34 for v in jumps.values())
    sr = config_sharpes(oos_arr[:, None])[0]
    return {"windows": windows, "oos": oos_arr, "oos_mean": oos_mean, "is_mean": float(is_mean), "wfe": wfe,
            "n_windows": len(windows), "selection_changes": changes, "param_jumps": jumps,
            "unstable": bool(unstable), "oos_sharpe": float(sr), "anchored": anchored}
