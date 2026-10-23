"""Trade reshuffle, trade bootstrap and block bootstraps of daily P&L."""

from __future__ import annotations

from typing import Any

import numpy as np

from nqlab.results.metrics import max_drawdown


def reshuffle_drawdowns(trade_pnl: np.ndarray, n: int = 5000, seed: int = 0) -> dict[str, Any]:
    """Permute the order of realized trades: the drawdown distribution GIVEN this edge.
    NOT a test of edge (the total is identical in every permutation)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(trade_pnl, float)
    if len(x) == 0:
        return {"dd": np.array([]), "p50": float("nan"), "p95": float("nan")}
    dd = np.array([max_drawdown(np.cumsum(rng.permutation(x))) for _ in range(n)])
    return {"dd": dd, "p50": float(np.median(dd)), "p95": float(np.quantile(dd, 0.95)),
            "realized": max_drawdown(np.cumsum(x))}


def trade_bootstrap(trade_pnl: np.ndarray, n: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Resample trades with replacement: CI on expectancy and P(final < 0)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(trade_pnl, float)
    if len(x) < 2:
        return {"exp_ci": (float("nan"), float("nan")), "p_final_negative": float("nan")}
    idx = rng.integers(0, len(x), size=(n, len(x)))
    s = x[idx]
    means = s.mean(axis=1)
    finals = s.sum(axis=1)
    return {"exp_mean": float(x.mean()), "exp_ci": (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))),
            "p_final_negative": float((finals < 0).mean())}


def block_bootstrap_sharpe(daily: np.ndarray, n: int = 2000, block: int = 5, seed: int = 0,
                           kind: str = "stationary") -> dict[str, Any]:
    from arch.bootstrap import CircularBlockBootstrap, StationaryBootstrap

    x = np.asarray(daily, float)
    if len(x) < 20 or x.std() == 0:
        return {"sharpe_ci": (float("nan"), float("nan"))}
    B = StationaryBootstrap if kind == "stationary" else CircularBlockBootstrap
    bs = B(block, x, seed=seed)

    def f(a):
        sd = a.std(ddof=1)
        return np.array([a.mean() / sd if sd > 0 else 0.0])

    res = np.array([f(*pos)[0] for pos, _ in bs.bootstrap(n)]) * np.sqrt(252)
    return {"sharpe_ci": (float(np.quantile(res, 0.025)), float(np.quantile(res, 0.975))), "kind": kind}
