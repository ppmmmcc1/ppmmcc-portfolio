"""Performance metrics. Every Sharpe is printed with its standard error."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

ANN = 252


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def sharpe_se(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of a per-period Sharpe (Mertens 2002 / Lo 2002, non-normal)."""
    if n < 2 or not np.isfinite(sr):
        return float("nan")
    return math.sqrt(max((1 - skew * sr + (kurt - 1) / 4 * sr * sr) / (n - 1), 0.0))


def max_consecutive_losses(pnl: np.ndarray) -> int:
    best = cur = 0
    for v in pnl:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    return float((peak[1:] - equity).max())


def required_n(expectancy_r: float, sd_r: float = 1.0, t: float = 2.0) -> float:
    if not expectancy_r or expectancy_r <= 0:
        return float("inf")
    return (t * sd_r / expectancy_r) ** 2


def summarize(trades: pl.DataFrame, daily: pl.DataFrame) -> dict[str, Any]:
    d = daily["pnl_usd"].to_numpy() if daily.height else np.array([])
    pnl = trades["pnl_usd"].to_numpy() if trades.height else np.array([])
    r = trades["r_multiple"].to_numpy() if trades.height else np.array([])
    r = r[np.isfinite(r)]
    n_days = len(d)
    sr = sharpe(d)
    sk = float(stats.skew(d)) if n_days > 2 else 0.0
    ku = float(stats.kurtosis(d, fisher=False)) if n_days > 3 else 3.0
    se = sharpe_se(sr, n_days, sk, ku)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    exp_r = float(r.mean()) if len(r) else float("nan")
    sd_r = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    t_stat = float(exp_r / (sd_r / math.sqrt(len(r)))) if len(r) > 1 and sd_r > 0 else float("nan")
    eq = np.cumsum(d)
    amb = float(trades["ambiguous"].mean()) if trades.height else 0.0
    return {
        "n_trades": int(trades.height),
        "n_sessions": int(n_days),
        "trades_per_session": float(trades.height / n_days) if n_days else 0.0,
        "net_usd": float(d.sum()),
        "expectancy_usd": float(pnl.mean()) if len(pnl) else float("nan"),
        "expectancy_r": exp_r,
        "sd_r": sd_r,
        "t_stat_r": t_stat,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else float("nan"),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else float("nan"),
        "max_consec_losses": max_consecutive_losses(pnl),
        "max_drawdown_usd": max_drawdown(eq),
        "sharpe_daily": sr,
        "sharpe_daily_se": se,
        "sharpe_ann": sr * math.sqrt(ANN) if np.isfinite(sr) else float("nan"),
        "sharpe_ann_se": se * math.sqrt(ANN) if np.isfinite(se) else float("nan"),
        "skew_daily": sk,
        "kurt_daily": ku,
        "fees_usd": float(trades["fees_usd"].sum()) if trades.height else 0.0,
        "ambiguous_frac": amb,
        "required_n_t2": required_n(exp_r, sd_r if np.isfinite(sd_r) else 1.0, 2.0),
        "required_n_t3": required_n(exp_r, sd_r if np.isfinite(sd_r) else 1.0, 3.0),
    }


SAMPLE_SIZE_TABLE = [(0.30, 45, 100), (0.20, 100, 225), (0.10, 400, 900), (0.05, 1600, 3600)]
