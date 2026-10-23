"""Fill-optimism diagnostic: run the strategy on a pure random walk (matched intraday
volatility) under each fill tier. A mean-reversion limit strategy with touch fills is
often profitable on random data; if it is, the profit comes from the fill model."""

from __future__ import annotations

from typing import Any

import numpy as np

from nqlab.engine.backtest import run_backtest
from nqlab.results.metrics import summarize
from nqlab.validation.synthetic import random_walk_matched

FILL_WARNING = "your fill model is generating this profit, not the market."


def fill_optimism(md, strategy: str, params: dict, settings, n_seeds: int = 3, seed: int = 99) -> dict[str, Any]:
    out: dict[str, Any] = {"tiers": {}}
    for tier in (1, 2):
        nets, trades, exp_r = [], 0, []
        for k in range(n_seeds):
            rw = random_walk_matched(md, seed + k)
            r = run_backtest(rw, strategy, params, settings, fill_tier=tier)
            m = summarize(r.trades, r.daily)
            nets.append(m["net_usd"])
            trades += m["n_trades"]
            exp_r.append(m["expectancy_usd"])
        out["tiers"][tier] = {"net_usd": float(np.sum(nets)), "n_trades": trades,
                              "expectancy_usd": float(np.nanmean(exp_r)) if trades else float("nan")}
    t1, t2 = out["tiers"][1], out["tiers"][2]
    out["tier1_profitable_on_noise"] = t1["n_trades"] > 0 and t1["net_usd"] > 0
    out["tier2_profitable_on_noise"] = t2["n_trades"] > 0 and t2["net_usd"] > 0
    out["warning"] = FILL_WARNING if (out["tier1_profitable_on_noise"] or out["tier2_profitable_on_noise"]) else ""
    return out
