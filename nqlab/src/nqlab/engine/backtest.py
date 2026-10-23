"""One strategy x one parameter set x one fill tier -> trades + daily P&L."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from nqlab.engine.bars import EngineBars, build_engine_bars
from nqlab.engine.simulate import apply_costs, daily_pnl, simulate
from nqlab.hashing import stable_hash
from nqlab.settings import Settings
from nqlab.strategies.registry import get_strategy


@dataclass
class BacktestResult:
    strategy: str
    params: dict[str, Any]
    fill_tier: int
    trades: pl.DataFrame
    daily: pl.DataFrame
    excluded_sessions: list
    data_version: str
    run_id: str
    instrument: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return self.trades.height


def engine_bars_for(md, settings: Settings, start_ct: int) -> EngineBars:
    cache = md.meta.setdefault("_engine_bars", {})
    if start_ct not in cache:
        cache[start_ct] = build_engine_bars(md, settings, start_ct=start_ct)
    return cache[start_ct]


def make_run_id(strategy: str, params: dict, data_version: str, fill_tier: int, settings: Settings) -> str:
    return stable_hash({
        "s": strategy, "sv": get_strategy(strategy).version, "p": params, "d": data_version, "f": fill_tier,
        "cost": settings.section("costs"), "slip": settings.section("slippage"),
        "rules": settings.topstep_rules, "v": 1,
    })


def apply_daily_stop(trades: pl.DataFrame, stop_usd: float | None) -> pl.DataFrame:
    """Drop trades that start after the day's realized loss reached ``stop_usd``."""
    if not stop_usd or trades.height == 0:
        return trades
    t = trades.sort("entry_ts").with_columns(
        pl.col("pnl_usd").cum_sum().shift(1, fill_value=0.0).over("session_date").alias("_before"))
    return t.filter(pl.col("_before") > -float(stop_usd)).drop("_before")


def run_backtest(md, strategy_name: str, params: dict[str, Any], settings: Settings,
                 fill_tier: int = 2) -> BacktestResult:
    strat = get_strategy(strategy_name)
    strat.validate(params)
    eb = engine_bars_for(md, settings, strat.start_ct)
    sig = strat.signals(md, eb, params)
    tick = settings.contract(strat.sizing(params).instrument).tick_size
    raw = simulate(eb, sig, strat.engine_params(params, fill_tier), tick=tick)
    sizing = strat.sizing(params)
    sized = apply_costs(raw, sizing, settings)
    out = sized.filter(pl.col("contracts") == 0)
    pv = settings.contract(sizing.instrument).point_value
    sized_out = {"n_sized_out": out.height,
                 "sized_out_pnl_1c": float((out["pnl_points"] * pv).sum() - out.height * settings.round_turn_cost(sizing.instrument))
                 if out.height else 0.0}
    trades = apply_daily_stop(sized.filter(pl.col("contracts") > 0), params.get("sizing.daily_stop_usd"))
    daily = daily_pnl(trades, eb.session_dates, eb.excluded_sessions)
    return BacktestResult(
        strategy=strategy_name, params=params, fill_tier=fill_tier, trades=trades, daily=daily,
        excluded_sessions=eb.excluded_sessions, data_version=md.data_version,
        run_id=make_run_id(strategy_name, params, md.data_version, fill_tier, settings),
        instrument=sizing.instrument,
        meta=sized_out,
    )
