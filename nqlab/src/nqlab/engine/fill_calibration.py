"""Fill-tier comparison on tick data (spec 6: "Use [tier 4] on a sample of ~200 sessions
to calibrate tiers 2 and 3").

Every tier is run on the SAME sessions and the SAME 09:25-10:45 ET window (08:25-09:45 CT),
so the comparison is apples to apples; positions still open at the end of the window are
closed at its last bar in every tier."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from nqlab.engine.backtest import engine_bars_for
from nqlab.engine.bars import EngineBars
from nqlab.engine.fills import MBOReplayFill, QueueFill, TouchFill, TradeThroughFill
from nqlab.engine.orders import Signals
from nqlab.strategies.registry import get_strategy

WINDOW_CT = (8 * 60 + 25, 9 * 60 + 45)


def subset(eb: EngineBars, sig: Signals, mask: np.ndarray) -> tuple[EngineBars, Signals]:
    idx = np.nonzero(mask)[0]
    codes = eb.sess[idx]
    uniq = np.unique(codes)
    remap = {int(c): k for k, c in enumerate(uniq)}
    eb2 = EngineBars(idx=eb.idx[idx], ts=eb.ts[idx], o=eb.o[idx], h=eb.h[idx], l=eb.l[idx], c=eb.c[idx],
                     volume=eb.volume[idx], sess=np.array([remap[int(c)] for c in codes], np.int64),
                     session_dates=[eb.session_dates[int(c)] for c in uniq], ct_minute=eb.ct_minute[idx],
                     slip=eb.slip[idx], force_flat=eb.force_flat[idx], entry_ok=eb.entry_ok[idx],
                     excluded_sessions=eb.excluded_sessions)
    s2 = Signals(len(idx))
    for k in ("l_type", "l_px", "l_lim", "l_stop", "l_tgt", "s_type", "s_px", "s_lim", "s_stop", "s_tgt",
              "cancel_l", "cancel_s", "exit_now"):
        setattr(s2, k, getattr(sig, k)[idx])
    return eb2, s2


def compare_tiers(md, strategy: str, params: dict[str, Any], settings, trades: pl.DataFrame,
                  mbo: pl.DataFrame | None = None, n_sessions: int = 200, seed: int = 0) -> dict[str, Any]:
    strat = get_strategy(strategy)
    strat.validate(params)
    eb = engine_bars_for(md, settings, strat.start_ct)
    sig = strat.signals(md, eb, params)
    tick_days = set(trades.select(pl.col("ts_utc").dt.convert_time_zone("America/Chicago").dt.date().unique())
                    .to_series().to_list())
    have = [d for d in eb.session_dates if d in tick_days and d not in set(eb.excluded_sessions)]
    rng = np.random.default_rng(seed)
    sample = set(rng.choice(np.array(have, dtype=object), size=min(n_sessions, len(have)), replace=False)) if have else set()
    sd = np.array(eb.session_dates, dtype=object)[eb.sess]
    mask = np.array([d in sample for d in sd]) & (eb.ct_minute >= WINDOW_CT[0]) & (eb.ct_minute < WINDOW_CT[1])
    if not mask.any():
        return {"n_sessions": 0, "tiers": {}, "note": "no tick data overlaps the strategy's sessions"}
    e2, s2 = subset(eb, sig, mask)
    e2.force_flat[:] = False
    ep = strat.engine_params(params, 2)
    tick = settings.contract(strat.sizing(params).instrument).tick_size
    models = [TouchFill(), TradeThroughFill(), QueueFill(float(settings.section("fill").get("queue_cancel_haircut", 0.3)))]
    if mbo is not None:
        models.append(MBOReplayFill())
    pv = settings.contract(strat.sizing(params).instrument).point_value
    fee = settings.round_turn_cost(strat.sizing(params).instrument)
    out = {}
    for m in models:
        tr = m.run(e2, s2, ep, tick, trades=trades, mbo=mbo)
        pnl = tr["pnl_points"].to_numpy() * pv - fee if tr.height else np.array([])
        out[m.tier] = {"label": m.label, "n_trades": int(tr.height), "net_usd": float(pnl.sum()),
                       "expectancy_usd": float(pnl.mean()) if len(pnl) else float("nan")}
    return {"n_sessions": len(sample), "window_et": "09:25-10:45", "tiers": out}
