"""Run the kernel and turn its output into trade and daily tables."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from nqlab.engine.bars import EngineBars
from nqlab.engine.kernel import K_LIMIT, K_OPEN, K_STOP, run_kernel
from nqlab.engine.orders import EngineParams, Signals
from nqlab.settings import Settings

REASONS = {1: "stop", 2: "target", 3: "flatten_1510", 4: "exit_signal", 5: "session_end"}
KINDS = {K_OPEN: "at_open", K_LIMIT: "limit_intrabar", K_STOP: "stop_intrabar"}


def simulate(eb: EngineBars, sig: Signals, ep: EngineParams, tick: float = 0.25) -> pl.DataFrame:
    """Trades in POINTS per contract (sizing and costs are applied separately)."""
    if sig.n != eb.n:
        raise ValueError("signals and engine bars differ in length")
    if ep.fill_tier not in (1, 2):
        raise ValueError("the bar kernel implements tiers 1 and 2; tiers 3/4 use engine.tick_replay")
    out = run_kernel(
        eb.o, eb.h, eb.l, eb.c, eb.sess, eb.slip.astype(np.float64), eb.force_flat, eb.entry_ok,
        sig.l_type, sig.l_px, sig.l_lim, sig.l_stop, sig.l_tgt,
        sig.s_type, sig.s_px, sig.s_lim, sig.s_stop, sig.s_tgt,
        sig.cancel_l, sig.cancel_s, sig.exit_now,
        int(ep.fill_tier), float(tick), int(ep.expiry_bars), int(ep.max_trades_per_day),
        bool(ep.stop_after_win), bool(ep.stop_after_loss), float(ep.be_trigger_r), int(ep.be_offset_ticks),
    )
    ei, xi, side, epx, xpx, sd, td, mae, mfe, reason, kind, amb, be = out
    dates = np.array(eb.session_dates, dtype="datetime64[D]")
    df = pl.DataFrame({
        "session_date": dates[eb.sess[ei]] if len(ei) else np.array([], "datetime64[D]"),
        "entry_bar": ei, "exit_bar": xi,
        "entry_ts": eb.ts[ei], "exit_ts": eb.ts[xi],
        "entry_ct_minute": eb.ct_minute[ei],
        "side": side, "entry_price": epx, "exit_price": xpx,
        "stop_dist": sd, "target_dist": td,
        "pnl_points": (xpx - epx) * side,
        "mae_points": mae, "mfe_points": mfe,
        "exit_reason": [REASONS[int(r)] for r in reason],
        "entry_type": [KINDS[int(k)] for k in kind],
        "ambiguous": amb, "breakeven_moved": be,
        "fill_tier": np.full(len(ei), ep.fill_tier),
    }, schema_overrides={"session_date": pl.Date})
    return df.with_columns(pl.col("entry_ts").dt.replace_time_zone("UTC"), pl.col("exit_ts").dt.replace_time_zone("UTC"))


@dataclass(frozen=True)
class Sizing:
    instrument: str = "NQ"
    contracts: int = 1
    risk_usd: float | None = None      # if set: contracts = floor(risk / (stop pts * point value))
    max_contracts: int | None = None


def apply_costs(trades: pl.DataFrame, sizing: Sizing, settings: Settings) -> pl.DataFrame:
    spec = settings.contract(sizing.instrument)
    rt = settings.round_turn_cost(sizing.instrument)
    if sizing.risk_usd is not None:
        risk_per = trades["stop_dist"] * spec.point_value
        n = (sizing.risk_usd / risk_per).floor().fill_nan(0).fill_null(0).cast(pl.Int64)
    else:
        n = pl.Series([sizing.contracts] * trades.height, dtype=pl.Int64)
    if sizing.max_contracts is not None:
        n = n.clip(upper_bound=sizing.max_contracts)
    t = trades.with_columns(n.alias("contracts"))
    t = t.with_columns(
        (pl.col("pnl_points") * spec.point_value * pl.col("contracts")).alias("gross_usd"),
        (pl.lit(rt) * pl.col("contracts")).alias("fees_usd"),
    ).with_columns(
        (pl.col("gross_usd") - pl.col("fees_usd")).alias("pnl_usd"),
        (pl.col("pnl_points") / pl.col("stop_dist")).alias("r_gross"),
        ((pl.col("gross_usd") - pl.col("fees_usd")) / (pl.col("stop_dist") * spec.point_value * pl.col("contracts"))).alias("r_multiple"),
        pl.lit(sizing.instrument).alias("instrument"),
    )
    skipped = int((t["contracts"] == 0).sum())
    if skipped:
        t = t.filter(pl.col("contracts") > 0)
    return t


def daily_pnl(trades: pl.DataFrame, session_dates: list, excluded: list) -> pl.DataFrame:
    """One row per ELIGIBLE session, zero-filled. Excluded sessions are not traded and not
    counted as zero days (they are reported separately)."""
    ex = set(excluded)
    days = pl.DataFrame({"session_date": [d for d in session_dates if d not in ex]}, schema={"session_date": pl.Date})
    if trades.height:
        agg = trades.group_by("session_date").agg(pl.col("pnl_usd").sum(), pl.len().alias("n_trades"))
    else:
        agg = pl.DataFrame(schema={"session_date": pl.Date, "pnl_usd": pl.Float64, "n_trades": pl.UInt32})
    return days.join(agg, on="session_date", how="left").with_columns(
        pl.col("pnl_usd").fill_null(0.0), pl.col("n_trades").fill_null(0).cast(pl.Int64)
    ).sort("session_date")


def contracts_for_risk(risk_usd: float, stop_points: float, point_value: float) -> int:
    if not (stop_points > 0) or math.isnan(stop_points):
        return 0
    return int(math.floor(risk_usd / (stop_points * point_value)))
