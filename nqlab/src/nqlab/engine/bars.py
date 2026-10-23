"""Engine bar arrays: the strategy's trading window of every session, with per-bar
slippage and the forced-flatten flag."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nqlab.settings import Settings


@dataclass
class EngineBars:
    idx: np.ndarray          # row index into md.bars
    ts: np.ndarray           # bar open, datetime64[us] UTC
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    volume: np.ndarray
    sess: np.ndarray         # dense session code (int64)
    session_dates: list      # code -> date (index = code)
    ct_minute: np.ndarray
    slip: np.ndarray         # adverse ticks for stop / market fills at this bar
    force_flat: np.ndarray   # first bar at/after the flatten deadline
    entry_ok: np.ndarray     # session not excluded and bar not flagged bad
    excluded_sessions: list  # dates excluded by data quality (reported, never traded)

    @property
    def n(self) -> int:
        return int(self.o.shape[0])


def slippage_ticks(df: pl.DataFrame, cfg: dict) -> pl.Expr:
    """Max over the applicable time-of-day buckets (all CT minutes)."""
    m = pl.col("ct_minute")
    rth = pl.col("_rth")
    base = pl.when(rth).then(cfg["default_rth_ticks"]).otherwise(cfg["overnight_ticks"])
    first_min = (m == 510)
    rel_0830 = pl.col("is_release_0830") & ((m == 450) | (m == 451))   # 07:30 CT = 08:30 ET
    rel_1000 = pl.col("is_release_1000") & ((m == 540) | (m == 541))   # 09:00 CT = 10:00 ET
    fomc_day = pl.col("is_fomc") & rth
    fomc_ann = pl.col("is_fomc") & (m >= 780) & (m < 785)              # 13:00-13:05 CT
    return pl.max_horizontal(
        base,
        pl.when(first_min).then(cfg["open_first_minute_ticks"]).otherwise(0),
        pl.when(rel_0830 | rel_1000).then(cfg["release_ticks"]).otherwise(0),
        pl.when(fomc_day).then(cfg["fomc_day_ticks"]).otherwise(0),
        pl.when(fomc_ann).then(cfg["fomc_announcement_ticks"]).otherwise(0),
    ).cast(pl.Int64)


def build_engine_bars(md, settings: Settings, start_ct: int = 8 * 60, flatten: bool = True) -> EngineBars:
    """Bars from ``start_ct`` (CT minutes; may be before the RTH open) through the first
    bar at/after the flatten deadline. ``flatten=False`` runs to the session close (the
    position is then closed at the last bar before the maintenance break)."""
    sess_cols = ["session_date", "rth_open_utc", "rth_close_utc", "flatten_utc", "close_utc",
                 "is_fomc", "is_release_0830", "is_release_1000", "excluded"]
    b = md.bars.select("ts_utc", "session_date", "open", "high", "low", "close", "volume", "ct_minute", "bad_bar") \
        .with_row_index("_idx").join(md.sessions.select(sess_cols), on="session_date", how="inner")
    b = b.with_columns(
        ((pl.col("ts_utc") >= pl.col("rth_open_utc")) & (pl.col("ts_utc") < pl.col("rth_close_utc"))).alias("_rth"),
    )
    # window start in CT minutes relative to the session's own trade date (overnight
    # minutes >= 17:00 belong to the previous calendar day and are excluded)
    in_window = (pl.col("ct_minute") >= start_ct) & (pl.col("ct_minute") < 17 * 60)
    end_col = "flatten_utc" if flatten else "close_utc"
    b = b.filter(in_window & (pl.col("ts_utc") <= pl.col(end_col)))
    b = b.sort("ts_utc")
    # first bar at/after deadline (may be later than the deadline if a bar is missing)
    after = pl.col("ts_utc") >= pl.col("flatten_utc")
    b = b.with_columns(after.alias("_after"))
    b = b.with_columns((pl.col("_after") & (pl.col("_after").cum_sum().over("session_date") == 1)).alias("_ff"))
    if flatten:
        b = b.filter(~pl.col("_after") | pl.col("_ff"))
    else:
        b = b.with_columns(pl.lit(False).alias("_ff"))
    b = b.with_columns(slippage_ticks(b, settings.section("slippage")).alias("_slip"))
    codes = b["session_date"].rank("dense").cast(pl.Int64) - 1
    dates = b["session_date"].unique().sort().to_list()
    bad_sess = set(b.filter(pl.col("bad_bar"))["session_date"].unique().to_list())
    excluded = sorted(set(b.filter(pl.col("excluded"))["session_date"].unique().to_list()) | bad_sess)
    entry_ok = (~b["excluded"] & ~b["session_date"].is_in(list(bad_sess))).to_numpy()
    return EngineBars(
        idx=b["_idx"].to_numpy().astype(np.int64),
        ts=b["ts_utc"].dt.replace_time_zone(None).to_numpy(),
        o=b["open"].to_numpy(), h=b["high"].to_numpy(), l=b["low"].to_numpy(), c=b["close"].to_numpy(),
        volume=b["volume"].to_numpy(),
        sess=codes.to_numpy(), session_dates=dates, ct_minute=b["ct_minute"].to_numpy(),
        slip=b["_slip"].to_numpy(), force_flat=b["_ff"].to_numpy(), entry_ok=entry_ok,
        excluded_sessions=excluded,
    )
