"""Synthetic NQ-like 1-minute bars for tests, the fill-optimism diagnostic and the null
calibration harness. Driftless random walk with a U-shaped intraday volatility curve.
Prices sit on the 0.25 tick grid."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from nqlab.calendar.roll import cme_convention_front, contract_symbol, parse_contract
from nqlab.calendar.sessions import build_sessions
from nqlab.settings import Settings

TICK = 0.25


def u_shape(ct_minute: np.ndarray) -> np.ndarray:
    """Relative 1m volatility by CT minute: high at the 08:30 open, dip midday, rise into
    the 15:00 close, quiet overnight."""
    m = ct_minute.astype(float)
    rth = (m >= 510) & (m < 900)
    x = np.clip((m - 510) / 390.0, 0, 1)
    rth_curve = 0.7 + 2.2 * np.exp(-x * 12) + 0.6 * np.exp((x - 1) * 10)
    return np.where(rth, rth_curve, 0.35)


def session_minutes(sess_row: dict) -> np.ndarray:
    start = sess_row["globex_open_utc"]
    end = sess_row["close_utc"]
    n = int((end - start).total_seconds() // 60)
    return np.array([start + dt.timedelta(minutes=i) for i in range(n)], dtype="datetime64[us]")


def random_walk_bars(
    settings: Settings,
    start: dt.date,
    end: dt.date,
    seed: int = 0,
    vol_points: float = 2.0,
    start_price: float = 15000.0,
    drift_points_per_bar: float = 0.0,
    rth_only: bool = False,
    per_contract: bool = False,
    symbol: str = "NQ_CONT",
    sessions: pl.DataFrame | None = None,
    momentum_edge: float = 0.0,
) -> pl.DataFrame:
    """Canonical-schema bars. ``per_contract=True`` emits real contract symbols, with the
    next contract also trading (at a carry premium and rising volume) in the 10 sessions
    before each CME-convention roll, so volume-based roll detection can be tested."""
    rng = np.random.default_rng(seed)
    sess = sessions if sessions is not None else build_sessions(start, end, settings)
    frames = []
    price = start_price
    front = dict(cme_convention_front(sess["session_date"].to_list(), "NQ").iter_rows()) if per_contract else {}
    for row in sess.iter_rows(named=True):
        lo = row["rth_open_utc"] - dt.timedelta(minutes=30) if rth_only else row["globex_open_utc"]
        hi = row["rth_close_utc"] if rth_only else row["close_utc"]
        n = int((hi - lo).total_seconds() // 60)
        if n <= 0:
            continue
        ts = np.array([np.datetime64(lo.replace(tzinfo=None), "us") + np.timedelta64(i, "m") for i in range(n)])
        ct_min = _ct_minutes(ts)
        sig = vol_points * u_shape(ct_min)
        rets = rng.standard_normal(n) * sig + drift_points_per_bar
        if momentum_edge:
            # planted structure for power tests: the 09:00-10:00 CT drift follows the sign
            # of the 08:30-09:00 CT return
            first = (ct_min >= 510) & (ct_min < 540)
            nxt = (ct_min >= 540) & (ct_min < 600)
            if first.any():
                rets[nxt] += momentum_edge * np.sign(rets[first].sum())
        closes = price + np.cumsum(rets)
        opens = np.concatenate([[price], closes[:-1]])
        wig_hi = np.abs(rng.standard_normal(n)) * sig * 0.6
        wig_lo = np.abs(rng.standard_normal(n)) * sig * 0.6
        highs = np.maximum(opens, closes) + wig_hi
        lows = np.minimum(opens, closes) - wig_lo
        o, h, l, c = (np.round(x / TICK) * TICK for x in (opens, highs, lows, closes))
        h = np.maximum.reduce([h, o, c])
        l = np.minimum.reduce([l, o, c])
        vol = np.maximum(1, (rng.poisson(200 * u_shape(ct_min)))).astype(np.int64)
        price = float(c[-1])
        sym = front.get(row["session_date"], symbol) if per_contract else symbol
        frames.append(pl.DataFrame({"ts_utc": ts, "symbol": [sym] * n, "open": o, "high": h, "low": l,
                                    "close": c, "volume": vol}))
        if per_contract:
            _, y, mth = parse_contract(sym)
            nxt_m = mth + 3
            nxt = contract_symbol("NQ", y + (nxt_m > 12), (nxt_m - 1) % 12 + 1)
            # does nxt become front within the next 10 sessions?
            later = [d for d, f in front.items() if f == nxt and 0 < (d - row["session_date"]).days <= 14]
            if later:
                days_left = (min(later) - row["session_date"]).days
                carry = 60.0
                share = max(0.05, 1.0 - days_left / 14)
                frames.append(pl.DataFrame({
                    "ts_utc": ts, "symbol": [nxt] * n, "open": o + carry, "high": h + carry,
                    "low": l + carry, "close": c + carry,
                    "volume": np.maximum(1, (vol * share * 2).astype(np.int64)),
                }))
    df = pl.concat(frames).with_columns(
        pl.col("ts_utc").dt.replace_time_zone("UTC"),
        pl.lit("synthetic").alias("source"),
        pl.lit("synthetic").alias("provenance"),
        pl.lit("").alias("data_version"),
    )
    return df.sort("ts_utc", "symbol")


def _ct_minutes(ts_utc_naive: np.ndarray) -> np.ndarray:
    s = pl.Series(ts_utc_naive).dt.replace_time_zone("UTC").dt.convert_time_zone("America/Chicago")
    return (s.dt.hour().cast(pl.Int32) * 60 + s.dt.minute().cast(pl.Int32)).to_numpy()
