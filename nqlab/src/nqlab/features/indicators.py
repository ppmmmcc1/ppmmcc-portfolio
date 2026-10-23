"""Causal indicators: SMA/EMA, Bollinger, MACD, ATR, session VWAP, opening range."""

from __future__ import annotations

import polars as pl

from nqlab.features.registry import col, feature


def _ema(x: pl.Expr, span: int) -> pl.Expr:
    return x.ewm_mean(span=span, adjust=False, min_samples=span)


@feature("sma", series="param", defaults={"length": 20, "source": "close", "series": "back_adjusted"})
def sma(md, length: int, source: str, series: str) -> pl.DataFrame:
    return md.bars.select(col(source, series).rolling_mean(length, min_samples=length).alias("sma"))


@feature("ema", series="param", defaults={"length": 20, "source": "close", "series": "back_adjusted"})
def ema(md, length: int, source: str, series: str) -> pl.DataFrame:
    return md.bars.select(_ema(col(source, series), length).alias("ema"))


def bollinger_expr(x: pl.Expr, length: int, mult: float, basis: str, stdev: str) -> list[pl.Expr]:
    if stdev not in ("population", "sample"):
        raise ValueError("stdev must be 'population' or 'sample' (they differ; choose explicitly)")
    ddof = 0 if stdev == "population" else 1
    mid = x.rolling_mean(length, min_samples=length) if basis == "sma" else _ema(x, length) if basis == "ema" else None
    if mid is None:
        raise ValueError("basis must be 'sma' or 'ema'")
    sd = x.rolling_std(length, min_samples=length, ddof=ddof)
    return [mid.alias("bb_basis"), (mid + mult * sd).alias("bb_upper"), (mid - mult * sd).alias("bb_lower")]


@feature("bollinger", series="param",
         defaults={"length": 20, "mult": 2.0, "basis": "sma", "stdev": "population",
                   "source": "close", "series": "back_adjusted"})
def bollinger(md, length: int, mult: float, basis: str, stdev: str, source: str, series: str) -> pl.DataFrame:
    return md.bars.select(bollinger_expr(col(source, series), length, mult, basis, stdev))


def macd_expr(x: pl.Expr, fast: int, slow: int, signal: int) -> list[pl.Expr]:
    line = _ema(x, fast) - _ema(x, slow)
    sig = line.ewm_mean(span=signal, adjust=False, min_samples=signal)
    return [line.alias("macd_line"), sig.alias("macd_signal"), (line - sig).alias("macd_hist")]


@feature("macd", series="param",
         defaults={"fast": 12, "slow": 26, "signal": 9, "source": "close", "series": "back_adjusted"})
def macd(md, fast: int, slow: int, signal: int, source: str, series: str) -> pl.DataFrame:
    return md.bars.select(macd_expr(col(source, series), fast, slow, signal))


@feature("atr", series="back_adjusted", defaults={"length": 14})
def atr(md, length: int) -> pl.DataFrame:
    h, l, c = pl.col("high_badj"), pl.col("low_badj"), pl.col("close_badj")
    tr = pl.max_horizontal(h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs())
    return md.bars.select(tr.ewm_mean(alpha=1.0 / length, adjust=False, min_samples=length).alias("atr"))


def _anchor_join(md, anchor: str) -> pl.DataFrame:
    cols = {"rth": "rth_open_utc", "globex": "globex_open_utc"}
    if anchor not in cols:
        raise ValueError("anchor must be 'rth' or 'globex'")
    return md.bars.select("ts_utc", "session_date").join(
        md.sessions.select("session_date", pl.col(cols[anchor]).alias("_anchor")), on="session_date", how="left"
    )


@feature("session_vwap", series="unadjusted", defaults={"anchor": "rth"})
def session_vwap(md, anchor: str) -> pl.DataFrame:
    a = _anchor_join(md, anchor)["_anchor"]
    b = md.bars.with_columns(a.alias("_anchor"))
    active = pl.col("ts_utc") >= pl.col("_anchor")
    tp = col("hlc3", "unadjusted")
    pv = pl.when(active).then(tp * pl.col("volume")).otherwise(0.0)
    vv = pl.when(active).then(pl.col("volume").cast(pl.Float64)).otherwise(0.0)
    out = b.select(
        pl.when(active)
        .then(pv.cum_sum().over("session_date") / vv.cum_sum().over("session_date"))
        .otherwise(None).alias("vwap")
    )
    return out


@feature("opening_range", series="unadjusted", defaults={"minutes": 15})
def opening_range(md, minutes: int) -> pl.DataFrame:
    """High/low of the first ``minutes`` of RTH. Null until the window has fully closed
    (a value visible at 09:32 ET would be the future 15-minute high)."""
    a = _anchor_join(md, "rth")["_anchor"]
    b = md.bars.with_columns(a.alias("_anchor"))
    in_win = (pl.col("ts_utc") >= pl.col("_anchor")) & (pl.col("ts_utc") < pl.col("_anchor") + pl.duration(minutes=minutes))
    hi = pl.when(in_win).then(pl.col("high")).otherwise(None).cum_max().forward_fill().over("session_date")
    lo = pl.when(in_win).then(pl.col("low")).otherwise(None).cum_min().forward_fill().over("session_date")
    done = pl.col("ts_close_utc") >= pl.col("_anchor") + pl.duration(minutes=minutes)
    return b.select(
        pl.when(done).then(hi).otherwise(None).alias("or_high"),
        pl.when(done).then(lo).otherwise(None).alias("or_low"),
    )
