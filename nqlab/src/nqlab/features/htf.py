"""Higher-timeframe bars and their alignment onto 1-minute rows.

Each HTF bar is indexed by its CLOSE timestamp and joined with a backward as-of join on
the 1m row's close timestamp. At 10:07 (1m bar 10:06-10:07 just closed) the newest 15m
bar visible is 09:45-10:00; 10:00-10:15 appears only on the row whose close is 10:15.

``align``: ``clock`` buckets start on the hour (10:00, 11:00); ``session`` buckets start
at the half hour, aligned with the 08:30 CT / 09:30 ET cash open (09:30, 10:30 ET). They
produce materially different hourly signals; the strategy spec must choose.
"""

from __future__ import annotations

import polars as pl

from nqlab.features.indicators import bollinger_expr, macd_expr
from nqlab.features.registry import SERIES, _SUFFIX, feature

_ALLOWED_TF = (5, 15, 30, 60)


def htf_bars(md, tf: int, align: str = "clock", series: str = "back_adjusted") -> pl.DataFrame:
    if tf not in _ALLOWED_TF:
        raise ValueError(f"tf must be one of {_ALLOWED_TF}")
    if align not in ("clock", "session"):
        raise ValueError("align must be 'clock' or 'session'")
    if series not in SERIES:
        raise ValueError(series)
    sfx = _SUFFIX[series]
    # only hourly bars differ between alignments (5/15/30m buckets already align with :30)
    offset_min = 30 if (align == "session" and tf == 60) else 0
    tf_ns = tf * 60 * 1_000_000_000
    off_ns = offset_min * 60 * 1_000_000_000
    ns = pl.col("ts_utc").dt.epoch("ns")
    bucket = ((ns - off_ns) // tf_ns) * tf_ns + off_ns
    b = md.bars.with_columns(bucket.alias("_bucket"))
    agg = (
        b.group_by("_bucket", maintain_order=True)
        .agg(
            pl.col("open" + sfx).first().alias("open"),
            pl.col("high" + sfx).max().alias("high"),
            pl.col("low" + sfx).min().alias("low"),
            pl.col("close" + sfx).last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        )
        .sort("_bucket")
        .with_columns(pl.from_epoch(pl.col("_bucket") + tf_ns, time_unit="ns")
                      .dt.replace_time_zone("UTC").dt.cast_time_unit("us").alias("htf_close_utc"))
        .drop("_bucket")
    )
    return agg


def align_to_1m(md, htf: pl.DataFrame, cols: list[str], prefix: str) -> pl.DataFrame:
    left = md.bars.select(pl.col("ts_close_utc").cast(pl.Datetime("us", "UTC")), pl.int_range(pl.len()).alias("_i"))
    right = htf.select(["htf_close_utc", *cols]).rename({c: f"{prefix}{c}" for c in cols})
    j = left.sort("ts_close_utc").join_asof(right.sort("htf_close_utc"), left_on="ts_close_utc",
                                             right_on="htf_close_utc", strategy="backward")
    return j.sort("_i").select([f"{prefix}{c}" for c in cols] + [pl.col("htf_close_utc").alias(f"{prefix}close_ts")])


@feature("htf_macd", series="param",
         defaults={"tf": 15, "fast": 12, "slow": 26, "signal": 9, "source": "close",
                   "align": "clock", "series": "back_adjusted"})
def htf_macd(md, tf, fast, slow, signal, source, align, series) -> pl.DataFrame:
    h = htf_bars(md, tf, align, series)
    x = pl.col(source) if source != "hlc3" else (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    h = h.with_columns(macd_expr(x, fast, slow, signal))
    return align_to_1m(md, h, ["macd_line", "macd_signal", "macd_hist"], f"m{tf}_")


@feature("htf_bollinger", series="param",
         defaults={"tf": 5, "length": 20, "mult": 2.0, "basis": "sma", "stdev": "population",
                   "source": "close", "align": "clock", "series": "back_adjusted"})
def htf_bollinger(md, tf, length, mult, basis, stdev, source, align, series) -> pl.DataFrame:
    h = htf_bars(md, tf, align, series)
    x = pl.col(source) if source != "hlc3" else (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    h = h.with_columns(bollinger_expr(x, length, mult, basis, stdev))
    return align_to_1m(md, h, ["bb_basis", "bb_upper", "bb_lower"], f"b{tf}_")
