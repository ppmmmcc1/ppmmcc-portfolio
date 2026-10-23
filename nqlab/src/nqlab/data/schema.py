"""Canonical bar schema. ``ts_utc`` is the bar's OPEN instant (tz-aware UTC)."""

from __future__ import annotations

import polars as pl

CANONICAL = {
    "ts_utc": pl.Datetime("us", "UTC"),
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "source": pl.Utf8,
    "provenance": pl.Utf8,
    "data_version": pl.Utf8,
}

PROVENANCE = ("primary", "roll_unknown", "low_confidence", "synthetic")


def validate(df: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in CANONICAL if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical columns: {missing}")
    df = df.select([pl.col(c).cast(t) for c, t in CANONICAL.items()])
    tz = df.schema["ts_utc"].time_zone
    if tz != "UTC":
        raise ValueError(f"ts_utc must be tz-aware UTC, got {tz!r}")
    if df["provenance"].is_in(list(PROVENANCE)).not_().any():
        raise ValueError("unknown provenance value")
    return df
