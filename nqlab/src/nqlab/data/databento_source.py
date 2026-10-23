"""Databento GLBX.MDP3 source: ohlcv-1m, trades, mbo. Complete but INACTIVE until
DATABENTO_API_KEY is set and settings.data.source = databento.

The cost of every request is estimated with ``metadata.get_cost`` and printed BEFORE any
download; requests above ``data.databento.max_cost_usd`` are refused unless confirmed.
This is the upgrade path and the only source that enables fill tiers 3 and 4."""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

import polars as pl

from nqlab.calendar.roll import canonical_contract
from nqlab.data.schema import validate
from nqlab.data.sources import SourceCapabilities
from nqlab.log import get_logger

log = get_logger("data.databento")


class CostRefused(RuntimeError):
    pass


class DatabentoSource:
    name = "databento"

    def __init__(self, settings, client: Any = None, confirm_cost: bool = False):
        self.settings = settings
        self.cfg = settings.section("data")["databento"]
        if client is None:
            import databento as db

            key = os.environ.get("DATABENTO_API_KEY")
            if not key:
                raise RuntimeError("DATABENTO_API_KEY is not set")
            client = db.Historical(key)
        self.client = client
        self.confirm_cost = confirm_cost
        self.notes: list[str] = []
        self.files: list = []

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(True, True, dt.date(2010, 6, 6), True, "primary")

    # ---- cost gate --------------------------------------------------------------------
    def estimate(self, schema: str, start: dt.date, end: dt.date) -> float:
        cost = float(self.client.metadata.get_cost(
            dataset=self.cfg["dataset"], start=start.isoformat(), end=(end + dt.timedelta(days=1)).isoformat(),
            symbols=self.cfg["symbols"], schema=schema, stype_in=self.cfg["stype_in"]))
        msg = f"Databento {schema} {start}..{end} {self.cfg['symbols']}: estimated cost ${cost:,.2f}"
        print(msg)
        log.warning(msg)
        cap = float(self.cfg.get("max_cost_usd", 50))
        if cost > cap and not self.confirm_cost:
            raise CostRefused(f"{msg} exceeds max_cost_usd=${cap:,.2f}; re-run with --confirm-cost to proceed")
        return cost

    def _get(self, schema: str, start: dt.date, end: dt.date):
        self.estimate(schema, start, end)
        store = self.client.timeseries.get_range(
            dataset=self.cfg["dataset"], start=start.isoformat(), end=(end + dt.timedelta(days=1)).isoformat(),
            symbols=self.cfg["symbols"], schema=schema, stype_in=self.cfg["stype_in"])
        pdf = store.to_df(price_type="float", pretty_ts=True, map_symbols=True)
        return pl.from_pandas(pdf.reset_index())

    @staticmethod
    def _contracts(df: pl.DataFrame, ts_col: str) -> pl.DataFrame:
        first = df.group_by("symbol").agg(pl.col(ts_col).min().dt.year().alias("y"))
        mapping = {s: canonical_contract(s, int(y)) for s, y in first.iter_rows()}
        return df.with_columns(pl.col("symbol").replace_strict(mapping, default=None).alias("symbol")) \
                 .filter(pl.col("symbol").is_not_null())          # drops calendar spreads

    def fetch_bars(self, symbol: str | None = None, start: dt.date | None = None, end: dt.date | None = None,
                   interval: str = "1m") -> pl.DataFrame:
        if interval != "1m":
            raise ValueError("only 1m bars are supported")
        start = start or dt.date(2019, 1, 1)
        end = end or dt.date.today() - dt.timedelta(days=1)
        raw = self._get("ohlcv-1m", start, end)
        raw = self._contracts(raw, "ts_event")
        out = pl.DataFrame({
            "ts_utc": raw["ts_event"].dt.convert_time_zone("UTC").dt.cast_time_unit("us"),
            "symbol": raw["symbol"], "open": raw["open"], "high": raw["high"], "low": raw["low"],
            "close": raw["close"], "volume": raw["volume"].cast(pl.Int64),
            "source": pl.Series(["databento"] * raw.height), "provenance": pl.Series(["primary"] * raw.height),
            "data_version": pl.Series([""] * raw.height)})
        self.files = []
        return validate(out.sort("ts_utc", "symbol"))

    def fetch_trades(self, symbol: str | None, start: dt.date, end: dt.date) -> pl.DataFrame:
        raw = self._contracts(self._get("trades", start, end), "ts_event")
        return raw.select(pl.col("ts_event").dt.convert_time_zone("UTC").alias("ts_utc"), "symbol",
                          pl.col("price").cast(pl.Float64), pl.col("size").cast(pl.Int64), pl.col("side").cast(pl.Utf8))

    def fetch_mbo(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        raw = self._contracts(self._get("mbo", start, end), "ts_event")
        return raw.select(pl.col("ts_event").dt.convert_time_zone("UTC").alias("ts_utc"), "symbol",
                          pl.col("action").cast(pl.Utf8), pl.col("side").cast(pl.Utf8), pl.col("price").cast(pl.Float64),
                          pl.col("size").cast(pl.Int64), pl.col("order_id").cast(pl.UInt64))
