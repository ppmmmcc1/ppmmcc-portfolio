"""Data sources. Upgrading from free CSV to Databento is a config change
(``data.source`` in settings.yaml), not a rewrite."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl

from nqlab.data.schema import validate
from nqlab.log import get_logger
from nqlab.settings import Settings

log = get_logger("data.sources")


@dataclass(frozen=True)
class SourceCapabilities:
    has_trades: bool
    has_book: bool
    history_start: dt.date | None
    per_contract: bool          # True if bars carry real contract symbols (enables volume roll)
    provenance: str


class DataSource(Protocol):
    name: str

    def fetch_bars(self, symbol: str, start: dt.date, end: dt.date, interval: str = "1m") -> pl.DataFrame: ...
    def fetch_trades(self, symbol: str, start: dt.date, end: dt.date) -> pl.DataFrame | None: ...
    def capabilities(self) -> SourceCapabilities: ...


# ---------------------------------------------------------------------------------------
# CSV (primary for v1)
# ---------------------------------------------------------------------------------------
_GUESSES = {
    "timestamp": [["timestamp"], ["datetime"], ["date_time"], ["time"], ["date", "time"], ["date"]],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "last", "c"],
    "volume": ["volume", "vol", "v"],
}


class CsvSource:
    name = "csv"

    def __init__(self, settings: Settings, files: list[Path] | None = None):
        self.settings = settings
        self.cfg = settings.section("data")["csv"]
        raw_dir = settings.path("data_raw")
        self.files = files or sorted(raw_dir.glob(self.cfg.get("pattern", "*.csv")))
        self.notes: list[str] = []

    def capabilities(self) -> SourceCapabilities:
        per_contract = bool(self.cfg["columns"].get("symbol"))
        return SourceCapabilities(False, False, None, per_contract,
                                  "primary" if per_contract else "roll_unknown")

    def fetch_trades(self, symbol, start, end):
        return None

    def _resolve_columns(self, headers: list[str]) -> dict[str, list[str] | str | None]:
        lower = {h.lower().strip(): h for h in headers}
        cols = dict(self.cfg["columns"])
        out: dict[str, list[str] | str | None] = {}
        for key in ("timestamp", "open", "high", "low", "close", "volume"):
            want = cols.get(key)
            want_list = want if isinstance(want, list) else [want]
            if want and all(str(w).lower() in lower for w in want_list):
                found = [lower[str(w).lower()] for w in want_list]
                out[key] = found if key == "timestamp" else found[0]
                continue
            for g in _GUESSES[key]:
                gl = g if isinstance(g, list) else [g]
                if all(x in lower for x in gl):
                    found = [lower[x] for x in gl]
                    out[key] = found if key == "timestamp" else found[0]
                    self.notes.append(f"column '{key}' not found as configured ({want}); GUESSED {found}")
                    break
            else:
                raise ValueError(f"cannot map column {key!r}; headers are {headers}. Edit data.csv.columns")
        sym = cols.get("symbol")
        out["symbol"] = lower.get(str(sym).lower()) if sym else None
        return out

    def _parse_ts(self, s: pl.Series) -> pl.Series:
        fmt = self.cfg.get("timestamp_format")
        tz = self.cfg.get("source_timezone") or "UTC"
        if s.dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.Float64):
            # epoch: guess unit by magnitude
            v = s.cast(pl.Int64)
            mx = v.max() or 0
            unit = "s" if mx < 10**11 else "ms" if mx < 10**14 else "us" if mx < 10**17 else "ns"
            self.notes.append(f"epoch timestamps interpreted as unit={unit}, UTC")
            return pl.from_epoch(v, time_unit=unit).dt.replace_time_zone("UTC")
        s = s.cast(pl.Utf8).str.strip_chars()
        if fmt:
            parsed = s.str.strptime(pl.Datetime("us"), fmt, strict=True)
        else:
            parsed = s.str.to_datetime(time_unit="us", strict=True)
        if parsed.dtype.time_zone is None:  # type: ignore[union-attr]
            parsed = parsed.dt.replace_time_zone(tz, ambiguous="earliest", non_existent="raise")
        return parsed.dt.convert_time_zone("UTC")

    def _load_one(self, path: Path) -> pl.DataFrame:
        df = pl.read_csv(path, infer_schema_length=10000, try_parse_dates=False)
        m = self._resolve_columns(df.columns)
        tcols = m["timestamp"]
        assert isinstance(tcols, list)
        if len(tcols) == 1:
            ts_raw = df[tcols[0]]
        else:
            ts_raw = df.select(pl.concat_str([pl.col(c).cast(pl.Utf8) for c in tcols], separator=" ")).to_series()
        ts = self._parse_ts(ts_raw)
        if self.cfg.get("bar_label", "open") == "close":
            ts = ts - dt.timedelta(minutes=1)
        sym_col = m["symbol"]
        prov = "primary" if sym_col else "roll_unknown"
        out = pl.DataFrame({
            "ts_utc": ts,
            "symbol": df[sym_col].cast(pl.Utf8).str.to_uppercase() if sym_col else
                      pl.Series([self.cfg.get("contract_symbol_default", "NQ_CONT")] * df.height),
            "open": df[m["open"]].cast(pl.Float64),
            "high": df[m["high"]].cast(pl.Float64),
            "low": df[m["low"]].cast(pl.Float64),
            "close": df[m["close"]].cast(pl.Float64),
            "volume": df[m["volume"]].cast(pl.Float64).round(0).cast(pl.Int64),
            "source": pl.Series(["csv"] * df.height),
            "provenance": pl.Series([prov] * df.height),
            "data_version": pl.Series([""] * df.height),
        })
        return validate(out)

    def fetch_bars(self, symbol: str | None = None, start: dt.date | None = None,
                   end: dt.date | None = None, interval: str = "1m") -> pl.DataFrame:
        if interval != "1m":
            raise ValueError("CsvSource supplies 1-minute bars only")
        if not self.files:
            raise FileNotFoundError(f"no CSV files in {self.settings.path('data_raw')}")
        df = pl.concat([self._load_one(f) for f in self.files]).sort("ts_utc", "symbol")
        if start:
            df = df.filter(pl.col("ts_utc") >= dt.datetime.combine(start, dt.time(), dt.timezone.utc))
        if end:
            df = df.filter(pl.col("ts_utc") < dt.datetime.combine(end + dt.timedelta(days=1), dt.time(), dt.timezone.utc))
        return df


# ---------------------------------------------------------------------------------------
# yfinance (fallback / top-up) - LOW CONFIDENCE
# ---------------------------------------------------------------------------------------
class YFinanceSource:
    """Yahoo serves only a short recent window of 1m data for NQ=F, and its roll handling
    is undocumented. Everything it returns is marked ``low_confidence``."""

    name = "yfinance"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ticker = settings.section("data").get("yfinance", {}).get("ticker", "NQ=F")
        self.notes = ["yfinance data is LOW CONFIDENCE: short window, undocumented roll handling"]

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(False, False, dt.date.today() - dt.timedelta(days=29), False, "low_confidence")

    def fetch_trades(self, symbol, start, end):
        return None

    def fetch_bars(self, symbol: str | None = None, start: dt.date | None = None,
                   end: dt.date | None = None, interval: str = "1m") -> pl.DataFrame:
        import yfinance as yf

        start = start or dt.date.today() - dt.timedelta(days=7)
        end = end or dt.date.today()
        frames = []
        cur = start
        while cur <= end:  # Yahoo caps 1m requests at 8 days
            nxt = min(cur + dt.timedelta(days=7), end + dt.timedelta(days=1))
            pdf = yf.download(self.ticker, start=cur.isoformat(), end=nxt.isoformat(),
                              interval="1m", progress=False, auto_adjust=False, multi_level_index=False)
            if pdf is not None and len(pdf):
                pdf = pdf.reset_index()
                frames.append(pl.from_pandas(pdf))
            cur = nxt
        if not frames:
            raise RuntimeError("yfinance returned no data")
        raw = pl.concat(frames, how="diagonal_relaxed")
        tcol = "Datetime" if "Datetime" in raw.columns else raw.columns[0]
        out = pl.DataFrame({
            "ts_utc": raw[tcol].dt.convert_time_zone("UTC"),
            "symbol": pl.Series(["NQ_CONT"] * raw.height),
            "open": raw["Open"], "high": raw["High"], "low": raw["Low"], "close": raw["Close"],
            "volume": raw["Volume"].cast(pl.Int64),
            "source": pl.Series(["yfinance"] * raw.height),
            "provenance": pl.Series(["low_confidence"] * raw.height),
            "data_version": pl.Series([""] * raw.height),
        })
        return validate(out.unique("ts_utc").sort("ts_utc"))


def get_source(settings: Settings) -> DataSource:
    name = settings.section("data").get("source", "csv")
    if name == "csv":
        return CsvSource(settings)
    if name == "yfinance":
        return YFinanceSource(settings)
    if name == "databento":
        from nqlab.data.databento_source import DatabentoSource

        if not os.environ.get("DATABENTO_API_KEY"):
            raise RuntimeError("data.source=databento but DATABENTO_API_KEY is not set")
        return DatabentoSource(settings)
    raise ValueError(f"unknown data source {name!r}")
