"""MarketData: the research dataset a strategy sees (never the holdout, unless the
holdout command builds one inside ``holdout_access``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from nqlab.calendar.sessions import ct_minute_expr
from nqlab.settings import Settings


@dataclass
class MarketData:
    bars: pl.DataFrame            # full 1m history, sorted by ts_utc
    sessions: pl.DataFrame        # sessions with data (incl. excluded flag/reason)
    data_version: str
    provenance: str = "primary"
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        b = self.bars
        if "ct_minute" not in b.columns:
            b = b.with_columns(ct_minute_expr().alias("ct_minute"))
        if "ts_close_utc" not in b.columns:
            b = b.with_columns((pl.col("ts_utc") + pl.duration(minutes=1)).alias("ts_close_utc"))
        if "bad_bar" not in b.columns:
            b = b.with_columns(pl.lit(False).alias("bad_bar"))
        for c in ("open", "high", "low", "close"):
            if f"{c}_badj" not in b.columns:
                b = b.with_columns(pl.col(c).alias(f"{c}_badj"), pl.col(c).alias(f"{c}_radj"))
        if "adj_offset" not in b.columns:
            b = b.with_columns((pl.col("close_badj") - pl.col("close")).alias("adj_offset"))
        self.bars = b.sort("ts_utc")
        s = self.sessions
        if "has_data" in s.columns:
            s = s.filter(pl.col("has_data"))
        if "excluded" not in s.columns:
            s = s.with_columns(pl.lit(False).alias("excluded"), pl.lit("").alias("exclude_reason"))
        present = self.bars["session_date"].unique()
        self.sessions = s.filter(pl.col("session_date").is_in(present.implode())).sort("session_date")

    # ------------------------------------------------------------------------------
    @classmethod
    def from_store(cls, settings: Settings, holdout: bool = False) -> "MarketData":
        from nqlab.data.store import DataStore

        store = DataStore(settings)
        man = store.manifest()
        bars = store.scan_front(man["root"], holdout=holdout).collect()
        sess = store.sessions(holdout=holdout)
        return cls(bars, sess, man["data_version"] + ("-holdout" if holdout else ""),
                   provenance=man.get("provenance", ""), warnings=list(man.get("warnings", [])),
                   meta={"roll_method": man.get("roll_method"), "manifest": man})

    @classmethod
    def from_raw(cls, raw: pl.DataFrame, settings: Settings, data_version: str = "inmem") -> "MarketData":
        """Run the ingest transformation in memory (no files, no lockbox). Tests only."""
        from nqlab.data.ingest import process

        series, sessions, rolls, rep, roll_method, provenance, *_ = process(raw, settings)
        return cls(series, sessions, data_version, provenance=provenance, warnings=rep.warnings,
                   meta={"roll_method": roll_method, "rolls": rolls, "quality": rep})

    # ------------------------------------------------------------------------------
    def truncate(self, ts_close_max) -> "MarketData":
        """Data as it existed at ``ts_close_max`` (bars whose close <= that instant)."""
        b = self.bars.filter(pl.col("ts_close_utc") <= ts_close_max)
        return MarketData(b, self.sessions, self.data_version + f"@{ts_close_max}", self.provenance,
                          self.warnings, self._fresh_meta())

    def with_bars(self, bars: pl.DataFrame, data_version: str) -> "MarketData":
        return MarketData(bars, self.sessions, data_version, self.provenance, self.warnings, self._fresh_meta())

    def _fresh_meta(self) -> dict[str, Any]:
        """Derived datasets never inherit cached features (they were computed on other bars)."""
        return {k: v for k, v in self.meta.items() if not k.startswith("_")}

    def session_dates(self, include_excluded: bool = False) -> list:
        s = self.sessions if include_excluded else self.sessions.filter(~pl.col("excluded"))
        return s["session_date"].to_list()

    def ts_ns(self) -> np.ndarray:
        return self.bars["ts_utc"].dt.epoch("ns").to_numpy()
