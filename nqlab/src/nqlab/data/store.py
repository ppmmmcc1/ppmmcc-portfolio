"""Parquet store. Every read of market data goes through here (holdout guard).

Layout::

    data/parquet/ohlcv-1m/{SYMBOL}/{YYYY-MM}/part.parquet   per-contract (or as supplied)
    data/parquet/front-1m/{ROOT}/{YYYY-MM}/part.parquet     continuous front month + adj
    data/parquet/sessions.parquet                           session table (incl. flags)
    data/parquet/manifest.json                              data_version, provenance, rolls
    data/holdout/front-1m/{ROOT}/...                        the lockbox (same layout)
    data/lockbox.json                                       holdout boundary (fixed forever)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl

from nqlab.data.holdout import _ingest_write_access, assert_readable, register_guard
from nqlab.settings import Settings


class DataStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.parquet = settings.path("parquet")
        self.holdout = settings.path("holdout")
        self.lockbox_file = self.parquet.parent / "lockbox.json"
        register_guard(self.holdout)

    # ---- writing -------------------------------------------------------------------
    def _write_partitioned(self, base: Path, df: pl.DataFrame) -> None:
        if df.height == 0:
            return
        df = df.with_columns(pl.col("ts_utc").dt.strftime("%Y-%m").alias("_ym"))
        for (ym,), part in df.group_by(["_ym"], maintain_order=True):
            d = base / ym
            d.mkdir(parents=True, exist_ok=True)
            part.drop("_ym").sort("ts_utc").write_parquet(d / "part.parquet")

    def write_raw(self, df: pl.DataFrame, schema: str = "ohlcv-1m") -> None:
        base = self.parquet / schema
        if base.exists():
            shutil.rmtree(base)
        for (sym,), part in df.group_by(["symbol"]):
            self._write_partitioned(base / str(sym), part)

    def write_front(self, research: pl.DataFrame, holdout: pl.DataFrame, root: str) -> None:
        base = self.parquet / "front-1m" / root
        if base.exists():
            shutil.rmtree(base)
        self._write_partitioned(base, research)
        with _ingest_write_access():
            hbase = self.holdout / "front-1m" / root
            if hbase.exists():
                shutil.rmtree(hbase)
            self._write_partitioned(hbase, holdout)
            self.holdout.mkdir(parents=True, exist_ok=True)
            (self.holdout / "README_LOCKED.txt").write_text(
                "LOCKED HOLDOUT. Do not open, load, chart or summarize anything in this folder.\n"
                "Looking at it even once turns it into in-sample data and the final test becomes meaningless.\n"
                "The only sanctioned read is `nqlab holdout <family> --confirm`, which records the look forever.\n",
                encoding="utf-8")

    def write_sessions(self, sessions: pl.DataFrame, holdout_sessions: pl.DataFrame) -> None:
        self.parquet.mkdir(parents=True, exist_ok=True)
        sessions.write_parquet(self.parquet / "sessions.parquet")
        with _ingest_write_access():
            self.holdout.mkdir(parents=True, exist_ok=True)
            holdout_sessions.write_parquet(self.holdout / "sessions.parquet")

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.parquet.mkdir(parents=True, exist_ok=True)
        (self.parquet / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # ---- lockbox boundary ------------------------------------------------------------
    def lockbox(self) -> dict[str, Any] | None:
        if self.lockbox_file.exists():
            return json.loads(self.lockbox_file.read_text(encoding="utf-8"))
        return None

    def set_lockbox(self, info: dict[str, Any]) -> None:
        if self.lockbox_file.exists():
            raise RuntimeError("lockbox boundary already set; it never moves")
        self.lockbox_file.parent.mkdir(parents=True, exist_ok=True)
        self.lockbox_file.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    # ---- reading ---------------------------------------------------------------------
    def manifest(self) -> dict[str, Any]:
        p = self.parquet / "manifest.json"
        if not p.exists():
            raise FileNotFoundError("no data ingested yet; run `nqlab data ingest`")
        return json.loads(p.read_text(encoding="utf-8"))

    def _scan(self, base: Path) -> pl.LazyFrame:
        assert_readable(base)
        files = sorted(base.glob("*/part.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet under {base}")
        return pl.scan_parquet([str(f) for f in files])

    def scan_front(self, root: str = "NQ", holdout: bool = False) -> pl.LazyFrame:
        base = (self.holdout if holdout else self.parquet) / "front-1m" / root
        return self._scan(base)

    def scan_raw(self, symbol: str, schema: str = "ohlcv-1m") -> pl.LazyFrame:
        return self._scan(self.parquet / schema / symbol)

    def sessions(self, holdout: bool = False) -> pl.DataFrame:
        p = (self.holdout if holdout else self.parquet) / "sessions.parquet"
        assert_readable(p)
        return pl.read_parquet(p)
