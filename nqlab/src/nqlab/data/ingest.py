"""Ingest pipeline: source -> canonical -> sessions -> roll -> quality -> lockbox -> parquet."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import polars as pl

from nqlab.calendar.roll import build_continuous, cme_convention_front, volume_front, apply_adjustment
from nqlab.calendar.sessions import build_sessions, ct_minute_expr, session_date_expr
from nqlab.data.quality import QUALITY_VERSION, QualityReport, check_raw_duplicates, run_checks
from nqlab.data.store import DataStore
from nqlab.hashing import file_hash, stable_hash
from nqlab.log import get_logger
from nqlab.settings import Settings

log = get_logger("data.ingest")


@dataclass
class IngestResult:
    data_version: str
    n_bars: int
    n_sessions: int
    n_holdout_sessions: int
    n_excluded: int
    holdout_start: dt.date
    report: QualityReport
    roll_method: str
    provenance: str


def _months_before(d: dt.date, months: int) -> dt.date:
    y, m = d.year, d.month - months
    while m <= 0:
        y, m = y - 1, m + 12
    day = min(d.day, 28)
    return dt.date(y, m, day)


def ingest(settings: Settings, raw: pl.DataFrame | None = None, source_notes: list[str] | None = None,
           source_fingerprint: Any = None) -> IngestResult:
    from nqlab.data.sources import get_source

    notes = list(source_notes or [])
    if raw is None:
        src = get_source(settings)
        raw = src.fetch_bars()
        notes += getattr(src, "notes", [])
        files = getattr(src, "files", [])
        source_fingerprint = [file_hash(f) for f in files] if files else stable_hash(raw.select("ts_utc").tail(1).rows())
    root = settings.section("data").get("root_symbol", "NQ")
    series, sessions, rolls, rep, roll_method, provenance, dates, first, last, raw = process(raw, settings, notes)

    data_version = stable_hash({
        "src": source_fingerprint, "roll": roll_method, "csv": settings.section("data").get("csv"),
        "q": QUALITY_VERSION, "events": str(settings.path("events").stat().st_mtime_ns) if settings.path("events").exists() else None,
    })
    series = series.with_columns(pl.lit(data_version).alias("data_version"))

    excluded = {dt.date.fromisoformat(d): ", ".join(r) for d, r in rep.excluded.items()}

    # ---- lockbox ----------------------------------------------------------------------
    store = DataStore(settings)
    lb = store.lockbox()
    if lb is None:
        months = int(settings.section("data").get("holdout_months", 6))
        hstart = _months_before(last, months)
        store.set_lockbox({"holdout_start": hstart.isoformat(), "set_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                           "months": months, "first_data_version": data_version})
        log.warning("Holdout lockbox created: sessions on/after %s are locked", hstart)
    else:
        hstart = dt.date.fromisoformat(lb["holdout_start"])
    research = series.filter(pl.col("session_date") < hstart)
    hold = series.filter(pl.col("session_date") >= hstart)
    sess_r = sessions.filter(pl.col("session_date") < hstart)
    sess_h = sessions.filter(pl.col("session_date") >= hstart)

    store.write_raw(raw.filter(pl.col("session_date") < hstart).drop("session_date", "ct_minute"))
    store.write_front(research, hold, root)
    store.write_sessions(sess_r, sess_h)
    manifest = {
        "data_version": data_version, "root": root, "roll_method": roll_method,
        "provenance": provenance, "rolls": rolls.to_dicts(), "quality": rep.counts,
        "warnings": rep.warnings, "excluded_sessions": {str(k): v for k, v in excluded.items() if k < hstart},
        "first_session": first, "last_research_session": max([d for d in dates if d < hstart], default=None),
        "holdout_start": hstart, "events_verified": False, "ingested_at": dt.datetime.now(dt.timezone.utc),
    }
    store.write_manifest(manifest)

    rep_dir = settings.path("reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"quality_{data_version}.md").write_text(rep.to_markdown(), encoding="utf-8")

    from nqlab.results.db import connect

    with connect(settings.path("results_db")) as con:
        con.execute("DELETE FROM excluded_sessions WHERE data_version = ?", [data_version])
        rows = [(data_version, d, r) for d, r in excluded.items()]
        if rows:
            con.executemany("INSERT INTO excluded_sessions VALUES (?, ?, ?)", rows)

    return IngestResult(
        data_version=data_version, n_bars=research.height, n_sessions=sess_r.filter(pl.col("has_data")).height,
        n_holdout_sessions=sess_h.filter(pl.col("has_data")).height, n_excluded=len(manifest["excluded_sessions"]),
        holdout_start=hstart, report=rep, roll_method=roll_method, provenance=provenance,
    )


def process(raw: pl.DataFrame, settings: Settings, notes: list[str] | None = None):
    """Pure transformation shared by ingest and in-memory datasets (tests, synthetic)."""
    notes = list(notes or [])
    root = settings.section("data").get("root_symbol", "NQ")
    rep = QualityReport()
    for n in notes:
        rep.warnings.append(n)
    provenance = ",".join(sorted(set(raw["provenance"].unique().to_list())))
    if "low_confidence" in provenance:
        rep.warnings.append("LOW CONFIDENCE DATA (yfinance): short history, undocumented roll handling. "
                            "Do not draw conclusions from it.")

    raw = check_raw_duplicates(raw, rep)
    raw = raw.with_columns(session_date_expr().alias("session_date"), ct_minute_expr().alias("ct_minute"))
    first, last = raw["session_date"].min(), raw["session_date"].max()
    sessions = build_sessions(first, last, settings)

    known = set(sessions["session_date"].to_list())
    orphan = raw.filter(~pl.col("session_date").is_in(list(known)))
    rep.counts["bars_outside_any_session"] = orphan.height
    raw = raw.filter(pl.col("session_date").is_in(list(known)))

    # ---- roll / front month ---------------------------------------------------------
    per_contract = not raw["symbol"].str.ends_with("_CONT").all()
    roll_cfg = settings.section("data")["roll"]
    dates = sorted(raw["session_date"].unique().to_list())
    if per_contract and roll_cfg.get("method", "volume") == "volume":
        dv = raw.group_by("session_date", "symbol").agg(pl.col("volume").sum())
        front = volume_front(dv, root)
        roll_method = "volume"
    elif per_contract:
        front = cme_convention_front(dates, root, int(roll_cfg.get("cme_days_before_expiry", 8)))
        roll_method = "cme_convention"
    else:
        front = pl.DataFrame({"session_date": dates, "front": [raw["symbol"][0]] * len(dates)},
                             schema={"session_date": pl.Date, "front": pl.Utf8})
        roll_method = "source_continuous(unknown)"
        rep.warnings.append(
            "Source is a pre-spliced continuous series with no contract column: roll dates and roll "
            "gaps are UNKNOWN. Back-adjusted series equal unadjusted; lookbacks spanning a roll "
            "(e.g. 1h MACD) may be contaminated by roll gaps on roll days."
        )
    if per_contract:
        series, rolls = build_continuous(raw, front)
    else:
        series = raw.sort("ts_utc")
        rolls = pl.DataFrame(schema={"roll_session": pl.Date, "from": pl.Utf8, "to": pl.Utf8,
                                     "gap": pl.Float64, "ratio": pl.Float64, "gap_method": pl.Utf8})
        series = apply_adjustment(series, rolls)
    # contract label for reporting even when the source is continuous
    conv = cme_convention_front(dates, root).rename({"front": "contract_conv"})
    series = series.join(conv, on="session_date", how="left")

    series = run_checks(series, sessions, rep)
    rep.info["roll_method"] = roll_method
    rep.info["provenance"] = provenance

    sessions = mark_sessions(sessions, series, rep, last)
    return series, sessions, rolls, rep, roll_method, provenance, dates, first, last, raw


def mark_sessions(sessions: pl.DataFrame, series: pl.DataFrame, rep: QualityReport, last: dt.date) -> pl.DataFrame:
    excluded = {dt.date.fromisoformat(d): ", ".join(r) for d, r in rep.excluded.items()}
    data_dates = set(series["session_date"].unique().to_list())
    reasons = pl.DataFrame({"session_date": list(excluded), "exclude_reason": list(excluded.values())},
                           schema={"session_date": pl.Date, "exclude_reason": pl.Utf8})
    return (
        sessions.filter(pl.col("session_date") <= last)
        .join(reasons, on="session_date", how="left")
        .with_columns(
            pl.col("session_date").is_in(list(data_dates)).alias("has_data"),
            pl.col("exclude_reason").fill_null(""),
        )
        .with_columns((pl.col("exclude_reason") != "").alias("excluded"))
    )
