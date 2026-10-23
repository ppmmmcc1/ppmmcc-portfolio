"""Ingest-time data quality checks.

A bar that fails a check never silently triggers a trade: its whole session is excluded
and the exclusion is logged (manifest, DuckDB ``excluded_sessions``, every report shows
the count). Exclusion rules are deliberately narrow so that genuinely volatile sessions
are NOT dropped (dropping them would flatter a strategy): an out-of-range print only
counts when it is a one-bar spike that reverts, not a sustained move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

QUALITY_VERSION = "q1"
TICK = 0.25


@dataclass
class QualityReport:
    counts: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, list[str]] = field(default_factory=dict)  # date -> reasons
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def exclude(self, date, reason: str) -> None:
        self.excluded.setdefault(str(date), [])
        if reason not in self.excluded[str(date)]:
            self.excluded[str(date)].append(reason)

    def to_markdown(self, research_before=None) -> str:
        """``research_before``: hide dates on/after the holdout boundary (count only)."""
        lines = ["# Data quality report", ""]
        for w in self.warnings:
            lines.append(f"> **WARNING:** {w}")
        if self.warnings:
            lines.append("")
        lines.append("| check | count |")
        lines.append("|---|---|")
        for k, v in sorted(self.counts.items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")
        shown = {d: r for d, r in self.excluded.items() if research_before is None or d < str(research_before)}
        hidden = len(self.excluded) - len(shown)
        lines.append(f"**Excluded sessions: {len(self.excluded)}** (never traded, never silently dropped)"
                     + (f"; {hidden} of them are in the locked holdout (dates hidden)" if hidden else ""))
        lines.append("")
        for d, rs in sorted(shown.items()):
            lines.append(f"- {d}: {', '.join(rs)}")
        for k, v in self.info.items():
            lines.append(f"\n**{k}:** {v}")
        return "\n".join(lines) + "\n"


def check_raw_duplicates(raw: pl.DataFrame, rep: QualityReport) -> pl.DataFrame:
    """Exact duplicates are dropped (counted). Conflicting duplicates exclude the session."""
    dup_mask = raw.select(pl.struct("ts_utc", "symbol").is_duplicated()).to_series()
    n_dup = int(dup_mask.sum())
    rep.counts["duplicate_timestamps"] = n_dup
    if n_dup:
        conflicting = (
            raw.filter(dup_mask)
            .group_by("ts_utc", "symbol")
            .agg(pl.struct("open", "high", "low", "close", "volume").n_unique().alias("nu"))
            .filter(pl.col("nu") > 1)
        )
        rep.counts["duplicate_timestamps_conflicting"] = conflicting.height
        rep.info["_conflicting_dup_ts"] = conflicting["ts_utc"].to_list()
    return raw.unique(subset=["ts_utc", "symbol"], keep="first", maintain_order=True)


def run_checks(series: pl.DataFrame, sessions: pl.DataFrame, rep: QualityReport) -> pl.DataFrame:
    """``series``: continuous front month with session_date, ct_minute. Returns series with
    a boolean ``bad_bar`` column; fills ``rep``."""
    s = series.sort("ts_utc")
    sess = sessions.select("session_date", "rth_open_utc", "rth_close_utc")
    s = s.join(sess, on="session_date", how="left")
    in_rth = (pl.col("ts_utc") >= pl.col("rth_open_utc")) & (pl.col("ts_utc") < pl.col("rth_close_utc"))
    s = s.with_columns(in_rth.fill_null(False).alias("in_rth"))

    # --- tick grid ------------------------------------------------------------------
    off_grid = pl.lit(False)
    for c in ("open", "high", "low", "close"):
        x = pl.col(c) / TICK
        off_grid = off_grid | ((x - x.round(0)).abs() > 1e-6)
    # --- OHLC consistency -------------------------------------------------------------
    incoherent = (
        (pl.col("high") < pl.max_horizontal("open", "close") - 1e-9)
        | (pl.col("low") > pl.min_horizontal("open", "close") + 1e-9)
        | (pl.col("high") < pl.col("low"))
        | (pl.min_horizontal("open", "high", "low", "close") <= 0)
    )
    # --- one-bar spikes ("> 10 sigma from rolling median" that revert) ----------------
    med = pl.col("close").rolling_median(window_size=31, min_samples=10)
    absdiff = pl.col("close").diff().abs()
    scale = 1.4826 * absdiff.rolling_median(window_size=390, min_samples=30) * np.sqrt(15) + TICK
    s = s.with_columns(med.shift(1).alias("_med"), scale.shift(1).alias("_scale"))
    dev_hi = (pl.col("high") - pl.col("_med")) / pl.col("_scale")
    dev_lo = (pl.col("_med") - pl.col("low")) / pl.col("_scale")
    nxt_close_dev = ((pl.col("close").shift(-1) - pl.col("_med")) / pl.col("_scale")).abs()
    spike = ((dev_hi > 10) | (dev_lo > 10)) & (nxt_close_dev < 3)
    zero_vol = pl.col("volume") <= 0
    # --- maintenance break / outside session --------------------------------------
    boundary = (pl.col("ct_minute") >= 16 * 60) & (pl.col("ct_minute") < 17 * 60)

    s = s.with_columns(
        off_grid.alias("q_off_grid"),
        incoherent.alias("q_incoherent"),
        spike.fill_null(False).alias("q_spike"),
        zero_vol.alias("q_zero_vol"),
        boundary.alias("q_boundary"),
    )
    conflicting = set(rep.info.pop("_conflicting_dup_ts", []) or [])
    s = s.with_columns(pl.col("ts_utc").is_in(list(conflicting)).alias("q_dup_conflict")) if conflicting \
        else s.with_columns(pl.lit(False).alias("q_dup_conflict"))

    for col, label in (("q_off_grid", "off_tick_grid"), ("q_incoherent", "ohlc_incoherent"),
                       ("q_spike", "out_of_range_spike"), ("q_zero_vol", "zero_volume_bars"),
                       ("q_boundary", "session_boundary_violation"), ("q_dup_conflict", "conflicting_duplicates")):
        rep.counts[label] = int(s[col].sum())

    # exclusions: any hard failure anywhere in the session
    hard = s.filter(pl.col("q_off_grid") | pl.col("q_incoherent") | pl.col("q_spike")
                    | pl.col("q_boundary") | pl.col("q_dup_conflict"))
    for r in hard.select("session_date", "q_off_grid", "q_incoherent", "q_spike", "q_boundary",
                         "q_dup_conflict").iter_rows(named=True):
        for k, lab in (("q_off_grid", "off-grid price"), ("q_incoherent", "incoherent OHLC"),
                       ("q_spike", "bad-tick spike"), ("q_boundary", "bar in maintenance break"),
                       ("q_dup_conflict", "conflicting duplicate bars")):
            if r[k]:
                rep.exclude(r["session_date"], lab)

    # missing RTH bars and RTH zero-volume share, per session
    per = (
        s.filter(pl.col("in_rth")).group_by("session_date")
        .agg(pl.len().alias("n"), pl.col("q_zero_vol").sum().alias("zv"))
    )
    exp = sessions.select(
        "session_date",
        ((pl.col("rth_close_utc") - pl.col("rth_open_utc")).dt.total_minutes()).alias("expected"),
    )
    per = exp.join(per, on="session_date", how="left").with_columns(pl.col("n").fill_null(0), pl.col("zv").fill_null(0))
    missing_total = int((per["expected"] - per["n"]).clip(lower_bound=0).sum())
    rep.counts["missing_rth_bars"] = missing_total
    no_data = per.filter(pl.col("n") == 0)
    rep.counts["sessions_without_data"] = no_data.height
    for d in no_data["session_date"].to_list():
        rep.exclude(d, "no RTH data (holiday gap or missing file)")
    gappy = per.filter((pl.col("n") > 0) & ((pl.col("expected") - pl.col("n")) > 0.05 * pl.col("expected")))
    for d in gappy["session_date"].to_list():
        rep.exclude(d, ">5% of RTH bars missing")
    zv = per.filter((pl.col("n") > 0) & (pl.col("zv") > 0.01 * pl.col("n")))
    for d in zv["session_date"].to_list():
        rep.exclude(d, ">1% zero-volume RTH bars")

    _timezone_sanity(s, rep)

    s = s.with_columns(
        (pl.col("q_off_grid") | pl.col("q_incoherent") | pl.col("q_spike")
         | pl.col("q_boundary") | pl.col("q_dup_conflict")).alias("bad_bar")
    )
    return s.drop([c for c in s.columns if c.startswith("q_") or c.startswith("_")] + ["rth_open_utc", "rth_close_utc"])


def _timezone_sanity(s: pl.DataFrame, rep: QualityReport) -> None:
    """NQ volume spikes at the 08:30 CT cash open. If the biggest volume jump sits an
    hour away, the CSV timezone setting is almost certainly wrong."""
    by_min = s.group_by("ct_minute").agg(pl.col("volume").mean().alias("v")).sort("ct_minute")
    if by_min.height < 120:
        return
    v = dict(by_min.iter_rows())
    jumps = {m: v.get(m, 0) - v.get(m - 1, 0) for m in v if 6 * 60 <= m <= 11 * 60}
    if not jumps:
        return
    peak = max(jumps, key=jumps.get)
    rep.info["largest_volume_jump_ct"] = f"{peak // 60:02d}:{peak % 60:02d}"
    if abs(peak - 510) > 5:
        rep.warnings.append(
            f"largest volume jump is at {peak // 60:02d}:{peak % 60:02d} CT, not 08:30 CT. "
            "The CSV source_timezone or bar_label setting is probably wrong."
        )
