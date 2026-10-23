"""DuckDB connection handling.

DuckDB allows one writing process per database file. Worker processes never open the
results database; they return results to the parent, which writes. Short-lived
connections plus a retry loop keep the CLI and the overnight runner from colliding.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

SCHEMA = [
    """CREATE SEQUENCE IF NOT EXISTS ledger_seq START 1""",
    """CREATE TABLE IF NOT EXISTS trial_ledger (
        seq BIGINT PRIMARY KEY,
        ts_utc TIMESTAMPTZ NOT NULL,
        trial_key VARCHAR NOT NULL,
        run_id VARCHAR,
        family_id VARCHAR,
        strategy VARCHAR,
        params_json VARCHAR,
        kind VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        origin VARCHAR NOT NULL,
        data_version VARCHAR,
        fill_tier INTEGER,
        sharpe DOUBLE,
        n_obs INTEGER,
        note VARCHAR,
        prev_hash VARCHAR NOT NULL,
        row_hash VARCHAR NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        family_id VARCHAR,
        strategy VARCHAR,
        params_json VARCHAR,
        spec_hash VARCHAR,
        data_version VARCHAR,
        fill_tier INTEGER,
        cost_model VARCHAR,
        rules_model VARCHAR,
        git_commit VARCHAR,
        ts_utc TIMESTAMPTZ,
        status VARCHAR,
        origin VARCHAR,
        kind VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS trades (
        run_id VARCHAR, trade_no INTEGER, session_date DATE,
        entry_ts TIMESTAMPTZ, exit_ts TIMESTAMPTZ, side INTEGER, contracts INTEGER,
        entry_price DOUBLE, exit_price DOUBLE, stop_dist DOUBLE,
        pnl_points DOUBLE, pnl_usd DOUBLE, fees_usd DOUBLE, r_multiple DOUBLE,
        mae_points DOUBLE, mfe_points DOUBLE, exit_reason VARCHAR, entry_type VARCHAR,
        fill_tier INTEGER, ambiguous BOOLEAN
    )""",
    """CREATE TABLE IF NOT EXISTS daily (
        run_id VARCHAR, session_date DATE, pnl_usd DOUBLE, n_trades INTEGER,
        balance DOUBLE, mll_level DOUBLE, dll_hit BOOLEAN, consistency_ratio DOUBLE,
        rule_events VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS metrics (
        run_id VARCHAR, key VARCHAR, value DOUBLE, text VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS families (
        family_id VARCHAR PRIMARY KEY,
        strategy VARCHAR,
        prereg_path VARCHAR,
        prereg_hash VARCHAR,
        status VARCHAR,
        current_stage INTEGER,
        created_ts TIMESTAMPTZ,
        selected_params_json VARCHAR,
        origin VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS stage_results (
        family_id VARCHAR, stage INTEGER, ts_utc TIMESTAMPTZ, status VARCHAR,
        summary VARCHAR, details_json VARCHAR, data_version VARCHAR, report_path VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS holdout_log (
        family_id VARCHAR, ts_utc TIMESTAMPTZ, data_version VARCHAR, result_json VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS excluded_sessions (
        data_version VARCHAR, session_date DATE, reason VARCHAR
    )""",
]


@contextmanager
def connect(db_path: Path, read_only: bool = False, retries: int = 80) -> Iterator[duckdb.DuckDBPyConnection]:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    delay = 0.05
    last: Exception | None = None
    for _ in range(retries):
        try:
            con = duckdb.connect(str(db_path), read_only=read_only and db_path.exists())
            break
        except duckdb.IOException as exc:  # file locked by another process
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 2.0)   # ~2.5 minutes in total before giving up
    else:
        raise RuntimeError(f"could not open {db_path}: {last}")
    try:
        if not read_only:
            for stmt in SCHEMA:
                con.execute(stmt)
        yield con
    finally:
        con.close()
