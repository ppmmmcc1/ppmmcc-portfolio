"""Persist runs, trades, daily account state and metrics; log every run to the ledger."""

from __future__ import annotations

import datetime as dt
import math

import polars as pl

from nqlab.hashing import canonical_json, git_commit, stable_hash
from nqlab.results.db import connect
from nqlab.results.ledger import LedgerEntry, TrialLedger, trial_key
from nqlab.results.metrics import sharpe


def save_run(settings, result, metrics: dict, account_daily: pl.DataFrame | None = None, family_id: str | None = None,
             origin: str = "human", kind: str = "config_run", status: str = "completed") -> str:
    db = settings.path("results_db")
    TrialLedger(db).append([LedgerEntry(
        trial_key(family_id, result.strategy, result.params), kind, status, origin=origin, run_id=result.run_id,
        family_id=family_id, strategy=result.strategy, params=result.params, data_version=result.data_version,
        fill_tier=result.fill_tier, sharpe=sharpe(result.daily["pnl_usd"].to_numpy()), n_obs=result.daily.height)])
    with connect(db) as con:
        con.execute("DELETE FROM runs WHERE run_id=?", [result.run_id])
        con.execute("DELETE FROM trades WHERE run_id=?", [result.run_id])
        con.execute("DELETE FROM daily WHERE run_id=?", [result.run_id])
        con.execute("DELETE FROM metrics WHERE run_id=?", [result.run_id])
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            result.run_id, family_id, result.strategy, canonical_json(result.params), stable_hash(result.params),
            result.data_version, result.fill_tier, stable_hash(settings.section("costs")),
            stable_hash(settings.topstep_rules), git_commit(settings.root), dt.datetime.now(dt.timezone.utc),
            status, origin, kind])
        t = result.trades
        if t.height:
            rows = t.select(
                pl.lit(result.run_id).alias("run_id"), pl.int_range(pl.len()).cast(pl.Int32).alias("trade_no"), "session_date", "entry_ts", "exit_ts",
                "side", "contracts", "entry_price", "exit_price", "stop_dist", "pnl_points", "pnl_usd", "fees_usd",
                "r_multiple", "mae_points", "mfe_points", "exit_reason", "entry_type", "fill_tier", "ambiguous").to_arrow()
            con.register("_t", rows)
            con.execute("INSERT INTO trades SELECT * FROM _t")
            con.unregister("_t")
        d = result.daily
        if account_daily is not None and account_daily.height:
            d = d.join(account_daily.select("session_date", "balance", "mll_level", "rule_events", "consistency_ratio"),
                       on="session_date", how="left")
        else:
            d = d.with_columns(pl.lit(None, pl.Float64).alias("balance"), pl.lit(None, pl.Float64).alias("mll_level"),
                               pl.lit(None, pl.Utf8).alias("rule_events"), pl.lit(None, pl.Float64).alias("consistency_ratio"))
        rows = d.select(pl.lit(result.run_id).alias("run_id"), "session_date", "pnl_usd", "n_trades", "balance",
                        "mll_level", (pl.col("rule_events").fill_null("").str.contains("DLL")).alias("dll"),
                        "consistency_ratio", "rule_events").to_arrow()
        if rows.num_rows:
            con.register("_d", rows)
            con.execute("INSERT INTO daily SELECT * FROM _d")
            con.unregister("_d")
        mrows = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mrows.append((result.run_id, k, float(v) if math.isfinite(float(v)) else None, None))
            else:
                mrows.append((result.run_id, k, None, str(v)))
        if mrows:
            con.executemany("INSERT INTO metrics VALUES (?,?,?,?)", mrows)
    return result.run_id


def ledger_log(settings, result, family_id: str | None = None, kind: str = "config_run", note: str | None = None) -> None:
    """Ledger-only record of a real-data backtest (parity, post-mortem, ad-hoc). Same
    trial key as the config's family run, so re-running a known config never inflates N
    while a NEW config always counts."""
    TrialLedger(settings.path("results_db")).append([LedgerEntry(
        trial_key(family_id, result.strategy, result.params), kind, "completed", run_id=result.run_id,
        family_id=family_id, strategy=result.strategy, params=result.params, data_version=result.data_version,
        fill_tier=result.fill_tier, sharpe=sharpe(result.daily["pnl_usd"].to_numpy()), n_obs=result.daily.height,
        note=note)])


def ledger_error(settings, strategy: str, params: dict, exc: BaseException, family_id: str | None = None,
                 status: str = "errored") -> None:
    TrialLedger(settings.path("results_db")).append([LedgerEntry(
        trial_key(family_id, strategy, params), "config_run", status, family_id=family_id, strategy=strategy,
        params=params, note=f"{type(exc).__name__}: {exc}"[:500])])
