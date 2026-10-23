"""Stage 7: the single scripted look at the holdout. Consumption is recorded BEFORE any
holdout data is read, so a crash mid-way still counts as a look."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import numpy as np
import polars as pl

from nqlab.data.holdout import holdout_access
from nqlab.data.market import MarketData
from nqlab.engine.backtest import run_backtest
from nqlab.results.db import connect
from nqlab.results.ledger import LedgerEntry, trial_key
from nqlab.results.metrics import summarize
from nqlab.selection.family import FamilyError, FamilyStore
from nqlab.selection.funnel import FAIL, INSUFFICIENT, PASS, StageResult

WARMUP_SESSIONS = 30


def consume_holdout(settings, fid: str, confirm: bool = False) -> StageResult:
    store = FamilyStore(settings)
    f = store.family(fid)
    if f["status"] != "active" or int(f["current_stage"]) != 7:
        raise FamilyError(f"{fid} is not at stage 7 (status={f['status']}, stage={f['current_stage']}); "
                          "only a family that passed and was promoted through stage 6 may look at the holdout.")
    with connect(settings.path("results_db")) as con:
        prior = con.execute("SELECT family_id, ts_utc FROM holdout_log").fetchall()
    if any(p[0] == fid for p in prior):
        raise FamilyError(f"{fid} has already consumed the holdout. There is exactly one look.")
    if not confirm:
        raise FamilyError("this is the ONE look at the holdout and cannot be undone; re-run with --confirm")
    contaminated = [p[0] for p in prior]
    research = MarketData.from_store(settings)
    cfg = json.loads(f["selected_params_json"])
    # record the look first
    with connect(settings.path("results_db")) as con:
        con.execute("INSERT INTO holdout_log VALUES (?,?,?,?)",
                    [fid, dt.datetime.now(dt.timezone.utc), research.data_version, json.dumps({"status": "started"})])
    store.ledger.append([LedgerEntry(trial_key(fid, f["strategy"], cfg), "holdout", "reserved", family_id=fid,
                                     strategy=f["strategy"], params=cfg, note="holdout look started")])
    with holdout_access(f"nqlab holdout {fid}"):
        hold = MarketData.from_store(settings, holdout=True)
        # warm indicators up on the tail of research data; trade only holdout sessions
        warm_dates = research.sessions["session_date"].tail(WARMUP_SESSIONS).to_list()
        warm = research.bars.filter(pl.col("session_date").is_in(warm_dates))
        cols = [c for c in hold.bars.columns if c in warm.columns]
        bars = pl.concat([warm.select(cols), hold.bars.select(cols)], how="vertical_relaxed")
        md = MarketData(bars, hold.sessions, hold.data_version, hold.provenance, hold.warnings)
        r = run_backtest(md, f["strategy"], cfg, settings, fill_tier=2)
        r1 = run_backtest(md, f["strategy"], cfg, settings, fill_tier=1)
    is_r = run_backtest(research, f["strategy"], cfg, settings, fill_tier=2)
    m, m1, mis = summarize(r.trades, r.daily), summarize(r1.trades, r1.daily), summarize(is_r.trades, is_r.daily)
    n = r.daily.height
    is_daily = is_r.daily["pnl_usd"].to_numpy()
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(is_daily, n).mean() for _ in range(5000)]) if n and len(is_daily) else np.array([np.nan])
    p5 = float(np.quantile(boot, 0.05))
    mean_h = float(r.daily["pnl_usd"].mean()) if n else float("nan")
    d: dict[str, Any] = {"holdout": m, "holdout_tier1": m1, "in_sample": mis, "is_bootstrap_p5_mean": p5,
                         "n_sessions": n, "contaminated_by": contaminated, "params": cfg}
    if n < 40 or m["n_trades"] < 20:
        res = StageResult(7, INSUFFICIENT, f"Holdout has only {n} sessions / {m['n_trades']} trades.", d)
    elif mean_h > 0 and mean_h >= p5:
        res = StageResult(7, PASS, f"Holdout mean ${mean_h:.2f}/session is positive and consistent with the "
                          "in-sample estimate (low power: this is a sanity check, not proof).", d)
    else:
        res = StageResult(7, FAIL, f"Holdout mean ${mean_h:.2f}/session " + ("is negative." if mean_h <= 0 else
                          f"is below the 5th percentile (${p5:.2f}) of what the in-sample result predicts."), d)
    if contaminated:
        res.summary = f"CONTAMINATED (holdout already consumed by {', '.join(contaminated)}). " + res.summary
    store.record(fid, res, md.data_version)
    store.ledger.append([LedgerEntry(trial_key(fid, f["strategy"], cfg), "holdout", "completed", family_id=fid,
                                     strategy=f["strategy"], params=cfg, data_version=md.data_version,
                                     sharpe=m["sharpe_daily"], n_obs=n)])
    with connect(settings.path("results_db")) as con:
        con.execute("UPDATE holdout_log SET result_json=? WHERE family_id=?",
                    [json.dumps({"status": res.status, "summary": res.summary}), fid])
        con.execute("UPDATE families SET status='holdout_consumed' WHERE family_id=?", [fid])
    return res
