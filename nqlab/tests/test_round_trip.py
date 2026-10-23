"""Round trip with a KNOWN answer (spec 14): CSV -> ingest -> features -> engine -> metrics.

Every RTH session rises exactly 10 points from the 08:30 CT open to 15:00 CT in a
deterministic staircase. Buying the open (market order, 3 ticks of first-minute slippage)
and selling at the open of the 15:00 CT bar (after the cash close: 2 ticks of overnight
slippage) must earn 10 - 0.75 - 0.50 = 8.75 points = $175 gross per NQ, minus $3.78 =
$171.22, every research session (the last 6 months are locked in the holdout)."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from nqlab.calendar.sessions import build_sessions
from nqlab.data.ingest import ingest
from nqlab.data.market import MarketData
from nqlab.engine.backtest import run_backtest
from nqlab.results.metrics import summarize


def staircase_csv(settings, start, end):
    sess = build_sessions(start, end, settings).filter(~pl.col("early_close"))
    rows = []
    price = 15000.0
    for r in sess.iter_rows(named=True):
        t = r["rth_open_utc"] - dt.timedelta(minutes=30)
        end_t = r["rth_close_utc"] + dt.timedelta(minutes=30)
        n_rth = int((r["rth_close_utc"] - r["rth_open_utc"]).total_seconds() // 60)
        k = 0
        while t < end_t:
            if r["rth_open_utc"] <= t < r["rth_close_utc"]:
                o = price + round(10.0 * k / n_rth * 4) / 4
                c = price + round(10.0 * (k + 1) / n_rth * 4) / 4
                k += 1
            elif t < r["rth_open_utc"]:
                o = c = price
            else:
                o = c = price + 10.0
            rows.append((t, o, max(o, c), min(o, c), c, 100))
            t += dt.timedelta(minutes=1)
        price += 10.0
    df = pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume"], orient="row")
    df = df.with_columns(pl.col("ts").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
                         .dt.strftime("%Y-%m-%d %H:%M:%S").alias("timestamp"))
    df.select("timestamp", "open", "high", "low", "close", "volume").write_csv(settings.path("data_raw") / "stairs.csv")
    return sess.height


def test_known_answer(settings):
    staircase_csv(settings, dt.date(2024, 1, 2), dt.date(2024, 12, 31))
    r = ingest(settings)
    md = MarketData.from_store(settings)
    res = run_backtest(md, "session_hold", {"direction": 1, "instrument": "NQ", "contracts": 1, "max_trades": 1}, settings)
    m = summarize(res.trades, res.daily)
    traded = res.daily.filter(pl.col("n_trades") > 0)
    assert m["n_trades"] == traded.height > 50
    assert np.allclose(res.trades["pnl_points"].to_numpy(), 8.75)
    assert np.allclose(res.trades["pnl_usd"].to_numpy(), 171.22)
    assert m["net_usd"] == pytest.approx(171.22 * m["n_trades"])
    assert m["win_rate"] == 1.0 and m["max_consec_losses"] == 0
