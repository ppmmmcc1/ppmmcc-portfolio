import datetime as dt

import duckdb
import pytest

from nqlab.data.market import MarketData
from nqlab.data.synthetic import random_walk_bars
from nqlab.errors import Refusal
from nqlab.results.report import family_verdict
from nqlab.selection.family import FamilyStore
from nqlab.selection.funnel import FAIL, INSUFFICIENT, PASS
from nqlab.selection.prereg import make_prereg


@pytest.fixture()
def losing(settings):
    raw = random_walk_bars(settings, dt.date(2023, 1, 3), dt.date(2023, 8, 31), seed=3, rth_only=True,
                           drift_points_per_bar=-0.05)
    settings.raw["validation"]["gates"].update({"min_sessions": 20, "min_trades": 20})
    return MarketData.from_raw(raw, settings, "losing")


def test_failed_family_stays_archived_even_if_db_is_edited(settings, losing):
    store = FamilyStore(settings)
    store.register(make_prereg(settings, "hold_v1", "session_hold", "Buying the open earns the session drift."))
    res = store.run_stage("hold_v1", losing)
    assert res.status == FAIL
    with pytest.raises(Refusal):
        store.promote("hold_v1")
    # the operator edits the database by hand to resurrect the family
    con = duckdb.connect(str(settings.path("results_db")))
    con.execute("UPDATE families SET status='active' WHERE family_id='hold_v1'")
    con.execute("DELETE FROM stage_results WHERE family_id='hold_v1'")
    con.close()
    with pytest.raises(Refusal, match="ARCHIVED"):
        store.run_stage("hold_v1", losing)
    with pytest.raises(Refusal):
        store.promote("hold_v1")


def test_promote_requires_ledger_pass(settings, losing):
    store = FamilyStore(settings)
    store.register(make_prereg(settings, "hold_v2", "session_hold", "Buying the open earns the session drift."))
    con = duckdb.connect(str(settings.path("results_db")))
    con.execute("INSERT INTO stage_results VALUES ('hold_v2', 1, now(), 'PASS', 'forged', '{}', 'x', NULL)")
    con.close()
    with pytest.raises(Refusal, match="ledger"):
        store.promote("hold_v2")


def test_family_verdict():
    rows = [{"stage": 1, "status": PASS, "summary": "a"}, {"stage": 2, "status": INSUFFICIENT, "summary": "b"}]
    assert family_verdict(rows)[0] == INSUFFICIENT
    assert family_verdict(rows + [{"stage": 3, "status": FAIL, "summary": "c"}])[0] == FAIL
    assert family_verdict([{"stage": s, "status": PASS, "summary": ""} for s in range(1, 4)])[0] == INSUFFICIENT
    assert family_verdict([{"stage": s, "status": PASS, "summary": ""} for s in range(1, 7)])[0] == PASS
    assert family_verdict([])[0] == INSUFFICIENT


def test_sized_out_trades_are_reported(small_md, session_settings):
    from nqlab.engine.backtest import run_backtest

    p = {**__import__("nqlab.strategies.registry", fromlist=["x"]).get_strategy("orb").defaults(),
         "risk_usd": 50.0}
    r = run_backtest(small_md, "orb", p, session_settings)
    assert r.meta["n_sized_out"] > 0          # $50 cannot buy one NQ at an opening-range stop
