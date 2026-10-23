"""Targeted tests for paths the main suites do not reach (XFA payouts, DLL, tick-replay
exits, registry guards)."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from nqlab.engine.orders import EngineParams, Signals
from nqlab.features import compute
from nqlab.features.registry import col, to_unadjusted
from nqlab.risk.topstep import (AccountRules, SizingPolicy, TradeTape, bootstrap_combines, optimal_risk_search,
                                run_combine, run_xfa)
from tests.engine_helpers import make_bars
from tests.test_topstep import tape_from_days


@pytest.fixture()
def rules(settings):
    return AccountRules.from_config(settings.topstep_rules, "NQ")


def _xfa(tape, rules, policy=SizingPolicy.fixed(1)):
    return run_xfa(tape.day_ptr, tape.pnl_pts, tape.mae_pts, tape.stop_pts, tape.fee_pc, tape.point_value,
                   tape.tick_value, 2.0, 0, 10**6, *rules.xfa_tuple(), rules.max_contracts, policy.mode, policy.param)


def test_xfa_standard_payouts(rules):
    # +$300/day winners: five winning days >= $150 -> payout of 50% of balance
    tape = tape_from_days([[(15.2, 0, 10)]] * 12)
    n, paid, breached, used, first = _xfa(tape, rules)
    assert n == 2 and not breached and first == 5
    assert paid == pytest.approx((0.5 * 5 * 300.22) * 0.9 + 0.5 * (5 * 300.22 + 0.5 * 5 * 300.22) * 0.9, rel=1e-3)


def test_xfa_breach_after_payout_resets_mll(rules):
    days = [[(15.2, 0, 10)]] * 5 + [[(-40, 40, 50)]] * 3
    n, paid, breached, used, first = _xfa(tape_from_days(days), rules)
    assert n == 1 and breached          # MLL reset to $0 after the first payout


def test_xfa_consistency_path(settings):
    cfg = settings.topstep_rules
    cfg["xfa"]["path"] = "consistency"
    r = AccountRules.from_config(cfg, "NQ")
    n, *_ = _xfa(tape_from_days([[(15.2, 0, 10)]] * 6), r)
    assert n >= 1


def test_dll_stops_the_day(settings):
    cfg = settings.topstep_rules
    cfg["combine"]["daily_loss_limit"]["enabled"] = True
    r = AccountRules.from_config(cfg, "NQ")
    tape = tape_from_days([[(-30, 30, 50), (-30, 30, 50), (10, 0, 10)]])
    d = np.zeros(5)
    ev = np.zeros(5, np.int64)
    out = run_combine(tape.day_ptr, tape.pnl_pts, tape.mae_pts, tape.stop_pts, tape.fee_pc, tape.point_value,
                      tape.tick_value, 2.0, 0, 10, *r.combine_tuple(), 0, 1.0, True, d, d.copy(), d.copy(), ev)
    assert ev[0] == 2                     # DLL event, third trade never taken


def test_tape_from_backtest_and_risk_search(small_md, session_settings, rules):
    from nqlab.engine.backtest import run_backtest

    r = run_backtest(small_md, "null_fixed_time", {**__import__("nqlab.strategies.registry", fromlist=["x"])
                                                   .get_strategy("null_fixed_time").defaults()}, session_settings)
    tape = TradeTape.from_backtest(r.trades, r.daily, 20.0, 5.0, 3.78)
    assert tape.n_days == r.daily.height and len(tape.pnl_pts) == r.trades.height
    rs = optimal_risk_search(tape, rules, fracs=np.array([0.05, 0.2]), n_paths=50, horizon=60)
    assert rs.height == 2 and set(rs.columns) >= {"p_first_payout", "expected_paid_usd"}
    st = bootstrap_combines(tape, rules, SizingPolicy.risk_usd(500), n_paths=50, horizon=60, xfa_horizon=10)
    assert st.n_paths == 50
    assert SizingPolicy.risk_usd(500).label().startswith("$")


def test_registry_guards(small_md):
    with pytest.raises(ValueError):
        compute(small_md, "bollinger", series="weird")
    with pytest.raises(ValueError):
        col("median", "unadjusted")
    with pytest.raises(ValueError):
        compute(small_md, "bollinger", stdev="both")
    with pytest.raises(ValueError):
        compute(small_md, "session_vwap", anchor="noon")
    df = small_md.bars.select(col("ohlc4", "ratio_adjusted").alias("a"),
                              to_unadjusted(pl.col("close_radj"), "ratio_adjusted").alias("b"),
                              to_unadjusted(pl.col("close_badj"), "back_adjusted").alias("c"))
    assert np.allclose(df["b"].to_numpy(), small_md.bars["close"].to_numpy())
    assert np.allclose(df["c"].to_numpy(), small_md.bars["close"].to_numpy())


T0 = dt.datetime(2024, 6, 3, 13, 30, tzinfo=dt.timezone.utc)


def _ticks(rows):
    return pl.DataFrame({"ts_utc": [T0 + dt.timedelta(seconds=s) for s, _, _ in rows],
                         "price": [float(p) for _, p, _ in rows], "size": [int(z) for _, _, z in rows]})


def test_tick_replay_exits():
    from nqlab.engine.fills import QueueFill

    eb = make_bars([(100, 100, 100, 100)] * 6, slip=1, force_flat_at=5)
    sig = Signals(eb.n)
    sig.place(np.array([True] + [False] * 5), +1, "market", stop_dist=2.0, target_dist=3.0)
    # market fill at 100 (+1 tick), target 103.25 trades through at bar 2
    t = _ticks([(60, 100.0, 1), (130, 103.5, 1)])
    tr = QueueFill().run(eb, sig, EngineParams(), trades=t)
    assert tr["exit_reason"].to_list() == ["target"]
    # stop path
    t = _ticks([(60, 100.0, 1), (130, 97.75, 1)])
    assert QueueFill().run(eb, sig, EngineParams(), trades=t)["exit_reason"].to_list() == ["stop"]
    # exit signal at bar 2 close -> exit at bar 3's first tick
    sig.exit_now[2] = True
    t = _ticks([(60, 100.0, 1), (190, 101.0, 1)])
    tr = QueueFill().run(eb, sig, EngineParams(), trades=t)
    assert tr["exit_reason"].to_list() == ["exit_signal"] and tr["exit_bar"][0] == 3
    # no exits at all -> forced flat at the deadline bar
    sig.exit_now[:] = False
    t = _ticks([(60, 100.0, 1), (310, 100.5, 1)])
    assert QueueFill().run(eb, sig, EngineParams(), trades=t)["exit_reason"].to_list() == ["flatten_1510"]


def test_tick_replay_stop_entry_and_expiry():
    from nqlab.engine.fills import QueueFill

    eb = make_bars([(100, 100, 100, 100)] * 5, slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False, False, False]), +1, "stop", price=101.0, stop_dist=2.0)
    t = _ticks([(70, 101.0, 1)])
    tr = QueueFill().run(eb, sig, EngineParams(), trades=t)
    assert tr["entry_price"][0] == 101.0 and tr["exit_reason"][0] == "session_end"
    t = _ticks([(250, 101.0, 1)])
    assert QueueFill().run(eb, sig, EngineParams(expiry_bars=2), trades=t).height == 0
