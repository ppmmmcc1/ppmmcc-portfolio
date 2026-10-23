"""Golden engine tests: hand-computed trade lists (spec section 6)."""

import math

import numpy as np
import pytest

from nqlab.engine.orders import EngineParams, Signals
from nqlab.engine.simulate import simulate
from tests.engine_helpers import make_bars

T = 0.25


def one(trades):
    assert trades.height == 1, trades
    return trades.row(0, named=True)


def test_no_same_bar_fill():
    """A market order decided at bar 0's close fills at bar 1's open, never at bar 0."""
    eb = make_bars([(100, 101, 99, 100.5), (102, 103, 101, 102.5), (102.5, 104, 102, 103)], slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False]), +1, "market", stop_dist=10.0)
    tr = simulate(eb, sig, EngineParams())
    r = one(tr)
    assert r["entry_bar"] == 1 and r["entry_price"] == 102.0
    # last bar of the session closes the position (session_end) at its close
    assert r["exit_reason"] == "session_end" and r["exit_price"] == 103.0


def test_limit_fills_tier1_not_tier2():
    """Buy limit at 100.00; next bar's low is exactly 100.00 (touch, no trade-through)."""
    eb = make_bars([(101, 101.5, 100.5, 101), (100.75, 101, 100.0, 100.5), (100.5, 101, 100.25, 101)], slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False]), +1, "limit", price=100.0, stop_dist=5.0)
    assert simulate(eb, sig, EngineParams(fill_tier=2)).height == 0
    r = one(simulate(eb, sig, EngineParams(fill_tier=1)))
    assert r["entry_bar"] == 1 and r["entry_price"] == 100.0 and r["entry_type"] == "limit_intrabar"


def test_limit_trade_through_fills_tier2():
    eb = make_bars([(101, 101.5, 100.5, 101), (100.75, 101, 99.75, 100.5), (100.5, 101, 100.25, 101)], slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False]), +1, "limit", price=100.0, stop_dist=5.0)
    r = one(simulate(eb, sig, EngineParams(fill_tier=2)))
    assert r["entry_price"] == 100.0


def test_limit_gap_fills_at_open():
    eb = make_bars([(101, 101.5, 100.5, 101), (99.0, 99.5, 98.75, 99.25), (99.25, 99.5, 99, 99)], slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False]), +1, "limit", price=100.0, stop_dist=5.0)
    r = one(simulate(eb, sig, EngineParams(fill_tier=2)))
    assert r["entry_price"] == 99.0 and r["entry_type"] == "at_open"


def test_bracket_both_legs_touched_stop_first():
    """Long from 100 (market at bar 1 open), stop 98, target 103. Bar 2 touches 97.5 and
    103.5: ambiguous -> stop assumed first, at 98 - 1 tick slippage."""
    eb = make_bars([(100, 100.5, 99.5, 100), (100, 100.5, 99.5, 100), (100, 103.5, 97.5, 101), (101, 102, 100, 101)], slip=1)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False, False]), +1, "market", stop_dist=2.0, target_dist=3.0)
    r = one(simulate(eb, sig, EngineParams(fill_tier=2)))
    assert r["entry_price"] == 100.0 + T          # market slippage 1 tick
    assert r["exit_reason"] == "stop" and r["ambiguous"]
    assert r["exit_price"] == pytest.approx(100.25 - 2.0 - T)
    assert r["exit_bar"] == 2


def test_target_needs_trade_through_under_tier2():
    eb = make_bars([(100, 100.5, 99.5, 100), (100, 100.5, 99.75, 100), (100, 103.25, 99.5, 103), (103, 103.5, 102.5, 103)], slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False, False]), +1, "market", stop_dist=2.0, target_dist=3.25)
    # entry 100.00, target 103.25: bar 2 high is exactly 103.25 -> tier 2 no fill; bar 3 high 103.5 fills
    r = one(simulate(eb, sig, EngineParams(fill_tier=2)))
    assert r["exit_bar"] == 3 and r["exit_price"] == 103.25 and r["exit_reason"] == "target"
    r1 = one(simulate(eb, sig, EngineParams(fill_tier=1)))
    assert r1["exit_bar"] == 2


def test_force_flatten_at_1510():
    """Position open at the 15:10 CT bar is liquidated at that bar's open minus slippage."""
    eb = make_bars([(100, 100.5, 99.5, 100), (100, 101, 99.75, 100.5), (101, 101.5, 100.5, 101), (101.5, 102, 101, 102)],
                   slip=2, force_flat_at=2, ct_start=908)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False, False]), +1, "market", stop_dist=10.0)
    # the 4th bar (after the deadline) is excluded by build_engine_bars in real runs; here
    # we also verify nothing trades after the flatten bar.
    sig.place(np.array([False, False, True, False]), +1, "market", stop_dist=10.0)
    tr = simulate(eb, sig, EngineParams(fill_tier=2, max_trades_per_day=5))
    r = one(tr)
    assert r["exit_reason"] == "flatten_1510" and r["exit_bar"] == 2
    assert r["exit_price"] == 101.0 - 2 * T


def test_trade_never_spans_maintenance_break():
    """Session A's last bar is 15:59 CT; session B opens 17:00 CT with a gap. A position
    cannot survive the break: it is closed at A's last close; B starts flat."""
    eb = make_bars([(100, 100.5, 99.5, 100), (100, 101, 99.75, 100.5), (100.5, 101, 100, 100.75),
                    (110, 111, 109, 110.5), (110.5, 111, 110, 111)], sess=[0, 0, 0, 1, 1], slip=1)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False, False, False]), +1, "market", stop_dist=50.0)
    r = one(simulate(eb, sig, EngineParams(fill_tier=2)))
    assert r["exit_bar"] == 2 and r["exit_reason"] == "session_end"
    assert r["exit_price"] == 100.75 - T
    assert r["pnl_points"] < 10   # did NOT capture the overnight gap


def test_stop_gap_through_fills_at_open():
    eb = make_bars([(100, 100.5, 99.5, 100), (100, 100.5, 99.75, 100), (95, 96, 94, 95.5)], slip=1)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False]), +1, "market", stop_dist=2.0)
    r = one(simulate(eb, sig, EngineParams(fill_tier=2)))
    assert r["exit_reason"] == "stop" and r["exit_price"] == 95.0 - T   # open, not the stop


def test_limit_entry_bar_target_not_allowed_stop_allowed():
    """Buy limit 100 fills intrabar on bar 1; bar 1 high (106) might be before the fill,
    so the 105 target cannot fill on bar 1. Bar 1 low 97.5 is below the 98 stop: price
    had to pass 100 to get there, so the stop IS hit on the entry bar."""
    eb = make_bars([(102, 102.5, 101.5, 102), (102, 106, 97.5, 99), (99, 100, 98.5, 99)], slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False]), +1, "limit", price=100.0, stop_dist=2.0, target_dist=5.0)
    r = one(simulate(eb, sig, EngineParams(fill_tier=2)))
    assert r["exit_bar"] == 1 and r["exit_reason"] == "stop" and r["exit_price"] == 98.0


def test_oco_breakout_both_triggered_same_bar_is_ambiguous_and_pessimistic():
    eb = make_bars([(100, 100.5, 99.5, 100), (100, 103, 97, 100), (100, 100.5, 99.5, 100)], slip=0)
    sig = Signals(eb.n)
    m = np.array([True, False, False])
    sig.place(m, +1, "stop", price=102.0, stop_dist=4.0)
    sig.place(m, -1, "stop", price=98.0, stop_dist=4.0)
    r = one(simulate(eb, sig, EngineParams(fill_tier=2)))
    assert r["ambiguous"]


def test_expiry_and_max_trades():
    eb = make_bars([(100, 100.5, 99.5, 100)] * 4 + [(100, 100.5, 98.0, 99)], slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False, False, False]), +1, "limit", price=99.0, stop_dist=2.0)
    assert simulate(eb, sig, EngineParams(expiry_bars=3)).height == 0
    assert simulate(eb, sig, EngineParams(expiry_bars=0)).height == 1


def test_breakeven_moves_next_bar_not_same_bar():
    eb = make_bars([(100, 100.5, 99.5, 100), (100, 100.25, 99.75, 100), (100, 104.5, 99.75, 104),
                    (104, 104.25, 99.5, 100), (100, 100.5, 99.5, 100)], slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False, False, False]), +1, "market", stop_dist=2.0)
    r = one(simulate(eb, sig, EngineParams(be_trigger_r=1.0)))
    # MFE reached 4.5 pts on bar 2; stop moves to 100 effective bar 3; bar 3 low 99.5 -> exit at 100
    assert r["exit_bar"] == 3 and r["exit_price"] == 100.0 and r["breakeven_moved"]


def test_mae_includes_bar_lows():
    eb = make_bars([(100, 100.5, 99.5, 100), (100, 100.5, 99.0, 100), (100, 101, 97.0, 100.5), (100.5, 101, 100, 100.75)], slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False, False]), +1, "market", stop_dist=5.0)
    r = one(simulate(eb, sig, EngineParams()))
    assert r["mae_points"] == pytest.approx(3.0)
