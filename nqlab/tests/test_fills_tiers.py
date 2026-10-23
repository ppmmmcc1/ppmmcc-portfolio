"""Fill tiers 3 (queue, trades data) and 4 (MBO replay) on synthetic ticks."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from nqlab.engine.fills import FillModelUnavailable, MBOReplayFill, QueueFill, TouchFill, TradeThroughFill, available_models
from nqlab.engine.orders import EngineParams, Signals
from tests.engine_helpers import make_bars

T0 = dt.datetime(2024, 6, 3, 13, 30, tzinfo=dt.timezone.utc)


def ticks(rows):
    """rows: (seconds_from_T0, price, size)."""
    return pl.DataFrame({"ts_utc": [T0 + dt.timedelta(seconds=s) for s, _, _ in rows],
                         "price": [float(p) for _, p, _ in rows], "size": [int(z) for _, _, z in rows]})


def limit_setup():
    eb = make_bars([(101, 101.5, 100.5, 101), (101, 101, 100.0, 100.5), (100.5, 101, 100.0, 100.5)], slip=0)
    sig = Signals(eb.n)
    sig.place(np.array([True, False, False]), +1, "limit", price=100.0, stop_dist=5.0)
    return eb, sig


def test_queue_fills_only_after_queue_ahead_trades():
    eb, sig = limit_setup()
    # history before the order joins: 30 lots typically trade per price -> queue ahead ~ 30*(1-0.3)=21
    hist = [(-600, 100.25, 30), (-500, 100.5, 30), (-400, 100.75, 30)]
    bar1 = [(60 + k, 100.0, 5) for k in range(4)]                  # only 20 lots at our price in bar 1
    bar2 = [(120 + k, 100.0, 5) for k in range(4)]                 # 20 more -> 40 > 21 ahead
    t = ticks(hist + bar1 + bar2)
    tr = QueueFill(haircut=0.3).run(eb, sig, EngineParams(), trades=t)
    assert tr.height == 1 and tr["entry_bar"][0] == 2               # tier 3 fills one bar later
    assert TouchFill().run(eb, sig, EngineParams()).height == 1     # tier 1 fills on bar 1 (touch)
    assert TradeThroughFill().run(eb, sig, EngineParams()).height == 0   # never traded through 100.00


def test_queue_trade_through_fills_immediately():
    eb, sig = limit_setup()
    t = ticks([(-30, 100.5, 100), (65, 99.75, 1)])
    tr = QueueFill().run(eb, sig, EngineParams(), trades=t)
    assert tr.height == 1 and tr["entry_bar"][0] == 1 and tr["entry_price"][0] == 100.0


def test_mbo_exact_queue_position():
    eb, sig = limit_setup()
    # two resting bids at 100.00 ahead of us (10 + 15 lots), placed before we join at bar 1
    mbo = pl.DataFrame({
        "ts_utc": [T0 + dt.timedelta(seconds=s) for s in (-100, -90, 61, 62, 63, 121)],
        "action": ["A", "A", "F", "C", "F", "F"], "side": ["B", "B", "B", "B", "B", "B"],
        "price": [100.0] * 6, "size": [10, 15, 10, 15, 0, 3], "order_id": [1, 2, 1, 2, 9, 9]})
    # trades at 100: bar 1 consumes order 1 (10 lots) while order 2 cancels; our turn comes in bar 2
    t = ticks([(61, 100.0, 10), (121, 100.0, 3)])
    tr = MBOReplayFill().run(eb, sig, EngineParams(), trades=t, mbo=mbo)
    assert tr.height == 1 and tr["entry_bar"][0] == 2


def test_tiers_require_data():
    eb, sig = limit_setup()
    with pytest.raises(FillModelUnavailable):
        QueueFill().run(eb, sig, EngineParams())
    assert [m.tier for m in available_models()] == [2, 1]
    assert [m.tier for m in available_models(True, True)] == [2, 1, 3, 4]
    assert available_models()[0].headline
