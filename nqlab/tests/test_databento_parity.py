import datetime as dt

import pandas as pd
import polars as pl
import pytest

from nqlab.calendar.roll import canonical_contract
from nqlab.data.databento_source import CostRefused, DatabentoSource
from nqlab.parity import add_label, compare_fills, compare_labels, import_topstepx, labels_path, LABEL_COLS


class FakeStore:
    def __init__(self, df):
        self.df = df

    def to_df(self, **kw):
        return self.df


class FakeClient:
    def __init__(self, cost):
        self.cost = cost
        self.calls = []
        outer = self

        class M:
            def get_cost(self, **kw):
                outer.calls.append(("cost", kw))
                return outer.cost

        class T:
            def get_range(self, **kw):
                outer.calls.append(("get", kw))
                idx = pd.DatetimeIndex(pd.to_datetime(["2015-03-02 14:30", "2015-03-02 14:30", "2015-03-02 14:30"], utc=True), name="ts_event")
                return FakeStore(pd.DataFrame({"open": [4200.0, 4210.0, 10.0], "high": [4201.0, 4211.0, 10.0],
                                               "low": [4199.0, 4209.0, 10.0], "close": [4200.5, 4210.5, 10.0],
                                               "volume": [100, 20, 5], "symbol": ["NQH5", "NQM5", "NQH5-NQM5"]}, index=idx))

        self.metadata, self.timeseries = M(), T()


def test_cost_is_estimated_and_printed_before_download(settings, capsys):
    c = FakeClient(cost=1.25)
    src = DatabentoSource(settings, client=c)
    df = src.fetch_bars(start=dt.date(2015, 3, 2), end=dt.date(2015, 3, 2))
    assert [k for k, _ in c.calls] == ["cost", "get"]
    assert "estimated cost $1.25" in capsys.readouterr().out
    assert sorted(df["symbol"].to_list()) == ["NQH15", "NQM15"]      # spread dropped, years resolved


def test_expensive_request_refused(settings):
    c = FakeClient(cost=999.0)
    with pytest.raises(CostRefused):
        DatabentoSource(settings, client=c).fetch_bars(start=dt.date(2015, 3, 2), end=dt.date(2015, 3, 2))
    assert [k for k, _ in c.calls] == ["cost"]                          # nothing downloaded


def test_canonical_contract():
    assert canonical_contract("NQZ5", 2015) == "NQZ15"
    assert canonical_contract("NQH0", 2019) == "NQH20"
    assert canonical_contract("NQZ5-NQH6", 2015) is None


def _model_trades():
    t0 = dt.datetime(2024, 3, 5, 14, 47, tzinfo=dt.timezone.utc)   # 09:47 ET
    return pl.DataFrame({"session_date": [dt.date(2024, 3, 5), dt.date(2024, 3, 6)],
                         "entry_ts": [t0, t0 + dt.timedelta(days=1, minutes=30)], "side": [1, -1],
                         "entry_price": [18000.0, 18100.0], "pnl_usd": [100.0, -50.0], "contracts": [1, 1]},
                        schema_overrides={"session_date": pl.Date})


def test_parity_labels(settings):
    add_label(settings, dt.date(2024, 3, 5), "09:45", "long")
    add_label(settings, dt.date(2024, 3, 6), "10:30", "long")       # coded is short -> divergence
    labels = pl.read_csv(labels_path(settings), schema=LABEL_COLS)
    r = compare_labels(labels, _model_trades())
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["decision_required"]
    assert r["human_not_coded"][0]["date"] == "2024-03-06"


def test_topstepx_import_and_compare(tmp_path):
    p = tmp_path / "trades.csv"
    pl.DataFrame({"Id": [1], "ContractName": ["NQH4"], "EnteredAt": ["2024-03-05 09:48:10"],
                  "ExitedAt": ["2024-03-05 10:05:00"], "EntryPrice": [18001.0], "ExitPrice": [18006.0],
                  "Fees": [3.78], "PnL": [96.22], "Size": [1], "Type": ["Long"]}).write_csv(p)
    live = import_topstepx(p)
    assert live["side"][0] == 1 and live["session_date"][0] == dt.date(2024, 3, 5)
    r = compare_fills(live, _model_trades())
    assert r["paired"] == 1 and r["entry_slippage_pts_mean"] == 1.0
    assert r["model_fill_rate_realized"] == 1.0
