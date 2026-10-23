import datetime as dt
import json

import polars as pl
import pytest

from nqlab.data.holdout import HoldoutAccessError, holdout_access
from nqlab.data.ingest import ingest
from nqlab.data.market import MarketData
from nqlab.data.store import DataStore
from nqlab.data.synthetic import random_walk_bars


def _write_csv(settings, raw: pl.DataFrame, name="nq.csv", with_symbol=False):
    """Write synthetic bars as an ET-stamped CSV with typical free-data headers."""
    et = raw.with_columns(pl.col("ts_utc").dt.convert_time_zone("America/New_York")
                          .dt.strftime("%Y-%m-%d %H:%M:%S").alias("Datetime"))
    cols = ["Datetime", "open", "high", "low", "close", "volume"] + (["symbol"] if with_symbol else [])
    out = et.select(cols).rename({"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    p = settings.path("data_raw") / name
    out.write_csv(p)
    return p


@pytest.fixture()
def ingested(settings):
    raw = random_walk_bars(settings, dt.date(2024, 1, 2), dt.date(2024, 9, 30), seed=3, rth_only=True)
    # inject one bad tick (spike that reverts) on 2024-02-06 at 10:00 ET
    t = dt.datetime(2024, 2, 6, 15, 0, tzinfo=dt.timezone.utc)
    raw = raw.with_columns(pl.when(pl.col("ts_utc") == t).then(pl.col("high") + 500).otherwise(pl.col("high")).alias("high"))
    _write_csv(settings, raw)
    settings.raw["data"]["csv"]["columns"]["timestamp"] = ["Datetime"]
    return ingest(settings)


def test_ingest_end_to_end(settings, ingested):
    r = ingested
    assert r.n_sessions > 50 and r.n_holdout_sessions > 100
    assert r.holdout_start == dt.date(2024, 3, 28) or r.holdout_start.month == 3
    assert "2024-02-06" in r.report.excluded
    assert "bad-tick spike" in r.report.excluded["2024-02-06"]
    md = MarketData.from_store(settings)
    assert md.bars["session_date"].max() < r.holdout_start
    assert md.bars.schema["ts_utc"].time_zone == "UTC"
    excl = md.sessions.filter(pl.col("excluded"))["session_date"].to_list()
    assert dt.date(2024, 2, 6) in excl
    # timezone sanity: CSV written in ET and configured as ET -> volume jump at 08:30 CT
    assert r.report.info["largest_volume_jump_ct"] == "08:30"
    assert (settings.path("reports") / f"quality_{r.data_version}.md").exists()


def test_holdout_unreachable_outside_command(settings, ingested):
    store = DataStore(settings)
    hdir = settings.path("holdout")
    with pytest.raises(HoldoutAccessError):
        store.scan_front("NQ", holdout=True)
    with pytest.raises(HoldoutAccessError):
        open(next(iter((hdir / "front-1m" / "NQ").iterdir())) / "part.parquet", "rb")  # noqa: SIM115
    with pytest.raises(HoldoutAccessError):
        list(hdir.iterdir())
    with pytest.raises(HoldoutAccessError):
        MarketData.from_store(settings, holdout=True)
    with holdout_access("test of the lockbox itself"):
        md = MarketData.from_store(settings, holdout=True)
        assert md.bars["session_date"].min() >= ingested.holdout_start


def test_lockbox_boundary_never_moves(settings, ingested):
    lb = json.loads((settings.path("parquet").parent / "lockbox.json").read_text(encoding="utf-8"))
    r2 = ingest(settings)
    assert r2.holdout_start.isoformat() == lb["holdout_start"]


def test_only_holdout_command_grants_access():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "nqlab"
    users = [p.relative_to(src).as_posix() for p in src.rglob("*.py")
             if "holdout_access(" in p.read_text(encoding="utf-8")]
    assert sorted(users) == ["data/holdout.py", "selection/holdout_cmd.py"], users


def test_wrong_timezone_is_flagged(settings):
    raw = random_walk_bars(settings, dt.date(2024, 1, 2), dt.date(2024, 2, 28), seed=4)
    _write_csv(settings, raw)
    settings.raw["data"]["csv"]["columns"]["timestamp"] = ["Datetime"]
    settings.raw["data"]["csv"]["source_timezone"] = "America/Chicago"   # wrong: file is ET
    r = ingest(settings)
    assert any("timezone" in w for w in r.report.warnings)


def test_per_contract_volume_roll(settings):
    from nqlab.data.ingest import process

    raw = random_walk_bars(settings, dt.date(2024, 2, 1), dt.date(2024, 4, 30), seed=5, per_contract=True, rth_only=True)
    series, sessions, rolls, rep, method, *_ = process(raw, settings)
    assert method == "volume"
    assert rolls.height == 1 and rolls["from"][0] == "NQH24" and rolls["to"][0] == "NQM24"
    assert abs(rolls["gap"][0] - 60.0) < 1e-9
    # unadjusted trades the real contract; back-adjusted removes the 60pt carry jump
    r = rolls.row(0, named=True)
    before = series.filter(pl.col("session_date") < r["roll_session"]).tail(1)
    after = series.filter(pl.col("session_date") >= r["roll_session"]).head(1)
    assert after["close"][0] - before["close"][0] > 30
    assert abs(after["open_badj"][0] - before["close_badj"][0]) < 30


def test_cross_check_bars_vs_trades(small_md):
    from nqlab.data.quality import cross_check_trades

    b = small_md.bars.filter(pl.col("session_date") == small_md.sessions["session_date"][3])
    rows = []
    for r in b.iter_rows(named=True):
        for k, px in enumerate((r["open"], r["low"], r["high"], r["close"])):
            rows.append((r["ts_utc"] + dt.timedelta(seconds=5 + 10 * k), px, r["volume"] // 4 + (1 if k == 0 else 0) * (r["volume"] % 4)))
    tr = pl.DataFrame(rows, schema={"ts_utc": pl.Datetime("us", "UTC"), "price": pl.Float64, "size": pl.Int64}, orient="row")
    ok = cross_check_trades(b, tr)
    assert ok["bars_compared"] == b.height and ok["price_disagreements"] == 0 and ok["volume_disagreements"] == 0
    bad = tr.with_columns(pl.when(pl.int_range(pl.len()) == 41).then(pl.col("price") + 5).otherwise(pl.col("price")).alias("price"))
    assert cross_check_trades(b, bad)["price_disagreements"] == 1
