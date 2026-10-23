import datetime as dt

import polars as pl

from nqlab.calendar.events import EventCalendar
from nqlab.calendar.roll import cme_convention_front, expiry, parse_contract, third_friday, volume_front
from nqlab.calendar.sessions import build_sessions, session_date_expr


def _row(sess, d):
    return sess.filter(pl.col("session_date") == d).row(0, named=True)


def test_rth_open_is_0830_ct_across_us_dst(settings):
    s = build_sessions(dt.date(2024, 3, 1), dt.date(2024, 3, 22), settings)
    winter = _row(s, dt.date(2024, 3, 8))
    summer = _row(s, dt.date(2024, 3, 11))
    assert winter["rth_open_utc"].hour == 14 and winter["rth_open_utc"].minute == 30
    assert summer["rth_open_utc"].hour == 13 and summer["rth_open_utc"].minute == 30
    # Topstep flatten 15:10 CT
    assert summer["flatten_utc"].hour == 20 and summer["flatten_utc"].minute == 10


def test_us_eu_dst_mismatch_weeks(settings):
    """US springs forward ~2-3 weeks before Europe and falls back a week after. In those
    weeks ET-CT is unchanged (both US) so RTH stays 08:30 CT / 09:30 ET, while the UTC
    instant has already moved. Globex open stays 17:00 CT."""
    s = build_sessions(dt.date(2024, 3, 11), dt.date(2024, 3, 29), settings)  # US DST, EU not yet
    for r in s.iter_rows(named=True):
        ct = r["rth_open_utc"].astimezone(dt.timezone(dt.timedelta(hours=-5)))
        assert (ct.hour, ct.minute) == (8, 30)
        g = r["globex_open_utc"].astimezone(dt.timezone(dt.timedelta(hours=-5)))
        assert g.hour == 17
    s2 = build_sessions(dt.date(2024, 10, 28), dt.date(2024, 11, 1), settings)  # EU back, US not yet
    for r in s2.iter_rows(named=True):
        assert (r["rth_open_utc"].hour, r["rth_open_utc"].minute) == (13, 30)


def test_holidays_and_early_close(settings):
    s = build_sessions(dt.date(2024, 11, 25), dt.date(2024, 12, 31), settings)
    dates = set(s["session_date"].to_list())
    assert dt.date(2024, 11, 28) not in dates or _row(s, dt.date(2024, 11, 28))["early_close"]
    fri = _row(s, dt.date(2024, 11, 29))
    assert fri["early_close"]
    # early close 12:00 CT -> flatten 5 min before, RTH ends at the close
    assert fri["flatten_utc"] < fri["close_utc"]
    assert fri["rth_close_utc"] == fri["close_utc"]
    assert dt.date(2024, 12, 25) not in dates


def test_session_date_assignment():
    df = pl.DataFrame({"ts_utc": [dt.datetime(2024, 3, 10, 22, 0, tzinfo=dt.timezone.utc),   # Sun 17:00 CDT
                                  dt.datetime(2024, 3, 11, 14, 0, tzinfo=dt.timezone.utc)]})
    out = df.select(session_date_expr()).to_series().to_list()
    assert out == [dt.date(2024, 3, 11), dt.date(2024, 3, 11)]


def test_contract_math():
    assert third_friday(2025, 12) == dt.date(2025, 12, 19)
    assert parse_contract("NQZ25") == ("NQ", 2025, 12)
    assert parse_contract("MNQH6", ref_year=2025) == ("MNQ", 2026, 3)
    assert expiry("NQH24") == dt.date(2024, 3, 15)


def test_cme_convention_roll_thursday_8_days_before():
    days = [dt.date(2024, 3, 6), dt.date(2024, 3, 7), dt.date(2024, 3, 8)]
    f = dict(cme_convention_front(days, "NQ").iter_rows())
    assert f[dt.date(2024, 3, 6)] == "NQH24"
    assert f[dt.date(2024, 3, 7)] == "NQM24"   # 2024-03-15 - 8 days = Thu 03-07


def test_volume_roll_is_causal():
    dv = pl.DataFrame({
        "session_date": [dt.date(2024, 3, d) for d in (4, 4, 5, 5, 6, 6)],
        "symbol": ["NQH24", "NQM24"] * 3,
        "volume": [100, 50, 100, 150, 50, 200],
    })
    f = dict(volume_front(dv, "NQ").iter_rows())
    assert f[dt.date(2024, 3, 4)] == "NQH24"
    assert f[dt.date(2024, 3, 5)] == "NQH24"   # M24 exceeded on the 5th: switch applies from the 6th
    assert f[dt.date(2024, 3, 6)] == "NQM24"


def test_events_flags(settings):
    ev = EventCalendar.load(settings)
    assert ev.flags(dt.date(2024, 9, 18))["is_fomc"]
    f = ev.flags(dt.date(2024, 9, 6))  # first Friday -> NFP by rule
    assert f["is_release_0830"] and f["events_approximate"]
