"""CME equity-index session calendar.

Conventions
-----------
* A *session* is identified by its trade date D. Globex for D runs from 17:00 CT on the
  prior calendar day to 16:00 CT on D, with the 16:00-17:00 CT maintenance break.
* RTH is 08:30-15:00 CT (09:30-16:00 ET). Early-close days end RTH at the early close.
* The Topstep flatten deadline is 15:10 CT, or ``early_close_flatten_minutes_before``
  minutes before an early close, whichever is earlier.
* All instants are tz-aware UTC; all wall-clock logic is America/Chicago.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import polars as pl

from nqlab.calendar.events import EventCalendar
from nqlab.settings import Settings, load_yaml

CT = ZoneInfo("America/Chicago")
ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc


def _hm(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


def ct_instant(day: dt.date, t: dt.time) -> dt.datetime:
    """Wall-clock CT time on ``day`` -> UTC instant (DST-correct)."""
    return dt.datetime.combine(day, t, tzinfo=CT).astimezone(UTC)


@lru_cache(maxsize=32)
def _mcal_schedule(start: dt.date, end: dt.date) -> pd.DataFrame:
    cal = mcal.get_calendar("CME_Equity")
    return cal.schedule(start_date=start.isoformat(), end_date=end.isoformat())


def build_sessions(start: dt.date, end: dt.date, settings: Settings) -> pl.DataFrame:
    """One row per CME equity session in [start, end].

    Columns: session_date, globex_open_utc, close_utc, rth_open_utc, rth_close_utc,
    flatten_utc, early_close, is_fomc, is_release_0830, is_release_1000, release_kinds,
    events_approximate.
    """
    cfg = settings.section("calendar")
    rules = settings.topstep_rules
    flatten_t = _hm(rules["combine"]["flatten_deadline_ct"])
    rth_open_t, rth_close_t = _hm(cfg["rth_open"]), _hm(cfg["rth_close"])
    early_buf = dt.timedelta(minutes=int(cfg.get("early_close_flatten_minutes_before", 5)))

    overrides_path = settings.path("calendar_overrides")
    overrides = load_yaml(overrides_path) if overrides_path.exists() else {}
    closed = {dt.date.fromisoformat(str(d)) for d in overrides.get("closed", []) or []}
    early_over = {
        dt.date.fromisoformat(str(k)): _hm(str(v))
        for k, v in (overrides.get("early_close", {}) or {}).items()
    }

    events = EventCalendar.load(settings)
    sched = _mcal_schedule(start, end)
    rows = []
    for idx, r in sched.iterrows():
        d = idx.date()
        if d in closed:
            continue
        gopen = r["market_open"].to_pydatetime().astimezone(UTC)
        close = r["market_close"].to_pydatetime().astimezone(UTC)
        if d in early_over:
            close = ct_instant(d, early_over[d])
        close_ct = close.astimezone(CT).time()
        early = close_ct < dt.time(16, 0)
        rth_open = ct_instant(d, rth_open_t)
        rth_close = ct_instant(d, rth_close_t)
        flatten = ct_instant(d, flatten_t)
        if early:
            rth_close = min(rth_close, close)
            flatten = min(flatten, close - early_buf)
        ev = events.flags(d)
        rows.append({
            "session_date": d,
            "globex_open_utc": gopen,
            "close_utc": close,
            "rth_open_utc": rth_open,
            "rth_close_utc": rth_close,
            "flatten_utc": flatten,
            "early_close": early,
            **ev,
        })
    schema = {
        "session_date": pl.Date,
        "globex_open_utc": pl.Datetime("us", "UTC"),
        "close_utc": pl.Datetime("us", "UTC"),
        "rth_open_utc": pl.Datetime("us", "UTC"),
        "rth_close_utc": pl.Datetime("us", "UTC"),
        "flatten_utc": pl.Datetime("us", "UTC"),
        "early_close": pl.Boolean,
        "is_fomc": pl.Boolean,
        "is_release_0830": pl.Boolean,
        "is_release_1000": pl.Boolean,
        "release_kinds": pl.Utf8,
        "events_approximate": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)


def session_date_expr(ts_col: str = "ts_utc") -> pl.Expr:
    """Trade date of a bar that *opens* at ``ts_col``: CT date, rolled forward one day
    for bars at or after 17:00 CT (they belong to the next day's Globex session)."""
    ct = pl.col(ts_col).dt.convert_time_zone("America/Chicago")
    return (
        pl.when(ct.dt.hour() >= 17)
        .then(ct.dt.date() + pl.duration(days=1))
        .otherwise(ct.dt.date())
    )


def ct_minute_expr(ts_col: str = "ts_utc") -> pl.Expr:
    """Minutes since CT midnight for the bar's open time."""
    ct = pl.col(ts_col).dt.convert_time_zone("America/Chicago")
    return (ct.dt.hour().cast(pl.Int32) * 60 + ct.dt.minute().cast(pl.Int32))
