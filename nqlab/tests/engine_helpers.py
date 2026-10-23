import datetime as dt

import numpy as np

from nqlab.engine.bars import EngineBars


def make_bars(ohlc, start=dt.datetime(2024, 6, 3, 13, 30), sess=None, slip=1, force_flat_at=None, ct_start=510):
    """ohlc: list of (o,h,l,c). One session unless ``sess`` given."""
    a = np.array(ohlc, dtype=float)
    n = len(a)
    ts = np.array([np.datetime64(start) + np.timedelta64(i, "m") for i in range(n)])
    s = np.zeros(n, np.int64) if sess is None else np.asarray(sess, np.int64)
    ff = np.zeros(n, bool)
    if force_flat_at is not None:
        ff[force_flat_at] = True
    return EngineBars(idx=np.arange(n), ts=ts, o=a[:, 0], h=a[:, 1], l=a[:, 2], c=a[:, 3],
                      volume=np.ones(n, np.int64), sess=s,
                      session_dates=[dt.date(2024, 6, 3) + dt.timedelta(days=int(k)) for k in range(int(s.max()) + 1)],
                      ct_minute=ct_start + np.arange(n), slip=np.full(n, slip, np.int64), force_flat=ff,
                      entry_ok=np.ones(n, bool), excluded_sessions=[])
