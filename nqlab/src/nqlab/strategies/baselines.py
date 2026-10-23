"""Baseline strategies (spec 13.2). The operator's strategy must beat these. If it cannot
beat random entry, that is the finding."""

from __future__ import annotations

import numpy as np

from nqlab.features import compute
from nqlab.strategies.base import Strategy, at_minute, gather, session_rng, session_slices

RTH_OPEN = 8 * 60 + 30  # CT minutes


class ORB(Strategy):
    name = "orb"
    description = "Opening-range breakout: stop entries one tick beyond the first N minutes' high/low."
    param_space = {"or_minutes": [5, 15, 30], "target_r": [1.0, 1.5, 2.0], "entry_cutoff_min": [60, 120]}
    fixed = {"time_stop_ct": 10 * 60, "instrument": "NQ", "contracts": 1, "max_trades": 1}

    def signals(self, md, eb, p):
        from nqlab.engine.orders import Signals

        f = compute(md, "opening_range", minutes=int(p["or_minutes"]))
        hi, lo = gather(md, eb, f, "or_high"), gather(md, eb, f, "or_low")
        sig = Signals(eb.n)
        cutoff = RTH_OPEN + int(p["entry_cutoff_min"])
        live = ~np.isnan(hi) & (eb.ct_minute < cutoff) & (eb.ct_minute < p["time_stop_ct"])
        rng = hi - lo
        stop = rng + 2 * 0.25
        sig.place(live, +1, "stop", price=hi + 0.25, stop_dist=stop, target_dist=stop * p["target_r"])
        sig.place(live, -1, "stop", price=lo - 0.25, stop_dist=stop, target_dist=stop * p["target_r"])
        sig.cancel_l |= eb.ct_minute == cutoff
        sig.cancel_s |= eb.ct_minute == cutoff
        sig.exit_now |= at_minute(eb, int(p["time_stop_ct"]) - 1)
        return sig


class FirstHalfHourMomentum(Strategy):
    name = "fhh_momentum"
    description = "Sign of the 09:30-10:00 ET return, traded at market in the following window."
    param_space = {"hold_minutes": [30, 60, 90], "stop_mult": [0.5, 0.75, 1.0]}
    fixed = {"instrument": "NQ", "contracts": 1, "max_trades": 1}

    def signals(self, md, eb, p):
        from nqlab.engine.orders import Signals

        sig = Signals(eb.n)
        decide = at_minute(eb, RTH_OPEN + 29)       # 08:59 CT bar closes at 09:00 CT (10:00 ET)
        first_open = np.full(eb.n, np.nan)
        hi = np.full(eb.n, np.nan)
        lo = np.full(eb.n, np.nan)
        for _, a, b in session_slices(eb):
            cm = eb.ct_minute[a:b]
            j = np.nonzero(cm == RTH_OPEN)[0]
            k = np.nonzero(cm == RTH_OPEN + 29)[0]
            if not len(j) or not len(k):
                continue
            j0, i = a + j[0], a + k[0]
            first_open[i] = eb.o[j0]
            hi[i] = eb.h[j0:i + 1].max()
            lo[i] = eb.l[j0:i + 1].min()
        ret = eb.c - first_open
        rng = hi - lo
        ok = decide & ~np.isnan(ret) & (ret != 0)
        stop = np.maximum(rng * p["stop_mult"], 2.0)
        sig.place(ok & (ret > 0), +1, "market", stop_dist=stop)
        sig.place(ok & (ret < 0), -1, "market", stop_dist=stop)
        sig.exit_now |= at_minute(eb, RTH_OPEN + 30 + int(p["hold_minutes"]) - 1)
        return sig


class RandomEntry(Strategy):
    name = "random_entry"
    description = ("Random entry time in the operator's window, coin-flip direction, the operator's "
                   "bracket exits. Matched trade frequency via p_trade.")
    param_space = {"stop_ticks": [24, 40, 60], "target_r": [1.0, 2.0]}
    fixed = {"p_trade": 1.0, "window_start_ct": RTH_OPEN, "window_minutes": 60, "seed": 0,
             "instrument": "NQ", "contracts": 1, "max_trades": 1}

    def signals(self, md, eb, p):
        from nqlab.engine.orders import Signals

        sig = Signals(eb.n)
        side_arr = np.zeros(eb.n, np.int64)
        for s, a, b in session_slices(eb):
            rng = session_rng(int(p["seed"]), eb.session_dates[s])
            if rng.random() >= p["p_trade"]:
                continue
            minute = int(p["window_start_ct"]) + int(rng.integers(0, int(p["window_minutes"])))
            side = 1 if rng.random() < 0.5 else -1
            j = np.nonzero(eb.ct_minute[a:b] == minute - 1)[0]
            if len(j):
                side_arr[a + j[0]] = side
        sd = p["stop_ticks"] * 0.25
        sig.place(side_arr > 0, +1, "market", stop_dist=sd, target_dist=sd * p["target_r"])
        sig.place(side_arr < 0, -1, "market", stop_dist=sd, target_dist=sd * p["target_r"])
        return sig


class SessionHold(Strategy):
    name = "session_hold"
    description = "Buy the RTH open, sell the RTH close. No stop (the MLL is the stop)."
    param_space = {"direction": [1]}
    fixed = {"instrument": "NQ", "contracts": 1, "max_trades": 1}

    def signals(self, md, eb, p):
        from nqlab.engine.orders import Signals

        sig = Signals(eb.n)
        m = at_minute(eb, RTH_OPEN - 1)
        sig.place(m, int(p["direction"]), "market")
        sig.exit_now |= at_minute(eb, 15 * 60 - 1)
        return sig


class NullFixedTime(Strategy):
    """Calibration junk: fixed-time entry with coin-flip direction."""

    name = "null_fixed_time"
    description = "Null: coin-flip direction at a fixed time with a symmetric bracket."
    param_space = {"entry_minute": [RTH_OPEN + 5, RTH_OPEN + 15, RTH_OPEN + 30], "stop_ticks": [20, 40]}
    fixed = {"seed": 0, "target_r": 1.0, "instrument": "NQ", "contracts": 1, "max_trades": 1}

    def signals(self, md, eb, p):
        from nqlab.engine.orders import Signals

        sig = Signals(eb.n)
        side_arr = np.zeros(eb.n, np.int64)
        m = np.nonzero(at_minute(eb, int(p["entry_minute"]) - 1))[0]
        for i in m:
            rng = session_rng(int(p["seed"]), eb.session_dates[eb.sess[i]])
            side_arr[i] = 1 if rng.random() < 0.5 else -1
        sd = p["stop_ticks"] * 0.25
        sig.place(side_arr > 0, +1, "market", stop_dist=sd, target_dist=sd * p["target_r"])
        sig.place(side_arr < 0, -1, "market", stop_dist=sd, target_dist=sd * p["target_r"])
        return sig
