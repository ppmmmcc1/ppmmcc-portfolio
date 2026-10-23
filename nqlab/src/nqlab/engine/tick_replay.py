"""Tick replay for fill tiers 3 (queue model, trades data) and 4 (MBO replay).

Same order semantics as the bar kernel (signals decided at a bar's close act from the
next bar; one position; flat by the deadline; ambiguous cases resolved against the
trader), but fills are decided tick by tick inside each bar:

* tier 3: when a limit joins at price P, the queue ahead is ESTIMATED as the median volume
  traded at a price over the previous 30 minutes, minus a cancellation haircut. The order
  fills once cumulative volume traded at P exceeds that, or immediately if price trades
  through P.
* tier 4: the queue ahead is EXACT: resting MBO orders on our side at P that arrived
  before us. It shrinks as those orders fill or cancel (FIFO).

Slow (pure Python) by design: use it on the 09:25-10:45 ET window over a sample of ~200
sessions to calibrate tiers 2 and 3.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl

from nqlab.engine.bars import EngineBars
from nqlab.engine.kernel import LIMIT, MARKET, NONE, STOP, STOPLIMIT
from nqlab.engine.orders import EngineParams, Signals

TICK = 0.25


class TradesQueue:
    """Tier 3 queue estimator from trades only."""

    def __init__(self, trades: pl.DataFrame, haircut: float = 0.3, lookback_s: int = 1800):
        self.ts = trades["ts_utc"].dt.epoch("ns").to_numpy()
        self.px = trades["price"].to_numpy()
        self.sz = trades["size"].to_numpy()
        self.haircut = haircut
        self.lookback = lookback_s * 1_000_000_000

    def join(self, side: int, price: float, t_ns: int) -> dict[str, Any]:
        a = np.searchsorted(self.ts, t_ns - self.lookback)
        b = np.searchsorted(self.ts, t_ns)
        if b <= a:
            ahead = 0.0
        else:
            px, sz = self.px[a:b], self.sz[a:b]
            vols: dict[float, float] = defaultdict(float)
            for p_, s_ in zip(px, sz):
                vols[float(p_)] += float(s_)
            ahead = float(np.median(list(vols.values())))
        return {"side": side, "price": price, "ahead": ahead * (1 - self.haircut)}

    def on_trade(self, st: dict[str, Any], price: float, size: float, t_ns: int = 0) -> bool:
        side, p = st["side"], st["price"]
        if (side > 0 and price < p - 1e-9) or (side < 0 and price > p + 1e-9):
            return True                        # traded through: we were filled
        if abs(price - p) < 1e-9:
            st["ahead"] -= size
            return st["ahead"] < 0
        return False


class MBOQueue:
    """Tier 4: exact queue position from market-by-order events."""

    def __init__(self, mbo: pl.DataFrame):
        self.events = mbo.sort("ts_utc")
        self.ts = self.events["ts_utc"].dt.epoch("ns").to_numpy()
        self._book: dict[int, tuple[str, float, int, int]] = {}
        self._pos = 0

    def _advance(self, t_ns: int) -> None:
        ev = self.events
        while self._pos < len(self.ts) and self.ts[self._pos] <= t_ns:
            r = ev.row(self._pos, named=True)
            self._apply(r)
            self._pos += 1

    def _apply(self, r) -> None:
        a, oid = r["action"], int(r["order_id"] or 0)
        if a == "A":
            self._book[oid] = (r["side"], float(r["price"]), int(r["size"]), self._pos)
        elif a in ("C", "F"):
            if oid in self._book:
                s, p, q, k = self._book[oid]
                q -= int(r["size"])
                if q <= 0:
                    self._book.pop(oid, None)
                else:
                    self._book[oid] = (s, p, q, k)
        elif a == "R":
            self._book.clear()
        elif a == "M" and oid in self._book:
            s, p, q, k = self._book[oid]
            # a price change or size increase loses priority
            k2 = self._pos if (float(r["price"]) != p or int(r["size"]) > q) else k
            self._book[oid] = (s, float(r["price"]), int(r["size"]), k2)

    def join(self, side: int, price: float, t_ns: int) -> dict[str, Any]:
        self._advance(t_ns)
        bs = "B" if side > 0 else "A"
        ahead_ids = {oid for oid, (s, p, q, k) in self._book.items() if s == bs and abs(p - price) < 1e-9}
        return {"side": side, "price": price, "ids": ahead_ids, "extra_ahead": 0}

    def ahead(self, st) -> int:
        return sum(self._book[o][2] for o in st["ids"] if o in self._book)

    def on_trade(self, st: dict[str, Any], price: float, size: float, t_ns: int = 0) -> bool:
        side, p = st["side"], st["price"]
        self._advance(t_ns - 1)                # everything strictly before this print
        ahead_before = self.ahead(st)          # queue state BEFORE this print's own fills
        self._advance(t_ns)
        if (side > 0 and price < p - 1e-9) or (side < 0 and price > p + 1e-9):
            return True
        if abs(price - p) < 1e-9:
            return ahead_before <= 0 and size > 0
        return False


def replay(eb: EngineBars, sig: Signals, ep: EngineParams, ticks: pl.DataFrame, queue, tick: float = TICK,
           mbo_driven: bool = False) -> pl.DataFrame:
    """Tick-level simulation. ``ticks``: ts_utc, price, size (trades). For tier 4 the MBO
    queue object advances its own book as time passes."""
    tts = ticks["ts_utc"].dt.epoch("ns").to_numpy()
    tpx = ticks["price"].to_numpy().astype(float)
    tsz = ticks["size"].to_numpy().astype(float)
    bar_ns = eb.ts.astype("datetime64[ns]").astype(np.int64)
    trades: list[dict[str, Any]] = []
    pend: dict[int, dict[str, Any] | None] = {1: None, -1: None}
    pos, e_px, stop_px, tgt_px, e_idx, mae, mfe = 0, 0.0, np.nan, np.nan, 0, 0.0, 0.0
    tgt_q = None
    exit_flag, trades_today, done_today = False, 0, False

    def close(i, px, reason):
        nonlocal pos, exit_flag, done_today, tgt_q
        trades.append({"entry_bar": e_idx, "exit_bar": i, "side": pos, "entry_price": e_px, "exit_price": px,
                       "pnl_points": (px - e_px) * pos, "mae_points": mae, "mfe_points": mfe, "exit_reason": reason,
                       "session": int(eb.sess[e_idx])})
        pnl = (px - e_px) * pos
        if (ep.stop_after_win and pnl > 0) or (ep.stop_after_loss and pnl < 0):
            done_today = True
        pos, exit_flag, tgt_q = 0, False, None

    for i in range(eb.n):
        new = i == 0 or eb.sess[i] != eb.sess[i - 1]
        last = i == eb.n - 1 or eb.sess[i + 1] != eb.sess[i]
        if new:
            pend = {1: None, -1: None}
            trades_today, done_today, exit_flag = 0, False, False
        a = np.searchsorted(tts, bar_ns[i])
        b = np.searchsorted(tts, bar_ns[i] + 60_000_000_000)
        slip = float(eb.slip[i]) * tick
        if pos != 0 and (eb.force_flat[i] or exit_flag):
            px = (tpx[a] if b > a else eb.o[i]) - pos * slip
            close(i, px, "flatten_1510" if eb.force_flat[i] else "exit_signal")
        if eb.force_flat[i]:
            pend = {1: None, -1: None}
        for k in range(a, b):
            p, s, t = tpx[k], tsz[k], tts[k]
            if pos != 0:
                adv = (e_px - p) * pos
                mae = max(mae, adv)
                mfe = max(mfe, -adv)
                if stop_px == stop_px and ((pos > 0 and p <= stop_px) or (pos < 0 and p >= stop_px)):
                    close(i, p - pos * slip, "stop")
                    continue
                if tgt_q is not None and queue.on_trade(tgt_q, p, s, int(t)):
                    close(i, tgt_px, "target")
                    continue
            elif not eb.force_flat[i]:
                for side in (1, -1):
                    o = pend[side]
                    if o is None:
                        continue
                    filled, fpx = False, 0.0
                    if o["type"] == MARKET:
                        filled, fpx = True, p + side * slip
                    elif o["type"] in (STOP, STOPLIMIT) and ((side > 0 and p >= o["px"]) or (side < 0 and p <= o["px"])):
                        if o["type"] == STOP:
                            filled, fpx = True, p + side * slip
                        else:
                            o["type"], o["px"] = LIMIT, o["lim"]
                            o["q"] = queue.join(side, o["px"], int(t))
                    elif o["type"] == LIMIT:
                        if o.get("q") is None:      # submitted at the previous close = this bar's open
                            o["q"] = queue.join(side, o["px"], int(bar_ns[i]))
                        if queue.on_trade(o["q"], p, s, int(t)):
                            filled, fpx = True, o["px"]
                    if filled:
                        pos, e_px, e_idx, mae, mfe = side, fpx, i, 0.0, 0.0
                        stop_px = fpx - side * o["sd"] if o["sd"] == o["sd"] else np.nan
                        tgt_px = fpx + side * o["td"] if o["td"] == o["td"] else np.nan
                        tgt_q = queue.join(-side, tgt_px, int(t)) if tgt_px == tgt_px else None
                        trades_today += 1
                        pend = {1: None, -1: None}
                        break
        # expiry
        for side in (1, -1):
            o = pend[side]
            if o is not None and ep.expiry_bars > 0 and i - o["at"] >= ep.expiry_bars:
                pend[side] = None
        if last:
            if pos != 0:
                close(i, eb.c[i] - pos * slip, "session_end")
            pend = {1: None, -1: None}
            continue
        if eb.force_flat[i]:
            continue
        if pos != 0:
            if sig.exit_now[i]:
                exit_flag = True
            continue
        if sig.cancel_l[i]:
            pend[1] = None
        if sig.cancel_s[i]:
            pend[-1] = None
        if done_today or trades_today >= ep.max_trades_per_day or not eb.entry_ok[i]:
            continue
        for side, pre in ((1, "l"), (-1, "s")):
            t_ = int(getattr(sig, f"{pre}_type")[i])
            if t_ != NONE:
                pend[side] = {"type": t_, "px": float(getattr(sig, f"{pre}_px")[i]), "lim": float(getattr(sig, f"{pre}_lim")[i]),
                              "sd": float(getattr(sig, f"{pre}_stop")[i]), "td": float(getattr(sig, f"{pre}_tgt")[i]),
                              "at": i, "q": None}
    return pl.DataFrame(trades, schema={"entry_bar": pl.Int64, "exit_bar": pl.Int64, "side": pl.Int64,
                                        "entry_price": pl.Float64, "exit_price": pl.Float64, "pnl_points": pl.Float64,
                                        "mae_points": pl.Float64, "mfe_points": pl.Float64, "exit_reason": pl.Utf8,
                                        "session": pl.Int64})
