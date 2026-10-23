"""The operator's strategy family (spec 13.1). Parameters are the FLATTENED spec
(``specs/bb_vp_macd.yaml``). Any REQUIRED_UNDECIDED value makes ``validate`` raise:
the platform refuses to run a half-specified strategy."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from nqlab.calendar.roll import cme_convention_front
from nqlab.engine.orders import EngineParams, Signals
from nqlab.engine.simulate import Sizing
from nqlab.features import compute
from nqlab.strategies.base import Strategy, gather, session_slices
from nqlab.strategies.spec import UNDECIDED, SpecIncomplete

TICK = 0.25


def _et_to_ct_minutes(hhmm: str) -> int:
    h, m = (int(x) for x in str(hhmm).split(":"))
    return (h - 1) * 60 + m


def _prev_distinct(x: np.ndarray) -> np.ndarray:
    """Value of the previous DISTINCT run. For a higher-timeframe series repeated on every
    1m row this is the previous HTF bar's value; for a 1m series it is the previous bar."""
    x = np.asarray(x, float)
    same = np.r_[False, (x[1:] == x[:-1]) | (np.isnan(x[1:]) & np.isnan(x[:-1]))]
    run = np.cumsum(~same) - 1
    starts = np.nonzero(~same)[0]
    vals = x[starts]
    prev_vals = np.r_[np.nan, vals[:-1]]
    return prev_vals[run]


def _direction(line, sig, hist, how: str) -> np.ndarray:
    """Direction on FULL-length arrays aligned to md.bars (continuous through the night),
    never on gathered engine bars (whose neighbours can be 17 hours apart)."""
    if how == "line_vs_signal":
        d = np.sign(line - sig)
    elif how == "hist_sign":
        d = np.sign(hist)
    elif how == "line_vs_zero":
        d = np.sign(line)
    elif how == "line_slope":
        d = np.sign(line - _prev_distinct(line))
    else:
        raise ValueError(f"unknown direction_definition {how!r}")
    return np.nan_to_num(d, nan=0.0)


def _crosses(line, sig, hist, how: str, min_mag: float) -> tuple[np.ndarray, np.ndarray]:
    """Bullish / bearish cross at each 1m row of the FULL history (decided on the row's
    close). For an HTF series the cross appears on the row where the new HTF bar does."""
    if how == "line_cross_signal":
        a = line - sig
    elif how == "hist_sign_change":
        a = hist
    elif how == "line_cross_zero":
        a = line
    else:
        raise ValueError(f"unknown cross_definition {how!r}")
    b = np.r_[np.nan, a[:-1]]
    with np.errstate(invalid="ignore"):
        mag = np.abs(hist) >= float(min_mag)
        up = (b <= 0) & (a > 0) & mag
        dn = (b >= 0) & (a < 0) & mag
    return up, dn


def _recent(mask: np.ndarray, window: int, sess: np.ndarray) -> np.ndarray:
    """True where ``mask`` fired within the last ``window`` bars of the same session."""
    out = np.zeros_like(mask)
    last = -10**9
    last_s = -1
    for i in range(len(mask)):
        if sess[i] != last_s:
            last, last_s = -10**9, sess[i]
        if mask[i]:
            last = i
        out[i] = i - last < window
    return out


class _BBVPMacd(Strategy):
    variant = "limit"
    param_space: dict[str, list[Any]] = {}

    def validate(self, p: dict[str, Any]) -> None:
        from nqlab.strategies.spec import REQUIRED_FIELDS

        missing = sorted(k for k, v in p.items() if v == UNDECIDED)
        missing += [f"{k} (omitted from the spec)" for k in REQUIRED_FIELDS["bb_vp_macd"]
                    if k not in p and k != "variant"]
        if missing:
            raise SpecIncomplete(missing)
        if p.get("variant") not in (None, self.variant):
            raise ValueError(f"spec variant {p.get('variant')!r} does not match strategy {self.name}")
        if str(p.get("orders.partials", "none")).lower() not in ("none", "false", "no"):
            raise ValueError("partial exits are not supported by the engine yet")
        if int(p.get("orders.max_concurrent_positions", 1)) != 1:
            raise ValueError("the engine supports exactly one concurrent position")


    def engine_params(self, p, fill_tier):
        be = bool(p["orders.breakeven.enabled"])
        exp = int(p["orders.expiry_minutes"])
        return EngineParams(fill_tier=fill_tier, expiry_bars=exp,
                            max_trades_per_day=int(p["orders.max_trades_per_day"]),
                            stop_after_win=bool(p["orders.stop_after_first_win"]),
                            stop_after_loss=bool(p["orders.stop_after_first_loss"]),
                            be_trigger_r=float(p["orders.breakeven.trigger_r"]) if be else float("nan"))

    def sizing(self, p):
        return Sizing(instrument=p["sizing.instrument"], risk_usd=float(p["sizing.risk_per_trade_usd"]))

    def topstep_policy(self, p):
        """In the Combine simulation the operator sizes from the MLL buffer when he
        declared a percentage; otherwise from his fixed dollar risk."""
        from nqlab.risk.topstep import SizingPolicy

        pct = p.get("sizing.risk_pct_of_mll_buffer")
        if pct not in (None, 0, UNDECIDED):
            return SizingPolicy.buffer_frac(float(pct))
        return SizingPolicy.risk_usd(float(p["sizing.risk_per_trade_usd"]))

    # ------------------------------------------------------------------------------------
    def _context(self, md, eb, p):
        dirs = {}
        for tf, key in ((15, "macd_15m"), (60, "macd_1h")):
            f = compute(md, "htf_macd", tf=tf, fast=int(p[f"context.{key}.fast"]), slow=int(p[f"context.{key}.slow"]),
                        signal=int(p[f"context.{key}.signal"]), source=p[f"context.{key}.source"],
                        align=p["context.hourly_alignment"] if tf == 60 else "clock", series="back_adjusted")
            pre = f"m{tf}_"
            full = _direction(f[pre + "macd_line"].to_numpy().astype(float), f[pre + "macd_signal"].to_numpy().astype(float),
                              f[pre + "macd_hist"].to_numpy().astype(float), p["context.direction_definition"])
            dirs[tf] = full[eb.idx]
        rule = p["context.agreement_rule"]
        if rule == "both":
            ctx = np.where(dirs[15] == dirs[60], dirs[15], 0)
        elif rule == "either":
            ctx = np.where((dirs[15] == 0) | (dirs[60] == 0) | (dirs[15] == dirs[60]), dirs[15] + dirs[60], 0)
            ctx = np.sign(ctx)
        elif rule == "1h_dominates":
            ctx = dirs[60]
        else:
            raise ValueError(f"unknown agreement_rule {rule!r}")
        return ctx

    def _bands(self, md, eb, p):
        tf = str(p["bollinger.timeframe"])
        kw = dict(length=int(p["bollinger.length"]), mult=float(p["bollinger.mult"]), basis=p["bollinger.basis"],
                  stdev=p["bollinger.stdev"], source=p["bollinger.source"], series="back_adjusted")
        if tf == "1m":
            f = compute(md, "bollinger", **kw)
            cols = ("bb_upper", "bb_lower", "bb_basis")
        elif tf == "5m":
            f = compute(md, "htf_bollinger", tf=5, **kw)
            cols = ("b5_bb_upper", "b5_bb_lower", "b5_bb_basis")
        else:
            raise ValueError("bollinger.timeframe must be 1m or 5m")
        # bands are computed on the back-adjusted series; convert to tradable prices
        off = md.bars["adj_offset"].to_numpy()
        return tuple((f[c].to_numpy().astype(float) - off)[eb.idx] for c in cols)

    def _levels(self, md, eb, p):
        src = p["volume_profile.source"]
        kw = dict(bin_size=float(p["volume_profile.bin_size"]), va_pct=float(p["volume_profile.value_area_pct"]),
                  poc_tie=p["volume_profile.poc_tie"])
        name = {"prior_rth": "vp_prior_rth", "prior_globex": "vp_prior_globex",
                "composite": "vp_composite", "developing": "vp_developing"}[src]
        if src == "composite":
            kw["n_days"] = int(p["volume_profile.composite_days"])
        kw["build_from"] = p.get("volume_profile.build_from", "bars")
        f = compute(md, name, **kw)
        return {"VAL": gather(md, eb, f, "vp_val"), "POC": gather(md, eb, f, "vp_poc"), "VAH": gather(md, eb, f, "vp_vah")}

    def _trigger(self, md, eb, p):
        kw = dict(fast=int(p["trigger.macd.fast"]), slow=int(p["trigger.macd.slow"]),
                  signal=int(p["trigger.macd.signal"]), source=p["trigger.macd.source"], series="back_adjusted")
        ups, dns = [], []
        for tf in p["trigger.timeframes"]:
            if tf == "1m":
                f = compute(md, "macd", **kw)
                pre = ""
            elif tf == "5m":
                f = compute(md, "htf_macd", tf=5, align="clock", **kw)
                pre = "m5_"
            else:
                raise ValueError("trigger.timeframes entries must be 1m or 5m")
            up, dn = _crosses(f[pre + "macd_line"].to_numpy().astype(float), f[pre + "macd_signal"].to_numpy().astype(float),
                              f[pre + "macd_hist"].to_numpy().astype(float), p["trigger.cross_definition"],
                              float(p["trigger.min_hist_magnitude"]))
            ups.append(up[eb.idx])
            dns.append(dn[eb.idx])
        if len(ups) == 1:
            return ups[0], dns[0]
        w = int(p["trigger.both_within_bars"])
        up = (ups[0] & _recent(ups[1], w, eb.sess)) | (ups[1] & _recent(ups[0], w, eb.sess))
        dn = (dns[0] & _recent(dns[1], w, eb.sess)) | (dns[1] & _recent(dns[0], w, eb.sess))
        return up, dn

    def _skip_mask(self, md, eb, p) -> np.ndarray:
        s = md.sessions
        skip = set()
        if p["orders.skip_days.fomc"]:
            skip |= set(s.filter(pl.col("is_fomc"))["session_date"].to_list())
        if p["orders.skip_days.release_0830"]:
            skip |= set(s.filter(pl.col("is_release_0830"))["session_date"].to_list())
        if p["orders.skip_days.half_days"]:
            skip |= set(s.filter(pl.col("early_close"))["session_date"].to_list())
        if p["orders.skip_days.roll_week"]:
            dates = s["session_date"].to_list()
            fr = cme_convention_front(dates, "NQ")["front"].to_list()
            for i in range(1, len(dates)):
                if fr[i] != fr[i - 1]:
                    skip |= set(dates[max(0, i - 4):i + 1])
        sd = np.array(eb.session_dates, dtype=object)[eb.sess]
        return np.array([d in skip for d in sd]) if skip else np.zeros(eb.n, bool)

    def _stop_target(self, p, side, price, levels, upper, lower, basis, atr):
        st, tg = p["orders.stop.type"], p["orders.target.type"]
        if st == "fixed_ticks":
            sd = np.full_like(price, float(p["orders.stop.value"]) * TICK)
        elif st == "atr_multiple":
            sd = atr * float(p["orders.stop.value"])
        elif st == "beyond_level":
            buf = float(p["orders.stop.buffer_ticks"]) * TICK
            stack = np.vstack([levels["VAL"], levels["POC"], levels["VAH"]])
            if side > 0:
                cand = np.where(stack < price - TICK, stack, -np.inf).max(axis=0)
                sd = price - (cand - buf)
            else:
                cand = np.where(stack > price + TICK, stack, np.inf).min(axis=0)
                sd = (cand + buf) - price
            sd = np.where(np.isfinite(sd), sd, np.nan)
        else:
            raise ValueError(f"unknown stop.type {st!r}")
        if tg == "fixed_r":
            td = sd * float(p["orders.target.value"])
        elif tg == "opposite_band":
            td = (upper - price) if side > 0 else (price - lower)
        elif tg == "basis":
            td = (basis - price) if side > 0 else (price - basis)
        elif tg == "poc":
            td = (levels["POC"] - price) if side > 0 else (price - levels["POC"])
        else:
            raise ValueError(f"unknown target.type {tg!r}")
        td = np.where(td > TICK, td, np.nan)      # a target on the wrong side is no target
        sd = np.where(sd >= TICK, sd, np.nan)
        return sd, td

    def signals(self, md, eb, p):
        self.validate(p)
        ctx = self._context(md, eb, p)
        upper, lower, basis = self._bands(md, eb, p)
        levels = self._levels(md, eb, p)
        up_x, dn_x = self._trigger(md, eb, p)
        atr = gather(md, eb, compute(md, "atr", length=14), "atr")
        w0, w1 = _et_to_ct_minutes(p["orders.window_start_et"]), _et_to_ct_minutes(p["orders.window_end_et"])
        in_win = (eb.ct_minute >= w0 - 1) & (eb.ct_minute < w1 - 1) & ~self._skip_mask(md, eb, p)
        x = float(p["volume_profile.align_ticks"]) * TICK
        sig = Signals(eb.n)
        for side, band, lvl_key, xs in ((+1, lower, "volume_profile.long_levels", up_x),
                                        (-1, upper, "volume_profile.short_levels", dn_x)):
            lv = [levels[k] for k in p[lvl_key]]
            if not lv:
                continue
            stack = np.vstack(lv)
            dist = np.abs(stack - band)
            near_i = np.nanargmin(np.where(np.isnan(dist), np.inf, dist), axis=0)
            near_lvl = stack[near_i, np.arange(eb.n)]
            aligned = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=0) <= x
            how = p["volume_profile.order_price"]
            if how == "band":
                price = band
            elif how == "level":
                price = near_lvl
            elif how == "better":
                price = np.minimum(band, near_lvl) if side > 0 else np.maximum(band, near_lvl)
            else:
                raise ValueError(f"unknown order_price {how!r}")
            price = np.round(price / TICK) * TICK
            ok = in_win & aligned & (ctx == side) & ~np.isnan(price)
            sd, td = self._stop_target(p, side, price, levels, upper, lower, basis, atr)
            ok &= ~np.isnan(sd)
            if self.variant == "limit":
                conf = _recent(xs, int(p["trigger.confirmation_window_bars"]), eb.sess)
                ok &= conf
                if not bool(p["bollinger.reprice_each_bar"]):
                    price, sd, td, ok = self._freeze(eb, ok, price, band, sd, td,
                                                     int(p["orders.cancel_if_band_moves_ticks"]))
                sig.place(ok, side, "limit", price=price, stop_dist=sd, target_dist=td)
                # withdraw the resting order as soon as its conditions lapse
                lapse = ~ok & in_win
                if side > 0:
                    sig.cancel_l |= lapse
                else:
                    sig.cancel_s |= lapse
            else:
                touched = (eb.l <= band) if side > 0 else (eb.h >= band)
                near_ext = _recent(touched & aligned, int(p["trigger.extreme_lookback_bars"]), eb.sess)
                ok = in_win & (ctx == side) & xs & near_ext & ~np.isnan(sd)
                sig.place(ok, side, "market", stop_dist=sd, target_dist=td)
        sig.exit_now |= eb.ct_minute == _et_to_ct_minutes(p["orders.time_stop_et"]) - 1
        end = eb.ct_minute == w1 - 1
        sig.cancel_l |= end
        sig.cancel_s |= end
        return sig

    @staticmethod
    def _freeze(eb, ok, price, band, sd, td, cancel_ticks):
        """Frozen orders: price/bracket fixed at the first signal bar of each episode; the
        order is withdrawn if the band moves more than ``cancel_ticks`` away."""
        fp, fsd, ftd, fok = price.copy(), sd.copy(), td.copy(), ok.copy()
        for _, a, b in session_slices(eb):
            cur = None
            for i in range(a, b):
                if not ok[i]:
                    cur = None
                    continue
                if cur is None:
                    cur = (price[i], sd[i], td[i], band[i])
                fp[i], fsd[i], ftd[i] = cur[0], cur[1], cur[2]
                if cancel_ticks > 0 and abs(band[i] - cur[3]) > cancel_ticks * TICK:
                    fok[i] = False
                    cur = None
        return fp, fsd, ftd, fok


class BBVPMacdLimit(_BBVPMacd):
    name = "bb_vp_macd_limit"
    variant = "limit"
    description = "Operator strategy, limit variant: resting limit at band/level, MACD cross as confirmation."


class BBVPMacdTrigger(_BBVPMacd):
    name = "bb_vp_macd_trigger"
    variant = "trigger"
    description = "Operator strategy, trigger variant: MACD cross triggers a market entry near a band/level extreme."


class RandomEntrySpecExits(_BBVPMacd):
    """Baseline (spec 13.2): random entry with the OPERATOR'S exit logic at matched trade
    frequency. Everything except the entry decision comes from his spec: window, skip
    days, stop/target definitions, breakeven, time stop, trade caps, sizing. If his
    strategy cannot beat this, its entries add nothing."""

    name = "random_entry_spec"
    variant = "any"
    description = "Random entries (coin-flip side) with the operator's spec exits, matched frequency."

    def validate(self, p):
        q = dict(p)
        q["variant"] = self.variant
        super().validate(q)
        if not 0 < float(p.get("rate", 0)) <= 5:
            raise ValueError("rate (expected entries per session, matched to the real strategy) must be in (0, 5]")

    def signals(self, md, eb, p):
        self.validate(p)
        need_bands = p["orders.target.type"] in ("opposite_band", "basis")
        upper, lower, basis = self._bands(md, eb, p) if need_bands else (np.full(eb.n, np.nan),) * 3
        levels = self._levels(md, eb, p) if (p["orders.stop.type"] == "beyond_level" or p["orders.target.type"] == "poc") \
            else {k: np.full(eb.n, np.nan) for k in ("VAL", "POC", "VAH")}
        atr = gather(md, eb, compute(md, "atr", length=14), "atr")
        w0, w1 = _et_to_ct_minutes(p["orders.window_start_et"]), _et_to_ct_minutes(p["orders.window_end_et"])
        in_win = (eb.ct_minute >= w0 - 1) & (eb.ct_minute < w1 - 1) & ~self._skip_mask(md, eb, p)
        side = np.zeros(eb.n, np.int64)
        from nqlab.strategies.base import session_rng

        for s_, a, b in session_slices(eb):
            rng = session_rng(int(p.get("seed", 0)), eb.session_dates[s_])
            idx = np.nonzero(in_win[a:b])[0]
            if not len(idx):
                continue
            prob = float(p["rate"]) / len(idx)
            draws = rng.random(len(idx))
            sides = np.where(rng.random(len(idx)) < 0.5, 1, -1)
            side[a + idx[draws < prob]] = sides[draws < prob]
        sig = Signals(eb.n)
        for sd_, band in ((1, lower), (-1, upper)):
            sdist, tdist = self._stop_target(p, sd_, eb.c.copy(), levels, upper, lower, basis, atr)
            m = (side == sd_) & ~np.isnan(sdist)
            sig.place(m, sd_, "market", stop_dist=sdist, target_dist=tdist)
        sig.exit_now |= eb.ct_minute == _et_to_ct_minutes(p["orders.time_stop_et"]) - 1
        return sig
