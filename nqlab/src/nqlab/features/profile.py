"""Volume profile: POC / VAH / VAL.

Built from trades at 0.25-point bins when trades data exists. With only 1-minute bars,
each bar's volume is spread uniformly across its high-low range and the profile is
flagged ``approximate`` (the ``vp_approximate`` column is True).

Profiles are ALWAYS built on unadjusted prices: profile levels are real resting-liquidity
prices and any adjustment moves them.

Tie-breaking is explicit:
* POC ties: ``mid`` (closest to the midpoint of the profile's traded range; then the
  lower price), ``low`` or ``high``.
* Value-area expansion: add the larger adjacent bin; on a tie add both.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import polars as pl

from nqlab.features.registry import feature

_TIE = {"mid": 0, "low": 1, "high": 2}


@nb.njit(cache=True)
def _value_area(hist, va_pct, tie):
    n = hist.shape[0]
    total = 0.0
    first, last = -1, -1
    for i in range(n):
        if hist[i] > 0:
            total += hist[i]
            if first < 0:
                first = i
            last = i
    if total <= 0:
        return -1, -1, -1
    mx = 0.0
    for i in range(first, last + 1):
        if hist[i] > mx:
            mx = hist[i]
    mid2 = first + last  # twice the midpoint, avoids fractions
    poc = -1
    best = 1 << 60
    for i in range(first, last + 1):
        if hist[i] == mx:
            if tie == 1:
                poc = i
                break
            elif tie == 2:
                poc = i
            else:
                d = abs(2 * i - mid2)
                if d < best:
                    best = d
                    poc = i
    lo, hi = poc, poc
    acc = hist[poc]
    target = va_pct * total
    while acc < target:
        up = hist[hi + 1] if hi + 1 <= last else -1.0
        dn = hist[lo - 1] if lo - 1 >= first else -1.0
        if up < 0 and dn < 0:
            break
        if up > dn:
            hi += 1
            acc += up
        elif dn > up:
            lo -= 1
            acc += dn
        else:
            hi += 1
            lo -= 1
            acc += up + dn
    return poc, hi, lo


@nb.njit(cache=True)
def _add_bar(hist, base, lo_bin, hi_bin, vol):
    k = hi_bin - lo_bin + 1
    share = vol / k
    for b in range(lo_bin, hi_bin + 1):
        hist[b - base] += share


@nb.njit(cache=True)
def _developing(lo_bins, hi_bins, vols, sess, include, va_pct, tie, out_poc, out_vah, out_val):
    n = lo_bins.shape[0]
    i = 0
    while i < n:
        j = i
        while j < n and sess[j] == sess[i]:
            j += 1
        base = 1 << 60
        top = -(1 << 60)
        for k in range(i, j):
            if include[k]:
                if lo_bins[k] < base:
                    base = lo_bins[k]
                if hi_bins[k] > top:
                    top = hi_bins[k]
        if top >= base:
            hist = np.zeros(top - base + 1)
            have = False
            for k in range(i, j):
                if include[k]:
                    _add_bar(hist, base, lo_bins[k], hi_bins[k], vols[k])
                    have = True
                if have:
                    p, h, l = _value_area(hist, va_pct, tie)
                    if p >= 0:
                        out_poc[k] = base + p
                        out_vah[k] = base + h
                        out_val[k] = base + l
        i = j


def _bins(md, bin_size: float):
    b = md.bars
    lo = np.floor(b["low"].to_numpy() / bin_size + 1e-9).astype(np.int64)
    hi = np.floor(b["high"].to_numpy() / bin_size + 1e-9).astype(np.int64)
    return lo, hi, b["volume"].to_numpy().astype(np.float64)


def _rth_mask(md) -> np.ndarray:
    j = md.bars.select("ts_utc", "session_date").join(
        md.sessions.select("session_date", "rth_open_utc", "rth_close_utc"), on="session_date", how="left")
    return ((j["ts_utc"] >= j["rth_open_utc"]) & (j["ts_utc"] < j["rth_close_utc"])).fill_null(False).to_numpy()


def _session_codes(md) -> np.ndarray:
    return md.bars["session_date"].rank("dense").cast(pl.Int64).to_numpy()


def _session_profiles(md, mask, bin_size, va_pct, tie, lookback_sessions):
    """Profile of the previous ``lookback_sessions`` sessions (bars in ``mask``), broadcast
    to every bar of the following session."""
    lo, hi, vol = _bins(md, bin_size)
    codes = _session_codes(md)
    uniq = np.unique(codes)
    hists = {}
    for c in uniq:
        m = (codes == c) & mask
        if not m.any():
            continue
        base, top = int(lo[m].min()), int(hi[m].max())
        h = np.zeros(top - base + 1)
        for a, b_, v in zip(lo[m], hi[m], vol[m]):
            _add_bar(h, base, int(a), int(b_), float(v))
        hists[int(c)] = (base, h)
    res = {}
    for idx, c in enumerate(uniq):
        prev = [int(x) for x in uniq[max(0, idx - lookback_sessions):idx] if int(x) in hists]
        if len(prev) < lookback_sessions or not prev:
            continue
        base = min(hists[p][0] for p in prev)
        top = max(hists[p][0] + len(hists[p][1]) - 1 for p in prev)
        comb = np.zeros(top - base + 1)
        for p in prev:
            b0, h = hists[p]
            comb[b0 - base:b0 - base + len(h)] += h
        p_, h_, l_ = _value_area(comb, va_pct, tie)
        res[int(c)] = ((base + p_) * bin_size, (base + h_) * bin_size, (base + l_) * bin_size)
    poc = np.full(len(codes), np.nan)
    vah = np.full(len(codes), np.nan)
    val = np.full(len(codes), np.nan)
    for c, (p, h, l) in res.items():
        m = codes == c
        poc[m], vah[m], val[m] = p, h, l
    return poc, vah, val


def _frame(poc, vah, val, approximate=True) -> pl.DataFrame:
    return pl.DataFrame({"vp_poc": poc, "vp_vah": vah, "vp_val": val,
                         "vp_approximate": np.full(len(poc), approximate)})


_DEF = {"bin_size": 0.25, "va_pct": 0.70, "poc_tie": "mid"}
_NOTE = "built from 1m bars (volume spread uniformly over each bar's range)"


@feature("vp_prior_rth", series="unadjusted", unadjusted_only=True, defaults=_DEF, approximate_note=_NOTE)
def vp_prior_rth(md, bin_size, va_pct, poc_tie):
    return _frame(*_session_profiles(md, _rth_mask(md), bin_size, va_pct, _TIE[poc_tie], 1))


@feature("vp_prior_globex", series="unadjusted", unadjusted_only=True, defaults=_DEF, approximate_note=_NOTE)
def vp_prior_globex(md, bin_size, va_pct, poc_tie):
    return _frame(*_session_profiles(md, np.ones(md.bars.height, bool), bin_size, va_pct, _TIE[poc_tie], 1))


@feature("vp_composite", series="unadjusted", unadjusted_only=True, defaults={**_DEF, "n_days": 5},
         approximate_note=_NOTE)
def vp_composite(md, bin_size, va_pct, poc_tie, n_days):
    return _frame(*_session_profiles(md, _rth_mask(md), bin_size, va_pct, _TIE[poc_tie], int(n_days)))


@feature("vp_developing", series="unadjusted", unadjusted_only=True, defaults={**_DEF, "rth_only": True},
         approximate_note=_NOTE)
def vp_developing(md, bin_size, va_pct, poc_tie, rth_only):
    lo, hi, vol = _bins(md, bin_size)
    include = _rth_mask(md) if rth_only else np.ones(md.bars.height, bool)
    n = md.bars.height
    poc = np.full(n, -(1 << 62), np.int64)
    vah = poc.copy()
    val = poc.copy()
    _developing(lo, hi, vol, _session_codes(md), include, float(va_pct), _TIE[poc_tie], poc, vah, val)
    to_px = lambda a: np.where(a == -(1 << 62), np.nan, a * bin_size)  # noqa: E731
    return _frame(to_px(poc), to_px(vah), to_px(val))
