"""Other synthetic nulls: matched-volatility random walk and GJR-GARCH(1,1) with t
errors fitted after removing intraday seasonality."""

from __future__ import annotations

import numpy as np
import polars as pl

TICK = 0.25


def _scale_by_minute(sd: dict[int, float], minutes: np.ndarray) -> np.ndarray:
    lut = np.full(1440, float(np.median(list(sd.values()))))
    for k, v in sd.items():
        lut[k] = v
    return lut[minutes.astype(np.int64)]


def _seasonal_sd(md) -> dict[int, float]:
    b = md.bars.with_columns((pl.col("close").log() - pl.col("close").shift(1).log()).alias("_r"))
    b = b.filter(pl.col("session_date") == pl.col("session_date").shift(1))
    by = b.group_by("ct_minute").agg(pl.col("_r").std().alias("sd"))
    return {int(m): float(s) for m, s in by.iter_rows() if s is not None and s > 0}


def _rebuild(md, rets: np.ndarray, seed: int):
    """Close-to-close log returns -> bars, borrowing real intrabar shapes (permuted within
    time-of-day) so high/low ranges stay realistic."""
    b = md.bars
    o = b["open"].to_numpy().astype(float)
    c = b["close"].to_numpy().astype(float)
    sess = b["session_date"].to_physical().to_numpy()
    first = np.ones(len(o), bool)
    first[1:] = sess[1:] != sess[:-1]
    rng = np.random.default_rng(seed)
    wick_up = np.log(b["high"].to_numpy() / np.maximum(o, c))
    wick_dn = np.log(np.minimum(o, c) / b["low"].to_numpy())
    perm = rng.permutation(len(o))
    step = np.where(first, np.log(c), rets)
    seg = np.cumsum(first) - 1
    cs = np.cumsum(step)
    base = cs[np.nonzero(first)[0]] - step[first]
    logc = cs - base[seg]
    newc = np.exp(logc)
    newo = np.where(first, o, np.exp(np.roll(logc, 1)))
    hi = np.maximum(newo, newc) * np.exp(np.abs(wick_up[perm]))
    lo = np.minimum(newo, newc) * np.exp(-np.abs(wick_dn[perm]))
    r = lambda x: np.round(x / TICK) * TICK  # noqa: E731
    newo, hi, lo, newc = r(newo), r(hi), r(lo), r(newc)
    hi = np.maximum.reduce([hi, newo, newc])
    lo = np.minimum.reduce([lo, newo, newc])
    off = b["adj_offset"].to_numpy()
    cols = {"open": newo, "high": hi, "low": lo, "close": newc}
    upd = [pl.Series(k, x) for k, x in cols.items()] + [pl.Series(f"{k}_badj", x + off) for k, x in cols.items()] \
        + [pl.Series(f"{k}_radj", x) for k, x in cols.items()]
    return b.with_columns(upd)


def random_walk_matched(md, seed: int):
    """i.i.d. normal returns scaled to the real U-shaped intraday volatility curve. A
    sanity floor only: nothing should make money here."""
    sd = _seasonal_sd(md)
    m = md.bars["ct_minute"].to_numpy()
    scale = _scale_by_minute(sd, m)
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal(len(m)) * scale
    return md.with_bars(_rebuild(md, rets, seed + 1), f"{md.data_version}~rw{seed}")


def garch_synthetic(md, seed: int, fit_rows: int = 50000):
    """GJR-GARCH(1,1) with Student-t errors on deseasonalized 1m returns; simulate, then
    re-apply the intraday seasonality."""
    from arch import arch_model

    sd = _seasonal_sd(md)
    b = md.bars
    lc = np.log(b["close"].to_numpy())
    r = np.diff(lc, prepend=lc[0])
    m = b["ct_minute"].to_numpy()
    scale = _scale_by_minute(sd, m)
    z = r / scale
    fit = z[-fit_rows:] * 100
    res = arch_model(fit, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    sim = res.model.simulate(res.params, len(z), burn=500)
    zs = sim["data"].to_numpy() / 100
    zs = zs / zs.std() * np.std(fit / 100)
    return md.with_bars(_rebuild(md, zs * scale, seed + 7), f"{md.data_version}~garch{seed}")
