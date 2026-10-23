"""Masters intrabar-preserving permutation (the PRIMARY null for MCPT).

Each bar is decomposed into its log gap from the prior close plus the intrabar tuple
(high-open, low-open, close-open) in logs. Gaps and intrabar tuples are permuted
SEPARATELY and WITHIN time-of-day buckets, so the U-shaped intraday volatility and the
distribution of bar shapes survive while any serial structure (the thing a strategy
would exploit) is destroyed. Volume travels with its intrabar tuple. The first bar of
each session is left untouched, so overnight levels and roll gaps are preserved.
"""

from __future__ import annotations

import numpy as np
import polars as pl

TICK = 0.25


def permute_bars(md, seed: int, bucket_minutes: int = 30):
    b = md.bars
    o = b["open"].to_numpy().astype(float)
    h = b["high"].to_numpy().astype(float)
    lo = b["low"].to_numpy().astype(float)
    c = b["close"].to_numpy().astype(float)
    v = b["volume"].to_numpy()
    sess = b["session_date"].to_physical().to_numpy()
    first = np.ones(len(o), bool)
    first[1:] = sess[1:] != sess[:-1]
    prev_c = np.roll(c, 1)
    gap = np.log(o / prev_c)
    rh, rl, rc = np.log(h / o), np.log(lo / o), np.log(c / o)
    bucket = (b["ct_minute"].to_numpy() // bucket_minutes).astype(np.int64)
    rng = np.random.default_rng(seed)
    ng, nh, nl, nc, nv = gap.copy(), rh.copy(), rl.copy(), rc.copy(), v.copy()
    movable = ~first
    for bk in np.unique(bucket[movable]):
        idx = np.nonzero(movable & (bucket == bk))[0]
        if len(idx) < 2:
            continue
        p1 = rng.permutation(idx)
        ng[idx] = gap[p1]
        p2 = rng.permutation(idx)
        nh[idx], nl[idx], nc[idx], nv[idx] = rh[p2], rl[p2], rc[p2], v[p2]
    # rebuild: log close is a within-session cumulative sum anchored on the real first bar
    step = np.where(first, 0.0, ng + nc)
    step[first] = np.log(c[first])
    seg = np.cumsum(first) - 1
    cs = np.cumsum(step)
    base = cs[np.nonzero(first)[0]] - step[first]
    logc = cs - base[seg]
    newc = np.exp(logc)
    newo = np.where(first, o, np.exp(np.roll(logc, 1) + ng))
    newh = np.where(first, h, newo * np.exp(nh))
    newl = np.where(first, lo, newo * np.exp(nl))
    newc = np.where(first, c, newc)
    r = lambda x: np.round(x / TICK) * TICK  # noqa: E731
    newo, newh, newl, newc = r(newo), r(newh), r(newl), r(newc)
    newh = np.maximum.reduce([newh, newo, newc])
    newl = np.minimum.reduce([newl, newo, newc])
    off = b["adj_offset"].to_numpy()
    ratio = (b["close"] / b["close_radj"]).to_numpy()
    cols = {"open": newo, "high": newh, "low": newl, "close": newc}
    upd = {k: pl.Series(k, x) for k, x in cols.items()}
    upd.update({f"{k}_badj": pl.Series(f"{k}_badj", x + off) for k, x in cols.items()})
    upd.update({f"{k}_radj": pl.Series(f"{k}_radj", x / ratio) for k, x in cols.items()})
    upd["volume"] = pl.Series("volume", nv)
    nb = b.with_columns(list(upd.values()))
    return md.with_bars(nb, f"{md.data_version}~perm{seed}")
