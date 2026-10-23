"""MANDATORY lookahead property test (spec section 5).

For random timestamps t: compute every registered feature on the full dataset and on
data truncated at t; every value at times <= t must be bit-identical. If this test does
not exist or does not pass, the platform is not trustworthy.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from nqlab.features import FEATURES, compute

# parameter variants so both alignments / stdev types / anchors get exercised
VARIANTS = {
    "htf_macd": [dict(tf=15, align="clock"), dict(tf=60, align="session"), dict(tf=60, align="clock")],
    "htf_bollinger": [dict(tf=5), dict(tf=15, stdev="sample", basis="ema")],
    "bollinger": [dict(), dict(stdev="sample", basis="ema", source="hlc3")],
    "session_vwap": [dict(anchor="rth"), dict(anchor="globex")],
    "vp_developing": [dict(), dict(rth_only=False, poc_tie="low")],
    "vp_composite": [dict(n_days=3)],
}


def _all(md):
    out = {}
    for name in FEATURES:
        for i, v in enumerate(VARIANTS.get(name, [dict()])):
            out[f"{name}#{i}"] = compute(md, name, **v)
    return out


@pytest.fixture(scope="module")
def full(small_md):
    return _all(small_md)


def _check(small_md, full, k):
    ts_close = small_md.bars["ts_close_utc"]
    t = ts_close[k]
    trunc = small_md.truncate(t)
    n = trunc.bars.height
    assert n == k + 1
    got = _all(trunc)
    for key, df in got.items():
        ref = full[key].head(n)
        for c in df.columns:
            a, b = df[c], ref[c]
            if a.dtype in (pl.Float64, pl.Float32):
                av, bv = a.to_numpy(), b.to_numpy()
                same = (av == bv) | (np.isnan(av) & np.isnan(bv))
                assert same.all(), f"{key}.{c} differs at t={t} (first idx {np.argmin(same)})"
            else:
                assert a.equals(b, check_names=False), f"{key}.{c} differs at t={t}"


def test_feature_registry_declares_series():
    from nqlab.features.registry import SERIES

    for name, spec in FEATURES.items():
        assert spec.series in SERIES or spec.series == "param", name
        if name.startswith("vp_"):
            assert spec.series == "unadjusted", "volume profile must use unadjusted prices"


def test_undeclared_series_rejected():
    from nqlab.features.registry import feature

    with pytest.raises(TypeError):
        feature("bad", series=None)
    with pytest.raises(ValueError):
        feature("bad_vp", series="back_adjusted", unadjusted_only=True)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture,
                                                                   HealthCheck.too_slow])
@given(frac=st.floats(min_value=0.02, max_value=0.999))
def test_truncation_bit_identical(small_md, full, frac):
    k = int(frac * (small_md.bars.height - 1))
    _check(small_md, full, k)


def test_truncation_at_htf_boundaries(small_md, full):
    """Edge cases the random draw may miss: the minute before / at / after a 15m and a
    session-aligned 60m boundary, the RTH open and the first bar of a session."""
    cm = small_md.bars["ct_minute"].to_numpy()
    idx = []
    for target in (509, 510, 511, 524, 525, 554, 555, 569, 570, 1020):
        hits = np.nonzero(cm == target)[0]
        if len(hits) > 3:
            idx.append(int(hits[3]))
    for k in idx:
        _check(small_md, full, k)


def test_htf_exposes_only_closed_bar(small_md):
    """At the 1m row closing 10:07 CT, the newest 15m bar is the one that closed at 10:00."""
    df = compute(small_md, "htf_macd", tf=15).with_columns(small_md.bars["ts_close_utc"], small_md.bars["ct_minute"])
    row = df.filter(pl.col("ct_minute") == 10 * 60 + 6).row(0, named=True)   # bar 10:06-10:07
    close_ct = row["m15_close_ts"].astimezone(__import__("zoneinfo").ZoneInfo("America/Chicago"))
    assert (close_ct.hour, close_ct.minute) == (10, 0)
    row = df.filter(pl.col("ct_minute") == 10 * 60 + 14).row(0, named=True)  # bar 10:14-10:15
    close_ct = row["m15_close_ts"].astimezone(__import__("zoneinfo").ZoneInfo("America/Chicago"))
    assert (close_ct.hour, close_ct.minute) == (10, 15)


def test_opening_range_hidden_until_complete(small_md):
    df = compute(small_md, "opening_range", minutes=15).with_columns(small_md.bars["ct_minute"])
    early = df.filter((pl.col("ct_minute") >= 510) & (pl.col("ct_minute") < 524))
    assert early["or_high"].null_count() == early.height
    late = df.filter(pl.col("ct_minute") == 524)  # bar 08:44-08:45 closes the window
    assert late["or_high"].null_count() == 0


def test_bollinger_stdev_types_differ(small_md):
    a = compute(small_md, "bollinger", stdev="population")["bb_upper"]
    b = compute(small_md, "bollinger", stdev="sample")["bb_upper"]
    assert (b.drop_nulls() > a.drop_nulls()).all()


def test_profile_from_trades_is_exact_and_causal(small_md):
    """build_from='trades': exact volume at price, approximate flag off, and truncation-safe."""
    import datetime as dt

    from nqlab.data.market import MarketData

    b = small_md.bars.filter(pl.col("session_date") <= small_md.sessions["session_date"][8])
    rows = []
    for r in b.iter_rows(named=True):
        for k, px in enumerate((r["open"], r["low"], r["high"], r["close"])):
            rows.append((r["ts_utc"] + dt.timedelta(seconds=10 + 10 * k), px, 3))
    trades = pl.DataFrame(rows, schema={"ts_utc": pl.Datetime("us", "UTC"), "price": pl.Float64, "size": pl.Int64},
                          orient="row")
    md = MarketData(b, small_md.sessions, "tr", meta={"trades": trades})
    f = compute(md, "vp_developing", build_from="trades")
    assert not f["vp_approximate"].any() and f["vp_poc"].drop_nans().len() > 0
    g = compute(md, "vp_prior_rth", build_from="trades")
    assert g["vp_poc"].drop_nans().len() > 0
    k = md.bars.height // 2
    t = md.truncate(md.bars["ts_close_utc"][k])
    ft = compute(t, "vp_developing", build_from="trades")
    a, c = ft["vp_poc"].to_numpy(), f["vp_poc"].head(k + 1).to_numpy()
    assert ((a == c) | (np.isnan(a) & np.isnan(c))).all()
    with pytest.raises(ValueError, match="trades"):
        compute(small_md, "vp_prior_rth", build_from="trades")
