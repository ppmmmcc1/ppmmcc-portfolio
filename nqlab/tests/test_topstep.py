"""Topstep layer: numba kernel vs an independent pure-Python reference, plus rule cases."""

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings as hsettings, strategies as st

from nqlab.risk.topstep import (OUT_FAIL_MLL, OUT_PASS, OUT_UNRESOLVED, AccountRules, SizingPolicy, TradeTape,
                                bootstrap_combines, historical_combines, run_combine, zero_edge_pass_probability)


@pytest.fixture()
def rules(settings):
    return AccountRules.from_config(settings.topstep_rules, "NQ")


def tape_from_days(days, pv=20.0, fee=3.78):
    """days: list of lists of (pnl_pts, mae_pts, stop_pts)."""
    ptr = [0]
    pnl, mae, stop = [], [], []
    for d in days:
        for a, b, c in d:
            pnl.append(a); mae.append(b); stop.append(c)
        ptr.append(len(pnl))
    return TradeTape(np.array(ptr, np.int64), np.array(pnl, float), np.array(mae, float), np.array(stop, float),
                     fee, pv, 5.0, list(range(len(days))))


def ref_combine(days, r: AccountRules, n_contracts=1, pv=20.0, fee=3.78, liq=2.0, tv=5.0):
    bal, high, mll, best, blocked = r.start, r.start, r.start - r.mll_amount, 0.0, False
    for i, d in enumerate(days):
        b0 = bal
        failed = False
        for pnl, mae, _ in d:
            if bal - mae * pv * n_contracts - fee * n_contracts <= mll:
                bal = mll - liq * tv * n_contracts - fee * n_contracts
                failed = True
                break
            bal += pnl * pv * n_contracts - fee * n_contracts
        if failed:
            return OUT_FAIL_MLL, i + 1
        best = max(best, bal - b0)
        high = max(high, bal)
        mll = r.mll_lock_at if high >= r.mll_lock_trigger else max(mll, high - r.mll_amount)
        prof = bal - r.start
        if prof >= r.target:
            if best <= r.consistency_pct * prof:
                return OUT_PASS, i + 1
            blocked = True
    return OUT_UNRESOLVED, len(days)


def _run(tape, r, n=1):
    d = np.zeros(1)
    out, end, used, blocked, best = run_combine(tape.day_ptr, tape.pnl_pts, tape.mae_pts, tape.stop_pts, tape.fee_pc,
                                                tape.point_value, tape.tick_value, 2.0, 0, 10**6, *r.combine_tuple(),
                                                0, float(n), False, d, d, d, np.zeros(1, np.int64))
    return out, used


day_st = st.lists(st.tuples(st.floats(-40, 60), st.floats(0, 60), st.just(10.0)), max_size=3)


@hsettings(max_examples=300, deadline=None)
@given(days=st.lists(day_st, min_size=1, max_size=60))
def test_kernel_matches_reference(days, session_settings):
    r = AccountRules.from_config(session_settings.topstep_rules, "NQ")
    days = [[(p, max(m, -p if p < 0 else 0.0), s) for p, m, s in d] for d in days]  # MAE >= realized loss
    assert _run(tape_from_days(days), r) == ref_combine(days, r)


def test_mll_breach_on_unrealized(rules):
    # +20pt winner (+$400 - fees) then a trade whose MAE hits the MLL even though it closed green
    days = [[(20, 0, 10)], [(5, 130, 10)]]
    out, used = _run(tape_from_days(days), rules)
    assert out == OUT_FAIL_MLL and used == 2


def test_trailing_mll_locks(rules):
    # +$2,600 day: EOD high 52,600 >= 52,000 trigger -> MLL locks at 50,000
    t = tape_from_days([[(130.2, 0, 10)], [(-1, 0, 10)]])
    from nqlab.risk.topstep import combine_trace
    tr = combine_trace(t, rules, SizingPolicy.fixed(1))
    assert tr["mll_level"][1] == 50000.0


def test_consistency_blocks_then_passes(rules):
    # one +$3,100 day: target met but best day = 100% of profit -> blocked; then grind to pass
    days = [[(155.2, 0, 10)]] + [[(50, 0, 10)]] * 3
    out, used = _run(tape_from_days(days), rules)
    assert out == OUT_PASS and used > 1


def test_zero_edge_reference_is_40pct(rules):
    assert zero_edge_pass_probability(rules) == pytest.approx(0.4)


def test_zero_edge_walk_passes_at_most_40pct(rules):
    """A symmetric +/-$250 walk passes ~40% against a STATIC floor (gambler's ruin). The
    MLL trails end-of-day highs, which makes it strictly harder: expect ~30%."""
    rng = np.random.default_rng(0)
    days = [[(12.5 + 3.78 / 20 if rng.random() < 0.5 else -12.5 + 3.78 / 20, 0.0, 12.5)] for _ in range(4000)]
    days = [[(p, max(0.0, -p), s) for p, _, s in d] for d in days]
    tape = tape_from_days(days)
    stt = bootstrap_combines(tape, rules, SizingPolicy.fixed(1), n_paths=2000, horizon=400, xfa_horizon=10, mean_block=1.0)
    assert 0.25 < stt.p_pass < 0.42, stt.p_pass


def test_historical_unresolved_never_counts_as_pass(rules):
    tape = tape_from_days([[(1, 0, 10)]] * 30)
    stt = historical_combines(tape, rules, SizingPolicy.fixed(1), horizon=250)
    assert stt.p_pass == 0 and stt.p_unresolved == 1.0


def test_risk_sizing_from_buffer(rules):
    from nqlab.risk.topstep import _size
    # 10% of a $2,000 buffer = $200; 10pt stop on MNQ ($2/pt) = $20/contract -> 10 contracts
    assert _size(1, 0.10, 2000.0, 10.0, 2.0, 50) == 10
    assert _size(1, 0.10, 2000.0, 10.0, 20.0, 5) == 1
    assert _size(1, 0.01, 2000.0, 10.0, 20.0, 5) == 0   # cannot size -> skip, never round up
