import copy

import numpy as np
import pytest
import yaml

from nqlab.engine.backtest import run_backtest
from nqlab.strategies.spec import SpecIncomplete, expand, flatten, grid_dimensions, load_spec, undecided

FILLED = {
    "variant": "limit",
    "context": {"direction_definition": "line_vs_signal", "agreement_rule": "both", "hourly_alignment": "session",
                "macd_15m": {"fast": 12, "slow": 26, "signal": 9, "source": "close"},
                "macd_1h": {"fast": 12, "slow": 26, "signal": 9, "source": "close"}},
    "bollinger": {"timeframe": "1m", "length": 20, "mult": 2.0, "basis": "sma", "stdev": "population",
                  "source": "close", "reprice_each_bar": True},
    "volume_profile": {"source": "prior_rth", "composite_days": None, "value_area_pct": 0.7, "bin_size": 0.25,
                       "poc_tie": "mid", "build_from": "bars", "align_ticks": 40, "long_levels": ["VAL", "POC"],
                       "short_levels": ["VAH", "POC"], "order_price": "better"},
    "trigger": {"timeframes": ["1m"], "both_within_bars": 3, "cross_definition": "line_cross_signal",
                "min_hist_magnitude": 0, "confirmation_window_bars": 10, "extreme_lookback_bars": 5,
                "macd": {"fast": 12, "slow": 26, "signal": 9, "source": "close"}},
    "orders": {"window_start_et": "09:30", "window_end_et": "10:30", "expiry_minutes": 0,
               "cancel_if_band_moves_ticks": 0, "max_concurrent_positions": 1, "max_trades_per_day": 2,
               "stop_after_first_win": True, "stop_after_first_loss": False,
               "stop": {"type": "fixed_ticks", "value": 40, "buffer_ticks": 4},
               "target": {"type": "fixed_r", "value": 1.5}, "partials": "none",
               "breakeven": {"enabled": False, "trigger_r": 1.0}, "time_stop_et": "11:00",
               "skip_days": {"fomc": True, "release_0830": False, "half_days": True, "roll_week": False}},
    "sizing": {"instrument": "MNQ", "risk_per_trade_usd": 200, "risk_pct_of_mll_buffer": 0.1, "daily_stop_usd": 400},
}


def params(**over):
    d = copy.deepcopy(FILLED)
    flat = flatten(d)
    flat.update(over)
    return flat


def test_scaffold_is_fully_undecided(settings):
    spec = load_spec(settings.path("specs") / "bb_vp_macd.yaml")
    u = undecided(spec)
    for k in ("variant", "context.hourly_alignment", "bollinger.stdev", "bollinger.reprice_each_bar",
              "volume_profile.align_ticks", "trigger.confirmation_window_bars", "orders.stop.type", "sizing.instrument"):
        assert k in u
    # every leaf that is not structural is undecided: nothing was given a silent default
    assert set(flatten(spec)) - set(u) <= {"spec_version", "strategy", "orders.max_concurrent_positions"}


def test_refuses_to_run_undecided(small_md, session_settings):
    spec = load_spec(session_settings.path("specs") / "bb_vp_macd.yaml")
    with pytest.raises(SpecIncomplete):
        run_backtest(small_md, "bb_vp_macd_limit", flatten(spec), session_settings)


@pytest.mark.parametrize("variant,strategy", [("limit", "bb_vp_macd_limit"), ("trigger", "bb_vp_macd_trigger")])
def test_runs_when_filled(small_md, session_settings, variant, strategy):
    p = params(variant=variant)
    r = run_backtest(small_md, strategy, p, session_settings)
    assert r.trades.height > 0
    assert (r.trades["contracts"] == 10).all()         # $200 risk / (40 ticks = 10 pts x $2/pt MNQ)
    assert set(r.trades["exit_reason"].unique()) <= {"stop", "target", "exit_signal", "flatten_1510", "session_end"}
    et = (r.trades["entry_ct_minute"] + 60).to_numpy()
    assert et.min() >= 9 * 60 + 30 and et.max() <= 10 * 60 + 30


def test_variant_mismatch_refused(small_md, session_settings):
    with pytest.raises(ValueError):
        run_backtest(small_md, "bb_vp_macd_trigger", params(variant="limit"), session_settings)


def test_frozen_vs_repriced_differ(small_md, session_settings):
    a = run_backtest(small_md, "bb_vp_macd_limit", params(**{"bollinger.reprice_each_bar": True}), session_settings)
    b = run_backtest(small_md, "bb_vp_macd_limit", params(**{"bollinger.reprice_each_bar": False,
                                                             "orders.cancel_if_band_moves_ticks": 8}), session_settings)
    assert a.trades.height != b.trades.height or not np.allclose(a.trades["entry_price"].to_numpy(), b.trades["entry_price"].to_numpy())


def test_grid_dimensions_from_lists():
    d = copy.deepcopy(FILLED)
    d["bollinger"]["length"] = [14, 20, 30]
    d["volume_profile"]["long_levels"] = [["VAL"], ["VAL", "POC"]]
    dims = grid_dimensions(d)
    assert set(dims) == {"bollinger.length", "volume_profile.long_levels"}
    assert len(expand(d)) == 6


def test_no_cross_fabricated_across_sessions():
    """A1 regression: the previous value of a session's first window bar is the previous
    MINUTE (continuous history), not the previous session's last window bar."""
    from nqlab.strategies.bb_vp_macd import _crosses, _prev_distinct

    line = np.array([1.0, 1.0, -1.0, -1.0, 2.0])
    sig = np.zeros(5)
    up, dn = _crosses(line, sig, line - sig, "line_cross_signal", 0)
    assert dn.tolist() == [False, False, True, False, False] and up.tolist() == [False] * 4 + [True]
    htf = np.array([1.0, 1.0, 1.0, 3.0, 3.0, 2.0])
    assert np.isnan(_prev_distinct(htf)[0]) and _prev_distinct(htf)[3:].tolist() == [1.0, 1.0, 3.0]


def test_lint_flags_omitted_and_invalid_fields(settings, tmp_path):
    from nqlab.llm.roles import lint_spec

    d = copy.deepcopy(FILLED)
    d["strategy"] = "bb_vp_macd"
    del d["bollinger"]["stdev"]                       # silently deleted decision
    d["context"]["hourly_alignment"] = "clok"         # typo
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(d))
    settings.raw["llm"]["enabled"] = False
    r = lint_spec(settings, p, use_llm=False)
    fields = {q["field"] for q in r["questions"]}
    assert "bollinger.stdev" in fields and "context.hourly_alignment" in fields and not r["runnable"]
    filled = copy.deepcopy(FILLED)
    filled["strategy"] = "bb_vp_macd"
    p.write_text(yaml.safe_dump(filled))
    assert lint_spec(settings, p, use_llm=False)["runnable"]


def test_omitted_field_refused_at_run(small_md, session_settings):
    p = params()
    del p["bollinger.stdev"]
    with pytest.raises(SpecIncomplete, match="omitted"):
        run_backtest(small_md, "bb_vp_macd_limit", p, session_settings)


def test_random_entry_with_operator_exits(small_md, session_settings):
    p = params(rate=1.0, seed=3)
    r = run_backtest(small_md, "random_entry_spec", p, session_settings)
    assert r.trades.height > 20
    assert (r.trades["stop_dist"] == 10.0).all()                       # his 40-tick stop
    assert np.allclose(r.trades["target_dist"].to_numpy(), 15.0)       # his 1.5R target
    et = (r.trades["entry_ct_minute"] + 60).to_numpy()
    assert et.min() >= 9 * 60 + 30 and et.max() <= 10 * 60 + 30         # his window


def test_topstep_sizing_uses_buffer_pct():
    from nqlab.risk.topstep import SZ_BUFFER_FRAC, SZ_RISK_USD
    from nqlab.strategies.registry import get_strategy

    st = get_strategy("bb_vp_macd_limit")
    assert st.topstep_policy(params()).mode == SZ_BUFFER_FRAC
    assert st.topstep_policy(params(**{"sizing.risk_pct_of_mll_buffer": 0})).mode == SZ_RISK_USD
