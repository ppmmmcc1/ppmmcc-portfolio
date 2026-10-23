import numpy as np
import polars as pl
import pytest

from nqlab.selection.plateau import Grid, config_sharpes, plateau_scores, select
from nqlab.validation.cpcv import cpcv
from nqlab.validation.permutation import permute_bars
from nqlab.validation.stats import dsr, expected_max_sharpe, pbo_cscv, spa_test
from nqlab.validation.walkforward import walk_forward


def grid3x3():
    return Grid.build({"a": [1, 2, 3], "b": [10, 20, 30]})


def test_plateau_beats_spike():
    g = grid3x3()
    sh = np.zeros(9)
    sh[0] = 5.0                     # lone spike in a corner
    sh[[4, 5, 7, 8]] = 0.4          # a modest plateau
    sh[[1, 3]] = 0.1
    pl_ = plateau_scores(sh, g)
    assert int(np.argmax(sh)) == 0
    assert int(np.argmax(pl_)) in (4, 5, 7, 8)


def test_categorical_dims_are_not_neighbors():
    g = Grid.build({"a": [1, 2, 3], "mode": ["x", "y"]}, categorical={"mode"})
    for i, nb in enumerate(g.neighbors):
        assert all(g.configs[j]["mode"] == g.configs[i]["mode"] for j in nb)


def test_expected_max_sharpe_grows_with_n():
    v = 1 / 250
    assert expected_max_sharpe(1, v) == 0
    assert expected_max_sharpe(10, v) < expected_max_sharpe(1000, v) < expected_max_sharpe(100000, v)


def test_dsr_penalizes_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.08, 1, 500)
    a, b = dsr(r, 1)["dsr"], dsr(r, 1000)["dsr"]
    assert a > b


def test_pbo_noise_is_high_edge_is_low():
    rng = np.random.default_rng(2)
    g = Grid.build({"a": list(range(6)), "b": list(range(4))})
    noise = rng.normal(0, 100, (480, g.n))
    p_noise = pbo_cscv(noise, g, 12)["pbo"]
    edge = noise.copy()
    edge[:, 9] += 40; edge[:, [8, 10, 5, 13]] += 25
    p_edge = pbo_cscv(edge, g, 12)["pbo"]
    assert p_noise > 0.3 and p_edge < 0.1, (p_noise, p_edge)


def test_walk_forward_and_cpcv_shapes():
    rng = np.random.default_rng(3)
    g = grid3x3()
    m = rng.normal(0, 1, (300, g.n))
    w = walk_forward(m, g, 120, 20, 20)
    assert w["n_windows"] == 9 and len(w["oos"]) == 180
    wa = walk_forward(m, g, 120, 20, 20, anchored=True)
    assert wa["windows"][-1]["train"][0] == 0
    c = cpcv(m, g, 10, 2, 1)
    assert c["n_paths"] == 9 and all(len(p) == 300 for p in c["paths"])


def test_spa_runs():
    rng = np.random.default_rng(4)
    m = rng.normal(0, 1, (300, 5))
    res = spa_test(m, reps=200)
    assert 0 <= res["spa_p_consistent"] <= 1


def test_permutation_properties(small_md):
    pm = permute_bars(small_md, 1)
    a, b = small_md.bars, pm.bars
    assert a.height == b.height
    first = (a["session_date"] != a["session_date"].shift(1)).fill_null(True)
    assert (a.filter(first)["close"] == b.filter(first)["close"]).all()
    for c in ("open", "high", "low", "close"):
        x = b[c].to_numpy() * 4
        assert np.allclose(x, np.round(x))
    assert (b["high"] >= b["low"]).all() and (b["high"] >= b["close"]).all() and (b["low"] <= b["open"]).all()
    # the permutation changes the path
    assert not np.array_equal(a["close"].to_numpy(), b["close"].to_numpy())
    # per-bucket return dispersion is preserved approximately
    def bucket_sd(df):
        r = df.with_columns((pl.col("close").log().diff()).alias("r"), (pl.col("ct_minute") // 30).alias("bk"))
        r = r.filter(pl.col("session_date") == pl.col("session_date").shift(1))
        return dict(r.group_by("bk").agg(pl.col("r").std()).iter_rows())
    sa, sb = bucket_sd(a), bucket_sd(b)
    for k in sa:
        assert sb[k] == pytest.approx(sa[k], rel=0.25)


def test_mcpt_small(small_md, session_settings):
    from nqlab.validation.mcpt import run_mcpt

    cps = []
    res = run_mcpt(small_md, "null_fixed_time", {"stop_ticks": [20, 40]}, {"seed": 0, "entry_minute": 520,
                   "target_r": 1.0, "instrument": "NQ", "contracts": 1, "max_trades": 1}, (), session_settings,
                   n_perm=9, checkpoint_every=4, save_checkpoint=cps.append)
    assert len(res.null["wf_oos_mean"]) == 9
    assert 0 < res.p <= 1
    assert len(cps) >= 3
    # resume from a mid-run checkpoint: only the remaining permutations are run
    seen = []
    res2 = run_mcpt(small_md, "null_fixed_time", {"stop_ticks": [20, 40]}, {"seed": 0, "entry_minute": 520,
                    "target_r": 1.0, "instrument": "NQ", "contracts": 1, "max_trades": 1}, (), session_settings,
                    n_perm=9, checkpoint=cps[1], checkpoint_every=4, on_permutation=seen.append)
    assert len(seen) == 9 - len(cps[1]["done"])
    assert res2.p_values == res.p_values


def test_permuted_data_never_carries_real_trades(small_md):
    import datetime as dt

    from nqlab.data.market import MarketData
    from nqlab.features import compute

    trades = pl.DataFrame({"ts_utc": [small_md.bars["ts_utc"][600] + dt.timedelta(seconds=5)], "price": [15000.0],
                           "size": [10]})
    md = MarketData(small_md.bars, small_md.sessions, "x", meta={"trades": trades})
    pm = permute_bars(md, 3)
    assert "trades" not in pm.meta and pm.meta["bars_only"]
    f = compute(pm, "vp_prior_rth", build_from="trades")    # falls back to bars, consistently
    assert f["vp_approximate"].all()
