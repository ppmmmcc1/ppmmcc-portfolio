"""Account-farming economics: fees, resets, activation, payouts net of all costs."""

import numpy as np
import pytest

from nqlab.risk.farming import Fees, demeaned, farm, haircut, lifecycle
from nqlab.risk.topstep import AccountRules, SizingPolicy
from tests.test_topstep import tape_from_days


@pytest.fixture()
def cfg(settings):
    return settings.topstep_rules


def test_rules_match_dashboard(cfg):
    r = AccountRules.from_config(cfg, "NQ")
    assert (r.target, r.mll_amount, r.consistency_pct, r.max_contracts) == (3000, 2000, 0.55, 5)
    assert r.dll_enabled and r.dll_amount == 1000 and r.x_dll_enabled
    assert (r.x_max_payout, r.x_cons_max_payout, r.x_cons_pct, r.x_cons_min_days) == (4000, 6000, 0.40, 3)
    assert AccountRules.from_config(cfg, "MNQ").max_contracts == 50
    f = Fees.from_config(cfg)
    assert (f.subscription_per_month, f.reset_fee, f.reset_credits_per_month, f.xfa_activation_fee) == (49, 49, 1, 149)


def test_fee_accounting(cfg):
    f = Fees.from_config(cfg)
    assert f.combine_cost(10, 0) == (49, 1, 0)            # first month, no reset
    assert f.combine_cost(10, 1) == (49, 1, 0)            # the monthly reset credit covers it
    assert f.combine_cost(10, 3) == (49 + 2 * 49, 1, 2)
    assert f.combine_cost(43, 2) == (3 * 49, 3, 0)        # 43 sessions = 3 billed months


def test_lifecycle_pass_and_payouts(cfg):
    r = AccountRules.from_config(cfg, "NQ")
    f = Fees.from_config(cfg)
    # +$300/day: passes after enough days (consistency 55%), then five $150+ days per payout
    days = [[(15.2, 0, 10)]] * 60
    lc = lifecycle(tape_from_days(days), r, f, SizingPolicy.fixed(1), SizingPolicy.fixed(1), 250, 30)
    assert lc.passed and lc.payouts >= 2 and lc.cost == 49 + 149
    assert lc.net == pytest.approx(lc.paid - 198)


def test_lifecycle_resets_until_budget(cfg):
    r = AccountRules.from_config(cfg, "NQ")
    f = Fees.from_config(cfg)
    days = [[(-25, 25, 30)]] * 200          # -$500/day: the DLL is not hit, the MLL fails every 4 days
    lc = lifecycle(tape_from_days(days), r, f, SizingPolicy.fixed(1), SizingPolicy.fixed(1), 40, 30)
    assert not lc.passed and lc.resets >= 8 and lc.paid == 0
    months = 2
    assert lc.cost == months * 49 + (lc.resets - months) * 49


def test_zero_edge_loses_money_and_edge_makes_it(cfg):
    r = AccountRules.from_config(cfg, "NQ")
    f = Fees.from_config(cfg)
    rng = np.random.default_rng(0)
    edge = [[(rng.normal(3.0, 15.0), 0.0, 10.0)] for _ in range(1500)]
    edge = [[(p, max(0.0, -p), s) for p, _, s in d] for d in edge]
    t = tape_from_days(edge)
    pol = SizingPolicy.fixed(1)
    s = farm(t, r, f, pol, pol, n_paths=300, combine_budget=120, xfa_horizon=120)
    z = farm(demeaned(t), r, f, pol, pol, n_paths=300, combine_budget=120, xfa_horizon=120, seed=1)
    h = farm(haircut(t, 1.0), r, f, pol, pol, n_paths=300, combine_budget=120, xfa_horizon=120, seed=2)
    assert z.ev_net < 0 < s.ev_net
    assert h.ev_net <= s.ev_net
    assert 0 <= s.p_net_positive <= 1 and s.mean_months > 0


def test_xfa_scaling_plan_limits_size(cfg):
    from nqlab.risk.topstep import run_xfa

    days = [[(10.2, 0, 10)]] * 40
    t = tape_from_days(days)

    def paid(c):
        r = AccountRules.from_config(c, "NQ")
        n, p_, *_ = run_xfa(t.day_ptr, t.pnl_pts, t.mae_pts, t.stop_pts, t.fee_pc, t.point_value, t.tick_value, 2.0, 0,
                            40, *r.xfa_tuple(), r.max_contracts, 0, 5.0)
        return p_

    free = paid(cfg)
    cfg["xfa"]["scaling_plan"] = [{"balance": -2000, "minis": 1}, {"balance": 3000, "minis": 3}]
    assert AccountRules.from_config(cfg, "MNQ").x_scale_contracts == (10, 30)
    assert 0 < paid(cfg) < free
