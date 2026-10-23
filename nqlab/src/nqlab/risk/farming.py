"""Account farming economics: what one purchased 50K account is worth, after every fee.

One *lifecycle* = buy a Combine, trade it (resetting after each failure) until it passes
or the Combine budget runs out, pay the activation fee, then trade the Express Funded
Account collecting every payout you are eligible for, until its MLL is breached or the
horizon ends. Net = payouts to you - subscription months - paid resets - activation.

Why this is the right objective for farming: the most you can lose on an account is the
fees, so the quantity to maximize is expected net dollars per account (and per month of
your time). It is NOT a reason to oversize: the search below finds the risk per trade
that maximizes expected net, and the report always shows the zero-edge baseline (a
trader with no edge pays the fees and occasionally gets lucky; with these rules that is
negative), plus the result with the edge haircut to its one-standard-error lower bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from nqlab.risk.topstep import (OUT_FAIL_MLL, OUT_PASS, AccountRules, SizingPolicy, TradeTape, run_combine,
                                run_xfa, stationary_indices)


@dataclass(frozen=True)
class Fees:
    subscription_per_month: float
    reset_fee: float
    reset_credits_per_month: int
    xfa_activation_fee: float
    sessions_per_month: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Fees":
        f = cfg.get("fees", {})
        return cls(float(f.get("subscription_per_month", 49)), float(f.get("reset_fee", 49)),
                   int(f.get("reset_credits_per_month", 1)), float(f.get("xfa_activation_fee", 149)),
                   int(f.get("sessions_per_month", 21)))

    def combine_cost(self, sessions: int, resets: int) -> tuple[float, int, int]:
        """(dollars, billed months, paid resets) for a Combine phase."""
        months = max(1, math.ceil(sessions / self.sessions_per_month))
        paid = max(0, resets - months * self.reset_credits_per_month)
        return months * self.subscription_per_month + paid * self.reset_fee, months, paid


@dataclass
class Lifecycle:
    passed: bool
    combine_sessions: int
    resets: int
    cost: float
    payouts: int
    paid: float
    xfa_sessions: int
    breached: bool

    @property
    def net(self) -> float:
        return self.paid - self.cost


def _args(tape: TradeTape, liq: float):
    return (tape.day_ptr, tape.pnl_pts, tape.mae_pts, np.nan_to_num(tape.stop_pts, nan=-1.0), float(tape.fee_pc),
            float(tape.point_value), float(tape.tick_value), float(liq))


def lifecycle(tape: TradeTape, rules: AccountRules, fees: Fees, combine_policy: SizingPolicy,
              xfa_policy: SizingPolicy, combine_budget: int, xfa_horizon: int, d0: int = 0,
              liq_ticks: float = 2.0) -> Lifecycle:
    a = _args(tape, liq_ticks)
    dummy = np.zeros(1)
    d, used_total, resets, passed = d0, 0, 0, False
    while used_total < combine_budget and d < tape.n_days:
        out, end, used, _blocked, _best = run_combine(
            *a, d, combine_budget - used_total, *rules.combine_tuple(), combine_policy.mode, combine_policy.param,
            False, dummy, dummy, dummy, np.zeros(1, np.int64))
        used_total += used
        d = end
        if out == OUT_PASS:
            passed = True
            break
        if out == OUT_FAIL_MLL and used_total < combine_budget and d < tape.n_days:
            resets += 1                      # buy a reset and try again
            continue
        break
    cost, _months, _paid_resets = fees.combine_cost(used_total, resets)
    if not passed:
        return Lifecycle(False, used_total, resets, cost, 0, 0.0, 0, False)
    cost += fees.xfa_activation_fee
    n_pay, paid, breached, xused, _first = run_xfa(*a, d, xfa_horizon, *rules.xfa_tuple(), rules.max_contracts,
                                                  xfa_policy.mode, xfa_policy.param)
    return Lifecycle(True, used_total, resets, cost, int(n_pay), float(paid), int(xused), bool(breached))


@dataclass
class FarmStats:
    n_paths: int
    ev_net: float
    ev_net_se: float
    median_net: float
    p_net_positive: float
    p_funded: float
    mean_cost: float
    mean_paid: float
    mean_payouts: float
    p_any_payout: float
    mean_months: float
    ev_per_month: float
    nets: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items() if not isinstance(v, np.ndarray)}


def farm(tape: TradeTape, rules: AccountRules, fees: Fees, combine_policy: SizingPolicy, xfa_policy: SizingPolicy,
         n_paths: int = 2000, combine_budget: int = 250, xfa_horizon: int = 250, mean_block: float = 5.0,
         seed: int = 0) -> FarmStats:
    """Bootstrap lifecycles from the strategy's own sessions (stationary blocks)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_paths):
        idx = stationary_indices(tape.n_days, combine_budget + xfa_horizon, mean_block, rng)
        out.append(lifecycle(tape.resample(idx), rules, fees, combine_policy, xfa_policy, combine_budget, xfa_horizon))
    nets = np.array([o.net for o in out])
    months = np.array([(o.combine_sessions + o.xfa_sessions) / fees.sessions_per_month for o in out])
    return FarmStats(
        n_paths=n_paths, ev_net=float(nets.mean()), ev_net_se=float(nets.std(ddof=1) / math.sqrt(len(nets))),
        median_net=float(np.median(nets)), p_net_positive=float((nets > 0).mean()),
        p_funded=float(np.mean([o.passed for o in out])), mean_cost=float(np.mean([o.cost for o in out])),
        mean_paid=float(np.mean([o.paid for o in out])), mean_payouts=float(np.mean([o.payouts for o in out])),
        p_any_payout=float(np.mean([o.payouts > 0 for o in out])), mean_months=float(months.mean()),
        ev_per_month=float(nets.sum() / max(months.sum(), 1e-9)), nets=nets)


def demeaned(tape: TradeTape) -> TradeTape:
    """Same trade sizes and timing, zero net expectancy after fees (the zero-edge trader)."""
    if len(tape.pnl_pts) == 0:
        return tape
    shift = tape.pnl_pts.mean() - tape.fee_pc / tape.point_value
    p = tape.pnl_pts - shift
    return TradeTape(tape.day_ptr, p, np.maximum(tape.mae_pts, np.maximum(0.0, -p)), tape.stop_pts, tape.fee_pc,
                     tape.point_value, tape.tick_value, tape.session_dates)


def haircut(tape: TradeTape, n_se: float = 1.0) -> TradeTape:
    """The edge reduced to its n-standard-error lower bound (the observed edge is an
    estimate; farming on the point estimate oversizes and overpays)."""
    if len(tape.pnl_pts) < 2:
        return tape
    se = tape.pnl_pts.std(ddof=1) / math.sqrt(len(tape.pnl_pts))
    p = tape.pnl_pts - n_se * se
    return TradeTape(tape.day_ptr, p, np.maximum(tape.mae_pts, np.maximum(0.0, -p)), tape.stop_pts, tape.fee_pc,
                     tape.point_value, tape.tick_value, tape.session_dates)


def to_micros(tape: TradeTape, settings) -> TradeTape:
    m = settings.contract("MNQ")
    return TradeTape(tape.day_ptr, tape.pnl_pts, tape.mae_pts, tape.stop_pts, settings.round_turn_cost("MNQ"),
                     m.point_value, m.tick_value, tape.session_dates)


def optimize_farming(tape: TradeTape, rules_cfg: dict[str, Any], settings, fracs: list[float] | None = None,
                     n_paths: int = 600, combine_budget: int = 250, xfa_horizon: int = 250, seed: int = 0) -> dict[str, Any]:
    """Grid over risk per trade (fraction of the live MLL buffer) in the Combine and in the
    funded account, for both payout paths, in MNQ for sizing granularity. Returns every
    cell; the choice is made on expected net per account WITH the edge haircut, so the
    selected size survives the edge being one SE smaller than measured."""
    fracs = fracs or [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    rules = AccountRules.from_config(rules_cfg, "MNQ")
    fees = Fees.from_config(rules_cfg)
    mt = to_micros(tape, settings)
    hc = haircut(mt)
    rows = []
    for standard in (True, False):
        r = rules.with_path(standard)
        for fc in fracs:
            for fx in fracs:
                cp, xp = SizingPolicy.buffer_frac(fc), SizingPolicy.buffer_frac(fx)
                s = farm(mt, r, fees, cp, xp, n_paths, combine_budget, xfa_horizon, seed=seed)
                h = farm(hc, r, fees, cp, xp, n_paths, combine_budget, xfa_horizon, seed=seed)
                rows.append({"path": "standard" if standard else "consistency", "combine_risk": fc, "xfa_risk": fx,
                             "ev_net": s.ev_net, "ev_net_haircut": h.ev_net, "p_funded": s.p_funded,
                             "p_any_payout": s.p_any_payout, "ev_per_month": s.ev_per_month,
                             "p_net_positive": s.p_net_positive})
    df = pl.DataFrame(rows)
    best = df.sort("ev_net_haircut", descending=True).row(0, named=True)
    naive = df.sort("ev_net", descending=True).row(0, named=True)
    return {"grid": df, "best": best, "best_unhaircut": naive}


def farm_report(tape: TradeTape, settings, policy: SizingPolicy, instrument: str, n_paths: int = 2000,
                seed: int = 0) -> dict[str, Any]:
    """Expected economics per purchased account for the strategy as configured, the same
    trades with zero edge, and the edge haircut, under both payout paths."""
    rules_cfg = settings.topstep_rules
    ts = settings.section("topstep")
    fees = Fees.from_config(rules_cfg)
    base = AccountRules.from_config(rules_cfg, instrument)
    out: dict[str, Any] = {"fees": fees.__dict__, "paths": {}}
    for standard in (True, False):
        r = base.with_path(standard)
        name = "standard" if standard else "consistency"
        out["paths"][name] = {
            "strategy": farm(tape, r, fees, policy, policy, n_paths, ts["combine_horizon_sessions"], ts["xfa_horizon_sessions"], seed=seed),
            "zero_edge": farm(demeaned(tape), r, fees, policy, policy, n_paths, ts["combine_horizon_sessions"], ts["xfa_horizon_sessions"], seed=seed + 1),
            "haircut": farm(haircut(tape), r, fees, policy, policy, n_paths, ts["combine_horizon_sessions"], ts["xfa_horizon_sessions"], seed=seed + 2),
        }
    best = max(out["paths"], key=lambda k: out["paths"][k]["haircut"].ev_net)
    out["best_path"] = best
    return out
