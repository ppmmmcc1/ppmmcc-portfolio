"""Topstep rules layer: a stateful account between the strategy's trades and the P&L.

Driven entirely by ``config/topstep_50k.yaml``. The trade list comes from the engine (one
position at a time, flat by 15:10 CT), so the real-time Maximum Loss Limit check can be
done exactly per trade: equity at the trade's worst point (bar lows/highs, i.e. its
MAE) is balance + realized-today - MAE. Touching the MLL (or DLL) liquidates at the
limit plus market slippage; an MLL touch fails the account, a DLL touch ends the day.

One numba implementation is used for everything (single backtest trace, every-start
historical Combines, bootstrap Monte Carlo, the sizing search). ``tests/test_topstep.py``
cross-checks it against an independent pure-Python reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numba as nb
import numpy as np
import polars as pl

OUT_PASS, OUT_FAIL_MLL, OUT_UNRESOLVED = 1, 2, 3
EV_NONE, EV_MLL, EV_DLL, EV_PASS, EV_PAYOUT, EV_CONSISTENCY_BLOCK = 0, 1, 2, 3, 4, 5
SZ_FIXED, SZ_BUFFER_FRAC, SZ_RISK_USD = 0, 1, 2


@dataclass(frozen=True)
class AccountRules:
    start: float
    target: float
    mll_amount: float
    mll_lock_at: float
    mll_lock_trigger: float
    dll_enabled: bool
    dll_amount: float
    consistency_pct: float
    max_contracts: int
    # XFA
    x_mll_amount: float
    x_lock_at: float
    x_reset_after_payout: float
    x_split: float
    x_min_payout: float
    x_path_standard: bool
    x_win_days: int
    x_win_min: float
    x_max_pct: float
    x_max_payout: float
    x_cons_min_days: int
    x_cons_pct: float
    x_cons_max_payout: float
    x_dll_enabled: bool
    x_dll_amount: float
    verify_note: str = ""

    @classmethod
    def from_config(cls, cfg: dict[str, Any], instrument: str = "NQ") -> "AccountRules":
        c, x = cfg["combine"], cfg["xfa"]
        mc = c["max_contracts"]
        maxc = int(mc["minis"] if instrument == "NQ" else mc["micros"])
        return cls(
            start=float(c["account_size"]), target=float(c["profit_target"]),
            mll_amount=float(c["max_loss_limit"]["amount"]), mll_lock_at=float(c["max_loss_limit"]["locks_at"]),
            mll_lock_trigger=float(c["max_loss_limit"].get("lock_trigger_balance", c["max_loss_limit"]["locks_at"] + c["max_loss_limit"]["amount"])),
            dll_enabled=bool(c["daily_loss_limit"].get("enabled", False)), dll_amount=float(c["daily_loss_limit"]["amount"]),
            consistency_pct=float(c["consistency"]["best_day_max_pct"]), max_contracts=maxc,
            x_mll_amount=float(x["max_loss_limit"]["amount"]), x_lock_at=float(x["max_loss_limit"]["locks_at"]),
            x_reset_after_payout=float(x["max_loss_limit"]["reset_to_after_first_payout"]),
            x_split=float(x["payout_split"]), x_min_payout=float(x["min_payout"]),
            x_path_standard=x.get("path", "standard") == "standard",
            x_win_days=int(x["standard"]["winning_days_required"]), x_win_min=float(x["standard"]["winning_day_min"]),
            x_max_pct=float(x["standard"]["max_pct_of_balance"]), x_max_payout=float(x["standard"]["max_payout"]),
            x_cons_min_days=int(x["consistency_path"]["min_trading_days"]), x_cons_pct=float(x["consistency_path"]["best_day_max_pct"]),
            x_cons_max_payout=float(x["consistency_path"]["max_payout"]),
            x_dll_enabled=bool(x.get("daily_loss_limit", {}).get("enabled", False)),
            x_dll_amount=float(x.get("daily_loss_limit", {}).get("amount", 1e18)),
            verify_note=str(cfg.get("verify_note", "")),
        )

    def combine_tuple(self):
        return (self.start, self.target, self.mll_amount, self.mll_lock_at, self.mll_lock_trigger,
                self.dll_enabled, self.dll_amount, self.consistency_pct, self.max_contracts)

    def xfa_tuple(self):
        return (self.x_mll_amount, self.x_lock_at, self.x_reset_after_payout, self.x_split, self.x_min_payout,
                self.x_path_standard, self.x_win_days, self.x_win_min, self.x_max_pct, self.x_max_payout,
                self.x_cons_min_days, self.x_cons_pct, self.x_cons_max_payout, self.x_dll_enabled, self.x_dll_amount)


@dataclass
class TradeTape:
    """Trades grouped by session (CSR layout). Values are PER CONTRACT."""
    day_ptr: np.ndarray       # len n_days+1
    pnl_pts: np.ndarray
    mae_pts: np.ndarray
    stop_pts: np.ndarray
    fee_pc: float             # round-turn fee per contract
    point_value: float
    tick_value: float
    session_dates: list

    @property
    def n_days(self) -> int:
        return len(self.day_ptr) - 1

    @classmethod
    def from_backtest(cls, trades: pl.DataFrame, daily: pl.DataFrame, point_value: float, tick_value: float,
                      fee_pc: float) -> "TradeTape":
        dates = daily["session_date"].to_list()
        t = trades.sort("entry_ts")
        pos = {d: i for i, d in enumerate(dates)}
        counts = np.zeros(len(dates), np.int64)
        for d in t["session_date"].to_list():
            counts[pos[d]] += 1
        ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        order = np.argsort([pos[d] for d in t["session_date"].to_list()], kind="stable")
        return cls(ptr, t["pnl_points"].to_numpy()[order].astype(float), t["mae_points"].to_numpy()[order].astype(float),
                   t["stop_dist"].to_numpy()[order].astype(float), fee_pc, point_value, tick_value, dates)

    def resample(self, day_index: np.ndarray) -> "TradeTape":
        cnt = np.diff(self.day_ptr)[day_index]
        csum = np.cumsum(cnt)
        ptr = np.concatenate([[0], csum]).astype(np.int64)
        total = int(csum[-1]) if len(csum) else 0
        idx = (np.repeat(self.day_ptr[day_index] - (csum - cnt), cnt) + np.arange(total)).astype(np.int64)
        return TradeTape(ptr, self.pnl_pts[idx], self.mae_pts[idx], self.stop_pts[idx], self.fee_pc,
                         self.point_value, self.tick_value, [self.session_dates[d] for d in day_index])


# ---------------------------------------------------------------------------------------
# the account kernel
# ---------------------------------------------------------------------------------------
@nb.njit(cache=True)
def _size(mode, param, buffer, stop_pts, point_value, max_c):
    if mode == SZ_FIXED:
        n = int(param)
    else:
        if not (stop_pts > 0):
            return 0
        budget = param * buffer if mode == SZ_BUFFER_FRAC else param
        n = int(np.floor(budget / (stop_pts * point_value)))
    if n > max_c:
        n = max_c
    return n if n > 0 else 0


@nb.njit(cache=True)
def _trade_day(balance, mll, day_real, dll_on, dll_amt, t0, t1, pnl, mae, stop, fee_pc, pv, tv, liq_ticks,
               sz_mode, sz_param, max_c):
    """Run one session's trades. Returns (balance, day_real, event, n_traded)."""
    traded = 0
    for k in range(t0, t1):
        n = _size(sz_mode, sz_param, balance - mll, stop[k], pv, max_c)
        if n == 0:
            continue
        traded += 1
        worst = balance - mae[k] * pv * n - fee_pc * n
        if worst <= mll:
            loss = (balance - mll) + liq_ticks * tv * n + fee_pc * n
            balance -= loss
            day_real -= loss
            return balance, day_real, EV_MLL, traded
        if dll_on and day_real - mae[k] * pv * n <= -dll_amt:
            loss = (dll_amt + day_real) + liq_ticks * tv * n + fee_pc * n
            balance -= loss
            day_real -= loss
            return balance, day_real, EV_DLL, traded
        r = pnl[k] * pv * n - fee_pc * n
        balance += r
        day_real += r
    return balance, day_real, EV_NONE, traded


@nb.njit(cache=True)
def run_combine(day_ptr, pnl, mae, stop, fee_pc, pv, tv, liq_ticks, d0, max_days,
                start, target, mll_amt, lock_at, lock_trig, dll_on, dll_amt, cons_pct, max_c,
                sz_mode, sz_param, trace, tr_bal, tr_mll, tr_pnl, tr_ev):
    """Returns (outcome, end_day_exclusive, sessions_used, consistency_blocked, best_day)."""
    n_days = day_ptr.shape[0] - 1
    balance = start
    eod_high = start
    mll = start - mll_amt
    best = 0.0
    blocked = False
    used = 0
    d = d0
    while d < n_days and used < max_days:
        bal0 = balance
        balance, day_real, ev, traded = _trade_day(balance, mll, 0.0, dll_on, dll_amt, day_ptr[d], day_ptr[d + 1],
                                                   pnl, mae, stop, fee_pc, pv, tv, liq_ticks, sz_mode, sz_param, max_c)
        day_pnl = balance - bal0
        used += 1
        if trace:
            tr_bal[used - 1] = balance
            tr_mll[used - 1] = mll
            tr_pnl[used - 1] = day_pnl
            tr_ev[used - 1] = ev
        if ev == EV_MLL:
            return OUT_FAIL_MLL, d + 1, used, blocked, best
        if day_pnl > best:
            best = day_pnl
        if balance > eod_high:
            eod_high = balance
        if eod_high >= lock_trig:
            mll = lock_at
        elif eod_high - mll_amt > mll:
            mll = eod_high - mll_amt
        profit = balance - start
        if profit >= target:
            if best <= cons_pct * profit:
                if trace:
                    tr_ev[used - 1] = EV_PASS
                return OUT_PASS, d + 1, used, blocked, best
            blocked = True
            if trace and tr_ev[used - 1] == EV_NONE:
                tr_ev[used - 1] = EV_CONSISTENCY_BLOCK
        d += 1
    return OUT_UNRESOLVED, d, used, blocked, best


@nb.njit(cache=True)
def run_xfa(day_ptr, pnl, mae, stop, fee_pc, pv, tv, liq_ticks, d0, max_days,
            mll_amt, lock_at, reset_after, split, min_pay, standard, win_days, win_min, max_pct, max_pay,
            cons_days, cons_pct, cons_max, dll_on, dll_amt, max_c, sz_mode, sz_param):
    """Express Funded Account from a $0 balance. Returns (n_payouts, paid_to_trader,
    breached, sessions_used, first_payout_session)."""
    n_days = day_ptr.shape[0] - 1
    balance = 0.0
    eod_high = 0.0
    mll = -mll_amt
    locked = False
    payouts = 0
    paid = 0.0
    wins = 0
    days_since = 0
    best_since = 0.0
    profit_since = 0.0
    first = -1
    used = 0
    d = d0
    while d < n_days and used < max_days:
        bal0 = balance
        balance, day_real, ev, traded = _trade_day(balance, mll, 0.0, dll_on, dll_amt, day_ptr[d], day_ptr[d + 1],
                                                   pnl, mae, stop, fee_pc, pv, tv, liq_ticks, sz_mode, sz_param, max_c)
        used += 1
        day_pnl = balance - bal0
        if ev == EV_MLL:
            return payouts, paid, True, used, first
        if traded > 0:
            days_since += 1
        if day_pnl >= win_min:
            wins += 1
        profit_since += day_pnl
        if day_pnl > best_since:
            best_since = day_pnl
        if not locked:
            if balance > eod_high:
                eod_high = balance
            if eod_high - mll_amt > mll:
                mll = eod_high - mll_amt
            if mll > lock_at:
                mll = lock_at
        eligible = False
        cap = 0.0
        if standard:
            if wins >= win_days:
                eligible = True
                cap = max_pay
        else:
            if days_since >= cons_days and profit_since > 0 and best_since <= cons_pct * profit_since:
                eligible = True
                cap = cons_max
        if eligible and balance > 0:
            amt = max_pct * balance
            if amt > cap:
                amt = cap
            if amt >= min_pay:
                balance -= amt
                payouts += 1
                paid += amt * split
                if first < 0:
                    first = used
                wins = 0
                days_since = 0
                best_since = 0.0
                profit_since = 0.0
                if payouts == 1:
                    mll = reset_after
                    locked = True
        d += 1
    return payouts, paid, False, used, first


# ---------------------------------------------------------------------------------------
# Python-facing API
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SizingPolicy:
    mode: int = SZ_FIXED
    param: float = 1.0

    @classmethod
    def fixed(cls, n: int) -> "SizingPolicy":
        return cls(SZ_FIXED, float(n))

    @classmethod
    def buffer_frac(cls, f: float) -> "SizingPolicy":
        return cls(SZ_BUFFER_FRAC, float(f))

    @classmethod
    def risk_usd(cls, usd: float) -> "SizingPolicy":
        return cls(SZ_RISK_USD, float(usd))

    def label(self) -> str:
        return {SZ_FIXED: f"{int(self.param)} contract(s)", SZ_BUFFER_FRAC: f"{self.param:.0%} of MLL buffer",
                SZ_RISK_USD: f"${self.param:.0f} risk/trade"}[self.mode]


def _args(tape: TradeTape, liq_ticks: float):
    return (tape.day_ptr, tape.pnl_pts, tape.mae_pts, np.nan_to_num(tape.stop_pts, nan=-1.0),
            float(tape.fee_pc), float(tape.point_value), float(tape.tick_value), float(liq_ticks))


def combine_trace(tape: TradeTape, rules: AccountRules, policy: SizingPolicy, liq_ticks: float = 2.0) -> pl.DataFrame:
    """Sequential Combine attempts across the whole history (restart the day after each
    pass or failure). Feeds the ``daily`` table and the equity/MLL chart."""
    rows = []
    d = 0
    attempt = 0
    while d < tape.n_days:
        attempt += 1
        n = tape.n_days - d
        bal, mll, pnl, ev = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
        out, end, used, blocked, best = run_combine(*_args(tape, liq_ticks), d, 10**9, *rules.combine_tuple(),
                                                    policy.mode, policy.param, True, bal, mll, pnl, ev)
        best = 0.0
        for k in range(used):
            best = max(best, pnl[k])
            profit = bal[k] - rules.start
            rows.append({"session_date": tape.session_dates[d + k], "attempt": attempt, "balance": bal[k],
                         "mll_level": mll[k], "pnl_usd": pnl[k], "event": int(ev[k]),
                         "consistency_ratio": best / profit if profit > 0 else None})
        d = end
    df = pl.DataFrame(rows, schema={"session_date": pl.Date, "attempt": pl.Int64, "balance": pl.Float64,
                                    "mll_level": pl.Float64, "pnl_usd": pl.Float64, "event": pl.Int64,
                                    "consistency_ratio": pl.Float64})
    names = {EV_NONE: "", EV_MLL: "MLL_BREACH", EV_DLL: "DLL_HIT", EV_PASS: "PASS", EV_CONSISTENCY_BLOCK: "CONSISTENCY_BLOCK"}
    return df.with_columns(pl.col("event").replace_strict(names, return_dtype=pl.Utf8).alias("rule_events"))


@dataclass
class CombineStats:
    n_paths: int
    p_pass: float
    p_fail_mll: float
    p_unresolved: float
    p_fail_consistency: float
    median_sessions_to_pass: float
    p_first_payout_given_pass: float
    expected_payouts: float
    expected_paid_given_pass: float
    max_consec_losses: np.ndarray
    sessions_to_pass: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items() if not isinstance(v, np.ndarray)}


def _simulate_paths(tapes_and_starts, rules, policy, horizon, xfa_horizon, liq_ticks):
    outs, used_l, blocked_l, firsts, pays, paids, passed = [], [], [], [], [], [], []
    for tape, d0 in tapes_and_starts:
        a = _args(tape, liq_ticks)
        dummy = np.zeros(1)
        out, end, used, blocked, best = run_combine(*a, d0, horizon, *rules.combine_tuple(),
                                                    policy.mode, policy.param, False, dummy, dummy, dummy, np.zeros(1, np.int64))
        outs.append(out)
        used_l.append(used)
        blocked_l.append(blocked)
        if out == OUT_PASS:
            n_pay, paid, breached, xused, first = run_xfa(*a, end, xfa_horizon, *rules.xfa_tuple(), rules.max_contracts,
                                                          policy.mode, policy.param)
            firsts.append(first >= 0)
            pays.append(n_pay)
            paids.append(paid)
    return np.array(outs), np.array(used_l), np.array(blocked_l), np.array(firsts), np.array(pays), np.array(paids)


def _stats(outs, used, blocked, firsts, pays, paids, mcl) -> CombineStats:
    n = len(outs)
    passed = outs == OUT_PASS
    return CombineStats(
        n_paths=n,
        p_pass=float(passed.mean()) if n else float("nan"),
        p_fail_mll=float((outs == OUT_FAIL_MLL).mean()) if n else float("nan"),
        p_unresolved=float((outs == OUT_UNRESOLVED).mean()) if n else float("nan"),
        p_fail_consistency=float(((outs == OUT_FAIL_MLL) & blocked).mean()) if n else float("nan"),
        median_sessions_to_pass=float(np.median(used[passed])) if passed.any() else float("nan"),
        p_first_payout_given_pass=float(firsts.mean()) if len(firsts) else float("nan"),
        expected_payouts=float(pays.sum() / n) if n else float("nan"),
        expected_paid_given_pass=float(paids.mean()) if len(paids) else float("nan"),
        max_consec_losses=mcl,
        sessions_to_pass=used[passed],
    )


def historical_combines(tape: TradeTape, rules: AccountRules, policy: SizingPolicy, horizon: int = 250,
                        xfa_horizon: int = 250, liq_ticks: float = 2.0) -> CombineStats:
    """A Combine started on EVERY session in history (overlapping windows). Paths that
    run out of data are 'unresolved', never counted as passes."""
    res = _simulate_paths([(tape, d) for d in range(tape.n_days)], rules, policy, horizon, xfa_horizon, liq_ticks)
    return _stats(*res, mcl=np.array([]))


@nb.njit(cache=True)
def _stationary(n_days, starts, jumps):
    h = starts.shape[0]
    idx = np.empty(h, np.int64)
    cur = starts[0]
    for k in range(h):
        if k == 0 or jumps[k]:
            cur = starts[k]
        else:
            cur = (cur + 1) % n_days
        idx[k] = cur
    return idx


def stationary_indices(n_days: int, horizon: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (geometric block lengths, wrap-around)."""
    starts = rng.integers(0, n_days, horizon)
    jumps = rng.random(horizon) < 1.0 / mean_block
    return _stationary(n_days, starts, jumps)


def bootstrap_combines(tape: TradeTape, rules: AccountRules, policy: SizingPolicy, n_paths: int = 10000,
                       horizon: int = 250, xfa_horizon: int = 250, mean_block: float = 5.0, seed: int = 0,
                       liq_ticks: float = 2.0) -> CombineStats:
    """Stationary block bootstrap of whole sessions (keeps some loss clustering)."""
    rng = np.random.default_rng(seed)
    paths = []
    mcl = np.zeros(n_paths, np.int64)
    for k in range(n_paths):
        idx = stationary_indices(tape.n_days, horizon + xfa_horizon, mean_block, rng)
        t = tape.resample(idx)
        paths.append((t, 0))
        # max consecutive losing trades in the combine part of the path
        lp = t.pnl_pts[: t.day_ptr[horizon]]
        cur = best = 0
        for v in lp:
            cur = cur + 1 if v < 0 else 0
            best = max(best, cur)
        mcl[k] = best
    res = _simulate_paths(paths, rules, policy, horizon, xfa_horizon, liq_ticks)
    return _stats(*res, mcl=mcl)


ZERO_EDGE_LINE = ("A zero-edge trader passes this Combine roughly 40% of the time (gambler's ruin with a "
                  "$2,000 floor and a $3,000 target). Passing is weak evidence of skill.")


def zero_edge_pass_probability(rules: AccountRules) -> float:
    """Gambler's ruin with zero drift: P(hit +target before -MLL) = MLL / (MLL + target)."""
    return rules.mll_amount / (rules.mll_amount + rules.target)


def zero_edge_benchmark(tape: TradeTape, rules: AccountRules, policy: SizingPolicy, n_paths: int = 2000,
                        horizon: int = 250, seed: int = 1) -> CombineStats:
    """The same trade-size distribution with its edge removed (P&L demeaned so the net
    expectancy after fees is zero). This is the honest baseline for P(pass): the 40%
    gambler's-ruin figure assumes a static floor, and the trailing MLL lowers it."""
    if len(tape.pnl_pts) == 0:
        return bootstrap_combines(tape, rules, policy, n_paths=1, horizon=1, xfa_horizon=1)
    shift = tape.pnl_pts.mean() - tape.fee_pc / tape.point_value
    z = TradeTape(tape.day_ptr, tape.pnl_pts - shift, np.maximum(tape.mae_pts, np.maximum(0.0, -(tape.pnl_pts - shift))),
                  tape.stop_pts, tape.fee_pc, tape.point_value, tape.tick_value, tape.session_dates)
    return bootstrap_combines(z, rules, policy, n_paths=n_paths, horizon=horizon, xfa_horizon=horizon, seed=seed)


def optimal_risk_search(tape: TradeTape, rules: AccountRules, fracs: np.ndarray | None = None,
                        n_paths: int = 2000, horizon: int = 250, seed: int = 0) -> pl.DataFrame:
    """Sweep risk per trade as a fraction of the CURRENT MLL buffer (2%..40%) and report
    P(first payout) and expected profit. Effective capital is the distance to the MLL,
    not the $50,000 headline: risking 1% of $50k ($500) means four losses end the account."""
    fracs = np.round(np.arange(0.02, 0.401, 0.02), 3) if fracs is None else fracs
    rows = []
    for f in fracs:
        st = bootstrap_combines(tape, rules, SizingPolicy.buffer_frac(float(f)), n_paths=n_paths, horizon=horizon,
                                xfa_horizon=horizon, seed=seed)
        rows.append({"risk_frac_of_buffer": float(f), "p_pass": st.p_pass,
                     "p_first_payout": st.p_pass * (st.p_first_payout_given_pass if st.p_pass > 0 else 0.0),
                     "expected_paid_usd": st.p_pass * (st.expected_paid_given_pass if st.p_pass > 0 else 0.0)})
    return pl.DataFrame(rows)
