"""The selection funnel (spec section 9). Stage logic only; persistence is in family.py.

Stage 0  REGISTER  (prereg.py)
Stage 1  SCREEN    tier-2 grid; kill negative expectancy of the plateau pick
Stage 2  PLATEAU   score = median Sharpe of the neighborhood; heatmap
Stage 3  WALKFWD   walk-forward the SELECTION RULE; WFE; CPCV path distribution
Stage 4  PERMUTE   MCPT re-running the entire selection on permuted bars
Stage 5  DEFLATE   DSR (N from the ledger), PBO, SPA, family t-stat bar
Stage 6  PROPSIM   Topstep Monte Carlo vs a zero-edge benchmark
Stage 7  HOLDOUT   one scripted look (holdout_cmd.py)

Every gate resolves pessimistically: missing or too little evidence is INSUFFICIENT,
never PASS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nqlab.engine.backtest import run_backtest
from nqlab.results.ledger import TrialLedger
from nqlab.results.metrics import summarize
from nqlab.selection.plateau import Grid, config_sharpes, heatmap, plateau_scores, select
from nqlab.validation.cpcv import cpcv
from nqlab.validation.matrix import FamilyMatrix, evaluate_grid
from nqlab.validation.stats import dsr, pbo_cscv, spa_test
from nqlab.validation.walkforward import walk_forward

PASS, FAIL, INSUFFICIENT, ERRORED = "PASS", "FAIL", "INSUFFICIENT DATA", "ERRORED"
STAGE_NAMES = {0: "REGISTER", 1: "SCREEN", 2: "PLATEAU", 3: "WALKFWD", 4: "PERMUTE", 5: "DEFLATE",
               6: "PROPSIM", 7: "HOLDOUT"}


@dataclass
class StageResult:
    stage: int
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FamilyContext:
    prereg: dict[str, Any]
    md: Any
    settings: Any
    ledger: TrialLedger
    cache: dict[str, Any] = field(default_factory=dict)
    mcpt_checkpoint: dict[str, Any] | None = None
    save_checkpoint: Callable[[dict[str, Any]], None] | None = None
    n_perm: int | None = None
    n_jobs: int = 1
    min_perm: int = 99        # below this a permutation p-value cannot resolve 0.05 honestly

    @property
    def grid(self) -> Grid:
        p = self.prereg
        return Grid.build(p["param_space"], p["fixed"], tuple(p.get("categorical", [])))

    @property
    def crit(self) -> dict[str, Any]:
        return self.prereg["criteria"]

    def matrix(self, tier: int) -> FamilyMatrix:
        key = f"matrix_t{tier}"
        if key not in self.cache:
            p = self.prereg
            self.cache[key] = evaluate_grid(
                self.md, p["strategy"], self.grid, self.settings, fill_tier=tier, ledger=self.ledger,
                family_id=p["family_id"], origin=p.get("origin", "human"), keep_results=False,
                variant=p.get("llm_variant_id"))
        return self.cache[key]

    def selected(self) -> tuple[int, dict[str, Any]]:
        fm = self.matrix(2)
        i, _ = select(fm.pnl, self.grid)
        return i, self.grid.configs[i]


# ------------------------------------------------------------------------------------------
def stage_screen(ctx: FamilyContext) -> StageResult:
    fm2 = ctx.matrix(2)
    fm1 = ctx.matrix(1)
    i, cfg = ctx.selected()
    col = fm2.pnl[:, i]
    n_tr = int(fm2.n_trades[i])
    exp_day = float(col.mean()) if len(col) else float("nan")
    d = {"selected_index": i, "selected_params": cfg, "n_trades": n_tr, "n_sessions": fm2.T,
         "mean_daily_usd_t2": exp_day, "mean_daily_usd_t1": float(fm1.pnl[:, i].mean()) if fm1.T else float("nan"),
         "configs_positive_t2": int((fm2.pnl.mean(axis=0) > 0).sum()), "n_configs": fm2.pnl.shape[1]}
    if fm2.T < ctx.crit["min_sessions"] or n_tr < ctx.crit["min_trades"]:
        return StageResult(1, INSUFFICIENT, f"Only {n_tr} trades over {fm2.T} sessions; the pre-registered minimum "
                           f"is {ctx.crit['min_trades']} trades / {ctx.crit['min_sessions']} sessions.", d)
    if not exp_day > 0:
        return StageResult(1, FAIL, "The plateau-selected configuration loses money after costs under "
                           "trade-through (tier 2) fills.", d)
    return StageResult(1, PASS, "Positive tier-2 expectancy after costs at the plateau pick (this proves nothing yet).", d)


def stage_plateau(ctx: FamilyContext) -> StageResult:
    fm = ctx.matrix(2)
    g = ctx.grid
    sh = config_sharpes(fm.pnl)
    ps = plateau_scores(sh, g)
    i = int(np.argmax(ps))
    peak = int(np.argmax(sh))
    d = {"sharpes": sh.tolist(), "plateau": ps.tolist(), "selected_index": i, "peak_index": peak,
         "selected_plateau": float(ps[i]), "peak_sharpe": float(sh[peak]), "peak_plateau": float(ps[peak])}
    big = [k for k in g.keys if len(g.values[k]) > 1]
    if len(big) >= 2:
        xs, ys, raw, pla = heatmap(sh, g, big[0], big[1])
        d["heatmap"] = {"x": big[0], "y": big[1], "xs": xs, "ys": ys, "raw": raw.tolist(), "plateau": pla.tolist()}
    spike = sh[peak] > 0 and ps[peak] <= 0
    d["peak_is_spike"] = bool(spike)
    if ps[i] <= 0:
        return StageResult(2, FAIL, "No parameter plateau has a positive median Sharpe; any profitable config is "
                           "an isolated spike (noise).", d)
    return StageResult(2, PASS, f"Best plateau median daily Sharpe {ps[i]:.3f} "
                       f"({'peak config is a spike, ignored' if spike else 'plateau is broad'}).", d)


def stage_walkforward(ctx: FamilyContext) -> StageResult:
    fm = ctx.matrix(2)
    wfc = ctx.settings.section("validation")["walk_forward"]
    w = walk_forward(fm.pnl, ctx.grid, wfc["train_sessions"], wfc["test_sessions"], wfc["step_sessions"])
    wa = walk_forward(fm.pnl, ctx.grid, wfc["train_sessions"], wfc["test_sessions"], wfc["step_sessions"], anchored=True)
    cc = ctx.settings.section("validation")["cpcv"]
    cv = cpcv(fm.pnl, ctx.grid, cc["n_groups"], cc["k_test"], cc["embargo_sessions"])
    d = {"rolling": {k: v for k, v in w.items() if k != "oos"}, "anchored": {k: v for k, v in wa.items() if k != "oos"},
         "oos_daily": w["oos"].tolist() if len(w["oos"]) else [],
         "cpcv_path_sharpes": cv.get("path_sharpes", np.array([])).tolist(),
         "cpcv_frac_positive": cv.get("frac_paths_positive")}
    if w["n_windows"] < 3:
        return StageResult(3, INSUFFICIENT, f"Only {w['n_windows']} walk-forward windows fit in {fm.T} sessions "
                           "(need at least 3).", d)
    notes = []
    if w["unstable"]:
        notes.append("RED FLAG: the selected parameters jump wildly between windows")
    wfe = w["wfe"]
    if not w["oos_mean"] > 0:
        return StageResult(3, FAIL, "The selection rule, walked forward, loses money out of sample. " + "; ".join(notes), d)
    if not (wfe >= ctx.crit["wfe_min"]):
        return StageResult(3, FAIL, f"Walk-forward efficiency {wfe:.2f} is below {ctx.crit['wfe_min']}: most of the "
                           "in-sample edge does not survive out of sample. " + "; ".join(notes), d)
    return StageResult(3, PASS, f"Out-of-sample mean ${w['oos_mean']:.2f}/session, WFE {wfe:.2f}. " + "; ".join(notes), d)


def stage_permute(ctx: FamilyContext) -> StageResult:
    from nqlab.validation.mcpt import run_mcpt

    p = ctx.prereg
    v = ctx.settings.section("validation")["mcpt"]
    n_perm = int(ctx.n_perm or v["n_permutations"])
    res = run_mcpt(ctx.md, p["strategy"], p["param_space"], p["fixed"], tuple(p.get("categorical", [])), ctx.settings,
                   n_perm=n_perm, fill_tier=2, checkpoint=ctx.mcpt_checkpoint, save_checkpoint=ctx.save_checkpoint,
                   checkpoint_every=int(v["checkpoint_every"]), n_jobs=ctx.n_jobs, bucket_minutes=int(v["tod_bucket_minutes"]))
    from nqlab.results.ledger import LedgerEntry

    ctx.ledger.append([LedgerEntry(f"perm:{p['family_id']}:{n_perm}", "permutation", "completed",
                                   family_id=p["family_id"], strategy=p["strategy"],
                                   data_version=ctx.md.data_version, note=f"{n_perm} permutations")])
    d = {"real": res.real, "null": res.null, "p_values": res.p_values, "n_perm": n_perm,
         "percentile": {k: res.percentile(k) for k in res.real}}
    if n_perm < ctx.min_perm:
        return StageResult(4, INSUFFICIENT, f"Only {n_perm} permutations; at least {ctx.min_perm} are needed for "
                           "p < 0.05 to be resolvable.", d)
    if not res.p < ctx.crit["mcpt_p_max"]:
        return StageResult(4, FAIL, f"Permutation p = {res.p:.3f}: the whole selection procedure does about as well "
                           "on structure-free shuffled bars.", d)
    return StageResult(4, PASS, f"Permutation p = {res.p:.3f} (re-optimized on every permutation).", d)


def stage_deflate(ctx: FamilyContext) -> StageResult:
    fm = ctx.matrix(2)
    i, cfg = ctx.selected()
    n_trials = ctx.ledger.n_trials()
    sharpes = [s for s, _ in ctx.ledger.trial_sharpes()]
    ds = dsr(fm.pnl[:, i], n_trials, sharpes)
    pb = pbo_cscv(fm.pnl, ctx.grid, int(ctx.settings.section("validation")["pbo"]["n_blocks"]))
    sp_cfg = ctx.settings.section("validation")["spa"]
    sp = spa_test(fm.pnl, reps=int(sp_cfg["reps"]), block_size=int(sp_cfg["block_size"]))
    r = run_backtest(ctx.md, ctx.prereg["strategy"], cfg, ctx.settings, fill_tier=2)
    m = summarize(r.trades, r.daily)
    d = {"dsr": ds, "pbo": {k: v for k, v in pb.items() if k != "logits"}, "pbo_logits": pb.get("logits", []),
         "spa": sp, "t_stat_r": m["t_stat_r"], "n_trials_ledger": n_trials,
         "n_trials_family": ctx.ledger.n_trials(ctx.prereg["family_id"])}
    if not np.isfinite(ds["dsr"]):
        return StageResult(5, INSUFFICIENT, "Deflated Sharpe could not be computed (too little data).", d)
    if ds["dsr"] < ctx.crit["dsr_min"]:
        return StageResult(5, FAIL, f"Deflated Sharpe {ds['dsr']:.2f} < {ctx.crit['dsr_min']} after {n_trials} trials "
                           "in the ledger: the result is consistent with the best of many lucky tries.", d)
    if not pb["pbo"] < ctx.crit["pbo_max"]:
        worse = " Above 0.5 the selection process is worse than picking a config at random." if pb["pbo"] > 0.5 else ""
        return StageResult(5, FAIL, f"Probability of backtest overfitting {pb['pbo']:.2f} >= {ctx.crit['pbo_max']}.{worse}", d)
    t = m["t_stat_r"]
    if not np.isfinite(t):        # no stop -> R undefined: use the t-stat of daily P&L instead
        dp = fm.pnl[:, i]
        t = float(dp.mean() / (dp.std(ddof=1) / math.sqrt(len(dp)))) if len(dp) > 1 and dp.std(ddof=1) > 0 else float("nan")
        d["t_stat_basis"] = "daily_pnl"
    if not (np.isfinite(t) and t >= ctx.crit["family_t_stat"]):
        return StageResult(5, INSUFFICIENT, f"t-statistic {t:.2f} is below the family bar of {ctx.crit['family_t_stat']} "
                           "(Harvey-Liu-Zhu); more data is needed before this can be called an edge.", d)
    return StageResult(5, PASS, f"DSR {ds['dsr']:.3f} with N={n_trials}, PBO {pb['pbo']:.2f}, t={t:.2f}.", d)


def stage_propsim(ctx: FamilyContext) -> StageResult:
    from nqlab.risk.topstep import (AccountRules, SizingPolicy, TradeTape, bootstrap_combines, historical_combines,
                                    optimal_risk_search, zero_edge_benchmark)
    from nqlab.strategies.registry import get_strategy

    _, cfg = ctx.selected()
    strat = get_strategy(ctx.prereg["strategy"])
    r = run_backtest(ctx.md, ctx.prereg["strategy"], cfg, ctx.settings, fill_tier=2)
    sz = strat.sizing(cfg)
    spec = ctx.settings.contract(sz.instrument)
    tape = TradeTape.from_backtest(r.trades, r.daily, spec.point_value, spec.tick_value,
                                   ctx.settings.round_turn_cost(sz.instrument))
    rules = AccountRules.from_config(ctx.settings.topstep_rules, sz.instrument)
    policy = strat.topstep_policy(cfg)
    ts = ctx.settings.section("topstep")
    n_paths = int(ctx.settings.section("validation")["bootstrap"]["n_paths"])
    hist = historical_combines(tape, rules, policy, ts["combine_horizon_sessions"], ts["xfa_horizon_sessions"])
    boot = bootstrap_combines(tape, rules, policy, n_paths=n_paths, horizon=ts["combine_horizon_sessions"],
                              xfa_horizon=ts["xfa_horizon_sessions"])
    zero = zero_edge_benchmark(tape, rules, policy, n_paths=min(n_paths, 4000), horizon=ts["combine_horizon_sessions"])
    mspec = ctx.settings.contract("MNQ")
    mtape = TradeTape(tape.day_ptr, tape.pnl_pts, tape.mae_pts, tape.stop_pts, ctx.settings.round_turn_cost("MNQ"),
                      mspec.point_value, mspec.tick_value, tape.session_dates)
    risk = optimal_risk_search(mtape, AccountRules.from_config(ctx.settings.topstep_rules, "MNQ"),
                               n_paths=min(n_paths, 1000), horizon=ts["combine_horizon_sessions"])
    pfp = boot.p_pass * (boot.p_first_payout_given_pass if boot.p_pass > 0 else 0)
    zfp = zero.p_pass * (zero.p_first_payout_given_pass if zero.p_pass > 0 else 0)
    d = {"policy": policy.label(), "historical": hist.as_dict(), "bootstrap": boot.as_dict(), "zero_edge": zero.as_dict(),
         "p_first_payout": pfp, "zero_edge_p_first_payout": zfp, "risk_search": risk.to_dicts(),
         "max_consec_losses_q": np.quantile(boot.max_consec_losses, [0.5, 0.9, 0.99]).tolist() if len(boot.max_consec_losses) else [],
         "sessions_to_pass": boot.sessions_to_pass.tolist()[:5000]}
    if hist.n_paths < 60:
        return StageResult(6, INSUFFICIENT, "Too few sessions for a meaningful Combine simulation.", d)
    if not pfp > zfp + 0.05:
        return StageResult(6, FAIL, f"P(first payout) {pfp:.1%} is not clearly better than the same trades with the edge "
                           f"removed ({zfp:.1%}).", d)
    return StageResult(6, PASS, f"P(first payout) {pfp:.1%} vs {zfp:.1%} for a zero-edge trader with the same sizing.", d)


STAGES: dict[int, Callable[[FamilyContext], StageResult]] = {
    1: stage_screen, 2: stage_plateau, 3: stage_walkforward, 4: stage_permute, 5: stage_deflate, 6: stage_propsim,
}


def run_through(ctx: FamilyContext, last_stage: int = 6) -> list[StageResult]:
    """Run stages in order until one does not PASS (used by the null calibration harness)."""
    out = []
    for s in range(1, last_stage + 1):
        r = STAGES[s](ctx)
        out.append(r)
        if r.status != PASS:
            break
    return out
