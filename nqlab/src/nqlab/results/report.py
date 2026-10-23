"""Self-contained single-file HTML report (Jinja2 + Plotly inline).

Opens with a verdict block: PASS / FAIL / INSUFFICIENT DATA against pre-registered
criteria, with the reason in one plain sentence."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from nqlab import __version__
from nqlab.engine.backtest import run_backtest
from nqlab.results.metrics import SAMPLE_SIZE_TABLE, summarize
from nqlab.selection.funnel import FAIL, INSUFFICIENT, PASS, STAGE_NAMES

FILL_LINE = "This strategy's profit comes from the fill assumption, not from the market."
LLM_NOISE_LINE = "An LLM will always find a pattern, including in pure noise. This is a hypothesis, not a finding."
_TEMPLATES = Path(__file__).parent / "templates"


def _fmt(x, kind="num", nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if kind == "usd":
        sign = "-" if x < 0 else ""
        return f"{sign}${abs(x):,.0f}" if abs(x) >= 100 else f"{sign}${abs(x):,.2f}"
    if kind == "pct":
        return f"{x:.1%}"
    if kind == "int":
        return f"{int(x):,}"
    return f"{x:.{nd}f}"


def _fig(fig: go.Figure, title: str, caption: str = "") -> dict:
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=50, r=20, t=30, b=40),
                      font=dict(family="system-ui, sans-serif", size=12), legend=dict(orientation="h", y=1.1))
    return {"title": title, "caption": caption, "html": fig.to_html(full_html=False, include_plotlyjs=False)}


def family_verdict(stage_rows: list[dict]) -> tuple[str, str]:
    by_stage: dict[int, dict] = {}
    for r in stage_rows:
        by_stage[r["stage"]] = r
    for s in sorted(by_stage):
        r = by_stage[s]
        if r["status"] == FAIL:
            return FAIL, f"Stage {s} ({STAGE_NAMES[s]}): {r['summary']}"
    for s in sorted(by_stage):
        r = by_stage[s]
        if r["status"] != PASS:
            return INSUFFICIENT, f"Stage {s} ({STAGE_NAMES[s]}): {r['summary']}"
    done = max(by_stage) if by_stage else 0
    if done >= 6:
        return PASS, ("Passed every pre-registered gate through the Topstep simulation"
                      + (" and the holdout." if done >= 7 else "; the holdout has not been looked at yet."))
    return INSUFFICIENT, (f"Passed stages 1-{done} only; not validated until it clears stage 6." if done
                          else "Not yet run through the selection funnel.")


def build_report(settings, md, strategy: str, params: dict[str, Any], out_path: Path, *, n_trials: int,
                 stage_rows: list[dict] | None = None, family_id: str | None = None, run_diag: bool = True,
                 propsim: bool = True, n_paths: int | None = None, save: bool = True,
                 grid_keys: list[str] | None = None) -> Path:
    from nqlab.risk.topstep import (ZERO_EDGE_LINE, AccountRules, SizingPolicy, TradeTape, bootstrap_combines,
                                    combine_trace, historical_combines, optimal_risk_search, zero_edge_benchmark)
    from nqlab.strategies.registry import get_strategy
    from nqlab.validation.diagnostics import fill_optimism

    strat = get_strategy(strategy)
    r2 = run_backtest(md, strategy, params, settings, fill_tier=2)
    r1 = run_backtest(md, strategy, params, settings, fill_tier=1)
    m2, m1 = summarize(r2.trades, r2.daily), summarize(r1.trades, r1.daily)
    crit = settings.section("validation")["gates"]
    warnings, notes, figures = [], [], []

    # ---- verdict -------------------------------------------------------------------------
    if stage_rows is not None:
        status, reason = family_verdict(stage_rows)
    elif m2["n_trades"] < crit["min_trades"]:
        status, reason = INSUFFICIENT, (f"Only {m2['n_trades']} trades; at least {crit['min_trades']} are needed before "
                                        "anything can be said.")
    elif not m2["expectancy_usd"] > 0:
        status, reason = FAIL, "It loses money after costs under trade-through (tier 2) fills."
    else:
        status, reason = INSUFFICIENT, ("A single backtest is not validation: this configuration has not been through "
                                        "the pre-registered selection funnel (nqlab preregister / run).")
    verdict = {"status": status, "reason": reason, "css": status.split()[0]}

    # ---- fill tiers ------------------------------------------------------------------------
    def tier_row(label, m):
        return {"label": label, "n": m["n_trades"], "net": _fmt(m["net_usd"], "usd"), "exp": _fmt(m["expectancy_usd"], "usd"),
                "exp_r": _fmt(m["expectancy_r"]), "sharpe": f"{_fmt(m['sharpe_ann'])} ± {_fmt(m['sharpe_ann_se'])}"}
    tiers = [tier_row("Tier 2 trade-through (HEADLINE)", m2), tier_row("Tier 1 touch (optimistic upper bound only)", m1)]
    fill_line = FILL_LINE if (m1["net_usd"] > 0 and not m2["net_usd"] > 0) else ""
    try:   # tiers 3/4 whenever tick data covers part of the period (Databento)
        from nqlab.data.store import DataStore

        ds = DataStore(settings)
        if ds.has_ticks("trades"):
            from nqlab.engine.fill_calibration import compare_tiers

            mbo = ds.scan_ticks("mbo").collect() if ds.has_ticks("mbo") else None
            ct = compare_tiers(md, strategy, params, settings, ds.scan_ticks("trades").collect(), mbo, n_sessions=60)
            for k, v in sorted(ct["tiers"].items()):
                tiers.append({"label": f"{v['label']} [{ct['n_sessions']} sampled sessions, 09:25-10:45 ET only]",
                              "n": v["n_trades"], "net": _fmt(v["net_usd"], "usd"), "exp": _fmt(v["expectancy_usd"], "usd"),
                              "exp_r": "n/a", "sharpe": "n/a"})
    except Exception as exc:  # pragma: no cover - tick data is optional
        notes.append(f"Tick-data fill tiers unavailable: {type(exc).__name__}: {exc}")
    diag = None
    if run_diag:
        fo = fill_optimism(md, strategy, params, settings, n_seeds=2)
        diag = {"t1": _fmt(fo["tiers"][1]["net_usd"], "usd"), "t2": _fmt(fo["tiers"][2]["net_usd"], "usd"),
                "warning": fo["warning"]}
        if fo["warning"]:
            warnings.append("Fill-optimism diagnostic: on a pure random walk this strategy makes money: " + fo["warning"])
    if m2["ambiguous_frac"] > settings.section("engine")["ambiguous_warn_fraction"]:
        warnings.append(f"{m2['ambiguous_frac']:.0%} of trades were resolved by the ambiguous-bar rule (stop assumed first). "
                        "Above ~15% the result is not trustworthy at 1-minute resolution; it needs tick data.")
    else:
        notes.append(f"{m2['ambiguous_frac']:.1%} of trades resolved by the ambiguous-bar rule (stop first).")

    # ---- equity --------------------------------------------------------------------------------
    d2, d1 = r2.daily, r1.daily
    eq = go.Figure()
    eq.add_scatter(x=d2["session_date"].to_list(), y=np.cumsum(d2["pnl_usd"].to_numpy()), name="tier 2 (headline)")
    eq.add_scatter(x=d1["session_date"].to_list(), y=np.cumsum(d1["pnl_usd"].to_numpy()), name="tier 1 (optimistic)",
                   line=dict(dash="dot"))
    figures.append(_fig(eq, "Equity curve (USD, after costs)", f"Sharpe {_fmt(m2['sharpe_ann'])} ± {_fmt(m2['sharpe_ann_se'])} (SE)."))

    # ---- topstep ------------------------------------------------------------------------------
    sz = strat.sizing(params)
    spec = settings.contract(sz.instrument)
    tape = TradeTape.from_backtest(r2.trades, r2.daily, spec.point_value, spec.tick_value, settings.round_turn_cost(sz.instrument))
    rules_cfg = settings.topstep_rules
    rules = AccountRules.from_config(rules_cfg, sz.instrument)
    policy = strat.topstep_policy(params)
    trace = combine_trace(tape, rules, policy)
    topstep_rows, kpi_pass, kpi_payout = [], None, None
    if trace.height:
        f = go.Figure()
        f.add_scatter(x=trace["session_date"].to_list(), y=trace["balance"].to_list(), name="balance (sequential Combines)")
        f.add_scatter(x=trace["session_date"].to_list(), y=trace["mll_level"].to_list(), name="MLL floor", line=dict(dash="dash"))
        ev = trace.filter(pl.col("rule_events") != "")
        f.add_scatter(x=ev["session_date"].to_list(), y=ev["balance"].to_list(), mode="markers" + ("+text" if ev.height <= 15 else ""),
                      name="events (hover)", text=ev["rule_events"].to_list(), textposition="top center",
                      hovertemplate="%{text}<extra></extra>")
        figures.append(_fig(f, "Account balance with the trailing MLL floor", "A new Combine starts the session after each pass or failure."))
    if propsim and tape.n_days > 20:
        ts = settings.section("topstep")
        npth = int(n_paths or settings.section("validation")["bootstrap"]["n_paths"])
        boot = bootstrap_combines(tape, rules, policy, n_paths=npth, horizon=ts["combine_horizon_sessions"], xfa_horizon=ts["xfa_horizon_sessions"])
        zero = zero_edge_benchmark(tape, rules, policy, n_paths=min(npth, 4000), horizon=ts["combine_horizon_sessions"])
        hist = historical_combines(tape, rules, policy, ts["combine_horizon_sessions"], ts["xfa_horizon_sessions"])
        fp = lambda s: s.p_pass * (s.p_first_payout_given_pass if s.p_pass > 0 else 0)  # noqa: E731
        kpi_pass, kpi_payout = boot.p_pass, fp(boot)
        for k, fn in (("P(pass)", lambda s: _fmt(s.p_pass, "pct")), ("P(fail by MLL)", lambda s: _fmt(s.p_fail_mll, "pct")),
                      ("P(fail after consistency block)", lambda s: _fmt(s.p_fail_consistency, "pct")),
                      ("P(unresolved at horizon / end of data)", lambda s: _fmt(s.p_unresolved, "pct")),
                      ("Median sessions to pass", lambda s: _fmt(s.median_sessions_to_pass, "num", 0)),
                      ("P(first payout)", lambda s: _fmt(fp(s), "pct")),
                      ("P(first payout | passed)", lambda s: _fmt(s.p_first_payout_given_pass, "pct")),
                      ("Expected payouts before MLL breach", lambda s: _fmt(s.expected_payouts))):
            topstep_rows.append({"k": k, "a": fn(boot), "b": fn(zero), "c": fn(hist)})
        if len(boot.sessions_to_pass):
            h = go.Figure(go.Histogram(x=boot.sessions_to_pass, nbinsx=40))
            figures.append(_fig(h, "Sessions to pass (bootstrap paths that passed)"))
        if len(boot.max_consec_losses):
            avg_loss = -float(np.mean(tape.pnl_pts[tape.pnl_pts < 0]) * tape.point_value) if (tape.pnl_pts < 0).any() else 0
            h = go.Figure(go.Histogram(x=boot.max_consec_losses * avg_loss, nbinsx=30, name="max losing streak x avg loss"))
            h.add_vline(x=rules.mll_amount, line_dash="dash", annotation_text="MLL buffer $2,000")
            figures.append(_fig(h, "Worst losing streak (in $) vs the MLL buffer", "Approximation: streak length x average loss."))
        # sweep in MNQ so small risk fractions are expressible (10 MNQ = 1 NQ of exposure)
        mspec = settings.contract("MNQ")
        mtape = TradeTape(tape.day_ptr, tape.pnl_pts, tape.mae_pts, tape.stop_pts, settings.round_turn_cost("MNQ"),
                          mspec.point_value, mspec.tick_value, tape.session_dates)
        rs = optimal_risk_search(mtape, AccountRules.from_config(rules_cfg, "MNQ"), n_paths=min(npth, 1000),
                                 horizon=ts["combine_horizon_sessions"])
        f = go.Figure()
        f.add_scatter(x=rs["risk_frac_of_buffer"].to_list(), y=rs["p_first_payout"].to_list(), name="P(first payout)")
        f.add_scatter(x=rs["risk_frac_of_buffer"].to_list(), y=(rs["expected_paid_usd"] / max(rs["expected_paid_usd"].max(), 1e-9)).to_list(),
                      name="expected payout $ (scaled)", line=dict(dash="dot"))
        f.update_xaxes(tickformat=".0%", title="risk per trade as a share of the current MLL buffer")
        best_p = rs.row(int(np.argmax(rs["p_first_payout"].to_numpy())), named=True)
        best_e = rs.row(int(np.argmax(rs["expected_paid_usd"].to_numpy())), named=True)
        figures.append(_fig(f, "Sizing: P(first payout) vs risk per trade",
                            f"P(first payout) peaks at {best_p['risk_frac_of_buffer']:.0%} of buffer; expected payout $ peaks at "
                            f"{best_e['risk_frac_of_buffer']:.0%}. Size for the first, not the second. Effective capital is the "
                            "$2,000 distance to the MLL, not $50,000. Simulated in MNQ for sizing granularity; trades without a stop "
                            "cannot be risk-sized and are skipped."))

    # ---- account farming economics ---------------------------------------------------------
    farm_rows, farm_best, kpi_ev = [], "", None
    if propsim and tape.n_days > 20:
        from nqlab.risk.farming import farm_report, optimize_farming

        fr = farm_report(tape, settings, policy, sz.instrument, n_paths=int(min(npth, 2000)))
        for path, v in fr["paths"].items():
            for label, key in (("this strategy", "strategy"), ("same trades, zero edge", "zero_edge"),
                               ("edge cut to its 1-SE lower bound", "haircut")):
                st = v[key]
                farm_rows.append({"path": path, "who": label, "ev": _fmt(st.ev_net, "usd"), "p_pos": _fmt(st.p_net_positive, "pct"),
                                  "p_funded": _fmt(st.p_funded, "pct"), "p_pay": _fmt(st.p_any_payout, "pct"),
                                  "cost": _fmt(st.mean_cost, "usd"), "paid": _fmt(st.mean_paid, "usd"),
                                  "months": _fmt(st.mean_months, "num", 1), "per_month": _fmt(st.ev_per_month, "usd")})
        bp = fr["best_path"]
        kpi_ev = fr["paths"][bp]["strategy"].ev_net
        zev = fr["paths"][bp]["zero_edge"].ev_net
        if zev > 0:
            warnings.append(f"The same trades with ZERO edge still show {_fmt(zev, 'usd')} per account: losses are capped at "
                            "the fees while payouts are not, so variance alone has value under these rules. Only the part "
                            "above the zero-edge line is your edge; the rest is the payout structure, and it depends on rules "
                            "nqlab could not verify (payout %, split, scaling).")
        h = go.Figure()
        for path, v in fr["paths"].items():
            h.add_histogram(x=v["strategy"].nets, name=f"{path} path", opacity=.6, nbinsx=50)
        h.add_vline(x=0, line_dash="dash")
        figures.append(_fig(h, "Net $ per purchased account (distribution over simulated lifecycles)",
                            "Left of zero: the account cost more in fees than it paid out."))
        opt = optimize_farming(tape, rules_cfg, settings, fracs=[0.05, 0.10, 0.20, 0.30, 0.40], n_paths=300,
                               combine_budget=ts["combine_horizon_sessions"], xfa_horizon=ts["xfa_horizon_sessions"])
        b, bu = opt["best"], opt["best_unhaircut"]
        g = opt["grid"].filter(pl.col("path") == b["path"]).sort("combine_risk", "xfa_risk")
        xs = sorted(set(g["xfa_risk"].to_list()))
        ys = sorted(set(g["combine_risk"].to_list()))
        z = [[g.filter((pl.col("combine_risk") == y) & (pl.col("xfa_risk") == x))["ev_net_haircut"][0] for x in xs] for y in ys]
        f = go.Figure(go.Heatmap(z=z, x=[f"{x:.0%}" for x in xs], y=[f"{y:.0%}" for y in ys], colorscale="RdBu", zmid=0,
                                 colorbar=dict(title="$")))
        f.update_xaxes(title="funded-account risk per trade (share of MLL buffer)")
        f.update_yaxes(title="Combine risk per trade (share of MLL buffer)")
        farm_best = (f"Best sizing on the {b['path']} path, judged with the edge haircut: {b['combine_risk']:.0%} of the MLL "
                     f"buffer per trade in the Combine and {b['xfa_risk']:.0%} in the funded account (MNQ), expected net "
                     f"{_fmt(b['ev_net_haircut'], 'usd')} per account haircut / {_fmt(b['ev_net'], 'usd')} as measured. "
                     f"Sizing on the measured edge alone would pick {bu['combine_risk']:.0%} / {bu['xfa_risk']:.0%}.")
        if b["combine_risk"] == max(opt["grid"]["combine_risk"]) or b["xfa_risk"] == max(opt["grid"]["xfa_risk"]):
            farm_best += (" The best cell is at the edge of the grid (largest size tested): bigger may score higher in "
                          "simulation, but every extra contract leans harder on the edge being real.")
        figures.append(_fig(f, f"Farming sizing grid ({b['path']} path): expected net $ per account, edge haircut", farm_best))

    # ---- family-stage figures --------------------------------------------------------------
    wf_rows = []
    for row in stage_rows or []:
        d = row["details"]
        if row["stage"] == 4 and d.get("null"):
            for k, label in (("wf_oos_mean", "walk-forward OOS mean $/session"), ("is_plateau", "best in-sample plateau score")):
                null = [x for x in d["null"][k] if x is not None and np.isfinite(x)]
                h = go.Figure(go.Histogram(x=null, nbinsx=40, name="permuted"))
                h.add_vline(x=d["real"][k], line_color="red", annotation_text="real")
                figures.append(_fig(h, f"Permutation null: {label}",
                                    f"p = {_fmt(d['p_values'][k], 'num', 3)}; the real result beats {_fmt(d['percentile'][k], 'pct')} of "
                                    f"{d['n_perm']} permutations (selection re-run on each)."))
        if row["stage"] == 2 and d.get("heatmap"):
            hm = d["heatmap"]
            for key, label in (("raw", "raw Sharpe per config"), ("plateau", "plateau score (median of neighborhood)")):
                f = go.Figure(go.Heatmap(z=hm[key], x=[str(x) for x in hm["xs"]], y=[str(y) for y in hm["ys"]], colorscale="RdBu",
                                         zmid=0))
                f.update_xaxes(title=hm["x"]); f.update_yaxes(title=hm["y"])
                figures.append(_fig(f, f"Parameter heatmap: {label}", "A lone bright cell surrounded by dark ones is a spike (noise)."))
        if row["stage"] == 3:
            for w in d.get("rolling", {}).get("windows", []):
                wf_rows.append({"train": f"{w['train'][0]}-{w['train'][1]}", "test": f"{w['test'][0]}-{w['test'][1]}",
                                "params": json.dumps({k: v for k, v in w["params"].items() if not grid_keys or k in grid_keys}),
                                "is_mean": _fmt(w["is_mean"], "usd"), "oos_mean": _fmt(w["oos_mean"], "usd")})

    # ---- time of day / MAE-MFE ----------------------------------------------------------------
    t = r2.trades
    if t.height:
        tod = t.with_columns(((pl.col("entry_ct_minute") + 60) // 15 * 15).alias("b")).group_by("b").agg(
            pl.len().alias("n"), pl.col("pnl_usd").mean().alias("mean")).sort("b")
        lab = [f"{int(b) // 60:02d}:{int(b) % 60:02d} ET" for b in tod["b"].to_list()]
        f = go.Figure()
        f.add_bar(x=lab, y=tod["mean"].to_list(), name="mean $/trade")
        f.add_scatter(x=lab, y=tod["n"].to_list(), name="trades", yaxis="y2", mode="lines+markers")
        f.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
        figures.append(_fig(f, "Time-of-day entry breakdown (15-minute buckets)"))
        f = go.Figure(go.Scatter(x=t["mae_points"].to_list(), y=t["mfe_points"].to_list(), mode="markers",
                                 marker=dict(color=np.where(t["pnl_usd"].to_numpy() > 0, "#1f7a3a", "#b3261e"), size=6, opacity=.6)))
        f.update_xaxes(title="MAE (points)"); f.update_yaxes(title="MFE (points)")
        figures.append(_fig(f, "MAE / MFE per trade (green = winner)"))

    # ---- Monte Carlo / synthetic nulls (spec 8.4) ---------------------------------------
    from nqlab.validation.bootstrap import block_bootstrap_sharpe, reshuffle_drawdowns, trade_bootstrap

    mc_rows = []
    tp = t["pnl_usd"].to_numpy() if t.height else np.array([])
    dp = d2["pnl_usd"].to_numpy()
    rs_dd = reshuffle_drawdowns(tp, n=2000)
    mc_rows.append({"k": "Trade-order reshuffle: max drawdown p50 / p95 (NOT a test of edge)",
                    "v": f"{_fmt(rs_dd['p50'], 'usd')} / {_fmt(rs_dd['p95'], 'usd')} (realized {_fmt(rs_dd.get('realized'), 'usd')})"})
    tb = trade_bootstrap(tp, n=5000)
    mc_rows.append({"k": "Trade bootstrap: 95% CI of expectancy per trade",
                    "v": f"{_fmt(tb['exp_ci'][0], 'usd')} to {_fmt(tb['exp_ci'][1], 'usd')}"})
    mc_rows.append({"k": "Trade bootstrap: P(final P&L < 0)", "v": _fmt(tb["p_final_negative"], "pct")})
    for kind in ("stationary", "circular"):
        bb = block_bootstrap_sharpe(dp, n=1000, kind=kind)
        mc_rows.append({"k": f"{kind.title()} block bootstrap: 95% CI of annualized Sharpe",
                        "v": f"{_fmt(bb['sharpe_ci'][0])} to {_fmt(bb['sharpe_ci'][1])}"})
    if run_diag:
        try:
            from nqlab.validation.synthetic import garch_synthetic

            g = run_backtest(garch_synthetic(md, 5), strategy, params, settings, fill_tier=2)
            mg = summarize(g.trades, g.daily)
            mc_rows.append({"k": "GJR-GARCH(1,1)-t null (deseasonalized): net on synthetic data, tier 2",
                            "v": f"{_fmt(mg['net_usd'], 'usd')} over {mg['n_trades']} trades"})
            if mg["n_trades"] and mg["net_usd"] > 0:
                warnings.append("The strategy makes money on GARCH-simulated data (volatility clustering, no "
                                "directional structure). Part of the edge may be a volatility artefact.")
        except Exception as exc:  # pragma: no cover - GARCH fit can fail on tiny samples
            mc_rows.append({"k": "GJR-GARCH null", "v": f"not available ({type(exc).__name__})"})

    # ---- sample size -----------------------------------------------------------------------
    e = m2["expectancy_r"]
    here = None
    if np.isfinite(e) and e > 0:
        here = min(range(len(SAMPLE_SIZE_TABLE)), key=lambda i: abs(SAMPLE_SIZE_TABLE[i][0] - e))
    ss = {"exp_r": _fmt(e), "n": m2["n_trades"], "sd_r": _fmt(m2["sd_r"]),
          "req2": _fmt(m2["required_n_t2"], "int") if np.isfinite(m2["required_n_t2"]) else "infinite (no positive edge)",
          "req3": _fmt(m2["required_n_t3"], "int") if np.isfinite(m2["required_n_t3"]) else "infinite (no positive edge)",
          "table": [{"e": f"{a:.2f}R", "n2": f"~{b:,}", "n3": f"~{c:,}", "here": i == here} for i, (a, b, c) in enumerate(SAMPLE_SIZE_TABLE)]}

    if r2.meta.get("n_sized_out"):
        warnings.append(f"{r2.meta['n_sized_out']} signal(s) were skipped because the risk budget could not buy one "
                        f"contract at that stop width. At 1 contract they would have made "
                        f"{_fmt(r2.meta['sized_out_pnl_1c'], 'usd')} after costs. Skipping wide-stop trades changes the "
                        "result; check that this is really your rule.")

    # ---- micros warning ---------------------------------------------------------------------
    if sz.instrument == "MNQ" and t.height and int(t["contracts"].max()) > 10:
        nmax = int(t["contracts"].max())
        mini_rt, micro_rt = settings.round_turn_cost("NQ"), settings.round_turn_cost("MNQ")
        over = t.filter(pl.col("contracts") > 10)
        pen = float((over["contracts"] * micro_rt - over["contracts"] / 10 * mini_rt).sum())
        warnings.append(f"Uses up to {nmax} MNQ (more than 10 micros = more than 1 NQ of exposure). Commission penalty vs "
                        f"trading minis: ${pen:,.2f} over {over.height} trades. 10 MNQ cost ${10 * micro_rt:.2f} per round "
                        f"turn vs ${mini_rt:.2f} for 1 NQ.")

    # ---- assumptions --------------------------------------------------------------------------
    rv = rules_cfg.get("rules_version", "?")
    cons = rules_cfg["combine"]["consistency"]["best_day_max_pct"]
    assumptions = [
        f"Data version {md.data_version}; provenance: {md.provenance}; roll method: {md.meta.get('roll_method')}.",
        f"{len(r2.excluded_sessions)} session(s) excluded by data-quality checks (never traded, listed in the quality report).",
        "Costs (round turn): " + ", ".join(f"{k} ${v}" for k, v in settings.section("costs")["round_turn"].items())
        + ". Stops/markets pay time-of-day slippage; limits pay none but must trade through (tier 2).",
        "Ambiguous bars (stop and target both touched) resolve to the stop.",
        f"10 MNQ cost ${10 * settings.round_turn_cost('MNQ'):.2f} per round turn versus ${settings.round_turn_cost('NQ'):.2f} "
        "for 1 NQ, for identical exposure.",
        f"Topstep rules {rv}; consistency target {cons:.0%}. Not shown on the Topstep page and still unverified: "
        "payout split, 50%-of-balance payout limit, MLL lock level, XFA scaling.",
        "Release calendar is transcribed, not verified; NFP/ISM dates are rule-based approximations.",
    ]
    assumptions += [f"Data warning: {w}" for w in md.warnings]
    if params.get("volume_profile.build_from") == "bars" or "vp" in strategy:
        assumptions.append("Volume profile built from 1-minute bars is APPROXIMATE (volume spread over each bar's range).")

    kpis = [
        {"k": "Expected net $ per purchased account, after all fees (primary)", "v": _fmt(kpi_ev, "usd")},
        {"k": "Sharpe (annualized) ± SE", "v": f"{_fmt(m2['sharpe_ann'])} ± {_fmt(m2['sharpe_ann_se'])}"},
        {"k": "Expectancy per trade", "v": f"{_fmt(m2['expectancy_r'])} R / {_fmt(m2['expectancy_usd'], 'usd')}"},
        {"k": "Profit factor", "v": _fmt(m2["profit_factor"])},
        {"k": "Trades / sessions", "v": f"{m2['n_trades']} / {m2['n_sessions']}"},
        {"k": "Max consecutive losses", "v": str(m2["max_consec_losses"])},
        {"k": "Max drawdown", "v": _fmt(m2["max_drawdown_usd"], "usd")},
        {"k": "P(pass Combine)", "v": _fmt(kpi_pass, "pct")},
        {"k": "P(first payout) (primary rank)", "v": _fmt(kpi_payout, "pct")},
    ]
    dsr_row = next((r for r in (stage_rows or []) if r["stage"] == 5), None)
    if dsr_row:
        kpis.insert(1, {"k": "Deflated Sharpe (N from ledger)", "v": _fmt(dsr_row["details"].get("dsr", {}).get("dsr"))})

    stages = [{"stage": r["stage"], "name": STAGE_NAMES[r["stage"]], "status": r["status"], "summary": r["summary"]}
              for r in (stage_rows or [])]
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES)), autoescape=select_autoescape(["html", "j2"]))
    from plotly.offline import get_plotlyjs

    html = env.get_template("report.html.j2").render(
        title=f"{family_id or strategy}: verdict", subtitle=f"{strategy} · {json.dumps(params)[:300]}",
        verdict=verdict, n_trials=n_trials, warnings=warnings, notes=notes, kpis=kpis, stages=stages, tiers=tiers,
        fill_line=fill_line, diag=diag, figures=figures, wf_rows=wf_rows, ss=ss, topstep=topstep_rows, mc_rows=mc_rows,
        farm_rows=farm_rows, farm_best=farm_best,
        zero_edge_line=ZERO_EDGE_LINE + " (With the trailing MLL simulated here the zero-edge rate is lower still; see the "
        "middle column, which uses this strategy's own trade sizes with the edge removed.)",
        sizing_label=policy.label(), rules_version=rv, rules_note=rules_cfg.get("verify_note", ""),
        assumptions=assumptions, generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        version=__version__, run_id=r2.run_id, data_version=md.data_version, plotly_js=get_plotlyjs())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    if save:
        from nqlab.results.store import save_run

        acct = trace.rename({"pnl_usd": "acct_pnl"}) if trace.height else None
        save_run(settings, r2, {**m2, "verdict": status, "p_pass": kpi_pass, "p_first_payout": kpi_payout},
                 account_daily=acct, family_id=family_id)
        save_run(settings, r1, m1, family_id=family_id)
    return out_path

