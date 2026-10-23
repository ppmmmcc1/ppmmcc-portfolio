"""nqlab command line. ``python -m nqlab.cli ...`` or ``nqlab ...``."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from nqlab.log import setup_logging
from nqlab.settings import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Statistically honest NQ/MNQ backtesting.")
data_app = typer.Typer(no_args_is_help=True, help="Ingest and inspect market data.")
spec_app = typer.Typer(no_args_is_help=True, help="Strategy spec files.")
queue_app = typer.Typer(no_args_is_help=True, help="Overnight job queue.")
ledger_app = typer.Typer(no_args_is_help=True, help="The trial ledger (DSR's N).")
parity_app = typer.Typer(no_args_is_help=True, help="Discretionary and live-fill parity.")
llm_app = typer.Typer(no_args_is_help=True, help="Ollama roles (text only; never numbers).")
app.add_typer(data_app, name="data")
app.add_typer(spec_app, name="spec")
app.add_typer(queue_app, name="queue")
app.add_typer(ledger_app, name="ledger")
app.add_typer(parity_app, name="parity")
app.add_typer(llm_app, name="llm")
con = Console()


def _settings():
    setup_logging()
    return get_settings()


def _die(msg: str, code: int = 1):
    con.print(f"[bold red]refused:[/bold red] {msg}")
    raise typer.Exit(code)


# ------------------------------------------------------------------------------- data
@data_app.command("ingest")
def data_ingest(source: Optional[str] = typer.Option(None, help="csv | yfinance | databento (overrides settings)"),
                confirm_cost: bool = typer.Option(False, help="Databento: allow downloads above max_cost_usd")):
    """Load raw data, normalize, build the roll calendar, run quality checks, lock the holdout."""
    from nqlab.data.ingest import ingest

    s = _settings()
    if source:
        s.raw["data"]["source"] = source
    raw = None
    notes: list[str] = []
    fp = None
    if s.section("data")["source"] == "databento":
        from nqlab.data.databento_source import DatabentoSource

        src = DatabentoSource(s, confirm_cost=confirm_cost)
        raw = src.fetch_bars()
        fp = ["databento", str(raw["ts_utc"].max())]
    r = ingest(s, raw=raw, source_notes=notes, source_fingerprint=fp)
    con.print(f"data_version [bold]{r.data_version}[/bold]: {r.n_bars:,} research bars, {r.n_sessions} sessions "
              f"(+{r.n_holdout_sessions} LOCKED in the holdout from {r.holdout_start}); {r.n_excluded} session(s) excluded; "
              f"roll method {r.roll_method}; provenance {r.provenance}")
    for w in r.report.warnings:
        con.print(f"[yellow]WARNING[/yellow] {w}")
    con.print(f"quality report: {s.path('reports') / f'quality_{r.data_version}.md'}")


@data_app.command("ingest-ticks")
def data_ingest_ticks(schema: str = typer.Option("trades", help="trades | mbo"), start: str = typer.Option(...),
                      end: str = typer.Option(...), confirm_cost: bool = typer.Option(False)):
    """Download Databento tick data (cost estimated and printed first) for fill tiers 3/4."""
    from nqlab.data.databento_source import DatabentoSource
    from nqlab.data.store import DataStore

    s = _settings()
    if schema not in ("trades", "mbo"):
        _die("schema must be trades or mbo")
    src = DatabentoSource(s, confirm_cost=confirm_cost)
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    df = src.fetch_trades(None, a, b) if schema == "trades" else src.fetch_mbo(a, b)
    store = DataStore(s)
    con.print(store.write_ticks(schema, df))
    if schema == "trades":
        import polars as pl

        from nqlab.data.quality import cross_check_trades

        man = store.manifest()
        bars = store.scan_front(man["root"]).filter(
            (pl.col("session_date") >= a) & (pl.col("session_date") <= b)).collect()
        res = cross_check_trades(bars, df)
        out = s.path("reports") / f"quality_trades_vs_bars_{man['data_version']}_{a}_{b}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        con.print(f"bars vs trades: {res['price_disagreements']} of {res['bars_compared']} bars disagree on price, "
                  f"{res['volume_disagreements']} on volume (details: {out})")


@data_app.command("quality")
def data_quality():
    """Print the latest data-quality report."""
    from nqlab.data.store import DataStore

    s = _settings()
    man = DataStore(s).manifest()
    p = s.path("reports") / f"quality_{man['data_version']}.md"
    con.print(p.read_text(encoding="utf-8") if p.exists() else json.dumps(man["quality"], indent=2))


@data_app.command("roll-calendar")
def data_roll_calendar():
    """Show the roll calendar used for the current data version."""
    from nqlab.data.store import DataStore

    s = _settings()
    man = DataStore(s).manifest()
    t = Table("roll session", "from", "to", "gap (pts)", "gap method", title=f"method: {man['roll_method']}")
    for r in man["rolls"]:
        t.add_row(str(r["roll_session"]), r["from"], r["to"], f"{r['gap']:.2f}", r["gap_method"])
    con.print(t if man["rolls"] else f"no measured rolls (method: {man['roll_method']})")


@data_app.command("events-import")
def data_events_import(csv: Path):
    """Import a VERIFIED release calendar (columns: date, kind; kind in fomc, cpi, ppi, nfp, retail_sales, ism_manufacturing)."""
    import polars as pl
    import yaml

    s = _settings()
    df = pl.read_csv(csv)
    p = s.path("events")
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    for kind, grp in df.group_by("kind"):
        k = kind[0]
        dates = sorted({str(d) for d in grp["date"].to_list()})
        if k == "fomc":
            doc["fomc"] = dates
        elif k in ("ism_manufacturing",):
            doc["release_1000"][k] = dates
        else:
            doc["release_0830"][k] = dates
    doc["verified"] = True
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    con.print(f"imported {df.height} events; marked verified. Re-run `nqlab data ingest` to refresh session flags.")


# ------------------------------------------------------------------------------- spec
@spec_app.command("lint")
def spec_lint(paths: list[Path], no_llm: bool = typer.Option(False, "--no-llm"),
              strict: bool = typer.Option(False, help="exit non-zero if any question remains (pre-commit)")):
    """List undefined, ambiguous or contradictory parameters as QUESTIONS."""
    from nqlab.llm.roles import lint_spec

    s = _settings()
    bad = 0
    for p in paths:
        r = lint_spec(s, p, use_llm=not no_llm)
        con.print(f"[bold]{p}[/bold]: {r['n_undecided']} undecided field(s); runnable: {r['runnable']}")
        for i, q in enumerate(r["questions"], 1):
            con.print(f"  {i:>2}. [{q['source']}] {q['field']}: {q['question']}")
        bad += 0 if r["runnable"] else 1
    if strict and bad:
        raise typer.Exit(1)


@spec_app.command("scaffold")
def spec_scaffold(name: str, template: str = "bb_vp_macd"):
    """Copy a spec template with every decision marked REQUIRED_UNDECIDED."""
    s = _settings()
    src = s.path("specs") / f"{template}.yaml"
    dst = s.path("specs") / f"{name}.yaml"
    if dst.exists():
        _die(f"{dst} exists")
    shutil.copy(src, dst)
    con.print(f"wrote {dst}; fill it in, then `nqlab spec lint {dst}`")


# ------------------------------------------------------------------------------- families
@app.command()
def preregister(family: str, strategy: Optional[str] = typer.Option(None),
                hypothesis: Optional[str] = typer.Option(None, help="one sentence"),
                from_draft: Optional[Path] = typer.Option(None, help="register an LLM-variant draft (keeps its reserved trial keys)"),
                spec: Optional[Path] = typer.Option(None, help="strategy spec YAML (operator strategy)"),
                space: Optional[str] = typer.Option(None, help="JSON param grid; default = the strategy's declared grid"),
                fixed: Optional[str] = typer.Option(None, help="JSON fixed params")):
    """Stage 0: register a strategy FAMILY with its full grid declared up front (hashed, locked)."""
    from nqlab.selection.family import FamilyStore
    from nqlab.selection.prereg import make_prereg

    s = _settings()
    sp = json.loads(space) if space else None
    fx = json.loads(fixed) if fixed else {}
    spec_hash = None
    origin, variant_id = "human", None
    if from_draft:
        import yaml

        dr = yaml.safe_load(Path(from_draft).read_text(encoding="utf-8"))
        if not dr.get("llm_variant_id"):
            _die("not an LLM-variant draft (no llm_variant_id)")
        if family != dr["family_id"]:
            _die(f"an LLM draft must be registered under its reserved id {dr['family_id']!r}")
        strategy, sp, fx = dr["strategy"], dr["param_space"], dr.get("fixed") or {}
        origin, variant_id = "llm_variant", dr["llm_variant_id"]
        hypothesis = hypothesis or f"LLM variant '{dr.get('name', '')}': {dr.get('rationale', '')}"
    if not hypothesis:
        _die("--hypothesis is required: state what you expect before you test it")
    if spec:
        from nqlab.strategies.spec import META_KEYS, SpecIncomplete, flatten, grid_dimensions, load_spec, spec_hash as sh, undecided

        doc = load_spec(spec)
        und = undecided(doc)
        if und:
            _die(str(SpecIncomplete(und)))
        strategy = f"{doc['strategy']}_{doc['variant']}"
        dims = grid_dimensions(doc)
        flat = flatten(doc)
        fx = {**{k: v for k, v in flat.items() if k not in dims and k not in ("spec_version", "strategy")}, **fx}
        sp = dims
        spec_hash = sh(doc)
    if not strategy:
        _die("--strategy or --spec is required")
    from nqlab.errors import Refusal

    try:
        pre = make_prereg(s, family, strategy, hypothesis, param_space=sp, fixed=fx, spec_hash=spec_hash,
                          origin=origin, llm_variant_id=variant_id)
        path = FamilyStore(s).register(pre)
    except Refusal as e:
        _die(str(e))
    con.print(f"registered {family} ({pre['n_configs']} configs, all reserved in the trial ledger): {path}")


@app.command()
def run(family: str, stage: Optional[int] = typer.Option(None), auto: bool = typer.Option(False, help="promote and continue while gates pass (never into the holdout)"),
        workers: str = typer.Option("1"), n_perm: Optional[int] = typer.Option(None)):
    """Run the family's current stage of the funnel."""
    from nqlab.data.market import MarketData
    from nqlab.runner.runner import resolve_workers
    from nqlab.selection.family import FamilyStore
    from nqlab.errors import Refusal as FamilyError
    from nqlab.selection.funnel import PASS, STAGE_NAMES

    s = _settings()
    store = FamilyStore(s)
    md = MarketData.from_store(s)
    n = resolve_workers(workers)
    while True:
        try:
            res = store.run_stage(family, md, stage, n_jobs=n, n_perm=n_perm)
        except FamilyError as e:
            _die(str(e))
        con.print(f"stage {res.stage} {STAGE_NAMES[res.stage]}: [bold]{res.status}[/bold] - {res.summary}")
        if not auto or res.status != PASS or res.stage >= 6:
            break
        store.promote(family)
        stage = None
    _report_family(s, family)


def _logged_backtest(s, strategy: str, params: dict, md=None, family_id: str | None = None, note: str = ""):
    """Every real-data backtest goes into the trial ledger, including failed ones."""
    from nqlab.data.market import MarketData
    from nqlab.engine.backtest import run_backtest
    from nqlab.results.store import ledger_error, ledger_log

    try:
        r = run_backtest(md or MarketData.from_store(s), strategy, params, s)
    except BaseException as exc:
        ledger_error(s, strategy, params, exc, family_id, "abandoned" if isinstance(exc, KeyboardInterrupt) else "errored")
        raise
    ledger_log(s, r, family_id, note=note)
    return r


def _report_family(s, family: str, out: Path | None = None) -> Path:
    from nqlab.runner.jobs import job_report

    r = job_report(s, {"payload": {"family_id": family}, "batch": None}, None, 1)
    path = Path(r["report"]) if out is None else Path(shutil.copy(r["report"], out))
    con.print(f"report: {path}")
    return path


@app.command()
def promote(family: str):
    """Advance a family one stage. Refuses unless the current stage PASSED."""
    from nqlab.selection.family import FamilyStore
    from nqlab.errors import Refusal as FamilyError

    s = _settings()
    try:
        new = FamilyStore(s).promote(family)
    except FamilyError as e:
        _die(str(e))
    con.print(f"{family} promoted to stage {new}")


@app.command()
def holdout(family: str, confirm: bool = typer.Option(False, help="acknowledge this is the ONE look")):
    """Stage 7: one scripted look at the locked holdout; consumption is recorded permanently."""
    from nqlab.errors import Refusal as FamilyError
    from nqlab.selection.holdout_cmd import consume_holdout

    s = _settings()
    try:
        res = consume_holdout(s, family, confirm=confirm)
    except FamilyError as e:
        _die(str(e))
    con.print(f"HOLDOUT {family}: [bold]{res.status}[/bold] - {res.summary}")
    _report_family(s, family)


@app.command()
def families():
    """List families and their funnel state."""
    from nqlab.selection.family import FamilyStore

    t = Table("family", "strategy", "status", "stage", "origin")
    for f in FamilyStore(_settings()).list():
        t.add_row(f["family_id"], f["strategy"], f["status"], str(f["current_stage"]), f["origin"])
    con.print(t)


@app.command()
def rank():
    """Families ranked by expected net $ per purchased account (edge haircut), never by Sharpe."""
    from nqlab.selection.ranking import rank_families

    f = lambda x, fmt="{:.2f}": "n/a" if x is None else fmt.format(x)  # noqa: E731
    t = Table("family", "status", "stage", "E\\[net]/acct (haircut)", "E\\[net]/acct", "zero-edge", "path", "P(1st payout)",
              "Sharpe ± SE", "DSR", "exp R", "PF", "max L", "trades",
              title="Ranked by expected net $ per purchased account after all fees, never by Sharpe")
    for r in rank_families(_settings()):
        t.add_row(r["family_id"], r["status"], f"{r['furthest_stage']} {r['last_result']}",
                  f(r["ev_net_per_account_haircut"], "${:,.0f}"), f(r["ev_net_per_account"], "${:,.0f}"),
                  f(r["zero_edge_ev_net"], "${:,.0f}"), r["payout_path"] or "n/a", f(r["p_first_payout"], "{:.1%}"),
                  f"{f(r['sharpe_ann'])} ± {f(r['sharpe_ann_se'])}", f(r["dsr"]), f(r["expectancy_r"]),
                  f(r["profit_factor"]), f(r["max_consec_losses"], "{:.0f}"), f(r["n_trades"], "{:.0f}"))
    con.print(t)


@app.command()
def farm(family_or_strategy: str, params: Optional[str] = None, n_paths: int = 2000, optimize: bool = False):
    """Account-farming economics: expected net $ per purchased 50K account after every fee,
    for both payout paths, vs a zero-edge trader and vs the edge cut to its 1-SE lower bound."""
    from nqlab.errors import Refusal
    from nqlab.risk.farming import farm_report, optimize_farming
    from nqlab.risk.topstep import TradeTape
    from nqlab.selection.family import FamilyStore
    from nqlab.strategies.registry import get_strategy

    s = _settings()
    family_id = None
    try:
        fam = FamilyStore(s).family(family_or_strategy)
        strat_name, p = fam["strategy"], json.loads(fam["selected_params_json"] or "{}")
        family_id = family_or_strategy
    except Refusal:
        strat_name = family_or_strategy
        p = {**get_strategy(strat_name).defaults(), **(json.loads(params) if params else {})}
    strat = get_strategy(strat_name)
    r = _logged_backtest(s, strat_name, p, family_id=family_id, note="farm")
    sz = strat.sizing(p)
    spec = s.contract(sz.instrument)
    tape = TradeTape.from_backtest(r.trades, r.daily, spec.point_value, spec.tick_value, s.round_turn_cost(sz.instrument))
    fr = farm_report(tape, s, strat.topstep_policy(p), sz.instrument, n_paths=n_paths)
    t = Table("path", "trader", "E\\[net]/account", "P(net>0)", "P(funded)", "P(any payout)", "E\\[fees]", "E\\[paid]",
              "months", "E\\[net]/month", title=f"{strat_name}: one purchased 50K account, all fees included")
    for path, v in fr["paths"].items():
        for label, key in (("strategy", "strategy"), ("zero edge", "zero_edge"), ("edge -1 SE", "haircut")):
            x = v[key]
            t.add_row(path, label, f"${x.ev_net:,.0f}", f"{x.p_net_positive:.0%}", f"{x.p_funded:.0%}",
                      f"{x.p_any_payout:.0%}", f"${x.mean_cost:,.0f}", f"${x.mean_paid:,.0f}", f"{x.mean_months:.1f}",
                      f"${x.ev_per_month:,.0f}")
    con.print(t)
    con.print(f"Fees: {fr['fees']}. Best path judged on the haircut edge: [bold]{fr['best_path']}[/bold].")
    bz = fr["paths"][fr["best_path"]]["zero_edge"]
    if bz.ev_net > 0:
        con.print(f"[yellow]Note:[/yellow] the same trades with ZERO edge still show ${bz.ev_net:,.0f} per account: losses "
                  "are capped at the fees while payouts are not, so variance alone has value under these rules. Only the "
                  "part above the zero-edge line is your edge; the rest is the payout structure (and depends on rules "
                  "nqlab could not verify).")
    if optimize:
        ts = s.section("topstep")
        opt = optimize_farming(tape, s.topstep_rules, s, n_paths=max(200, n_paths // 4),
                               combine_budget=ts["combine_horizon_sessions"], xfa_horizon=ts["xfa_horizon_sessions"])
        b, bu = opt["best"], opt["best_unhaircut"]
        if b["combine_risk"] == max(opt["grid"]["combine_risk"]) or b["xfa_risk"] == max(opt["grid"]["xfa_risk"]):
            con.print("[yellow]The best cell is at the edge of the grid (largest size tested).[/yellow] Bigger may score "
                      "higher in simulation, but every extra contract leans harder on the edge being real.")
        con.print(f"Best sizing (MNQ, judged on the haircut edge): {b['path']} path, {b['combine_risk']:.0%} of the MLL "
                  f"buffer per trade in the Combine, {b['xfa_risk']:.0%} funded -> E[net] ${b['ev_net_haircut']:,.0f} haircut / "
                  f"${b['ev_net']:,.0f} measured. On the measured edge alone it would pick {bu['combine_risk']:.0%} / "
                  f"{bu['xfa_risk']:.0%}.")


@app.command()
def report(target: str, out: Optional[Path] = typer.Option(None)):
    """HTML report for a family id, or for a run id from the results store."""
    from nqlab.results.db import connect
    from nqlab.selection.family import FamilyStore
    from nqlab.errors import Refusal as FamilyError

    s = _settings()
    try:
        FamilyStore(s).family(target)
        _report_family(s, target, out)
        return
    except FamilyError:
        pass
    with connect(s.path("results_db")) as c:
        row = c.execute("SELECT strategy, params_json, family_id FROM runs WHERE run_id=?", [target]).fetchone()
    if not row:
        _die(f"{target!r} is neither a family nor a run id")
    from nqlab.data.market import MarketData
    from nqlab.results.ledger import TrialLedger
    from nqlab.results.report import build_report

    md = MarketData.from_store(s)
    p = build_report(s, md, row[0], json.loads(row[1]), out or s.path("reports") / f"run_{target}.html",
                     n_trials=TrialLedger(s.path("results_db")).n_trials(), family_id=row[2])
    con.print(f"report: {p}")


@app.command()
def backtest(strategy: str, params: Optional[str] = typer.Option(None, help="JSON overrides of the defaults"),
             no_diag: bool = typer.Option(False, "--no-diag"), out: Optional[Path] = typer.Option(None)):
    """One backtest + HTML report (logged to the ledger; NOT validation)."""
    from nqlab.data.market import MarketData
    from nqlab.results.ledger import TrialLedger
    from nqlab.results.report import build_report
    from nqlab.strategies.registry import get_strategy

    s = _settings()
    st = get_strategy(strategy)
    p = {**st.defaults(), **(json.loads(params) if params else {})}
    md = MarketData.from_store(s)
    try:
        path = build_report(s, md, strategy, p, out or s.path("reports") / f"{strategy}.html",
                            n_trials=TrialLedger(s.path("results_db")).n_trials(), run_diag=not no_diag)
    except BaseException as exc:
        from nqlab.results.store import ledger_error

        ledger_error(s, strategy, p, exc, None, "abandoned" if isinstance(exc, KeyboardInterrupt) else "errored")
        raise
    con.print(f"report: {path}")


@app.command()
def calibrate(n_datasets: int = 40, n_perm: int = 49, workers: str = "auto"):
    """Null calibration harness: junk must be rejected at ~the nominal rate."""
    from nqlab.runner.runner import resolve_workers
    from nqlab.validation.calibration import calibrate as cal, power_check

    s = _settings()
    c = cal(s, n_datasets=n_datasets, n_perm=n_perm, n_jobs=resolve_workers(workers))
    pw = power_check(s, n_perm=n_perm)
    con.print({k: v for k, v in c.items() if k != "rows"})
    con.print(f"power check (planted edge) MCPT p: {pw['mcpt_p']}")
    ok = c["funnel_fpr"] <= 0.05 and max(pw["mcpt_p"].values()) < 0.05
    con.print("[green]validators calibrated[/green]" if ok else "[red]VALIDATORS ARE BROKEN: do not trust any result[/red]")
    if not ok:
        raise typer.Exit(1)


@app.command("fills-calibrate")
def fills_calibrate(family_or_strategy: str, params: Optional[str] = None, n_sessions: int = 200):
    """Compare fill tiers 1/2/3(/4) on a sample of sessions with tick data (Databento)."""
    from nqlab.data.market import MarketData
    from nqlab.data.store import DataStore
    from nqlab.engine.fill_calibration import compare_tiers
    from nqlab.errors import Refusal
    from nqlab.selection.family import FamilyStore
    from nqlab.strategies.registry import get_strategy

    s = _settings()
    store = DataStore(s)
    if not store.has_ticks("trades"):
        _die("no trades data: run `nqlab data ingest-ticks --schema trades ...` (Databento) first")
    try:
        f = FamilyStore(s).family(family_or_strategy)
        strat, p = f["strategy"], json.loads(f["selected_params_json"] or "{}")
    except Refusal:
        strat = family_or_strategy
        p = {**get_strategy(strat).defaults(), **(json.loads(params) if params else {})}
    mbo = store.scan_ticks("mbo").collect() if store.has_ticks("mbo") else None
    res = compare_tiers(MarketData.from_store(s), strat, p, s, store.scan_ticks("trades").collect(), mbo, n_sessions)
    t = Table("tier", "trades", "net $", "expectancy $", title=f"{res['n_sessions']} sessions, {res.get('window_et')} ET")
    for k, v in sorted(res["tiers"].items()):
        t.add_row(v["label"], str(v["n_trades"]), f"{v['net_usd']:,.0f}", f"{v['expectancy_usd']:,.2f}")
    con.print(t)


# ------------------------------------------------------------------------------- ledger
@ledger_app.command("stats")
def ledger_stats():
    """Current N (distinct trials, all statuses and origins) used by the Deflated Sharpe Ratio."""
    from nqlab.results.ledger import TrialLedger

    st = TrialLedger(_settings().path("results_db")).stats()
    con.print(f"N for DSR: [bold]{st['n_trials']}[/bold]")
    con.print(st)


@ledger_app.command("verify")
def ledger_verify():
    """Check the ledger's hash chain (detects edits/deletions)."""
    from nqlab.results.ledger import TrialLedger

    ok, msg = TrialLedger(_settings().path("results_db")).verify()
    con.print(("[green]OK[/green] " if ok else "[red]TAMPERED[/red] ") + msg)
    if not ok:
        raise typer.Exit(1)


# ------------------------------------------------------------------------------- queue
@queue_app.command("add")
def queue_add(type_: str = typer.Argument(..., metavar="TYPE", help="family | backtest | report | calibrate | morning_summary"),
              payload: str = typer.Option("{}", help="JSON payload"), depends: str = typer.Option("", help="comma-separated job ids"),
              batch: Optional[str] = None):
    from nqlab.runner.queue import JobQueue

    q = JobQueue(_settings().path("queue_db"))
    jid = q.add(type_, json.loads(payload), [d for d in depends.split(",") if d], batch=batch)
    con.print(jid)


@queue_app.command("batch")
def queue_batch(families: str = typer.Option("", help="comma-separated family ids"),
                baselines: str = typer.Option("orb,fhh_momentum,random_entry,session_hold"),
                n_perm: Optional[int] = None):
    """Queue tonight's batch: families through the funnel (never the holdout), baseline reports, morning summary."""
    from nqlab.runner.runner import queue_overnight_batch

    b = queue_overnight_batch(_settings(), [f for f in families.split(",") if f], [x for x in baselines.split(",") if x], n_perm)
    con.print(f"queued batch {b}")


@queue_app.command("list")
def queue_list():
    from nqlab.runner.queue import JobQueue

    t = Table("job", "type", "status", "attempts", "deps", "error")
    for j in JobQueue(_settings().path("queue_db")).list():
        t.add_row(j["job_id"], j["type"], j["status"], str(j["attempts"]), ",".join(j["depends_on"]), (j["error"] or "")[-60:])
    con.print(t)


def _run_queue(workers: str, log: Optional[Path]):
    from nqlab.runner.runner import run_queue

    s = get_settings()
    setup_logging(logfile=log)
    c = run_queue(s, workers=workers, logfile=log)
    con.print(c)


@queue_app.command("run-queue")
def queue_run(workers: str = "auto", log: Optional[Path] = None):
    _run_queue(workers, log)


@app.command("run-queue")
def run_queue_cmd(workers: str = typer.Option("auto"), log: Optional[Path] = typer.Option(None)):
    """Drain the job queue (what run_overnight.bat calls)."""
    _run_queue(workers, log)


# ------------------------------------------------------------------------------- parity
@parity_app.command("label")
def parity_label(date: str, time_et: str, side: str, note: str = ""):
    """Record 'I would have taken this trade here' (side: long/short/none)."""
    from nqlab.parity import add_label

    p = add_label(_settings(), dt.date.fromisoformat(date), time_et, side, note)
    con.print(f"label saved to {p}")


@parity_app.command("compare")
def parity_compare(family_or_strategy: str, params: Optional[str] = None, tolerance_min: int = 5):
    """Precision/recall of coded signals vs your hand labels, with every divergence listed."""
    import polars as pl

    from nqlab.data.market import MarketData
    from nqlab.engine.backtest import run_backtest
    from nqlab.parity import LABEL_COLS, compare_labels, labels_path
    from nqlab.selection.family import FamilyStore
    from nqlab.errors import Refusal as FamilyError
    from nqlab.strategies.registry import get_strategy

    s = _settings()
    try:
        f = FamilyStore(s).family(family_or_strategy)
        strat, p = f["strategy"], json.loads(f["selected_params_json"] or "{}")
    except FamilyError:
        strat = family_or_strategy
        p = {**get_strategy(strat).defaults(), **(json.loads(params) if params else {})}
    labels = pl.read_csv(labels_path(s), schema=LABEL_COLS)
    r = _logged_backtest(s, strat, p, note="parity compare")
    res = compare_labels(labels, r.trades, tolerance_min)
    con.print(res)
    if res["decision_required"]:
        con.print("[yellow]The code and you disagree. Decide explicitly which version is being tested.[/yellow]")


@parity_app.command("import")
def parity_import(csv: Path, family_or_strategy: str = typer.Option(...), params: Optional[str] = None):
    """TopstepX fills vs model fills: fill rate, slippage and expectancy differences with CIs."""
    from nqlab.data.market import MarketData
    from nqlab.engine.backtest import run_backtest
    from nqlab.parity import compare_fills, import_topstepx
    from nqlab.selection.family import FamilyStore
    from nqlab.errors import Refusal as FamilyError
    from nqlab.strategies.registry import get_strategy

    s = _settings()
    try:
        f = FamilyStore(s).family(family_or_strategy)
        strat, p = f["strategy"], json.loads(f["selected_params_json"] or "{}")
    except FamilyError:
        strat = family_or_strategy
        p = {**get_strategy(strat).defaults(), **(json.loads(params) if params else {})}
    live = import_topstepx(csv)
    r = _logged_backtest(s, strat, p, note="parity import")
    con.print(compare_fills(live, r.trades))


# ------------------------------------------------------------------------------- llm
@llm_app.command("status")
def llm_status():
    from nqlab.llm.client import OllamaClient

    c = OllamaClient(_settings())
    con.print(f"Ollama available, model {c.model}" if c.available() else "Ollama unavailable: LLM features are skipped")


@llm_app.command("postmortem")
def llm_postmortem(family: str, n: int = 20):
    """Worst-trade post-mortem -> a NEW hypothesis draft (never applied as a filter)."""
    from nqlab.data.market import MarketData
    from nqlab.engine.backtest import run_backtest
    from nqlab.llm.roles import NOISE_LINE, postmortem, trade_context
    from nqlab.selection.family import FamilyStore

    s = _settings()
    f = FamilyStore(s).family(family)
    md = MarketData.from_store(s)
    r = _logged_backtest(s, f["strategy"], json.loads(f["selected_params_json"] or "{}"), md=md, family_id=family,
                         note="llm post-mortem")
    p = postmortem(s, family, trade_context(md, r.trades, s, n))
    con.print(f"draft: {p}\n{NOISE_LINE}" if p else "skipped (Ollama unavailable)")


@llm_app.command("variants")
def llm_variants(family: str, n: int = 5):
    """Generate structural variants; each is ledgered at generation time and the DSR penalty is printed."""
    from nqlab.llm.roles import generate_variants
    from nqlab.selection.family import FamilyStore
    from nqlab.selection.prereg import load_prereg

    s = _settings()
    f = FamilyStore(s).family(family)
    pre = load_prereg(s, family, f["prereg_hash"])
    res = generate_variants(s, pre["strategy"], pre["param_space"], pre["fixed"], n)
    con.print(res.get("note", ""))
    for v in res["variants"]:
        con.print(f"- {v['family_id']}: {v['name']} (reserved in the ledger)")
    if res.get("dsr_penalty"):
        pen = res["dsr_penalty"]
        con.print(f"N {pen['n_trials_before']} -> {pen['n_trials_after']}; the Sharpe a strategy must beat by luck alone rises "
                  f"to {pen['sr0_annualized_after']:.2f} (annualized, 1 year of data).")


@llm_app.command("summary")
def llm_summary(batch: Optional[str] = None):
    """(Re)write the morning summary for a batch from code-computed results."""
    from nqlab.llm.roles import morning_summary
    from nqlab.runner.queue import JobQueue

    s = _settings()
    jobs = [j for j in JobQueue(s.path("queue_db")).list() if batch is None or j.get("batch") == batch]
    facts = [{"job": j["payload"].get("family_id") or j["payload"].get("strategy") or j["job_id"], "type": j["type"],
              "status": j["status"], "error": (j["error"] or "")[:300],
              "result": json.loads(j["result_json"]) if j["result_json"] else None} for j in jobs]
    con.print(f"wrote {morning_summary(s, facts, batch)}")


if __name__ == "__main__":
    app()
