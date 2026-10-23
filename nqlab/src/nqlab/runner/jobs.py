"""Job implementations, executed inside a worker subprocess."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from nqlab.log import get_logger

log = get_logger("runner.jobs")


def _batch_dir(settings, batch: str | None) -> Path:
    d = settings.path("reports") / f"overnight_{batch or dt.date.today().strftime('%Y%m%d')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_family(settings, job, queue, n_jobs: int) -> dict[str, Any]:
    """Run a family's current stage (optionally promoting and continuing through stage 6)."""
    from nqlab.data.market import MarketData
    from nqlab.selection.family import FamilyStore
    from nqlab.selection.funnel import PASS

    p = job["payload"]
    fid = p["family_id"]
    store = FamilyStore(settings)
    md = MarketData.from_store(settings)
    out = []
    while True:
        f = store.family(fid)
        stage = int(f["current_stage"])
        if f["status"] != "active" or stage >= 7:
            break
        prior = [r for r in store.results(fid) if r["stage"] == stage]
        if prior and prior[-1]["status"] == PASS:
            store.promote(fid)
            continue
        cp = job["checkpoint"] if (job["checkpoint"] or {}).get("stage") == stage else None
        res = store.run_stage(
            fid, md, stage, n_jobs=n_jobs, n_perm=p.get("n_perm"),
            mcpt_checkpoint=(cp or {}).get("mcpt"),
            save_checkpoint=lambda c, s=stage: queue.save_checkpoint(job["job_id"], {"stage": s, "mcpt": c}))
        out.append({"stage": stage, "status": res.status, "summary": res.summary})
        if res.status != PASS or not p.get("auto", True):
            break
        store.promote(fid)
        if p.get("stop_before") and int(store.family(fid)["current_stage"]) >= int(p["stop_before"]):
            break
    rep = job_report(settings, {"payload": {"family_id": fid, "batch": job.get("batch")}}, queue, n_jobs)
    return {"family_id": fid, "stages": out, **rep}


def job_mcpt(settings, job, queue, n_jobs: int) -> dict[str, Any]:
    """Standalone permutation test of a family's whole selection procedure (diagnostic:
    it does not advance the funnel). Checkpointed; resumes after a crash."""
    from nqlab.data.market import MarketData
    from nqlab.results.ledger import LedgerEntry, TrialLedger
    from nqlab.selection.family import FamilyStore
    from nqlab.selection.prereg import load_prereg
    from nqlab.validation.mcpt import run_mcpt

    p = job["payload"]
    fid = p["family_id"]
    pre = load_prereg(settings, fid, FamilyStore(settings).family(fid)["prereg_hash"])
    md = MarketData.from_store(settings)
    cp = (job["checkpoint"] or {}).get("mcpt")
    res = run_mcpt(md, pre["strategy"], pre["param_space"], pre["fixed"], tuple(pre.get("categorical", [])), settings,
                   n_perm=int(p.get("n_perm") or settings.section("validation")["mcpt"]["n_permutations"]),
                   checkpoint=cp, save_checkpoint=lambda c: queue.save_checkpoint(job["job_id"], {"mcpt": c}),
                   checkpoint_every=int(settings.section("validation")["mcpt"]["checkpoint_every"]), n_jobs=n_jobs)
    TrialLedger(settings.path("results_db")).append([LedgerEntry(
        f"perm:{fid}:{res.n_perm}", "permutation", "completed", family_id=fid, strategy=pre["strategy"],
        data_version=md.data_version, note=f"diagnostic MCPT, {res.n_perm} permutations")])
    return {"family_id": fid, "p_values": res.p_values, "real": res.real}


def job_report(settings, job, queue, n_jobs: int) -> dict[str, Any]:
    from nqlab.data.market import MarketData
    from nqlab.results.ledger import TrialLedger
    from nqlab.results.report import build_report
    from nqlab.selection.family import FamilyStore
    from nqlab.selection.prereg import load_prereg

    p = job["payload"]
    md = MarketData.from_store(settings)
    ledger = TrialLedger(settings.path("results_db"))
    out_dir = _batch_dir(settings, p.get("batch") or job.get("batch")) if (p.get("batch") or job.get("batch")) else settings.path("reports")
    if p.get("family_id"):
        store = FamilyStore(settings)
        fid = p["family_id"]
        f = store.family(fid)
        pre = load_prereg(settings, fid, f["prereg_hash"])
        if f["selected_params_json"]:
            params = json.loads(f["selected_params_json"])
        else:
            from nqlab.selection.plateau import Grid

            params = Grid.build(pre["param_space"], pre["fixed"]).configs[0]
        path = build_report(settings, md, pre["strategy"], params, out_dir / f"{fid}.html", n_trials=ledger.n_trials(),
                            stage_rows=store.results(fid), family_id=fid, grid_keys=list(pre["param_space"]))
    else:
        from nqlab.strategies.registry import get_strategy

        strat = get_strategy(p["strategy"])
        params = {**strat.defaults(), **(p.get("params") or {})}
        path = build_report(settings, md, p["strategy"], params, out_dir / f"{p['strategy']}.html", n_trials=ledger.n_trials())
    return {"report": str(path)}


def job_backtest(settings, job, queue, n_jobs):
    return job_report(settings, job, queue, n_jobs)


def job_calibrate(settings, job, queue, n_jobs):
    from nqlab.validation.calibration import calibrate, power_check

    p = job["payload"]
    c = calibrate(settings, n_datasets=int(p.get("n_datasets", 40)), n_perm=int(p.get("n_perm", 49)), n_jobs=n_jobs)
    pw = power_check(settings, n_perm=int(p.get("n_perm", 49)))
    return {k: v for k, v in c.items() if k != "rows"} | {"power_mcpt_p": pw["mcpt_p"]}


def job_morning_summary(settings, job, queue, n_jobs):
    from nqlab.llm.roles import morning_summary

    batch = job.get("batch")
    jobs = [j for j in queue.list() if j.get("batch") == batch and j["job_id"] != job["job_id"]]
    facts = [{"job": j["job_id"], "type": j["type"], "status": j["status"], "error": (j["error"] or "")[:300],
              "result": json.loads(j["result_json"]) if j["result_json"] else None} for j in jobs]
    path = morning_summary(settings, facts, batch=batch)
    return {"summary": str(path)}


JOBS = {"family": job_family, "mcpt": job_mcpt, "report": job_report, "backtest": job_backtest, "calibrate": job_calibrate,
        "morning_summary": job_morning_summary}
