"""Family lifecycle: register -> run stage -> promote, persisted in DuckDB.

Rules enforced here (the operator cannot talk his way around them from the CLI):
* stages run strictly in order; a stage cannot be skipped;
* a FAILED stage archives the family permanently; ``promote`` refuses;
* a stage that already produced PASS/FAIL is never re-run (no fishing); only ERRORED
  stages, or INSUFFICIENT DATA stages on a NEW data version, may run again;
* the pre-registration hash is checked on every action;
* registering a family reserves every config of its grid in the trial ledger.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from nqlab.errors import Refusal
from nqlab.hashing import canonical_json
from nqlab.log import get_logger
from nqlab.results.db import connect
from nqlab.results.ledger import LedgerEntry, TrialLedger, trial_key
from nqlab.selection.funnel import ERRORED, FAIL, INSUFFICIENT, PASS, STAGE_NAMES, STAGES, FamilyContext, StageResult
from nqlab.selection.plateau import Grid
from nqlab.selection.prereg import load_prereg, prereg_path, write_prereg

log = get_logger("selection.family")


class FamilyError(Refusal):
    pass


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    return str(o)


class FamilyStore:
    def __init__(self, settings):
        self.settings = settings
        self.db = settings.path("results_db")
        self.ledger = TrialLedger(self.db)
        self.art_dir = self.db.parent / "families"

    # ---- registration ------------------------------------------------------------------
    def register(self, prereg: dict[str, Any]) -> Path:
        fid = prereg["family_id"]
        with connect(self.db) as con:
            if con.execute("SELECT 1 FROM families WHERE family_id=?", [fid]).fetchone():
                raise FamilyError(f"family {fid!r} already exists")
        path = write_prereg(self.settings, prereg)
        grid = Grid.build(prereg["param_space"], prereg["fixed"], tuple(prereg.get("categorical", [])))
        self.ledger.append([
            LedgerEntry(trial_key(fid, prereg["strategy"], cfg, prereg.get("llm_variant_id")), "config_run", "reserved",
                        origin=prereg.get("origin", "human"), family_id=fid, strategy=prereg["strategy"], params=cfg,
                        note="reserved at pre-registration")
            for cfg in grid.configs])
        with connect(self.db) as con:
            con.execute("INSERT INTO families VALUES (?,?,?,?,?,?,?,?,?)",
                        [fid, prereg["strategy"], str(path), prereg["hash"], "active", 1,
                         dt.datetime.now(dt.timezone.utc), None, prereg.get("origin", "human")])
        return path

    # ---- state -------------------------------------------------------------------------
    def family(self, fid: str) -> dict[str, Any]:
        with connect(self.db) as con:
            r = con.execute("SELECT family_id, strategy, prereg_path, prereg_hash, status, current_stage, "
                            "selected_params_json FROM families WHERE family_id=?", [fid]).fetchone()
        if not r:
            raise FamilyError(f"unknown family {fid!r}")
        keys = ("family_id", "strategy", "prereg_path", "prereg_hash", "status", "current_stage", "selected_params_json")
        return dict(zip(keys, r))

    def results(self, fid: str) -> list[dict[str, Any]]:
        with connect(self.db) as con:
            rows = con.execute("SELECT stage, ts_utc, status, summary, details_json, data_version FROM stage_results "
                               "WHERE family_id=? ORDER BY ts_utc", [fid]).fetchall()
        return [{"stage": s, "ts": t, "status": st, "summary": su, "details": json.loads(d) if d else {},
                 "data_version": dv} for s, t, st, su, d, dv in rows]

    def list(self) -> list[dict[str, Any]]:
        with connect(self.db) as con:
            rows = con.execute("SELECT family_id, strategy, status, current_stage, origin FROM families ORDER BY created_ts").fetchall()
        return [dict(zip(("family_id", "strategy", "status", "current_stage", "origin"), r)) for r in rows]

    def _set(self, fid: str, **kw) -> None:
        sets = ", ".join(f"{k}=?" for k in kw)
        with connect(self.db) as con:
            con.execute(f"UPDATE families SET {sets} WHERE family_id=?", [*kw.values(), fid])

    def record(self, fid: str, res: StageResult, data_version: str) -> None:
        self.ledger.append([LedgerEntry(f"stage:{fid}:{res.stage}", "diagnostic", "completed", family_id=fid,
                                        data_version=data_version,
                                        note=f"STAGE {res.stage} {res.status}: {res.summary}"[:500])])
        self.art_dir.mkdir(parents=True, exist_ok=True)
        details = json.dumps(res.details, default=_json_default)
        with connect(self.db) as con:
            con.execute("INSERT INTO stage_results VALUES (?,?,?,?,?,?,?,?)",
                        [fid, res.stage, dt.datetime.now(dt.timezone.utc), res.status, res.summary, details,
                         data_version, None])

    # ---- running ------------------------------------------------------------------------
    def _ledger_failed(self, fid: str) -> int | None:
        for st, status in self.ledger.stage_outcomes(fid):
            if status == FAIL:
                return st
        return None

    def check_can_run(self, fid: str, stage: int | None, data_version: str) -> int:
        f = self.family(fid)
        load_prereg(self.settings, fid, f["prereg_hash"])
        if f["status"] == "archived" or self._ledger_failed(fid) is not None:
            raise FamilyError(f"{fid} is ARCHIVED (it failed a gate). Failed families are archived, not tweaked; "
                              "a modified idea is a new family and joins the trial count.")
        if f["status"] in ("completed", "holdout_consumed"):
            raise FamilyError(f"{fid} has finished the funnel")
        cur = int(f["current_stage"])
        stage = cur if stage is None else stage
        if stage != cur:
            raise FamilyError(f"{fid} is at stage {cur} ({STAGE_NAMES[cur]}); stages run in order and cannot be "
                              f"skipped or re-run (asked for {stage}).")
        if stage == 7:
            raise FamilyError("stage 7 is the holdout: use `nqlab holdout <family>` (one look, recorded)")
        prior = [r for r in self.results(fid) if r["stage"] == stage]
        for r in prior:
            if r["status"] in (PASS, FAIL):
                raise FamilyError(f"stage {stage} already produced {r['status']}; it is never re-run. "
                                  + ("Run `nqlab promote`." if r["status"] == PASS else ""))
            if r["status"] == INSUFFICIENT and r["data_version"] == data_version:
                raise FamilyError(f"stage {stage} was INSUFFICIENT DATA on this data version; re-running on the same "
                                  "data cannot change that. Ingest more data first.")
        return stage

    def context(self, fid: str, md, **kw) -> FamilyContext:
        f = self.family(fid)
        pre = load_prereg(self.settings, fid, f["prereg_hash"])
        return FamilyContext(pre, md, self.settings, self.ledger, **kw)

    def run_stage(self, fid: str, md, stage: int | None = None, **ctx_kw) -> StageResult:
        stage = self.check_can_run(fid, stage, md.data_version)
        ctx = self.context(fid, md, **ctx_kw)
        try:
            res = STAGES[stage](ctx)
        except KeyboardInterrupt:
            res = StageResult(stage, ERRORED, "abandoned by the operator (interrupted)", {})
            self.record(fid, res, md.data_version)
            self.ledger.append([LedgerEntry(f"stage_abandoned:{fid}:{stage}", "diagnostic", "abandoned", family_id=fid,
                                            strategy=ctx.prereg["strategy"], data_version=md.data_version)])
            raise
        except Exception as exc:
            res = StageResult(stage, ERRORED, f"{type(exc).__name__}: {exc}", {})
            self.record(fid, res, md.data_version)
            self.ledger.append([LedgerEntry(f"stage_error:{fid}:{stage}", "diagnostic", "errored", family_id=fid,
                                            strategy=ctx.prereg["strategy"], data_version=md.data_version,
                                            note=res.summary[:500])])
            raise
        self.record(fid, res, md.data_version)
        if res.status == FAIL:
            self._set(fid, status="archived")
            log.warning("%s FAILED stage %d (%s): archived. %s", fid, stage, STAGE_NAMES[stage], res.summary)
        if res.status in (PASS, FAIL, INSUFFICIENT):
            # the plateau pick on the full research sample, whatever the outcome, so that
            # every report describes the config the funnel actually selected
            _, cfg = ctx.selected()
            self._set(fid, selected_params_json=canonical_json(cfg))
        return res

    def promote(self, fid: str) -> int:
        f = self.family(fid)
        load_prereg(self.settings, fid, f["prereg_hash"])
        if f["status"] == "archived" or self._ledger_failed(fid) is not None:
            raise FamilyError(f"{fid} is archived: it failed a gate and can never be promoted.")
        cur = int(f["current_stage"])
        led = [status for st, status in self.ledger.stage_outcomes(fid) if st == cur]
        if not led or led[-1] != PASS:
            raise FamilyError(f"the trial ledger has no PASS for stage {cur}; promotion refused "
                              "(the results tables disagree with the tamper-evident ledger)")
        res = [r for r in self.results(fid) if r["stage"] == cur]
        if not res:
            raise FamilyError(f"stage {cur} ({STAGE_NAMES[cur]}) has not been run")
        last = res[-1]
        if last["status"] != PASS:
            raise FamilyError(f"stage {cur} result is {last['status']}: promotion refused. {last['summary']}")
        self._set(fid, current_stage=cur + 1)
        return cur + 1
