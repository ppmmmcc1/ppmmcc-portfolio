"""SQLite-backed job queue for overnight batches.

Jobs: {job_id, type, payload_json, status, attempts, checkpoint_json, created, started,
finished} plus dependency, heartbeat and error columns. Each job runs in its OWN
subprocess (``python -m nqlab.runner.worker <job_id>``): Windows-safe (no fork), a crash
or memory blow-up cannot take the runner down, and the job resumes from its last
checkpoint when the runner restarts.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    checkpoint_json TEXT,
    depends_on TEXT NOT NULL DEFAULT '[]',
    created TEXT NOT NULL,
    started TEXT,
    finished TEXT,
    heartbeat TEXT,
    worker_pid INTEGER,
    error TEXT,
    result_json TEXT,
    batch TEXT
);
"""
STATUSES = ("pending", "running", "done", "errored", "blocked")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    try:  # a zombie child still answers kill(0); check /proc where available
        with open(f"/proc/{int(pid)}/stat", "rb") as fh:
            return fh.read().split()[2] != b"Z"
    except OSError:
        return True


class JobQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._con() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _con(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    def add(self, type_: str, payload: dict[str, Any], depends_on: list[str] | None = None,
            batch: str | None = None) -> str:
        jid = f"{type_}-{uuid.uuid4().hex[:8]}"
        with self._con() as c:
            c.execute("INSERT INTO jobs(job_id,type,payload_json,depends_on,created,batch) VALUES (?,?,?,?,?,?)",
                      (jid, type_, json.dumps(payload), json.dumps(depends_on or []), _now(), batch))
        return jid

    def get(self, jid: str) -> dict[str, Any]:
        with self._con() as c:
            r = c.execute("SELECT * FROM jobs WHERE job_id=?", (jid,)).fetchone()
        if r is None:
            raise KeyError(jid)
        d = dict(r)
        d["payload"] = json.loads(d["payload_json"])
        d["checkpoint"] = json.loads(d["checkpoint_json"]) if d["checkpoint_json"] else None
        d["depends_on"] = json.loads(d["depends_on"])
        return d

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._con() as c:
            q = "SELECT job_id FROM jobs" + (" WHERE status=?" if status else "") + " ORDER BY created"
            ids = [r[0] for r in c.execute(q, (status,) if status else ())]
        return [self.get(i) for i in ids]

    # ---- state transitions ---------------------------------------------------------------
    def recover_stale(self, stale_minutes: float = 15.0) -> list[str]:
        """Jobs left 'running' by a dead runner/worker go back to 'pending' (they resume
        from their checkpoint)."""
        out = []
        now = dt.datetime.now(dt.timezone.utc)
        for j in self.list("running"):
            hb = dt.datetime.fromisoformat(j["heartbeat"]) if j["heartbeat"] else None
            dead = not pid_alive(j["worker_pid"])
            stale = hb is None or (now - hb).total_seconds() > stale_minutes * 60
            if dead or stale:
                with self._con() as c:
                    c.execute("UPDATE jobs SET status='pending', worker_pid=NULL WHERE job_id=?", (j["job_id"],))
                out.append(j["job_id"])
        return out

    def next_ready(self) -> dict[str, Any] | None:
        jobs = self.list()
        status = {j["job_id"]: j["status"] for j in jobs}
        for j in jobs:
            if j["status"] != "pending":
                continue
            deps = j["depends_on"]
            if j["payload"].get("after_any"):
                # e.g. the morning summary: runs once its deps have FINISHED, in any state
                if all(status.get(d) in ("done", "errored", "blocked") for d in deps):
                    return j
                continue
            if any(status.get(d) in ("errored", "blocked") for d in deps):
                self._set(j["job_id"], status="blocked", error="a dependency errored")
                return self.next_ready()      # re-scan: blocking may release after_any jobs
            if all(status.get(d) == "done" for d in deps):
                return j
        return None

    def _set(self, jid: str, **kw) -> None:
        sets = ", ".join(f"{k}=?" for k in kw)
        with self._con() as c:
            c.execute(f"UPDATE jobs SET {sets} WHERE job_id=?", (*kw.values(), jid))

    def start(self, jid: str, pid: int) -> None:
        with self._con() as c:
            c.execute("UPDATE jobs SET status='running', attempts=attempts+1, started=COALESCE(started, ?), "
                      "heartbeat=?, worker_pid=? WHERE job_id=?", (_now(), _now(), pid, jid))

    def heartbeat(self, jid: str) -> None:
        self._set(jid, heartbeat=_now())

    def save_checkpoint(self, jid: str, cp: dict[str, Any]) -> None:
        self._set(jid, checkpoint_json=json.dumps(cp, default=str), heartbeat=_now())

    def finish(self, jid: str, result: dict[str, Any] | None = None) -> None:
        self._set(jid, status="done", finished=_now(), result_json=json.dumps(result or {}, default=str), worker_pid=None)

    def fail(self, jid: str, error: str) -> None:
        self._set(jid, status="errored", finished=_now(), error=error[-4000:], worker_pid=None)
