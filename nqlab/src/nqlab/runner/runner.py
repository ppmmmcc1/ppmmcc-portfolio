"""The overnight runner: drains the queue one job at a time, each in a subprocess."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

from nqlab.log import get_logger
from nqlab.runner.queue import JobQueue

log = get_logger("runner")


def resolve_workers(w: str | int) -> int:
    if str(w) == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(w))


def run_queue(settings, workers: str | int = "auto", poll: float = 1.0, max_jobs: int | None = None,
              logfile: Path | None = None) -> dict[str, int]:
    q = JobQueue(settings.path("queue_db"))
    n = resolve_workers(workers)
    rec = q.recover_stale()
    if rec:
        log.warning("recovered %d interrupted job(s); they resume from checkpoint: %s", len(rec), rec)
    counts = {"done": 0, "errored": 0}
    env = dict(os.environ)
    env["NQLAB_ROOT"] = str(settings.root)
    ran = 0
    while True:
        job = q.next_ready()
        if job is None:
            break
        log.info("running %s (%s), attempt %d", job["job_id"], job["type"], job["attempts"] + 1)
        out = open(logfile, "a", encoding="utf-8") if logfile else None  # noqa: SIM115
        try:
            proc = subprocess.run([sys.executable, "-m", "nqlab.runner.worker", job["job_id"], str(n)],
                                  env=env, stdout=out, stderr=subprocess.STDOUT if out else None)
        finally:
            if out:
                out.close()
        st = q.get(job["job_id"])["status"]
        if proc.returncode != 0 and st == "running":  # hard crash: worker could not record it
            q.fail(job["job_id"], f"worker exited with code {proc.returncode}")
            st = "errored"
        counts["done" if st == "done" else "errored"] += 1
        ran += 1
        if max_jobs and ran >= max_jobs:
            break
        time.sleep(0.01)
    return counts


def queue_overnight_batch(settings, families: list[str], baselines: list[str], n_perm: int | None = None) -> str:
    """Convenience: one batch = every listed family through the funnel + baseline reports
    + the morning summary that depends on all of them."""
    q = JobQueue(settings.path("queue_db"))
    batch = dt.date.today().strftime("%Y%m%d")
    ids = [q.add("family", {"family_id": f, "auto": True, "n_perm": n_perm, "stop_before": 7}, batch=batch) for f in families]
    ids += [q.add("backtest", {"strategy": s, "batch": batch}, batch=batch) for s in baselines]
    q.add("morning_summary", {"after_any": True}, depends_on=ids, batch=batch)
    return batch
