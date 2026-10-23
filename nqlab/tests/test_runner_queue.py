"""Job queue: dependency ordering, crash isolation (errored -> ledger), and a kill
mid-run that resumes from the checkpoint instead of restarting."""

import datetime as dt
import os
import signal
import subprocess
import sys
import time

import pytest

from nqlab.results.ledger import TrialLedger
from nqlab.runner.queue import JobQueue, pid_alive
from nqlab.runner.runner import run_queue


def test_dependencies_and_errors(settings):
    q = JobQueue(settings.path("queue_db"))
    a = q.add("report", {"strategy": "does_not_exist"})
    b = q.add("report", {"strategy": "orb"}, depends_on=[a])
    c = q.add("morning_summary", {"after_any": True}, depends_on=[a, b])
    counts = run_queue(settings, workers=1)
    assert q.get(a)["status"] == "errored"
    assert q.get(b)["status"] == "blocked"
    assert q.get(c)["status"] == "done"            # the summary still runs
    st = TrialLedger(settings.path("results_db")).stats()
    assert st["by_status"].get("errored", 0) >= 1  # crash logged in the ledger
    assert counts["errored"] == 1
    assert list(settings.path("reports").glob("morning_*.md"))


@pytest.fixture()
def ingested(settings):
    import polars as pl

    from nqlab.data.ingest import ingest
    from nqlab.data.synthetic import random_walk_bars

    raw = random_walk_bars(settings, dt.date(2023, 1, 3), dt.date(2024, 3, 28), seed=21, rth_only=True)
    et = raw.with_columns(pl.col("ts_utc").dt.convert_time_zone("America/New_York").dt.strftime("%Y-%m-%d %H:%M:%S").alias("timestamp"))
    et.select("timestamp", "open", "high", "low", "close", "volume").write_csv(settings.path("data_raw") / "nq.csv")
    return ingest(settings)


@pytest.mark.slow
def test_kill_midrun_resumes_from_checkpoint(settings, ingested):
    from nqlab.selection.family import FamilyStore
    from nqlab.selection.prereg import make_prereg

    settings.raw["validation"]["mcpt"]["checkpoint_every"] = 5
    # persist the faster checkpoint cadence for the worker subprocess
    import yaml
    p = settings.root / "config" / "settings.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["validation"]["mcpt"]["checkpoint_every"] = 5
    p.write_text(yaml.safe_dump(raw))
    store = FamilyStore(settings)
    pre = make_prereg(settings, "kill_v1", "null_fixed_time", "junk family for the kill test",
                      param_space={"stop_ticks": [20, 40]}, fixed={"seed": 1, "entry_minute": 520})
    store.register(pre)
    q = JobQueue(settings.path("queue_db"))
    jid = q.add("mcpt", {"family_id": "kill_v1", "n_perm": 60})
    env = {**os.environ, "NQLAB_ROOT": str(settings.root)}
    proc = subprocess.Popen([sys.executable, "-m", "nqlab.cli", "run-queue", "--workers", "1"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    done_before = 0
    for _ in range(600):
        time.sleep(0.5)
        cp = q.get(jid)["checkpoint"]
        if cp and len((cp.get("mcpt") or {}).get("done", {})) >= 10:
            done_before = len(cp["mcpt"]["done"])
            break
    assert done_before >= 10, "never reached a checkpoint"
    os.killpg(proc.pid, signal.SIGKILL)          # kill runner AND worker (laptop lid closed)
    proc.wait()
    time.sleep(0.5)
    assert q.get(jid)["status"] == "running" and not pid_alive(q.get(jid)["worker_pid"])
    proc2 = subprocess.run([sys.executable, "-m", "nqlab.cli", "run-queue", "--workers", "1"], env=env,
                           capture_output=True, text=True, timeout=1800)
    j = q.get(jid)
    assert j["status"] == "done", (proc2.stdout, proc2.stderr, j["error"])
    assert j["attempts"] == 2
    assert len(j["checkpoint"]["mcpt"]["done"]) == 60
    assert j["checkpoint"]["mcpt"].get("resumed_with", 0) >= done_before
