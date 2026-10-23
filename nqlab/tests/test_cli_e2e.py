"""Round trip through the real CLI on a tiny synthetic dataset: ingest -> backtest report
-> preregister -> run -> refusals. (Spec 14: ingest -> features -> engine -> metrics.)"""

import datetime as dt
import os
import subprocess
import sys

import polars as pl
import pytest
import yaml

from nqlab.data.synthetic import random_walk_bars


def nq(root, *args, ok=True):
    env = {**os.environ, "NQLAB_ROOT": str(root), "COLUMNS": "200"}
    r = subprocess.run([sys.executable, "-m", "nqlab.cli", *args], env=env, capture_output=True, text=True, timeout=900)
    if ok:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


@pytest.fixture()
def project(settings):
    p = settings.root / "config" / "settings.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["validation"]["bootstrap"]["n_paths"] = 300
    raw["llm"]["base_url"] = "http://127.0.0.1:9"
    p.write_text(yaml.safe_dump(raw))
    bars = random_walk_bars(settings, dt.date(2023, 1, 3), dt.date(2023, 12, 29), seed=5, rth_only=True)
    et = bars.with_columns(pl.col("ts_utc").dt.convert_time_zone("America/New_York").dt.strftime("%Y-%m-%d %H:%M:%S").alias("timestamp"))
    et.select("timestamp", "open", "high", "low", "close", "volume").write_csv(settings.path("data_raw") / "nq.csv")
    return settings.root


def test_cli_round_trip(project):
    out = nq(project, "data", "ingest").stdout
    assert "LOCKED in the holdout" in out
    nq(project, "data", "quality")
    r = nq(project, "backtest", "session_hold", "--no-diag").stdout
    html = next((project / "reports").glob("session_hold.html")).read_text()
    assert "INSUFFICIENT" in html or "FAIL" in html           # a single backtest is never PASS
    assert "A zero-edge trader passes this Combine roughly 40% of the time" in html
    assert "Tier 1 touch (optimistic upper bound only)" in html
    st = nq(project, "ledger", "stats").stdout
    assert "N for DSR: 1" in st
    nq(project, "preregister", "orb_t", "--strategy", "orb", "--hypothesis", "Opening range breakouts continue.",
       "--space", '{"or_minutes": [5, 15], "target_r": [1.0, 2.0]}')
    assert "N for DSR: 5" in nq(project, "ledger", "stats").stdout    # 4 configs reserved at registration
    r = nq(project, "run", "orb_t", ok=False)
    assert "stage 1 SCREEN" in r.stdout
    assert "refused" in nq(project, "run", "orb_t", "--stage", "3", ok=False).stdout
    lint = nq(project, "spec", "lint", "specs/bb_vp_macd.yaml", "--no-llm", "--strict", ok=False)
    assert lint.returncode == 1 and "hourly" in lint.stdout
    assert "OK" in nq(project, "ledger", "verify").stdout
    # editing a locked pre-registration voids the family
    pre = project / "research_log" / "orb_t.yaml"
    os.chmod(pre, 0o644)
    pre.write_text(pre.read_text().replace("Opening range breakouts continue.", "Something else entirely."))
    assert "modified after it was locked" in nq(project, "promote", "orb_t", ok=False).stdout
