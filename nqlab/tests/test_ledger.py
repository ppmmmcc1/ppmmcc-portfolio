import re
from pathlib import Path

import duckdb
import pytest

from nqlab.results.ledger import LedgerEntry, TrialLedger, trial_key

SRC = Path(__file__).resolve().parents[1] / "src"


def test_append_count_and_dedupe(tmp_path):
    led = TrialLedger(tmp_path / "r.duckdb")
    k1 = trial_key("fam", "orb", {"a": 1})
    k2 = trial_key("fam", "orb", {"a": 2})
    led.append([LedgerEntry(k1, "config_run", "completed", family_id="fam", sharpe=0.1, n_obs=100),
                LedgerEntry(k1, "config_run", "completed", family_id="fam", fill_tier=1),   # same trial, other tier
                LedgerEntry(k2, "config_run", "errored", family_id="fam"),                 # errored runs count
                LedgerEntry("p", "permutation", "completed")])            # nulls do not count
    assert led.n_trials() == 2
    assert led.n_trials("fam") == 2
    ok, msg = led.verify()
    assert ok, msg


def test_llm_reservation_counts_without_running(tmp_path):
    led = TrialLedger(tmp_path / "r.duckdb")
    led.append([LedgerEntry(trial_key("v", "orb", {"a": i}, "llm1"), "llm_variant", "reserved", origin="llm_variant")
                for i in range(5)])
    assert led.n_trials() == 5


def test_tampering_breaks_chain(tmp_path):
    db = tmp_path / "r.duckdb"
    led = TrialLedger(db)
    led.append([LedgerEntry(f"k{i}", "config_run", "completed") for i in range(3)])
    con = duckdb.connect(str(db))
    con.execute("UPDATE trial_ledger SET status='abandoned' WHERE seq=2")  # simulated tampering
    con.close()
    ok, _ = led.verify()
    assert not ok


def test_no_delete_path_exists():
    """The ledger exposes no delete/update method and no source file mutates it."""
    for name in dir(TrialLedger):
        assert not re.search(r"delete|remove|update|truncate|drop|purge|clear", name, re.I), name
    pat = re.compile(r"(DELETE\s+FROM|UPDATE|DROP\s+TABLE|TRUNCATE)\s+(TABLE\s+)?trial_ledger", re.I)
    for f in SRC.rglob("*.py"):
        assert not pat.search(f.read_text(encoding="utf-8")), f"ledger mutation in {f}"


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        LedgerEntry("k", "made_up", "completed")
