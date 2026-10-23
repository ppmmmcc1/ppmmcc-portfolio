"""Null calibration harness as a CI test (spec 8.5): worthless strategies on
structure-free data must be rejected at about the nominal rate, and a planted edge
must be detected (otherwise "reject everything" would pass)."""

import os

import pytest
from scipy import stats

from nqlab.validation.calibration import calibrate, power_check

N_JOBS = max(1, (os.cpu_count() or 2) - 1)


def test_calibration_smoke(session_settings, tmp_path):
    c = calibrate(session_settings, n_datasets=2, n_days=200, n_perm=19, n_jobs=1, ledger_dir=tmp_path)
    assert c["n"] == 4
    assert c["funnel_fpr"] == 0.0


@pytest.mark.slow
def test_junk_rejected_at_nominal_rate(session_settings, tmp_path):
    c = calibrate(session_settings, n_datasets=30, n_days=260, n_perm=39, n_jobs=N_JOBS, ledger_dir=tmp_path)
    n = c["n"]
    # 99% binomial upper bound on false positives at alpha = 0.05
    ub = stats.binom.ppf(0.99, n, 0.05) / n
    assert c["mcpt_fpr_wf"] <= ub, c
    assert c["mcpt_fpr_is"] <= ub, c
    assert c["funnel_fpr"] <= 0.05, c
    # p-values must look uniform (not stuck at 1.0, which would mean a broken test)
    assert 0.3 < c["mean_p_wf"] < 0.7, c
    assert 0.3 < c["mean_p_is"] < 0.7, c


@pytest.mark.slow
def test_planted_edge_is_detected(session_settings, tmp_path):
    r = power_check(session_settings, n_perm=39, ledger_dir=tmp_path)
    assert r["mcpt_p"]["wf_oos_mean"] < 0.05 and r["mcpt_p"]["is_plateau"] < 0.05, r
    assert r["statuses"][1] == r["statuses"][2] == r["statuses"][3] == "PASS", r
