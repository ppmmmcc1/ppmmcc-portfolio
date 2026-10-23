"""Null calibration harness (spec 8.5). Build deliberately worthless strategies (random
entries with the operator's exits, coin-flip direction, fixed-time entry), run the full
validation pipeline on structure-free data, and measure how often they get through.

If the validators pass junk, the validators are broken. The harness also has a power
check (a planted edge must be detectable) so "reject everything" cannot pass either.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np

from nqlab.data.market import MarketData
from nqlab.data.synthetic import random_walk_bars
from nqlab.results.ledger import TrialLedger
from nqlab.selection.funnel import PASS, FamilyContext, STAGES
from nqlab.selection.prereg import make_prereg

JUNK = {
    "random_entry": {"space": {"stop_ticks": [24, 40, 60], "target_r": [1.0, 2.0]}},
    "null_fixed_time": {"space": {"entry_minute": [515, 525, 540], "stop_ticks": [20, 40]}},
}


def _dataset(settings, seed: int, n_days: int, momentum_edge: float = 0.0) -> MarketData:
    start = dt.date(2021, 1, 4)
    end = start + dt.timedelta(days=int(n_days * 7 / 5) + 10)
    raw = random_walk_bars(settings, start, end, seed=seed, rth_only=True, momentum_edge=momentum_edge)
    return MarketData.from_raw(raw, settings, data_version=f"null{seed}")


def run_family(md, settings, strategy: str, space: dict, seed: int, ledger_dir: Path, n_perm: int,
               fixed_extra: dict | None = None) -> dict[str, Any]:
    fx = {"seed": seed} if strategy in ("random_entry", "null_fixed_time") else {}
    fx.update(fixed_extra or {})
    pre = make_prereg(settings, f"cal_{strategy}_{seed}", strategy, "calibration junk: should be rejected",
                      param_space=space, fixed=fx)
    ledger = TrialLedger(ledger_dir / f"ledger_{strategy}_{seed}.duckdb")
    ctx = FamilyContext(pre, md, settings, ledger, n_perm=n_perm, min_perm=min(19, n_perm))
    statuses = {}
    funnel_pass = True
    for s in range(1, 6):
        r = STAGES[s](ctx)
        statuses[s] = r.status
        if s == 4:
            mcpt_p = r.details["p_values"]
        if r.status != PASS:
            funnel_pass = False
    return {"strategy": strategy, "seed": seed, "statuses": statuses, "mcpt_p": mcpt_p,
            "funnel_pass": funnel_pass and all(v == PASS for v in statuses.values())}


def _one(settings_root: str, seed: int, n_days: int, n_perm: int, ledger_dir: str, strategies: list[str]):
    from nqlab.settings import get_settings

    settings = get_settings(Path(settings_root))
    md = _dataset(settings, seed, n_days)
    return [run_family(md, settings, s, JUNK[s]["space"], seed, Path(ledger_dir), n_perm) for s in strategies]


def calibrate(settings, n_datasets: int = 40, n_days: int = 260, n_perm: int = 49, n_jobs: int = 1,
              seed0: int = 1000, ledger_dir: Path | None = None, strategies: list[str] | None = None) -> dict[str, Any]:
    import tempfile

    strategies = strategies or list(JUNK)
    ledger_dir = Path(ledger_dir or tempfile.mkdtemp(prefix="nqlab_cal_"))
    seeds = [seed0 + i for i in range(n_datasets)]
    if n_jobs == 1:
        rows = [r for s in seeds for r in _one(str(settings.root), s, n_days, n_perm, str(ledger_dir), strategies)]
    else:
        from joblib import Parallel, delayed

        parts = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one)(str(settings.root), s, n_days, n_perm, str(ledger_dir), strategies) for s in seeds)
        rows = [r for p in parts for r in p]
    p_wf = np.array([r["mcpt_p"]["wf_oos_mean"] for r in rows])
    p_is = np.array([r["mcpt_p"]["is_plateau"] for r in rows])
    p_max = np.maximum(p_wf, p_is)
    return {
        "n": len(rows),
        "mcpt_fpr_wf": float((p_wf < 0.05).mean()),
        "mcpt_fpr_is": float((p_is < 0.05).mean()),
        "mcpt_fpr_gate": float((p_max < 0.05).mean()),
        "mean_p_wf": float(p_wf.mean()), "mean_p_is": float(p_is.mean()),
        "funnel_fpr": float(np.mean([r["funnel_pass"] for r in rows])),
        "rows": rows,
    }


def power_check(settings, seed: int = 7, n_days: int = 260, n_perm: int = 49, edge: float = 0.6,
                ledger_dir: Path | None = None) -> dict[str, Any]:
    import tempfile

    md = _dataset(settings, seed, n_days, momentum_edge=edge)
    ledger_dir = Path(ledger_dir or tempfile.mkdtemp(prefix="nqlab_pow_"))
    return run_family(md, settings, "fhh_momentum", {"hold_minutes": [30, 60, 90], "stop_mult": [0.5, 0.75, 1.0]},
                      seed, ledger_dir, n_perm)
