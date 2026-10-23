"""Evaluate a whole parameter grid -> daily P&L matrix (sessions x configs).

Intraday flat-at-close strategies make each session's P&L independent of earlier
sessions, so walk-forward, CPCV, PBO, SPA and MCPT can all work from this one matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nqlab.engine.backtest import BacktestResult, run_backtest
from nqlab.results.ledger import LedgerEntry, TrialLedger, trial_key
from nqlab.results.metrics import sharpe
from nqlab.selection.plateau import Grid


@dataclass
class FamilyMatrix:
    strategy: str
    grid: Grid
    dates: list
    pnl: np.ndarray                  # (T, N) daily USD
    n_trades: np.ndarray             # (N,)
    fill_tier: int
    data_version: str
    results: list[BacktestResult] = field(default_factory=list)

    @property
    def T(self) -> int:
        return self.pnl.shape[0]


def evaluate_grid(md, strategy: str, grid: Grid, settings, fill_tier: int = 2,
                  ledger: TrialLedger | None = None, family_id: str | None = None, kind: str = "config_run",
                  origin: str = "human", keep_results: bool = False, variant: str | None = None) -> FamilyMatrix:
    cols, ntr, results, entries = [], [], [], []
    dates = None
    for cfg in grid.configs:
        key = trial_key(family_id, strategy, cfg, variant)
        try:
            r = run_backtest(md, strategy, cfg, settings, fill_tier=fill_tier)
        except Exception as exc:
            if ledger is not None:
                entries.append(LedgerEntry(key, kind, "errored", origin=origin, family_id=family_id,
                                           strategy=strategy, params=cfg, data_version=md.data_version,
                                           fill_tier=fill_tier, note=f"{type(exc).__name__}: {exc}"[:500]))
                ledger.append(entries)
            raise
        d = r.daily
        if dates is None:
            dates = d["session_date"].to_list()
        col = d["pnl_usd"].to_numpy()
        cols.append(col)
        ntr.append(r.n_trades)
        if keep_results:
            results.append(r)
        if ledger is not None:
            entries.append(LedgerEntry(key, kind, "completed", origin=origin, run_id=r.run_id, family_id=family_id,
                                       strategy=strategy, params=cfg, data_version=md.data_version,
                                       fill_tier=fill_tier, sharpe=sharpe(col), n_obs=len(col)))
    if ledger is not None and entries:
        ledger.append(entries)
    pnl = np.column_stack(cols) if cols else np.zeros((0, 0))
    return FamilyMatrix(strategy, grid, dates or [], pnl, np.array(ntr), fill_tier, md.data_version, results)
