"""Multiple-testing corrections: PSR, DSR, PBO (CSCV), SPA / StepM."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from scipy import stats

from nqlab.selection.plateau import Grid, plateau_scores

EULER = 0.5772156649015329


def psr(sr: float, sr_star: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012). Per-period Sharpe."""
    if n < 3 or not np.isfinite(sr):
        return float("nan")
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR] of n_trials independent zero-skill trials with Sharpe variance var_sr."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2)


def dsr(returns: np.ndarray, n_trials: int, trial_sharpes: list[float] | None = None) -> dict[str, Any]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    ``n_trials`` MUST come from the trial ledger. The variance of trial Sharpes is the
    larger of the ledger's empirical variance and the null variance 1/T (so a ledger of
    near-identical configs cannot shrink the deflation)."""
    r = np.asarray(returns, float)
    T = len(r)
    if T < 3 or r.std(ddof=1) == 0:
        return {"dsr": float("nan"), "sr": float("nan"), "sr0": float("nan"), "n_trials": n_trials, "T": T}
    sr = float(r.mean() / r.std(ddof=1))
    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r, fisher=False))
    var_null = 1.0 / T
    var_emp = float(np.var(trial_sharpes, ddof=1)) if trial_sharpes is not None and len(trial_sharpes) > 1 else 0.0
    v = max(var_null, var_emp)
    sr0 = expected_max_sharpe(max(int(n_trials), 1), v)
    return {"dsr": psr(sr, sr0, T, sk, ku), "psr0": psr(sr, 0.0, T, sk, ku), "sr": sr, "sr0": sr0,
            "n_trials": int(n_trials), "T": T, "var_sr_used": v, "skew": sk, "kurt": ku}


def pbo_cscv(matrix: np.ndarray, grid: Grid, n_blocks: int = 16) -> dict[str, Any]:
    """Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado, Zhu).

    The IS 'winner' is chosen by the SAME plateau rule the funnel uses. PBO = share of
    symmetric splits in which the IS winner ranks at or below the OOS median."""
    M = np.asarray(matrix, float)
    T, N = M.shape
    if N < 2:
        return {"pbo": 0.0, "note": "single configuration: no selection, nothing to overfit", "logits": []}
    S = n_blocks - (n_blocks % 2)
    S = min(S, T - (T % 2)) if T < S else S
    if S < 4:
        return {"pbo": float("nan"), "note": "too few sessions for CSCV", "logits": []}
    bounds = np.linspace(0, T, S + 1).astype(int)
    sums = np.array([M[bounds[i]:bounds[i + 1]].sum(axis=0) for i in range(S)])
    sq = np.array([(M[bounds[i]:bounds[i + 1]] ** 2).sum(axis=0) for i in range(S)])
    cnt = np.diff(bounds)

    def sharpe_of(blocks):
        n = cnt[blocks].sum()
        s, q = sums[blocks].sum(axis=0), sq[blocks].sum(axis=0)
        mu = s / n
        var = (q - n * mu * mu) / max(n - 1, 1)
        sd = np.sqrt(np.maximum(var, 0))
        return np.where(sd > 0, mu / np.where(sd > 0, sd, 1), 0.0)

    logits = []
    all_b = np.arange(S)
    for combo in itertools.combinations(range(S), S // 2):
        is_b = np.array(combo)
        oos_b = np.setdiff1d(all_b, is_b)
        is_sr, oos_sr = sharpe_of(is_b), sharpe_of(oos_b)
        star = int(np.argmax(plateau_scores(is_sr, grid)))
        rank = stats.rankdata(oos_sr)[star]          # 1..N, average ties
        w = rank / (N + 1)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "logits": logits.tolist(), "n_splits": len(logits), "S": S}


def spa_test(matrix: np.ndarray, reps: int = 1000, block_size: int = 5, seed: int = 0) -> dict[str, Any]:
    """Hansen's SPA and Romano-Wolf StepM against a do-nothing benchmark (0 P&L), fed the
    daily P&L of EVERY variant tested."""
    from arch.bootstrap import SPA, StepM

    M = np.asarray(matrix, float)
    if M.shape[0] < 10:
        return {"spa_p_consistent": float("nan"), "stepm_superior": []}
    losses = -M
    bench = np.zeros(M.shape[0])
    if np.all(M.std(axis=0) == 0):
        return {"spa_p_consistent": 1.0, "spa_p_lower": 1.0, "spa_p_upper": 1.0, "stepm_superior": []}
    keep = M.std(axis=0) > 0
    spa = SPA(bench, losses[:, keep], block_size=block_size, reps=reps, seed=seed)
    spa.compute()
    pv = spa.pvalues
    st = StepM(bench, losses[:, keep], size=0.05, block_size=block_size, reps=reps, seed=seed)
    st.compute()
    idx = np.nonzero(keep)[0]
    sup = [int(idx[int(str(i).split("_")[-1]) if not isinstance(i, (int, np.integer)) else int(i)])
           for i in st.superior_models] if len(st.superior_models) else []
    return {"spa_p_consistent": float(pv["consistent"]), "spa_p_lower": float(pv["lower"]),
            "spa_p_upper": float(pv["upper"]), "stepm_superior": sup}
