"""Monte Carlo permutation test of the WHOLE selection procedure (Masters).

On the real data and on each permuted dataset we run the full family: evaluate every
config in the grid, select by plateau, walk the selection rule forward. Re-optimizing
on every permutation is what makes the test honest about selection bias.

    p = (1 + #{perm >= real}) / (1 + n_perm)

Two statistics are recorded and BOTH must clear the gate (conservative):
  * ``wf_oos_mean``  - mean daily OOS P&L of the walk-forward selection rule
  * ``is_plateau``   - best in-sample plateau score (Masters' in-sample MCPT)

Progress is checkpointed every ``checkpoint_every`` permutations through a callback so a
crash or a closed laptop lid resumes instead of restarting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nqlab.selection.plateau import Grid, select
from nqlab.validation.matrix import evaluate_grid
from nqlab.validation.permutation import permute_bars
from nqlab.validation.walkforward import walk_forward

STATS = ("wf_oos_mean", "is_plateau")


def family_statistics(md, strategy: str, grid: Grid, settings, fill_tier: int, wf: dict[str, int]) -> dict[str, float]:
    fm = evaluate_grid(md, strategy, grid, settings, fill_tier=fill_tier)
    _, is_score = select(fm.pnl, grid)
    w = walk_forward(fm.pnl, grid, wf["train_sessions"], wf["test_sessions"], wf["step_sessions"])
    return {"wf_oos_mean": float(w.get("oos_mean", float("nan"))), "is_plateau": float(is_score)}


def _perm_worker(md, strategy, space, fixed, categorical, settings_root, fill_tier, wf, seeds, bucket):
    """Runs in a worker process: everything arrives by value / by name (spawn-safe)."""
    from nqlab.settings import get_settings

    settings = get_settings(settings_root)
    grid = Grid.build(space, fixed, categorical)
    out = []
    for s in seeds:
        pm = permute_bars(md, int(s), bucket)
        out.append((int(s), family_statistics(pm, strategy, grid, settings, fill_tier, wf)))
    return out


@dataclass
class MCPTResult:
    real: dict[str, float]
    null: dict[str, list[float]]
    n_perm: int
    p_values: dict[str, float] = field(default_factory=dict)

    def finalize(self) -> "MCPTResult":
        for k in STATS:
            arr = np.array(self.null[k], float)
            r = self.real[k]
            arr = arr[np.isfinite(arr)]
            if not np.isfinite(r):
                self.p_values[k] = 1.0
            else:
                self.p_values[k] = float((1 + (arr >= r).sum()) / (1 + len(arr)))
        return self

    @property
    def p(self) -> float:
        return max(self.p_values.values()) if self.p_values else float("nan")

    def percentile(self, k: str) -> float:
        arr = np.array(self.null[k], float)
        return float((arr < self.real[k]).mean()) if len(arr) else float("nan")


def run_mcpt(md, strategy: str, space: dict, fixed: dict, categorical: tuple, settings, n_perm: int,
             fill_tier: int = 2, seed: int = 12345, checkpoint: dict[str, Any] | None = None,
             save_checkpoint: Callable[[dict[str, Any]], None] | None = None, checkpoint_every: int = 25,
             n_jobs: int = 1, bucket_minutes: int = 30, on_permutation: Callable[[int], None] | None = None) -> MCPTResult:
    wf = settings.section("validation")["walk_forward"]
    grid = Grid.build(space, fixed, categorical)
    cp = dict(checkpoint or {})
    real = cp.get("real") or family_statistics(md, strategy, grid, settings, fill_tier, wf)
    done: dict[int, dict[str, float]] = {int(k): v for k, v in (cp.get("done") or {}).items()}
    resumed_with = len(done)
    seeds = [seed + i for i in range(n_perm)]
    todo = [s for s in seeds if s not in done]
    chunks = [todo[i:i + checkpoint_every] for i in range(0, len(todo), checkpoint_every)]

    def _save():
        if save_checkpoint:
            save_checkpoint({"real": real, "done": {str(k): v for k, v in done.items()}, "n_perm": n_perm, "seed": seed,
                             "resumed_with": resumed_with})

    _save()
    if n_jobs == 1:
        for ch in chunks:
            for s, st in _perm_worker(md, strategy, space, fixed, categorical, str(settings.root), fill_tier, wf, ch, bucket_minutes):
                done[s] = st
                if on_permutation:
                    on_permutation(s)
            _save()
    else:
        from joblib import Parallel, delayed

        # submit in waves of n_jobs chunks so checkpoints land regularly
        for w in range(0, len(chunks), n_jobs):
            wave = chunks[w:w + n_jobs]
            res = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_perm_worker)(md, strategy, space, fixed, categorical, str(settings.root), fill_tier, wf, ch,
                                      bucket_minutes) for ch in wave)
            for part in res:
                for s, st in part:
                    done[s] = st
                    if on_permutation:
                        on_permutation(s)
            _save()
    null = {k: [done[s][k] for s in seeds] for k in STATS}
    return MCPTResult(real=real, null=null, n_perm=n_perm).finalize()
