# Review 1: senior quant review of the initial build

Reviewer stance: this code decides whether a 21-year-old risks money he does not have.
Every finding below is either a way the platform could flatter a strategy, a way the
operator could fool himself, or a way it breaks on his Windows laptop. Paths are
relative to `src/nqlab/`.

Test state at review time: 89 passed (incl. slow null calibration and kill/resume).
Passing tests prove only what they test; several of the findings are untested paths.

---

## A. Lookahead and causality

### A1. Cross / slope detection on gathered engine bars compares across sessions (HIGH)
`strategies/bb_vp_macd.py:35,44-48`. `_direction(..., "line_slope")` uses `np.diff` and
`_crosses` uses `np.roll` on arrays already gathered onto engine bars. Engine bars are
only the trading window, so the "previous" value of the first bar of a session is the
last window bar of the *previous* session (~17 hours earlier). That manufactures
spurious MACD crosses / slopes at every window open. Not lookahead, but a fabricated
signal that the operator does not see on his chart.
- Fix (a): compute direction and crosses on the full-length feature arrays (aligned to
  `md.bars`, continuous through the overnight session), then gather. Correct, cheap.
- Fix (b): mask the first bar of each session window. Hides the symptom but still
  differs from the chart (a real cross at 08:00 would be lost).
- **Chosen: (a)**; it reproduces what the operator's chart shows bar for bar.

### A2. Quality exclusions use the rest of the session (LOW, documented)
`data/quality.py`: a bad tick at 14:00 excludes the whole session, including the 08:30
entry that happened before it. This is data cleaning, not a signal, and it is reported;
but it is a selection on future information.
- Fix (a): exclude only bars after the bad bar. Leaves a trade on a session whose data
  is known corrupt.
- Fix (b): keep whole-session exclusion, but report the excluded sessions' count in
  every report and list the dates (already done), and make the spike rule narrow (only
  reverting one-bar spikes; already done).
- **Chosen: (b)**, unchanged; corrupt sessions should not be traded in a backtest at all.

Re-derived by hand and found clean: HTF join (bucket close label, backward as-of on the
1m close), opening range (null until the window closes), prior-session profiles
(mapped to the following session only), causal additive adjustment (gaps applied from
the roll forward), volume roll (D's volume switches D+1), engine order of events
(signals at close -> pending from i+1).

## B. Silent optimism

### B1. Risk-sized trades that round to zero contracts vanish silently (MEDIUM)
`engine/simulate.py:81`, `engine/backtest.py:76`. With `risk_usd` sizing, a wide stop
gives `floor(risk / (stop * pv)) == 0` and the trade is dropped. Dropping
systematically the wide-stop (high-volatility) trades can flatter the result, and the
count is only kept in `meta`.
- Fix (a): size to at least 1 contract. Overstates risk, misrepresents the operator's rule.
- Fix (b): keep the drop (it is the operator's rule) but report the count and the P&L
  those trades would have made at 1 contract, so the effect is visible.
- **Chosen: (b)**; honest about the rule and its consequence.

### B2. Family t-stat gate is impossible for strategies without a stop (MEDIUM)
`selection/funnel.py:198`. `t_stat_r` is NaN when there is no stop (R undefined), so
`session_hold`-style families sit at INSUFFICIENT DATA forever, which hides whether
they actually fail.
- Fix (a): fall back to the t-stat of daily P&L when R is undefined.
- Fix (b): skip the t gate when R is undefined. Removes a guard.
- **Chosen: (a)**.

### B3. Monte Carlo methods in 8.4 are implemented but never reported (MEDIUM)
`validation/bootstrap.py`, `validation/synthetic.py:garch_synthetic` are unused by the
report and the funnel. The spec requires the reshuffle drawdown distribution (labelled
"not a test of edge"), trade-bootstrap CI and P(final < 0), block-bootstrap Sharpe CIs
and a GARCH null.
- Fix (a): add a "Monte Carlo" section to every report.
- Fix (b): add a separate `nqlab montecarlo` command. Easy to forget to run.
- **Chosen: (a)**.

### B4. Volume-profile "approximate" note never fires (LOW)
`results/report.py:258` looks for parameter names starting with `vp_`; the operator's
spec uses `volume_profile.*`. The report would never say the profile is approximate.
- Fix: detect `volume_profile.build_from == "bars"` or any `vp_*` feature use.

### B5. Holdout information leaks through the quality report (LOW)
`data/ingest.py:95`. The markdown quality report lists excluded sessions and counts
computed over the whole history, holdout included. It reveals nothing about strategy
performance, but it is still a look at locked data.
- Fix (a): run the checks on research data only. Then holdout sessions are never
  quality-checked before the one look.
- Fix (b): check everything, but write only research-period dates; holdout sessions
  appear as a count.
- **Chosen: (b)**.

## C. Trial-ledger leaks

### C1. Several code paths run backtests without logging them (HIGH)
`cli.py:435,460,484` (`parity compare`, `parity import`, `llm postmortem`) call
`run_backtest` directly. An errored `nqlab backtest` is not logged either, and there is
no `abandoned` status anywhere (Ctrl-C during a stage leaves no trace).
- Fix (a): log inside `run_backtest` itself. It is also used inside permutation loops
  on synthetic data, where per-run logging would flood the ledger with nulls that must
  not count.
- Fix (b): one helper `logged_backtest()` used by every CLI/LLM path; `run_stage` and
  the CLI catch `KeyboardInterrupt` and append an `abandoned` row; `backtest` errors are
  appended as `errored`.
- **Chosen: (b)**; keeps synthetic-data runs out of N while closing every real-data path.

### C2. Archive state lives in a mutable table (HIGH)
`selection/family.py:112,163`. `families.status` is an ordinary DuckDB column. One
`UPDATE families SET status='active'` (or deleting a `stage_results` row) un-archives a
failed family, and nothing would notice.
- Fix (a): hash-chain `stage_results` like the ledger. More machinery.
- Fix (b): write every stage outcome into the (already hash-chained) ledger and derive
  "archived" from the ledger on every action, ignoring the table if it disagrees.
- **Chosen: (b)**; reuses the tamper-evident structure that already exists.

## D. Windows-specific

### D1. MCPT ships cached features to every worker (MEDIUM)
`validation/mcpt.py:113`. `md` is pickled to loky workers with `meta["_feature_cache"]`
and `meta["_engine_bars"]`, multiplying memory per worker. `runner.max_worker_memory_mb`
is configured but unused (dead config).
- Fix: send a cache-free copy, and cap `n_jobs` so `n_jobs x estimated md size x 3`
  fits `max_worker_memory_mb`.

### D2. `run_overnight.bat` is missing; the spec's `%DATE%` slicing is locale-dependent (MEDIUM)
On a non-US Windows locale `%DATE:~-4%...` produces garbage file names.
- Fix (a): use the spec's line verbatim. Breaks on other locales.
- Fix (b): get the date from PowerShell `Get-Date -Format yyyyMMdd`.
- **Chosen: (b)**, and create `reports\` first.

### D3. Kill/resume test is POSIX-only (LOW)
`tests/test_runner_queue.py` uses `os.killpg`. Make it kill the runner and the worker
PIDs explicitly so it runs on Windows too.

## E. Statistical correctness

### E1. StepM index mapping is fragile (LOW)
`validation/stats.py:117` parses labels with string splitting. Use integer indices directly.

### E2. `daily.consistency_ratio` is always NULL (LOW)
Compute best-day / total-profit per Combine attempt in `combine_trace`.

Checked and correct: MCPT re-optimizes on every permutation; DSR's N comes from the
ledger (global, distinct trial keys, errored included); CPCV purges with an embargo on
both sides; PBO selects the IS winner with the same plateau rule; Sharpe SE uses
Mertens with raw kurtosis.

## F. The operator fooling himself

- F1 = C2 (un-archiving by editing the DB).
- F2. Nothing tells him the holdout files are off limits when he browses the folder.
  Write a `README_LOCKED.txt` into `data/holdout/` at ingest.

## G. Missing pieces / dead code

- G1. README, Task Scheduler and power settings, pre-commit spec-lint hook: missing.
- G2. `features/htf.py:32` dead branch.
- G3. No unit test for `family_verdict`, for ledger-derived archiving, or for the
  coverage targets (>90% engine/features/risk).
