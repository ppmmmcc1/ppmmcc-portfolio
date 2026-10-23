# Review 3: the spec, re-read clause by clause against the code

Reviews 1-2 read the code; this pass read SPEC.md top to bottom and asked, for every
sentence, "where is this implemented, and where is it tested?". Status after the fixes
in this round. Paths relative to `src/nqlab/`.

## Gaps found and fixed in this round

| Spec | Gap | Candidate fixes (chosen in bold) | Where now |
|---|---|---|---|
| 3.4 | No cross-check of 1m bars against trades-rebuilt bars when both sources exist | silently prefer one source / **rebuild bars from trades and report every disagreement with examples** (deciding which source is wrong needs a human) | `data/quality.py:cross_check_trades`, run by `nqlab data ingest-ticks` |
| 5 | Volume profile never built from trades even when trades exist | always approximate / **exact profile from trades at 0.25 bins with `approximate=False`, bars only as fallback** | `features/profile.py` (`build_from`), trades attached lazily by `MarketData.from_store` |
| 13 | Linter only caught `REQUIRED_UNDECIDED`; deleting a field from the YAML made the decision silently vanish | trust the template / **canonical required-field list + allowed values; omitted or invalid fields are questions and the strategy refuses to run** | `strategies/spec.py:REQUIRED_FIELDS, ALLOWED`; `llm/roles.py:lint_spec`; `_BBVPMacd.validate` |
| 13.1 | `sizing.risk_pct_of_mll_buffer` declared but never used | drop the field / **use it as the Topstep sizing policy (fraction of the live MLL buffer)** | `Strategy.topstep_policy`, funnel stage 6, report |
| 13.2 | "Random entry with **his** exit logic" only approximated with fixed ticks | keep the approximation / **a random-entry strategy that reuses his spec's window, skip days, stop/target types, breakeven, time stop and caps; only the entry is random** | `strategies/bb_vp_macd.py:RandomEntrySpecExits` (`random_entry_spec`) |
| 9 | Primary ranking metric P(first payout) had no ranking view | Sharpe leaderboard (forbidden) / **`nqlab rank`: ordered by P(first payout), with zero-edge benchmark, P(pass), Sharpe ± SE, DSR, expectancy R and $, PF, max losing streak beside it** | `selection/ranking.py`, `cli.py:rank` |
| 11 (role 2) | Post-mortem context lacked "distance to level" | **distance from entry to the nearest prior-RTH POC/VAH/VAL** | `llm/roles.py:trade_context` |
| 14 | Round-trip test had no known answer | **deterministic staircase data where buy-the-open/sell-15:00 must earn exactly 8.75 pts / $171.22 per session** (the test caught my own hand calculation: the 15:00 CT exit bar is after the cash close and pays 2 ticks) | `tests/test_round_trip.py` |
| 6 | "10 MNQ costs $12.20 vs $3.78" only appeared when >10 micros were used | **always stated in the assumptions** | `results/report.py` |
| 8.3 | PBO > 0.5 not described as "worse than random" | **said in the stage-5 failure sentence** | `selection/funnel.py` |
| 10 | run_id did not include a strategy version | **`Strategy.version` hashed into every run_id** | `engine/backtest.py:make_run_id` |
| 17 | `run_overnight.bat` could not be exercised on Linux | **queued a real batch (2 families + 4 baselines + summary) and drained it with the same command the .bat runs; added a Windows CI workflow** | `.github/workflows/nqlab.yml` |

Review-2 items fixed in the same round: resting limits no longer receive price
improvement on gaps (R2-1), `tzdata` declared (R2-2), plateau pick stored after every
stage (R2-3), `preregister --from-draft` keeps LLM-variant trial keys (R2-4), tick data
ingest + `nqlab fills-calibrate` + tier-3/4 rows in reports (R2-5), longer DuckDB lock
wait (R2-6), DLL check includes fees (R2-7).

## Re-verified clause by clause (no change needed)

- 0: all eight invariants hold; `CLAUDE.md` carries them.
- 1: pathlib only; no fork/symlink/nohup; workers resolve strategies by name; Ollama
  detected via `/api/tags`, everything runs without it (tested with a dead port).
- 3: source protocol; CSV mapper; yfinance marked low confidence; Databento cost printed
  before download (mock-tested); canonical schema; partitioned parquet; volume roll with
  CME fallback recorded in data_version; three price series; lockbox with tests.
- 4: CME calendar + overrides, maintenance break, RTH, 15:10 flatten, early closes, DST
  mismatch weeks tested, release calendar (flagged unverified).
- 5: all indicators; explicit stdev type; HTF close-indexed backward joins with both
  hourly alignments; 200-example truncation property test plus boundary cases.
- 6: all order types; four fill classes; ambiguous-bar rule counted and warned; golden
  tests (tier 1 vs 2 limit, both legs in one bar, 15:10 flatten, maintenance break).
- 7: rules from YAML; real-time MLL via MAE (cross-checked against an independent
  reference on 300 random cases); every-session + 10,000-path Monte Carlo; XFA payouts;
  sizing search on the MLL buffer; mandatory zero-edge line.
- 8: walk-forward (rolling + anchored, WFE, stability), CPCV, PSR/DSR with ledger N,
  PBO, SPA/StepM, all 8.4 methods, MCPT re-optimizing on every permutation, fill-optimism
  diagnostic, null calibration (passes at the nominal rate; planted edge detected),
  sample-size table and Sharpe SE.
- 9-12: funnel stages 0-7 with refusals; DuckDB tables; append-only hash-chained ledger;
  self-contained HTML report with verdict block; four LLM roles with guardrails; SQLite
  queue with checkpoints, dependencies, crash isolation and kill/resume test.

## Remaining (accepted) limitations

- The engine holds one position at a time and has no partial exits; the spec refuses to
  run a strategy that declares partials rather than approximating them.
- Release calendar and Topstep rules are transcribed, not verified (flagged in every report).
- Bar-level results cannot resolve intrabar order; tiers 3/4 need Databento tick data.
