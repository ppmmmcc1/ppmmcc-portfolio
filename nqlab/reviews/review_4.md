# Review 4

Scope: the code added in round 3 (tick data, trades-based profiles, ranking, random entry
with the operator's exits) and the first run of the suite on real Windows (GitHub
Actions, `windows-latest`). Two real defects, both fixed; the rest cosmetic.

### R4-1. The results package was never committed (CRITICAL, packaging)
`.gitignore` contained `results/` (unanchored), which matches `src/nqlab/results/` as
well as the top-level output folder. The ledger, DuckDB schema, metrics and report code
existed only on the build machine; every local test passed, and a fresh clone could not
even import. Windows CI caught it on its first run.
- Fix (a): rename the package. Churn, and the next unanchored pattern bites again.
- Fix (b): anchor every output pattern to the project root (`/results/`, `/reports/`,
  `/data/raw/*`, ...).
- **Chosen: (b)**, plus a check that every source file under `src/` is tracked.

### R4-2. Real trades leaked into permuted / synthetic datasets (HIGH, null-model leak)
`data/market.py:with_bars` kept `meta["trades"]`. A strategy with
`volume_profile.build_from: trades` would build profiles from the REAL tape on top of
PERMUTED bars, carrying real structure into the MCPT null (and into the random-walk
fill-optimism and GARCH diagnostics).
- Fix (a): permute the trades too. Possible in principle (trades follow their bar), but
  large and slow for a diagnostic.
- Fix (b): derived datasets never carry trades and are flagged `bars_only`; trades-based
  profiles fall back to bars there, and the MCPT computes the REAL statistic on the same
  bars-only copy, so real and null use identical feature construction.
- **Chosen: (b)**: consistent and honest; the precision lost applies equally to real and null.

### R4-3. The holdout look could not build exact profiles (MEDIUM)
`selection/holdout_cmd.py` built the holdout dataset without trades, so a
`build_from: trades` strategy would have errored during the one look (and a failed look
still counts as consumed). Holdout trades are now attached inside `holdout_access`.

### R4-4. Tier-3 replay on every report could take many minutes (LOW)
Reports now sample 60 sessions for the tier 3/4 rows; `nqlab fills-calibrate` keeps 200.

### Cosmetic
- Morning summary listed queue job ids instead of family / strategy names (fixed).

Nothing else found in this pass that changes a number a report shows.
