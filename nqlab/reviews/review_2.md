# Review 2

Scope: a line-by-line re-read of the kernel, the account simulator, the funnel, the
holdout command and the Windows runtime, after the review-1 fixes. Fewer findings than
review 1, but one of them is a genuine optimistic bias in the fill model.

---

### R2-1. Resting limits get price improvement on gaps (HIGH, silent optimism)
`engine/kernel.py:_fill_entry` (LIMIT branch) and `_exit_check` (target branch). When a
bar opens through a resting buy limit at P (open <= P - through), the kernel fills at
the OPEN, i.e. below P. Same for targets: a long target at T fills at an open above T.
A resting limit sits in the book; when the market moves through it, it is filled at its
own price. Only an order that was already marketable when submitted gets the market
price. On 1-minute bars inside RTH this is usually a tick, but at the open, around
releases and after halts it is real money, always in the strategy's favour.
- Fix (a): always fill limits at their limit price. Slightly pessimistic for orders that
  were marketable on arrival.
- Fix (b): fill at the limit price if the order was resting (previous close on the
  far side of the limit) and at min(open, P) only if it was marketable at submission
  (previous close already through the limit by at least a tick).
- **Chosen: (b)**: it is exactly what a matching engine does, and it never improves a
  resting order. Targets are always resting, so they always fill at the target.

### R2-2. `tzdata` is not a declared dependency (HIGH on Windows)
`calendar/sessions.py` and `parity.py` use `zoneinfo.ZoneInfo("America/Chicago")`.
Windows has no system timezone database; without the `tzdata` package this raises
`ZoneInfoNotFoundError` on the operator's laptop even though every test passes on Linux.
- Fix (a): catch and fall back to a fixed UTC-6 offset. Wrong half the year (DST).
- Fix (b): declare `tzdata` as a dependency.
- **Chosen: (b)**.

### R2-3. Reports of failed families show an arbitrary config (MEDIUM)
`runner/jobs.py:job_report` falls back to `configs[0]` when `selected_params_json` is
empty, which it is for every family that failed before its first PASS. The report then
describes a configuration the funnel never selected.
- Fix (a): compute the plateau pick in the report job. Duplicates logic.
- Fix (b): store the plateau pick after every stage, whatever its status.
- **Chosen: (b)**.

### R2-4. LLM-variant drafts cannot be registered with their reserved trial keys (MEDIUM)
`llm/roles.py:generate_variants` reserves keys with `variant=<id>`; `nqlab preregister`
has no way to pass that id, so registering the draft creates a second set of keys and
the trials are counted twice. Over-counting is the safe direction, but it makes the
DSR penalty of LLM variants look worse than it is and hides the link to the generation.
- Fix: `nqlab preregister --from-draft research_log/hypotheses/llmv_x.yaml`, which
  keeps `family_id`, `llm_variant_id` and `origin=llm_variant`.

### R2-5. Fill tiers 3/4 exist but are unreachable from the CLI and reports (MEDIUM)
The classes and tick replay are tested, but nothing stores tick data or runs them.
"Every report shows results under all available tiers" is only true for tiers 1-2.
- Fix: `nqlab data ingest-ticks` (Databento trades/mbo -> parquet, cost-gated),
  `nqlab fills calibrate` (tiers 2/3/4 on a session sample), and a tier-3 row in the
  report whenever trades data covers part of the period.

### R2-6. DuckDB lock contention during an overnight run (MEDIUM, Windows)
`results/db.py:connect` retries for ~30 s. If the operator runs a read command while a
worker writes, the worker can give up and the job errors. Lengthen the retry window to
about two minutes.

### R2-7. DLL check ignores fees at the worst point (LOW)
`risk/topstep.py:_trade_day`: the MLL check includes fees at the worst point, the DLL
check does not. Make them consistent (include fees).

### R2-8. LLM number filter also removes clock times not in its input (LOW, accepted)
"09:30" in LLM prose becomes two removed numbers unless the times are in the facts.
Acceptable: the facts passed to the LLM include entry times, and removing an unsourced
time is the safe failure.

Checked again and correct: no same-bar re-entry after an exit (pendings are cancelled on
fill and signals are ignored while in a position); stop gaps fill at the worse open;
MLL check per trade via MAE is exact for one position at a time; holdout consumption is
recorded before any read; promotion now requires a PASS in the hash-chained ledger.
