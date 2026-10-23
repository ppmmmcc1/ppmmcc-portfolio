# nqlab

A local, Windows-native platform for backtesting intraday NQ/MNQ strategies, with
statistically honest validation, Topstep 50K rule simulation and an overnight batch
runner. It exists to answer one question: **does your strategy have a measurable edge,
or not?** It is built to say "no" or "not enough data" whenever that is the honest
answer.

Read `CLAUDE.md` for the non-negotiable invariants and `SPEC.md` for the full design.

---

## 1. Install (Windows, plain PowerShell, no WSL)

```powershell
cd path\to\nqlab
py -3.12 -m venv .venv            # Python 3.11 or 3.12
.venv\Scripts\activate
pip install -e ".[dev]"
pytest                            # fast suite (about 3 minutes)
pytest -m slow                    # null calibration + kill/resume (about 10 minutes)
```

Ollama is optional. If it is running at `http://localhost:11434`, nqlab picks an
instruct model in the 7B-14B range automatically (`nqlab llm status`). If it is not,
every LLM feature is skipped with a warning and everything else works.

## 2. First run

1. Put your 1-minute NQ CSV files in `data\raw\`.
2. Open `config\settings.yaml` and set `data.csv`:
   - `columns`: the header names in your file (`timestamp` can be one column or
     `[date, time]`). Unknown headers are guessed and the guess is reported loudly.
   - `source_timezone`: the timezone the timestamps are written in. **Check this.**
     The quality report warns if the volume spike is not at 08:30 CT, which almost
     always means this setting is wrong.
   - `bar_label`: `open` if a timestamp marks the start of the minute, `close` if the end.
3. `nqlab data ingest`

On the first ingest the **final 6 months are moved into `data\holdout\` and locked**.
The boundary never moves. Nothing in nqlab can read that folder except
`nqlab holdout <family> --confirm`, which records the look forever. Do not open it in
Excel or a notebook either; the moment you look, it stops being a holdout.

Then read the quality report (`nqlab data quality`). Excluded sessions are never
traded and never silently dropped; every report shows how many there were.

## 3. Daily workflow

```text
nqlab backtest orb                      # one run + HTML report (NOT validation)
nqlab preregister orb_v1 --strategy orb --hypothesis "Opening-range breakouts follow through."
nqlab run orb_v1 --auto --workers auto  # stages 1-6, stops at the first gate that does not pass
nqlab promote orb_v1                    # only after a PASS; refused otherwise
nqlab holdout orb_v1 --confirm          # stage 7: ONE look, recorded permanently
nqlab ledger stats                      # N, the number every DSR is deflated by
```

Every run, including errored and abandoned ones and every LLM-generated variant, goes
into the trial ledger. Registering a family reserves its whole grid in the ledger up
front, so trying an idea and then pretending you did not is impossible. A family that
fails a gate is **archived, not tweaked**; a tweak is a new family and raises N.

### Your strategy

`specs\bb_vp_macd.yaml` lists every decision the strategy needs, each marked
`REQUIRED_UNDECIDED`. nqlab refuses to run it until each one is filled:

```text
nqlab spec lint specs\bb_vp_macd.yaml   # lists the open decisions as questions
nqlab parity label 2024-03-05 09:47 long  # label 20-30 sessions by hand ...
nqlab parity compare orb_v1               # ... and see where the code disagrees with you
```

The limit-with-confirmation and cross-as-trigger readings are different strategies
(`variant: limit` / `variant: trigger`); register and test them separately. Any field
can be a list, which declares a grid dimension at pre-registration time.

Install the pre-commit hook so a half-specified spec cannot be committed by accident:
`python scripts\install_hooks.py`.

## 4. How to read a verdict block

Every report opens with one of three words and one sentence.

| Verdict | Meaning | What to do |
|---|---|---|
| **PASS** | Cleared every pre-registered gate through the Topstep simulation (and, if run, the holdout). | Paper trade it, then `nqlab parity import` your TopstepX fills to check the fill model against reality. A PASS is evidence, not proof. |
| **FAIL** | A gate rejected it; the sentence names the gate. | Stop. The family is archived. If you have a genuinely new idea, register a new family; it will count against every future DSR. |
| **INSUFFICIENT DATA** | Not enough trades, sessions, permutations, or not yet through the funnel. A single backtest is always at best this. | Get more data or keep collecting. Do not trade it. |

Things in every report that should change your mind on their own:

- **Fill tiers side by side.** If tier 1 (touch) is profitable and tier 2 (trade-through)
  is not, the report says: *"This strategy's profit comes from the fill assumption, not
  from the market."*
- **Fill-optimism diagnostic.** Profit on a pure random walk means the fill model, not
  the market, is generating it.
- **Ambiguous-bar share.** Above ~15% of trades the 1-minute result is not trustworthy.
- **Sharpe ± SE.** One year of daily data gives SE(annualized Sharpe) ≈ 1.0.
- **Sample-size table** with your strategy's row highlighted.
- **Topstep block.** P(pass) and P(first payout) next to the same trades with the edge
  removed. *A zero-edge trader passes this Combine roughly 40% of the time (gambler's
  ruin with a $2,000 floor and a $3,000 target). Passing is weak evidence of skill.*
  With the trailing MLL, the simulated zero-edge figure is lower (around 30%).
- **Sizing curve.** Risk per trade as a share of the *$2,000 distance to the MLL*, not of
  $50,000. P(first payout) usually peaks well below the size that maximizes expected
  profit; size for the former.

## 5. Overnight batches on Windows

Queue the night's work, then let Task Scheduler run it:

```text
nqlab queue batch --families orb_v1,fhh_v1
nqlab queue list
```

`scripts\run_overnight.bat` drains the queue: each job runs in its own process, crashes
are logged as `errored` (in the queue and the trial ledger) and the queue continues; a
permutation test checkpoints every 25 permutations and resumes after a crash or a closed
lid. In the morning read `reports\morning_YYYYMMDD.md` and the HTML reports in
`reports\overnight_YYYYMMDD\`.

**Task Scheduler**

1. Open *Task Scheduler* → *Create Task...* (not "Basic Task").
2. *General*: name `nqlab overnight`; select *Run whether user is logged on or not*.
3. *Triggers*: *New...* → *Daily*, e.g. 23:00.
4. *Actions*: *Start a program* → Program: `C:\path\to\nqlab\scripts\run_overnight.bat`;
   *Start in*: `C:\path\to\nqlab`.
5. *Conditions*: tick **Wake the computer to run this task**; untick *Start the task only
   if the computer is on AC power* only if you accept battery drain (better: stay plugged in).
6. *Settings*: tick *Run task as soon as possible after a scheduled start is missed*; set
   *Stop the task if it runs longer than* to 8 hours.

**Power settings** (so the laptop does not sleep mid-run)

- `run-queue` asks Windows not to sleep while it is running (`SetThreadExecutionState`),
  but that does not override closing the lid.
- *Control Panel → Power Options → Choose what closing the lid does*: set *When I close
  the lid* (Plugged in) to **Do nothing**, or leave the lid open.
- *Change plan settings → Change advanced power settings → Sleep → Allow wake timers*:
  **Enable** (needed for "Wake the computer to run this task").
- Or from an elevated prompt: `powercfg /change standby-timeout-ac 0` while batches run.

If the machine sleeps anyway, nothing is lost: the next run resumes interrupted jobs from
their checkpoints.

## 6. Upgrading data: the Databento path

Free 1-minute bars cannot show queue position, so they support fill tiers 1 and 2 only,
and the volume profile is approximate. Databento (`GLBX.MDP3`) adds real per-contract
bars (volume-based rolls), trades (tier 3 queue model, exact volume profile) and MBO
(tier 4 exact queue replay, used to calibrate tiers 2 and 3 on ~200 sessions).

1. `setx DATABENTO_API_KEY db-XXXXXXXX` (new terminal afterwards).
2. In `config\settings.yaml`: `data.source: databento`; check `data.databento.max_cost_usd`.
3. `nqlab data ingest`. **The cost of every request is estimated and printed before
   anything is downloaded**; requests above `max_cost_usd` are refused unless you add
   `--confirm-cost`.

The holdout boundary set by your first ingest does not move when the data source
changes.

## 7. What nqlab will not do

- Report tier-1 (touch) fills as a headline number.
- Let the LLM pick a strategy, choose a parameter, or produce a number (numbers in LLM
  text that are not in its input are removed, and its variants are ledgered before they
  run).
- Rank strategies by peak Sharpe (selection is on parameter plateaus; the primary ranking
  is P(first payout)).
- Re-run a stage that already passed or failed, skip a stage, or promote a failed family.
- Read the holdout outside `nqlab holdout`.

## 8. Known limitations (read before trusting anything)

- Topstep rules in `config\topstep_50k.yaml` and the release calendar in
  `config\events.yaml` were transcribed, not verified. Check them against Topstep's Help
  Center and an official release calendar (`nqlab data events-import`).
- A pre-spliced continuous CSV hides roll dates and gaps; indicators with lookbacks
  across a roll may be contaminated on roll days. Per-contract data fixes this.
- One position at a time; no partial exits yet.
- The engine resolves a bar that touches both stop and target as a loss. That is the
  right default for a 1-minute backtest, and it is why tick data matters.
