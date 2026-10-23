# Review 5

Scope: a second read of the operator-strategy semantics (spec 13.1) and the account rules
(spec 7), plus the results of the first green-ish Windows CI runs.

### R5-1. Repricing defeated `orders.expiry_minutes` (MEDIUM)
`strategies/bb_vp_macd.py` re-sends the limit every bar while its conditions hold (that
is how "reprice each bar" and "keep the frozen order alive" are expressed). Every re-send
resets the engine's order clock, so an order declared to expire after N minutes could
rest for the whole window. That makes fills more likely than the operator's own rule
allows.
- Fix (a): give the engine a "replace keeps the original timestamp" flag. Touches the
  kernel for one strategy's needs.
- Fix (b): limit each signal episode (run of consecutive "order wanted" bars) to its
  declared lifetime in the strategy, then let the lapse rule cancel it.
- **Chosen: (b)**; the kernel stays simple and the rule is enforced where it is defined.
  Regression test: `tests/test_bb_vp_macd.py::test_expiry_survives_repricing`.

### R5-2. Windows-only failures found by CI (fixed in round 4/5)
- Tests relied on `python -m pytest` putting the project on `sys.path`; plain `pytest`
  (what the README tells the operator to run) failed. Fixed with `pythonpath = ["."]`.
- One test read the UTF-8 report with the Windows default code page. Source code already
  specified UTF-8 everywhere; tests now do too, and `run_overnight.bat` sets
  `PYTHONUTF8=1`.
- The kill/resume test now runs on Windows on every push (process handling and
  `pid_alive` are the Windows-specific code paths).

### Checked, nothing found
- Limit variant: confirmation must precede the fill within N bars (order is only live
  while a qualifying cross happened in the last N bars); frozen orders cancel when the
  band moves more than M ticks; orders cancel at the window end.
- Trigger variant: the cross itself triggers a market entry at the next open, only near
  a band extreme aligned with a qualifying level.
- Account rules: trailing MLL on end-of-day highs with lock; MLL checked at every
  trade's worst point including fees; DLL optional and day-stopping; consistency blocks
  a pass rather than failing it; XFA payouts on both paths.

Remaining findings in this pass are cosmetic. Per spec 16 the loop stops here; the
limitations that remain are documented in the README ("Known limitations").
