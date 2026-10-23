# nqlab: invariants for anyone (human or model) editing this code

nqlab exists to answer one question honestly: does the operator's NQ/MNQ strategy
have a measurable edge? A platform that flatters a bad strategy costs real money.
Every ambiguous default resolves pessimistically. Every uncertain result reads
INSUFFICIENT DATA rather than PASS. Convenience never beats statistical honesty.

## Non-negotiable invariants

1. **No same-bar fills.** An order submitted on the basis of bar N's close can only
   fill on bar N+1 or later. (Enforced structurally in `engine/kernel.py`; tested in
   `tests/test_engine_golden.py`.)
2. **Higher timeframes expose only the last fully closed bar.** At 10:07 the 15m bar
   covering 10:00-10:15 does not exist; the latest available is 09:45-10:00.
   HTF bars are indexed by their close timestamp and joined backward-asof
   (`features/htf.py`).
3. **Conservative fill model is the default.** Tier 2 (trade-through) is the headline.
   Tier 1 (touch) is only ever reported labelled as an optimistic upper bound.
4. **Every run is logged to the trial ledger**, including errored and abandoned runs
   and every LLM-generated variant (ledgered at generation time). The ledger is
   append-only and hash-chained (`results/ledger.py`). DSR pulls N from it.
5. **The LLM never selects strategies, tunes parameters, or computes metrics.** It
   writes text and proposes hypotheses. Numbers in LLM output that are not present
   in its input are stripped (`llm/guardrails.py`).
6. **The holdout set is never read** except through `nqlab holdout`, which records
   the consumption permanently (`data/holdout.py`, audit hook + store guard).
7. **Selection is on parameter plateaus, never on peak Sharpe** (`selection/plateau.py`).
8. All timestamps stored in UTC. All session logic in `America/Chicago`. No naive
   datetimes.

## Practical rules

- Windows-native: `pathlib` everywhere, no symlinks, no fork. Every parallel entry
  point has an `if __name__ == "__main__":` guard; strategies are resolved **by name**
  from `strategies/registry.py` inside workers, never passed as closures.
- Prop-firm rules and fees live in `config/topstep_50k.yaml`, never as constants in code.
- The operator's objective (2026-10-23) is farming: pass Combines, extract as much as
  possible from the funded accounts. The primary metric is therefore expected net dollars
  per purchased account after ALL fees (`risk/farming.py`), judged with the edge cut to
  its one-standard-error lower bound, and always shown next to the zero-edge trader.
  P(first payout) and every other metric stay beside it. Never rank by Sharpe.
- Every feature declares which price series it uses (unadjusted / back_adjusted /
  ratio_adjusted). Volume profile may only use unadjusted prices.
- A family that fails a gate is archived, not tweaked. A tweak is a new family with a
  new pre-registration that joins the trial count.
- Run `pytest` (fast) and `pytest -m slow` (null calibration) before trusting a change.
