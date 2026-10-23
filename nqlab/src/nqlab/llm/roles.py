"""The four LLM roles. The LLM writes text and proposes hypotheses; code produces every
number. If Ollama is down, each role degrades to its deterministic part (or is skipped)."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nqlab.llm.client import LLMUnavailable, OllamaClient
from nqlab.llm.guardrails import filter_variant, scrub_selection, strip_unsourced_numbers
from nqlab.log import get_logger

log = get_logger("llm.roles")
NOISE_LINE = "An LLM will always find a pattern, including in pure noise. This is a hypothesis, not a finding."


def _client(settings) -> OllamaClient | None:
    c = OllamaClient(settings)
    return c if c.available() else None


# ---------------------------------------------------------------------------------------
# Role 1: spec linter
# ---------------------------------------------------------------------------------------
QUESTIONS = {
    "variant": "Is the MACD cross your ENTRY TRIGGER (market/stop entry) or a CONFIRMATION that must happen before a resting limit fills? These are different strategies; each is tested separately.",
    "context.direction_definition": "What exactly does 'MACD direction' mean on the 15m/1h: line above signal, histogram sign, line above zero, or line slope?",
    "context.agreement_rule": "Must the 15m and 1h agree, is either enough, or does the 1h override the 15m?",
    "context.hourly_alignment": "Are your hourly bars clock-aligned (10:00, 11:00 ET) or session-aligned (09:30, 10:30 ET)? They give materially different signals.",
    "bollinger.timeframe": "Are the bands on the 1-minute or the 5-minute chart?",
    "bollinger.stdev": "Does your charting platform use population or sample standard deviation for the bands? (TradingView uses population.)",
    "bollinger.reprice_each_bar": "Does your resting order move with the band every bar, or stay at the price where you placed it?",
    "volume_profile.source": "Which profile do you use: prior RTH session, prior full Globex session, a composite of N days, or today's developing profile?",
    "volume_profile.align_ticks": "How close (in ticks) must the band be to a profile level to count as 'aligned'?",
    "volume_profile.long_levels": "Which levels qualify for longs (VAL, POC, VAH)?",
    "volume_profile.short_levels": "Which levels qualify for shorts (VAL, POC, VAH)?",
    "volume_profile.order_price": "Is your limit at the band, at the level, or at whichever is the better price?",
    "volume_profile.build_from": "Your platform builds the profile from trades; nqlab can only approximate it from 1-minute bars until tick data is available. Accept 'bars' (approximate) for now?",
    "trigger.timeframes": "Which cross do you watch: 1m, 5m, or both?",
    "trigger.both_within_bars": "If both 1m and 5m must cross, within how many minutes of each other?",
    "trigger.confirmation_window_bars": "If the cross is a confirmation: how many minutes before the fill may it have happened and still count?",
    "orders.stop.type": "Where is your stop: fixed ticks, an ATR multiple, or just beyond the next profile level?",
    "orders.target.type": "Where is your target: the opposite band, the basis, the POC, or a fixed R multiple?",
    "orders.time_stop_et": "At what time do you flatten if neither stop nor target hits?",
    "orders.max_trades_per_day": "How many trades per day at most?",
    "sizing.instrument": "NQ or MNQ?",
    "sizing.risk_per_trade_usd": "How many dollars do you risk per trade (stop distance x contracts x point value)?",
}


def _question_for(field: str) -> str:
    if field in QUESTIONS:
        return QUESTIONS[field]
    return f"`{field}` is undecided. What value do you actually use? (Do not guess: a guess is a hidden parameter.)"


def lint_spec(settings, spec_path: Path, use_llm: bool = True) -> dict[str, Any]:
    from nqlab.strategies.spec import flatten, load_spec, undecided

    spec = load_spec(spec_path)
    flat = flatten(spec)
    questions = [{"field": f, "question": _question_for(f), "source": "code"} for f in undecided(spec)]
    # deterministic contradiction checks
    contra = []
    if flat.get("variant") == "limit" and flat.get("bollinger.reprice_each_bar") is False and flat.get("orders.cancel_if_band_moves_ticks") in (None, 0):
        contra.append("Frozen limit orders with no cancel rule can rest far from a moved band for the whole window. Intended?")
    if flat.get("orders.stop.type") == "beyond_level" and flat.get("orders.stop.buffer_ticks") in (None,):
        contra.append("stop.type is beyond_level: how many ticks beyond the level (stop.buffer_ticks)?")
    if flat.get("volume_profile.source") != "composite" and flat.get("volume_profile.composite_days") not in (None, "REQUIRED_UNDECIDED"):
        contra.append("composite_days is set but the profile source is not 'composite': which one is it?")
    for c in contra:
        questions.append({"field": "(consistency)", "question": c, "source": "code"})
    llm_qs: list[str] = []
    if use_llm and (cl := _client(settings)):
        text = Path(spec_path).read_text(encoding="utf-8")
        prompt = ("You review trading strategy specifications for ambiguity. Below is a YAML spec. List up to 8 "
                  "QUESTIONS the author must answer because a field is ambiguous, underspecified or contradictory. "
                  "Do not suggest values. Do not evaluate whether the strategy is good. One question per line.\n\n" + text)
        try:
            out = cl.generate("spec_lint", prompt, system="You only ask clarifying questions. You never propose numbers.")
            out, _ = strip_unsourced_numbers(out, text)
            llm_qs = [q.strip(" -*\t") for q in out.splitlines() if q.strip().endswith("?")][:8]
        except LLMUnavailable:
            pass
    for q in llm_qs:
        questions.append({"field": "(llm)", "question": q, "source": "llm"})
    return {"spec": str(spec_path), "n_undecided": len(undecided(spec)), "questions": questions,
            "runnable": len(undecided(spec)) == 0 and not contra}


# ---------------------------------------------------------------------------------------
# Role 2: losing-trade post-mortem
# ---------------------------------------------------------------------------------------
def trade_context(md, trades, settings, n: int = 20) -> list[dict[str, Any]]:
    """Code-computed context for the N worst trades (the LLM only reads this)."""
    import polars as pl

    from nqlab.features import compute

    worst = trades.sort("pnl_usd").head(n)
    atr = compute(md, "atr", length=14)["atr"].to_numpy()
    h15 = compute(md, "htf_macd", tf=15)["m15_macd_hist"].to_numpy()
    ts = md.bars["ts_utc"].dt.epoch("us").to_numpy()
    sess = md.sessions.select("session_date", "is_fomc", "is_release_0830", "is_release_1000")
    flags = {r["session_date"]: r for r in sess.iter_rows(named=True)}
    out = []
    for r in worst.iter_rows(named=True):
        i = int(np.searchsorted(ts, int(r["entry_ts"].timestamp() * 1e6)) - 1)
        f = flags.get(r["session_date"], {})
        m = int(r["entry_ct_minute"]) + 60
        out.append({"date": str(r["session_date"]), "time_et": f"{m // 60:02d}:{m % 60:02d}",
                    "side": "long" if r["side"] > 0 else "short", "pnl_usd": round(r["pnl_usd"], 2),
                    "r": None if r["r_multiple"] is None or not np.isfinite(r["r_multiple"]) else round(r["r_multiple"], 2),
                    "mae_pts": r["mae_points"], "mfe_pts": r["mfe_points"], "exit": r["exit_reason"],
                    "atr_1m_pts": round(float(atr[i]), 2) if i >= 0 and np.isfinite(atr[i]) else None,
                    "m15_hist_sign": int(np.sign(h15[i])) if i >= 0 and np.isfinite(h15[i]) else None,
                    "fomc": bool(f.get("is_fomc")), "release_0830": bool(f.get("is_release_0830")),
                    "release_1000": bool(f.get("is_release_1000"))})
    return out


def postmortem(settings, family_id: str, context: list[dict[str, Any]]) -> Path | None:
    """Writes a NEW pre-registration DRAFT in research_log/hypotheses/. It is never applied
    as a filter to the existing strategy."""
    cl = _client(settings)
    if cl is None:
        log.warning("post-mortem skipped: Ollama unavailable")
        return None
    facts = json.dumps(context, indent=1, default=str)
    prompt = ("Here are the worst trades of a strategy, with context computed by code. Describe in prose what they "
              "have in common. Phrase EVERY finding as a hypothesis to test on new data ('Hypothesis: ...'), never as a "
              "rule to apply. Do not compute statistics. Do not recommend changing the strategy.\n\n" + facts)
    text = cl.generate("postmortem", prompt, system="You write testable hypotheses. You never compute numbers.")
    text, removed = strip_unsourced_numbers(text, facts)
    text, _ = scrub_selection(text)
    d = settings.path("research_log") / "hypotheses"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{dt.date.today().isoformat()}_{family_id}_postmortem_{uuid.uuid4().hex[:6]}.yaml"
    doc = {"status": "DRAFT (not registered, not applied)", "source": f"LLM post-mortem of {family_id}",
           "warning": NOISE_LINE, "model": cl.model, "numbers_removed": removed,
           "hypotheses_prose": text,
           "to_test": "Register as a NEW family with `nqlab preregister`; it will join the trial count."}
    p.write_text(f"# {NOISE_LINE}\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10000), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------------------
# Role 3: variant generation
# ---------------------------------------------------------------------------------------
def generate_variants(settings, strategy: str, space: dict[str, list[Any]], fixed: dict[str, Any], n: int,
                      template_text: str = "") -> dict[str, Any]:
    """Ask for up to ``n`` structural variants restricted to the declared space. Every
    variant's grid is RESERVED in the trial ledger at generation time (origin=llm_variant),
    before anything runs, and the resulting DSR penalty is reported."""
    from nqlab.results.ledger import LedgerEntry, TrialLedger, trial_key
    from nqlab.selection.plateau import Grid
    from nqlab.validation.stats import expected_max_sharpe

    cap = int(settings.section("llm").get("max_variants_per_session", 20))
    ledger = TrialLedger(settings.path("results_db"))
    today = dt.date.today().isoformat()
    used_today = _variants_today(settings, today)
    n = max(0, min(n, cap - used_today))
    if n == 0:
        return {"variants": [], "note": f"per-session cap of {cap} LLM variants reached"}
    cl = _client(settings)
    if cl is None:
        return {"variants": [], "note": "Ollama unavailable; no variants generated"}
    prompt = ("Propose structurally different variants of this strategy family as JSON: "
              '{"variants": [{"name": str, "rationale": str, "param_space": {param: [values...]}}]}. '
              f"Use ONLY these parameters and ONLY these values: {json.dumps(space)}. Propose at most {n}. "
              "You may restrict which values are explored; you may not invent new parameters or values.\n\n" + template_text)
    raw = cl.generate("variants", prompt, system="You output JSON only.", json_mode=True)
    try:
        props = json.loads(raw).get("variants", [])[:n]
    except Exception:
        props = []
    n_before = ledger.n_trials()
    out = []
    for v in props:
        sub = filter_variant(v, space)
        if sub is None:
            continue
        vid = uuid.uuid4().hex[:10]
        fam = f"llmv_{vid}"
        grid = Grid.build(sub, fixed)
        ledger.append([LedgerEntry(trial_key(fam, strategy, cfg, vid), "llm_variant", "reserved", origin="llm_variant",
                                   family_id=fam, strategy=strategy, params=cfg, note=f"generated {today}")
                       for cfg in grid.configs])
        rationale, _ = strip_unsourced_numbers(str(v.get("rationale", "")), json.dumps(space))
        draft = {"status": "DRAFT llm_variant (reserved in the trial ledger)", "family_id": fam, "llm_variant_id": vid,
                 "strategy": strategy, "name": str(v.get("name", ""))[:80], "rationale": rationale,
                 "param_space": sub, "fixed": fixed, "warning": NOISE_LINE, "model": cl.model}
        d = settings.path("research_log") / "hypotheses"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{fam}.yaml").write_text(yaml.safe_dump(draft, sort_keys=False), encoding="utf-8")
        out.append(draft)
    n_after = ledger.n_trials()
    T = 250
    pen = {"n_trials_before": n_before, "n_trials_after": n_after,
           "sr0_daily_before": expected_max_sharpe(max(n_before, 1), 1 / T),
           "sr0_daily_after": expected_max_sharpe(max(n_after, 1), 1 / T)}
    pen["sr0_annualized_after"] = pen["sr0_daily_after"] * np.sqrt(252)
    return {"variants": out, "dsr_penalty": pen}


def _variants_today(settings, today: str) -> int:
    from nqlab.results.db import connect

    db = settings.path("results_db")
    if not db.exists():
        return 0
    with connect(db) as con:
        return int(con.execute("SELECT COUNT(DISTINCT family_id) FROM trial_ledger WHERE origin='llm_variant' "
                               "AND kind='llm_variant' AND CAST(ts_utc AS DATE) = CAST(? AS DATE)", [today]).fetchone()[0])


# ---------------------------------------------------------------------------------------
# Role 4: morning summary
# ---------------------------------------------------------------------------------------
def _deterministic_summary(facts: list[dict[str, Any]]) -> str:
    lines = []
    for f in facts:
        res = f.get("result") or {}
        stages = res.get("stages") or []
        tail = "; ".join(f"stage {s['stage']}: {s['status']} - {s['summary']}" for s in stages)
        lines.append(f"- **{f['job']}** ({f['type']}): {f['status']}" + (f". {tail}" if tail else "")
                     + (f" Error: {f['error'][:200]}" if f.get("error") else "")
                     + (f" Report: {res.get('report')}" if res.get("report") else ""))
    return "\n".join(lines)


def morning_summary(settings, facts: list[dict[str, Any]], batch: str | None = None) -> Path:
    date = batch or dt.date.today().strftime("%Y%m%d")
    det = _deterministic_summary(facts)
    body = det
    model = None
    cl = _client(settings)
    if cl is not None:
        src = json.dumps(facts, default=str, indent=1)
        prompt = ("Write a short plain-English morning summary of an overnight backtesting batch for a trader. Cover: "
                  "what ran, what passed, what failed and why, what is worth looking at. Use ONLY the facts below; every "
                  "number you mention must appear in them. Do not recommend trading anything.\n\n" + src)
        try:
            text = cl.generate("morning_summary", prompt, system="You narrate computed results. You never compute or rank.")
            text, removed = strip_unsourced_numbers(text, src)
            text, _ = scrub_selection(text)
            body = text + (f"\n\n_{removed} unsourced number(s) were removed from the LLM text._" if removed else "") \
                + "\n\n## Facts (computed by code)\n\n" + det
            model = cl.model
        except Exception as exc:
            log.warning("morning summary LLM failed (%s); deterministic summary only", exc)
    header = f"# Morning summary {date}\n\n" + (f"_Narrated by {model}; numbers come from code._\n\n" if model
                                                  else "_Ollama unavailable: deterministic summary._\n\n")
    out = settings.path("reports") / f"morning_{date}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + body + "\n", encoding="utf-8")
    od = settings.path("reports") / f"overnight_{date}"
    if od.exists():
        (od / out.name).write_text(header + body + "\n", encoding="utf-8")
    return out
