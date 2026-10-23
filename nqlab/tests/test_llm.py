import json

import httpx
import pytest

from nqlab.llm import roles
from nqlab.llm.client import OllamaClient, choose_model
from nqlab.llm.guardrails import REMOVED, filter_variant, scrub_selection, strip_unsourced_numbers
from nqlab.results.ledger import TrialLedger

TAGS = {"models": [{"name": "nomic-embed-text", "details": {"parameter_size": "137M"}},
                   {"name": "llama3.1:70b", "details": {"parameter_size": "70B"}},
                   {"name": "qwen2.5:7b-instruct", "details": {"parameter_size": "7.6B"}}]}


def mock_transport(response_text: str):
    def handler(req: httpx.Request):
        if req.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS)
        if req.url.path == "/api/generate":
            return httpx.Response(200, json={"response": response_text})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


@pytest.fixture()
def fake_ollama(monkeypatch):
    def install(text):
        orig = OllamaClient.__init__

        def init(self, settings, base_url=None, transport=None):
            orig(self, settings, base_url, mock_transport(text))
        monkeypatch.setattr(OllamaClient, "__init__", init)
    return install


def test_choose_model_prefers_instruct_in_range():
    assert choose_model(TAGS["models"]) == "qwen2.5:7b-instruct"


def test_strip_unsourced_numbers():
    out, n = strip_unsourced_numbers("Sharpe was 1.8 on 120 trades and win rate 63%", "trades: 120")
    assert "120" in out and "1.8" not in out and "63%" not in out and n == 2 and REMOVED in out


def test_scrub_selection():
    t, n = scrub_selection("Results are mixed. You should trade ORB tomorrow. Done.")
    assert n == 1 and "ORB tomorrow" not in t


def test_variant_cannot_escape_space():
    space = {"a": [1, 2, 3], "b": ["x", "y"]}
    v = filter_variant({"param_space": {"a": [2, 99], "c": [5], "b": "y"}}, space)
    assert v == {"a": [2], "b": ["y"]}


def test_everything_runs_with_ollama_down(settings):
    settings.raw["llm"]["base_url"] = "http://127.0.0.1:9"   # nothing listens here
    assert not OllamaClient(settings).available()
    r = roles.lint_spec(settings, settings.path("specs") / "bb_vp_macd.yaml")
    assert r["n_undecided"] > 40 and not r["runnable"]
    assert all(q["source"] == "code" for q in r["questions"])
    p = roles.morning_summary(settings, [{"job": "j1", "type": "backtest", "status": "done", "error": "", "result": {}}], "20990101")
    assert "deterministic" in p.read_text(encoding="utf-8")
    assert roles.generate_variants(settings, "orb", {"or_minutes": [5, 15]}, {}, 3)["variants"] == []


def test_lint_lists_undecided_as_questions(settings, fake_ollama):
    fake_ollama("Is the band computed on closes or typical price?\nUse 20 periods.")
    r = roles.lint_spec(settings, settings.path("specs") / "bb_vp_macd.yaml")
    fields = {q["field"] for q in r["questions"]}
    assert "context.hourly_alignment" in fields and "bollinger.stdev" in fields and "variant" in fields
    assert all("?" in q["question"] for q in r["questions"])
    llm = [q for q in r["questions"] if q["source"] == "llm"]
    assert len(llm) == 1  # the non-question line is dropped


def test_variants_ledgered_at_generation(settings, fake_ollama):
    fake_ollama(json.dumps({"variants": [{"name": "short OR", "rationale": "tighter range, Sharpe 2.5",
                                          "param_space": {"or_minutes": [5], "target_r": [1.0, 7.0]}},
                                         {"name": "long OR", "rationale": "wider", "param_space": {"or_minutes": [30]}}]}))
    space = {"or_minutes": [5, 15, 30], "target_r": [1.0, 2.0]}
    led = TrialLedger(settings.path("results_db"))
    before = led.n_trials()
    res = roles.generate_variants(settings, "orb", space, {}, 2)
    assert len(res["variants"]) == 2
    # variant 1: or=5 x target=1.0 (7.0 discarded) -> 1 config; variant 2: or=30 x 2 targets -> 2 configs
    assert led.n_trials() - before == 3
    assert "2.5" not in res["variants"][0]["rationale"]
    assert res["dsr_penalty"]["sr0_daily_after"] > res["dsr_penalty"]["sr0_daily_before"]
    st = led.stats()
    assert st["by_origin"]["llm_variant"] == 3


def test_morning_summary_numbers_come_from_code(settings, fake_ollama):
    fake_ollama("ORB failed screening with expectancy -6.68. ORB had a Sharpe of 3.1. You should trade ORB.")
    facts = [{"job": "orb", "type": "family", "status": "done", "error": "", "result": {"expectancy": -6.68}}]
    text = roles.morning_summary(settings, facts, "20990102").read_text(encoding="utf-8")
    assert "-6.68" in text and "3.1" not in text and "You should trade" not in text


def test_postmortem_writes_draft_with_noise_line(settings, fake_ollama):
    fake_ollama("Hypothesis: losers cluster on release days.")
    p = roles.postmortem(settings, "fam", [{"date": "2024-01-02", "pnl_usd": -100.0}])
    assert p.parent.name == "hypotheses" and roles.NOISE_LINE in p.read_text(encoding="utf-8")


def test_every_llm_call_logged(settings, fake_ollama):
    fake_ollama("Is X defined?")
    roles.lint_spec(settings, settings.path("specs") / "bb_vp_macd.yaml")
    logs = list((settings.path("research_log") / "llm").glob("*_spec_lint.json"))
    assert logs and json.loads(logs[0].read_text(encoding="utf-8"))["model"] == "qwen2.5:7b-instruct"


def test_llm_draft_registers_with_reserved_keys(settings, fake_ollama):
    import yaml

    from nqlab.selection.family import FamilyStore
    from nqlab.selection.prereg import make_prereg

    fake_ollama(json.dumps({"variants": [{"name": "short OR", "rationale": "tighter",
                                          "param_space": {"or_minutes": [5]}}]}))
    space = {"or_minutes": [5, 15, 30], "target_r": [1.0, 2.0]}
    from nqlab.strategies.registry import get_strategy
    fixed = {k: v for k, v in get_strategy("orb").defaults().items() if k not in space}
    res = roles.generate_variants(settings, "orb", space, fixed, 1)
    led = TrialLedger(settings.path("results_db"))
    n0 = led.n_trials()
    dr = res["variants"][0]
    pre = make_prereg(settings, dr["family_id"], "orb", "LLM variant test", param_space=dr["param_space"],
                      fixed=dr["fixed"], origin="llm_variant", llm_variant_id=dr["llm_variant_id"])
    FamilyStore(settings).register(pre)
    assert led.n_trials() == n0          # registering the draft did not double count
