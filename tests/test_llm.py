from resumesort.llm import (
    TinkerLLM,
    fallback_interview_questions,
    parse_json_object,
    sanitize_claims,
    strip_reasoning,
)
from resumesort.schemas import ClaimVerdict, EvidenceItem, FraudSignal


def test_parse_json_object_from_noisy_output():
    data = parse_json_object('Here is JSON: {"verdict": "SUPPORTED", "confidence": 0.8}')
    assert data["verdict"] == "SUPPORTED"
    assert data["confidence"] == 0.8


def test_parse_json_object_strips_think_blocks():
    data = parse_json_object('<think>let me reason...</think>{"verdict": "REFUTED"}')
    assert data["verdict"] == "REFUTED"


def test_strip_reasoning_removes_unclosed_think():
    assert strip_reasoning("prefix <think>endless reasoning") == "prefix "
    assert strip_reasoning("<think>a</think>answer") == "answer"


def test_no_key_fallback(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    llm = TinkerLLM(enabled=True)

    assert llm.status.enabled is False
    assert llm.extract_claims("resume", ["Built FastAPI app"], max_claims=3) == ["Built FastAPI app"]


def test_judge_claim_resolves_evidence_index(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    llm = TinkerLLM(enabled=True)
    canned = (
        '{"verdict": "SUPPORTED", "confidence": 0.9, "evidence_index": 2,'
        ' "evidence_quote": "uses FastAPI", "explanation": "readme mentions it"}'
    )
    monkeypatch.setattr(TinkerLLM, "complete", lambda self, *a, **k: canned)

    evidence = [
        EvidenceItem("languages", "repo-a", "https://github.com/x/repo-a", "Python"),
        EvidenceItem("readme", "repo-b", "https://github.com/x/repo-b", "FastAPI service"),
    ]
    verdict = llm.judge_claim("Built a FastAPI service", evidence)

    assert verdict is not None
    assert verdict.verdict == "SUPPORTED"
    assert verdict.evidence_source == "https://github.com/x/repo-b"
    assert verdict.evidence == "uses FastAPI"


def test_sanitize_drops_junk_claims():
    tagged = [
        {"claim": "Maintains a GPA of 3.4/4.0", "kind": "experience"},
        {"claim": "Maintains a public GitHub repository at https://github.com/PP-695", "kind": "project"},
        {"claim": "Possesses 7+ years of software engineering experience", "kind": "experience"},
        {"claim": "Built a REST API with FastAPI", "kind": "project"},
    ]
    result = sanitize_claims(tagged, max_claims=8)
    assert [item["claim"] for item in result] == ["Built a REST API with FastAPI"]


def test_sanitize_splits_compound_skill_lists():
    tagged = [{"claim": "Proficient in Python, Rust, FastAPI, MongoDB, and Machine Learning", "kind": "skill"}]
    result = sanitize_claims(tagged, max_claims=8)
    claims = [item["claim"] for item in result]
    assert len(claims) == 5
    assert "Proficient in Rust" in claims
    assert all(item["kind"] == "skill" for item in result)


def test_sanitize_keeps_normal_claims_and_dedupes():
    tagged = [
        {"claim": "Built a chat backend with Redis", "kind": "project"},
        {"claim": "Built a chat backend with Redis", "kind": "project"},
    ]
    result = sanitize_claims(tagged, max_claims=8)
    assert len(result) == 1


def test_truncation_retries_with_doubled_budget(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    llm = TinkerLLM(enabled=True)
    calls = []

    def fake_request(self, prompt, max_tokens, temperature, system, model, json_mode):
        calls.append(max_tokens)
        if len(calls) == 1:
            return "", "length"
        return '{"ok": true}', "stop"

    monkeypatch.setattr(TinkerLLM, "_request", fake_request)
    out = llm.complete("prompt", max_tokens=400)
    assert calls == [400, 800]
    assert out == '{"ok": true}'
    assert llm.truncations == 1


def test_parse_failures_counted(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    llm = TinkerLLM(enabled=True)
    monkeypatch.setattr(TinkerLLM, "complete", lambda self, *a, **k: "not json at all")
    data = llm._complete_json("prompt", max_tokens=100)
    assert data == {}
    assert llm.parse_failures == 1
    assert llm.status.parse_failures == 1


def test_extract_claims_tagged_parses_kinds(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    llm = TinkerLLM(enabled=True)
    canned = (
        '{"claims": [{"claim": "Built a chat backend", "kind": "project"},'
        ' {"claim": "Reduced latency by 40%", "kind": "metric"},'
        ' {"claim": "Weird kind", "kind": "banana"}]}'
    )
    monkeypatch.setattr(TinkerLLM, "complete", lambda self, *a, **k: canned)
    tagged = llm.extract_claims_tagged("resume", [], max_claims=8)
    kinds = {item["claim"]: item["kind"] for item in tagged}
    assert kinds["Built a chat backend"] == "project"
    assert kinds["Reduced latency by 40%"] == "metric"
    assert kinds["Weird kind"] == "project"  # unknown kind normalized


def test_fallback_questions_vary_and_break_on_word_boundaries():
    gaps = [
        ClaimVerdict(claim=f"Implemented a quantum digital signature scheme number {i} " + "x" * 100,
                     verdict="NOT_ENOUGH_INFO", confidence=0.3)
        for i in range(3)
    ]
    signals = [FraudSignal("account_age_gap", "warn", "Account newer than claimed experience", "detail")]
    questions = fallback_interview_questions(gaps, signals, k=4)

    assert len(questions) == 4
    # Templates rotate: consecutive claim questions must differ in phrasing.
    assert questions[0]["question"].split('"')[-1] != questions[1]["question"].split('"')[-1]
    # No mid-word truncation: the shortened claim ends with '...' not a cut word.
    assert "..." in questions[0]["question"]
    assert all(q.get("listen_for") for q in questions)


def test_judge_claim_index_out_of_range(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    llm = TinkerLLM(enabled=True)
    canned = '{"verdict": "NOT_ENOUGH_INFO", "confidence": 0.4, "evidence_index": 99}'
    monkeypatch.setattr(TinkerLLM, "complete", lambda self, *a, **k: canned)

    evidence = [EvidenceItem("readme", "repo-a", "https://github.com/x/repo-a", "text")]
    verdict = llm.judge_claim("Some claim", evidence)

    assert verdict is not None
    assert verdict.evidence_source == ""
