from resumesort.llm import TinkerLLM, fallback_interview_questions, parse_json_object, strip_reasoning
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
