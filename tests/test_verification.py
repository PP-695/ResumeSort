from resumesort.llm import TinkerLLM
from resumesort.schemas import EvidenceItem
from resumesort.verification import top_evidence_matches, verify_claims


EVIDENCE = [
    EvidenceItem("readme", "api-repo", "https://github.com/x/api-repo", "FastAPI REST service with MongoDB"),
    EvidenceItem("readme", "ml-repo", "https://github.com/x/ml-repo", "PyTorch transformer training pipeline"),
    EvidenceItem("languages", "misc", "https://github.com/x/misc", "HTML, CSS"),
]


def test_top_evidence_orders_by_similarity():
    top = top_evidence_matches("Built a FastAPI REST service", EVIDENCE, k=2)
    assert len(top) == 2
    assert top[0].repo_name == "api-repo"


def test_verify_claims_caps_llm_judgments(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    llm = TinkerLLM(enabled=True)
    calls = {"count": 0}

    def fake_judge(self, claim, evidence):
        calls["count"] += 1
        return None

    monkeypatch.setattr(TinkerLLM, "judge_claim", fake_judge)
    claims = [f"Unrelated quantum blockchain claim {i}" for i in range(10)]
    verdicts = verify_claims(claims, EVIDENCE, llm, max_llm_judgments=3)

    assert len(verdicts) == 10
    assert calls["count"] == 3


def test_verify_claims_all_heuristic_without_llm(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    llm = TinkerLLM(enabled=True)
    verdicts = verify_claims(["Built a FastAPI REST service with MongoDB"], EVIDENCE, llm)
    assert verdicts[0].verdict == "SUPPORTED"
    assert verdicts[0].evidence_source == "https://github.com/x/api-repo"
