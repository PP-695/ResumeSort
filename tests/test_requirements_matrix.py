from resumesort.llm import TinkerLLM
from resumesort.requirements_matrix import (
    Requirement,
    build_matrix,
    coverage_for_report,
    matrix_markdown,
    parse_jd_requirements,
)
from resumesort.sample_data import load_sample_reports


JD = """We are hiring backend engineers.

The ideal candidate will have:
- Hands-on experience with FastAPI or Flask (must)
- Worked with NoSQL databases like MongoDB
- Basic ML pipeline knowledge
"""


def test_fallback_parses_bullet_requirements(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    llm = TinkerLLM(enabled=True)
    requirements = parse_jd_requirements(JD, llm)
    assert len(requirements) == 3
    assert any("FastAPI" in r.text for r in requirements)
    assert any(r.must_have for r in requirements)


def test_llm_parse_path(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    llm = TinkerLLM(enabled=True)
    canned = '{"requirements": [{"requirement": "FastAPI experience", "must_have": true}]}'
    monkeypatch.setattr(TinkerLLM, "complete", lambda self, *a, **k: canned)
    requirements = parse_jd_requirements(JD, llm)
    assert requirements == [Requirement(text="FastAPI experience", must_have=True)]


def test_coverage_levels():
    strong, inflated, no_github = load_sample_reports()

    fastapi_req = Requirement("Hands-on experience with FastAPI", must_have=True)
    assert coverage_for_report(fastapi_req, strong).level == "met"
    assert coverage_for_report(fastapi_req, no_github).level == "none"

    rust_req = Requirement("Production Rust experience", must_have=True)
    coverage = coverage_for_report(rust_req, inflated)
    # Rust claim was REFUTED -> at best partial, never met.
    assert coverage.level in {"partial", "none"}


def test_matrix_markdown_renders_badges():
    reports = load_sample_reports()[:2]
    requirements = [
        Requirement("FastAPI experience", must_have=True),
        Requirement("Quantum computing research", must_have=False),
    ]
    matrix = build_matrix(requirements, reports)
    markdown = matrix_markdown(matrix, ["A", "B"])
    assert ":red-badge[must]" in markdown
    assert markdown.count("|") >= 12
    assert ":green-badge" in markdown or ":yellow-badge" in markdown
    # must-haves sort first
    assert markdown.index("FastAPI") < markdown.index("Quantum")
