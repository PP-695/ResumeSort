import json
from io import BytesIO

from resumesort.github_evidence import GitHubSnapshot, RepoSnapshot
from resumesort.pipeline import analyze_resumes, build_audit_log
from resumesort.schemas import AnalysisSettings


def _fake_snapshot() -> GitHubSnapshot:
    return GitHubSnapshot(
        username="example",
        account_created_at="2020-01-01T00:00:00+00:00",
        repos=[
            RepoSnapshot(
                name="api",
                html_url="https://github.com/example/api",
                fork=False,
                stars=3,
                subscribers=1,
                forks_count=0,
                created_at="2023-01-01T00:00:00+00:00",
                updated_at="2024-01-01T00:00:00+00:00",
                pushed_at="2024-01-01T00:00:00+00:00",
                languages={"Python": 1000},
                readme_text="FastAPI service",
                readme_url="https://github.com/example/api#readme",
            )
        ],
    )


def test_pipeline_with_mocks(monkeypatch):
    from resumesort import pipeline

    def fake_parse(_file_obj, source_name="", jd_text=""):
        from resumesort.schemas import CandidateProfile

        return CandidateProfile(
            name="Candidate One",
            email="one@example.com",
            github_url="https://github.com/example",
            cgpa=8.2,
            skills=["Python", "FastAPI"],
            projects=["Built FastAPI service"],
            experience=["Backend intern"],
            leadership=False,
            raw_text="Built FastAPI service",
            source_name=source_name,
        )

    def fake_snapshot(_url, token=None, max_repos=10, deep=True):
        return _fake_snapshot(), []

    monkeypatch.setattr(pipeline, "parse_resume_pdf", fake_parse)
    monkeypatch.setattr(pipeline, "fetch_github_snapshot", fake_snapshot)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    fake_file = BytesIO(b"fake pdf")
    fake_file.name = "candidate.pdf"
    progress_messages: list[str] = []
    reports = analyze_resumes(
        [fake_file],
        "Need Python FastAPI backend",
        AnalysisSettings(use_tinker=False),
        progress_callback=progress_messages.append,
    )

    assert len(reports) == 1
    assert reports[0].profile.name == "Candidate One"
    assert reports[0].scores.final_score > 0
    assert reports[0].evidence, "evidence should be attached to the report"
    assert progress_messages, "progress callback should be invoked"


def test_audit_log_schema(monkeypatch):
    from resumesort import pipeline

    monkeypatch.setattr(pipeline, "parse_resume_pdf", lambda f, source_name="", jd_text="": __import__(
        "resumesort.schemas", fromlist=["CandidateProfile"]
    ).CandidateProfile(name="X", raw_text="x", source_name=source_name))
    monkeypatch.setattr(pipeline, "fetch_github_snapshot", lambda *a, **k: (None, ["Missing GitHub link"]))
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    fake_file = BytesIO(b"fake pdf")
    fake_file.name = "x.pdf"
    settings = AnalysisSettings(use_tinker=False, blind_mode=True)
    reports = analyze_resumes([fake_file], "JD", settings)

    audit = json.loads(build_audit_log(reports, settings, "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"))
    assert audit["app"] == "grifter-filter"
    assert audit["app_version"]
    assert audit["blind_mode"] is True
    assert audit["weights"]["jd_fit"] == settings.jd_fit_weight
    assert len(audit["candidates"]) == 1
    assert "disclaimer" in audit
    # Blind mode: audit log must not contain the candidate's name (regression A2).
    assert audit["candidates"][0]["name"] == "Candidate 1"
    assert audit["candidates"][0]["source"] == "candidate_1.pdf"


def test_llm_counters_are_per_candidate(monkeypatch):
    """Regression A1: each report shows its own call count, not the cumulative total."""
    from resumesort import pipeline
    from resumesort.llm import TinkerLLM
    from resumesort.schemas import CandidateProfile

    def fake_parse(_file_obj, source_name="", jd_text=""):
        return CandidateProfile(name=source_name, raw_text="x", source_name=source_name)

    monkeypatch.setattr(pipeline, "parse_resume_pdf", fake_parse)
    monkeypatch.setattr(pipeline, "fetch_github_snapshot", lambda *a, **k: (None, []))
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    calls = {"n": 0}

    def fake_complete(self, *args, **kwargs):
        calls["n"] += 1
        self.api_calls += 1
        self.api_successes += 1
        return ""

    monkeypatch.setattr(TinkerLLM, "complete", fake_complete)

    files = []
    for name in ("a.pdf", "b.pdf"):
        f = BytesIO(b"pdf")
        f.name = name
        files.append(f)
    reports = analyze_resumes(files, "JD", AnalysisSettings(use_tinker=True))

    total = sum(r.llm_api_calls for r in reports)
    assert total == calls["n"], "per-report deltas must sum to the true total"
    assert all(r.llm_api_calls < calls["n"] for r in reports), "no report may claim the cumulative count"


def test_blind_exports_contain_no_pii(monkeypatch):
    """Regression A2: blind JSON/CSV exports must not leak name/email/github."""
    from resumesort import pipeline
    from resumesort.pipeline import reports_to_dataframe, reports_to_json
    from resumesort.schemas import CandidateProfile

    def fake_parse(_file_obj, source_name="", jd_text=""):
        return CandidateProfile(
            name="Jane Secret",
            email="jane.secret@example.com",
            github_url="https://github.com/janesecret",
            raw_text="Jane Secret resume",
            source_name=source_name,
        )

    monkeypatch.setattr(pipeline, "parse_resume_pdf", fake_parse)
    monkeypatch.setattr(pipeline, "fetch_github_snapshot", lambda *a, **k: (None, []))
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    f = BytesIO(b"pdf")
    f.name = "jane.pdf"
    reports = analyze_resumes([f], "JD", AnalysisSettings(use_tinker=False, blind_mode=True))

    json_out = reports_to_json(reports, blind=True)
    csv_out = reports_to_dataframe(reports, blind=True).to_csv(index=False)
    for output in (json_out, csv_out):
        assert "Jane Secret" not in output
        assert "jane.secret@example.com" not in output
        assert "janesecret" not in output
