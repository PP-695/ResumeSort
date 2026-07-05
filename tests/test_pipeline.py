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
                watchers=1,
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

    def fake_parse(_file_obj, source_name=""):
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

    monkeypatch.setattr(pipeline, "parse_resume_pdf", lambda f, source_name="": __import__(
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
