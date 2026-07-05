from resumesort.privacy import blind_display_name, redact_text_for_llm
from resumesort.schemas import CandidateProfile


def test_redaction_removes_identity_keeps_skills():
    profile = CandidateProfile(
        name="Jane Doe",
        email="jane@example.com",
        github_url="https://github.com/janedoe",
        cgpa=8.7,
    )
    text = (
        "Jane Doe\n"
        "jane@example.com | https://github.com/janedoe\n"
        "CGPA: 8.7/10 at Example University\n"
        "Skills: Python, FastAPI, MongoDB\n"
        "Built a real-time analytics API"
    )
    redacted = redact_text_for_llm(text, profile)

    assert "Jane Doe" not in redacted
    assert "jane@example.com" not in redacted
    assert "janedoe" not in redacted
    assert "8.7" not in redacted
    assert "University" not in redacted
    assert "Python" in redacted
    assert "real-time analytics API" in redacted


def test_blind_display_name():
    assert blind_display_name(0) == "Candidate 1"
    assert blind_display_name(4) == "Candidate 5"
