from resumesort.report_pdf import build_candidate_pdf
from resumesort.schemas import (
    CandidateProfile,
    CandidateReport,
    CandidateScores,
    ClaimVerdict,
    FraudSignal,
)


def _report(name: str) -> CandidateReport:
    return CandidateReport(
        profile=CandidateProfile(name=name, source_name="resume.pdf"),
        scores=CandidateScores(70.0, 55.5, 80.0, 68.2),
        verdicts=[
            ClaimVerdict(
                claim="Built a FastAPI service",
                verdict="SUPPORTED",
                confidence=0.9,
                evidence="FastAPI REST service",
                evidence_source="https://github.com/x/api",
                explanation="README documents the service.",
            )
        ],
        flags=["1 repository/repositories are forks"],
        summary="Solid backend candidate with verified API work.",
        fraud_signals=[
            FraudSignal("mostly_forks", "warn", "3 of 4 repos are forks", "Focus on non-fork repos.")
        ],
        interview_questions=[{"question": "Walk me through the API project?", "targets": "FastAPI claim"}],
    )


def test_pdf_magic_bytes():
    data = build_candidate_pdf(_report("Jane Doe"))
    assert data[:5] == b"%PDF-"
    assert len(data) > 1000


def test_pdf_handles_non_latin_name():
    data = build_candidate_pdf(_report("José Müller 张伟"))
    assert data[:5] == b"%PDF-"
