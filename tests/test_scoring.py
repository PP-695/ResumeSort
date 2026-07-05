from resumesort.schemas import CandidateProfile, ClaimVerdict, EvidenceItem
from resumesort.scoring import combine_scores, score_authenticity, score_jd_fit, score_verification


def test_scores_are_normalized():
    profile = CandidateProfile(
        skills=["Python", "FastAPI", "MongoDB"],
        projects=["Built FastAPI backend with MongoDB"],
        experience=["Backend developer intern"],
        cgpa=8.5,
        leadership=True,
    )
    jd_score = score_jd_fit(profile, "Python FastAPI MongoDB backend leadership")
    verification = score_verification(
        [
            ClaimVerdict("Built API", "SUPPORTED", 0.8),
            ClaimVerdict("Served 1M users", "NOT_ENOUGH_INFO", 0.5),
        ]
    )
    authenticity = score_authenticity(
        [EvidenceItem("readme", "repo", "url", "FastAPI MongoDB backend")],
        ["README-focused evidence; code-level proof may be limited"],
    )
    combined = combine_scores(jd_score, verification, authenticity, (0.45, 0.35, 0.20))

    assert 0 <= combined.jd_fit_score <= 100
    assert 0 <= combined.verification_score <= 100
    assert 0 <= combined.authenticity_score <= 100
    assert 0 <= combined.final_score <= 100
