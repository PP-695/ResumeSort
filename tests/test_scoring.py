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


def test_nei_confidence_does_not_raise_score():
    """Regression A3: confidence in 'we can't verify' must not boost verification."""
    confident_nei = score_verification([ClaimVerdict("c", "NOT_ENOUGH_INFO", 0.95)])
    hesitant_nei = score_verification([ClaimVerdict("c", "NOT_ENOUGH_INFO", 0.30)])
    assert confident_nei == hesitant_nei

    supported = score_verification([ClaimVerdict("c", "SUPPORTED", 0.95)])
    assert supported > confident_nei


def test_fraud_signal_penalties():
    from resumesort.schemas import FraudSignal

    clean = score_authenticity([], [])
    with_high = score_authenticity([], [], [FraudSignal("x", "high", "t", "d")])
    with_info = score_authenticity([], [], [FraudSignal("x", "info", "t", "d")])
    assert with_high < with_info < clean
