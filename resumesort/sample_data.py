"""Canned demo reports for the landing-page "Try sample data" CTA.

All three candidates are fictional; the data shape is exactly what the real
pipeline produces, so every UI surface (ranking, detail tabs, compare,
exports) works instantly with zero network or LLM calls.
"""

from __future__ import annotations

from .schemas import (
    CandidateProfile,
    CandidateReport,
    CandidateScores,
    ClaimVerdict,
    FraudSignal,
)


def load_sample_reports() -> list[CandidateReport]:
    strong = CandidateReport(
        profile=CandidateProfile(
            name="Asha Iyer",
            email="asha.iyer@example.com",
            github_url="https://github.com/asha-iyer-demo",
            cgpa=8.9,
            claimed_years_experience=2.0,
            skills=["Python", "FastAPI", "MongoDB", "Docker", "PyTorch"],
            projects=["Built a real-time inventory API with FastAPI and MongoDB"],
            experience=["Backend intern at a logistics startup"],
            leadership=True,
            raw_text="(sample)",
            source_name="asha_iyer_sample.pdf",
        ),
        scores=CandidateScores(jd_fit_score=78.4, verification_score=71.2, authenticity_score=93.0, final_score=78.8),
        verdicts=[
            ClaimVerdict(
                claim="Built a real-time inventory API with FastAPI and MongoDB",
                verdict="SUPPORTED",
                confidence=0.88,
                evidence="FastAPI service with Motor (async MongoDB) and WebSocket inventory pushes",
                evidence_source="https://github.com/asha-iyer-demo/inventory-api",
                explanation="README and dependency manifest document the claimed stack; commit history spans 7 months.",
            ),
            ClaimVerdict(
                claim="Reduced API p95 latency by 40%",
                verdict="NOT_ENOUGH_INFO",
                confidence=0.55,
                explanation="No benchmark artifacts in public repos; metric is plausible but unverifiable.",
            ),
            ClaimVerdict(
                claim="Trained a PyTorch demand-forecasting model",
                verdict="SUPPORTED",
                confidence=0.74,
                evidence="notebooks importing torch, training loop with saved checkpoints",
                evidence_source="https://github.com/asha-iyer-demo/demand-forecast",
                explanation="Real model code present, not just the word PyTorch in a README.",
            ),
        ],
        flags=[],
        summary=(
            "Asha's core backend claims are verified against real repository evidence: the FastAPI/MongoDB "
            "inventory service exists with a months-long commit history, and the PyTorch project contains "
            "genuine training code.\n\nHer latency-improvement metric could not be verified from public "
            "sources - worth one interview question, not a concern."
        ),
        fraud_signals=[],
        interview_questions=[
            {
                "question": "Your p95 latency claim - how did you measure it, and what would make that number regress?",
                "targets": "Reduced API p95 latency by 40%",
                "listen_for": "A specific load-testing setup and a bottleneck she can name.",
            }
        ],
        llm_provider="Tinker",
        llm_api_calls=6,
        llm_api_successes=6,
    )

    inflated = CandidateReport(
        profile=CandidateProfile(
            name="Rohan Mehta",
            email="rohan.mehta@example.com",
            github_url="https://github.com/rohan-mehta-demo",
            cgpa=9.1,
            claimed_years_experience=6.0,
            skills=["Python", "Rust", "Kubernetes", "Kafka", "AWS"],
            projects=["Architected a Rust trading engine handling 1M orders/sec"],
            experience=["Self-employed systems consultant"],
            leadership=False,
            raw_text="(sample)",
            source_name="rohan_mehta_sample.pdf",
        ),
        scores=CandidateScores(jd_fit_score=69.5, verification_score=22.4, authenticity_score=41.0, final_score=47.3),
        verdicts=[
            ClaimVerdict(
                claim="Architected a Rust trading engine handling 1M orders/sec",
                verdict="REFUTED",
                confidence=0.90,
                evidence="No Rust code exists in any inspected repository; languages are Python and HTML only",
                evidence_source="https://github.com/rohan-mehta-demo?tab=repositories",
                explanation="The claimed flagship project contradicts the account's actual language footprint.",
            ),
            ClaimVerdict(
                claim="Deployed Kubernetes clusters serving 10M daily users",
                verdict="NOT_ENOUGH_INFO",
                confidence=0.60,
                explanation="Infrastructure work can be private, but nothing public corroborates it.",
            ),
            ClaimVerdict(
                claim="Proficient in Python",
                verdict="SUPPORTED",
                confidence=0.62,
                evidence="Several Python repositories with real commits",
                evidence_source="https://github.com/rohan-mehta-demo/scraper-toolkit",
                explanation="Keyword-level match plus consistent language footprint.",
            ),
        ],
        flags=["2 repository/repositories are forks"],
        summary=(
            "Rohan's Python competence is supported, but his flagship claim - a Rust trading engine at "
            "1M orders/sec - is contradicted by his public footprint, which contains zero Rust.\n\nCombined "
            "with a bulk-pushed commit history and an account far younger than his claimed experience, this "
            "profile needs direct verification before proceeding."
        ),
        fraud_signals=[
            FraudSignal(
                signal_id="language_not_found",
                severity="warn",
                title="Claimed language 'Rust' has zero bytes across inspected repos",
                detail="The resume lists Rust, but none of the inspected repositories contain any Rust code.",
                evidence_url="https://github.com/rohan-mehta-demo?tab=repositories",
            ),
            FraudSignal(
                signal_id="commit_clustering",
                severity="high",
                title="93% of trading-sim's yearly commits landed in <=2 weeks",
                detail="An 11-month-old repo whose entire history arrived in two bulk pushes - consistent with imported or backdated work.",
                evidence_url="https://github.com/rohan-mehta-demo/trading-sim/graphs/commit-activity",
            ),
            FraudSignal(
                signal_id="account_age_gap",
                severity="warn",
                title="Resume claims 6 years of experience; GitHub account is 1.4 years old",
                detail="Accounts get recreated and work happens off-GitHub, but combined with other signals this warrants a question.",
                evidence_url="https://github.com/rohan-mehta-demo",
            ),
        ],
        interview_questions=[
            {
                "question": "On the Rust trading engine - what was the hardest ownership/borrowing problem you hit, and how did you profile the hot path?",
                "targets": "Architected a Rust trading engine handling 1M orders/sec",
                "listen_for": "Fluent, specific Rust vocabulary; a bluffer will stay abstract.",
            },
            {
                "question": "Your trading-sim repo's history arrived in two bulk pushes - where was the code developed before that?",
                "targets": "Commit clustering signal",
                "listen_for": "A concrete, checkable origin (private repo, employer, migration).",
            },
        ],
        llm_provider="Tinker",
        llm_api_calls=7,
        llm_api_successes=7,
    )

    no_github = CandidateReport(
        profile=CandidateProfile(
            name="Meera Krishnan",
            email="meera.k@example.com",
            github_url=None,
            cgpa=9.5,
            skills=["Java", "Spring Boot", "SQL"],
            projects=["Built a hospital-management system in Java"],
            experience=["Software engineering intern at a healthtech company"],
            leadership=True,
            raw_text="(sample)",
            source_name="meera_krishnan_sample.pdf",
        ),
        scores=CandidateScores(jd_fit_score=48.7, verification_score=35.0, authenticity_score=55.0, final_score=45.2),
        verdicts=[
            ClaimVerdict(
                claim="Built a hospital-management system in Java",
                verdict="NOT_ENOUGH_INFO",
                confidence=0.30,
                explanation="No GitHub link on the resume; nothing public to verify against.",
            ),
        ],
        flags=["Missing GitHub link"],
        summary=(
            "Meera's resume provides no public code footprint, so her claims are neither supported nor "
            "refuted - the honest verdict is NOT ENOUGH INFO across the board.\n\nStrong academics and a "
            "relevant internship suggest a normal next step: ask for a code sample or portfolio link."
        ),
        fraud_signals=[],
        interview_questions=[
            {
                "question": "Could you share any code you're proud of - a repo, gist, or take-home - from the hospital-management project?",
                "targets": "Built a hospital-management system in Java",
                "listen_for": "Willingness to show real artifacts; private work is a fine answer.",
            }
        ],
        llm_provider="Tinker",
        llm_api_calls=3,
        llm_api_successes=3,
    )

    return [strong, inflated, no_github]
