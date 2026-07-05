from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Verdict = Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]

Severity = Literal["info", "warn", "high"]


@dataclass
class CandidateProfile:
    name: str | None = None
    email: str | None = None
    github_url: str | None = None
    cgpa: float | None = None
    claimed_years_experience: float | None = None
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    leadership: bool = False
    raw_text: str = ""
    source_name: str = ""


@dataclass
class EvidenceItem:
    source_type: str
    repo_name: str
    path_or_url: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FraudSignal:
    signal_id: str
    severity: Severity
    title: str
    detail: str
    evidence_url: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimVerdict:
    claim: str
    verdict: Verdict
    confidence: float
    evidence: str = ""
    evidence_source: str = ""
    explanation: str = ""


@dataclass
class CandidateScores:
    jd_fit_score: float
    verification_score: float
    authenticity_score: float
    final_score: float


@dataclass
class CandidateReport:
    profile: CandidateProfile
    scores: CandidateScores
    verdicts: list[ClaimVerdict]
    flags: list[str]
    summary: str
    fraud_signals: list[FraudSignal] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    interview_questions: list[dict[str, str]] = field(default_factory=list)
    llm_provider: str = "heuristic fallback"
    llm_api_calls: int = 0
    llm_api_successes: int = 0
    llm_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisSettings:
    tinker_base_model: str = "openai/gpt-oss-20b"
    github_token: str | None = None
    use_tinker: bool = True
    max_claims: int = 8
    max_repos: int = 10
    jd_fit_weight: float = 0.45
    verification_weight: float = 0.35
    authenticity_weight: float = 0.20
    deep_fraud_checks: bool = True
    blind_mode: bool = False
