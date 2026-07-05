from __future__ import annotations

import re
from collections import Counter

from .schemas import CandidateProfile, CandidateScores, ClaimVerdict, EvidenceItem, FraudSignal


def score_jd_fit(profile: CandidateProfile, job_description: str) -> float:
    jd_lower = job_description.lower()
    skill_score = 0.0
    if profile.skills:
        skill_score = sum(1 for skill in profile.skills if skill.lower() in jd_lower) / len(profile.skills)

    project_text = " ".join(profile.projects + profile.experience)
    semantic_score = cosine_text_similarity(project_text, job_description)
    cgpa_score = 0.0 if profile.cgpa is None else min(profile.cgpa / 10.0, 1.0)
    leadership_score = 1.0 if profile.leadership and any(word in jd_lower for word in ["lead", "leader", "leadership", "mentor"]) else 0.0
    return _pct(0.35 * skill_score + 0.35 * semantic_score + 0.20 * cgpa_score + 0.10 * leadership_score)


def score_verification(verdicts: list[ClaimVerdict]) -> float:
    if not verdicts:
        return 0.0
    weights = {"SUPPORTED": 1.0, "NOT_ENOUGH_INFO": 0.35, "REFUTED": 0.0}
    score = sum(weights[v.verdict] * v.confidence for v in verdicts) / len(verdicts)
    return _pct(score)


def score_authenticity(
    evidence: list[EvidenceItem],
    flags: list[str],
    fraud_signals: list[FraudSignal] | None = None,
) -> float:
    score = 1.0
    flag_text = " ".join(flags).lower()
    if "missing github" in flag_text:
        score -= 0.35
    if "no matching repo evidence" in flag_text:
        score -= 0.30
    if "fork" in flag_text:
        score -= 0.15
    if "low recent commit" in flag_text:
        score -= 0.10
    if "readme-focused" in flag_text:
        score -= 0.05

    if fraud_signals:
        high_count = sum(1 for signal in fraud_signals if signal.severity == "high")
        warn_count = sum(1 for signal in fraud_signals if signal.severity == "warn")
        info_count = sum(1 for signal in fraud_signals if signal.severity == "info")
        score -= min(0.45, high_count * 0.15)
        score -= min(0.21, warn_count * 0.07)
        score -= min(0.06, info_count * 0.02)

    if evidence:
        source_types = {item.source_type for item in evidence}
        if "languages" in source_types:
            score += 0.05
        if "commits" in source_types:
            score += 0.05
    return _pct(max(0.0, min(1.0, score)))


def combine_scores(jd_fit: float, verification: float, authenticity: float, weights: tuple[float, float, float]) -> CandidateScores:
    total_weight = sum(weights) or 1.0
    normalized = tuple(weight / total_weight for weight in weights)
    final = jd_fit * normalized[0] + verification * normalized[1] + authenticity * normalized[2]
    return CandidateScores(
        jd_fit_score=round(jd_fit, 2),
        verification_score=round(verification, 2),
        authenticity_score=round(authenticity, 2),
        final_score=round(final, 2),
    )


def cosine_text_similarity(left: str, right: str) -> float:
    left_counts = _token_counts(left)
    right_counts = _token_counts(right)
    if not left_counts or not right_counts:
        return 0.0
    shared = set(left_counts) & set(right_counts)
    numerator = sum(left_counts[token] * right_counts[token] for token in shared)
    left_norm = sum(value * value for value in left_counts.values()) ** 0.5
    right_norm = sum(value * value for value in right_counts.values()) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _token_counts(text: str) -> Counter[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}", text.lower())
    stop = {"and", "the", "with", "for", "are", "this", "that", "from", "have", "will", "your", "our"}
    return Counter(token for token in tokens if token not in stop)


def _pct(value: float) -> float:
    return round(max(0.0, min(1.0, value)) * 100, 2)
