from __future__ import annotations

from .llm import TinkerLLM
from .schemas import ClaimVerdict, EvidenceItem
from .scoring import cosine_text_similarity


def verify_claims(
    claims: list[str],
    evidence: list[EvidenceItem],
    llm: TinkerLLM,
    max_llm_judgments: int = 6,
) -> list[ClaimVerdict]:
    verdicts: list[ClaimVerdict] = []
    llm_judgments = 0
    for claim in claims:
        best = best_evidence_match(claim, evidence)
        heuristic = heuristic_verdict(claim, best)
        if (
            heuristic.verdict == "NOT_ENOUGH_INFO"
            and llm.status.enabled
            and evidence
            and llm_judgments < max_llm_judgments
        ):
            llm_judgments += 1
            judged = llm.judge_claim(claim, top_evidence_matches(claim, evidence, k=4))
            verdicts.append(judged or heuristic)
        else:
            verdicts.append(heuristic)
    return verdicts


def best_evidence_match(claim: str, evidence: list[EvidenceItem]) -> EvidenceItem | None:
    if not evidence:
        return None
    return max(evidence, key=lambda item: cosine_text_similarity(claim, item.text[:3000]))


def top_evidence_matches(claim: str, evidence: list[EvidenceItem], k: int = 4) -> list[EvidenceItem]:
    """The k most claim-relevant evidence items, most similar first."""
    ranked = sorted(
        evidence,
        key=lambda item: cosine_text_similarity(claim, item.text[:3000]),
        reverse=True,
    )
    return ranked[:k]


def heuristic_verdict(claim: str, evidence: EvidenceItem | None) -> ClaimVerdict:
    if evidence is None:
        return ClaimVerdict(
            claim=claim,
            verdict="NOT_ENOUGH_INFO",
            confidence=0.25,
            explanation="No GitHub evidence was available for this claim.",
        )

    similarity = cosine_text_similarity(claim, evidence.text[:3000])
    if similarity >= 0.22:
        verdict = "SUPPORTED"
        confidence = min(0.9, 0.55 + similarity)
        explanation = "Local similarity matched the claim to retrieved GitHub evidence."
    elif similarity >= 0.10:
        verdict = "NOT_ENOUGH_INFO"
        confidence = 0.45
        explanation = "Some related evidence was found, but it is not strong enough to prove the claim."
    else:
        verdict = "NOT_ENOUGH_INFO"
        confidence = 0.30
        explanation = "Retrieved evidence does not clearly support or refute the claim."

    return ClaimVerdict(
        claim=claim,
        verdict=verdict,
        confidence=round(confidence, 2),
        evidence=evidence.text[:500],
        evidence_source=evidence.path_or_url,
        explanation=explanation,
    )
