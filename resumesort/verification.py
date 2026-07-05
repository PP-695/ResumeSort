from __future__ import annotations

from .llm import TinkerLLM
from .schemas import ClaimVerdict, EvidenceItem
from .scoring import cosine_text_similarity


METRIC_NEI_EXPLANATION = (
    "Quantitative outcome claims (latency %, user counts, revenue) are not verifiable "
    "from public repositories - ask how the number was measured in the interview."
)


def verify_claims(
    claims: list[str] | list[dict],
    evidence: list[EvidenceItem],
    llm: TinkerLLM,
    max_llm_judgments: int = 6,
) -> list[ClaimVerdict]:
    verdicts: list[ClaimVerdict] = []
    llm_judgments = 0

    def judge(claim: str, fallback: ClaimVerdict) -> ClaimVerdict:
        nonlocal llm_judgments
        llm_judgments += 1
        judged = llm.judge_claim(claim, top_evidence_matches(claim, evidence, k=4))
        return judged or fallback

    # First pass: NEI claims get LLM priority — they're the ones a heuristic
    # cannot resolve.
    pending_supported: list[int] = []
    for entry in claims:
        if isinstance(entry, dict):
            claim, kind = entry.get("claim", ""), entry.get("kind", "")
        else:
            claim, kind = entry, ""
        if not claim:
            continue

        # Metric claims can never be proven from public code - honest NEI without
        # burning judge budget on them.
        if kind == "metric":
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.4,
                    explanation=METRIC_NEI_EXPLANATION,
                )
            )
            continue

        best = best_evidence_match(claim, evidence)
        heuristic = heuristic_verdict(claim, best)
        if (
            heuristic.verdict == "NOT_ENOUGH_INFO"
            and llm.status.enabled
            and evidence
            and llm_judgments < max_llm_judgments
        ):
            verdicts.append(judge(claim, heuristic))
        else:
            if heuristic.verdict == "SUPPORTED":
                pending_supported.append(len(verdicts))
            verdicts.append(heuristic)

    # Second pass: spend leftover budget double-checking keyword-SUPPORTED
    # verdicts — a README that parrots the claim's words should not buy an
    # unreviewed SUPPORTED. The LLM verdict wins.
    if llm.status.enabled and evidence:
        for index in pending_supported:
            if llm_judgments >= max_llm_judgments:
                break
            verdicts[index] = judge(verdicts[index].claim, verdicts[index])
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
        # Keyword-level match only — cap well below what an LLM judgment can earn.
        confidence = min(0.65, 0.40 + similarity)
        explanation = "Keyword-level match between the claim and retrieved GitHub evidence (not LLM-judged)."
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
