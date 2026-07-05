from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .schemas import ClaimVerdict, EvidenceItem, FraudSignal, Verdict


# Documented Tinker OpenAI-compatible endpoint (beta). Auth = Tinker API key.
# https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/
TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

DEFAULT_MODEL = "openai/gpt-oss-20b"

# gpt-oss models read the reasoning level from the system prompt (harmony format);
# "low" keeps latency and token spend down for structured-output tasks.
SYSTEM_PREAMBLE = "Reasoning: low\nYou are a precise assistant. Follow the output format exactly."


@dataclass
class LLMStatus:
    enabled: bool
    provider: str
    model: str
    reason: str = ""


class TinkerLLM:
    """Tinker inference via the documented OpenAI-compatible endpoint.

    The endpoint renders the model's chat template server-side and (for reasoning
    models) separates chain-of-thought into ``reasoning_content``, so ``content``
    arrives clean. Falls back to deterministic heuristics on any failure.
    """

    def __init__(self, base_model: str = DEFAULT_MODEL, enabled: bool = True):
        self.base_model = base_model
        self._client: Any | None = None
        self._error = ""
        self.api_calls = 0
        self.api_successes = 0
        self.enabled = enabled and bool(os.getenv("TINKER_API_KEY"))

    @property
    def status(self) -> LLMStatus:
        if not self.enabled:
            reason = "TINKER_API_KEY not set" if not os.getenv("TINKER_API_KEY") else self._error
            return LLMStatus(False, "heuristic fallback", self.base_model, reason)
        if self._error:
            return LLMStatus(False, "heuristic fallback", self.base_model, self._error)
        return LLMStatus(True, "Tinker", self.base_model)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
        system: str = SYSTEM_PREAMBLE,
    ) -> str:
        if not self.enabled:
            return ""
        try:
            self.api_calls += 1
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.base_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            self.api_successes += 1
            return strip_reasoning(text).strip()
        except Exception as exc:
            self._error = f"Tinker unavailable: {exc}"
            self.enabled = False
            return ""

    def extract_claims(self, resume_text: str, fallback_claims: list[str], max_claims: int = 8) -> list[str]:
        prompt = f"""Extract atomic, checkable technical claims from this resume.
Return only JSON with this shape: {{"claims": ["claim one", "claim two"]}}.
Prefer project, experience, stack, deployment, and metric claims. Max claims: {max_claims}.

Resume:
{resume_text[:6000]}
"""
        data = parse_json_object(self.complete(prompt, max_tokens=700))
        claims = [str(item).strip() for item in data.get("claims", []) if str(item).strip()]
        if claims:
            return _dedupe(claims)[:max_claims]
        return _dedupe(fallback_claims)[:max_claims]

    def judge_claim(self, claim: str, evidence: list[EvidenceItem]) -> ClaimVerdict | None:
        evidence = evidence[:8]
        evidence_block = "\n\n".join(
            f"[{idx}] {item.repo_name} ({item.source_type})\n{item.text[:1200]}"
            for idx, item in enumerate(evidence, start=1)
        )
        if not evidence_block:
            return None
        prompt = f"""You are a strict resume claim verifier.
Judge the claim only against the provided GitHub evidence. Do not use the resume claim itself as proof.
Rules:
- SUPPORTED requires evidence that affirmatively backs the claim.
- REFUTED requires evidence that CONTRADICTS the claim. Mere absence of evidence is NOT refutation.
- If the evidence neither proves nor contradicts the claim, return NOT_ENOUGH_INFO.
Return only JSON:
{{
  "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO",
  "confidence": 0.0,
  "evidence_index": 1,
  "evidence_quote": "short quote from the numbered evidence item that backs your verdict",
  "explanation": "one sentence"
}}
"evidence_index" must be the number of the evidence item you relied on, or 0 if none.

Claim: {claim}

Evidence:
{evidence_block}
"""
        data = parse_json_object(self.complete(prompt, max_tokens=500))
        verdict = normalize_verdict(data.get("verdict"))
        if not verdict:
            return None
        evidence_source = ""
        try:
            index = int(data.get("evidence_index", 0))
        except (TypeError, ValueError):
            index = 0
        if 1 <= index <= len(evidence):
            evidence_source = evidence[index - 1].path_or_url
        return ClaimVerdict(
            claim=claim,
            verdict=verdict,
            confidence=_clamp_float(data.get("confidence"), default=0.5),
            evidence=str(data.get("evidence_quote", ""))[:600],
            evidence_source=evidence_source,
            explanation=str(data.get("explanation", ""))[:600],
        )

    def summarize(
        self,
        name: str | None,
        resume_text: str,
        job_description: str,
        scores: dict[str, float],
        verdicts: list[ClaimVerdict] | None = None,
    ) -> str:
        verdict_block = ""
        if verdicts:
            verdict_block = "Claim verification results:\n" + "\n".join(
                f"- [{v.verdict}] {v.claim[:150]}" for v in verdicts
            )
        prompt = f"""Write a concise recruiter summary for this candidate.
Use 2 short paragraphs. Evidence discipline is mandatory:
- Only state as fact what is marked SUPPORTED below.
- Describe REFUTED or NOT_ENOUGH_INFO items as "claims" that could not be verified.
- Mention evidence limits if verification is weak.
Return plain text only.

Candidate: {name or "Unknown"}
Scores: {json.dumps(scores)}
{verdict_block}

Job Description:
{job_description[:2500]}

Resume:
{resume_text[:4000]}
"""
        text = self.complete(prompt, max_tokens=350, temperature=0.3)
        if text:
            return text
        return fallback_summary(name, scores)

    def generate_interview_questions(
        self,
        name: str | None,
        verdicts: list[ClaimVerdict],
        fraud_signals: list[FraudSignal],
        k: int = 5,
    ) -> list[dict[str, str]]:
        gaps = [v for v in verdicts if v.verdict in {"REFUTED", "NOT_ENOUGH_INFO"}]
        high_signals = [s for s in fraud_signals if s.severity in {"high", "warn"}]
        if not gaps and not high_signals:
            return []

        gap_block = "\n".join(f"- [{v.verdict}] {v.claim}" for v in gaps[:8])
        signal_block = "\n".join(f"- {s.title}" for s in high_signals[:5])
        prompt = f"""Generate {k} targeted interview questions for candidate {name or "Unknown"}.
Each question must probe ONE specific unverified claim or authenticity concern below.
Questions should be respectful and give the candidate a fair chance to explain
(private repos, team projects, and off-GitHub work are all legitimate answers).
Return only JSON: {{"questions": [{{"question": "...", "targets": "the claim or concern it probes"}}]}}

Unverified/refuted claims:
{gap_block or "(none)"}

Authenticity concerns:
{signal_block or "(none)"}
"""
        data = parse_json_object(self.complete(prompt, max_tokens=700, temperature=0.4))
        questions = []
        for item in data.get("questions", []):
            if isinstance(item, dict) and str(item.get("question", "")).strip():
                questions.append(
                    {
                        "question": str(item["question"]).strip()[:400],
                        "targets": str(item.get("targets", "")).strip()[:300],
                    }
                )
        if questions:
            return questions[:k]
        return fallback_interview_questions(gaps, high_signals, k=k)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ["TINKER_API_KEY"],
                base_url=os.getenv("TINKER_BASE_URL", TINKER_OAI_BASE_URL),
                timeout=60,
                max_retries=1,
            )
        return self._client


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks (and an unclosed leading one) from output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = strip_reasoning(text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def normalize_verdict(value: Any) -> Verdict | None:
    if value is None:
        return None
    normalized = str(value).strip().upper().replace(" ", "_")
    if normalized in {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}:
        return normalized  # type: ignore[return-value]
    if normalized in {"NEI", "UNKNOWN", "UNVERIFIABLE"}:
        return "NOT_ENOUGH_INFO"
    return None


def fallback_interview_questions(
    gaps: list[ClaimVerdict],
    signals: list[FraudSignal],
    k: int = 5,
) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    for verdict in gaps:
        questions.append(
            {
                "question": (
                    f'Your resume mentions "{verdict.claim[:120]}", but we could not confirm it from your '
                    "public GitHub. Can you walk me through that work - where it lives and what you built?"
                ),
                "targets": verdict.claim[:300],
            }
        )
    for signal in signals:
        questions.append(
            {
                "question": f"We noticed: {signal.title}. Could you give us some context on that?",
                "targets": signal.title[:300],
            }
        )
    return questions[:k]


def fallback_summary(name: str | None, scores: dict[str, float]) -> str:
    display_name = name or "This candidate"
    return (
        f"{display_name} was scored against the JD using parsed resume signals and GitHub evidence. "
        f"Final score: {scores.get('final_score', 0):.1f}/100.\n\n"
        "Tinker summary generation was unavailable, so this summary is deterministic and evidence-first."
    )


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = re.sub(r"\s+", " ", value).strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result
