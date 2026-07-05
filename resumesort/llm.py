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

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, base_model: str = DEFAULT_MODEL, enabled: bool = True):
        self.base_model = base_model
        self._client: Any | None = None
        self._error = ""
        self._consecutive_failures = 0
        self.api_calls = 0
        self.api_successes = 0
        self.enabled = enabled and bool(os.getenv("TINKER_API_KEY"))

    @property
    def status(self) -> LLMStatus:
        if not self.enabled:
            reason = "TINKER_API_KEY not set" if not os.getenv("TINKER_API_KEY") else self._error
            return LLMStatus(False, "heuristic fallback", self.base_model, reason)
        # Still enabled; surface any transient error as informational only.
        return LLMStatus(True, "Tinker", self.base_model, self._error)

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
            self._consecutive_failures = 0
            self._error = ""
            return strip_reasoning(text).strip()
        except Exception as exc:
            self._error = f"Tinker error: {exc}"
            if _is_auth_error(exc):
                # Bad/expired key: no retry will help — disable immediately.
                self.enabled = False
                return ""
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                self._error = f"Tinker disabled after {self._consecutive_failures} consecutive errors: {exc}"
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
        prompt = f"""Generate {k} sharp technical interview questions for candidate {name or "Unknown"}.
Rules for each question:
- Target exactly ONE unverified claim or authenticity concern from the lists below.
- Probe implementation depth: a specific design decision, tradeoff, failure mode, scaling
  limit, or how a stated metric was measured. Someone who actually did the work should
  answer easily; someone who copied or inflated it should struggle.
- Do NOT ask generic questions like "walk me through it" or "tell me about this project".
- Stay respectful: private repos, team projects, and off-GitHub work are legitimate answers.
Return only JSON:
{{"questions": [{{"question": "...", "targets": "the claim/concern probed", "listen_for": "1 sentence: what a credible answer includes"}}]}}

Unverified/refuted claims:
{gap_block or "(none)"}

Authenticity concerns:
{signal_block or "(none)"}
"""
        # Reasoning models spend output budget thinking before the JSON arrives;
        # a tight cap truncates content to empty and silently forces the fallback.
        data = parse_json_object(self.complete(prompt, max_tokens=1600, temperature=0.4))
        questions = []
        for item in data.get("questions", []):
            if isinstance(item, dict) and str(item.get("question", "")).strip():
                questions.append(
                    {
                        "question": str(item["question"]).strip()[:400],
                        "targets": str(item.get("targets", "")).strip()[:300],
                        "listen_for": str(item.get("listen_for", "")).strip()[:300],
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


FALLBACK_QUESTION_TEMPLATES = (
    'On "{claim}" - what was the hardest technical decision you made there, and what alternative did you reject? Where does the code live today?',
    'For "{claim}" - what broke first when you tested it under load or with messy real data, and how did you fix it?',
    'Regarding "{claim}" - if you had to rebuild it from scratch tomorrow, what would you change about the design, and why?',
    'On "{claim}" - how did you measure the result you describe, and what would make that number go down?',
    'For "{claim}" - which part did you personally write versus adapt from libraries, teammates, or tutorials?',
)


def fallback_interview_questions(
    gaps: list[ClaimVerdict],
    signals: list[FraudSignal],
    k: int = 5,
) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    for idx, verdict in enumerate(gaps):
        template = FALLBACK_QUESTION_TEMPLATES[idx % len(FALLBACK_QUESTION_TEMPLATES)]
        questions.append(
            {
                "question": template.format(claim=_shorten(verdict.claim, 110)),
                "targets": verdict.claim[:300],
                "listen_for": "Specific names, tradeoffs, and failure details - not a restated resume bullet.",
            }
        )
    for signal in signals:
        questions.append(
            {
                "question": (
                    f"We noticed: {signal.title}. Could you give us some context? "
                    "(Private repos, team projects, or a recreated account are all fine answers.)"
                ),
                "targets": signal.title[:300],
                "listen_for": "A concrete, checkable explanation for the pattern.",
            }
        )
    return questions[:k]


def shorten(text: str, limit: int) -> str:
    """Truncate at a word boundary so labels never end mid-word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."


# Backward-compat alias (internal callers/tests may use the old name).
_shorten = shorten


def _is_auth_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403):
        return True
    message = str(exc).lower()
    return "401" in message or "unauthorized" in message or "invalid api key" in message


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
