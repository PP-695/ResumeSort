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

# Task-specific system prompts. gpt-oss models read the reasoning level from the
# system prompt (harmony format); judging gets "medium" because verdicts deserve
# thought, everything else stays "low" for latency.
EXTRACTOR_SYSTEM = (
    "Reasoning: low\n"
    "You are a resume-claims auditor for technical hiring. You extract only claims "
    "that could in principle be checked against a candidate's public code. "
    "You respond with valid JSON only."
)

JUDGE_SYSTEM = (
    "Reasoning: medium\n"
    "You are a strict evidence auditor for technical hiring. You judge claims only "
    "against the evidence provided; a claim is never its own proof. You respond "
    "with valid JSON only."
)

WRITER_SYSTEM = (
    "Reasoning: low\n"
    "You write terse, evidence-grounded candidate briefs for recruiters. You never "
    "state an unverified claim as fact. You respond with plain text only."
)

INTERVIEWER_SYSTEM = (
    "Reasoning: low\n"
    "You are a senior technical interviewer who designs questions that separate "
    "people who did the work from people who inflated it. You respond with valid "
    "JSON only."
)

CLAIM_KINDS = {"project", "skill", "metric", "experience"}

# Deterministic net under the LLM exclusion rules: claims about these are parsed
# elsewhere and can never be verified from code, so they must not reach the judge.
_JUNK_CLAIM_RE = re.compile(
    r"\bC?GPA\b"
    r"|github\.com|linkedin|leetcode"
    r"|[\w.+-]+@[\w-]+\.\w"
    r"|\byears?\b.{0,40}\bexperience\b"
    r"|\bmaintains?\s+an?\s+(?:public\s+)?(?:github|profile|portfolio|account)\b"
    r"|\b(?:phone|contact|address)\b",
    re.IGNORECASE,
)

_COMPOUND_LEAD_RE = re.compile(
    r"^(?P<lead>.*?\b(?:proficient in|skilled in|experienced (?:in|with)|knowledge of|expertise in|familiar with)\s+)(?P<items>.+)$",
    re.IGNORECASE,
)


@dataclass
class LLMStatus:
    enabled: bool
    provider: str
    model: str
    reason: str = ""
    parse_failures: int = 0
    truncations: int = 0


class TinkerLLM:
    """Tinker inference via the documented OpenAI-compatible endpoint.

    The endpoint renders the model's chat template server-side and (for reasoning
    models) separates chain-of-thought into ``reasoning_content``, so ``content``
    arrives clean. Falls back to deterministic heuristics on any failure.

    Reliability model: ``api_successes`` counts HTTP-level successes;
    ``parse_failures`` counts calls whose output was unusable (the metric that
    actually predicts fallback quality); truncated responses are retried once
    with a doubled budget before counting as failures.
    """

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, base_model: str = DEFAULT_MODEL, enabled: bool = True):
        self.base_model = base_model
        self.judge_model = os.getenv("TINKER_JUDGE_MODEL") or base_model
        self._client: Any | None = None
        self._error = ""
        self._consecutive_failures = 0
        self._json_mode: bool | None = None  # None = unprobed
        self.api_calls = 0
        self.api_successes = 0
        self.parse_failures = 0
        self.truncations = 0
        self.enabled = enabled and bool(os.getenv("TINKER_API_KEY"))

    @property
    def status(self) -> LLMStatus:
        if not self.enabled:
            reason = "TINKER_API_KEY not set" if not os.getenv("TINKER_API_KEY") else self._error
            return LLMStatus(False, "heuristic fallback", self.base_model, reason,
                             self.parse_failures, self.truncations)
        return LLMStatus(True, "Tinker", self.base_model, self._error,
                         self.parse_failures, self.truncations)

    # ------------------------------------------------------------------ core

    def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        system: str = EXTRACTOR_SYSTEM,
        model: str | None = None,
        json_mode: bool = False,
    ) -> str:
        if not self.enabled:
            return ""
        text, finish_reason = self._request(prompt, max_tokens, temperature, system, model, json_mode)
        if finish_reason == "length" and not text.strip():
            # Reasoning consumed the whole budget before any content arrived.
            self.truncations += 1
            text, _ = self._request(prompt, max_tokens * 2, temperature, system, model, json_mode)
        return text

    def _request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system: str,
        model: str | None,
        json_mode: bool,
    ) -> tuple[str, str]:
        try:
            self.api_calls += 1
            client = self._get_client()
            kwargs: dict[str, Any] = {
                "model": model or self.base_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "extra_body": {"separate_reasoning": True},
            }
            if json_mode and self._json_mode is not False:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = client.chat.completions.create(**kwargs)
                if json_mode and self._json_mode is None:
                    self._json_mode = True
            except Exception as exc:
                # Endpoint may not support response_format: probe once, degrade forever.
                if json_mode and self._json_mode is None and _is_bad_request(exc):
                    self._json_mode = False
                    kwargs.pop("response_format", None)
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise
            choice = response.choices[0]
            text = choice.message.content or ""
            self.api_successes += 1
            self._consecutive_failures = 0
            self._error = ""
            return strip_reasoning(text).strip(), choice.finish_reason or ""
        except Exception as exc:
            self._error = f"Tinker error: {exc}"
            if _is_auth_error(exc):
                self.enabled = False
                return "", "error"
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                self._error = f"Tinker disabled after {self._consecutive_failures} consecutive errors: {exc}"
                self.enabled = False
            return "", "error"

    def _complete_json(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        system: str = EXTRACTOR_SYSTEM,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Single JSON gateway: every structured call routes here so parse
        failures are counted in exactly one place."""
        text = self.complete(prompt, max_tokens=max_tokens, temperature=temperature,
                             system=system, model=model, json_mode=True)
        if not text:
            return {}
        data = parse_json_object(text)
        if not data:
            self.parse_failures += 1
        return data

    # ----------------------------------------------------------- extraction

    def extract_claims_tagged(
        self, resume_text: str, fallback_claims: list[str], max_claims: int = 8
    ) -> list[dict[str, str]]:
        prompt = f"""Extract atomic, code-checkable claims from this resume.

Atomic = exactly ONE skill, technology, or deliverable per claim. Split compound
sentences: "Built X using A and deployed on B" becomes two claims.

Example:
  Bullet: "Proficient in Python and Rust; built a REST API with FastAPI"
  Claims: [
    {{"claim": "Proficient in Python", "kind": "skill"}},
    {{"claim": "Proficient in Rust", "kind": "skill"}},
    {{"claim": "Built a REST API with FastAPI", "kind": "project"}}
  ]

EXCLUDE (parsed elsewhere, never verifiable from code): GPA/CGPA, contact info,
"has a GitHub/LinkedIn profile", education pedigree, total years of experience,
soft skills, leadership titles.

"kind" is one of: "project" (built/deployed something), "skill" (knows a
technology), "metric" (quantitative outcome: latency, users, revenue),
"experience" (role at an organization).

Order by importance: projects first, then skills. Max claims: {max_claims}.
Return only JSON: {{"claims": [{{"claim": "...", "kind": "..."}}]}}

Resume:
{resume_text[:6000]}
"""
        data = self._complete_json(prompt, max_tokens=900, system=EXTRACTOR_SYSTEM)
        tagged: list[dict[str, str]] = []
        for item in data.get("claims", []):
            if isinstance(item, dict):
                claim = str(item.get("claim", "")).strip()
                kind = str(item.get("kind", "")).strip().lower()
            else:
                claim, kind = str(item).strip(), ""
            if claim:
                tagged.append({"claim": claim, "kind": kind if kind in CLAIM_KINDS else "project"})
        if not tagged:
            tagged = [{"claim": claim, "kind": ""} for claim in fallback_claims]
        return sanitize_claims(tagged, max_claims)

    def extract_claims(self, resume_text: str, fallback_claims: list[str], max_claims: int = 8) -> list[str]:
        return [item["claim"] for item in self.extract_claims_tagged(resume_text, fallback_claims, max_claims)]

    # -------------------------------------------------------------- judging

    def judge_claim(self, claim: str, evidence: list[EvidenceItem]) -> ClaimVerdict | None:
        evidence = evidence[:8]
        evidence_block = "\n\n".join(
            f"[{idx}] {_evidence_header(item)}\n{item.text[:1200]}"
            for idx, item in enumerate(evidence, start=1)
        )
        if not evidence_block:
            return None
        prompt = f"""Judge this resume claim strictly against the numbered GitHub evidence.

Rules:
- SUPPORTED requires evidence that affirmatively backs the claim.
- REFUTED requires evidence that CONTRADICTS the claim. Mere absence of evidence is NOT refutation.
- If the evidence neither proves nor contradicts the claim, return NOT_ENOUGH_INFO.

Confidence rubric (use the full range, do not default to 0.5):
- 0.9-1.0: direct evidence - code, dependency manifest, or commits show exactly this
- 0.7-0.8: strong indirect - README and commit history align with the claim
- 0.4-0.6: weak or partial evidence
- 0.1-0.3: near guesswork

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
        data = self._complete_json(prompt, max_tokens=700, system=JUDGE_SYSTEM, model=self.judge_model)
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

    # -------------------------------------------------------------- writing

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
        prompt = f"""Write a recruiter brief for this candidate. Max 120 words, plain text, exactly this shape:

VERIFIED STRENGTHS: what the evidence actually supports (only SUPPORTED claims; name the repos). If none: "None verified."
UNVERIFIED / CONTRADICTED: the notable claims that did not survive verification, one line each.
BOTTOM LINE: one sentence - fit for the role and what the interview must probe.

Never present an unverified claim as fact.

Candidate: {name or "Unknown"}
Scores: {json.dumps(scores)}
{verdict_block}

Job Description:
{job_description[:2500]}

Resume:
{resume_text[:4000]}
"""
        text = self.complete(prompt, max_tokens=500, temperature=0.3, system=WRITER_SYSTEM)
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
        gaps.sort(key=lambda v: 0 if v.verdict == "REFUTED" else 1)
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

Unverified/refuted claims (most serious first):
{gap_block or "(none)"}

Authenticity concerns:
{signal_block or "(none)"}
"""
        data = self._complete_json(prompt, max_tokens=1600, temperature=0.4, system=INTERVIEWER_SYSTEM)
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


# --------------------------------------------------------------- pure helpers


def sanitize_claims(tagged: list[dict[str, str]], max_claims: int) -> list[dict[str, str]]:
    """Deterministic net under the LLM rules: drop junk claims, split compound
    skill lists, dedupe, cap."""
    result: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(claim: str, kind: str) -> None:
        claim = re.sub(r"\s+", " ", claim).strip(" .;")
        key = claim.lower()
        if claim and key not in seen and not _JUNK_CLAIM_RE.search(claim):
            seen.add(key)
            result.append({"claim": claim, "kind": kind})

    for item in tagged:
        claim, kind = item.get("claim", ""), item.get("kind", "")
        for piece in _split_compound(claim):
            add(piece, kind or ("skill" if piece != claim else kind))
    return result[:max_claims]


def _split_compound(claim: str) -> list[str]:
    """Split "Proficient in A, B, C, and D" style lists into one claim per item."""
    match = _COMPOUND_LEAD_RE.match(claim.strip())
    if not match:
        return [claim]
    items = [part.strip(" .") for part in re.split(r",\s*|\s+and\s+", match.group("items")) if part.strip(" .")]
    if len(items) < 3 or any(len(part) > 40 for part in items):
        return [claim]
    lead = match.group("lead").strip()
    return [f"{lead} {part}" for part in items]


def _evidence_header(item: EvidenceItem) -> str:
    meta = item.metadata or {}
    parts = [f"{item.repo_name} ({item.source_type})"]
    languages = meta.get("languages")
    if languages:
        parts.append("langs: " + ",".join(sorted(languages)[:5]))
    stars = meta.get("stars")
    if stars:
        parts.append(f"stars: {stars}")
    if meta.get("fork"):
        parts.append("FORK")
    return " — ".join(parts)


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
                "question": template.format(claim=shorten(verdict.claim, 110)),
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


def fallback_summary(name: str | None, scores: dict[str, float]) -> str:
    display_name = name or "This candidate"
    return (
        f"{display_name} was scored against the JD using parsed resume signals and GitHub evidence. "
        f"Final score: {scores.get('final_score', 0):.1f}/100.\n\n"
        "Tinker summary generation was unavailable, so this summary is deterministic and evidence-first."
    )


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


def _is_bad_request(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 400:
        return True
    message = str(exc).lower()
    return "400" in message or "response_format" in message or "unsupported" in message


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))
