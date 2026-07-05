"""JD Requirements Matrix: parse a job description into discrete requirements,
then grade each candidate's evidence coverage per requirement.

This mirrors how hiring managers actually evaluate ("does anyone meet my five
must-haves?") rather than presenting one blended score. Coverage grades are
deliberately conservative:

- "met"     — a SUPPORTED claim (or parsed skill) clearly matches the requirement
- "partial" — related evidence exists but is unverified or weakly matched
- "none"    — nothing in the candidate's footprint touches the requirement
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import TinkerLLM, parse_json_object
from .schemas import CandidateReport
from .scoring import cosine_text_similarity

MET_THRESHOLD = 0.25
PARTIAL_THRESHOLD = 0.12
MAX_REQUIREMENTS = 10

_MUST_HINTS = ("must", "required", "requirement", "need", "essential", "strong fundamentals")


@dataclass
class Requirement:
    text: str
    must_have: bool


@dataclass
class Coverage:
    level: str  # "met" | "partial" | "none"
    evidence_url: str = ""
    matched_claim: str = ""


def parse_jd_requirements(job_description: str, llm: TinkerLLM) -> list[Requirement]:
    """LLM-first JD decomposition with a deterministic line-splitting fallback."""
    if llm.status.enabled:
        prompt = f"""Extract the discrete candidate requirements from this job description.
Return only JSON: {{"requirements": [{{"requirement": "one specific skill/experience", "must_have": true}}]}}
Rules: max {MAX_REQUIREMENTS} requirements; each atomic (one skill or experience per entry);
"must_have" true only for hard requirements, false for nice-to-haves.

Job description:
{job_description[:4000]}
"""
        data = parse_json_object(llm.complete(prompt, max_tokens=700))
        requirements = []
        for item in data.get("requirements", []):
            if isinstance(item, dict) and str(item.get("requirement", "")).strip():
                requirements.append(
                    Requirement(
                        text=str(item["requirement"]).strip()[:160],
                        must_have=bool(item.get("must_have", False)),
                    )
                )
        if requirements:
            return requirements[:MAX_REQUIREMENTS]
    return _fallback_requirements(job_description)


def _fallback_requirements(job_description: str) -> list[Requirement]:
    requirements: list[Requirement] = []
    for line in job_description.splitlines():
        cleaned = line.strip().lstrip("-•*· ").strip()
        if not (8 <= len(cleaned) <= 160):
            continue
        if not line.strip().startswith(("-", "•", "*", "·")):
            continue
        must = any(hint in cleaned.lower() for hint in _MUST_HINTS)
        requirements.append(Requirement(text=cleaned, must_have=must))
    if not requirements:
        # Last resort: sentences mentioning must-hints.
        for sentence in re.split(r"[.\n]", job_description):
            cleaned = sentence.strip()
            if 12 <= len(cleaned) <= 160 and any(hint in cleaned.lower() for hint in _MUST_HINTS):
                requirements.append(Requirement(text=cleaned, must_have=True))
    return requirements[:MAX_REQUIREMENTS]


def coverage_for_report(requirement: Requirement, report: CandidateReport) -> Coverage:
    supported = [v for v in report.verdicts if v.verdict == "SUPPORTED"]
    refuted = [v for v in report.verdicts if v.verdict == "REFUTED"]
    other = [v for v in report.verdicts if v.verdict != "SUPPORTED"]

    # A refuted claim touching this requirement vetoes any "met" shortcut —
    # a listed skill whose flagship claim was contradicted is at best partial.
    refuted_hit = any(
        cosine_text_similarity(requirement.text, v.claim) >= PARTIAL_THRESHOLD for v in refuted
    )

    # Direct skill hit counts as met (skills are vocabulary-filtered upstream).
    req_lower = requirement.text.lower()
    if not refuted_hit:
        for skill in report.profile.skills:
            if len(skill) >= 2 and re.search(rf"(?<![a-z0-9]){re.escape(skill.lower())}(?![a-z0-9])", req_lower):
                url = supported[0].evidence_source if supported and supported[0].evidence_source else ""
                return Coverage("met", url, f"skill: {skill}")

    best_supported, best_supported_score = None, 0.0
    for verdict in supported:
        score = cosine_text_similarity(requirement.text, verdict.claim)
        if score > best_supported_score:
            best_supported, best_supported_score = verdict, score
    if best_supported and best_supported_score >= MET_THRESHOLD:
        return Coverage("met", best_supported.evidence_source, best_supported.claim)

    best_other_score = best_supported_score
    best_other_claim = best_supported.claim if best_supported else ""
    for verdict in other:
        score = cosine_text_similarity(requirement.text, verdict.claim)
        if score > best_other_score:
            best_other_score, best_other_claim = score, verdict.claim
    for item in report.evidence:
        score = cosine_text_similarity(requirement.text, item.text[:1500])
        if score > best_other_score:
            best_other_score, best_other_claim = score, f"evidence: {item.repo_name}"
    if best_other_score >= PARTIAL_THRESHOLD:
        return Coverage("partial", "", best_other_claim)
    return Coverage("none")


def build_matrix(
    requirements: list[Requirement],
    reports: list[CandidateReport],
) -> list[tuple[Requirement, list[Coverage]]]:
    ordered = sorted(requirements, key=lambda r: not r.must_have)
    return [(req, [coverage_for_report(req, report) for report in reports]) for req in ordered]


COVERAGE_BADGE = {
    "met": ":green-badge[✓ met]",
    "partial": ":yellow-badge[◐ partial]",
    "none": ":gray-badge[— none]",
}


def matrix_markdown(matrix: list[tuple[Requirement, list[Coverage]]], labels: list[str]) -> str:
    """Markdown table with badge cells (badge directives render inside tables)."""
    header = "| Requirement | " + " | ".join(labels) + " |"
    divider = "|" + "---|" * (len(labels) + 1)
    rows = [header, divider]
    for requirement, coverages in matrix:
        prefix = ":red-badge[must] " if requirement.must_have else ""
        cells = []
        for coverage in coverages:
            badge = COVERAGE_BADGE.get(coverage.level, "")
            if coverage.level == "met" and coverage.evidence_url:
                cells.append(f"{badge} [↗]({coverage.evidence_url})")
            else:
                cells.append(badge)
        rows.append(f"| {prefix}{requirement.text} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
