"""Blind-screening support: redact identity signals before LLM prompts and display.

Motivated by resume-screening bias research (e.g. the 2024 University of Washington
study finding LLM screeners favored white-associated names in 85% of trials).
Redaction is best-effort - it removes direct identifiers (name, email, GitHub handle,
CGPA) and education-section lines, not every possible proxy.
"""

from __future__ import annotations

import re

from .schemas import CandidateProfile

EDUCATION_KEYWORDS = (
    "university", "college", "institute", "school", "b.tech", "btech", "b.e.",
    "m.tech", "mtech", "bachelor", "master", "cgpa", "gpa",
)


def redact_text_for_llm(text: str, profile: CandidateProfile) -> str:
    redacted = text
    if profile.name:
        redacted = re.sub(re.escape(profile.name), "[CANDIDATE]", redacted, flags=re.IGNORECASE)
    if profile.email:
        redacted = redacted.replace(profile.email, "[EMAIL]")
    redacted = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", redacted)
    if profile.github_url:
        redacted = redacted.replace(profile.github_url, "[GITHUB]")
    redacted = re.sub(r"(?:CGPA|GPA)\s*[:\-]?\s*\d+(?:\.\d+)?(?:\s*/\s*10)?", "[GPA]", redacted, flags=re.IGNORECASE)

    lines = []
    for line in redacted.splitlines():
        lowered = line.lower()
        if any(keyword in lowered for keyword in EDUCATION_KEYWORDS):
            lines.append("[EDUCATION LINE REDACTED]")
        else:
            lines.append(line)
    return "\n".join(lines)


def blind_display_name(index: int) -> str:
    return f"Candidate {index + 1}"
