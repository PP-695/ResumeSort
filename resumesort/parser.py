from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import BinaryIO


from .schemas import CandidateProfile


SKILL_ALIASES = [
    "Python", "Java", "JavaScript", "TypeScript", "React", "ReactJS", "Node.js",
    "FastAPI", "Flask", "Django", "MongoDB", "PostgreSQL", "MySQL", "Kafka",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "REST", "GraphQL",
    "Machine Learning", "ML", "NLP", "Transformers", "PyTorch", "TensorFlow",
    "scikit-learn", "Pandas", "NumPy", "SQL", "Git", "Linux",
]

LEADERSHIP_KEYWORDS = [
    "president", "secretary", "lead", "leader", "organizer", "founder",
    "coordinator", "captain", "mentor", "head",
]


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """Extract page text plus hyperlink-annotation URIs.

    Modern resumes hide URLs behind link text ("GitHub", "LinkedIn"), so the
    visible text alone often contains no github.com URL - the target lives in
    the PDF's link annotations. We append those URIs so downstream regexes see
    them.
    """
    import pdfplumber

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_obj.read())
        tmp_path = Path(tmp.name)
    try:
        text_parts: list[str] = []
        link_uris: dict[str, None] = {}
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
                try:
                    for link in page.hyperlinks:
                        uri = link.get("uri")
                        if uri:
                            link_uris[uri] = None
                except Exception:
                    pass
        if link_uris:
            text_parts.append("\n".join(link_uris))
        return "\n".join(text_parts).strip()
    finally:
        tmp_path.unlink(missing_ok=True)


def extract_github_url(text: str) -> str | None:
    match = re.search(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/?", text)
    if match:
        return match.group(0).rstrip("/")
    # Protocol-less mention, e.g. "github.com/username" in plain resume text.
    match = re.search(r"(?<![\w/])(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/?", text)
    if match:
        return f"https://github.com/{match.group(1)}"
    return None


def extract_email(text: str) -> str | None:
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return match.group(0) if match else None


def extract_cgpa(text: str) -> float | None:
    """CGPA normalized to a 10-point scale (US 4.0-scale GPAs are converted)."""
    # Explicit scale: "3.8/4.0", "8.74 / 10"
    match = re.search(
        r"(?:CGPA|GPA)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/\s*(4(?:\.0)?|10)",
        text,
        re.IGNORECASE,
    )
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(4(?:\.0)?|10)\s*(?:CGPA|GPA)", text, re.IGNORECASE)
    if match:
        value, scale = float(match.group(1)), float(match.group(2))
        if 0 <= value <= scale:
            return round(value * 10.0 / scale, 2)

    # No explicit scale: infer from magnitude (<= 4.3 on a GPA line means 4.0 scale).
    match = re.search(r"(?:CGPA|GPA)\s*[:\-]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        if 0 <= value <= 4.3:
            return round(value * 2.5, 2)
        if value <= 10:
            return value
    return None


def extract_name(text: str, email: str | None = None) -> str | None:
    for line in text.splitlines()[:12]:
        cleaned = line.strip()
        if not cleaned or "resume" in cleaned.lower() or "curriculum" in cleaned.lower():
            continue
        if email and email in cleaned:
            continue
        if re.search(r"https?://|github\.com|linkedin\.com|@", cleaned, re.IGNORECASE):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z.'-]+", cleaned)
        if 1 <= len(words) <= 5:
            return " ".join(words)
    return None


def extract_skills(text: str, jd_text: str = "") -> list[str]:
    """Skills = alias-list hits plus skills-line tokens that look like real tech.

    Free-text tokens are kept only if they appear in the known-tech vocabulary or
    in the JD — otherwise junk like section labels dilutes the JD-fit denominator.
    """
    found: set[str] = set()
    lowered = text.lower()
    for skill in SKILL_ALIASES:
        pattern = r"(?<![A-Za-z0-9+#.])" + re.escape(skill.lower()) + r"(?![A-Za-z0-9+#.])"
        if re.search(pattern, lowered):
            found.add(skill)

    vocab = {alias.lower() for alias in SKILL_ALIASES}
    jd_tokens = set(re.findall(r"[a-z][a-z0-9+#.]{1,}", jd_text.lower()))
    for line in text.splitlines():
        if "skill" in line.lower():
            for token in re.split(r"[,|;:/•·]", line):
                cleaned = token.strip(" -\t")
                key = cleaned.lower()
                if not (2 <= len(cleaned) <= 28) or "skill" in key:
                    continue
                if key in vocab or key in jd_tokens:
                    found.add(cleaned)
    return sorted(found, key=str.lower)


def extract_projects(text: str) -> list[str]:
    projects: list[str] = []
    in_projects = False
    for raw_line in text.splitlines():
        line = raw_line.strip(" \t-")
        if not line:
            continue
        lower = line.lower()
        if "project" in lower:
            in_projects = True
            if len(line) > 12:
                projects.append(line)
            continue
        if in_projects and any(key in lower for key in ["education", "experience", "skills", "certification"]):
            in_projects = False
        if in_projects and 18 <= len(line) <= 180:
            projects.append(line)
    return _dedupe(projects)[:10]


def extract_experience(text: str) -> list[str]:
    keywords = ["intern", "engineer", "developer", "analyst", "manager", "worked", "built", "led", "developed"]
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" \t-")
        if 12 <= len(line) <= 220 and any(keyword in line.lower() for keyword in keywords):
            lines.append(line)
    return _dedupe(lines)[:12]


def extract_years_experience(text: str) -> float | None:
    matches = re.findall(
        r"(\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+|work\s+|industry\s+)?experience",
        text,
        re.IGNORECASE,
    )
    values = [float(value) for value in matches if 0 < float(value) <= 50]
    return max(values) if values else None


def has_leadership(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in LEADERSHIP_KEYWORDS)


def parse_resume_text(text: str, source_name: str = "", jd_text: str = "") -> CandidateProfile:
    email = extract_email(text)
    return CandidateProfile(
        name=extract_name(text, email=email),
        email=email,
        github_url=extract_github_url(text),
        cgpa=extract_cgpa(text),
        claimed_years_experience=extract_years_experience(text),
        skills=extract_skills(text, jd_text=jd_text),
        projects=extract_projects(text),
        experience=extract_experience(text),
        leadership=has_leadership(text),
        raw_text=text,
        source_name=source_name,
    )


def parse_resume_pdf(file_obj: BinaryIO, source_name: str = "", jd_text: str = "") -> CandidateProfile:
    return parse_resume_text(extract_text_from_pdf(file_obj), source_name=source_name, jd_text=jd_text)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


