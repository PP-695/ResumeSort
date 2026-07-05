from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import BinaryIO, Callable, Iterable

import pandas as pd

from .fraud import analyze_fraud_signals
from .github_evidence import fetch_github_snapshot, snapshot_to_evidence
from .llm import TinkerLLM
from .parser import parse_resume_pdf
from .privacy import redact_text_for_llm
from .schemas import AnalysisSettings, CandidateReport
from .scoring import combine_scores, score_authenticity, score_jd_fit, score_verification
from .verification import verify_claims

EVIDENCE_EXPORT_TEXT_CAP = 2000


def analyze_resumes(
    files: Iterable[BinaryIO],
    job_description: str,
    settings: AnalysisSettings,
    progress_callback: Callable[[str], None] | None = None,
) -> list[CandidateReport]:
    llm = TinkerLLM(base_model=settings.tinker_base_model, enabled=settings.use_tinker)
    reports: list[CandidateReport] = []

    def progress(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    # Phase 1: parse every resume up front so we know all GitHub URLs.
    profiles = []
    for file_obj in files:
        source_name = getattr(file_obj, "name", "uploaded_resume.pdf")
        progress(f"Parsing {source_name}...")
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        profiles.append(parse_resume_pdf(file_obj, source_name=source_name, jd_text=job_description))

    # Phase 2: prefetch snapshots concurrently (network-bound); the loop below
    # then reads from the warm TTL cache. Workers touch no Streamlit state.
    unique_urls = {p.github_url for p in profiles if p.github_url}
    if len(unique_urls) > 1:
        progress(f"Fetching GitHub evidence for {len(unique_urls)} profiles in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    fetch_github_snapshot,
                    url,
                    token=settings.github_token,
                    max_repos=settings.max_repos,
                    deep=settings.deep_fraud_checks,
                )
                for url in unique_urls
            ]
            for future in futures:
                try:
                    future.result()
                except Exception:
                    pass  # per-candidate fetch below will surface the flag

    # Phase 3: per-candidate scoring (sequential; LLM calls dominate here).
    for profile in profiles:
        source_name = profile.source_name
        calls_before = llm.api_calls
        successes_before = llm.api_successes

        progress(f"Collecting evidence for {profile.name or source_name}...")
        snapshot, flags = fetch_github_snapshot(
            profile.github_url,
            token=settings.github_token,
            max_repos=settings.max_repos,
            deep=settings.deep_fraud_checks,
        )
        evidence = snapshot_to_evidence(snapshot)

        progress("Analyzing fraud signals...")
        fraud_signals = analyze_fraud_signals(snapshot, profile)

        llm_text = profile.raw_text
        if settings.blind_mode:
            llm_text = redact_text_for_llm(profile.raw_text, profile)

        progress("Extracting and verifying claims...")
        fallback_claims = profile.projects + profile.experience
        claims = llm.extract_claims(llm_text, fallback_claims=fallback_claims, max_claims=settings.max_claims)
        verdicts = verify_claims(claims, evidence, llm)

        jd_fit = score_jd_fit(profile, job_description)
        verification = score_verification(verdicts)
        authenticity = score_authenticity(evidence, flags, fraud_signals)
        scores = combine_scores(
            jd_fit,
            verification,
            authenticity,
            (settings.jd_fit_weight, settings.verification_weight, settings.authenticity_weight),
        )
        score_dict = asdict(scores)

        progress("Generating summary and interview questions...")
        summary_name = None if settings.blind_mode else profile.name
        summary = llm.summarize(summary_name, llm_text, job_description, score_dict, verdicts=verdicts)
        interview_questions = llm.generate_interview_questions(summary_name, verdicts, fraud_signals)

        llm_status = llm.status
        reports.append(
            CandidateReport(
                profile=profile,
                scores=scores,
                verdicts=verdicts,
                flags=flags,
                summary=summary,
                fraud_signals=fraud_signals,
                evidence=_cap_evidence(evidence),
                interview_questions=interview_questions,
                llm_provider=llm_status.provider,
                llm_api_calls=llm.api_calls - calls_before,
                llm_api_successes=llm.api_successes - successes_before,
                llm_error=llm_status.reason,
            )
        )

    return sorted(reports, key=lambda report: report.scores.final_score, reverse=True)


def reports_to_dataframe(reports: list[CandidateReport], blind: bool = False) -> pd.DataFrame:
    rows = []
    for index, report in enumerate(reports):
        rows.append(
            {
                "name": f"Candidate {index + 1}" if blind else (report.profile.name or report.profile.source_name),
                "email": None if blind else report.profile.email,
                "github": None if blind else report.profile.github_url,
                "cgpa": None if blind else report.profile.cgpa,
                "skills": ", ".join(report.profile.skills),
                "jd_fit_score": report.scores.jd_fit_score,
                "verification_score": report.scores.verification_score,
                "authenticity_score": report.scores.authenticity_score,
                "final_score": report.scores.final_score,
                "fraud_signals": len(report.fraud_signals),
                "flags": "; ".join(report.flags),
                "llm_provider": report.llm_provider,
                "llm_api_calls": report.llm_api_calls,
                "llm_api_successes": report.llm_api_successes,
                "llm_error": report.llm_error,
                "summary": report.summary,
            }
        )
    return pd.DataFrame(rows)


def reports_to_json(reports: list[CandidateReport], blind: bool = False) -> str:
    payload = [report.to_dict() for report in reports]
    if blind:
        for index, item in enumerate(payload):
            _blind_report_dict(item, index)
    return json.dumps(payload, indent=2)


def _blind_report_dict(item: dict, index: int) -> None:
    profile = item.get("profile", {})
    profile["name"] = f"Candidate {index + 1}"
    profile["email"] = None
    profile["github_url"] = None
    profile["cgpa"] = None
    profile["raw_text"] = "[redacted in blind mode]"
    profile["source_name"] = f"candidate_{index + 1}.pdf"


def build_audit_log(
    reports: list[CandidateReport],
    settings: AnalysisSettings,
    started_at: str,
    finished_at: str,
) -> str:
    """A machine-readable record of one analysis run, for accountability/export."""
    from . import __version__

    return json.dumps(
        {
            "app": "grifter-filter",
            "app_version": __version__,
            "started_at": started_at,
            "finished_at": finished_at,
            "model": settings.tinker_base_model,
            "use_tinker": settings.use_tinker,
            "blind_mode": settings.blind_mode,
            "deep_fraud_checks": settings.deep_fraud_checks,
            "weights": {
                "jd_fit": settings.jd_fit_weight,
                "verification": settings.verification_weight,
                "authenticity": settings.authenticity_weight,
            },
            "candidates": [
                {
                    "source": f"candidate_{index + 1}.pdf" if settings.blind_mode else report.profile.source_name,
                    "name": f"Candidate {index + 1}" if settings.blind_mode else report.profile.name,
                    "scores": asdict(report.scores),
                    "verdicts": [asdict(v) for v in report.verdicts],
                    "fraud_signals": [asdict(s) for s in report.fraud_signals],
                    "flags": report.flags,
                    "llm_provider": report.llm_provider,
                    "llm_api_calls": report.llm_api_calls,
                }
                for index, report in enumerate(reports)
            ],
            "disclaimer": (
                "Decision-support output only. Not a hiring decision. "
                "A human must review evidence before acting on any verdict."
            ),
        },
        indent=2,
    )


def _cap_evidence(evidence):
    capped = []
    for item in evidence:
        if len(item.text) > EVIDENCE_EXPORT_TEXT_CAP:
            item.text = item.text[:EVIDENCE_EXPORT_TEXT_CAP]
        capped.append(item)
    return capped
