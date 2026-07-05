"""Shared Streamlit rendering helpers and session-state accessors."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .privacy import blind_display_name
from .schemas import CandidateReport

SHORTLIST_OPTIONS = ["", "shortlist", "maybe", "reject"]

DISCLAIMER = (
    "**Decision support, not a hiring decision.** Verdicts and scores point a human "
    "reviewer at evidence; they must never auto-reject a candidate."
)


def render_disclaimer() -> None:
    st.caption(DISCLAIMER)


def candidate_label(report: CandidateReport, blind: bool, index: int) -> str:
    if blind:
        return blind_display_name(index)
    return report.profile.name or report.profile.source_name


def get_reports() -> list[CandidateReport]:
    return st.session_state.get("reports", [])


def get_shortlist() -> dict[str, str]:
    return st.session_state.setdefault("shortlist", {})


def get_overrides() -> dict[str, str]:
    return st.session_state.setdefault("overrides", {})


def shortlist_key(report: CandidateReport) -> str:
    return report.profile.source_name or report.profile.name or "unknown"


def render_verdicts(report: CandidateReport) -> None:
    verdict_rows = [
        {
            "claim": verdict.claim,
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            "evidence_source": verdict.evidence_source,
            "explanation": verdict.explanation,
        }
        for verdict in report.verdicts
    ]
    st.dataframe(
        pd.DataFrame(verdict_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "evidence_source": st.column_config.LinkColumn("evidence", display_text="open"),
        },
    )
    for idx, verdict in enumerate(report.verdicts, start=1):
        with st.expander(f"Evidence {idx}: {verdict.verdict} - {verdict.claim[:80]}"):
            if verdict.evidence_source:
                st.markdown(f"[{verdict.evidence_source}]({verdict.evidence_source})")
            else:
                st.caption("No evidence source")
            st.write(verdict.evidence or "No evidence snippet available.")


def render_fraud_panel(report: CandidateReport) -> None:
    if not report.fraud_signals:
        st.success("No fraud signals detected in the inspected repositories.")
        return
    renderers = {"high": st.error, "warn": st.warning, "info": st.info}
    for signal in report.fraud_signals:
        renderer = renderers.get(signal.severity, st.info)
        with st.container():
            renderer(f"**{signal.title}**\n\n{signal.detail}")
            if signal.evidence_url:
                st.link_button("View evidence on GitHub", signal.evidence_url)


def render_interview_questions(report: CandidateReport) -> None:
    if not report.interview_questions:
        st.caption("No targeted interview questions (all claims verified, no signals).")
        return
    for idx, item in enumerate(report.interview_questions, start=1):
        st.markdown(f"**{idx}. {item.get('question', '')}**")
        if item.get("targets"):
            st.caption(f"Probes: {item['targets']}")
