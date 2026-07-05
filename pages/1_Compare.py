from __future__ import annotations

import streamlit as st

from resumesort import ui

st.set_page_config(page_title="Compare - Grifter Filter", page_icon="🔎", layout="wide")

st.title("Compare candidates")
ui.render_disclaimer()

reports = ui.get_reports()
if not reports:
    st.info("Run an analysis on the main page first, then come back to compare candidates.")
    st.stop()

blind = bool(st.session_state.get("run_meta") and st.session_state["run_meta"]["settings"].blind_mode)
labels = [ui.candidate_label(report, blind, index) for index, report in enumerate(reports)]

selected_labels = st.multiselect(
    "Candidates to compare (2-5)",
    labels,
    default=labels[: min(3, len(labels))],
    max_selections=5,
)

if len(selected_labels) < 2:
    st.caption("Select at least two candidates.")
    st.stop()

selected_reports = [reports[labels.index(label)] for label in selected_labels]
shortlist = ui.get_shortlist()

columns = st.columns(len(selected_reports))
for column, label, report in zip(columns, selected_labels, selected_reports):
    with column:
        st.subheader(label)
        status = shortlist.get(ui.shortlist_key(report), "")
        if status:
            st.caption(f"Status: **{status}**")

        st.metric("Final score", f"{report.scores.final_score:.1f}")
        st.metric("JD fit", f"{report.scores.jd_fit_score:.1f}")
        st.metric("Verification", f"{report.scores.verification_score:.1f}")
        st.metric("Authenticity", f"{report.scores.authenticity_score:.1f}")

        supported = [v for v in report.verdicts if v.verdict == "SUPPORTED"]
        refuted = [v for v in report.verdicts if v.verdict == "REFUTED"]
        nei = [v for v in report.verdicts if v.verdict == "NOT_ENOUGH_INFO"]
        st.markdown(f"**Verdicts:** {len(supported)} supported / {len(nei)} NEI / {len(refuted)} refuted")

        st.markdown("**Top supported claims**")
        if supported:
            for verdict in supported[:3]:
                st.markdown(f"- {verdict.claim[:120]}")
        else:
            st.caption("None verified.")

        high = sum(1 for s in report.fraud_signals if s.severity == "high")
        warn = sum(1 for s in report.fraud_signals if s.severity == "warn")
        st.markdown(f"**Fraud signals:** {high} high / {warn} warn / {len(report.fraud_signals)} total")
        for signal in report.fraud_signals[:3]:
            st.caption(f"[{signal.severity}] {signal.title[:100]}")
