from __future__ import annotations

import pandas as pd
import streamlit as st

from resumesort import ui

st.set_page_config(page_title="Compare - Grifter Filter", page_icon="🔎", layout="wide")
ui.inject_css()

st.title("Compare candidates")
ui.render_disclaimer()

reports = ui.get_reports()
if not reports:
    st.info("Run an analysis on the main page first, then come back to compare candidates.")
    st.stop()

run_meta = st.session_state.get("run_meta", {})
run_settings = run_meta.get("settings")
blind = bool(run_settings and run_settings.blind_mode)
labels = [ui.candidate_label(report, blind, index) for index, report in enumerate(reports)]

if len(labels) <= 8:
    selected_labels = st.pills(
        "Candidates to compare (2-4)",
        labels,
        selection_mode="multi",
        default=labels[: min(3, len(labels))],
    )
else:
    selected_labels = st.multiselect(
        "Candidates to compare (2-4)",
        labels,
        default=labels[: min(3, len(labels))],
    )

if not selected_labels or len(selected_labels) < 2:
    st.caption("Select at least two candidates.")
    st.stop()
selected_labels = selected_labels[:4]

selected_reports = [reports[labels.index(label)] for label in selected_labels]
shortlist = ui.get_shortlist()

best = {
    "jd_fit_score": max(r.scores.jd_fit_score for r in selected_reports),
    "verification_score": max(r.scores.verification_score for r in selected_reports),
    "authenticity_score": max(r.scores.authenticity_score for r in selected_reports),
}
best_final = max(r.scores.final_score for r in selected_reports)

columns = st.columns(len(selected_reports))
for column, label, report in zip(columns, selected_labels, selected_reports):
    with column:
        with st.container(border=True):
            title = f"#### {label}"
            status = shortlist.get(ui.shortlist_key(report), "")
            if status:
                title += f" &nbsp; {ui.STATUS_BADGE.get(status, '')}"
            st.markdown(title)

            top_chip = " :primary-badge[Top]" if report.scores.final_score == best_final else ""
            st.markdown(
                f'<span class="gf-num" style="font-size:2rem; font-weight:650;">'
                f"{report.scores.final_score:.1f}</span>",
                unsafe_allow_html=True,
            )
            if top_chip:
                st.markdown(top_chip)
            ui.render_score_bars(report.scores, best=best)

            st.markdown(ui.verdict_chips(report.verdicts))
            st.markdown(ui.fraud_chips(report))

            supported = [v for v in report.verdicts if v.verdict == "SUPPORTED"]
            st.markdown("**Top verified claims**")
            if supported:
                for verdict in supported[:3]:
                    st.markdown(f"- {verdict.claim[:110]}")
            else:
                st.caption("None verified.")

with st.expander("Full score table"):
    score_table = pd.DataFrame(
        {
            label: {
                "Final": report.scores.final_score,
                "JD fit": report.scores.jd_fit_score,
                "Verification": report.scores.verification_score,
                "Authenticity": report.scores.authenticity_score,
                "Supported claims": sum(1 for v in report.verdicts if v.verdict == "SUPPORTED"),
                "High signals": sum(1 for s in report.fraud_signals if s.severity == "high"),
            }
            for label, report in zip(selected_labels, selected_reports)
        }
    )
    st.dataframe(score_table, width="stretch")
