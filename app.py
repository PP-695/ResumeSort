from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from resumesort import AnalysisSettings, analyze_resumes, ui
from resumesort.llm import TinkerLLM
from resumesort.pipeline import build_audit_log, reports_to_dataframe, reports_to_json
from resumesort.report_pdf import build_candidate_pdf
from resumesort.sample_data import load_sample_reports


load_dotenv()

DEFAULT_JD = """We are hiring Software Development Engineers who are passionate about building scalable backend systems,
designing APIs, and working with real-time data. Candidates must have strong fundamentals in Python
and experience with FastAPI, MongoDB, and RESTful services.

We value open-source contributions and expect candidates to showcase real-world projects, preferably on GitHub,
demonstrating skills in backend development, cloud-native systems, or machine learning pipelines.

The ideal candidate will have hands-on experience with FastAPI or Flask/Django, NoSQL databases like MongoDB,
real-time data processing or API integrations, basic ML pipeline knowledge, leadership, and a CGPA above 7.5.
"""

st.set_page_config(page_title="Grifter Filter", page_icon="🔎", layout="wide")
ui.inject_css()


def sidebar_settings() -> AnalysisSettings:
    st.sidebar.markdown('<div class="gf-kicker">Grifter Filter</div>', unsafe_allow_html=True)
    st.sidebar.caption("Evidence-backed resume ranking")

    github_token = os.getenv("GITHUB_TOKEN") or None

    blind_mode = st.sidebar.toggle(
        "Blind screening",
        value=False,
        help="Redacts name, email, GitHub handle, CGPA, and education lines from LLM prompts and all views/exports.",
    )
    deep_fraud = st.sidebar.toggle(
        "Deep fraud checks",
        value=bool(github_token),
        disabled=not github_token,
        help="Contributor share, commit timelines, and root-file checks. Requires a GitHub token.",
    )

    with st.sidebar.expander("Scoring weights"):
        jd_weight = st.slider("JD fit", 0.0, 1.0, 0.45, 0.05)
        verification_weight = st.slider("Verification", 0.0, 1.0, 0.35, 0.05)
        authenticity_weight = st.slider("Authenticity", 0.0, 1.0, 0.20, 0.05)
        total = (jd_weight + verification_weight + authenticity_weight) or 1.0
        st.caption(
            f"Normalized: {jd_weight / total:.0%} fit · {verification_weight / total:.0%} "
            f"verification · {authenticity_weight / total:.0%} authenticity"
        )

    with st.sidebar.expander("Engine & limits"):
        tinker_model = st.text_input("Tinker model", value=os.getenv("TINKER_BASE_MODEL", "openai/gpt-oss-20b"))
        use_tinker = st.toggle("Use Tinker", value=bool(os.getenv("TINKER_API_KEY")))
        max_claims = st.slider("Claims per resume", 3, 15, 8)
        max_repos = st.slider("GitHub repos to inspect", 5, 30, 10)

    tinker_badge = ":green-badge[Tinker connected]" if os.getenv("TINKER_API_KEY") and use_tinker else ":gray-badge[Tinker off — heuristics]"
    github_badge = ":green-badge[GitHub token]" if github_token else ":orange-badge[GitHub anonymous — shallow]"
    st.sidebar.markdown(f"{tinker_badge}  \n{github_badge}")

    return AnalysisSettings(
        tinker_base_model=tinker_model,
        github_token=github_token,
        use_tinker=use_tinker,
        max_claims=max_claims,
        max_repos=max_repos,
        jd_fit_weight=jd_weight,
        verification_weight=verification_weight,
        authenticity_weight=authenticity_weight,
        deep_fraud_checks=deep_fraud,
        blind_mode=blind_mode,
    )


def analyze_form(settings: AnalysisSettings, collapsed: bool) -> None:
    container = st.expander("New screening", expanded=False) if collapsed else st.container(border=True)
    with container:
        if not collapsed:
            st.markdown("**New screening**")
        upload_col, jd_col = st.columns([3, 2])
        with upload_col:
            uploaded_files = st.file_uploader("PDF resumes", type=["pdf"], accept_multiple_files=True)
        with jd_col:
            job_description = st.text_area("Job description", value=DEFAULT_JD, height=180)

        config_chips = [f":gray-badge[{settings.tinker_base_model.split('/')[-1]}]"]
        if settings.blind_mode:
            config_chips.append(":violet-badge[blind]")
        if settings.deep_fraud_checks:
            config_chips.append(":blue-badge[deep fraud]")
        weights_total = (settings.jd_fit_weight + settings.verification_weight + settings.authenticity_weight) or 1.0
        config_chips.append(
            f":gray-badge[{settings.jd_fit_weight / weights_total:.0%}/"
            f"{settings.verification_weight / weights_total:.0%}/"
            f"{settings.authenticity_weight / weights_total:.0%}]"
        )
        st.markdown(" ".join(config_chips))

        run = st.button(
            "Analyze resumes",
            type="primary",
            disabled=not uploaded_files or not job_description.strip(),
        )
        if run:
            started_at = datetime.now(timezone.utc).isoformat()
            with st.status("Analyzing resumes...", expanded=True) as status:
                reports = analyze_resumes(
                    uploaded_files,
                    job_description,
                    settings,
                    progress_callback=status.write,
                )
                status.update(label="Analysis complete", state="complete")
            st.session_state["reports"] = reports
            st.session_state["run_meta"] = {
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "job_description": job_description,
                "sample": False,
            }
            st.toast(f"Screened {len(reports)} candidate(s)", icon="✅")
            st.rerun()


def render_landing(settings: AnalysisSettings) -> None:
    ui.render_hero()
    st.write("")
    ui.render_value_props()
    st.write("")
    cta_col, status_col = st.columns([1, 2], vertical_alignment="center")
    with cta_col:
        if st.button("Try sample data", type="primary"):
            st.session_state["reports"] = load_sample_reports()
            st.session_state["run_meta"] = {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "job_description": DEFAULT_JD,
                "sample": True,
            }
            st.toast("Sample screening loaded — three fictional candidates", icon="🧪")
            st.rerun()
    with status_col:
        status = TinkerLLM(settings.tinker_base_model, enabled=settings.use_tinker).status
        suffix = f" ({status.reason})" if status.reason else ""
        st.caption(f"Engine: {status.provider}{suffix}")


def render_results(settings: AnalysisSettings) -> None:
    reports = ui.get_reports()
    run_meta = st.session_state.get("run_meta", {})
    run_settings: AnalysisSettings = run_meta.get("settings", settings)
    blind = bool(run_settings.blind_mode)
    if blind != settings.blind_mode:
        st.caption("Blind-screening change takes effect on the next analysis run.")
    if run_meta.get("sample"):
        st.markdown(":violet-badge[sample data] &nbsp; Fictional candidates for demo purposes.")

    shortlist = ui.get_shortlist()
    labels = [ui.candidate_label(report, blind, index) for index, report in enumerate(reports)]

    # --- KPI strip ---
    total_claims = sum(len(r.verdicts) for r in reports)
    supported_claims = sum(1 for r in reports for v in r.verdicts if v.verdict == "SUPPORTED")
    high_signals = sum(1 for r in reports for s in r.fraud_signals if s.severity == "high")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Candidates", len(reports), border=True)
    kpi_cols[1].metric("Top candidate", labels[0], delta=f"{reports[0].scores.final_score:.1f}", border=True)
    kpi_cols[2].metric("Claims supported", f"{supported_claims}/{total_claims}", border=True)
    kpi_cols[3].metric("High-risk signals", high_signals, border=True)

    # --- Toolbar: heading + run health + export ---
    heading_col, health_col, export_col = st.columns([4, 1, 1], vertical_alignment="center")
    heading_col.subheader("Ranking")
    llm_calls = sum(report.llm_api_calls for report in reports)
    llm_successes = sum(report.llm_api_successes for report in reports)
    llm_errors = [report.llm_error for report in reports if report.llm_error]
    with health_col.popover("Run health"):
        st.markdown(f"**Tinker calls:** {llm_successes}/{llm_calls} succeeded")
        st.markdown(f"**Provider:** {reports[0].llm_provider}")
        if llm_errors:
            st.error(f"Latest error: {llm_errors[-1]}")
        elif llm_calls:
            st.success("All LLM calls succeeded.")
        else:
            st.info("Run used heuristics only (no LLM calls).")
    with export_col.popover("Export"):
        df_export = reports_to_dataframe(reports, blind=blind)
        st.download_button(
            "CSV ranking",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name="grifter_final_ranking.csv",
            mime="text/csv",
        )
        st.download_button(
            "JSON reports",
            data=reports_to_json(reports, blind=blind).encode("utf-8"),
            file_name="grifter_reports.json",
            mime="application/json",
        )
        if run_meta:
            audit = build_audit_log(
                reports, run_settings, run_meta.get("started_at", ""), run_meta.get("finished_at", "")
            )
            st.download_button(
                "Audit log",
                data=audit.encode("utf-8"),
                file_name="grifter_audit_log.json",
                mime="application/json",
            )

    # --- Ranking table (inline status editing) ---
    table = pd.DataFrame(
        {
            "candidate": labels,
            "status": [shortlist.get(ui.shortlist_key(r), "") for r in reports],
            "final": [r.scores.final_score for r in reports],
            "jd_fit": [r.scores.jd_fit_score for r in reports],
            "verification": [r.scores.verification_score for r in reports],
            "authenticity": [r.scores.authenticity_score for r in reports],
            "signals": [len(r.fraud_signals) for r in reports],
            "github": [None if blind else (r.profile.github_url or "") for r in reports],
        }
    )
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=["candidate", "final", "jd_fit", "verification", "authenticity", "signals", "github"],
        column_config={
            "candidate": st.column_config.TextColumn("Candidate", pinned=True),
            "status": st.column_config.SelectboxColumn("Status", options=ui.SHORTLIST_OPTIONS, default=""),
            "final": st.column_config.ProgressColumn("Final", min_value=0, max_value=100, format="%.0f", color="#22B8A8"),
            "jd_fit": st.column_config.NumberColumn("JD fit", format="%.0f"),
            "verification": st.column_config.NumberColumn("Verification", format="%.0f"),
            "authenticity": st.column_config.NumberColumn("Authenticity", format="%.0f"),
            "signals": st.column_config.NumberColumn("Signals"),
            "github": st.column_config.LinkColumn("GitHub", display_text="profile"),
        },
        key="ranking_editor",
    )
    for index, report in enumerate(reports):
        new_status = edited.iloc[index]["status"] or ""
        key = ui.shortlist_key(report)
        if shortlist.get(key, "") != new_status:
            shortlist[key] = new_status
            st.toast(f"{labels[index]}: {new_status or 'status cleared'}")

    # --- Candidate detail ---
    st.subheader("Candidate detail")
    if len(labels) <= 8:
        selected_label = st.pills("Candidate", labels, default=labels[0], label_visibility="collapsed")
    else:
        selected_label = st.selectbox("Candidate", labels)
    if not selected_label:
        return
    selected_index = labels.index(selected_label)
    selected = reports[selected_index]
    selected_key = ui.shortlist_key(selected)

    ui.render_candidate_header(selected, selected_label, blind)

    action_cols = st.columns([2, 3, 2], vertical_alignment="bottom")
    with action_cols[0]:
        current_status = shortlist.get(selected_key, "")
        new_status = st.segmented_control(
            "Decision",
            ["shortlist", "maybe", "reject"],
            default=current_status or None,
            key=f"decision_{selected_key}",
        )
        shortlist[selected_key] = new_status or ""
    with action_cols[1]:
        overrides = ui.get_overrides()
        overrides[selected_key] = st.text_input(
            "Reviewer note (human override)",
            value=overrides.get(selected_key, ""),
            help="Recorded in the audit log. The human decision always wins.",
        )
    with action_cols[2]:
        pdf_display_name = selected_label if blind else None
        pdf_stem = f"candidate_{selected_index + 1}" if blind else selected.profile.source_name.replace(".pdf", "")
        st.download_button(
            "Download PDF report",
            data=build_candidate_pdf(selected, display_name=pdf_display_name),
            file_name=f"grifter_report_{pdf_stem}.pdf",
            mime="application/pdf",
        )

    overview_tab, claims_tab, authenticity_tab, interview_tab = st.tabs(
        ["Overview", "Claims & evidence", "Authenticity", "Interview kit"]
    )
    with overview_tab:
        st.markdown("**Summary**")
        st.write(selected.summary)
        if selected.flags:
            st.markdown(" ".join(f":orange-badge[{flag}]" for flag in selected.flags))
        if not blind:
            fields = {
                "Email": selected.profile.email,
                "GitHub": selected.profile.github_url,
                "CGPA (/10)": selected.profile.cgpa,
                "Claimed experience": (
                    f"{selected.profile.claimed_years_experience:.0f} yrs"
                    if selected.profile.claimed_years_experience
                    else None
                ),
                "Leadership signals": "yes" if selected.profile.leadership else "no",
            }
            field_cols = st.columns(2)
            for i, (label, value) in enumerate([(k, v) for k, v in fields.items() if v is not None]):
                field_cols[i % 2].markdown(
                    f'<p class="gf-field">{label}</p><p class="gf-value">{value}</p>',
                    unsafe_allow_html=True,
                )
        with st.popover("Raw data"):
            st.json(selected.to_dict() if not blind else {"note": "raw data hidden in blind mode"})
    with claims_tab:
        ui.render_claim_cards(selected)
    with authenticity_tab:
        ui.render_fraud_panel(selected)
    with interview_tab:
        ui.render_interview_questions(selected)


settings = sidebar_settings()
ui.render_disclaimer()

has_reports = bool(ui.get_reports())
if not has_reports:
    render_landing(settings)
    st.write("")
analyze_form(settings, collapsed=has_reports)
if has_reports:
    render_results(settings)
