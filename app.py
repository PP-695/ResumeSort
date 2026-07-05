from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from resumesort import AnalysisSettings, analyze_resumes
from resumesort.llm import TinkerLLM
from resumesort.pipeline import build_audit_log, reports_to_dataframe, reports_to_json
from resumesort.report_pdf import build_candidate_pdf
from resumesort import ui


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


def sidebar_settings() -> AnalysisSettings:
    st.sidebar.title("Grifter Filter")
    st.sidebar.caption("Evidence-backed resume ranking")

    tinker_model = st.sidebar.text_input(
        "Tinker base model", value=os.getenv("TINKER_BASE_MODEL", "openai/gpt-oss-20b")
    )
    use_tinker = st.sidebar.toggle("Use Tinker", value=bool(os.getenv("TINKER_API_KEY")))
    github_token = os.getenv("GITHUB_TOKEN") or None
    st.sidebar.write("Tinker key:", "configured" if os.getenv("TINKER_API_KEY") else "missing")
    st.sidebar.write("GitHub token:", "configured" if github_token else "anonymous (shallow mode)")

    st.sidebar.divider()
    blind_mode = st.sidebar.toggle(
        "Blind screening",
        value=False,
        help="Redacts name, email, GitHub handle, CGPA, and education lines from LLM prompts and the ranking view.",
    )
    deep_fraud = st.sidebar.toggle(
        "Deep fraud checks",
        value=bool(github_token),
        disabled=not github_token,
        help="Contributor share, commit timelines, and root-file checks. Requires a GitHub token.",
    )

    st.sidebar.divider()
    max_claims = st.sidebar.slider("Claims per resume", 3, 15, 8)
    max_repos = st.sidebar.slider("GitHub repos to inspect", 5, 30, 10)

    st.sidebar.divider()
    jd_weight = st.sidebar.slider("JD fit weight", 0.0, 1.0, 0.45, 0.05)
    verification_weight = st.sidebar.slider("Verification weight", 0.0, 1.0, 0.35, 0.05)
    authenticity_weight = st.sidebar.slider("Authenticity weight", 0.0, 1.0, 0.20, 0.05)

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


settings = sidebar_settings()

st.title("Grifter Filter")
st.caption("Rank resumes against a JD, verify claims with GitHub evidence, and separate fit from authenticity.")
ui.render_disclaimer()

uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)
job_description = st.text_area("Job description", value=DEFAULT_JD, height=220)
run = st.button("Analyze resumes", type="primary", disabled=not uploaded_files or not job_description.strip())

if run:
    started_at = datetime.now(timezone.utc).isoformat()
    with st.status("Analyzing resumes...", expanded=True) as status:
        def progress(message: str) -> None:
            status.write(message)

        reports = analyze_resumes(uploaded_files, job_description, settings, progress_callback=progress)
        status.update(label="Analysis complete", state="complete")
    st.session_state["reports"] = reports
    st.session_state["run_meta"] = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
    }

reports = ui.get_reports()

if reports:
    blind = settings.blind_mode
    df = reports_to_dataframe(reports)
    if blind:
        df = df.drop(columns=["name", "email", "github", "cgpa"], errors="ignore")
        df.insert(0, "candidate", [ui.candidate_label(r, True, i) for i, r in enumerate(reports)])

    top = reports[0]
    llm_calls = sum(report.llm_api_calls for report in reports)
    llm_successes = sum(report.llm_api_successes for report in reports)
    llm_errors = [report.llm_error for report in reports if report.llm_error]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Top candidate", ui.candidate_label(top, blind, 0))
    col2.metric("Final score", f"{top.scores.final_score:.1f}")
    col3.metric("JD fit", f"{top.scores.jd_fit_score:.1f}")
    col4.metric("Verification", f"{top.scores.verification_score:.1f}")
    col5.metric("Tinker calls", f"{llm_successes}/{llm_calls}")

    if llm_calls and llm_successes == llm_calls:
        st.success(f"Tinker API returned {llm_successes} successful response(s) in this run.")
    elif llm_calls:
        st.warning(f"Tinker API was attempted {llm_calls} time(s), with {llm_successes} success(es).")
    elif settings.use_tinker:
        st.warning("Tinker API was not called. The app fell back before sampling completed.")
    if llm_errors:
        st.error(f"Latest Tinker error: {llm_errors[-1]}")

    st.subheader("Ranking")
    shortlist = ui.get_shortlist()
    ranking_columns = [
        column
        for column in [
            "candidate" if blind else "name",
            "final_score",
            "jd_fit_score",
            "verification_score",
            "authenticity_score",
            "fraud_signals",
            None if blind else "github",
            "flags",
        ]
        if column
    ]
    display_df = df[ranking_columns].copy()
    display_df.insert(1, "status", [shortlist.get(ui.shortlist_key(r), "") for r in reports])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    json_data = reports_to_json(reports).encode("utf-8")
    run_meta = st.session_state.get("run_meta")
    export_col1, export_col2, export_col3 = st.columns(3)
    export_col1.download_button("Download CSV", data=csv_data, file_name="grifter_final_ranking.csv", mime="text/csv")
    export_col2.download_button("Download JSON", data=json_data, file_name="grifter_reports.json", mime="application/json")
    if run_meta:
        audit = build_audit_log(reports, run_meta["settings"], run_meta["started_at"], run_meta["finished_at"])
        export_col3.download_button("Download audit log", data=audit.encode("utf-8"), file_name="grifter_audit_log.json", mime="application/json")

    st.subheader("Candidate detail")
    labels = [ui.candidate_label(report, blind, index) for index, report in enumerate(reports)]
    selected_label = st.selectbox("Candidate", labels)
    selected = reports[labels.index(selected_label)]
    selected_key = ui.shortlist_key(selected)

    action_col1, action_col2 = st.columns([1, 2])
    with action_col1:
        current_status = shortlist.get(selected_key, "")
        new_status = st.radio(
            "Status",
            ui.SHORTLIST_OPTIONS,
            index=ui.SHORTLIST_OPTIONS.index(current_status),
            horizontal=True,
            format_func=lambda value: value or "unset",
        )
        shortlist[selected_key] = new_status
    with action_col2:
        overrides = ui.get_overrides()
        overrides[selected_key] = st.text_input(
            "Reviewer note (human override)",
            value=overrides.get(selected_key, ""),
            help="Recorded in the audit log. The human decision always wins.",
        )

    pdf_bytes = build_candidate_pdf(selected)
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name=f"grifter_report_{selected_key.replace('.pdf', '')}.pdf",
        mime="application/pdf",
    )

    detail_col1, detail_col2 = st.columns([1, 2])
    with detail_col1:
        st.markdown("**Parsed profile**")
        if blind:
            st.json({"skills": selected.profile.skills, "leadership": selected.profile.leadership})
        else:
            st.json(
                {
                    "name": selected.profile.name,
                    "email": selected.profile.email,
                    "github_url": selected.profile.github_url,
                    "cgpa": selected.profile.cgpa,
                    "claimed_years_experience": selected.profile.claimed_years_experience,
                    "skills": selected.profile.skills,
                    "leadership": selected.profile.leadership,
                }
            )
        st.markdown("**Scores**")
        st.json(asdict(selected.scores))
        st.markdown("**LLM status**")
        st.json(
            {
                "provider": selected.llm_provider,
                "api_calls": selected.llm_api_calls,
                "api_successes": selected.llm_api_successes,
                "error": selected.llm_error,
            }
        )
        if selected.flags:
            st.markdown("**Flags**")
            for flag in selected.flags:
                st.warning(flag)

    with detail_col2:
        st.markdown("**Summary**")
        st.write(selected.summary)
        st.markdown("**Authenticity signals**")
        ui.render_fraud_panel(selected)
        st.markdown("**Claim verification**")
        ui.render_verdicts(selected)
        st.markdown("**Suggested interview questions**")
        ui.render_interview_questions(selected)
else:
    status = TinkerLLM(settings.tinker_base_model, enabled=settings.use_tinker).status
    suffix = f" ({status.reason})" if status.reason else ""
    st.info(f"Upload PDF resumes and click Analyze. Tinker status: {status.provider}{suffix}")
