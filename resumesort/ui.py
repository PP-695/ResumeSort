"""Shared Streamlit rendering helpers and session-state accessors.

Design system: dark blue-charcoal surfaces, verification-teal accent, semantic
badge colors (green=SUPPORTED, yellow=NEI, red=REFUTED/high, orange=warn,
blue=info, gray=neutral chips). Score magnitudes render as single-hue HTML
meter bars (.gf-bar) — identity comes from labels, not rainbow colors.
"""

from __future__ import annotations

import streamlit as st

from .llm import shorten
from .privacy import blind_display_name
from .schemas import CandidateReport, ClaimVerdict

SHORTLIST_OPTIONS = ["", "shortlist", "maybe", "reject"]

STATUS_BADGE = {
    "shortlist": ":green-badge[shortlist]",
    "maybe": ":yellow-badge[maybe]",
    "reject": ":red-badge[reject]",
}

VERDICT_BADGE = {
    "SUPPORTED": ":green-badge[SUPPORTED]",
    "REFUTED": ":red-badge[REFUTED]",
    "NOT_ENOUGH_INFO": ":yellow-badge[NOT ENOUGH INFO]",
}

SEVERITY_BADGE = {
    "high": ":red-badge[HIGH]",
    "warn": ":orange-badge[WARN]",
    "info": ":blue-badge[INFO]",
}

DISCLAIMER = (
    "**Decision support, not a hiring decision.** Verdicts and scores point a human "
    "reviewer at evidence; they must never auto-reject a candidate."
)

_CSS = """<style>
  .block-container { padding-top: 2.2rem; }
  .gf-kicker { font-size:.72rem; font-weight:600; letter-spacing:.14em;
               text-transform:uppercase; color:#5AC8BA; margin-bottom:.25rem; }
  .gf-hero-h { font-size:2.6rem; font-weight:700; line-height:1.15; margin:0 0 .4rem 0; }
  .gf-hero-sub { color:#9FB0B5; font-size:1.05rem; max-width:46rem; }
  .gf-num { font-variant-numeric: tabular-nums; }
  .gf-bar { display:flex; align-items:center; gap:.6rem; margin:.3rem 0; }
  .gf-bar-label { flex:0 0 6.5rem; font-size:.8rem; color:#9FB0B5; }
  .gf-bar-track { flex:1; height:6px; background:#263135; border-radius:3px; }
  .gf-bar-fill  { height:6px; border-radius:3px; background:#22B8A8; }
  .gf-bar-val   { flex:0 0 2.4rem; text-align:right; font-size:.85rem;
                  font-weight:600; font-variant-numeric:tabular-nums; }
  .gf-bar-best .gf-bar-val { color:#0CA30C; }
  .gf-field { color:#9FB0B5; font-size:.8rem; margin-bottom:0; }
  .gf-value { font-size:.95rem; margin-bottom:.6rem; }
</style>"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


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
    """Stable per-candidate key; email disambiguates same-named upload files."""
    parts = [
        report.profile.source_name or "unknown",
        report.profile.email or report.profile.name or "",
    ]
    return "|".join(parts)


def verdict_chips(verdicts: list[ClaimVerdict]) -> str:
    supported = sum(1 for v in verdicts if v.verdict == "SUPPORTED")
    nei = sum(1 for v in verdicts if v.verdict == "NOT_ENOUGH_INFO")
    refuted = sum(1 for v in verdicts if v.verdict == "REFUTED")
    chips = []
    if supported:
        chips.append(f":green-badge[{supported} supported]")
    if nei:
        chips.append(f":yellow-badge[{nei} not enough info]")
    if refuted:
        chips.append(f":red-badge[{refuted} refuted]")
    return " ".join(chips) or ":gray-badge[no claims]"


def fraud_chips(report: CandidateReport) -> str:
    high = sum(1 for s in report.fraud_signals if s.severity == "high")
    warn = sum(1 for s in report.fraud_signals if s.severity == "warn")
    info = sum(1 for s in report.fraud_signals if s.severity == "info")
    chips = []
    if high:
        chips.append(f":red-badge[{high} high]")
    if warn:
        chips.append(f":orange-badge[{warn} warn]")
    if info:
        chips.append(f":blue-badge[{info} info]")
    return " ".join(chips) or ":green-badge[clear]"


def skill_chips(skills: list[str], limit: int = 10) -> str:
    shown = [f":gray-badge[{skill}]" for skill in skills[:limit]]
    extra = len(skills) - limit
    if extra > 0:
        shown.append(f":gray-badge[+{extra} more]")
    return " ".join(shown) if shown else ":gray-badge[no skills parsed]"


def render_score_bars(scores, best: dict[str, float] | None = None) -> None:
    """Three single-hue HTML meter bars; `best` green-bolds per-metric winners."""
    rows = [
        ("JD fit", scores.jd_fit_score, "jd_fit_score"),
        ("Verification", scores.verification_score, "verification_score"),
        ("Authenticity", scores.authenticity_score, "authenticity_score"),
    ]
    html_parts = []
    for label, value, key in rows:
        best_class = " gf-bar-best" if best and best.get(key) == value else ""
        html_parts.append(
            f'<div class="gf-bar{best_class}">'
            f'<span class="gf-bar-label">{label}</span>'
            f'<span class="gf-bar-track"><span class="gf-bar-fill" style="width:{min(100, max(0, value)):.0f}%; display:block;"></span></span>'
            f'<span class="gf-bar-val">{value:.0f}</span>'
            f"</div>"
        )
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_candidate_header(report: CandidateReport, label: str, blind: bool) -> None:
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            status = get_shortlist().get(shortlist_key(report), "")
            title = f"### {label}"
            if status:
                title += f" &nbsp; {STATUS_BADGE.get(status, '')}"
            st.markdown(title)
            if not blind and report.profile.github_url:
                st.markdown(f"[{report.profile.github_url}]({report.profile.github_url})")
            st.markdown(skill_chips(report.profile.skills))
        with right:
            st.metric("Final score", f"{report.scores.final_score:.1f}", border=False)
            render_score_bars(report.scores)


def render_claim_cards(report: CandidateReport) -> None:
    st.markdown(verdict_chips(report.verdicts))
    verdict_filter = st.pills(
        "Filter",
        ["All", "SUPPORTED", "NOT_ENOUGH_INFO", "REFUTED"],
        default="All",
        label_visibility="collapsed",
        key=f"verdict_filter_{shortlist_key(report)}",
    )
    order = {"REFUTED": 0, "NOT_ENOUGH_INFO": 1, "SUPPORTED": 2}
    verdicts = sorted(report.verdicts, key=lambda v: order.get(v.verdict, 3))
    if verdict_filter and verdict_filter != "All":
        verdicts = [v for v in verdicts if v.verdict == verdict_filter]
    if not verdicts:
        st.caption("No claims match this filter.")
        return
    for verdict in verdicts:
        with st.container(border=True):
            st.markdown(
                f"{VERDICT_BADGE.get(verdict.verdict, '')} "
                f"<span style='color:#9FB0B5; font-size:.8rem;'>confidence {verdict.confidence:.2f}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{verdict.claim}**")
            if verdict.explanation:
                st.caption(verdict.explanation)
            if verdict.evidence or verdict.evidence_source:
                with st.expander("Evidence"):
                    if verdict.evidence:
                        st.write(verdict.evidence)
                    if verdict.evidence_source:
                        st.link_button("Open on GitHub", verdict.evidence_source)


def render_fraud_panel(report: CandidateReport) -> None:
    if not report.fraud_signals:
        with st.container(border=True):
            st.markdown(":green-badge[Clear] &nbsp; No fraud signals detected in the inspected repositories.")
        return
    order = {"high": 0, "warn": 1, "info": 2}
    for signal in sorted(report.fraud_signals, key=lambda s: order.get(s.severity, 3)):
        with st.container(border=True):
            st.markdown(f"{SEVERITY_BADGE.get(signal.severity, '')} &nbsp; **{signal.title}**")
            st.caption(signal.detail)
            if signal.evidence_url:
                st.link_button("View evidence on GitHub", signal.evidence_url)


def render_interview_questions(report: CandidateReport) -> None:
    if not report.interview_questions:
        st.caption("No targeted interview questions (all claims verified, no signals).")
        return
    for idx, item in enumerate(report.interview_questions, start=1):
        with st.container(border=True):
            st.markdown(f"**{idx}. {item.get('question', '')}**")
            if item.get("listen_for"):
                st.caption(f"Listen for: {item['listen_for']}")
            if item.get("targets"):
                st.markdown(f":gray-badge[{shorten(item['targets'], 60)}]")
    all_questions = "\n\n".join(
        f"{idx}. {item.get('question', '')}" for idx, item in enumerate(report.interview_questions, start=1)
    )
    st.code(all_questions, language=None)


def render_hero() -> None:
    st.markdown(
        '<div class="gf-kicker">Grifter Filter</div>'
        '<div class="gf-hero-h">Screen resumes on evidence, not vibes.</div>'
        '<div class="gf-hero-sub">Upload resumes and a job description. Every claim gets a verdict '
        "backed by the candidate's actual GitHub — with fraud signals, fair interview questions, "
        "and honest <em>not-enough-info</em> answers instead of bluffing.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        ":green-badge[✓ claim verification] :primary-badge[GitHub evidence] :gray-badge[human-in-the-loop]"
    )


def render_value_props() -> None:
    cols = st.columns(3)
    props = [
        ("Rank against the JD", "Three separated scores — fit, verification, authenticity — never one opaque number."),
        ("Verify every claim", "FEVER-style verdicts with clickable GitHub evidence for each resume claim."),
        ("Surface fraud signals", "Commit backdating, contributor share, fake-star patterns — as questions, not accusations."),
    ]
    for col, (title, caption) in zip(cols, props):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(caption)
