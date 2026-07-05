"""Per-candidate PDF report (fpdf2, pure Python).

v1 uses the built-in latin-1 core fonts, so non-latin characters degrade to '?'.
Bundling a Unicode TTF (e.g. DejaVuSans) is the documented upgrade path.
"""

from __future__ import annotations

from fpdf import FPDF

from .schemas import CandidateReport

DISCLAIMER = "Decision support only - not a hiring decision. A human must review the evidence."

SEVERITY_LABEL = {"high": "HIGH", "warn": "WARN", "info": "INFO"}


class _ReportPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, _latin1(DISCLAIMER), align="C")


def build_candidate_pdf(report: CandidateReport, job_title: str = "") -> bytes:
    pdf = _ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    name = report.profile.name or report.profile.source_name or "Candidate"
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 9, _latin1(f"Grifter Filter Report - {name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(90, 90, 90)
    subtitle = job_title or "Evidence-verified resume screening"
    pdf.cell(0, 6, _latin1(subtitle), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    _section(pdf, "Scores")
    scores = report.scores
    rows = [
        ("Final score", f"{scores.final_score:.1f} / 100"),
        ("JD fit", f"{scores.jd_fit_score:.1f}"),
        ("Verification", f"{scores.verification_score:.1f}"),
        ("Authenticity", f"{scores.authenticity_score:.1f}"),
    ]
    pdf.set_font("Helvetica", "", 10)
    for label, value in rows:
        pdf.cell(60, 6, _latin1(label))
        pdf.cell(0, 6, _latin1(value), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    _section(pdf, "Summary")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, new_x="LMARGIN", new_y="NEXT", text=_latin1(report.summary or "(no summary)"))
    pdf.ln(2)

    _section(pdf, "Claim verification")
    for verdict in report.verdicts:
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(0, 5, new_x="LMARGIN", new_y="NEXT", text=_latin1(f"[{verdict.verdict}] {verdict.claim}"))
        pdf.set_font("Helvetica", "", 8)
        if verdict.explanation:
            pdf.multi_cell(0, 4.5, new_x="LMARGIN", new_y="NEXT", text=_latin1(f"    {verdict.explanation}"))
        if verdict.evidence_source:
            pdf.set_text_color(30, 80, 160)
            pdf.multi_cell(0, 4.5, new_x="LMARGIN", new_y="NEXT", text=_latin1(f"    Evidence: {verdict.evidence_source}"))
            pdf.set_text_color(0, 0, 0)
        pdf.ln(1)
    pdf.ln(1)

    if report.fraud_signals:
        _section(pdf, "Authenticity signals")
        for signal in report.fraud_signals:
            pdf.set_font("Helvetica", "B", 9)
            pdf.multi_cell(0, 5, new_x="LMARGIN", new_y="NEXT", text=_latin1(f"[{SEVERITY_LABEL.get(signal.severity, '?')}] {signal.title}"))
            pdf.set_font("Helvetica", "", 8)
            pdf.multi_cell(0, 4.5, new_x="LMARGIN", new_y="NEXT", text=_latin1(f"    {signal.detail}"))
            if signal.evidence_url:
                pdf.set_text_color(30, 80, 160)
                pdf.multi_cell(0, 4.5, new_x="LMARGIN", new_y="NEXT", text=_latin1(f"    {signal.evidence_url}"))
                pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        pdf.ln(1)

    if report.interview_questions:
        _section(pdf, "Suggested interview questions")
        pdf.set_font("Helvetica", "", 9)
        for idx, item in enumerate(report.interview_questions, start=1):
            pdf.multi_cell(0, 5, new_x="LMARGIN", new_y="NEXT", text=_latin1(f"{idx}. {item.get('question', '')}"))
            if item.get("listen_for"):
                pdf.set_font("Helvetica", "I", 8)
                pdf.multi_cell(0, 4.5, new_x="LMARGIN", new_y="NEXT", text=_latin1(f"    Listen for: {item['listen_for']}"))
                pdf.set_font("Helvetica", "", 9)
            pdf.ln(0.5)

    return bytes(pdf.output())


def _section(pdf: FPDF, title: str) -> None:
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 7, _latin1(title), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(1)


def _latin1(text: str) -> str:
    return (text or "").encode("latin-1", "replace").decode("latin-1")
