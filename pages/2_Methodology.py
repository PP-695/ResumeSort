from __future__ import annotations

import streamlit as st

from resumesort import ui

st.set_page_config(page_title="Methodology - Grifter Filter", page_icon="🔎", layout="wide")
ui.inject_css()

st.title("Methodology")
ui.render_disclaimer()

overview_tab, scores_tab, fraud_tab, limits_tab = st.tabs(
    ["Overview", "The three scores", "Fraud signals", "Limitations & bias"]
)

with overview_tab:
    st.markdown(
        """
Grifter Filter ranks PDF resumes against a job description and — unlike keyword screeners —
**verifies each factual claim against the candidate's public GitHub activity**, emitting a
FEVER-style verdict per claim:
"""
    )
    st.markdown(
        """
| Verdict | Meaning |
|---|---|
| :green-badge[SUPPORTED] | Retrieved GitHub evidence backs the claim, with a citation. |
| :red-badge[REFUTED] | Evidence **contradicts** the claim. Absence of evidence is never refutation. |
| :yellow-badge[NOT ENOUGH INFO] | Public evidence can neither prove nor disprove it. **This is the honest default** — most resume claims cannot be verified from public code, and forcing them into verified/not-verified would be lying with numbers. |
"""
    )
    cols = st.columns(3)
    cards = [
        ("JD fit — 45%", "Relevance of skills and projects to the role. Measures *fit*, not truth."),
        ("Verification — 35%", "How much of the resume survives contact with evidence."),
        ("Authenticity — 20%", "Fraud-signal penalties: is the footprint genuinely theirs?"),
    ]
    for col, (title, caption) in zip(cols, cards):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(caption)
    st.caption("Default weights shown; every weight is user-adjustable and recorded in the audit log.")

with scores_tab:
    st.markdown(
        """
**JD-fit** = 35% skill overlap with the JD + 35% semantic similarity of projects/experience to
the JD + 20% CGPA/10 + 10% leadership-if-JD-asks. Skills are filtered against a technology
vocabulary and the JD itself, so section labels and soft-skill phrases don't dilute the score.
CGPAs on the US 4.0 scale are normalized to /10.

**Verification** = mean over claims of a per-verdict contribution:
:green-badge[SUPPORTED] contributes its confidence (LLM-judged; keyword-level matches are
capped at 0.65 and labeled as such) · :yellow-badge[NOT ENOUGH INFO] contributes a flat 0.35 —
confidence in "we can't verify" deliberately does **not** raise the score ·
:red-badge[REFUTED] contributes 0.

**Authenticity** starts at 1.0 and subtracts fraud-signal penalties:
:red-badge[high] −0.15 each (capped −0.45) · :orange-badge[warn] −0.07 each (capped −0.21) ·
:blue-badge[info] −0.02 each (capped −0.06), plus legacy flags (missing GitHub −0.35,
no evidence −0.30, forks −0.15).

**Final score** = weighted mean of the three (weights normalized; defaults 45/35/20).

**Verification flow**: claims are extracted by the LLM (atomic, checkable), matched against
evidence heuristically first; NOT-ENOUGH-INFO cases go to the LLM judge (top-4 most relevant
evidence items, capped at 6 judgments per candidate), and leftover judge budget double-checks
keyword-supported claims. The judge may only cite evidence by index — links are resolved
locally, so a hallucinated URL cannot appear in a report.
"""
    )

with fraud_tab:
    st.markdown(
        """
| Signal | Trigger | Severity |
|---|---|---|
| Low contributor share | Candidate authored <25% of sampled contributions on a non-fork repo (≥5 contributions) | :red-badge[high] |
| Commit clustering | ≥80% of a year's commits inside ≤2 weeks on a repo older than 6 months | :red-badge[high] |
| Commit clustering (fallback) | All sampled commits within 2 days **and** the sample is the repo's whole history (used when GitHub's stats API hasn't warmed up) | :orange-badge[warn] |
| Identity mismatch | >50% of sampled commits on own repos authored by other logins/emails | :red-badge[high] |
| Mostly forks | >50% of recently updated repos are forks | :orange-badge[warn] |
| Star anomaly | ≥40 stars with a subscriber/star ratio <0.5% (purchased-star pattern, cf. CMU StarScout). Deep mode only — shallow mode lacks true subscriber counts | :orange-badge[warn] |
| Language not found | Resume claims a programming language absent from every inspected repo | :orange-badge[warn] |
| Account age gap | Claimed years of experience exceed GitHub account age by >1 year | :orange-badge[warn] |
| AI-assistance markers | `.claude/`, `.cursor/`, `AGENTS.md`, or AI co-author trailers | :blue-badge[info] — *explicitly not fraud* |

Every signal links to the GitHub page where a human can check it. Signals are **questions,
not verdicts**: private repos, team projects, recreated accounts, and off-GitHub work are all
legitimate explanations — which is why the interview-question generator turns each signal into
a fair, answerable question. Non-fork repositories get priority in the inspection budget so
forks can't crowd out a candidate's own work.
"""
    )

with limits_tab:
    st.markdown(
        """
## Known limitations

- Public GitHub is a *partial* view of anyone's work. Absence of evidence ≠ evidence of absence
  — that is exactly what :yellow-badge[NOT ENOUGH INFO] encodes.
- Contributor share samples the top 10 contributors only; treat it as approximate.
- The heuristic (non-LLM) verdict path is word-overlap similarity: it can be fooled by
  keyword-stuffed READMEs, which is why its confidence is capped and labeled, and why leftover
  LLM budget re-checks those verdicts.
- Resume parsing is regex-based and can mis-extract fields from unusual PDF layouts
  (hyperlink-only GitHub links are handled via PDF link annotations).
- Blind mode redacts direct identifiers, not every possible proxy for demographics.

## Bias stance

A 2024 University of Washington study found LLM resume screeners favored white-associated
names in 85% of trials. Mitigations here: an optional **blind screening mode** (redacts name,
email, GitHub handle, CGPA, and education lines from LLM prompts, all views, and all exports),
evidence-required verdicts, published formulas on this page, and a hard product rule:
**no auto-reject; a human reviews evidence before any decision**, and the human's note is
recorded in the audit log.

This is a portfolio/demo tool, not a production hiring system. Production use would
additionally require an independent bias audit (NYC Local Law 144), candidate notice/consent,
data-retention controls, and an adverse-impact dashboard (EEOC four-fifths rule).
"""
    )
