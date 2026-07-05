from __future__ import annotations

import streamlit as st

from resumesort import ui

st.set_page_config(page_title="Methodology - Grifter Filter", page_icon="🔎", layout="wide")

st.title("Methodology")
ui.render_disclaimer()

st.markdown(
    """
## What this tool does

Grifter Filter ranks PDF resumes against a job description and — unlike keyword screeners —
**verifies each factual claim against the candidate's public GitHub activity**, emitting a
FEVER-style verdict per claim:

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | Retrieved GitHub evidence backs the claim, with a citation. |
| `REFUTED` | Evidence contradicts the claim. |
| `NOT_ENOUGH_INFO` | Public evidence can neither prove nor disprove it. **This is the honest default** — most resume claims cannot be verified from public code, and forcing them into verified/not-verified would be lying with numbers. |

## The three scores (kept separate on purpose)

Collapsing fit, truthfulness, and authenticity into one number hides information, so we don't.

**JD-fit** = 35% skill overlap with the JD + 35% semantic similarity of projects/experience to
the JD + 20% CGPA/10 + 10% leadership-if-JD-asks. Measures *relevance*, not truth.

**Verification** = mean over claims of `verdict_weight x confidence`, where SUPPORTED = 1.0,
NOT_ENOUGH_INFO = 0.35, REFUTED = 0.0. Measures *how much of the resume survives contact with
evidence*.

**Authenticity** starts at 1.0 and subtracts penalties from fraud signals:
high severity −0.15 each (capped −0.45), warn −0.07 each (capped −0.21), info −0.02 each
(capped −0.06), plus legacy flags (missing GitHub −0.35, no evidence −0.30, forks −0.15).

**Final score** = weighted mean (default 45% JD-fit / 35% verification / 20% authenticity).
Weights are user-adjustable in the sidebar and recorded in the audit log.

## Fraud signals and thresholds

| Signal | Trigger | Severity |
|---|---|---|
| Low contributor share | Candidate authored <25% of sampled contributions on a non-fork repo (≥5 contributions) | high |
| Commit clustering | ≥80% of a year's commits inside ≤2 weeks on a repo older than 6 months | high |
| Commit clustering (fallback) | All ≥5 sampled recent commits within 2 days on an old repo (used when GitHub's stats API hasn't warmed up) | warn |
| Identity mismatch | >50% of sampled commits on own repos authored by other logins/emails | high |
| Mostly forks | >50% of recently updated repos are forks | warn |
| Star anomaly | ≥40 stars with a watcher/star ratio <0.5% (purchased-star pattern, cf. CMU StarScout) | warn |
| Language not found | Resume claims a programming language absent from every inspected repo | warn |
| Account age gap | Claimed years of experience exceed GitHub account age by >1 year | warn |
| AI-assistance markers | `.claude/`, `.cursor/`, `AGENTS.md`, or AI co-author trailers | info (explicitly *not* fraud) |

Every signal links to the GitHub page where a human can check it. Signals are questions,
not verdicts: private repos, team projects, recreated accounts, and off-GitHub work are all
legitimate explanations, which is why the interview-question generator turns each signal into
a fair, answerable question.

## Known limitations

- Public GitHub is a *partial* view of anyone's work. Absence of evidence ≠ evidence of absence
  — that is exactly what `NOT_ENOUGH_INFO` encodes.
- Contributor share samples the top 10 contributors only; treat it as approximate.
- The heuristic (non-LLM) verdict path is word-overlap similarity and can be fooled by
  keyword-stuffed READMEs; the LLM judge is the stronger path.
- Resume parsing is regex-based and can mis-extract fields from unusual PDF layouts.
- Blind mode redacts direct identifiers, not every possible proxy for demographics.

## Bias stance

A 2024 University of Washington study found LLM resume screeners favored white-associated
names in 85% of trials. Mitigations here: an optional **blind screening mode** (redacts name,
email, GitHub handle, CGPA, and education lines from LLM prompts and the ranking view),
evidence-required verdicts, published formulas on this page, and a hard product rule:
**no auto-reject; a human reviews evidence before any decision**, and the human's note is
recorded in the audit log.

This is a portfolio/demo tool, not a production hiring system. Production use would
additionally require an independent bias audit (NYC Local Law 144), candidate notice/consent,
data-retention controls, and an adverse-impact dashboard (EEOC four-fifths rule).
"""
)
