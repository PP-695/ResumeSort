# Grifter Filter — Product Requirements Document

**Version 0.2 · July 2026 · Status: shipped MVP (P0 + P1)**

---

## 1. Vision

> **Rank resumes by fit — and by whether the claims survive contact with the candidate's actual GitHub.**

Every resume screener scores *relevance*. Almost none score *truth*. Grifter Filter treats a resume as a set of factual claims, retrieves evidence from the candidate's public GitHub, and issues a FEVER-style verdict per claim — `SUPPORTED`, `REFUTED`, or `NOT_ENOUGH_INFO` — with a citation the reviewer can click. Fit, truthfulness, and authenticity are scored separately, because collapsing them into one number hides exactly the information a hiring manager needs.

## 2. The problem (why this matters)

- **44% of US job seekers admit lying during hiring**, and 24% falsified the resume itself (Resume Builder, Jan 2025). Only ~26% of embellishments are ever caught.
- **GitHub portfolios are gameable at scale**: a CMU study ("StarScout", arXiv:2412.13459) found **6M+ fake stars** across 18,600 repos; commit histories are trivially backdated with `GIT_AUTHOR_DATE`; forked tutorials get presented as original work.
- **Recruiters distrust AI screeners for a specific reason**: the #1 documented complaint (2025–26) is *"a high match score with no reasoning, so I read the resume anyway."* An opaque score saves nobody any time.

The opportunity: a screener whose every output is **evidence-linked and honest about uncertainty** is both more useful and more defensible than a black-box score.

## 3. Personas

| Persona | Need | What they get |
|---|---|---|
| **Startup founder / eng hiring manager** (primary) | Screen 100+ applicants without an ATS or a recruiting team | Ranked table, fraud flags, one-click evidence, PDF to forward |
| **Technical recruiter** | Artifacts to hand the interview panel; defensible shortlists | Per-candidate PDF report, targeted interview questions, audit log |
| **Candidate** (fairness stakeholder) | Not being auto-rejected by a black box | NEI-honest verdicts, no auto-reject rule, blind-screening mode, human-override record |

## 4. Product principles

1. **Never bluff.** If public evidence can't prove or disprove a claim, the verdict is `NOT_ENOUGH_INFO` — the honest default, not a failure mode.
2. **Every verdict cites its evidence.** A claim without a clickable source is an opinion, and we don't ship opinions.
3. **Signals are questions, not accusations.** Fraud signals link to the GitHub page where a human can check; the interview-question generator turns each into a fair, answerable prompt.
4. **Decision support, never decisions.** No auto-reject exists in the product. The disclaimer is always visible; the human's override note is recorded.
5. **Degrade gracefully, stay free.** No Tinker key → deterministic heuristics. No GitHub token → shallow mode. The public demo never hard-fails.

## 5. Features

### P0 — core (shipped)
- Multi-PDF resume upload + JD input; regex/heuristic field parsing (name, email, GitHub, CGPA, skills, projects, experience, claimed years).
- GitHub evidence retrieval (repos, languages, commits, contributors, READMEs) with 15-min cache and rate-limit-aware shallow/deep modes.
- LLM claim extraction and claim judging via **Tinker** (OpenAI-compatible endpoint, default `openai/gpt-oss-20b`), with strict evidence-only judging rules and full heuristic fallback.
- Three separated scores (JD-fit / Verification / Authenticity) + weighted final; user-adjustable weights.
- Ranked table, CSV/JSON export, per-candidate detail with per-claim evidence expanders.

### P1 — differentiators (shipped)
- **Fraud-signal engine (7 signals)** — see catalog below.
- **Evidence deep-links** on every verdict and signal (`LinkColumn`, `link_button`, URLs in PDF).
- **Targeted interview questions** generated from REFUTED/NEI claims and fraud signals — converts distrust into a concrete next step.
- **Per-candidate PDF report** (fpdf2) — the artifact that actually gets forwarded to panels.
- **Side-by-side comparison page** (2–5 candidates, defaults to top 3).
- **Shortlist/status tagging** (shortlist / maybe / reject) + reviewer override note, persisted per session and recorded in the audit log.
- **Blind-screening mode** — redacts name/email/GitHub handle/CGPA/education lines from LLM prompts and the ranking view.
- **Methodology page** — every formula and threshold published in-app.
- **Audit-log JSON export** — model, weights, verdicts, signals, timestamps, disclaimer.

### P2 — documented future (not built)
ATS write-back (Greenhouse/Lever/iCIMS), reference-check cross-validation, private-repo OAuth reconciliation, MOSS-style code-plagiarism detection, coder-percentile benchmarking, labeled eval benchmark with published precision/recall, independent bias audit + four-fifths dashboard, notice/consent + retention/erasure workflows.

## 6. Fraud-signal catalog

| Signal | Data source | Trigger | Severity |
|---|---|---|---|
| Low contributor share | `/repos/{r}/contributors` | Candidate <25% of contributions on own repo (≥5 total) | high |
| Commit clustering | `/repos/{r}/stats/commit_activity` | ≥80% of a year's commits in ≤2 weeks, repo >6 months old | high |
| Commit clustering (fallback) | sampled commit dates | ≥5 recent commits within 2 days on an old repo | warn |
| Identity mismatch | commit author fields | >50% of sampled commits by other logins/emails | high |
| Mostly forks | repo list | >50% of recent repos are forks | warn |
| Star anomaly | stars vs subscribers | ≥40 stars, watcher/star <0.5% (StarScout pattern) | warn |
| Language not found | `/repos/{r}/languages` | Claimed language, zero bytes anywhere | warn |
| Account age gap | user `created_at` | Claimed experience > account age + 1yr | warn |
| AI-assistance markers | root files, commit trailers | `.claude/`, `.cursor/`, AI co-author trailers | info — *explicitly not fraud* |

## 7. Success metrics

- **100% of non-NEI verdicts carry a clickable evidence link** (product invariant).
- Full 5-resume analysis completes in **< 3 minutes** on the free tier.
- LLM cost **< $0.01 per candidate** (measured: ~9 calls ≈ $0.002 on gpt-oss-20b).
- Demo credibility test: a planted fake claim ("1M orders/sec Rust engine" on a Python-only profile) is REFUTED or flagged — verified in E2E.
- Zero hard failures with no API keys configured (fallback invariant).

## 8. Compliance posture

This is a **portfolio/demo tool**, deliberately built inside the guardrails that real deployments require:

- **No automated rejection** — GDPR Art. 22's "solely automated decision" line is never crossed; a human override field exists and is logged.
- **Transparency** — scoring formulas and thresholds are published in-app (Methodology page), the direction NYC Local Law 144 and the EU AI Act (Annex III classifies CV-screening as high-risk) push toward.
- **Bias mitigation** — blind-screening mode responds directly to the 2024 UW study (LLM screeners favored white-associated names in 85% of trials); evidence-required verdicts limit name-driven halo effects.
- **Accountability** — per-run audit log (model, weights, verdicts, overrides, timestamps).

**Before any production hiring use**, an operator would additionally need: an independent bias audit with published results (NYC LL144), candidate notice/consent (Illinois AIVIA / NYC), an adverse-impact dashboard (EEOC four-fifths), data-retention and erasure controls (GDPR), and a validation study (EEOC business-necessity defense). These are documented, not built.

## 9. Rollout

| Phase | Contents | Status |
|---|---|---|
| 0 | Security fix, correct Tinker integration (OAI-compatible endpoint), dependency slim-down | ✅ |
| 1 | PRD + Architecture docs, HF Spaces deployment | ✅ (docs) / deploy step |
| 2 | Fraud engine, evidence deep-links, top-K judging | ✅ |
| 3 | Interview questions, PDF reports, compare view, shortlist | ✅ |
| 4 | Blind mode, methodology page, audit log, disclaimers | ✅ |
| 5 (next) | Labeled eval benchmark (~100 claims) with published precision/recall; sample-resume gallery | — |
