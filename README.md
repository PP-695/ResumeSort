---
title: Grifter Filter
emoji: 🔎
colorFrom: gray
colorTo: red
sdk: streamlit
sdk_version: "1.58.0"
app_file: app.py
pinned: false
license: mit
---

# Grifter Filter — ResumeSort

**Rank resumes by fit — and by whether the claims survive contact with the candidate's actual GitHub.**

Grifter Filter is a Streamlit app that ranks PDF resumes against a job description, verifies each factual claim against the candidate's public GitHub with a FEVER-style verdict (`SUPPORTED` / `REFUTED` / `NOT_ENOUGH_INFO`), runs a 7-signal fraud analysis (commit backdating, contributor share, fake-star patterns, stack mismatches…), and produces recruiter-ready artifacts: ranked CSV, per-candidate PDF reports, targeted interview questions, and an audit log.

- 📄 **[PRD.md](PRD.md)** — product vision, personas, feature priorities, compliance posture
- 🏗️ **[ARCHITECTURE.md](ARCHITECTURE.md)** — component design, Tinker integration, fallback matrix
- ⚖️ **Methodology page (in-app)** — every scoring formula and fraud threshold, published

## Highlights

- **Honest verdicts.** Most resume claims can't be proven from public code — the system says `NOT_ENOUGH_INFO` instead of pretending. `REFUTED` requires *contradicting* evidence, never mere absence.
- **Every verdict cites clickable evidence** (repo, README, commit page). No black-box scores.
- **Three separated scores** — JD-fit, Verification, Authenticity — with user-adjustable weights.
- **Fraud signals as questions, not accusations**, each deep-linked to the GitHub page where a human can check.
- **Targeted interview questions** generated from unverified claims and signals.
- **Blind-screening mode** (redacts name/email/CGPA/education from LLM prompts and ranking), no auto-reject by design, per-run audit-log export.
- **LLM powered by [Tinker](https://thinkingmachines.ai/tinker/)** (default `openai/gpt-oss-20b` via the OpenAI-compatible endpoint, ~$0.003/candidate) with a full deterministic fallback — the app works with zero API keys configured.

## Setup (local)

```powershell
cd ResumeSort
py -3.12 -m venv .venv312
.\.venv312\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env   # then edit .env
```

`.env`:

```text
TINKER_API_KEY=your-tinker-api-key        # optional — heuristic fallback without it
TINKER_BASE_MODEL=openai/gpt-oss-20b
GITHUB_TOKEN=your-github-token            # optional — enables deep fraud checks (public_repo read scope)
```

Verify Tinker connectivity (also prints the live model catalog):

```powershell
.\.venv312\Scripts\python scripts\tinker_smoke.py
```

## Run

```powershell
.\.venv312\Scripts\streamlit run app.py
```

Upload PDF resumes, paste a JD, click **Analyze resumes**. Then explore the ranking, per-candidate detail (verdicts → evidence → fraud signals → interview questions), the **Compare** page, and the **Methodology** page.

## Deploy (Hugging Face Spaces, free)

1. Create a Space → SDK **Streamlit**, hardware **CPU basic (free)**.
2. Space **Settings → Variables and secrets**: add `TINKER_API_KEY` and `GITHUB_TOKEN`.
3. Push this repo to the Space:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/grifter-filter
   git push hf main --force   # first push only (HF creates a stub commit)
   ```
Free Spaces sleep after ~48 h idle; the first visitor waits ~30–60 s for wake-up.

## Project layout

```text
app.py                          Streamlit main page (analyze / rank / detail)
pages/1_Compare.py              Side-by-side candidate comparison
pages/2_Methodology.py          Published formulas, thresholds, limitations
resumesort/parser.py            PDF + regex field extraction
resumesort/github_evidence.py   GitHub snapshot fetching + TTL cache
resumesort/fraud.py             7-signal fraud analysis
resumesort/llm.py               Tinker client (claims, judging, summary, questions)
resumesort/verification.py      Heuristic-first claim verification
resumesort/scoring.py           JD-fit / verification / authenticity formulas
resumesort/privacy.py           Blind-screening redaction
resumesort/report_pdf.py        Per-candidate PDF report (fpdf2)
resumesort/pipeline.py          Orchestration, exports, audit log
resumesort/ui.py                Shared Streamlit helpers
scripts/tinker_smoke.py         Tinker connectivity + model-catalog smoke test
tests/                          29 unit/integration tests (pytest)
```

## Tests

```powershell
.\.venv312\Scripts\python -m pip install -r requirements-dev.txt
.\.venv312\Scripts\python -m pytest -q
```

## Honest limitations

This is a portfolio/demo MVP, **not** a production hiring system. Public GitHub is a partial view of anyone's work; parsing is regex-based; the non-LLM verdict path is a word-overlap heuristic; blind mode removes direct identifiers, not every demographic proxy. Production use would require an independent bias audit, candidate notice/consent, retention controls, and a validation study — see [PRD.md §8](PRD.md). The original Colab-era notebook is preserved in `code.ipynb` / `README_legacy.md` (untracked) for history.
