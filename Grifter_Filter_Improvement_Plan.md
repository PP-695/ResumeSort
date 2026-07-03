# Grifter Filter (Resume Claim Verifier) — Improvement Plan v2

*Goal: turn your T5/BART/RAG resume-claim verifier into a rigorous, fraud-catching, explainable system. Budget assumption: ~$150 in API credits — so the LLM is used surgically, not for everything.*

> **Note:** I haven't seen the repo yet (couldn't find it via search). This plan is built from your description. Once you share the GitHub link (or paste `requirements.txt`, the main pipeline file, and a sample report), I'll mark up the **actual code** with concrete diffs.

---

## What you have today (as described)

1. Preprocess resume → extract structured fields (Name, LinkedIn, GitHub, GPA, College, Experience).
2. Identify skill/project claims → scrape GitHub repos + READMEs (BeautifulSoup).
3. Rank with NLP: **T5** for summary scoring, **RAG/BART** for project-claim verification.
4. Output: report of each Claim → Verification result → Score.

Solid skeleton. The weaknesses are all in *how verification is framed*, *how evidence is gathered*, and *how fraud is caught* — which is exactly where the value is.

---

## The 7 highest-leverage improvements

### 1. Reframe verification as FEVER-style fact-checking (biggest win)

Right now a generative model (BART/RAG) "verifies" claims — but generative models *narrate*, they don't *adjudicate*, and they can't say "I don't know." The academic-standard framing for exactly your problem is **FEVER (Fact Extraction and VERification)**: a 3-stage pipeline that outputs one of three labels with cited evidence.

```
Claim → [1] Evidence Retrieval → [2] Sentence/Span Selection → [3] NLI verdict
                                                         │
                              SUPPORTED  /  REFUTED  /  NOT ENOUGH INFO
```

The **NOT ENOUGH INFO (NEI)** label is the single most important upgrade. Most resume claims ("Built a scalable pipeline handling 1M records") can't be proven *or* disproven from a GitHub repo — they're NEI. Today your system almost certainly forces these into "verified" or "not," which is the core accuracy leak. Add NEI and your reports become honest and far more useful to recruiters.

**Implementation:**
- **Claim decomposition:** use the LLM to split each resume bullet into *atomic, checkable* sub-claims ("uses Python", "deployed on AWS", "achieved 67% latency reduction"). Verify each atom separately.
- **NLI verdict:** classify each atom with a dedicated entailment model — **DeBERTa-v3 fine-tuned on MNLI/ANLI** is cheap, runs locally/free, and is purpose-built for SUPPORTED/REFUTED/NEI. Reserve the paid LLM for only the hard/low-confidence atoms.
- **Always cite the evidence span** (the commit, file, README line) behind every verdict.

### 2. Replace BeautifulSoup with the GitHub API + real code analysis

BeautifulSoup on GitHub is fragile (JS-rendered pages, rate limits) and README-only evidence is shallow. Switch to the **official GitHub REST/GraphQL API** (auth token → 5,000 req/hr) and analyze the *actual code*, not just the README:

- **Languages breakdown** + **dependency manifests** (`requirements.txt`, `package.json`, `pom.xml`) to verify claimed tech stack against what the repo actually uses.
- **Commit history & timeline.**
- **Code embeddings** (CodeBERT / GraphCodeBERT / StarCoder) to semantically map code → claimed skills, instead of keyword presence. "Knows PyTorch" should be backed by `import torch` and real model code, not the word "PyTorch" in a README.
- **Deployed-URL check:** if they claim a live product, ping the URL.

### 3. Add fraud detection — this is what makes it a "Grifter Filter"

A verifier that only checks "is the keyword present" is gameable. The defensible product *catches lying*. Add an **Authenticity Score** per project, separate from skill-match:

- **Forked-tutorial detection:** is the repo a fork? Research convention treats forks with <100 stars as not original work. Flag "this is a fork of a popular tutorial."
- **Authorship verification:** contributor commit-share for *this candidate* (GitHub API), GPG-signed commits, `Co-authored-by` trailers. Did they write it, or just push someone else's code?
- **Commit-timeline anomalies:** all commits in one day / one bulk push = likely dumped, not built over time.
- **README/code plagiarism:** embed and compare against known tutorials and other repos.
- **Inflated-metric flag:** claim of "1M users / 67% improvement" on a 2-star repo with no deploy and no benchmark → REFUTED-or-NEI + "unverifiable metric."
- **Cross-source consistency:** resume vs LinkedIn vs GitHub (dead links, mismatched dates, nonexistent profiles).

Output per claim: `Verdict (S/R/NEI) · Evidence link · Authenticity score · Flags`.

### 4. If you use an LLM judge, neutralize its known biases

LLM-as-judge research documents specific failure modes that would wreck a verifier:

- **Authority bias** — LLMs rate claims higher just because they *contain* a citation, even a fabricated one. Critical for you: **judge only against retrieved evidence, never let the candidate's own claim text count as its own proof.**
- **Verbosity bias** — longer claims score higher. Mitigate with **structured form-filling output** (`verdict: …, evidence_span: …, confidence: …`) rather than free-text.
- **Position/self-enhancement bias** — when ranking candidates, **swap order and average** (position-swap calibration); shuffle.
- **Low confidence → human-in-the-loop**, don't auto-decide.
- Use **self-consistency** (sample a few times) on hard atoms.

### 5. Separate the scores (and de-risk hiring bias)

Collapsing everything into one "Score" hides information and invites legal/fairness problems in hiring. Emit three:

- **JD-fit score** (semantic match of evidenced skills ↔ job requirements).
- **Verification score** (% of claims SUPPORTED with evidence).
- **Authenticity score** (fraud signals).

Then make ranking **explainable** (why this rank, which evidence) and run a **bias audit** — don't let name/college/GPA act as hidden proxies; measure adverse impact. This is both an ethics and a "won't get the product banned" issue.

### 6. Build a real evaluation benchmark ("high accuracy" isn't measurable yet)

Create a labeled test set so you can prove improvement:

- ~100–200 resume claims with ground-truth labels (SUPPORTED / REFUTED / NEI) + evidence.
- An **adversarial subset**: forked tutorials presented as original, inflated metrics, plagiarized READMEs, dead links.
- Metrics: **precision/recall/F1 on fraud detection**, **FEVER score** (label correct *and* right evidence retrieved), and **calibration (ECE)** so scores mean something.

### 7. Cost-aware architecture for the $150 budget

Tiered so the paid LLM is the exception, not the rule:

| Stage | Tool | Cost |
|-------|------|------|
| Field extraction | small LLM / regex+spaCy | ~free |
| Evidence gathering | GitHub API + embeddings | free / cheap |
| Atomic-claim NLI (bulk) | DeBERTa-MNLI (local) | free |
| Code→skill mapping | CodeBERT/embeddings | cheap |
| Claim decomposition + hard-case judging | paid LLM, surgical | the only real spend |

Rough math: per candidate ≈ decomposition (2–5K tok) + judging a handful of hard atoms (5–15K tok) ≈ **$0.02–0.10**. **$150 = roughly a few thousand candidates** — plenty for build, eval, and a demo. Cache GitHub + embeddings aggressively.

---

## Target architecture (v2)

```
Resume (PDF/DOCX)
   │  parse + structured fields (spaCy/regex + small LLM)
   ▼
Claims extractor ── decompose into atomic sub-claims (LLM)
   │
   ▼
Evidence layer  ── GitHub API (commits, langs, deps, contributors, forks)
   │               + code embeddings + LinkedIn/URL checks   ── cache
   ▼
Verification  ── retrieve evidence → NLI (DeBERTa) → S/R/NEI + cited span
   │             └─ hard/low-confidence → LLM judge (structured, debiased)
   ▼
Fraud layer  ── fork/authorship/timeline/plagiarism/metric checks → Authenticity score
   ▼
Scoring  ── JD-fit | Verification | Authenticity  (kept separate, explainable)
   ▼
Report  ── per claim: verdict · evidence link · authenticity · flags  + bias-audited rank
```

---

## Suggested roadmap (fits a short build)

- **Phase 1 (must):** Swap BeautifulSoup→GitHub API; add NEI label; switch verification to retrieve→NLI with cited evidence.
- **Phase 2 (should):** Fraud layer (fork + authorship + timeline + inflated-metric flags); separate the three scores.
- **Phase 3 (could):** Code-embedding skill mapping; LLM-judge debiasing; evaluation benchmark + calibration; explainable ranking + bias audit.

---

## Synergy with your other two ideas

This is the **mirror image of the "honest resume builder" (Idea 1)**: that tool helps a candidate make *truthful* claims; this one lets a recruiter *verify* them. Same north star — **grounded, evidence-cited AI that refuses to bluff** — and the NEI label + citation discipline you'd build here is the identical trust machinery the insurance explainer (Idea 2) needs. You could pitch all three as one thesis: *trustworthy, evidence-grounded AI for high-stakes documents.*

---

## What I need to review the code

Send any of:
1. The **GitHub repo URL** (if private, add me/grant access or paste files), or
2. `requirements.txt` + the **main pipeline file** + the **GitHub-scraping module** + one **sample output report**.

Then I'll return concrete, line-level changes (where to insert NEI, how to swap BeautifulSoup for the API, where the fraud checks slot in).

---

## Sources

- [FEVER: Fact Extraction and VERification — arXiv 1803.05355](https://arxiv.org/pdf/1803.05355)
- [FEVER shared task (3-stage pipeline) — arXiv 1811.10971](https://arxiv.org/pdf/1811.10971)
- [Multi-hop Evidence Pursuit Meets the Web (FEVER 2024) — arXiv 2411.05762](https://arxiv.org/pdf/2411.05762)
- [Retrieval-Augmented Generation (RAG) — Lewis et al., arXiv 2005.11401](https://arxiv.org/pdf/2005.11401)
- [A Survey on LLM-as-a-Judge — arXiv 2411.15594](https://arxiv.org/html/2411.15594v6)
- [From Generation to Judgment: LLM-as-a-judge — arXiv 2411.16594](https://arxiv.org/pdf/2411.16594)
- [GitHub Repository Analyzer (official API approach) — Apify](https://apify.com/constant_quadruped/github-repository-analyzer)
- [Verifying commit authorship (GPG signing) — GitHub Community](https://github.com/orgs/community/discussions/123213)
- [Resume Screening RAG Pipeline (reference design) — GitHub](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline)
- [Context-Aware Explainable Multi-Agent Resume Screening — arXiv 2504.02870](https://arxiv.org/html/2504.02870v1)
