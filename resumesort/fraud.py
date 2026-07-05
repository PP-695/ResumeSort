"""Fraud-signal analysis over a candidate's GitHub snapshot.

Each signal is a heuristic grounded in published research on GitHub abuse
(fake stars, backdated commit histories, reputation farming) and resume fraud.
Signals are evidence with severity levels, never accusations: the UI presents
them as "look here", and the methodology page documents every threshold.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .github_evidence import AI_MARKER_FILES, GitHubSnapshot, RepoSnapshot
from .schemas import CandidateProfile, FraudSignal

# Skills (lowercase) -> GitHub language names for the language cross-check.
LANGUAGE_SKILLS = {
    "python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "c++": "C++",
    "c#": "C#",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "ruby": "Ruby",
    "php": "PHP",
    "swift": "Swift",
    "kotlin": "Kotlin",
    "scala": "Scala",
    "r": "R",
    "dart": "Dart",
    "elixir": "Elixir",
    "haskell": "Haskell",
}

AI_COAUTHOR_MARKERS = ("claude", "copilot", "cursor", "chatgpt", "openai", "aider", "windsurf", "devin")


def analyze_fraud_signals(snapshot: GitHubSnapshot | None, profile: CandidateProfile) -> list[FraudSignal]:
    if snapshot is None or not snapshot.repos:
        return []
    signals: list[FraudSignal] = []
    own_repos = [repo for repo in snapshot.repos if not repo.fork]

    signals.extend(_contributor_share(snapshot, own_repos))
    signals.extend(_commit_clustering(own_repos))
    signals.extend(_fork_ratios(snapshot))
    signals.extend(_identity_mismatch(snapshot, profile, own_repos))
    signals.extend(_language_crosscheck(snapshot, profile))
    signals.extend(_account_age_vs_experience(snapshot, profile))
    signals.extend(_ai_authorship(own_repos))
    return signals


def _contributor_share(snapshot: GitHubSnapshot, own_repos: list[RepoSnapshot]) -> list[FraudSignal]:
    signals = []
    username = snapshot.username.lower()
    for repo in own_repos:
        if not repo.contributors:
            continue
        total = sum(c.get("contributions", 0) for c in repo.contributors)
        own = sum(
            c.get("contributions", 0)
            for c in repo.contributors
            if (c.get("login") or "").lower() == username
        )
        if total < 5:
            continue
        share = own / total if total else 0.0
        if share < 0.25:
            signals.append(
                FraudSignal(
                    signal_id="low_contributor_share",
                    severity="high",
                    title=f"Candidate authored only {share:.0%} of commits in {repo.name}",
                    detail=(
                        f"Of {total} contributions sampled in {repo.name}, the candidate accounts for "
                        f"{own}. A repo presented as personal work is usually majority-authored by the "
                        "candidate; verify their actual role."
                    ),
                    evidence_url=f"{repo.html_url}/graphs/contributors",
                    metrics={"repo": repo.name, "share": round(share, 3), "total_contributions": total},
                )
            )
    return signals


def _commit_clustering(own_repos: list[RepoSnapshot]) -> list[FraudSignal]:
    signals = []
    for repo in own_repos:
        age_days = _age_days(repo.created_at)
        if age_days is None or age_days < 180:
            continue

        # Preferred source: 52 weeks of commit activity from the stats API.
        if repo.commit_weeks:
            total = sum(repo.commit_weeks)
            if total >= 10:
                top_two = sum(sorted(repo.commit_weeks, reverse=True)[:2])
                concentration = top_two / total
                if concentration >= 0.8:
                    signals.append(
                        FraudSignal(
                            signal_id="commit_clustering",
                            severity="high",
                            title=f"{concentration:.0%} of {repo.name}'s yearly commits landed in <=2 weeks",
                            detail=(
                                f"{repo.name} is {age_days // 30} months old but {top_two} of its {total} "
                                "commits in the past year fall inside a two-week window. Long-lived "
                                "projects normally accumulate history gradually; bulk pushes can indicate "
                                "imported or backdated work."
                            ),
                            evidence_url=f"{repo.html_url}/graphs/commit-activity",
                            metrics={"repo": repo.name, "concentration": round(concentration, 3), "total": total},
                        )
                    )
            continue

        # Fallback when stats are unavailable (API returns 202/None on cold caches):
        # cluster the sampled recent commit dates — but ONLY when the sample is
        # essentially the whole history. The last 10 commits landing in a weekend
        # is normal on an active repo and must not fire.
        if repo.total_commit_count is None or repo.total_commit_count > len(repo.recent_commits):
            continue
        dates = sorted(d for c in repo.recent_commits if (d := _parse_iso(c.get("date"))))
        if len(dates) >= 5 and (dates[-1] - dates[0]).days <= 2:
            signals.append(
                FraudSignal(
                    signal_id="commit_clustering",
                    severity="warn",
                    title=f"All sampled commits in {repo.name} fall within {max((dates[-1] - dates[0]).days, 1)} day(s)",
                    detail=(
                        f"{repo.name} is {age_days // 30} months old, yet its {len(dates)} most recent "
                        "commits were all made within two days. Worth checking whether the history was "
                        "bulk-imported."
                    ),
                    evidence_url=f"{repo.html_url}/commits",
                    metrics={"repo": repo.name, "sampled_commits": len(dates)},
                )
            )
    return signals


def _fork_ratios(snapshot: GitHubSnapshot) -> list[FraudSignal]:
    signals = []
    fork_count = sum(1 for repo in snapshot.repos if repo.fork)
    if snapshot.repos and fork_count / len(snapshot.repos) > 0.5:
        signals.append(
            FraudSignal(
                signal_id="mostly_forks",
                severity="warn",
                title=f"{fork_count} of {len(snapshot.repos)} recent repos are forks",
                detail=(
                    "The majority of the candidate's recently updated repositories are forks of other "
                    "projects. Forked tutorials presented as original work are a common padding pattern; "
                    "focus review on the non-fork repos."
                ),
                evidence_url=f"https://github.com/{snapshot.username}?tab=repositories",
                metrics={"fork_count": fork_count, "repo_count": len(snapshot.repos)},
            )
        )
    # Star anomaly needs true subscriber counts, which only deep mode fetches
    # (the shallow watchers_count field just mirrors stars).
    if not snapshot.deep:
        return signals
    for repo in snapshot.repos:
        if not repo.fork and repo.stars >= 40 and repo.subscribers is not None:
            watcher_ratio = repo.subscribers / repo.stars if repo.stars else 0
            if watcher_ratio < 0.005:
                signals.append(
                    FraudSignal(
                        signal_id="star_anomaly",
                        severity="warn",
                        title=f"{repo.name}: {repo.stars} stars but almost no watchers",
                        detail=(
                            f"{repo.name} has {repo.stars} stars and {repo.subscribers} watchers. Organic "
                            "repos typically keep a small but non-zero watcher base; near-zero ratios "
                            "correlate with purchased-star campaigns (CMU StarScout)."
                        ),
                        evidence_url=repo.html_url,
                        metrics={"repo": repo.name, "stars": repo.stars, "subscribers": repo.subscribers},
                    )
                )
    return signals


def _identity_mismatch(
    snapshot: GitHubSnapshot, profile: CandidateProfile, own_repos: list[RepoSnapshot]
) -> list[FraudSignal]:
    username = snapshot.username.lower()
    resume_email = (profile.email or "").lower()
    other_login_commits = 0
    sampled = 0
    for repo in own_repos:
        for commit in repo.recent_commits:
            login = (commit.get("author_login") or "").lower()
            email = (commit.get("author_email") or "").lower()
            if not login and not email:
                continue
            sampled += 1
            if login and login != username:
                other_login_commits += 1
            elif not login and email and "noreply" not in email and email != resume_email:
                other_login_commits += 1
    if sampled >= 5 and other_login_commits / sampled > 0.5:
        return [
            FraudSignal(
                signal_id="identity_mismatch",
                severity="high",
                title=f"{other_login_commits} of {sampled} sampled commits are authored by other identities",
                detail=(
                    "More than half of the sampled commits on the candidate's own repositories were "
                    "authored under a different GitHub login or email. This can mean team projects "
                    "presented as solo work, or copied history."
                ),
                evidence_url=f"https://github.com/{snapshot.username}",
                metrics={"other_author_commits": other_login_commits, "sampled": sampled},
            )
        ]
    return []


def _language_crosscheck(snapshot: GitHubSnapshot, profile: CandidateProfile) -> list[FraudSignal]:
    signals = []
    all_languages: set[str] = set()
    for repo in snapshot.repos:
        all_languages.update(repo.languages.keys())
    if not all_languages:
        return []
    for skill in profile.skills:
        language = LANGUAGE_SKILLS.get(skill.lower().strip())
        if language and language not in all_languages:
            signals.append(
                FraudSignal(
                    signal_id="language_not_found",
                    severity="warn",
                    title=f"Claimed language '{language}' has zero bytes across inspected repos",
                    detail=(
                        f"The resume lists {language}, but none of the {len(snapshot.repos)} inspected "
                        f"repositories contain any {language} code. Public GitHub is not exhaustive "
                        "(private/work code exists) - treat as a question, not a conclusion."
                    ),
                    evidence_url=f"https://github.com/{snapshot.username}?tab=repositories",
                    metrics={"language": language},
                )
            )
    return signals


def _account_age_vs_experience(snapshot: GitHubSnapshot, profile: CandidateProfile) -> list[FraudSignal]:
    if profile.claimed_years_experience is None:
        return []
    age_days = _age_days(snapshot.account_created_at)
    if age_days is None:
        return []
    account_years = age_days / 365.25
    if profile.claimed_years_experience > account_years + 1.0:
        return [
            FraudSignal(
                signal_id="account_age_gap",
                severity="warn",
                title=(
                    f"Resume claims {profile.claimed_years_experience:.0f} years of experience; "
                    f"GitHub account is {account_years:.1f} years old"
                ),
                detail=(
                    "A newer GitHub account than the claimed experience is common (accounts get recreated, "
                    "work happens off-GitHub), but combined with other signals it is worth a question."
                ),
                evidence_url=f"https://github.com/{snapshot.username}",
                metrics={
                    "claimed_years": profile.claimed_years_experience,
                    "account_years": round(account_years, 1),
                },
            )
        ]
    return []


def _ai_authorship(own_repos: list[RepoSnapshot]) -> list[FraudSignal]:
    signals = []
    for repo in own_repos:
        markers = [name for name in repo.root_files if name.lower() in AI_MARKER_FILES]
        coauthored = [
            c for c in repo.recent_commits
            if any(marker in (c.get("message") or "").lower() for marker in AI_COAUTHOR_MARKERS)
        ]
        if markers or len(coauthored) >= 3:
            found = ", ".join(markers) if markers else f"{len(coauthored)} AI-co-authored commit messages"
            signals.append(
                FraudSignal(
                    signal_id="ai_assisted",
                    severity="info",
                    title=f"AI-coding-agent markers in {repo.name} ({found})",
                    detail=(
                        "This repository shows signs of AI coding assistants (config files or co-author "
                        "trailers). That is normal modern practice, not fraud - but calibrate 'built this "
                        "myself' claims and probe understanding in the interview."
                    ),
                    evidence_url=repo.html_url,
                    metrics={"repo": repo.name, "markers": markers, "coauthored_commits": len(coauthored)},
                )
            )
    return signals


def _age_days(iso_date: str | None) -> int | None:
    parsed = _parse_iso(iso_date)
    if parsed is None:
        return None
    return (datetime.now(timezone.utc) - parsed).days


def _parse_iso(iso_date: str | None) -> datetime | None:
    if not iso_date:
        return None
    try:
        parsed = datetime.fromisoformat(iso_date)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
