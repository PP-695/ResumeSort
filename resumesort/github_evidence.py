from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schemas import EvidenceItem

_SNAPSHOT_TTL_SECONDS = 15 * 60
_SNAPSHOT_CACHE: dict[tuple[str, int, bool], tuple[float, "GitHubSnapshot | None", list[str]]] = {}

# Root-level files/dirs that indicate AI-coding-agent involvement (informational).
AI_MARKER_FILES = {".claude", ".cursor", "claude.md", "agents.md", ".aider", ".windsurf"}


@dataclass
class RepoSnapshot:
    name: str
    html_url: str
    fork: bool
    stars: int
    # True subscriber count in deep mode; None in shallow mode (the REST v3
    # watchers_count field just mirrors stars, which is useless for anomaly checks).
    subscribers: int | None
    forks_count: int
    created_at: str | None
    updated_at: str | None
    pushed_at: str | None
    languages: dict[str, int] = field(default_factory=dict)
    recent_commits: list[dict[str, Any]] = field(default_factory=list)
    contributors: list[dict[str, Any]] = field(default_factory=list)
    commit_weeks: list[int] | None = None
    total_commit_count: int | None = None
    root_files: list[str] = field(default_factory=list)
    readme_text: str = ""
    readme_url: str = ""


@dataclass
class GitHubSnapshot:
    username: str
    account_created_at: str | None = None
    public_repos: int = 0
    followers: int = 0
    deep: bool = False
    repos: list[RepoSnapshot] = field(default_factory=list)


def github_username(github_url: str | None) -> str | None:
    if not github_url:
        return None
    match = re.search(r"github\.com/([A-Za-z0-9_.-]+)", github_url)
    return match.group(1) if match else None


def clear_snapshot_cache() -> None:
    _SNAPSHOT_CACHE.clear()


def fetch_github_snapshot(
    github_url: str | None,
    token: str | None = None,
    max_repos: int = 10,
    deep: bool = True,
) -> tuple[GitHubSnapshot | None, list[str]]:
    """Fetch a candidate's GitHub activity snapshot.

    Deep mode adds contributors, weekly commit stats, and root-file listings
    (~6 API calls per repo) and requires a token to stay within rate limits.
    Without a token we degrade to shallow mode: 5 repos, no deep calls.
    """
    username = github_username(github_url)
    if not username:
        return None, ["Missing GitHub link"]

    deep = deep and bool(token)
    if not token and max_repos > 5:
        max_repos = 5

    cache_key = (username.lower(), max_repos, deep)
    cached = _SNAPSHOT_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < _SNAPSHOT_TTL_SECONDS:
        return cached[1], list(cached[2])

    flags: list[str] = []
    if not deep:
        flags.append("Anonymous/shallow GitHub mode: deep fraud checks skipped")

    try:
        from github import Github
    except Exception:
        return None, ["PyGithub is not installed; GitHub evidence unavailable"]

    try:
        client = Github(token) if token else Github()
        user = client.get_user(username)
        snapshot = GitHubSnapshot(
            username=username,
            account_created_at=_safe_iso(user.created_at),
            public_repos=user.public_repos,
            followers=user.followers,
            deep=deep,
        )
        # Own work first: forks must not occupy the max_repos budget while the
        # candidate's original repos get starved out.
        candidates = list(user.get_repos(sort="updated"))[: max_repos * 3]
        candidates.sort(key=lambda r: (bool(r.fork), -(r.pushed_at.timestamp() if r.pushed_at else 0)))
        repos = candidates[:max_repos]
    except Exception as exc:
        return None, [f"GitHub evidence unavailable: {exc}"]

    for repo in repos:
        repo_snap = RepoSnapshot(
            name=repo.name,
            html_url=repo.html_url,
            fork=bool(repo.fork),
            stars=repo.stargazers_count,
            subscribers=repo.subscribers_count if deep else None,
            forks_count=repo.forks_count,
            created_at=_safe_iso(repo.created_at),
            updated_at=_safe_iso(repo.updated_at),
            pushed_at=_safe_iso(repo.pushed_at),
        )

        try:
            repo_snap.languages = repo.get_languages() or {}
        except Exception:
            pass

        try:
            for commit in repo.get_commits()[:10]:
                author = commit.commit.author
                repo_snap.recent_commits.append(
                    {
                        "sha": commit.sha,
                        "html_url": commit.html_url,
                        "date": _safe_iso(getattr(author, "date", None)),
                        "message": commit.commit.message.splitlines()[0][:200],
                        "author_email": getattr(author, "email", None),
                        "author_login": commit.author.login if commit.author else None,
                    }
                )
        except Exception:
            pass

        if deep:
            try:
                repo_snap.contributors = [
                    {"login": c.login, "contributions": c.contributions}
                    for c in list(repo.get_contributors()[:10])
                ]
            except Exception:
                pass
            try:
                activity = repo.get_stats_commit_activity()
                if activity:
                    repo_snap.commit_weeks = [week.total for week in activity]
            except Exception:
                pass
            if repo_snap.commit_weeks is None:
                # Stats API cold (202). The clustering fallback needs to know
                # whether the sampled commits are the WHOLE history.
                try:
                    repo_snap.total_commit_count = repo.get_commits().totalCount
                except Exception:
                    pass
            try:
                repo_snap.root_files = [item.name for item in repo.get_contents("")][:50]
            except Exception:
                pass

        try:
            readme = repo.get_readme()
            repo_snap.readme_text = readme.decoded_content.decode("utf-8", errors="ignore")[:8000]
            repo_snap.readme_url = readme.html_url or repo.html_url
        except Exception:
            pass

        snapshot.repos.append(repo_snap)

    flags.extend(_legacy_flags(snapshot))
    _SNAPSHOT_CACHE[cache_key] = (time.time(), snapshot, list(flags))
    return snapshot, flags


def snapshot_to_evidence(snapshot: GitHubSnapshot | None) -> list[EvidenceItem]:
    if snapshot is None:
        return []
    evidence: list[EvidenceItem] = []
    for repo in snapshot.repos:
        base_metadata = {
            "html_url": repo.html_url,
            "fork": repo.fork,
            "stars": repo.stars,
            "subscribers": repo.subscribers,
            "forks": repo.forks_count,
            "created_at": repo.created_at,
            "updated_at": repo.updated_at,
        }
        if repo.languages:
            evidence.append(
                EvidenceItem(
                    source_type="languages",
                    repo_name=repo.name,
                    path_or_url=repo.html_url,
                    text=", ".join(sorted(repo.languages.keys())),
                    metadata={**base_metadata, "languages": repo.languages},
                )
            )
        if repo.recent_commits:
            commit_lines = [f"{c['date']} {c['message']}" for c in repo.recent_commits]
            evidence.append(
                EvidenceItem(
                    source_type="commits",
                    repo_name=repo.name,
                    path_or_url=f"{repo.html_url}/commits",
                    text="\n".join(commit_lines),
                    metadata={
                        **base_metadata,
                        "commit_urls": [c["html_url"] for c in repo.recent_commits],
                    },
                )
            )
        if repo.readme_text:
            evidence.append(
                EvidenceItem(
                    source_type="readme",
                    repo_name=repo.name,
                    path_or_url=repo.readme_url or repo.html_url,
                    text=repo.readme_text,
                    metadata=base_metadata,
                )
            )
    return evidence


def fetch_github_evidence(
    github_url: str | None,
    token: str | None = None,
    max_repos: int = 10,
) -> tuple[list[EvidenceItem], list[str]]:
    """Backward-compatible wrapper: snapshot -> evidence items + flags."""
    snapshot, flags = fetch_github_snapshot(github_url, token=token, max_repos=max_repos, deep=False)
    return snapshot_to_evidence(snapshot), flags


def _legacy_flags(snapshot: GitHubSnapshot) -> list[str]:
    flags: list[str] = []
    if not snapshot.repos:
        flags.append("No matching repo evidence found")
        return flags
    fork_count = sum(1 for repo in snapshot.repos if repo.fork)
    sparse_count = sum(1 for repo in snapshot.repos if len(repo.recent_commits) <= 1)
    if fork_count:
        flags.append(f"{fork_count} repository/repositories are forks")
    if sparse_count:
        flags.append(f"{sparse_count} repository/repositories have very low recent commit signal")
    only_readme = all(
        not repo.languages and not repo.recent_commits and repo.readme_text for repo in snapshot.repos
    )
    if only_readme:
        flags.append("README-focused evidence; code-level proof may be limited")
    return flags


def _safe_iso(value: datetime | None) -> str | None:
    if not value:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()
