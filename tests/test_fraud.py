from datetime import datetime, timedelta, timezone

from resumesort.fraud import analyze_fraud_signals
from resumesort.github_evidence import GitHubSnapshot, RepoSnapshot
from resumesort.schemas import CandidateProfile


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _repo(**kwargs) -> RepoSnapshot:
    defaults = dict(
        name="proj",
        html_url="https://github.com/cand/proj",
        fork=False,
        stars=1,
        subscribers=1,
        forks_count=0,
        created_at=_iso(400),
        updated_at=_iso(1),
        pushed_at=_iso(1),
    )
    defaults.update(kwargs)
    return RepoSnapshot(**defaults)


def _snapshot(repos, username="cand", account_days=1000, deep=True) -> GitHubSnapshot:
    return GitHubSnapshot(username=username, account_created_at=_iso(account_days), deep=deep, repos=repos)


PROFILE = CandidateProfile(name="Cand", email="cand@example.com", skills=["Python"])


def _ids(signals):
    return {signal.signal_id for signal in signals}


def test_no_snapshot_no_signals():
    assert analyze_fraud_signals(None, PROFILE) == []


def test_low_contributor_share():
    repo = _repo(contributors=[{"login": "other", "contributions": 90}, {"login": "cand", "contributions": 10}])
    signals = analyze_fraud_signals(_snapshot([repo]), PROFILE)
    assert "low_contributor_share" in _ids(signals)
    assert next(s for s in signals if s.signal_id == "low_contributor_share").severity == "high"


def test_commit_clustering_from_weeks():
    weeks = [0] * 50 + [45, 45]
    repo = _repo(commit_weeks=weeks)
    signals = analyze_fraud_signals(_snapshot([repo]), PROFILE)
    assert "commit_clustering" in _ids(signals)


def test_commit_clustering_fallback_on_missing_stats():
    commits = [
        {"date": _iso(2), "author_login": "cand", "author_email": None, "message": "x", "html_url": ""}
        for _ in range(6)
    ]
    # Sample IS the whole history (total 6 commits) -> genuine bulk-import pattern.
    repo = _repo(commit_weeks=None, recent_commits=commits, total_commit_count=6)
    signals = analyze_fraud_signals(_snapshot([repo]), PROFILE)
    clustering = [s for s in signals if s.signal_id == "commit_clustering"]
    assert clustering and clustering[0].severity == "warn"


def test_commit_clustering_fallback_skips_active_repos():
    """A weekend burst on a repo with a long history must NOT fire (regression A4)."""
    commits = [
        {"date": _iso(2), "author_login": "cand", "author_email": None, "message": "x", "html_url": ""}
        for _ in range(10)
    ]
    repo = _repo(commit_weeks=None, recent_commits=commits, total_commit_count=250)
    signals = analyze_fraud_signals(_snapshot([repo]), PROFILE)
    assert not [s for s in signals if s.signal_id == "commit_clustering"]


def test_mostly_forks():
    repos = [_repo(name=f"f{i}", fork=True) for i in range(3)] + [_repo(name="own")]
    signals = analyze_fraud_signals(_snapshot(repos), PROFILE)
    assert "mostly_forks" in _ids(signals)


def test_star_anomaly():
    repo = _repo(stars=500, subscribers=0)
    signals = analyze_fraud_signals(_snapshot([repo]), PROFILE)
    assert "star_anomaly" in _ids(signals)


def test_star_anomaly_gated_on_deep_mode():
    """Shallow mode has no true subscriber counts - signal must not fire (regression A5)."""
    repo = _repo(stars=500, subscribers=None)
    signals = analyze_fraud_signals(_snapshot([repo], deep=False), PROFILE)
    assert "star_anomaly" not in _ids(signals)


def test_identity_mismatch():
    commits = [
        {"date": _iso(5), "author_login": "someone_else", "author_email": "x@y.com", "message": "m", "html_url": ""}
        for _ in range(6)
    ]
    repo = _repo(recent_commits=commits)
    signals = analyze_fraud_signals(_snapshot([repo]), PROFILE)
    assert "identity_mismatch" in _ids(signals)


def test_language_crosscheck():
    profile = CandidateProfile(skills=["Rust", "Python"])
    repo = _repo(languages={"Python": 1000})
    signals = analyze_fraud_signals(_snapshot([repo]), profile)
    language_signals = [s for s in signals if s.signal_id == "language_not_found"]
    assert len(language_signals) == 1
    assert language_signals[0].metrics["language"] == "Rust"


def test_account_age_gap():
    profile = CandidateProfile(claimed_years_experience=8.0)
    snapshot = _snapshot([_repo()], account_days=365)
    signals = analyze_fraud_signals(snapshot, profile)
    assert "account_age_gap" in _ids(signals)


def test_ai_markers_are_info_only():
    repo = _repo(root_files=[".claude", "src", "README.md"])
    signals = analyze_fraud_signals(_snapshot([repo]), PROFILE)
    ai = [s for s in signals if s.signal_id == "ai_assisted"]
    assert ai and ai[0].severity == "info"


def test_clean_profile_no_high_signals():
    commits = [
        {"date": _iso(i * 30), "author_login": "cand", "author_email": "cand@example.com", "message": "m", "html_url": ""}
        for i in range(1, 7)
    ]
    repo = _repo(
        recent_commits=commits,
        contributors=[{"login": "cand", "contributions": 50}],
        languages={"Python": 5000},
        commit_weeks=[2] * 52,
    )
    signals = analyze_fraud_signals(_snapshot([repo]), PROFILE)
    assert not [s for s in signals if s.severity == "high"]
