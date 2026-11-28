"""Read-only GitHub context for a pull-request pre-review.

This module never approves, merges, or comments. The only GitHub write in the
process remains the optional contributor-role invite, which is not called here.
"""

import logging
import re
from dataclasses import dataclass, field

from .api import GITHUB_API, fetch_pull_diff, github_get_json, github_graphql

_SHA = re.compile(r"[0-9a-f]{40}")
_QUERY = """
query ReviewThreads($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        nodes {
          isResolved
          isOutdated
          path
          comments(first: 10) {
            nodes {
              author { login }
              body
              commit { oid }
              line
              originalLine
            }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


@dataclass
class Snapshot:
    number: int
    title: str
    head_sha: str
    draft: bool
    merged: bool
    state: str
    mergeable: bool | None
    mergeable_state: str | None
    author_login: str
    author_id: int
    base_sha: str = ""
    reviews: list = field(default_factory=list)
    comments: list = field(default_factory=list)
    threads: list | None = None
    check_runs: list | None = None
    combined_state: str | None = None
    diff: str = ""
    body: str = ""


def _pages(url: str, token: str | None, label: str) -> list:
    items: list = []
    page = 1
    while page <= 10:
        payload = github_get_json(f"{url}?per_page=100&page={page}", token, pace=True)
        if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
            raise RuntimeError(f"GitHub returned an unexpected {label} response")
        items.extend(payload)
        if len(payload) < 100:
            return items
        page += 1
    return items


def _pull(repository: str, number: int, token: str | None) -> dict:
    url = f"{GITHUB_API}/repos/{repository}/pulls/{number}"
    pull = github_get_json(url, token, pace=True)
    if not isinstance(pull, dict) or pull.get("number") != number:
        raise RuntimeError("GitHub returned an unexpected pull-request response")
    if pull.get("mergeable") is None and pull.get("state") == "open":
        pull = github_get_json(url, token, pace=True)
        if not isinstance(pull, dict) or pull.get("number") != number:
            raise RuntimeError("GitHub returned an unexpected pull-request response")
    return pull


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _check_runs(repository: str, sha: str, token: str | None) -> list:
    if not _SHA.fullmatch(sha):
        raise ValueError("Unexpected commit sha")
    runs: list = []
    page = 1
    while page <= 5:
        payload = github_get_json(
            f"{GITHUB_API}/repos/{repository}/commits/{sha}/check-runs?per_page=100&page={page}",
            token,
            pace=True,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("check_runs"), list):
            raise RuntimeError("GitHub returned an unexpected check-runs response")
        runs.extend(item for item in payload["check_runs"] if isinstance(item, dict))
        if len(payload["check_runs"]) < 100:
            return runs
        page += 1
    return runs


def _combined_state(repository: str, sha: str, token: str | None) -> str | None:
    """Commit statuses only. Empty status lists are ignored so check runs can stand alone."""
    if not _SHA.fullmatch(sha):
        raise ValueError("Unexpected commit sha")
    payload = github_get_json(f"{GITHUB_API}/repos/{repository}/commits/{sha}/status", token, pace=True)
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub returned an unexpected status response")
    if payload.get("total_count") == 0:
        return None
    state = payload.get("state")
    if state in {"success", "pending", "failure", "error"}:
        return state
    return None


def _threads(repository: str, number: int, token: str | None) -> list | None:
    owner, _, name = repository.partition("/")
    if not owner or not name:
        return None
    threads: list = []
    cursor = None
    for _page in range(5):
        try:
            payload = github_graphql(token, _QUERY, {
                "owner": owner,
                "name": name,
                "number": number,
                "cursor": cursor,
            })
        except Exception as error:
            logging.warning("Review thread resolution for #%s is unavailable: %s", number, error)
            return None
        if not isinstance(payload, dict) or payload.get("errors"):
            logging.warning("Review thread resolution for #%s returned no data", number)
            return None
        connection = (
            ((payload.get("data") or {}).get("repository") or {}).get("pullRequest") or {}
        ).get("reviewThreads") or {}
        nodes = connection.get("nodes")
        if not isinstance(nodes, list):
            return None
        threads.extend(node for node in nodes if isinstance(node, dict))
        page_info = connection.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            return threads
        cursor = page_info.get("endCursor")
        if not isinstance(cursor, str) or not cursor:
            return threads
    return threads


def _diff(repository: str, number: int, token: str | None) -> str:
    try:
        text = fetch_pull_diff(repository, number, token)
    except Exception as error:
        logging.warning("Diff for #%s is unavailable: %s", number, error)
        return ""
    if len(text) > 300_000:
        return text[:300_000] + "\n(diff truncated before review)\n"
    return text


def gather_snapshot(repository: str, number: int, token: str | None) -> Snapshot:
    """Fetch the pull request, bot reviews, CI, and diff. Reads only."""
    pull = _pull(repository, number, token)
    user = pull.get("user") if isinstance(pull.get("user"), dict) else {}
    login = user.get("login") if isinstance(user.get("login"), str) else ""
    author_id = user.get("id") if isinstance(user.get("id"), int) else 0
    head = pull.get("head") if isinstance(pull.get("head"), dict) else {}
    sha = head.get("sha") if isinstance(head.get("sha"), str) else ""
    base = pull.get("base") if isinstance(pull.get("base"), dict) else {}
    base_sha = base.get("sha") if isinstance(base.get("sha"), str) else ""
    title = pull.get("title") if isinstance(pull.get("title"), str) else ""
    body = pull.get("body") if isinstance(pull.get("body"), str) else ""
    merged = pull.get("merged") is True
    state = pull.get("state") if pull.get("state") in {"open", "closed"} else "closed"
    draft = pull.get("draft") is True
    mergeable_state = pull.get("mergeable_state") if isinstance(pull.get("mergeable_state"), str) else None
    snapshot = Snapshot(
        number=number,
        title=title,
        head_sha=sha,
        draft=draft,
        merged=merged,
        state=state,
        mergeable=_bool_or_none(pull.get("mergeable")),
        mergeable_state=mergeable_state,
        author_login=login,
        author_id=author_id,
        base_sha=base_sha if re.fullmatch(r"[0-9a-f]{40}", base_sha) else "",
        body=body,
    )
    if merged or state != "open":
        return snapshot
    if not _SHA.fullmatch(sha):
        raise RuntimeError("GitHub pull request is missing a head sha")
    base = f"{GITHUB_API}/repos/{repository}/pulls/{number}"
    snapshot.reviews = _pages(f"{base}/reviews", token, "reviews")
    snapshot.comments = _pages(f"{base}/comments", token, "comments")
    snapshot.threads = _threads(repository, number, token)
    try:
        snapshot.check_runs = _check_runs(repository, sha, token)
    except Exception as error:
        logging.warning("Check runs for #%s are unavailable: %s", number, error)
        snapshot.check_runs = None
    try:
        snapshot.combined_state = _combined_state(repository, sha, token)
    except Exception as error:
        logging.warning("Combined status for #%s is unavailable: %s", number, error)
        snapshot.combined_state = None
    # The full unified diff is not shown to the model. File patches are read
    # one path at a time, and only when a bot finding or a failing check names it.
    return snapshot
