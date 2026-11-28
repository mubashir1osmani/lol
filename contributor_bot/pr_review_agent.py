"""PR Review Agent (``pr_review_agent``).

Dedicated top-level OpenInference ``AGENT`` module owning everything review-related:
- PR review message handling & classifier (``pr_review_classifier`` child span)
- Review thread creation and multi-turn SDK review against
  ``review_standards/LITELLM_STANDARDS.md`` with root-cause analysis and
  actionable suggestions
- Thread follow-up Q&A and ``/recheck`` triggers
- Contributor PR board / backlog sync
- PR lifecycle tracking on review threads until the PR is merged AND included
  in a LiteLLM release, posting concise status updates to the thread
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from .api import (
    fetch_pull_request,
    fetch_recent_releases,
    fetch_release_entries,
    post_message,
)
from .config import Config, is_discord_id
from .releases import release_url
from .review_agent import (
    AgentResult,
    TOOLS,
    build_system_prompt,
    run_review_session,
)
from .schedule import UTC, iso, parse_time
from .store import Store
from .tracing import (
    AGENT_PR_REVIEW,
    OPENINFERENCE_SPAN_KIND_CHAIN,
    agent_span,
    child_span,
)

AGENT_NAME = AGENT_PR_REVIEW
PROMPT = build_system_prompt()


def build_prompt(standards: str | None = None) -> str:
    """Return the PR review agent system prompt backed by LITELLM_STANDARDS.md."""
    return build_system_prompt(standards)


def handle_routed_message(
    config: Config,
    store: Store,
    channel_id: str,
    message: dict,
    *,
    route_info: dict | None = None,
) -> None:
    """Entry point for a Discord channel message routed to ``pr_review_agent``."""
    from .review_job import handle_review_channel_message

    text = str(message.get("content") or "")
    message_id = str(message.get("id") or "")
    session_id = f"{AGENT_NAME}:{channel_id}:{message_id or 'msg'}"
    with agent_span(
        AGENT_NAME,
        session_id=session_id,
        input_value={"channel_id": str(channel_id), "message_id": message_id, "content": text[:800]},
        attributes={
            "discord.channel_id": str(channel_id),
            "discord.message_id": message_id,
            "pr.repository": config.github_repository,
        },
    ) as span:
        if route_info is not None:
            with child_span(
                "message_router",
                kind=OPENINFERENCE_SPAN_KIND_CHAIN,
                input_value={"channel_id": str(channel_id), "content": text[:500]},
                attributes={"router.target": AGENT_NAME},
            ) as router_span:
                router_span.set_output(route_info)
        outcome = handle_review_channel_message(config, store, channel_id, message)
        span.set_output(outcome or {"handled": True})


def handle_thread_followup(config: Config, store: Store, message: dict) -> None:
    """Entry point for a follow-up or recheck message inside an existing review thread."""
    from .review_job import handle_thread_message

    channel_id = str(message.get("channel_id") or "")
    author = message.get("author") or {}
    if not config.pr_review_enabled or author.get("bot") or message.get("webhook_id"):
        return
    records = [
        rec for rec in (store.review_threads_by_discord(channel_id) if is_discord_id(channel_id) else [])
        if rec["repository"].lower() == config.github_repository.lower()
    ]
    if not records:
        return
    numbers = [int(rec["number"]) for rec in records]
    primary = numbers[0]
    text = str(message.get("content") or "")
    with agent_span(
        AGENT_NAME,
        session_id=f"{AGENT_NAME}:{config.github_repository}#{primary}",
        input_value={
            "repository": config.github_repository,
            "pr_numbers": numbers,
            "thread_id": channel_id,
            "content": text[:800],
        },
        attributes={
            "pr.repository": config.github_repository,
            "pr.number": primary,
            "discord.thread_id": channel_id,
            "review.mode": "followup",
        },
    ) as span:
        handle_thread_message(config, store, message)
        span.set_output({"thread_id": channel_id, "pr_numbers": numbers})


def run_review(
    config: Config,
    store: Store,
    number: int,
    thread_id: str,
    *,
    mode: str = "review",
    history: str = "",
) -> AgentResult:
    """Run one PR review or follow-up session inside the ``pr_review_agent`` span."""
    return run_review_session(config, store, number, thread_id, mode=mode, history=history)


def sync_board(config: Config, store: Store, **kwargs) -> dict:
    """Synchronize tracked PRs with the GitHub Projects v2 contributor PR board."""
    from .pr_board import sync_pr_board

    return sync_pr_board(config, store, **kwargs)


def _pr_mentioned_in_text(text: str, repository: str, number: int) -> bool:
    if not text:
        return False
    repo_esc = re.escape(repository)
    patterns = (
        rf"https://github\.com/{repo_esc}/pull/{number}\b",
        rf"(?<![0-9A-Za-z_])#{number}\b",
        rf"\bpull/{number}\b",
    )
    return any(re.search(pat, text, re.IGNORECASE) for pat in patterns)


def _find_pr_in_github_releases(
    releases: list[dict],
    repository: str,
    number: int,
    merged_at: datetime | None,
) -> tuple[str, str] | None:
    """Return ``(tag_name, html_url)`` if any GitHub release after merge references ``#number``."""
    cutoff = merged_at - timedelta(days=1) if merged_at is not None else None
    for rel in releases:
        if rel.get("draft"):
            continue
        published_raw = str(rel.get("published_at") or rel.get("created_at") or "")
        if cutoff is not None and published_raw:
            try:
                if parse_time(published_raw) < cutoff:
                    continue
            except ValueError:
                pass
        body = str(rel.get("body") or "")
        name = str(rel.get("name") or "")
        tag = str(rel.get("tag_name") or name).strip()
        if not tag:
            continue
        if _pr_mentioned_in_text(body, repository, number) or _pr_mentioned_in_text(name, repository, number):
            url = str(rel.get("html_url") or "").strip()
            if not url:
                url = f"https://github.com/{repository}/releases/tag/{tag}"
            return tag, url
    return None


def _find_pr_in_docs_releases(
    entries: dict,
    repository: str,
    number: int,
    merged_at: datetime | None,
    *,
    release_prs_fetcher,
    page_cache: dict[str, list[int]],
) -> tuple[str, str] | None:
    """Check official docs.litellm.ai release notes pages for ``number``."""
    if not isinstance(entries, dict):
        return None
    cutoff_date = (merged_at - timedelta(days=1)).date() if merged_at is not None else None
    # Inspect newest release notes first (capped to keep polling light).
    sorted_items = sorted(entries.items(), reverse=True)[:10]
    for date_key, url in sorted_items:
        if not isinstance(url, str) or release_url(url) != url:
            continue
        if cutoff_date is not None and isinstance(date_key, str):
            try:
                if datetime.fromisoformat(date_key).date() < cutoff_date:
                    continue
            except ValueError:
                pass
        if url not in page_cache:
            try:
                page_cache[url] = list(release_prs_fetcher(url, repository))
            except Exception as error:
                logging.debug("Could not read docs release page %s: %s", url, error)
                page_cache[url] = []
        if number in page_cache[url]:
            tag = url.rstrip("/").rsplit("/", 1)[-1] or str(date_key)
            return tag, url
    return None


def _default_release_prs_fetcher(url: str, repository: str) -> list[int]:
    from .api import conditional_get
    from .releases import parse_new_contributors

    html = conditional_get(url, {}).decode("utf-8", errors="replace")
    found = set(parse_new_contributors(html, repository))
    repo_esc = re.escape(repository)
    for match in re.finditer(rf"https://github\.com/{repo_esc}/pull/([1-9][0-9]*)\b", html, re.IGNORECASE):
        found.add(int(match.group(1)))
    return sorted(found)


def poll_review_thread_lifecycle(
    config: Config,
    store: Store,
    *,
    now: datetime | None = None,
    pull_fetcher=None,
    release_entries_fetcher=None,
    release_prs_fetcher=None,
    github_releases_fetcher=None,
    message_poster=None,
) -> dict:
    """Watch active review threads until each PR is merged AND ships in a LiteLLM release.

    Posts one concise update to the Discord review thread when the PR is merged
    and one concise update with the release link once it is included in a
    LiteLLM release. Stops tracking PRs that are closed without merging.
    """
    summary = {
        "checked": 0,
        "merged_updates": 0,
        "release_updates": 0,
        "closed_unmerged": 0,
    }
    if not config.pr_review_enabled:
        return summary

    threads = store.active_lifecycle_review_threads(config.github_repository)
    if not threads:
        return summary

    moment = now or datetime.now(UTC)
    now_iso = iso(moment)
    token = config.effective_github_token
    fetch_pr = pull_fetcher or (lambda repo, num: fetch_pull_request(repo, num, token))
    fetch_entries = release_entries_fetcher or fetch_release_entries
    fetch_rel_prs = release_prs_fetcher or _default_release_prs_fetcher
    fetch_gh_rels = github_releases_fetcher or (lambda repo: fetch_recent_releases(repo, token, limit=10))
    post = message_poster or (
        lambda tid, content: post_message(
            tid,
            config.discord_token,
            {"content": content, "allowed_mentions": {"parse": []}},
        )
    )

    gh_releases_loaded = False
    gh_releases: list[dict] = []
    docs_entries_loaded = False
    docs_entries: dict = {}
    docs_page_cache: dict[str, list[int]] = {}

    for rec in threads:
        number = int(rec["number"])
        thread_id = str(rec["thread_id"])
        if not is_discord_id(thread_id):
            continue
        summary["checked"] += 1

        merged_at_str = rec.get("merged_at")
        merge_notified_at = rec.get("merge_notified_at")

        if not merged_at_str:
            try:
                pull = fetch_pr(config.github_repository, number)
            except Exception as error:
                logging.warning("Lifecycle poll could not fetch PR #%s: %s", number, error)
                continue
            if pull.get("merged"):
                merged_at_str = str(pull.get("merged_at") or now_iso)
                merge_sha = str(pull.get("merge_commit_sha") or "")
                notified = None
                if not merge_notified_at:
                    note = f"pr #{number} is merged — i'll post an update here once it's in a litellm release."
                    try:
                        post(thread_id, note)
                        notified = now_iso
                        summary["merged_updates"] += 1
                    except Exception as error:
                        logging.warning("Could not post merge update for #%s to %s: %s", number, thread_id, error)
                store.mark_review_thread_merged(
                    config.github_repository,
                    number,
                    merged_at=merged_at_str,
                    merge_commit_sha=merge_sha,
                    notified_at=notified,
                )
            elif str(pull.get("state") or "").lower() == "closed":
                store.mark_review_thread_closed_unmerged(
                    config.github_repository,
                    number,
                    closed_at=str(pull.get("closed_at") or now_iso),
                )
                summary["closed_unmerged"] += 1
                continue
            else:
                continue

        # PR is merged — check whether it has landed in a LiteLLM release.
        try:
            merged_dt = parse_time(merged_at_str) if merged_at_str else None
        except ValueError:
            merged_dt = None

        found_release: tuple[str, str] | None = None
        if not gh_releases_loaded:
            gh_releases_loaded = True
            try:
                gh_releases = list(fetch_gh_rels(config.github_repository) or [])
            except Exception as error:
                logging.debug("Could not fetch GitHub releases for lifecycle check: %s", error)
                gh_releases = []
        found_release = _find_pr_in_github_releases(gh_releases, config.github_repository, number, merged_dt)

        if found_release is None:
            if not docs_entries_loaded:
                docs_entries_loaded = True
                try:
                    docs_entries = dict(fetch_entries() or {})
                except Exception as error:
                    logging.debug("Could not fetch docs.litellm.ai release index: %s", error)
                    docs_entries = {}
            found_release = _find_pr_in_docs_releases(
                docs_entries,
                config.github_repository,
                number,
                merged_dt,
                release_prs_fetcher=fetch_rel_prs,
                page_cache=docs_page_cache,
            )

        if found_release is None:
            continue

        tag, rel_url = found_release
        note = f"pr #{number} is out in litellm {tag}: {rel_url}"
        try:
            post(thread_id, note)
            store.mark_review_thread_released(
                config.github_repository,
                number,
                release_tag=tag,
                release_url=rel_url,
                notified_at=now_iso,
            )
            summary["release_updates"] += 1
        except Exception as error:
            logging.warning("Could not post release update for #%s to %s: %s", number, thread_id, error)

    return summary


__all__ = [
    "AGENT_NAME",
    "PROMPT",
    "TOOLS",
    "AgentResult",
    "build_prompt",
    "handle_routed_message",
    "handle_thread_followup",
    "poll_review_thread_lifecycle",
    "run_review",
    "sync_board",
]
