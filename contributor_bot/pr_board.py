"""GitHub Projects (v2) backlog board sync for community contributor pull requests."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .api import github_graphql
from .reviews import DEFAULT_REVIEW_BOTS
from .schedule import UTC, iso, parse_time
from .weekly import (
    EXCLUDED_AUTHOR_ASSOCIATIONS,
    _bot_login,
    _org_staff_login,
    fetch_org_public_members,
)

STATUS_OPEN = "Open"
STATUS_IN_REVIEW = "In review"
STATUS_CHANGES_REQUESTED = "Changes requested"
STATUS_APPROVED = "Approved"
STATUS_MERGED = "Merged"
STATUS_CLOSED = "Closed"

BOARD_STATUSES = (
    STATUS_OPEN,
    STATUS_IN_REVIEW,
    STATUS_CHANGES_REQUESTED,
    STATUS_APPROVED,
    STATUS_MERGED,
    STATUS_CLOSED,
)

PROJECT_METADATA_TTL_SECONDS = 1800.0
BOARD_SYNC_DEADLINE_SECONDS = 60.0
BOARD_GRAPHQL_TIMEOUT_SECONDS = 15.0
BOARD_MIN_RATE_LIMIT_REMAINING = 50
DEFAULT_OPEN_PAGES = 3
DEFAULT_CLOSED_PAGES = 1
RECENT_CLOSED_LOOKBACK_DAYS = 7

_sync_lock = threading.Lock()
_meta_lock = threading.Lock()
_project_meta_cache: dict[tuple[str | None, str | None, int | None, str], tuple[ProjectMetadata, float]] = {}

_PROJECT_FIELDS_FRAGMENT = """
id
title
url
fields(first: 50) {
  nodes {
    ... on ProjectV2SingleSelectField {
      id
      name
      options {
        id
        name
      }
    }
  }
}
"""

_PROJECT_BY_ID_QUERY = f"""
query ($projectId: ID!) {{
  node(id: $projectId) {{
    ... on ProjectV2 {{
      {_PROJECT_FIELDS_FRAGMENT}
    }}
  }}
  rateLimit {{
    remaining
    resetAt
  }}
}}
"""

_ORG_PROJECT_QUERY = f"""
query ($owner: String!, $number: Int!) {{
  organization(login: $owner) {{
    projectV2(number: $number) {{
      {_PROJECT_FIELDS_FRAGMENT}
    }}
  }}
  rateLimit {{
    remaining
    resetAt
  }}
}}
"""

_USER_PROJECT_QUERY = f"""
query ($owner: String!, $number: Int!) {{
  user(login: $owner) {{
    projectV2(number: $number) {{
      {_PROJECT_FIELDS_FRAGMENT}
    }}
  }}
  rateLimit {{
    remaining
    resetAt
  }}
}}
"""

_PULLS_QUERY = """
query ($owner: String!, $name: String!, $states: [PullRequestState!], $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      first: 50
      after: $after
      states: $states
      orderBy: { field: UPDATED_AT, direction: DESC }
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        id
        number
        title
        url
        state
        updatedAt
        authorAssociation
        reviewDecision
        author {
          __typename
          login
        }
        reviews(last: 25) {
          nodes {
            state
            author {
              __typename
              login
            }
          }
        }
        comments(last: 25) {
          nodes {
            author {
              __typename
              login
            }
          }
        }
      }
    }
  }
  rateLimit {
    remaining
    resetAt
  }
}
"""

_ADD_ITEM_MUTATION = """
mutation ($projectId: ID!, $contentId: ID!) {
  addProjectV2ItemById(input: { projectId: $projectId, contentId: $contentId }) {
    item {
      id
    }
  }
}
"""

_UPDATE_STATUS_MUTATION = """
mutation ($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
  updateProjectV2ItemFieldValue(
    input: {
      projectId: $projectId
      itemId: $itemId
      fieldId: $fieldId
      value: { singleSelectOptionId: $optionId }
    }
  ) {
    projectV2Item {
      id
    }
  }
}
"""


@dataclass(frozen=True)
class ProjectMetadata:
    project_id: str
    title: str
    url: str
    status_field_id: str
    status_options: dict[str, str]  # canonical status label -> singleSelectOptionId


@dataclass(frozen=True)
class SyncResult:
    project_id: str
    project_url: str
    scanned: int
    community: int
    added: int
    updated: int
    unchanged: int
    skipped_missing_option: int
    rate_limited: bool = False


def reset_pr_board_cache() -> None:
    """Clear in-memory project metadata cache (for tests)."""
    with _meta_lock:
        _project_meta_cache.clear()


def _review_bot_set(config) -> frozenset[str]:
    configured = getattr(config, "review_bot_logins", ()) or ()
    return frozenset(login.lower() for login in (*DEFAULT_REVIEW_BOTS, *configured) if login)


def _is_review_bot_login(login: str, review_bots: frozenset[str]) -> bool:
    lowered = (login or "").strip().lower()
    if not lowered:
        return False
    if lowered in review_bots:
        return True
    return "bugbot" in lowered or "greptile" in lowered or "veria" in lowered


def _author_login_and_type(author: Any) -> tuple[str, str]:
    if not isinstance(author, dict):
        return "", ""
    login = str(author.get("login") or "").strip()
    typename = str(author.get("__typename") or author.get("type") or "").strip()
    return login, typename


def is_community_pr(
    pr: dict[str, Any],
    maintainer_logins: tuple[str, ...] | frozenset[str],
    *,
    org_members: frozenset[str] = frozenset(),
    org: str = "",
) -> bool:
    """True when the PR author is an external human community contributor."""
    login, author_type = _author_login_and_type(pr.get("author"))
    if not login:
        return False
    if _bot_login(login, author_type):
        return False
    lowered = login.lower()
    maintainers = {item.lower() for item in maintainer_logins if item}
    if lowered in maintainers:
        return False
    if org_members and lowered in org_members:
        return False
    if _org_staff_login(login, org):
        return False
    association = str(pr.get("authorAssociation") or pr.get("author_association") or "").upper()
    if association in EXCLUDED_AUTHOR_ASSOCIATIONS:
        return False
    return True


def classify_pr_status(
    pr: dict[str, Any],
    *,
    review_bots: frozenset[str] = frozenset(DEFAULT_REVIEW_BOTS),
) -> str:
    """Map a GitHub pull request GraphQL node to its single-select board Status."""
    state = str(pr.get("state") or "").upper()
    if state == "MERGED" or pr.get("merged") is True:
        return STATUS_MERGED
    if state == "CLOSED":
        return STATUS_CLOSED

    decision = str(pr.get("reviewDecision") or "").upper()
    if decision == "CHANGES_REQUESTED":
        return STATUS_CHANGES_REQUESTED
    if decision == "APPROVED":
        return STATUS_APPROVED

    pr_author, _ = _author_login_and_type(pr.get("author"))
    pr_author_lower = pr_author.lower()

    reviews_node = pr.get("reviews")
    review_nodes = (
        reviews_node.get("nodes")
        if isinstance(reviews_node, dict)
        else reviews_node
    ) or ()
    latest_human_state: dict[str, str] = {}
    has_any_review = False
    for review in review_nodes:
        if not isinstance(review, dict):
            continue
        r_state = str(review.get("state") or "").upper()
        if not r_state or r_state == "PENDING":
            continue
        r_login, r_type = _author_login_and_type(review.get("author"))
        r_lower = r_login.lower()
        if r_lower and r_lower != pr_author_lower:
            has_any_review = True
        if not r_login or r_lower == pr_author_lower:
            continue
        if _bot_login(r_login, r_type) or _is_review_bot_login(r_login, review_bots):
            continue
        if r_state in {"CHANGES_REQUESTED", "APPROVED"}:
            latest_human_state[r_lower] = r_state
        elif r_state == "DISMISSED":
            latest_human_state.pop(r_lower, None)

    if any(val == "CHANGES_REQUESTED" for val in latest_human_state.values()):
        return STATUS_CHANGES_REQUESTED
    if any(val == "APPROVED" for val in latest_human_state.values()):
        return STATUS_APPROVED
    if has_any_review:
        return STATUS_IN_REVIEW

    comments_node = pr.get("comments")
    comment_nodes = (
        comments_node.get("nodes")
        if isinstance(comments_node, dict)
        else comments_node
    ) or ()
    for comment in comment_nodes:
        if not isinstance(comment, dict):
            continue
        c_login, _ = _author_login_and_type(comment.get("author"))
        if _is_review_bot_login(c_login, review_bots):
            return STATUS_IN_REVIEW

    return STATUS_OPEN


def _run_graphql(
    graphql_fn,
    token: str | None,
    query: str,
    variables: dict[str, Any],
    *,
    timeout: float = BOARD_GRAPHQL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if graphql_fn is github_graphql:
        payload = github_graphql(
            token,
            query,
            variables,
            timeout=timeout,
            attempts=2,
            max_retry_delay=5.0,
        )
    else:
        payload = graphql_fn(token, query, variables)
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub GraphQL returned an unexpected payload")
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        message = first.get("message") if isinstance(first, dict) else str(first)
        raise RuntimeError(f"GitHub GraphQL error: {message}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("GitHub GraphQL response missing data")
    return data


def _extract_rate_remaining(data: dict[str, Any]) -> int | None:
    rl = data.get("rateLimit")
    if isinstance(rl, dict) and isinstance(rl.get("remaining"), int):
        return rl["remaining"]
    return None


def _parse_project_node(project: dict[str, Any], status_field_name: str) -> ProjectMetadata:
    project_id = str(project.get("id") or "").strip()
    if not project_id:
        raise RuntimeError("GitHub ProjectV2 node is missing an id")
    title = str(project.get("title") or "").strip()
    url = str(project.get("url") or "").strip()
    fields = ((project.get("fields") or {}).get("nodes") or ())
    target_lower = (status_field_name or "Status").strip().lower()
    status_field_id = ""
    raw_options: dict[str, str] = {}
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        if name.lower() != target_lower:
            continue
        if not isinstance(field.get("options"), list):
            continue
        status_field_id = str(field.get("id") or "").strip()
        for option in field["options"]:
            if isinstance(option, dict) and option.get("id") and option.get("name"):
                raw_options[str(option["name"]).strip().lower()] = str(option["id"]).strip()
        break
    if not status_field_id:
        raise RuntimeError(
            f"GitHub ProjectV2 {project_id} has no single-select field named '{status_field_name}'"
        )
    canonical_options: dict[str, str] = {}
    for label in BOARD_STATUSES:
        option_id = raw_options.get(label.lower())
        if option_id:
            canonical_options[label] = option_id
    return ProjectMetadata(
        project_id=project_id,
        title=title,
        url=url,
        status_field_id=status_field_id,
        status_options=canonical_options,
    )


def resolve_project_metadata(
    config,
    token: str,
    *,
    graphql=None,
    clock=time.monotonic,
    ttl_seconds: float = PROJECT_METADATA_TTL_SECONDS,
) -> ProjectMetadata:
    """Resolve ProjectV2 ID, URL, and Status single-select option IDs (cached in memory)."""
    cache_key = (
        config.pr_board_project_id,
        config.pr_board_owner,
        config.pr_board_number,
        config.pr_board_status_field,
    )
    now_mono = clock()
    if graphql is None:
        with _meta_lock:
            cached = _project_meta_cache.get(cache_key)
            if cached is not None and now_mono < cached[1]:
                return cached[0]

    gql = graphql or github_graphql
    project: dict[str, Any] | None = None
    if config.pr_board_project_id:
        data = _run_graphql(gql, token, _PROJECT_BY_ID_QUERY, {"projectId": config.pr_board_project_id})
        node = data.get("node")
        if isinstance(node, dict) and node.get("id"):
            project = node
    else:
        owner = config.pr_board_owner
        number = config.pr_board_number
        org_error: Exception | None = None
        try:
            data = _run_graphql(gql, token, _ORG_PROJECT_QUERY, {"owner": owner, "number": number})
            candidate = (data.get("organization") or {}).get("projectV2")
            if isinstance(candidate, dict) and candidate.get("id"):
                project = candidate
        except Exception as error:
            org_error = error
        if project is None:
            try:
                data = _run_graphql(gql, token, _USER_PROJECT_QUERY, {"owner": owner, "number": number})
                candidate = (data.get("user") or {}).get("projectV2")
                if isinstance(candidate, dict) and candidate.get("id"):
                    project = candidate
            except Exception as user_error:
                if org_error is not None:
                    raise RuntimeError(
                        f"Could not resolve GitHub ProjectV2 {owner}#{number} as org ({org_error}) or user ({user_error})"
                    ) from user_error
                raise

    if project is None:
        target = config.pr_board_project_id or f"{config.pr_board_owner}#{config.pr_board_number}"
        raise RuntimeError(
            f"GitHub ProjectV2 '{target}' not found or GITHUB_ACTOR_TOKEN lacks 'project' scope"
        )
    meta = _parse_project_node(project, config.pr_board_status_field)
    if graphql is None:
        with _meta_lock:
            _project_meta_cache[cache_key] = (meta, now_mono + ttl_seconds)
    return meta


def _fetch_pulls_batch(
    gql,
    token: str,
    owner: str,
    name: str,
    states: list[str],
    *,
    max_pages: int,
    since_cutoff: datetime | None = None,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    pulls: list[dict[str, Any]] = []
    cursor: str | None = None
    rate_limited = False
    for _page in range(max_pages):
        if deadline is not None and time.monotonic() >= deadline:
            break
        req_timeout = (
            min(BOARD_GRAPHQL_TIMEOUT_SECONDS, max(3.0, deadline - time.monotonic()))
            if deadline is not None
            else BOARD_GRAPHQL_TIMEOUT_SECONDS
        )
        data = _run_graphql(
            gql,
            token,
            _PULLS_QUERY,
            {"owner": owner, "name": name, "states": states, "after": cursor},
            timeout=req_timeout,
        )
        remaining = _extract_rate_remaining(data)
        if remaining is not None and remaining < BOARD_MIN_RATE_LIMIT_REMAINING:
            logging.warning(
                "PR board GraphQL rate limit low (remaining=%d); stopping pagination early",
                remaining,
            )
            rate_limited = True
        repo = data.get("repository") or {}
        conn = repo.get("pullRequests") or {}
        nodes = conn.get("nodes") or ()
        stop_for_cutoff = False
        for node in nodes:
            if not isinstance(node, dict) or not node.get("id") or not isinstance(node.get("number"), int):
                continue
            if since_cutoff is not None and isinstance(node.get("updatedAt"), str):
                try:
                    updated = parse_time(node["updatedAt"])
                    if updated < since_cutoff:
                        stop_for_cutoff = True
                        break
                except Exception:
                    pass
            pulls.append(node)
        page_info = conn.get("pageInfo") or {}
        if rate_limited or stop_for_cutoff or not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            break
    return pulls, rate_limited


def sync_pr_board(
    config,
    store,
    *,
    graphql=None,
    org_members_fetcher=None,
    now: datetime | None = None,
    deadline_seconds: float | None = BOARD_SYNC_DEADLINE_SECONDS,
    open_pages: int = DEFAULT_OPEN_PAGES,
    closed_pages: int = DEFAULT_CLOSED_PAGES,
) -> SyncResult | None:
    """Idempotently sync open and recently updated community PRs to GitHub Projects (v2)."""
    if not config.pr_board_enabled or store is None:
        return None
    token = config.github_actor_token
    if not token:
        logging.warning(
            "Contributor PR board sync skipped: GITHUB_ACTOR_TOKEN is unset "
            "(requires a token with 'project' scope)"
        )
        return None
    if not _sync_lock.acquire(blocking=False):
        logging.info("Contributor PR board sync skipped: another sync is already in progress")
        return None

    t_start = time.monotonic()
    deadline = (t_start + deadline_seconds) if deadline_seconds is not None else None
    moment = now or datetime.now(UTC)
    moment_iso = iso(moment)
    gql = graphql or github_graphql
    try:
        meta = resolve_project_metadata(config, token, graphql=graphql)
        store.set_pr_board_meta("project_id", meta.project_id, moment_iso)
        if meta.url:
            store.set_pr_board_meta("project_url", meta.url, moment_iso)
        if meta.title:
            store.set_pr_board_meta("project_title", meta.title, moment_iso)

        repo = config.github_repository
        owner, name = repo.split("/", 1)
        if org_members_fetcher is not None:
            org_members = frozenset(m.lower() for m in org_members_fetcher(owner, token))
        else:
            org_members = fetch_org_public_members(owner, token)

        last_synced_raw = store.get_pr_board_meta("last_synced_at")
        closed_cutoff = moment - timedelta(days=RECENT_CLOSED_LOOKBACK_DAYS)
        if last_synced_raw:
            try:
                parsed_last = parse_time(last_synced_raw) - timedelta(hours=1)
                if parsed_last > closed_cutoff:
                    closed_cutoff = parsed_last
            except Exception:
                pass

        open_pulls, rl_open = _fetch_pulls_batch(
            gql, token, owner, name, ["OPEN"], max_pages=open_pages, deadline=deadline,
        )
        closed_pulls: list[dict[str, Any]] = []
        rl_closed = False
        if not rl_open and (deadline is None or time.monotonic() < deadline):
            closed_pulls, rl_closed = _fetch_pulls_batch(
                gql,
                token,
                owner,
                name,
                ["MERGED", "CLOSED"],
                max_pages=closed_pages,
                since_cutoff=closed_cutoff,
                deadline=deadline,
            )

        deduped_by_number: dict[int, dict[str, Any]] = {}
        for pr in (*open_pulls, *closed_pulls):
            deduped_by_number.setdefault(int(pr["number"]), pr)

        all_pulls = list(deduped_by_number.values())
        community_pulls = [
            pr for pr in all_pulls
            if is_community_pr(
                pr,
                config.maintainer_logins,
                org_members=org_members,
                org=owner,
            )
        ]

        review_bots = _review_bot_set(config)
        added = 0
        updated = 0
        unchanged = 0
        skipped_missing_option = 0
        rate_limited = rl_open or rl_closed

        for pr in community_pulls:
            if deadline is not None and time.monotonic() >= deadline:
                logging.warning(
                    "PR board sync reached %.0fs deadline after %d/%d community PRs; resuming next cycle",
                    deadline_seconds or 0.0,
                    added + updated + unchanged,
                    len(community_pulls),
                )
                break
            number = int(pr["number"])
            node_id = str(pr["id"])
            updated_at = str(pr.get("updatedAt") or "")
            target_status = classify_pr_status(pr, review_bots=review_bots)
            cached = store.get_pr_board_item(meta.project_id, repo, number)

            if cached is not None and cached["status"] == target_status:
                unchanged += 1
                if cached["pr_updated_at"] != updated_at:
                    store.upsert_pr_board_item(
                        meta.project_id,
                        repo,
                        number,
                        node_id,
                        cached["item_id"],
                        target_status,
                        updated_at,
                        moment_iso,
                    )
                continue

            item_id = cached["item_id"] if cached is not None else ""
            if not item_id:
                req_timeout = (
                    min(BOARD_GRAPHQL_TIMEOUT_SECONDS, max(3.0, deadline - time.monotonic()))
                    if deadline is not None
                    else BOARD_GRAPHQL_TIMEOUT_SECONDS
                )
                add_data = _run_graphql(
                    gql,
                    token,
                    _ADD_ITEM_MUTATION,
                    {"projectId": meta.project_id, "contentId": node_id},
                    timeout=req_timeout,
                )
                item_id = str(((add_data.get("addProjectV2ItemById") or {}).get("item") or {}).get("id") or "").strip()
                if not item_id:
                    raise RuntimeError(f"addProjectV2ItemById returned no item id for {repo}#{number}")
                added += 1
                # Persist item_id immediately so retries never need to re-add.
                store.upsert_pr_board_item(
                    meta.project_id,
                    repo,
                    number,
                    node_id,
                    item_id,
                    "",
                    updated_at,
                    moment_iso,
                )

            option_id = meta.status_options.get(target_status)
            if not option_id:
                skipped_missing_option += 1
                logging.warning(
                    "PR board project %s is missing Status option '%s' (skipping status update for %s#%d)",
                    meta.project_id,
                    target_status,
                    repo,
                    number,
                )
                continue

            req_timeout = (
                min(BOARD_GRAPHQL_TIMEOUT_SECONDS, max(3.0, deadline - time.monotonic()))
                if deadline is not None
                else BOARD_GRAPHQL_TIMEOUT_SECONDS
            )
            _run_graphql(
                gql,
                token,
                _UPDATE_STATUS_MUTATION,
                {
                    "projectId": meta.project_id,
                    "itemId": item_id,
                    "fieldId": meta.status_field_id,
                    "optionId": option_id,
                },
                timeout=req_timeout,
            )
            if cached is not None:
                updated += 1
            store.upsert_pr_board_item(
                meta.project_id,
                repo,
                number,
                node_id,
                item_id,
                target_status,
                updated_at,
                moment_iso,
            )

        store.set_pr_board_meta("last_synced_at", moment_iso, moment_iso)
        store.set_pr_board_meta("last_error", "", moment_iso)
        logging.info(
            "PR board sync complete for %s (project=%s): scanned=%d community=%d "
            "added=%d updated=%d unchanged=%d missing_option=%d in %.2fs",
            repo,
            meta.project_id,
            len(all_pulls),
            len(community_pulls),
            added,
            updated,
            unchanged,
            skipped_missing_option,
            time.monotonic() - t_start,
        )
        return SyncResult(
            project_id=meta.project_id,
            project_url=meta.url,
            scanned=len(all_pulls),
            community=len(community_pulls),
            added=added,
            updated=updated,
            unchanged=unchanged,
            skipped_missing_option=skipped_missing_option,
            rate_limited=rate_limited,
        )
    except Exception as error:
        try:
            store.set_pr_board_meta("last_error", str(error)[:300], moment_iso)
        except Exception:
            pass
        raise
    finally:
        _sync_lock.release()


def format_pr_board_summary(config, store) -> str:
    """Format the ephemeral /pr-board reply from SQLite state (zero network calls)."""
    if not config.pr_board_enabled:
        return "contributor pr board is disabled (set PR_BOARD_PROJECT_ID or PR_BOARD_OWNER + PR_BOARD_NUMBER)."
    project_id = (store.get_pr_board_meta("project_id") if store is not None else None) or config.pr_board_project_id or ""
    project_url = (store.get_pr_board_meta("project_url") if store is not None else None) or ""
    if not project_url and config.pr_board_owner and config.pr_board_number is not None:
        project_url = f"https://github.com/orgs/{config.pr_board_owner}/projects/{config.pr_board_number}"
    last_synced = store.get_pr_board_meta("last_synced_at") if store is not None else None
    last_error = store.get_pr_board_meta("last_error") if store is not None else None

    counts = (
        store.pr_board_status_counts(project_id, config.github_repository)
        if store is not None and project_id
        else {}
    )
    total = sum(counts.values())

    lines = ["**Contributor PR backlog board**"]
    if project_url:
        lines.append(f"<{project_url}>")
    elif project_id:
        lines.append(f"Project ID: `{project_id}`")

    if total == 0 and not last_synced:
        lines.append("Initial sync has not completed yet.")
    else:
        for label in BOARD_STATUSES:
            lines.append(f"- {label}: **{counts.get(label, 0)}**")
        other = sum(v for k, v in counts.items() if k not in BOARD_STATUSES)
        if other:
            lines.append(f"- Unassigned: **{other}**")
        lines.append(f"Total tracked community PRs: **{total}**")
        if last_synced:
            lines.append(f"Last synced: `{last_synced}`")

    if last_error:
        lines.append(f"Last sync error: `{last_error[:180]}`")
    return "\n".join(lines)
