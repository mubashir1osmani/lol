"""Automatic contributor-role polling and historical backfill without OAuth or webhooks.

Discovers merged PRs on the configured repository via GitHub search, stores
number/author/merged_at/labels/changed files in SQLite, resolves community PR
authors to Discord members with the same matcher as the weekly summary
(exact_login, TypeSafe Jev, deterministic fallback, 30-day SQLite cache), and
assigns earned roles from ``roles.py``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .api import (
    TransientAPIError,
    fetch_author_merged_count,
    fetch_merged_numbers,
    fetch_pull_files,
    fetch_pull_request,
    github_get_json,
    post_message,
)
from .config import Config
from .matcher import match_community_contributors_with_diagnostics
from .rewards import _label_names, _models, apply_roles
from .roles import _agent_label, qualifies_agent_builder, qualifies_provider_wizard
from .schedule import UTC, iso, parse_time
from .store import Store
from .weekly import (
    EXCLUDED_AUTHOR_ASSOCIATIONS,
    _bot_login,
    _org_staff_login,
    fetch_org_public_members,
)

POLL_INITIAL_LOOKBACK_HOURS = 24
POLL_OVERLAP_MINUTES = 5
POLL_MAX_LOOKBACK_DAYS = 7
POLL_DEADLINE_SECONDS = 90.0
POLL_MAX_PULLS_PER_TICK = 50

AUTHOR_COUNT_CACHE_TTL_DAYS = 30

BACKFILL_WINDOW = timedelta(days=28)
BACKFILL_DEADLINE_SECONDS = 300.0
BACKFILL_MATCH_RESERVE_SECONDS = 45.0

_roles_poll_lock = threading.Lock()
_roles_backfill_lock = threading.Lock()


@dataclass(frozen=True)
class RolesPollResult:
    discovered: int
    stored: int
    skipped_existing: int
    skipped_excluded: int
    authors_evaluated: int
    authors_matched: int
    roles_granted: int
    files_skipped: int = 0
    rate_limited: bool = False
    timed_out: bool = False


@dataclass(frozen=True)
class BackfillResult:
    windows_completed: int
    windows_remaining: int
    pulls_stored: int
    authors_evaluated: int
    authors_matched: int
    contributors_granted: int
    roles_granted: int
    completed: bool
    summary_announced: bool
    rate_limited: bool = False
    timed_out: bool = False


def _checkpoint_key(repository: str) -> str:
    return f"last_merged_checkpoint:{repository.lower()}"


def _boot_backfill_key(repository: str) -> str:
    return f"backfill_on_boot_done:{repository.lower()}"


def should_run_boot_backfill(config: Config, store: Store | None) -> bool:
    """True when ROLES_BACKFILL_ON_BOOT=true and the one-time backfill has not finished yet."""
    if not config.contributor_roles_enabled or not config.roles_backfill_on_boot or store is None:
        return False
    return not store.get_roles_meta(_boot_backfill_key(config.github_repository))


def _is_excluded_author(
    login: str,
    user_type: str,
    author_association: str,
    maintainer_logins: tuple[str, ...] | frozenset[str],
    *,
    org_members: frozenset[str] = frozenset(),
    org: str = "",
) -> bool:
    if not login:
        return True
    if _bot_login(login, user_type):
        return True
    lowered = login.strip().lower()
    if lowered in {m.lower() for m in maintainer_logins if m}:
        return True
    if org_members and lowered in org_members:
        return True
    if _org_staff_login(login, org):
        return True
    if (author_association or "").upper() in EXCLUDED_AUTHOR_ASSOCIATIONS:
        return True
    return False


def _count_is_fresh(fetched_at: str | None, now: datetime) -> bool:
    if not fetched_at:
        return False
    try:
        recorded = parse_time(fetched_at)
    except Exception:
        return False
    return (now - recorded) <= timedelta(days=AUTHOR_COUNT_CACHE_TTL_DAYS)


def ensure_author_merged_count(
    config: Config,
    store: Store,
    login: str,
    user_id: int,
    *,
    now: datetime,
    count_fetcher=None,
) -> int | None:
    """Return cached lifetime merged PR total_count, fetching once via GitHub search on miss."""
    repo = config.github_repository
    cached = store.get_author_merged_count(repo, login)
    if cached is not None and _count_is_fresh(cached.get("fetched_at"), now):
        return int(cached["total_count"])
    fetcher = count_fetcher or fetch_author_merged_count
    try:
        total = int(fetcher(repo, login, config.effective_github_token))
    except (TransientAPIError, RuntimeError, ValueError) as error:
        logging.warning(
            "Could not fetch lifetime merged PR count for %s (%s); using stored rows",
            login,
            error,
        )
        return int(cached["total_count"]) if cached is not None else None
    stored_count = len(store.pulls_for_author(repo, user_id))
    effective = max(total, stored_count)
    store.set_author_merged_count(
        repo,
        login,
        effective,
        iso(now),
        github_user_id=user_id,
    )
    return effective


def _needs_pull_files(
    store: Store,
    repository: str,
    user_id: int,
    login: str,
    labels: list[str],
) -> bool:
    """True only when changed file paths could still unlock Agent Builder or Provider Wizard."""
    granted: set[str] = set()
    link = store.link_for_github_user(user_id)
    if link and link.get("discord_id"):
        granted = store.grant_keys(str(link["discord_id"]))
    else:
        matched = store.get_confirmed_match(login)
        if matched and matched.get("discord_id"):
            granted = store.grant_keys(str(matched["discord_id"]))

    if "agent_builder" in granted and "provider_wizard" in granted:
        return False

    existing_models = _models(store.pulls_for_author(repository, user_id))
    has_agent = (
        ("agent_builder" in granted)
        or _agent_label(tuple(labels))
        or qualifies_agent_builder(existing_models)
    )
    has_provider = (
        ("provider_wizard" in granted)
        or qualifies_provider_wizard(existing_models)
    )
    return not (has_agent and has_provider)


def _ingest_community_pull(
    config: Config,
    store: Store,
    number: int,
    *,
    org: str,
    org_members: frozenset[str],
    pull_fetcher=None,
    files_fetcher=None,
) -> tuple[dict[str, Any] | None, bool]:
    """Fetch and store one merged PR if authored by an external human contributor.

    Fetches changed file paths only when needed for Agent Builder or Provider
    Wizard, and incrementally updates any cached lifetime merged PR count.
    """
    get_pr = pull_fetcher or fetch_pull_request
    get_files = files_fetcher or fetch_pull_files
    token = config.effective_github_token
    pull = get_pr(config.github_repository, number, token)
    user = pull.get("user") if isinstance(pull.get("user"), dict) else {}
    if not pull.get("merged"):
        return None, True
    login = user.get("login")
    user_id = user.get("id")
    user_type = str(user.get("type") or "")
    association = str(pull.get("author_association") or "")
    merged_raw = pull.get("merged_at")
    if not isinstance(login, str) or not isinstance(user_id, int) or user_id <= 0 or not isinstance(merged_raw, str):
        return None, True
    if _is_excluded_author(
        login,
        user_type,
        association,
        config.maintainer_logins,
        org_members=org_members,
        org=org,
    ):
        return None, True
    labels = _label_names(pull)
    need_files = _needs_pull_files(store, config.github_repository, user_id, login, labels)
    files = get_files(config.github_repository, number, token) if need_files else []
    merged_iso = iso(parse_time(merged_raw))
    title = pull.get("title") if isinstance(pull.get("title"), str) else ""
    already_existed = store.get_pull(config.github_repository, number) is not None
    store.upsert_pull(
        repository=config.github_repository,
        number=number,
        github_user_id=user_id,
        github_login=login,
        merged_at=merged_iso,
        title=title,
        labels=labels,
        files=files,
    )
    if not already_existed:
        cached = store.get_author_merged_count(config.github_repository, login)
        if cached is not None and merged_iso > str(cached.get("fetched_at") or ""):
            store.increment_author_merged_count(config.github_repository, login, delta=1)
    return {
        "user_id": user_id,
        "login": login,
        "number": number,
        "merged_at": merged_iso,
        "files_fetched": need_files,
    }, False


def _match_authors(
    authors: list[dict[str, Any]],
    config: Config,
    store: Store,
    *,
    guild=None,
    loop=None,
    now: datetime,
    members=None,
    github_getter=None,
    jev_transport=None,
    haiku_complete=None,
    rest_fetcher=None,
    rest_searcher=None,
    deadline: float | None = None,
) -> tuple[dict[int, str], int]:
    """Resolve ``[{"user_id": int, "login": str, "pulls": [...]}]`` to ``{user_id: discord_id}``."""
    if not authors:
        return {}, 0
    matched_groups, _skipped, _details, diagnostics = match_community_contributors_with_diagnostics(
        authors,
        config,
        store,
        guild=guild,
        loop=loop,
        now=now,
        members=members,
        github_getter=github_getter or github_get_json,
        jev_transport=jev_transport,
        haiku_complete=haiku_complete,
        rest_fetcher=rest_fetcher,
        rest_searcher=rest_searcher,
        deadline=deadline,
    )
    resolved: dict[int, str] = {}
    for group in matched_groups:
        uid = group.get("user_id")
        discord_id = group.get("discord_id")
        if isinstance(uid, int) and uid > 0 and isinstance(discord_id, str) and discord_id:
            resolved[uid] = discord_id
    logging.info(
        "Contributor role auto-match: matched %d/%d community authors (members=%d, source=%s, llm=%s)",
        len(resolved),
        len(authors),
        diagnostics.member_count,
        diagnostics.members_source,
        diagnostics.llm_source,
    )
    return resolved, len(resolved)


def poll_merged_prs_for_roles(
    config: Config,
    store: Store,
    *,
    guild=None,
    loop=None,
    now: datetime | None = None,
    deadline_seconds: float | None = POLL_DEADLINE_SECONDS,
    max_pulls: int = POLL_MAX_PULLS_PER_TICK,
    search_merged_fn=None,
    pull_fetcher=None,
    files_fetcher=None,
    count_fetcher=None,
    org_members_fetcher=None,
    members=None,
    github_getter=None,
    jev_transport=None,
    haiku_complete=None,
    rest_fetcher=None,
    rest_searcher=None,
) -> RolesPollResult | None:
    """Poll newly merged PRs since the SQLite checkpoint and auto-assign newly earned roles."""
    if not config.contributor_roles_enabled or store is None:
        return None
    if not _roles_poll_lock.acquire(blocking=False):
        logging.info("Contributor roles poll skipped: another poll is already in progress")
        return None

    t_start = time.monotonic()
    deadline = (t_start + deadline_seconds) if deadline_seconds is not None else None
    moment = now or datetime.now(UTC)
    moment_iso = iso(moment)
    repo = config.github_repository
    org = repo.split("/", 1)[0] if "/" in repo else ""
    token = config.effective_github_token
    search_fn = search_merged_fn or fetch_merged_numbers
    try:
        checkpoint_raw = store.get_roles_meta(_checkpoint_key(repo))
        if checkpoint_raw:
            try:
                parsed = parse_time(checkpoint_raw)
                window_start = max(
                    parsed - timedelta(minutes=POLL_OVERLAP_MINUTES),
                    moment - timedelta(days=POLL_MAX_LOOKBACK_DAYS),
                )
            except Exception:
                window_start = moment - timedelta(hours=POLL_INITIAL_LOOKBACK_HOURS)
        else:
            window_start = moment - timedelta(hours=POLL_INITIAL_LOOKBACK_HOURS)
        if window_start >= moment:
            window_start = moment - timedelta(minutes=POLL_OVERLAP_MINUTES)

        if org_members_fetcher is not None:
            org_members = frozenset(m.lower() for m in org_members_fetcher(org, token))
        else:
            org_members = fetch_org_public_members(org, token) if org else frozenset()

        try:
            numbers = search_fn(repo, window_start, moment, token)
        except TransientAPIError as error:
            logging.warning(
                "Contributor roles poll deferred due to GitHub rate limit (%s); checkpoint unchanged",
                error,
            )
            return RolesPollResult(
                discovered=0,
                stored=0,
                skipped_existing=0,
                skipped_excluded=0,
                authors_evaluated=0,
                authors_matched=0,
                roles_granted=0,
                rate_limited=True,
            )

        stored = 0
        skipped_existing = 0
        skipped_excluded = 0
        files_skipped = 0
        rate_limited = False
        timed_out = False
        affected_authors: dict[int, dict[str, Any]] = {}
        fetch_deadline = (deadline - 20.0) if deadline is not None else None

        for number in sorted(numbers):
            if store.get_pull(repo, number) is not None:
                skipped_existing += 1
                continue
            if stored >= max_pulls or (fetch_deadline is not None and time.monotonic() >= fetch_deadline):
                timed_out = True
                break
            try:
                info, excluded = _ingest_community_pull(
                    config,
                    store,
                    number,
                    org=org,
                    org_members=org_members,
                    pull_fetcher=pull_fetcher,
                    files_fetcher=files_fetcher,
                )
            except TransientAPIError as error:
                logging.warning(
                    "Contributor roles poll hit rate limit on PR #%d (%s); will resume next cycle",
                    number,
                    error,
                )
                rate_limited = True
                break
            except RuntimeError as error:
                logging.warning("Contributor roles poll could not ingest PR #%d: %s", number, error)
                continue
            if excluded or info is None:
                skipped_excluded += 1
                continue
            stored += 1
            if not info.get("files_fetched", True):
                files_skipped += 1
            uid = info["user_id"]
            entry = affected_authors.setdefault(
                uid,
                {"user_id": uid, "login": info["login"], "pulls": []},
            )
            entry["login"] = info["login"]
            entry["pulls"].append({"number": info["number"], "merged_at": info["merged_at"]})

        if not rate_limited and not timed_out:
            store.set_roles_meta(_checkpoint_key(repo), moment_iso, moment_iso)
        store.set_roles_meta(f"last_roles_poll_at:{repo.lower()}", moment_iso, moment_iso)

        author_list = list(affected_authors.values())
        resolved_map, matched_count = _match_authors(
            author_list,
            config,
            store,
            guild=guild,
            loop=loop,
            now=moment,
            members=members,
            github_getter=github_getter,
            jev_transport=jev_transport,
            haiku_complete=haiku_complete,
            rest_fetcher=rest_fetcher,
            rest_searcher=rest_searcher,
            deadline=deadline,
        )

        roles_granted = 0
        for uid, discord_id in resolved_map.items():
            login = affected_authors[uid]["login"]
            lifetime_count = ensure_author_merged_count(
                config,
                store,
                login,
                uid,
                now=moment,
                count_fetcher=count_fetcher,
            )
            try:
                granted = apply_roles(
                    config,
                    store,
                    discord_id,
                    uid,
                    shoutout=False,
                    announce=True,
                    lifetime_merged_count=lifetime_count,
                )
                roles_granted += len(granted)
            except TransientAPIError as error:
                logging.warning("Role assignment deferred for user %s: %s", discord_id, error)
                rate_limited = True
                break

        logging.info(
            "Contributor roles poll for %s complete: discovered=%d stored=%d existing=%d "
            "excluded=%d files_skipped=%d matched=%d/%d roles_granted=%d in %.2fs",
            repo,
            len(numbers),
            stored,
            skipped_existing,
            skipped_excluded,
            files_skipped,
            matched_count,
            len(author_list),
            roles_granted,
            time.monotonic() - t_start,
        )
        return RolesPollResult(
            discovered=len(numbers),
            stored=stored,
            skipped_existing=skipped_existing,
            skipped_excluded=skipped_excluded,
            authors_evaluated=len(author_list),
            authors_matched=matched_count,
            roles_granted=roles_granted,
            files_skipped=files_skipped,
            rate_limited=rate_limited,
            timed_out=timed_out,
        )
    finally:
        _roles_poll_lock.release()


def _backfill_summary_line(contributors_granted: int, roles_granted: int) -> str:
    person_word = "contributor" if contributors_granted == 1 else "contributors"
    role_word = "role" if roles_granted == 1 else "roles"
    return (
        f"backfilled contributor roles for {contributors_granted} {person_word} "
        f"({roles_granted} {role_word} granted) 🎉"
    )


def run_roles_backfill(
    config: Config,
    store: Store,
    *,
    guild=None,
    loop=None,
    now: datetime | None = None,
    deadline_seconds: float | None = BACKFILL_DEADLINE_SECONDS,
    search_merged_fn=None,
    pull_fetcher=None,
    files_fetcher=None,
    count_fetcher=None,
    org_members_fetcher=None,
    members=None,
    github_getter=None,
    jev_transport=None,
    haiku_complete=None,
    rest_fetcher=None,
    rest_searcher=None,
    announce_summary: bool = True,
) -> BackfillResult | None:
    """Rate-limit-aware, time-bounded historical backfill with auto-matching.

    Posts at most ONE summary line to ``ROLE_ANNOUNCEMENTS_CHANNEL_ID`` (never
    one message per contributor).
    """
    if not config.contributor_roles_enabled or store is None:
        return None
    if not _roles_backfill_lock.acquire(blocking=False):
        logging.info("Contributor roles backfill skipped: already running")
        return None

    t_start = time.monotonic()
    deadline = (t_start + deadline_seconds) if deadline_seconds is not None else None
    ingest_deadline = (
        max(t_start + 5.0, deadline - BACKFILL_MATCH_RESERVE_SECONDS)
        if deadline is not None
        else None
    )
    moment = now or datetime.now(UTC)
    moment_iso = iso(moment)
    repo = config.github_repository
    org = repo.split("/", 1)[0] if "/" in repo else ""
    token = config.effective_github_token
    search_fn = search_merged_fn or fetch_merged_numbers

    try:
        if org_members_fetcher is not None:
            org_members = frozenset(m.lower() for m in org_members_fetcher(org, token))
        else:
            org_members = fetch_org_public_members(org, token) if org else frozenset()

        windows: list[tuple[datetime, datetime]] = []
        cursor = parse_time(config.backfill_since)
        while cursor < moment:
            end = min(moment, cursor + BACKFILL_WINDOW)
            windows.append((cursor, end))
            cursor = end

        windows_completed = 0
        pending_windows = [
            (start, end)
            for start, end in windows
            if not store.backfill_done(repo, iso(start), iso(end))
        ]
        pulls_stored = 0
        rate_limited = False
        timed_out = False

        logging.info(
            "Starting contributor roles backfill for %s: %d pending windows (since %s)",
            repo,
            len(pending_windows),
            config.backfill_since,
        )

        for start, end in pending_windows:
            if ingest_deadline is not None and time.monotonic() >= ingest_deadline:
                timed_out = True
                break
            start_iso, end_iso = iso(start), iso(end)
            try:
                numbers = search_fn(repo, start, end, token)
            except TransientAPIError as error:
                logging.warning(
                    "Backfill search %s..%s hit GitHub rate limit (%s); stopping ingest early",
                    start_iso,
                    end_iso,
                    error,
                )
                rate_limited = True
                break

            window_stored = 0
            window_incomplete = False
            for number in sorted(numbers):
                if store.get_pull(repo, number) is not None:
                    continue
                if ingest_deadline is not None and time.monotonic() >= ingest_deadline:
                    timed_out = True
                    window_incomplete = True
                    break
                try:
                    info, excluded = _ingest_community_pull(
                        config,
                        store,
                        number,
                        org=org,
                        org_members=org_members,
                        pull_fetcher=pull_fetcher,
                        files_fetcher=files_fetcher,
                    )
                except TransientAPIError as error:
                    logging.warning(
                        "Backfill PR #%d hit GitHub rate limit (%s); stopping ingest early",
                        number,
                        error,
                    )
                    rate_limited = True
                    window_incomplete = True
                    break
                except RuntimeError as error:
                    logging.warning("Backfill skipped PR #%d due to error: %s", number, error)
                    continue
                if not excluded and info is not None:
                    window_stored += 1
                    pulls_stored += 1
                    if window_stored % 25 == 0:
                        logging.info(
                            "Backfill stored %d new PRs in %s..%s",
                            window_stored,
                            start_iso,
                            end_iso,
                        )
            if window_incomplete:
                break
            store.mark_backfill_done(repo, start_iso, end_iso, iso(datetime.now(UTC)))
            windows_completed += 1
            logging.info(
                "Backfill window %s..%s complete (%d new community PRs)",
                start_iso,
                end_iso,
                window_stored,
            )

        windows_remaining = max(0, len(pending_windows) - windows_completed)
        completed = windows_remaining == 0 and not rate_limited and not timed_out

        # Evaluate all community PR authors currently recorded in SQLite.
        stored_authors = store.all_pull_authors(repo)
        eligible_authors = [
            {
                "user_id": row["github_user_id"],
                "login": row["github_login"],
                "pulls": [],
            }
            for row in stored_authors
            if not _is_excluded_author(
                row["github_login"],
                "User",
                "",
                config.maintainer_logins,
                org_members=org_members,
                org=org,
            )
        ]

        resolved_map, matched_count = _match_authors(
            eligible_authors,
            config,
            store,
            guild=guild,
            loop=loop,
            now=moment,
            members=members,
            github_getter=github_getter,
            jev_transport=jev_transport,
            haiku_complete=haiku_complete,
            rest_fetcher=rest_fetcher,
            rest_searcher=rest_searcher,
            deadline=deadline,
        )

        login_by_uid = {a["user_id"]: a["login"] for a in eligible_authors}
        contributors_granted = 0
        roles_granted = 0
        for uid, discord_id in resolved_map.items():
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
            lifetime_count = ensure_author_merged_count(
                config,
                store,
                login_by_uid[uid],
                uid,
                now=moment,
                count_fetcher=count_fetcher,
            )
            try:
                newly = apply_roles(
                    config,
                    store,
                    discord_id,
                    uid,
                    shoutout=False,
                    announce=False,
                    lifetime_merged_count=lifetime_count,
                )
            except TransientAPIError as error:
                logging.warning("Backfill role assignment hit rate limit for %s: %s", discord_id, error)
                rate_limited = True
                break
            if newly:
                contributors_granted += 1
                roles_granted += len(newly)

        summary_announced = False
        if announce_summary and contributors_granted > 0 and config.role_announcements_channel_id:
            try:
                post_message(
                    config.role_announcements_channel_id,
                    config.discord_token,
                    {
                        "content": _backfill_summary_line(contributors_granted, roles_granted),
                        "allowed_mentions": {"parse": []},
                    },
                )
                summary_announced = True
            except Exception:
                logging.exception(
                    "Could not post backfill summary to channel %s",
                    config.role_announcements_channel_id,
                )

        if completed:
            store.set_roles_meta(_boot_backfill_key(repo), moment_iso, moment_iso)
            if not store.get_roles_meta(_checkpoint_key(repo)):
                store.set_roles_meta(_checkpoint_key(repo), moment_iso, moment_iso)

        logging.info(
            "Contributor roles backfill finished for %s: windows=%d/%d pulls_stored=%d "
            "matched=%d/%d granted=%d contributors (%d roles) completed=%s rate_limited=%s timed_out=%s in %.2fs",
            repo,
            windows_completed,
            len(pending_windows),
            pulls_stored,
            matched_count,
            len(eligible_authors),
            contributors_granted,
            roles_granted,
            completed,
            rate_limited,
            timed_out,
            time.monotonic() - t_start,
        )
        return BackfillResult(
            windows_completed=windows_completed,
            windows_remaining=windows_remaining,
            pulls_stored=pulls_stored,
            authors_evaluated=len(eligible_authors),
            authors_matched=matched_count,
            contributors_granted=contributors_granted,
            roles_granted=roles_granted,
            completed=completed,
            summary_announced=summary_announced,
            rate_limited=rate_limited,
            timed_out=timed_out,
        )
    finally:
        _roles_backfill_lock.release()


def format_backfill_result(result: BackfillResult | None) -> str:
    """Ephemeral moderator summary for /roles-backfill."""
    if result is None:
        return "contributor roles backfill is already running or contributor roles are disabled."
    status = "complete" if result.completed else "partial (run /roles-backfill again to continue)"
    reasons = []
    if result.rate_limited:
        reasons.append("rate-limited")
    if result.timed_out:
        reasons.append("time-bounded")
    suffix = f" ({', '.join(reasons)})" if reasons else ""
    lines = [
        f"**Contributor roles backfill: {status}**{suffix}",
        f"- Windows completed this run: **{result.windows_completed}** (remaining: **{result.windows_remaining}**)",
        f"- New community PRs stored: **{result.pulls_stored}**",
        f"- Community authors matched in Discord: **{result.authors_matched}/{result.authors_evaluated}**",
        f"- New role grants: **{result.roles_granted}** across **{result.contributors_granted}** contributors",
    ]
    if result.summary_announced:
        lines.append("- Posted a single summary line to the role announcements channel.")
    return "\n".join(lines)
