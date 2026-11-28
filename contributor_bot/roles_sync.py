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
from zoneinfo import ZoneInfo

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
from .rewards import (
    ROLE_ANNOUNCEMENT_CHUNK_LIMIT,
    _label_names,
    _models,
    announce_pending_role_grants,
    apply_roles,
)
from .roles import _agent_label, qualifies_agent_builder, qualifies_provider_wizard
from .schedule import UTC, iso, parse_time
from .store import Store
from .weekly import (
    EXCLUDED_AUTHOR_ASSOCIATIONS,
    WeeklyFetchError,
    _bot_login,
    _org_staff_login,
    _search_merged_pages,
    fetch_org_public_members,
)

DAILY_LOOKBACK_DAYS = 7
POLL_INITIAL_LOOKBACK_HOURS = 24
POLL_OVERLAP_MINUTES = 5
POLL_MAX_LOOKBACK_DAYS = 14
POLL_DEADLINE_SECONDS = 180.0
POLL_MAX_PULLS_PER_TICK = 150
ROLES_DAILY_RETRY_SECONDS = 300.0

AUTHOR_COUNT_CACHE_TTL_DAYS = 30

BACKFILL_WINDOW = timedelta(days=28)
BACKFILL_DEADLINE_SECONDS = 300.0
BACKFILL_MATCH_RESERVE_SECONDS = 45.0

_roles_poll_lock = threading.Lock()
_roles_daily_lock = threading.Lock()
_roles_daily_state_lock = threading.Lock()
_roles_daily_retry_after: dict[tuple[str, str], float] = {}
_roles_daily_last_skip_log: dict[tuple[str, str], str] = {}
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
    total_windows: int = 0
    windows_done_total: int = 0
    enrichment_deferred: int = 0


def _checkpoint_key(repository: str) -> str:
    return f"last_merged_checkpoint:{repository.lower()}"


def _boot_backfill_key(repository: str) -> str:
    return f"backfill_on_boot_done:{repository.lower()}"


def _backfill_anchor_key(repository: str) -> str:
    return f"backfill_anchor:{repository.lower()}"


def should_run_boot_backfill(config: Config, store: Store | None) -> bool:
    """True when ROLES_BACKFILL_ON_BOOT=true and the one-time backfill has not finished yet."""
    if not config.contributor_roles_enabled or not config.roles_backfill_on_boot or store is None:
        return False
    return not store.get_roles_meta(_boot_backfill_key(config.github_repository))


def has_pending_backfill(config: Config, store: Store | None) -> bool:
    """True when a resumable backfill cursor is still in progress for this repository."""
    if not config.contributor_roles_enabled or store is None:
        return False
    anchor = (store.get_roles_meta(_backfill_anchor_key(config.github_repository)) or "").strip()
    return bool(anchor) or should_run_boot_backfill(config, store)


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
    stored_count = (
        len(store.pulls_for_author(repo, user_id))
        if isinstance(user_id, int) and user_id > 0
        else len(store.pulls_for_author_login(repo, login))
    )
    effective = max(total, stored_count)
    store.set_author_merged_count(
        repo,
        login,
        effective,
        iso(now),
        github_user_id=user_id if isinstance(user_id, int) and user_id > 0 else None,
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


def _resolve_stored_discord_id(config: Config, store: Store, login: str, user_id: int | None) -> str | None:
    """Reuse an already-known Discord link/mapping/github_discord_matches row without calling any LLM."""
    if isinstance(user_id, int) and user_id > 0:
        link = store.link_for_github_user(user_id)
        if link and link.get("discord_id"):
            return str(link["discord_id"])
    login_key = (login or "").strip().lower()
    if login_key:
        mapped = (getattr(config, "github_discord_map", None) or {}).get(login_key)
        if mapped:
            return str(mapped)
        confirmed = store.get_confirmed_match(login_key)
        if confirmed and confirmed.get("discord_id"):
            return str(confirmed["discord_id"])
    return None


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
    org_members: frozenset[str] = frozenset(),
) -> tuple[dict[int, str], int, int]:
    """Resolve ``[{"user_id": int, "login": str, "pulls": [...]}]`` to ``(resolved, matched_count, enrichment_deferred)``.

    Reuses stored ``github_discord_matches`` / ``github_links`` / ``GITHUB_DISCORD_MAP``
    directly without running GitHub enrichment or the LLM.
    """
    if not authors:
        return {}, 0, 0

    resolved: dict[int, str] = {}
    reused_count = 0
    unmatched: list[dict[str, Any]] = []
    for author in authors:
        uid = author.get("user_id")
        login = str(author.get("login") or "")
        stored_discord = _resolve_stored_discord_id(config, store, login, uid if isinstance(uid, int) else None)
        if stored_discord and isinstance(uid, int) and uid > 0:
            resolved[uid] = stored_discord
            reused_count += 1
        else:
            unmatched.append(author)

    if not unmatched:
        logging.info(
            "Contributor role auto-match: reused %d/%d stored Discord matches (0 LLM calls)",
            reused_count,
            len(authors),
        )
        return resolved, len(resolved), 0

    matched_groups, _skipped, _details, diagnostics = match_community_contributors_with_diagnostics(
        unmatched,
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
        org_members=org_members,
    )
    for group in matched_groups:
        uid = group.get("user_id")
        discord_id = group.get("discord_id")
        if isinstance(uid, int) and uid > 0 and isinstance(discord_id, str) and discord_id:
            resolved[uid] = discord_id
    logging.info(
        "Contributor role auto-match: matched %d/%d community authors "
        "(reused_stored=%d, members=%d, source=%s, llm=%s, enrichment_deferred=%d)",
        len(resolved),
        len(authors),
        reused_count,
        diagnostics.member_count,
        diagnostics.members_source,
        diagnostics.llm_source,
        diagnostics.enrichment_deferred,
    )
    return resolved, len(resolved), diagnostics.enrichment_deferred


def _grant_roles_for_stored_matches(
    config: Config,
    store: Store,
    *,
    now: datetime,
    org: str,
    org_members: frozenset[str],
    matched_since: str | None = None,
    count_fetcher=None,
    deadline: float | None = None,
) -> tuple[int, bool]:
    """Ensure confirmed matches in ``github_discord_matches`` (incl. weekly shoutouts) get earned roles."""
    repo = config.github_repository
    stored_matches = store.all_confirmed_matches(matched_since=matched_since)
    if not stored_matches:
        return 0, False

    roles_granted = 0
    rate_limited = False
    for match in stored_matches:
        if deadline is not None and time.monotonic() >= deadline:
            break
        login = str(match.get("github_login") or "").strip()
        discord_id = str(match.get("discord_id") or "").strip()
        if not login or not discord_id:
            continue
        if _is_excluded_author(
            login,
            "User",
            "",
            config.maintainer_logins,
            org_members=org_members,
            org=org,
        ):
            continue
        raw_uid = match.get("github_user_id")
        uid = int(raw_uid) if isinstance(raw_uid, int) and raw_uid > 0 else 0
        pulls = store.pulls_for_author(repo, uid) if uid > 0 else store.pulls_for_author_login(repo, login)
        if uid == 0 and pulls:
            uid = int(pulls[-1]["github_user_id"])
        already = store.grant_keys(discord_id)
        cached_count = store.get_author_merged_count(repo, login)
        lifetime_count: int | None = None
        if cached_count is not None and _count_is_fresh(cached_count.get("fetched_at"), now):
            lifetime_count = int(cached_count["total_count"])
        elif "contributor" not in already or "oss_champion" not in already:
            lifetime_count = ensure_author_merged_count(
                config,
                store,
                login,
                uid,
                now=now,
                count_fetcher=count_fetcher,
            )
            if lifetime_count is None and "contributor" not in already:
                # Confirmed weekly-summary match has at least 1 merged PR even if GitHub count lookup fails.
                lifetime_count = max(1, len(pulls))
        try:
            granted = apply_roles(
                config,
                store,
                discord_id,
                uid,
                shoutout=False,
                announce=False,
                mark_announced=False,
                lifetime_merged_count=lifetime_count,
            )
            roles_granted += len(granted)
        except TransientAPIError as error:
            logging.warning("Stored-match role assignment deferred for user %s: %s", discord_id, error)
            rate_limited = True
            break
    return roles_granted, rate_limited


def _search_merged_numbers_per_day(
    repository: str,
    start: datetime,
    end: datetime,
    token: str | None,
) -> set[int]:
    """Discover merged PR numbers across per-day ``merged:D..D`` windows (cap 10 pages/day)."""
    try:
        rows, rate_limited = _search_merged_pages(
            repository,
            start,
            token,
            until=end,
            pace=True,
            max_pages=10,
        )
    except WeeklyFetchError as error:
        if error.rate_limited:
            raise TransientAPIError(str(error), retry_after=error.retry_after) from error
        raise RuntimeError(str(error)) from error
    if rate_limited:
        raise TransientAPIError("GitHub search rate limit hit during per-day merged PR fetch")
    return {int(row["number"]) for row in rows if isinstance(row.get("number"), int)}


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
    announcement_chunk_limit: int = ROLE_ANNOUNCEMENT_CHUNK_LIMIT,
) -> RolesPollResult | None:
    """Check PRs merged in the rolling past 7 days and auto-assign newly earned roles."""
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
    search_fn = search_merged_fn or _search_merged_numbers_per_day
    try:
        window_start = moment - timedelta(days=DAILY_LOOKBACK_DAYS)
        window_start_iso = iso(window_start)

        if org_members_fetcher is not None:
            org_members = frozenset(m.lower() for m in org_members_fetcher(org, token))
        else:
            org_members = fetch_org_public_members(org, token) if org else frozenset()

        # Grant roles for recent confirmed matches (such as Friday's weekly summary shoutouts) first.
        stored_roles_granted, stored_rate_limited = _grant_roles_for_stored_matches(
            config,
            store,
            now=moment,
            org=org,
            org_members=org_members,
            matched_since=window_start_iso,
            count_fetcher=count_fetcher,
            deadline=(deadline - 30.0) if deadline is not None else None,
        )

        try:
            numbers = search_fn(repo, window_start, moment, token)
        except TransientAPIError as error:
            if stored_roles_granted:
                try:
                    announce_pending_role_grants(
                        config,
                        store,
                        now=moment,
                        limit=announcement_chunk_limit,
                    )
                except TransientAPIError:
                    pass
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
                roles_granted=stored_roles_granted,
                rate_limited=True,
            )

        stored = 0
        skipped_existing = 0
        skipped_excluded = 0
        files_skipped = 0
        rate_limited = stored_rate_limited
        timed_out = False
        affected_authors: dict[int, dict[str, Any]] = {}
        fetch_deadline = (deadline - 20.0) if deadline is not None else None

        for number in sorted(numbers, reverse=True):
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
            if info["merged_at"] >= window_start_iso:
                uid = info["user_id"]
                entry = affected_authors.setdefault(
                    uid,
                    {"user_id": uid, "login": info["login"], "pulls": []},
                )
                entry["login"] = info["login"]
                entry["pulls"].append({"number": info["number"], "merged_at": info["merged_at"]})

        # Also evaluate already-stored PR authors from the rolling 7-day window (never older history)
        # so contributors who joined Discord within a few days of merging still get matched & graded.
        for row in store.all_pull_authors(repo, merged_since=window_start_iso):
            uid = row["github_user_id"]
            if uid in affected_authors:
                continue
            login = row["github_login"]
            if _is_excluded_author(
                login,
                "User",
                "",
                config.maintainer_logins,
                org_members=org_members,
                org=org,
            ):
                continue
            affected_authors[uid] = {"user_id": uid, "login": login, "pulls": []}

        if not rate_limited and not timed_out:
            store.set_roles_meta(_checkpoint_key(repo), moment_iso, moment_iso)
        store.set_roles_meta(f"last_roles_poll_at:{repo.lower()}", moment_iso, moment_iso)

        author_list = list(affected_authors.values())
        resolved_map, matched_count, _enrichment_deferred = _match_authors(
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
            org_members=org_members,
        )

        roles_granted = stored_roles_granted
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
                    announce=False,
                    mark_announced=False,
                    lifetime_merged_count=lifetime_count,
                )
                roles_granted += len(granted)
            except TransientAPIError as error:
                logging.warning("Role assignment deferred for user %s: %s", discord_id, error)
                rate_limited = True
                break

        try:
            announce_pending_role_grants(
                config,
                store,
                now=moment,
                limit=announcement_chunk_limit,
            )
        except TransientAPIError as error:
            logging.warning("Batched role announcement deferred due to Discord rate limit: %s", error)
            rate_limited = True

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


def daily_is_due(now: datetime, hhmm: str, timezone: str, last_date: str | None) -> bool:
    local = now.astimezone(ZoneInfo(timezone))
    hour, minute = (int(part) for part in hhmm.split(":", 1))
    if (local.hour, local.minute) < (hour, minute):
        return False
    return local.date().isoformat() != (last_date or "")


def reset_roles_daily_scheduler_state() -> None:
    """Clear in-memory daily roles retry backoff and skip-log deduplication (for tests)."""
    with _roles_daily_state_lock:
        _roles_daily_retry_after.clear()
        _roles_daily_last_skip_log.clear()


def _log_roles_daily_skip(
    repository: str,
    run_date: str,
    reason_code: str,
    message: str,
    *args,
) -> None:
    cache_key = (repository.lower(), run_date)
    with _roles_daily_state_lock:
        if _roles_daily_last_skip_log.get(cache_key) == reason_code:
            return
        _roles_daily_last_skip_log[cache_key] = reason_code
    logging.info(message, *args)


def _clear_roles_daily_skip(repository: str, run_date: str) -> None:
    with _roles_daily_state_lock:
        _roles_daily_last_skip_log.pop((repository.lower(), run_date), None)


def roles_daily_schedule_status(config: Config, store: Store | None, now: datetime | None = None) -> str:
    """Human-readable one-line status of the daily contributor-roles scheduler."""
    moment = now or datetime.now(UTC)
    local = moment.astimezone(ZoneInfo(config.merged_timezone))
    run_date = local.date().isoformat()
    info = store.roles_day_info(config.github_repository, run_date) if store is not None else None
    if not config.contributor_roles_enabled:
        state = "disabled"
    elif info is not None:
        completed_at = info.get("completed_at") or "unknown time"
        via = info.get("via") or "scheduler"
        state = f"already completed at {completed_at} via {via}"
    elif daily_is_due(moment, config.roles_daily_time, config.merged_timezone, None):
        state = "due now (will catch up)"
    else:
        state = f"waiting for {config.roles_daily_time} {config.merged_timezone}"
    return (
        f"daily {config.roles_daily_time} {config.merged_timezone}, "
        f"now={local.strftime('%Y-%m-%d %H:%M %Z')}, date={run_date}, "
        f"status={state}, oauth={config.github_oauth_enabled}, "
        f"webhook={bool(config.github_webhook_secret)}, backfill_on_boot={config.roles_backfill_on_boot}"
    )


def log_roles_daily_boot_status(
    config: Config,
    store: Store | None,
    now: datetime | None = None,
) -> dict | None:
    """Log clear startup status for the daily contributor-roles run."""
    if not config.contributor_roles_enabled:
        return None
    moment = now or datetime.now(UTC)
    local = moment.astimezone(ZoneInfo(config.merged_timezone))
    run_date = local.date().isoformat()
    info = store.roles_day_info(config.github_repository, run_date) if store is not None else None
    if info is not None:
        logging.info(
            "Contributor roles daily run %s already completed at %s via %s; skipping",
            info["run_date"],
            info.get("completed_at") or "unknown time",
            info.get("via") or "scheduler",
        )
    logging.info("Contributor roles daily schedule: %s", roles_daily_schedule_status(config, store, moment))
    return info


def run_daily_roles_if_due(
    config: Config,
    store: Store | None,
    now: datetime | None = None,
    *,
    guild=None,
    loop=None,
    force: bool = False,
    via: str | None = None,
    clock=time.monotonic,
    poll_fn=None,
    **poll_kwargs,
) -> RolesPollResult | None:
    """Run the contributor-roles check once per local day at ``ROLES_DAILY_TIME``.

    Uses its own independent SQLite dedupe table (``roles_daily_runs``) so
    restarts/redeploys never run or announce the same day twice, while catching
    up if the bot was down at ``ROLES_DAILY_TIME``. On Fridays, runs right after
    the weekly merge summary post but independently if the summary fails or skips.
    """
    if store is None or not config.contributor_roles_enabled:
        return None
    if not _roles_daily_lock.acquire(blocking=False):
        logging.info("Contributor roles daily check skipped: another daily roles run is already in progress")
        return None
    try:
        moment = now or datetime.now(UTC)
        local = moment.astimezone(ZoneInfo(config.merged_timezone))
        run_date = local.date().isoformat()
        repo = config.github_repository

        if not force:
            done_info = store.roles_day_info(repo, run_date)
            if done_info is not None:
                _log_roles_daily_skip(
                    repo,
                    run_date,
                    "already_completed",
                    "Contributor roles daily run %s already completed at %s via %s; skipping",
                    run_date,
                    done_info.get("completed_at") or "unknown time",
                    done_info.get("via") or "scheduler",
                )
                return None
            hour, minute = (int(part) for part in config.roles_daily_time.split(":", 1))
            if (local.hour, local.minute) < (hour, minute):
                _log_roles_daily_skip(
                    repo,
                    run_date,
                    "before_time",
                    "Contributor roles daily run waiting for %s: local time %s is before %s %s",
                    run_date,
                    local.strftime("%H:%M %Z"),
                    config.roles_daily_time,
                    config.merged_timezone,
                )
                return None
            if not daily_is_due(moment, config.roles_daily_time, config.merged_timezone, None):
                return None
            retry_key = (repo.lower(), run_date)
            with _roles_daily_state_lock:
                retry_deadline = _roles_daily_retry_after.get(retry_key, 0.0)
            if clock() < retry_deadline:
                remaining = int(max(1.0, retry_deadline - clock()))
                _log_roles_daily_skip(
                    repo,
                    run_date,
                    "rate_limit_backoff",
                    "Contributor roles daily run for %s deferred after rate limit; retrying in ~%ds",
                    run_date,
                    remaining,
                )
                return None

        _clear_roles_daily_skip(repo, run_date)
        logging.info(
            "Contributor roles daily run %s for %s (local=%s); checking merged PRs",
            "forced" if force else "due",
            run_date,
            local.strftime("%Y-%m-%d %H:%M %Z"),
        )
        runner = poll_fn or poll_merged_prs_for_roles
        result = runner(
            config,
            store,
            guild=guild,
            loop=loop,
            now=moment,
            **poll_kwargs,
        )
        if result is None:
            return None
        if result.rate_limited or result.timed_out:
            if result.rate_limited and not force:
                with _roles_daily_state_lock:
                    _roles_daily_retry_after[(repo.lower(), run_date)] = clock() + ROLES_DAILY_RETRY_SECONDS
            logging.warning(
                "Contributor roles daily run for %s incomplete (rate_limited=%s, timed_out=%s; "
                "%d PRs, %d new grants so far); day left unmarked to resume automatically",
                run_date,
                result.rate_limited,
                result.timed_out,
                result.stored,
                result.roles_granted,
            )
            return result

        with _roles_daily_state_lock:
            _roles_daily_retry_after.pop((repo.lower(), run_date), None)
        mark_via = via or ("manual" if force else "scheduler")
        store.mark_roles_day(
            repo,
            run_date,
            iso(moment),
            via=mark_via,
            prs_discovered=result.discovered,
            roles_granted=result.roles_granted,
        )
        logging.info(
            "Contributor roles daily run for %s: %d PRs, %d new grants",
            run_date,
            result.stored,
            result.roles_granted,
        )
        return result
    finally:
        _roles_daily_lock.release()


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

        anchor_key = _backfill_anchor_key(repo)
        raw_anchor = (store.get_roles_meta(anchor_key) or "").strip()
        anchor = moment
        if raw_anchor:
            try:
                parsed_anchor = parse_time(raw_anchor)
                if parsed_anchor > parse_time(config.backfill_since):
                    anchor = parsed_anchor
            except Exception:
                anchor = moment
        if not raw_anchor or anchor == moment:
            store.set_roles_meta(anchor_key, iso(anchor), moment_iso)

        windows: list[tuple[datetime, datetime]] = []
        cursor = parse_time(config.backfill_since)
        while cursor < anchor:
            end = min(anchor, cursor + BACKFILL_WINDOW)
            windows.append((cursor, end))
            cursor = end
        # Process NEWEST-FIRST so recent contributors receive roles first.
        windows.reverse()

        total_windows = len(windows)
        windows_completed = 0
        pending_windows = [
            (start, end)
            for start, end in windows
            if not store.backfill_done(repo, iso(start), iso(end))
        ]
        done_before = total_windows - len(pending_windows)
        pulls_stored = 0
        rate_limited = False
        timed_out = False

        logging.info(
            "Starting contributor roles backfill for %s (newest-first): "
            "%d/%d windows already done, %d pending (since %s)",
            repo,
            done_before,
            total_windows,
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
            for number in sorted(numbers, reverse=True):
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
                "Backfill window %s..%s complete (%d new community PRs; %d/%d windows done)",
                start_iso,
                end_iso,
                window_stored,
                done_before + windows_completed,
                total_windows,
            )

        windows_done_total = done_before + windows_completed
        windows_remaining = max(0, total_windows - windows_done_total)

        # Evaluate all community PR authors currently recorded in SQLite (newest merged_at first).
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

        resolved_map, matched_count, enrichment_deferred = _match_authors(
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
            org_members=org_members,
        )
        if enrichment_deferred > 0:
            timed_out = True

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

        completed = (
            windows_remaining == 0
            and enrichment_deferred == 0
            and not rate_limited
            and not timed_out
        )

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
            store.set_roles_meta(anchor_key, "", moment_iso)
            store.set_roles_meta(_boot_backfill_key(repo), moment_iso, moment_iso)
            if not store.get_roles_meta(_checkpoint_key(repo)):
                store.set_roles_meta(_checkpoint_key(repo), moment_iso, moment_iso)

        logging.info(
            "Contributor roles backfill finished for %s: windows=%d/%d (this_run=%d) pulls_stored=%d "
            "matched=%d/%d granted=%d contributors (%d roles) enrichment_deferred=%d "
            "completed=%s rate_limited=%s timed_out=%s in %.2fs",
            repo,
            windows_done_total,
            total_windows,
            windows_completed,
            pulls_stored,
            matched_count,
            len(eligible_authors),
            contributors_granted,
            roles_granted,
            enrichment_deferred,
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
            total_windows=total_windows,
            windows_done_total=windows_done_total,
            enrichment_deferred=enrichment_deferred,
        )
    finally:
        _roles_backfill_lock.release()


def format_backfill_result(result: BackfillResult | None) -> str:
    """Ephemeral moderator summary for /roles-backfill."""
    if result is None:
        return "contributor roles backfill is already running or contributor roles are disabled."
    total = result.total_windows or (result.windows_done_total + result.windows_remaining)
    done = result.windows_done_total or result.windows_completed
    if result.completed:
        status = f"complete ({done}/{total} windows done)"
    else:
        status = f"{done}/{total} windows done, run /roles-backfill again to continue"
    reasons = []
    if result.rate_limited:
        reasons.append("rate-limited")
    if result.timed_out:
        reasons.append("time-bounded")
    suffix = f" ({', '.join(reasons)})" if reasons else ""
    lines = [
        f"**Contributor roles backfill: {status}**{suffix}",
        f"- Progress: **{done}/{total} windows done** (+{result.windows_completed} this run, {result.windows_remaining} remaining; newest-first)",
        f"- New community PRs stored: **{result.pulls_stored}**",
        f"- Community authors matched in Discord: **{result.authors_matched}/{result.authors_evaluated}**",
        f"- New role grants: **{result.roles_granted}** across **{result.contributors_granted}** contributors",
    ]
    if result.enrichment_deferred:
        lines.append(
            f"- Pending GitHub profile enrichment: **{result.enrichment_deferred}** authors deferred (cached progress saved; will continue on next run)"
        )
    if result.summary_announced:
        lines.append("- Posted a single summary line to the role announcements channel.")
    return "\n".join(lines)
