"""Automatic contributor-role polling and historical backfill without OAuth or webhooks.

Discovers merged PRs on the configured repository via GitHub search, stores
number/author/merged_at/labels/changed files in SQLite, resolves community PR
authors to Discord members with the same matcher as the weekly summary
(exact_login, TypeSafe Jev, deterministic fallback, 30-day SQLite cache), and
assigns earned roles from ``roles.py``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import uuid
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
    post_message,
)
from .config import Config, is_discord_id
from .matcher import (
    LLM_CONFIDENCE_THRESHOLD,
    LLM_MIN_SIGNALS,
    MESSAGE_EVIDENCE_STRONG_THRESHOLD,
    GitHubAuthorProfile,
    _MemberIndex,
    _cached_github_profile,
    _enrich_profiles_batch,
    _excluded_staff_discord_ids,
    _has_llm_credentials,
    _llm_selection_accepted,
    _pr_link_evidence_from_rows,
    _profile_has_enrichment_data,
    _select_with_llm,
    load_guild_members_with_source,
    match_community_contributors_with_diagnostics,
    multi_author_poster_ids,
    score_author_against_members,
)
from .rewards import (
    ROLE_ANNOUNCEMENT_CHUNK_LIMIT,
    _label_names,
    _models,
    announce_pending_role_grants,
    apply_roles,
    revoke_contributor_roles,
)
from .roles import (
    _agent_label,
    qualifies_agent_builder,
    qualifies_provider_wizard,
)
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
_grants_revalidation_lock = threading.Lock()
ROLES_JOB_LEASE_TTL_SECONDS = 600.0


def _lease_owner(job: str) -> str:
    return f"{job}:{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _lease_name(job: str, repository: str) -> str:
    return f"{job}:{(repository or '').strip().lower()}"


def _try_acquire_lease(
    store: Store | None,
    lock_name: str,
    owner_id: str,
    ttl_seconds: float,
    *,
    now: datetime | None = None,
) -> bool:
    if store is None or not hasattr(store, "acquire_lease"):
        return True
    try:
        return store.acquire_lease(lock_name, owner_id, ttl_seconds, now=now)
    except Exception as error:
        logging.warning("Could not acquire DB lease %s: %s", lock_name, error)
        return False


def _try_renew_lease(
    store: Store | None,
    lock_name: str,
    owner_id: str,
    ttl_seconds: float,
    *,
    now: datetime | None = None,
) -> None:
    if store is None or not hasattr(store, "renew_lease"):
        return
    try:
        store.renew_lease(lock_name, owner_id, ttl_seconds, now=now)
    except Exception as error:
        logging.warning("Could not renew DB lease %s: %s", lock_name, error)


def _try_release_lease(store: Store | None, lock_name: str, owner_id: str) -> None:
    if store is None or not hasattr(store, "release_lease"):
        return
    try:
        store.release_lease(lock_name, owner_id)
    except Exception as error:
        logging.warning("Could not release DB lease %s: %s", lock_name, error)


@dataclass(frozen=True)
class GrantsRevalidationResult:
    checked: int
    kept: int
    revoked: int
    roles_removed: int
    rate_limited: bool = False


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


def _detect_mapping_conflicts(
    config: Config,
    store: Store,
) -> tuple[set[str], set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Return ``(conflicting_discord_ids, conflicting_logins, discord_to_logins, login_to_discords)``.

    Enforces strict 1-to-1 bijection across github_discord_matches, github_links,
    and GITHUB_DISCORD_MAP (ignoring pairs recorded in match_rejections).
    """
    rejected = store.all_match_rejections() if hasattr(store, "all_match_rejections") else set()
    discord_to_logins: dict[str, set[str]] = {}
    login_to_discords: dict[str, set[str]] = {}

    def _add(login: str, uid: str) -> None:
        low = (login or "").strip().lower()
        uid_s = str(uid or "").strip()
        if not low or not is_discord_id(uid_s):
            return
        if (low, uid_s) in rejected:
            return
        discord_to_logins.setdefault(uid_s, set()).add(low)
        login_to_discords.setdefault(low, set()).add(uid_s)

    for map_login, map_uid in (getattr(config, "github_discord_map", None) or {}).items():
        _add(map_login, str(map_uid))
    if hasattr(store, "all_links"):
        for row in store.all_links():
            _add(str(row.get("github_login") or ""), str(row.get("discord_id") or ""))
    if hasattr(store, "all_confirmed_matches"):
        for row in store.all_confirmed_matches():
            _add(str(row.get("github_login") or ""), str(row.get("discord_id") or ""))

    conflicting_discords = {uid for uid, logins in discord_to_logins.items() if len(logins) > 1}
    conflicting_logins = {login for login, uids in login_to_discords.items() if len(uids) > 1}
    for uid in list(conflicting_discords):
        conflicting_logins.update(discord_to_logins.get(uid, set()))
    for login in list(conflicting_logins):
        conflicting_discords.update(login_to_discords.get(login, set()))
    return conflicting_discords, conflicting_logins, discord_to_logins, login_to_discords


def _resolve_stored_discord_id(
    config: Config,
    store: Store,
    login: str,
    user_id: int | None,
    *,
    conflicting_discords: set[str] | None = None,
    conflicting_logins: set[str] | None = None,
) -> str | None:
    """Reuse an already-known Discord link/mapping/github_discord_matches row without calling any LLM."""
    login_key = (login or "").strip().lower()
    if conflicting_logins and login_key in conflicting_logins:
        logging.warning(
            "Skipping stored match for github_login=%s due to 1-to-1 mapping conflict; granting nothing",
            login_key,
        )
        return None

    candidate: str | None = None
    if isinstance(user_id, int) and user_id > 0:
        link = store.link_for_github_user(user_id)
        if link and link.get("discord_id"):
            candidate = str(link["discord_id"])
    if candidate is None and login_key:
        mapped = (getattr(config, "github_discord_map", None) or {}).get(login_key)
        if mapped:
            candidate = str(mapped)
        else:
            confirmed = store.get_confirmed_match(login_key)
            if confirmed and confirmed.get("discord_id"):
                candidate = str(confirmed["discord_id"])

    if not candidate or not is_discord_id(candidate):
        return None
    if login_key and hasattr(store, "is_match_rejected") and store.is_match_rejected(login_key, candidate):
        return None
    if conflicting_discords and candidate in conflicting_discords:
        logging.warning(
            "Skipping stored match discord_id=%s for github_login=%s due to 1-to-1 mapping conflict; granting nothing",
            candidate,
            login_key,
        )
        return None
    return candidate


def _load_present_members(
    config: Config,
    *,
    guild=None,
    loop=None,
    members=None,
    rest_fetcher=None,
) -> tuple[list, frozenset[str]]:
    """Return ``(guild_members, present_member_ids)`` for verifying current guild membership."""
    if members is not None:
        member_list = list(members)
        return member_list, frozenset(m.discord_id for m in member_list if getattr(m, "discord_id", None))
    guild_id = config.discord_guild_id or (str(getattr(guild, "id", "")) if guild is not None else None)
    member_list, _source = load_guild_members_with_source(
        guild,
        loop=loop,
        guild_id=guild_id,
        discord_token=config.discord_token,
        rest_fetcher=rest_fetcher,
    )
    return member_list, frozenset(m.discord_id for m in member_list if getattr(m, "discord_id", None))


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
    graphql_caller=None,
    jev_transport=None,
    haiku_complete=None,
    rest_fetcher=None,
    rest_searcher=None,
    message_searcher=None,
    allow_live_message_search: bool | None = None,
    deadline: float | None = None,
    org_members: frozenset[str] = frozenset(),
    sleep_fn=time.sleep,
) -> tuple[dict[int, str], int, int]:
    """Resolve ``[{"user_id": int, "login": str, "pulls": [...]}]`` to ``(resolved, matched_count, enrichment_deferred)``.

    Reuses stored ``github_discord_matches`` / ``github_links`` / ``GITHUB_DISCORD_MAP``
    directly without running GitHub enrichment or the LLM, enforcing strict 1-to-1
    uniqueness and respecting ``match_rejections``.
    """
    if not authors:
        return {}, 0, 0

    conflicting_discords, conflicting_logins, _d2l, _l2d = _detect_mapping_conflicts(config, store)

    resolved: dict[int, str] = {}
    uid_to_login: dict[int, str] = {}
    reused_count = 0
    unmatched: list[dict[str, Any]] = []
    for author in authors:
        uid = author.get("user_id")
        login = str(author.get("login") or "")
        login_key = login.strip().lower()
        if isinstance(uid, int) and uid > 0:
            uid_to_login[uid] = login_key
        if login_key in conflicting_logins:
            logging.warning(
                "Skipping contributor role matching for github_login=%s due to existing 1-to-1 conflict; granting nothing",
                login_key,
            )
            continue
        stored_discord = _resolve_stored_discord_id(
            config,
            store,
            login,
            uid if isinstance(uid, int) else None,
            conflicting_discords=conflicting_discords,
            conflicting_logins=conflicting_logins,
        )
        if stored_discord and isinstance(uid, int) and uid > 0:
            resolved[uid] = stored_discord
            reused_count += 1
        else:
            unmatched.append(author)

    if unmatched:
        matched_groups, _skipped, _details, diagnostics = match_community_contributors_with_diagnostics(
            unmatched,
            config,
            store,
            guild=guild,
            loop=loop,
            now=now,
            members=members,
            github_getter=github_getter,
            graphql_caller=graphql_caller,
            jev_transport=jev_transport,
            haiku_complete=haiku_complete,
            rest_fetcher=rest_fetcher,
            rest_searcher=rest_searcher,
            message_searcher=message_searcher,
            allow_live_message_search=allow_live_message_search,
            deadline=deadline,
            org_members=org_members,
            sleep_fn=sleep_fn,
        )
        for group in matched_groups:
            uid = group.get("user_id")
            discord_id = group.get("discord_id")
            if isinstance(uid, int) and uid > 0 and isinstance(discord_id, str) and discord_id:
                resolved[uid] = discord_id
        member_count = diagnostics.member_count
        members_source = diagnostics.members_source
        llm_source = diagnostics.llm_source
        enrichment_deferred = diagnostics.enrichment_deferred
    else:
        member_count = 0
        members_source = "stored"
        llm_source = "none"
        enrichment_deferred = 0

    # Final strict 1-to-1 guard across all resolved authors: if two authors share a discord_id, grant nothing to either.
    by_discord: dict[str, list[int]] = {}
    for uid, discord_id in resolved.items():
        by_discord.setdefault(discord_id, []).append(uid)
    for discord_id, uids in by_discord.items():
        distinct_logins = {uid_to_login.get(u, str(u)) for u in uids}
        if len(distinct_logins) > 1 or discord_id in conflicting_discords:
            logging.warning(
                "1-to-1 role conflict for discord_id=%s across github_logins=%s; granting nothing",
                discord_id,
                sorted(distinct_logins),
            )
            for u in uids:
                resolved.pop(u, None)

    if not unmatched:
        logging.info(
            "Contributor role auto-match: reused %d/%d stored Discord matches (0 LLM calls)",
            len(resolved),
            len(authors),
        )
        return resolved, len(resolved), 0

    logging.info(
        "Contributor role auto-match: matched %d/%d community authors "
        "(reused_stored=%d, members=%d, source=%s, llm=%s, enrichment_deferred=%d)",
        len(resolved),
        len(authors),
        reused_count,
        member_count,
        members_source,
        llm_source,
        enrichment_deferred,
    )
    return resolved, len(resolved), enrichment_deferred


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
    present_member_ids: frozenset[str] | set[str] | None = None,
    guild=None,
    loop=None,
    members=None,
    rest_fetcher=None,
) -> tuple[int, bool]:
    """Ensure confirmed matches in ``github_discord_matches`` get earned roles (only if in guild with real GitHub PRs)."""
    repo = config.github_repository
    stored_matches = store.all_confirmed_matches(matched_since=matched_since)
    if not stored_matches:
        return 0, False

    if present_member_ids is None:
        _loaded, present_member_ids = _load_present_members(
            config,
            guild=guild,
            loop=loop,
            members=members,
            rest_fetcher=rest_fetcher,
        )
    if not present_member_ids:
        logging.warning("Skipping stored-match role grants for %s: guild member list unavailable", repo)
        return 0, False

    conflicting_discords, conflicting_logins, d2l, _l2d = _detect_mapping_conflicts(config, store)
    rejected = store.all_match_rejections() if hasattr(store, "all_match_rejections") else set()

    roles_granted = 0
    rate_limited = False
    for match in stored_matches:
        if deadline is not None and time.monotonic() >= deadline:
            break
        login = str(match.get("github_login") or "").strip()
        login_key = login.lower()
        discord_id = str(match.get("discord_id") or "").strip()
        if not login or not discord_id:
            continue
        if discord_id not in present_member_ids:
            logging.warning(
                "Skipping stored-match role grant for discord_id=%s github_login=%s: member not in guild",
                discord_id,
                login,
            )
            continue
        if (login_key, discord_id) in rejected:
            logging.info("Skipping rejected stored match discord_id=%s github_login=%s", discord_id, login)
            continue
        if discord_id in conflicting_discords or login_key in conflicting_logins:
            logging.warning(
                "1-to-1 stored-match conflict for discord_id=%s github_login=%s (all_logins=%s); granting nothing",
                discord_id,
                login,
                sorted(d2l.get(discord_id, {login_key})),
            )
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
        # GitHub is the ONLY source of truth: never fabricate PR counts when no GitHub PR evidence exists.
        if not pulls and (lifetime_count is None or lifetime_count <= 0):
            logging.info(
                "Skipping stored-match role grant for discord_id=%s github_login=%s: no GitHub merged PR evidence",
                discord_id,
                login,
            )
            continue
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
                github_login=login,
                match_method=str(match.get("method") or ""),
                match_confidence=float(match["confidence"]) if match.get("confidence") is not None else None,
                match_evidence=str(match.get("evidence") or ""),
                present_member_ids=present_member_ids,
            )
            roles_granted += len(granted)
        except TransientAPIError as error:
            logging.warning("Stored-match role assignment deferred for user %s: %s", discord_id, error)
            rate_limited = True
            break
    return roles_granted, rate_limited


def _search_merged_rows_per_day(
    repository: str,
    start: datetime,
    end: datetime,
    token: str | None,
) -> list[dict]:
    """Discover merged PR rows (with author metadata) across per-day ``merged:D..D`` windows."""
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
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get("number"), int)]


def _search_merged_numbers_per_day(
    repository: str,
    start: datetime,
    end: datetime,
    token: str | None,
) -> set[int]:
    """Discover merged PR numbers across per-day ``merged:D..D`` windows (cap 10 pages/day)."""
    return {int(row["number"]) for row in _search_merged_rows_per_day(repository, start, end, token)}


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
    graphql_caller=None,
    jev_transport=None,
    haiku_complete=None,
    rest_fetcher=None,
    rest_searcher=None,
    message_searcher=None,
    announcement_chunk_limit: int = ROLE_ANNOUNCEMENT_CHUNK_LIMIT,
    sleep_fn=time.sleep,
) -> RolesPollResult | None:
    """Check PRs merged in the rolling past 7 days and auto-assign newly earned roles."""
    if not config.contributor_roles_enabled or store is None:
        return None
    moment = now or datetime.now(UTC)
    repo = config.github_repository
    lease_key = _lease_name("roles_poll", repo)
    lease_owner = _lease_owner("roles_poll")
    lease_ttl = max(ROLES_JOB_LEASE_TTL_SECONDS, float(deadline_seconds or POLL_DEADLINE_SECONDS) + 120.0)
    if not _roles_poll_lock.acquire(blocking=False):
        logging.info("Contributor roles poll skipped: another poll is already in progress")
        return None
    if not _try_acquire_lease(store, lease_key, lease_owner, lease_ttl, now=moment):
        _roles_poll_lock.release()
        logging.info("Contributor roles poll skipped: another instance holds DB lease %s", lease_key)
        return None

    t_start = time.monotonic()
    deadline = (t_start + deadline_seconds) if deadline_seconds is not None else None
    moment_iso = iso(moment)
    org = repo.split("/", 1)[0] if "/" in repo else ""
    token = config.effective_github_token
    search_fn = search_merged_fn or _search_merged_rows_per_day
    try:
        window_start = moment - timedelta(days=DAILY_LOOKBACK_DAYS)
        window_start_iso = iso(window_start)

        if org_members_fetcher is not None:
            org_members = frozenset(m.lower() for m in org_members_fetcher(org, token))
        else:
            org_members = fetch_org_public_members(org, token) if org else frozenset()

        guild_members_loaded = False
        guild_members: list = []
        present_member_ids: frozenset[str] = frozenset()

        def _ensure_guild_members() -> tuple[list, frozenset[str]]:
            nonlocal guild_members_loaded, guild_members, present_member_ids
            if not guild_members_loaded:
                guild_members, present_member_ids = _load_present_members(
                    config,
                    guild=guild,
                    loop=loop,
                    members=members,
                    rest_fetcher=rest_fetcher,
                )
                guild_members_loaded = True
            return guild_members, present_member_ids

        # Grant roles for recent confirmed matches first (verifying current guild presence + GitHub PRs).
        stored_roles_granted = 0
        stored_rate_limited = False
        if store.all_confirmed_matches(matched_since=window_start_iso):
            _gm, present_ids = _ensure_guild_members()
            stored_roles_granted, stored_rate_limited = _grant_roles_for_stored_matches(
                config,
                store,
                now=moment,
                org=org,
                org_members=org_members,
                matched_since=window_start_iso,
                count_fetcher=count_fetcher,
                deadline=(deadline - 30.0) if deadline is not None else None,
                present_member_ids=present_ids,
            )

        try:
            raw_discovered = search_fn(repo, window_start, moment, token)
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
                if not config.roles_announce_enabled:
                    logging.info(
                        "Role announcements disabled (ROLES_ANNOUNCE_ENABLED=false); granted %d roles silently",
                        stored_roles_granted,
                    )
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

        rows_by_number: dict[int, dict[str, Any]] = {}
        numbers: set[int] = set()
        for entry in raw_discovered or ():
            if isinstance(entry, dict) and isinstance(entry.get("number"), int):
                num = int(entry["number"])
                numbers.add(num)
                rows_by_number[num] = entry
            elif isinstance(entry, int):
                numbers.add(entry)

        stored = 0
        skipped_existing = 0
        skipped_excluded = 0
        files_skipped = 0
        rate_limited = stored_rate_limited
        timed_out = False
        affected_authors: dict[int, dict[str, Any]] = {}
        if deadline is not None:
            match_reserve = min(60.0, max(20.0, float(deadline_seconds or POLL_DEADLINE_SECONDS) * 0.35))
            fetch_deadline = deadline - match_reserve
        else:
            fetch_deadline = None

        for number in sorted(numbers, reverse=True):
            if store.get_pull(repo, number) is not None:
                skipped_existing += 1
                continue
            search_row = rows_by_number.get(number)
            if search_row is not None:
                search_login = str(search_row.get("login") or "").strip()
                if search_login and _is_excluded_author(
                    search_login,
                    str(search_row.get("type") or ""),
                    str(search_row.get("author_association") or ""),
                    config.maintainer_logins,
                    org_members=org_members,
                    org=org,
                ):
                    skipped_excluded += 1
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
        if author_list:
            gm, present_ids = _ensure_guild_members()
        else:
            gm, present_ids = guild_members, present_member_ids
        resolved_map, matched_count, _enrichment_deferred = _match_authors(
            author_list,
            config,
            store,
            guild=guild,
            loop=loop,
            now=moment,
            members=gm if guild_members_loaded else members,
            github_getter=github_getter,
            graphql_caller=graphql_caller,
            jev_transport=jev_transport,
            haiku_complete=haiku_complete,
            rest_fetcher=rest_fetcher,
            rest_searcher=rest_searcher,
            message_searcher=message_searcher,
            allow_live_message_search=(members is None),
            deadline=deadline,
            org_members=org_members,
            sleep_fn=sleep_fn,
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
                    github_login=login,
                    present_member_ids=present_ids,
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
        if not config.roles_announce_enabled and roles_granted > 0:
            logging.info(
                "Role announcements disabled (ROLES_ANNOUNCE_ENABLED=false); granted %d roles silently",
                roles_granted,
            )

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
        _try_release_lease(store, lease_key, lease_owner)
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
        f"status={state}, announce={config.roles_announce_enabled}, oauth={config.github_oauth_enabled}, "
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
    moment = now or datetime.now(UTC)
    repo = config.github_repository
    lease_key = _lease_name("roles_daily", repo)
    lease_owner = _lease_owner("roles_daily")
    lease_ttl = max(ROLES_JOB_LEASE_TTL_SECONDS, float(POLL_DEADLINE_SECONDS) + 120.0)
    if not _roles_daily_lock.acquire(blocking=False):
        logging.info("Contributor roles daily check skipped: another daily roles run is already in progress")
        return None
    if not _try_acquire_lease(store, lease_key, lease_owner, lease_ttl, now=moment):
        _roles_daily_lock.release()
        logging.info("Contributor roles daily check skipped: another instance holds DB lease %s", lease_key)
        return None
    try:
        local = moment.astimezone(ZoneInfo(config.merged_timezone))
        run_date = local.date().isoformat()

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
        from contextlib import nullcontext
        from .tracing import AGENT_ENGAGEMENT, agent_span, current_agent_name

        runner = poll_fn or poll_merged_prs_for_roles
        eng_cm = (
            agent_span(
                AGENT_ENGAGEMENT,
                session_id=f"{AGENT_ENGAGEMENT}:daily_roles:{repo}:{run_date}",
                input_value={"task": "daily_roles", "repository": repo, "run_date": run_date, "force": force},
                attributes={
                    "engagement.task": "daily_roles",
                    "engagement.repository": repo,
                    "roles.announce_enabled": bool(config.roles_announce_enabled),
                },
            )
            if current_agent_name() is None
            else nullcontext()
        )
        with eng_cm as eng_span:
            result = runner(
                config,
                store,
                guild=guild,
                loop=loop,
                now=moment,
                **poll_kwargs,
            )
            if eng_span is not None and result is not None:
                eng_span.set_output({
                    "discovered": result.discovered,
                    "stored": result.stored,
                    "roles_granted": result.roles_granted,
                })
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
        _try_release_lease(store, lease_key, lease_owner)
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
    graphql_caller=None,
    jev_transport=None,
    haiku_complete=None,
    rest_fetcher=None,
    rest_searcher=None,
    message_searcher=None,
    announce_summary: bool = True,
    sleep_fn=time.sleep,
) -> BackfillResult | None:
    """Rate-limit-aware, time-bounded historical backfill with auto-matching.

    Posts at most ONE summary line to ``ROLE_ANNOUNCEMENTS_CHANNEL_ID`` (never
    one message per contributor).
    """
    if not config.contributor_roles_enabled or store is None:
        return None
    moment = now or datetime.now(UTC)
    repo = config.github_repository
    lease_key = _lease_name("roles_backfill", repo)
    lease_owner = _lease_owner("roles_backfill")
    lease_ttl = max(ROLES_JOB_LEASE_TTL_SECONDS, float(deadline_seconds or BACKFILL_DEADLINE_SECONDS) + 120.0)
    if not _roles_backfill_lock.acquire(blocking=False):
        logging.info("Contributor roles backfill skipped: already running")
        return None
    if not _try_acquire_lease(store, lease_key, lease_owner, lease_ttl, now=moment):
        _roles_backfill_lock.release()
        logging.info("Contributor roles backfill skipped: another instance holds DB lease %s", lease_key)
        return None

    t_start = time.monotonic()
    deadline = (t_start + deadline_seconds) if deadline_seconds is not None else None
    ingest_deadline = (
        max(t_start + 5.0, deadline - BACKFILL_MATCH_RESERVE_SECONDS)
        if deadline is not None
        else None
    )
    moment_iso = iso(moment)
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
            _try_renew_lease(store, lease_key, lease_owner, lease_ttl, now=now or datetime.now(UTC))
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

        if eligible_authors:
            bf_members, bf_present_ids = _load_present_members(
                config,
                guild=guild,
                loop=loop,
                members=members,
                rest_fetcher=rest_fetcher,
            )
        else:
            bf_members, bf_present_ids = [], frozenset()

        resolved_map, matched_count, enrichment_deferred = _match_authors(
            eligible_authors,
            config,
            store,
            guild=guild,
            loop=loop,
            now=moment,
            members=bf_members,
            github_getter=github_getter,
            graphql_caller=graphql_caller,
            jev_transport=jev_transport,
            haiku_complete=haiku_complete,
            rest_fetcher=rest_fetcher,
            rest_searcher=rest_searcher,
            message_searcher=message_searcher,
            allow_live_message_search=(members is None),
            deadline=deadline,
            org_members=org_members,
            sleep_fn=sleep_fn,
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
                    github_login=login_by_uid[uid],
                    present_member_ids=bf_present_ids,
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
        if not config.roles_announce_enabled:
            if roles_granted > 0:
                logging.info(
                    "Role announcements disabled (ROLES_ANNOUNCE_ENABLED=false); granted %d roles silently",
                    roles_granted,
                )
        elif announce_summary and contributors_granted > 0 and config.role_announcements_channel_id:
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
        _try_release_lease(store, lease_key, lease_owner)
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


def _grants_revalidation_key(repository: str) -> str:
    return f"grants_revalidated_v2:{repository.lower()}"


def should_run_grants_revalidation(config: Config, store: Store | None) -> bool:
    """True when the one-time boot re-validation of all existing grants has not yet completed."""
    if not config.contributor_roles_enabled or store is None:
        return False
    return not store.get_roles_meta(_grants_revalidation_key(config.github_repository))


def _stored_evidence_signals(raw_evidence: str) -> list[str]:
    text = (raw_evidence or "").strip()
    if not text:
        return []
    items: list[str] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            items = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            items = [text]
    except ValueError:
        items = [part.strip() for part in text.split(";") if part.strip()]
    return [item for item in items if not item.lower().startswith("reason:")]


def revalidate_all_grants_on_boot(
    config: Config,
    store: Store | None,
    *,
    guild=None,
    loop=None,
    members=None,
    now: datetime | None = None,
    org_members_fetcher=None,
    rest_fetcher=None,
    github_getter=None,
    graphql_caller=None,
    count_fetcher=None,
    llm_complete=None,
    skip_github_enrichment: bool = False,
    sleep_fn=time.sleep,
) -> GrantsRevalidationResult | None:
    """One-time boot job: re-validate ALL existing grants under the new matching & GitHub-truth rules.

    Silently removes Discord contributor roles and deletes ``role_grants`` / ``github_discord_matches``
    rows for any grant that no longer passes (recording rejected pairs in ``match_rejections``),
    logs every check and revocation, and posts NO announcements.
    """
    if not config.contributor_roles_enabled or store is None:
        return None
    repo = config.github_repository
    marker_key = _grants_revalidation_key(repo)
    if store.get_roles_meta(marker_key):
        return None
    moment = now or datetime.now(UTC)
    lease_key = _lease_name("roles_grants_revalidation", repo)
    lease_owner = _lease_owner("roles_grants_revalidation")
    if not _grants_revalidation_lock.acquire(blocking=False):
        return None
    if not _try_acquire_lease(store, lease_key, lease_owner, ROLES_JOB_LEASE_TTL_SECONDS, now=moment):
        _grants_revalidation_lock.release()
        logging.info("Boot all-grants re-validation skipped: another instance holds DB lease %s", lease_key)
        return None

    moment_iso = iso(moment)
    try:
        if store.get_roles_meta(marker_key):
            return None

        all_confirmed = store.all_confirmed_matches()
        granted_ids = set(store.all_granted_discord_ids()) if hasattr(store, "all_granted_discord_ids") else set()
        confirmed_by_discord: dict[str, list[dict]] = {}
        for row in all_confirmed:
            uid_s = str(row.get("discord_id") or "").strip()
            if uid_s:
                confirmed_by_discord.setdefault(uid_s, []).append(row)
        target_discord_ids = sorted(granted_ids | set(confirmed_by_discord.keys()))
        if not target_discord_ids:
            store.set_roles_meta(marker_key, moment_iso, moment_iso)
            logging.info("Boot all-grants re-validation for %s: 0 grants/matches to check", repo)
            return GrantsRevalidationResult(checked=0, kept=0, revoked=0, roles_removed=0)

        guild_members, _present_ids = _load_present_members(
            config,
            guild=guild,
            loop=loop,
            members=members,
            rest_fetcher=rest_fetcher,
        )
        if not guild_members:
            logging.warning(
                "Boot all-grants re-validation deferred for %s: guild members unavailable",
                repo,
            )
            return None

        member_index = _MemberIndex(guild_members)
        member_by_id = member_index.by_id
        token = config.effective_github_token
        org = repo.split("/", 1)[0] if "/" in repo else ""
        if org_members_fetcher is not None:
            org_members = frozenset(m.lower() for m in org_members_fetcher(org, token))
        else:
            org_members = fetch_org_public_members(org, token) if org else frozenset()

        excluded_staff_ids = _excluded_staff_discord_ids(
            config, store, guild_members, org_members=org_members, org=org,
        )
        all_repo_links = store.all_pr_link_posters(repo)
        all_linked_nums = {
            int(r["pr_number"])
            for r in all_repo_links
            if isinstance(r.get("pr_number"), int) and r["pr_number"] > 0
        }
        pr_number_to_author = store.pr_authors_for_numbers(repo, all_linked_nums)
        multi_author_ids = multi_author_poster_ids(all_repo_links, pr_number_to_author)
        rejected_pairs = store.all_match_rejections()
        conflicting_discords, conflicting_logins, _d2l, _l2d = _detect_mapping_conflicts(config, store)
        manual_map = getattr(config, "github_discord_map", None) or {}
        manual_by_discord: dict[str, list[str]] = {}
        for map_login, map_uid in manual_map.items():
            uid_s = str(map_uid or "").strip()
            low = str(map_login or "").strip().lower()
            if uid_s and low:
                manual_by_discord.setdefault(uid_s, []).append(low)

        # Pre-enrich all auto-matched GitHub logins in batch (never skipping lookups).
        enrich_groups: list[dict[str, Any]] = []
        seen_enrich_logins: set[str] = set()
        for row in all_confirmed:
            login = str(row.get("github_login") or "").strip()
            login_key = login.lower()
            method = str(row.get("method") or "").strip()
            discord_id = str(row.get("discord_id") or "").strip()
            if not login_key or method == "mod_override" or discord_id not in member_by_id:
                continue
            if login_key in seen_enrich_logins:
                continue
            seen_enrich_logins.add(login_key)
            raw_uid = row.get("github_user_id")
            uid = int(raw_uid) if isinstance(raw_uid, int) and raw_uid > 0 else 0
            pulls = store.pulls_for_author(repo, uid) if uid > 0 else store.pulls_for_author_login(repo, login)
            enrich_groups.append({
                "login": login,
                "user_id": uid or None,
                "pulls": [{"number": int(p["number"])} for p in pulls if isinstance(p.get("number"), int)],
            })

        profiles_by_login, failed_enrichment_logins = _enrich_profiles_batch(
            enrich_groups,
            repo,
            token,
            store=store,
            moment=moment,
            github_getter=github_getter,
            graphql_caller=graphql_caller,
            skip_github_enrichment=skip_github_enrichment,
            sleep_fn=sleep_fn,
        )

        kept = 0
        revoked = 0
        roles_removed_total = 0
        rate_limited = bool(failed_enrichment_logins)

        for discord_id in target_discord_ids:
            link_row = store.link_for_discord(discord_id)
            manual_logins = manual_by_discord.get(discord_id, [])
            confirmed_rows = confirmed_by_discord.get(discord_id, [])

            explicit_login = ""
            explicit_uid = 0
            explicit_source = ""
            if link_row and link_row.get("github_login"):
                explicit_login = str(link_row["github_login"]).strip()
                explicit_uid = int(link_row.get("github_user_id") or 0)
                explicit_source = "github_links"
            elif manual_logins:
                explicit_login = manual_logins[0]
                explicit_source = "config_map"
            else:
                for row in confirmed_rows:
                    if str(row.get("method") or "").strip() == "mod_override":
                        explicit_login = str(row.get("github_login") or "").strip()
                        explicit_uid = int(row.get("github_user_id") or 0)
                        explicit_source = "mod_override"
                        break

            primary_row = confirmed_rows[0] if confirmed_rows else None
            login = explicit_login or (str(primary_row.get("github_login") or "").strip() if primary_row else "")
            login_key = login.lower()
            raw_uid = explicit_uid or (primary_row.get("github_user_id") if primary_row else 0)
            uid = int(raw_uid) if isinstance(raw_uid, int) and raw_uid > 0 else 0
            old_method = explicit_source or (str(primary_row.get("method") or "").strip() if primary_row else "none")

            passed = False
            deferred = False
            reason = "unknown"

            if discord_id not in member_by_id:
                reason = "member_not_in_guild"
            elif not login_key:
                reason = "no_github_mapping"
            elif discord_id in conflicting_discords or login_key in conflicting_logins:
                reason = "mapping_conflict"
            elif (login_key, discord_id) in rejected_pairs:
                reason = "already_rejected"
            elif _is_excluded_author(login, "User", "", config.maintainer_logins, org_members=org_members, org=org):
                reason = "staff_or_bot_author"
            else:
                # Verify real GitHub merged-PR contribution evidence (GitHub is the ONLY source of truth).
                pulls = store.pulls_for_author(repo, uid) if uid > 0 else store.pulls_for_author_login(repo, login)
                if uid == 0 and pulls:
                    uid = int(pulls[-1]["github_user_id"])
                has_pr_evidence = len(pulls) > 0
                if not has_pr_evidence:
                    cached_count = store.get_author_merged_count(repo, login)
                    if cached_count is not None and _count_is_fresh(cached_count.get("fetched_at"), moment):
                        lifetime_count: int | None = int(cached_count["total_count"])
                    else:
                        lifetime_count = ensure_author_merged_count(
                            config,
                            store,
                            login,
                            uid,
                            now=moment,
                            count_fetcher=count_fetcher,
                        )
                    if lifetime_count is None:
                        deferred = True
                        rate_limited = True
                        reason = "github_count_lookup_deferred"
                    elif lifetime_count > 0:
                        has_pr_evidence = True

                if deferred:
                    pass
                elif not has_pr_evidence:
                    reason = "no_github_contributions"
                elif explicit_source:
                    passed = True
                    reason = f"verified:{explicit_source}"
                elif login_key in failed_enrichment_logins:
                    deferred = True
                    rate_limited = True
                    reason = "enrichment_failed_deferred"
                else:
                    profile = profiles_by_login.get(login_key) or _cached_github_profile(
                        store, login, moment, user_id=uid or None
                    ) or GitHubAuthorProfile(login=login, user_id=uid or None)
                    author_pr_nums = [int(p["number"]) for p in pulls if isinstance(p.get("number"), int)]
                    if author_pr_nums:
                        pr_rows = store.pr_link_posters_for_pulls(repo, author_pr_nums)
                    else:
                        pr_rows = [
                            row for row in store.pr_link_posters_for_discord(repo, discord_id)
                            if pr_number_to_author.get(int(row["pr_number"])) in (None, login_key)
                        ]
                    ev_list = _pr_link_evidence_from_rows(pr_rows)
                    rejected_for_login = {r_uid for r_log, r_uid in rejected_pairs if r_log == login_key}
                    strong, candidates = score_author_against_members(
                        profile,
                        guild_members,
                        member_index=member_index,
                        pr_link_posters=ev_list,
                        excluded_discord_ids=excluded_staff_ids,
                        multi_author_ids=multi_author_ids,
                        rejected_discord_ids=rejected_for_login,
                    )
                    if strong is not None:
                        if strong[0].discord_id == discord_id:
                            passed = True
                            reason = f"tier_verified:{strong[1]}"
                        else:
                            reason = f"strong_matched_other_member:{strong[0].discord_id}"
                    elif (
                        hasattr(store, "member_identity_notes_for_login")
                        and discord_id not in excluded_staff_ids
                        and discord_id not in multi_author_ids
                        and any(
                            str(n.get("discord_id") or "") == discord_id
                            and float(n.get("confidence") or 0.0) >= MESSAGE_EVIDENCE_STRONG_THRESHOLD
                            for n in store.member_identity_notes_for_login(login)
                        )
                        and len([
                            n for n in store.member_identity_notes_for_login(login)
                            if str(n.get("discord_id") or "") in member_by_id
                            and str(n.get("discord_id") or "") not in excluded_staff_ids
                            and str(n.get("discord_id") or "") not in multi_author_ids
                            and float(n.get("confidence") or 0.0) >= MESSAGE_EVIDENCE_STRONG_THRESHOLD
                        ]) == 1
                    ):
                        note = store.get_member_identity_note(discord_id)
                        passed = True
                        reason = f"message_evidence_verified:conf={float((note or {}).get('confidence') or 0.0):.2f}"
                    elif not skip_github_enrichment and not _profile_has_enrichment_data(profile):
                        reason = "no_enrichment_for_llm"
                    elif discord_id not in {c.member.discord_id for c in candidates}:
                        reason = "not_in_top5_candidates"
                    elif llm_complete is not None or _has_llm_credentials():
                        try:
                            from contextlib import nullcontext
                            from .tracing import AGENT_ENGAGEMENT, OPENINFERENCE_SPAN_KIND_CHAIN, agent_span, child_span, current_agent_name

                            outer_cm = (
                                agent_span(
                                    AGENT_ENGAGEMENT,
                                    session_id=f"{AGENT_ENGAGEMENT}:revalidate_grants:{repo}:{moment.date().isoformat()}",
                                    input_value={"task": "revalidate_grants", "repository": repo, "github_login": login},
                                    attributes={"engagement.task": "revalidate_grants", "engagement.repository": repo},
                                )
                                if current_agent_name() is None
                                else nullcontext()
                            )
                            with outer_cm:
                                with child_span(
                                    "role_matcher",
                                    kind=OPENINFERENCE_SPAN_KIND_CHAIN,
                                    input_value={"github_login": login, "discord_id": discord_id},
                                ) as rspan:
                                    llm_res = _select_with_llm(
                                        config,
                                        profile,
                                        candidates,
                                        llm_complete=llm_complete,
                                    )
                                    rspan.set_output({"selected": llm_res.discord_id, "confidence": llm_res.confidence})
                            if _llm_selection_accepted(llm_res) and llm_res.discord_id == discord_id:
                                passed = True
                                reason = f"llm_verified:conf={llm_res.confidence:.2f},signals={len(llm_res.evidence)}"
                            else:
                                reason = (
                                    f"llm_rejected:selected={llm_res.discord_id},"
                                    f"conf={llm_res.confidence:.2f},signals={len(llm_res.evidence)}"
                                )
                        except Exception as error:
                            logging.warning(
                                "Boot all-grants re-validation LLM error for discord_id=%s github_login=%s: %s",
                                discord_id,
                                login,
                                error,
                            )
                            deferred = True
                            rate_limited = True
                            reason = "llm_unavailable_deferred"
                    else:
                        stored_conf = float(primary_row.get("confidence") or 0.0) if primary_row else 0.0
                        stored_signals = _stored_evidence_signals(str(primary_row.get("evidence") or "")) if primary_row else []
                        if (
                            old_method == "llm"
                            and stored_conf >= LLM_CONFIDENCE_THRESHOLD
                            and len(stored_signals) >= LLM_MIN_SIGNALS
                        ):
                            passed = True
                            reason = f"stored_llm_verified:conf={stored_conf:.2f},signals={len(stored_signals)}"
                        else:
                            reason = f"failed_new_rules:old_method={old_method},conf={stored_conf:.2f},signals={len(stored_signals)}"

            if deferred:
                logging.info(
                    "Boot all-grants re-validation deferred discord_id=%s github_login=%s (%s)",
                    discord_id,
                    login or "unknown",
                    reason,
                )
                continue

            if passed:
                kept += 1
                logging.info(
                    "Boot all-grants re-validation KEPT discord_id=%s github_login=%s old_method=%s (%s)",
                    discord_id,
                    login,
                    old_method,
                    reason,
                )
                continue

            try:
                removed_roles, deleted_grants = revoke_contributor_roles(config, store, discord_id)
            except TransientAPIError as error:
                logging.warning(
                    "Boot all-grants re-validation hit Discord rate limit revoking discord_id=%s: %s",
                    discord_id,
                    error,
                )
                rate_limited = True
                break

            deleted_matches = store.delete_confirmed_matches_for_discord(discord_id)
            rejected_logins: list[str] = []
            for row in deleted_matches:
                bad_login = str(row.get("github_login") or "").strip().lower()
                if bad_login:
                    store.add_match_rejection(
                        bad_login,
                        discord_id,
                        moment_iso,
                        rejected_by="boot_revalidation",
                        reason=f"grants_revalidation_failed:{reason}",
                    )
                    rejected_logins.append(bad_login)
            if login_key and login_key not in rejected_logins and not explicit_source:
                store.delete_confirmed_match(login)
                store.add_match_rejection(
                    login_key,
                    discord_id,
                    moment_iso,
                    rejected_by="boot_revalidation",
                    reason=f"grants_revalidation_failed:{reason}",
                )

            revoked += 1
            roles_removed_total += len(removed_roles)
            logging.warning(
                "Boot all-grants re-validation REVOKED discord_id=%s github_login=%s "
                "old_method=%s reason=%s roles_removed=%s grants_deleted=%s",
                discord_id,
                login or "none",
                old_method,
                reason,
                removed_roles,
                deleted_grants,
            )

        if not rate_limited:
            store.set_roles_meta(marker_key, moment_iso, moment_iso)
        logging.info(
            "Boot all-grants re-validation for %s finished: checked=%d kept=%d revoked=%d "
            "roles_removed=%d rate_limited=%s",
            repo,
            len(target_discord_ids),
            kept,
            revoked,
            roles_removed_total,
            rate_limited,
        )
        return GrantsRevalidationResult(
            checked=len(target_discord_ids),
            kept=kept,
            revoked=revoked,
            roles_removed=roles_removed_total,
            rate_limited=rate_limited,
        )
    finally:
        _try_release_lease(store, lease_key, lease_owner)
        _grants_revalidation_lock.release()
