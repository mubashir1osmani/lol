"""Automatic GitHub-to-Discord contributor matching for the weekly summary.

Matches community PR authors to LiteLLM Discord guild members without requiring
contributors to run /link-github:
1. Optional /link-github store or persisted SQLite auto-matches (with 30-day TTL).
2. Discord guild members loaded via a multi-layer chain so empty/partial caches
   never silently yield zero matches:
   a. guild.members cache (when chunked or >=50% of guild.member_count)
   b. guild.chunk(cache=True)
   c. guild.fetch_members(limit=None)
   d. Discord REST GET /guilds/{id}/members (15-min TTL cache; works even when
      gateway intents.members was off or interactions arrive over HTTP)
   e. Per-author Discord REST GET /guilds/{id}/members/search fallback
3. GitHub profile (login, name, blog, twitter_username, bio) + PR commit author
   names and email local parts.
4. Deterministic strong signals (normalized login/name, Discord handle in bio/blog,
   matching twitter_username) against username, global_name, nick, display_name.
5. Single batched Jev (TypeSafe System One, REVIEW_DECISION_MODEL) noul request
   for ambiguous or fuzzy candidates (>= 0.85), with Haiku fallback when
   TYPESAFE_API_KEY is absent, and a deterministic fallback when no LLM is reachable.
"""

from __future__ import annotations

import asyncio
from collections import Counter
import difflib
import inspect
import json
import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta

from .api import (
    GITHUB_API,
    TransientAPIError,
    fetch_guild_members,
    github_get_json,
    github_rate_limit_status,
    search_guild_members,
)
from .config import is_discord_id
from .routing import FAST_MODEL, JEV_INPUT_USD_PER_TOKEN, _note_cost, call_jev_questions
from .schedule import UTC, iso, parse_time

JEV_MATCH_THRESHOLD = 0.85
JEV_RUNNER_UP_MARGIN = 0.05
MAX_CANDIDATES_PER_AUTHOR = 3
MAX_PAIRS_PER_BATCH = 30
FUZZY_CANDIDATE_THRESHOLD = 0.72
MATCH_CACHE_TTL_DAYS = 30
GITHUB_USER_CACHE_TTL_DAYS = 7
REST_MEMBERS_CACHE_TTL_SECONDS = 900.0
GUILD_FETCH_TIMEOUT_SECONDS = 60.0

_rest_members_lock = threading.Lock()
_rest_members_cache: dict[str, tuple[list["DiscordMemberInfo"], float]] = {}

_DISCORD_USER_URL = re.compile(
    r"https?://(?:www\.)?(?:discord\.com|discordapp\.com)/users/([1-9][0-9]{15,19})\b",
    re.IGNORECASE,
)
_DISCORD_INVITE_URL = re.compile(
    r"https?://(?:www\.)?(?:discord\.gg|discord\.com/invite|discordapp\.com/invite)/\S+",
    re.IGNORECASE,
)
_DISCORD_HANDLE_IN_TEXT = re.compile(
    r"\b(?:discord|dc)\b(?:\s*(?:username|handle|tag|id|is))?\s*(?:[:=@|/\-]\s*|\s+)@?([A-Za-z0-9_.-]{2,32})(?:#[0-9]{4})?",
    re.IGNORECASE,
)
_LEGACY_DISCORD_TAG = re.compile(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]{2,32})#[0-9]{4}\b")
_NOREPLY_ID_PREFIX = re.compile(r"^[0-9]+\+(.+)$")

_IGNORED_EMAIL_LOCALS = frozenset({
    "noreply", "no-reply", "donotreply", "action", "actions", "github", "github-actions",
    "bot", "dependabot", "renovate", "claude", "cursor", "admin", "root", "info",
    "contact", "hello", "hi", "mail", "email", "dev", "developer", "me", "support",
})
_IGNORED_COMMIT_NAMES = frozenset({
    "github", "web-flow", "github-actions", "github-actions[bot]",
    "dependabot[bot]", "renovate[bot]", "claude", "cursor",
})
_IGNORED_DISCORD_HANDLE_WORDS = frozenset({
    "server", "invite", "channel", "community", "guild", "bot", "link", "here", "join",
    "https", "http", "com", "gg", "org", "net", "io",
})
_LLM_API_KEY_ENVS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "LITELLM_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "VERTEXAI_PROJECT",
)


def reset_guild_members_cache() -> None:
    """Test helper."""
    with _rest_members_lock:
        _rest_members_cache.clear()


@dataclass(frozen=True)
class GitHubAuthorProfile:
    login: str
    user_id: int | None = None
    name: str = ""
    blog: str = ""
    twitter_username: str = ""
    bio: str = ""
    commit_names: tuple[str, ...] = ()
    email_locals: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscordMemberInfo:
    discord_id: str
    username: str = ""
    global_name: str = ""
    nick: str = ""
    display_name: str = ""


@dataclass(frozen=True)
class CandidatePair:
    profile: GitHubAuthorProfile
    member: DiscordMemberInfo
    heuristic_score: float
    signals: tuple[str, ...]


@dataclass(frozen=True)
class MatchDetail:
    github_login: str
    discord_id: str
    method: str
    confidence: float
    discord_username: str = ""
    pr_count: int = 0
    github_user_id: int | None = None


@dataclass(frozen=True)
class MatchDiagnostics:
    contributor_count: int
    member_count: int
    members_source: str
    candidate_pair_count: int
    matched_count: int
    skipped_count: int
    llm_source: str = "none"
    skip_reasons: tuple[tuple[str, int], ...] = ()
    method_counts: tuple[tuple[str, int], ...] = ()
    enrichment_deferred: int = 0

    def summary_line(self) -> str:
        parts = [f"checked {self.contributor_count} contributors against {self.member_count} members"]
        if self.members_source:
            parts.append(f"source: {self.members_source}")
        if self.candidate_pair_count:
            parts.append(f"{self.candidate_pair_count} candidates")
        if self.skip_reasons:
            reasons = ", ".join(f"{reason}={count}" for reason, count in self.skip_reasons)
            parts.append(f"skipped: {reasons}")
        return "(" + "; ".join(parts) + ")"


def _fold_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold()


def normalize_strict(text: str) -> str:
    """Lowercase and strip punctuation/spaces while preserving digits."""
    return re.sub(r"[^a-z0-9]+", "", _fold_ascii(text))


def normalize_base(text: str) -> str:
    """Lowercase, strip punctuation/spaces, and drop trailing digit suffixes."""
    strict = normalize_strict(text)
    if not strict:
        return ""
    stem = re.sub(r"\d+$", "", strict)
    return stem if len(stem) >= 3 else strict


def _name_variants(text: str) -> set[str]:
    out: set[str] = set()
    strict = normalize_strict(text)
    if strict:
        out.add(strict)
    base = normalize_base(text)
    if base:
        out.add(base)
    return out


def extract_discord_ids_from_text(text: str) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    for match in _DISCORD_USER_URL.finditer(text):
        uid = match.group(1)
        if is_discord_id(uid) and uid not in found:
            found.append(uid)
    return found


def extract_discord_handles_from_text(text: str) -> list[str]:
    """Extract explicit Discord usernames from a GitHub bio or blog string."""
    if not text:
        return []
    cleaned = _DISCORD_INVITE_URL.sub(" ", text)
    cleaned = _DISCORD_USER_URL.sub(" ", cleaned)
    found: list[str] = []
    for pattern in (_DISCORD_HANDLE_IN_TEXT, _LEGACY_DISCORD_TAG):
        for match in pattern.finditer(cleaned):
            raw = (match.group(1) or "").strip().strip(".,;:!?()[]{}\"'")
            lowered = raw.casefold()
            if (
                len(raw) >= 2
                and lowered not in _IGNORED_DISCORD_HANDLE_WORDS
                and not lowered.startswith("discord.")
                and lowered not in {h.casefold() for h in found}
            ):
                found.append(raw)
    return found


def _extract_email_local(email: str) -> str | None:
    if not isinstance(email, str) or "@" not in email:
        return None
    local, domain = email.strip().split("@", 1)
    domain_lower = domain.strip().casefold()
    if domain_lower.endswith("users.noreply.github.com"):
        m = _NOREPLY_ID_PREFIX.match(local)
        local = m.group(1) if m else local
    local = local.strip()
    if not local or "[bot]" in local.casefold():
        return None
    if local.casefold() in _IGNORED_EMAIL_LOCALS:
        return None
    if len(normalize_strict(local)) < 3:
        return None
    return local


def _is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, TransientAPIError):
        return True
    text = str(error)
    return "HTTP 403" in text or "HTTP 429" in text or "rate limit" in text.lower()


def _github_budget_available(
    *,
    sleep=time.sleep,
    clock=time.time,
    allow_sleep: bool = False,
) -> bool:
    """Return True if GitHub API budget is available; never blocks for long rate-limit resets."""
    remaining, reset = github_rate_limit_status()
    if remaining is None or remaining > 5:
        return True
    now_ts = clock()
    if reset is None or reset <= now_ts:
        return True
    delay = reset - now_ts + 0.25
    if allow_sleep and 0 < delay <= 10.0:
        logging.info("GitHub rate limit low (remaining=%d); waiting %.1fs for reset", remaining, delay)
        sleep(delay)
        return True
    logging.warning(
        "GitHub rate limit low (remaining=%d, resets in %.0fs); stopping optional profile lookups",
        remaining,
        delay,
    )
    return False


def _call_github_enrichment(fetch, url: str, token: str | None, *, timeout: float = 10.0):
    if fetch is github_get_json:
        return github_get_json(url, token, pace=False, attempts=1, timeout=timeout, max_retry_delay=0.0)
    return fetch(url, token, pace=False)


def _cached_github_profile(
    store,
    login: str,
    moment: datetime,
    *,
    user_id: int | None = None,
) -> GitHubAuthorProfile | None:
    if not login or not re.fullmatch(r"[A-Za-z0-9-]+", login):
        return None
    if store is None or not hasattr(store, "get_cached_github_user"):
        return None
    cached_user = store.get_cached_github_user(login)
    if not cached_user or not _match_is_fresh(
        cached_user.get("fetched_at"), moment, ttl_days=GITHUB_USER_CACHE_TTL_DAYS
    ):
        return None
    cached_email = (cached_user.get("email_local") or "").strip()
    cached_commit = (cached_user.get("commit_name") or "").strip()
    return GitHubAuthorProfile(
        login=login,
        user_id=user_id,
        name=(cached_user.get("name") or "").strip(),
        blog=(cached_user.get("blog") or "").strip(),
        twitter_username=(cached_user.get("twitter_username") or "").strip().lstrip("@"),
        bio=(cached_user.get("bio") or "").strip(),
        commit_names=(cached_commit,) if cached_commit else (),
        email_locals=(cached_email,) if cached_email else (),
    )


def fetch_github_author_profile(
    login: str,
    repository: str,
    pull_numbers: list[int],
    token: str | None,
    *,
    user_id: int | None = None,
    getter=None,
    store=None,
    now: datetime | None = None,
    deadline: float | None = None,
) -> tuple[GitHubAuthorProfile, bool]:
    """Fetch GitHub user profile (with 7-day SQLite cache) + PR commit author names/email locals."""
    fetch = getter or github_get_json
    using_real_github = fetch is github_get_json
    moment = now or datetime.now(UTC)
    name = ""
    blog = ""
    twitter = ""
    bio = ""
    email_locals: list[str] = []
    commit_names: list[str] = []
    rate_limited = False

    def _add_email(raw_email: object) -> None:
        if isinstance(raw_email, str):
            local = _extract_email_local(raw_email)
            if local and local.casefold() not in {e.casefold() for e in email_locals}:
                email_locals.append(local)

    def _add_commit_name(raw_name: object) -> None:
        if not isinstance(raw_name, str):
            return
        cleaned = raw_name.strip()
        if not cleaned or cleaned.casefold() in _IGNORED_COMMIT_NAMES:
            return
        if cleaned.casefold() not in {n.casefold() for n in commit_names}:
            commit_names.append(cleaned)

    def _time_left() -> float:
        if deadline is None:
            return 10.0
        return deadline - time.monotonic()

    used_cached_user = False
    profile_fetched = False
    if login and re.fullmatch(r"[A-Za-z0-9-]+", login):
        cached_profile = _cached_github_profile(store, login, moment, user_id=user_id)
        if cached_profile is not None:
            used_cached_user = True
            name = cached_profile.name
            blog = cached_profile.blog
            twitter = cached_profile.twitter_username
            bio = cached_profile.bio
            commit_names.extend(cached_profile.commit_names)
            email_locals.extend(cached_profile.email_locals)

        if not used_cached_user:
            if deadline is not None and _time_left() <= 1.0:
                rate_limited = True
            elif using_real_github and not _github_budget_available():
                rate_limited = True
            else:
                try:
                    req_timeout = min(10.0, max(2.0, _time_left()))
                    profile = _call_github_enrichment(fetch, f"{GITHUB_API}/users/{login}", token, timeout=req_timeout)
                    if isinstance(profile, dict):
                        profile_fetched = True
                        name = (profile.get("name") or "").strip() if isinstance(profile.get("name"), str) else ""
                        blog = (profile.get("blog") or "").strip() if isinstance(profile.get("blog"), str) else ""
                        twitter = (
                            (profile.get("twitter_username") or "").strip().lstrip("@")
                            if isinstance(profile.get("twitter_username"), str) else ""
                        )
                        bio = (profile.get("bio") or "").strip() if isinstance(profile.get("bio"), str) else ""
                        _add_email(profile.get("email"))
                        if store is not None and hasattr(store, "save_cached_github_user"):
                            try:
                                store.save_cached_github_user(
                                    login,
                                    name=name,
                                    blog=blog,
                                    twitter_username=twitter,
                                    bio=bio,
                                    email_local=email_locals[0] if email_locals else "",
                                    fetched_at=iso(moment),
                                )
                            except Exception as cache_error:
                                logging.warning("Could not cache GitHub profile for %s: %s", login, cache_error)
                except Exception as error:
                    if _is_rate_limit_error(error):
                        rate_limited = True
                    logging.warning("Could not fetch GitHub profile for %s: %s", login, error)

    should_fetch_commits = (
        not rate_limited
        and bool(repository)
        and bool(pull_numbers)
        and not (used_cached_user and (commit_names or email_locals))
    )
    if should_fetch_commits:
        commits_added = False
        for pr_number in pull_numbers[:2]:
            if not isinstance(pr_number, int) or pr_number <= 0:
                continue
            if deadline is not None and _time_left() <= 1.0:
                rate_limited = True
                break
            if using_real_github and not _github_budget_available():
                rate_limited = True
                break
            try:
                req_timeout = min(10.0, max(2.0, _time_left()))
                commits = _call_github_enrichment(
                    fetch,
                    f"{GITHUB_API}/repos/{repository}/pulls/{pr_number}/commits?per_page=30",
                    token,
                    timeout=req_timeout,
                )
            except Exception as error:
                if _is_rate_limit_error(error):
                    rate_limited = True
                logging.warning("Could not fetch commits for %s#%s: %s", repository, pr_number, error)
                break
            if not isinstance(commits, list):
                continue
            for entry in commits:
                if not isinstance(entry, dict):
                    continue
                author_obj = entry.get("author")
                if isinstance(author_obj, dict):
                    commit_login = author_obj.get("login")
                    if isinstance(commit_login, str) and commit_login.casefold() != login.casefold():
                        continue
                commit = entry.get("commit") if isinstance(entry.get("commit"), dict) else {}
                c_author = commit.get("author") if isinstance(commit.get("author"), dict) else {}
                before_counts = (len(commit_names), len(email_locals))
                _add_commit_name(c_author.get("name"))
                _add_email(c_author.get("email"))
                if (len(commit_names), len(email_locals)) != before_counts:
                    commits_added = True
            if commit_names and email_locals:
                break
        if commits_added and (profile_fetched or used_cached_user) and store is not None and hasattr(store, "save_cached_github_user"):
            try:
                store.save_cached_github_user(
                    login,
                    name=name,
                    blog=blog,
                    twitter_username=twitter,
                    bio=bio,
                    email_local=email_locals[0] if email_locals else "",
                    commit_name=commit_names[0] if commit_names else "",
                    fetched_at=iso(moment),
                )
            except Exception as cache_error:
                logging.warning("Could not update cached commit info for %s: %s", login, cache_error)

    return (
        GitHubAuthorProfile(
            login=login,
            user_id=user_id,
            name=name,
            blog=blog,
            twitter_username=twitter,
            bio=bio,
            commit_names=tuple(commit_names),
            email_locals=tuple(email_locals),
        ),
        rate_limited,
    )


def member_from_raw(member) -> DiscordMemberInfo | None:
    if isinstance(member, DiscordMemberInfo):
        return member
    if isinstance(member, dict):
        user = member.get("user") if isinstance(member.get("user"), dict) else {}
        if user.get("bot") or member.get("bot"):
            return None
        member_id = user.get("id") or member.get("id")
        if member_id is None or not is_discord_id(str(member_id)):
            return None
        username = str(user.get("username") or member.get("username") or member.get("name") or "").strip()
        global_name = str(user.get("global_name") or member.get("global_name") or "").strip()
        nick = str(member.get("nick") or "").strip()
        display_name = str(member.get("display_name") or "").strip()
        return DiscordMemberInfo(
            discord_id=str(member_id),
            username=username,
            global_name=global_name,
            nick=nick,
            display_name=display_name,
        )
    if getattr(member, "bot", False):
        return None
    member_id = getattr(member, "id", None)
    if member_id is None or not is_discord_id(str(member_id)):
        return None
    username = str(getattr(member, "name", None) or "").strip()
    global_name = str(getattr(member, "global_name", None) or "").strip()
    nick = str(getattr(member, "nick", None) or "").strip()
    display_name = str(getattr(member, "display_name", None) or "").strip()
    return DiscordMemberInfo(
        discord_id=str(member_id),
        username=username,
        global_name=global_name,
        nick=nick,
        display_name=display_name,
    )


async def _collect_async_members(result) -> list:
    if hasattr(result, "__aiter__"):
        found = []
        async for member in result:
            found.append(member)
        return found
    if inspect.isawaitable(result):
        resolved = await result
        return list(resolved) if resolved is not None else []
    return list(result) if result is not None else []


def _run_guild_coro(coro, loop):
    if loop is not None and loop.is_running():
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            coro.close()
            raise RuntimeError("Cannot block gateway event loop from its own thread")
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=GUILD_FETCH_TIMEOUT_SECONDS)
    return asyncio.run(coro)


def _cache_is_complete(guild, cached: list[DiscordMemberInfo]) -> bool:
    if not cached:
        return False
    if getattr(guild, "chunked", False) is True:
        return True
    member_count = getattr(guild, "member_count", None)
    if not isinstance(member_count, int) or member_count <= 0:
        return True
    if member_count <= 5:
        return len(cached) >= max(1, member_count - 1)
    return len(cached) >= max(2, int(member_count * 0.5))


def _load_rest_members_cached(
    guild_id: str,
    discord_token: str,
    *,
    rest_fetcher=None,
    ttl_seconds: float = REST_MEMBERS_CACHE_TTL_SECONDS,
    clock=time.monotonic,
) -> list[DiscordMemberInfo] | None:
    if not guild_id or not is_discord_id(str(guild_id)) or not discord_token:
        return None
    key = str(guild_id)
    now = clock()
    if rest_fetcher is None:
        with _rest_members_lock:
            cached = _rest_members_cache.get(key)
            if cached is not None and now < cached[1]:
                return list(cached[0])
    fetcher = rest_fetcher or fetch_guild_members
    try:
        raw = fetcher(key, discord_token)
    except Exception as error:
        logging.warning("Discord REST guild member list failed for guild %s: %s", key, error)
        return None
    parsed = [info for item in (raw or ()) if (info := member_from_raw(item)) is not None]
    if parsed and rest_fetcher is None:
        with _rest_members_lock:
            _rest_members_cache[key] = (list(parsed), now + ttl_seconds)
    return parsed


def load_guild_members_with_source(
    guild,
    loop=None,
    *,
    guild_id: str | None = None,
    discord_token: str | None = None,
    rest_fetcher=None,
) -> tuple[list[DiscordMemberInfo], str]:
    """Load non-bot guild members via cache -> chunk -> fetch_members -> REST /members."""
    resolved_guild_id = str(guild_id or getattr(guild, "id", "") or "")
    best: list[DiscordMemberInfo] = []
    best_source = "none"

    if guild is not None:
        raw_cached = list(getattr(guild, "members", ()) or ())
        cached = [info for raw in raw_cached if (info := member_from_raw(raw)) is not None]
        if cached:
            best = cached
            best_source = "guild_cache"
        if _cache_is_complete(guild, cached):
            return cached, "guild_cache"

        # Layer 2: gateway guild.chunk(cache=True) when members intent is on but guild isn't chunked yet.
        chunker = getattr(guild, "chunk", None)
        if callable(chunker) and not getattr(guild, "chunked", False):
            try:
                async def run_chunk():
                    result = chunker(cache=True)
                    if inspect.isawaitable(result):
                        await result
                    return list(getattr(guild, "members", ()) or ())

                chunked_raw = _run_guild_coro(run_chunk(), loop)
                chunked = [info for raw in chunked_raw if (info := member_from_raw(raw)) is not None]
                if len(chunked) > len(best):
                    best = chunked
                    best_source = "guild_chunk"
                if _cache_is_complete(guild, chunked):
                    return chunked, "guild_chunk"
            except Exception as error:
                logging.warning("guild.chunk() failed; falling back to fetch_members/REST: %s", error)

        # Layer 3: discord.py guild.fetch_members(limit=None).
        fetcher = getattr(guild, "fetch_members", None)
        if callable(fetcher):
            try:
                async def run_fetch():
                    return await _collect_async_members(fetcher(limit=None))

                fetched_raw = _run_guild_coro(run_fetch(), loop)
                fetched = [info for raw in fetched_raw if (info := member_from_raw(raw)) is not None]
                if len(fetched) > len(best):
                    best = fetched
                    best_source = "guild_fetch_members"
                if fetched:
                    return fetched, "guild_fetch_members"
            except Exception as error:
                logging.warning("guild.fetch_members() failed; falling back to REST /members: %s", error)

    # Layer 4: Discord REST GET /guilds/{id}/members (works whenever Server Members Intent
    # is enabled in the Discord Developer Portal, even if intents.members=False on the client
    # or the slash command came in over HTTP with guild=None).
    if resolved_guild_id and discord_token:
        rest_members = _load_rest_members_cached(
            resolved_guild_id,
            discord_token,
            rest_fetcher=rest_fetcher,
        )
        if rest_members is not None and len(rest_members) > len(best):
            return rest_members, "rest_members"
        if rest_members is not None and rest_members:
            return rest_members, "rest_members"

    return best, best_source


def load_guild_members(
    guild,
    loop=None,
    *,
    guild_id: str | None = None,
    discord_token: str | None = None,
    rest_fetcher=None,
) -> list[DiscordMemberInfo]:
    """Load non-bot Discord members from guild cache, chunking, fetch_members, or REST."""
    members, _source = load_guild_members_with_source(
        guild,
        loop=loop,
        guild_id=guild_id,
        discord_token=discord_token,
        rest_fetcher=rest_fetcher,
    )
    return members


def _member_fields(member: DiscordMemberInfo) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for val in (member.username, member.global_name, member.nick, member.display_name):
        cleaned = (val or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return tuple(out)


def _member_tokens(member: DiscordMemberInfo) -> tuple[set[str], set[str], set[str]]:
    raw_lower: set[str] = set()
    strict_set: set[str] = set()
    base_set: set[str] = set()
    for field in _member_fields(member):
        lowered = field.casefold()
        if lowered:
            raw_lower.add(lowered)
        s = normalize_strict(field)
        if s:
            strict_set.add(s)
        b = normalize_base(field)
        if b:
            base_set.add(b)
    return raw_lower, strict_set, base_set


class _MemberIndex:
    """Precomputed O(1) lookup tables over a guild's members for fast matching at scale."""

    def __init__(self, members: list[DiscordMemberInfo]):
        self.members = list(members)
        self.by_id: dict[str, DiscordMemberInfo] = {m.discord_id: m for m in members}
        self.entry_idx_by_id: dict[str, int] = {}
        self.by_raw: dict[str, list[DiscordMemberInfo]] = {}
        self.by_strict: dict[str, list[DiscordMemberInfo]] = {}
        self.by_base: dict[str, list[DiscordMemberInfo]] = {}
        self.by_bigram: dict[str, list[int]] = {}
        self.entries: list[tuple[DiscordMemberInfo, set[str], set[str], set[str], tuple[str, ...]]] = []

        for idx, member in enumerate(members):
            raw_l, strict_s, base_s = _member_tokens(member)
            norms = tuple(sorted({t for t in (strict_s | base_s) if len(t) >= 3}))
            self.entries.append((member, raw_l, strict_s, base_s, norms))
            self.entry_idx_by_id[member.discord_id] = idx
            for tok in raw_l:
                self.by_raw.setdefault(tok, []).append(member)
            for tok in strict_s:
                self.by_strict.setdefault(tok, []).append(member)
            for tok in base_s:
                self.by_base.setdefault(tok, []).append(member)
            seen_bigrams: set[str] = set()
            for norm in norms:
                if len(norm) < 4:
                    continue
                for pos in range(len(norm) - 1):
                    bg = norm[pos: pos + 2]
                    if bg not in seen_bigrams:
                        seen_bigrams.add(bg)
                        self.by_bigram.setdefault(bg, []).append(idx)

    def exact_tier_hits(self, value: str) -> list[DiscordMemberInfo]:
        """Match exact/strict normalized form first; strip trailing digit suffixes only if no strict hit."""
        cleaned = (value or "").strip()
        if not cleaned:
            return []
        hits_by_id: dict[str, DiscordMemberInfo] = {}
        lowered = cleaned.casefold()
        if len(lowered) >= 3:
            for member in self.by_raw.get(lowered, ()):
                hits_by_id[member.discord_id] = member
        strict = normalize_strict(cleaned)
        if len(strict) >= 4:
            for member in self.by_strict.get(strict, ()):
                hits_by_id[member.discord_id] = member
        if hits_by_id:
            return list(hits_by_id.values())
        base = normalize_base(cleaned)
        if len(base) >= 4:
            for member in self.by_base.get(base, ()):
                hits_by_id[member.discord_id] = member
        return list(hits_by_id.values())

    def exact_field_indices(self, value: str) -> set[int]:
        """Indices of members matching ``value`` via raw, strict, or base token."""
        cleaned = (value or "").strip()
        if not cleaned:
            return set()
        out: set[int] = set()
        lowered = cleaned.casefold()
        if len(lowered) >= 3:
            for member in self.by_raw.get(lowered, ()):
                out.add(self.entry_idx_by_id[member.discord_id])
        strict = normalize_strict(cleaned)
        if len(strict) >= 4:
            for member in self.by_strict.get(strict, ()):
                out.add(self.entry_idx_by_id[member.discord_id])
        base = normalize_base(cleaned)
        if len(base) >= 4:
            for member in self.by_base.get(base, ()):
                out.add(self.entry_idx_by_id[member.discord_id])
        return out

    def initial_pattern_indices(self, full_name: str) -> set[int]:
        parts = [re.sub(r"[^a-z]+", "", _fold_ascii(p)) for p in re.split(r"[\s._-]+", full_name or "")]
        parts = [p for p in parts if p]
        if len(parts) < 2:
            return set()
        first, last = parts[0], parts[-1]
        if len(first) < 2 or len(last) < 2:
            return set()
        out: set[int] = set()
        for pattern in (f"{first[0]}{last}", f"{first}{last[0]}", f"{last}{first}"):
            if len(pattern) >= 4:
                for member in self.by_strict.get(pattern, ()):
                    out.add(self.entry_idx_by_id[member.discord_id])
                for member in self.by_base.get(pattern, ()):
                    out.add(self.entry_idx_by_id[member.discord_id])
        return out


def _exact_strict_match(value: str, raw_lower: set[str], strict_set: set[str]) -> bool:
    cleaned = (value or "").strip()
    if not cleaned:
        return False
    lowered = cleaned.casefold()
    if len(lowered) >= 3 and lowered in raw_lower:
        return True
    strict = normalize_strict(cleaned)
    return len(strict) >= 4 and strict in strict_set


def _exact_base_match(value: str, base_set: set[str]) -> bool:
    cleaned = (value or "").strip()
    if not cleaned:
        return False
    base = normalize_base(cleaned)
    return len(base) >= 4 and base in base_set


def _exact_field_match(value: str, raw_lower: set[str], strict_set: set[str], base_set: set[str]) -> bool:
    return _exact_strict_match(value, raw_lower, strict_set) or _exact_base_match(value, base_set)


def _name_initial_match_tokens(full_name: str, all_member_norms: set[str]) -> float:
    """Check patterns like 'Jane Doe' <-> 'janed', 'jdoe', or 'Jane D'."""
    parts = [re.sub(r"[^a-z]+", "", _fold_ascii(p)) for p in re.split(r"[\s._-]+", full_name or "")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return 0.0
    first, last = parts[0], parts[-1]
    if len(first) < 2 or len(last) < 2:
        return 0.0
    patterns = (f"{first[0]}{last}", f"{first}{last[0]}", f"{last}{first}")
    for pattern in patterns:
        if len(pattern) >= 4 and pattern in all_member_norms:
            return 0.84
    return 0.0


def _fuzzy_score_entry(
    profile: GitHubAuthorProfile,
    source_norms: list[tuple[str, str]],
    raw_lower: set[str],
    strict_set: set[str],
    base_set: set[str],
    member_norms: tuple[str, ...],
) -> tuple[float, list[str]]:
    all_norms = strict_set | base_set
    best = 0.0
    signals: list[str] = []

    for commit_name in profile.commit_names:
        if _exact_field_match(commit_name, raw_lower, strict_set, base_set):
            if 0.88 > best:
                best = 0.88
            signals.append(f"commit_name_match:{commit_name}")
        else:
            init_score = _name_initial_match_tokens(commit_name, all_norms)
            if init_score > best:
                best = init_score
                signals.append(f"commit_initial_match:{commit_name}")

    for email_local in profile.email_locals:
        if _exact_field_match(email_local, raw_lower, strict_set, base_set):
            if 0.86 > best:
                best = 0.86
            signals.append(f"email_local_match:{email_local}")

    if profile.name:
        init_score = _name_initial_match_tokens(profile.name, all_norms)
        if init_score > best:
            best = init_score
            signals.append(f"name_initial_match:{profile.name}")

    for label, src_norm in source_norms:
        src_len = len(src_norm)
        for mem_norm in member_norms:
            mem_len = len(mem_norm)
            if mem_len < 4:
                continue
            # Cheap length-ratio filter before SequenceMatcher: ratio <= 2*min/(a+b).
            if (2 * min(src_len, mem_len)) / (src_len + mem_len) < FUZZY_CANDIDATE_THRESHOLD:
                continue
            sm = difflib.SequenceMatcher(None, src_norm, mem_norm)
            if sm.quick_ratio() >= FUZZY_CANDIDATE_THRESHOLD:
                ratio = sm.ratio()
            else:
                ratio = 0.0
            if ratio >= FUZZY_CANDIDATE_THRESHOLD:
                if ratio > best:
                    best = ratio
                tag = f"fuzzy_{label}:{ratio:.2f}"
                if tag not in signals:
                    signals.append(tag)
            elif (
                src_len >= 5
                and mem_len >= 5
                and (src_norm in mem_norm or mem_norm in src_norm)
                and (min(src_len, mem_len) / max(src_len, mem_len)) >= 0.65
            ):
                if 0.76 > best:
                    best = 0.76
                tag = f"substring_{label}"
                if tag not in signals:
                    signals.append(tag)

    return best, signals[:5]


def _profile_source_norms(profile: GitHubAuthorProfile) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = [("login", profile.login)]
    if profile.name:
        sources.append(("name", profile.name))
    if profile.twitter_username:
        sources.append(("twitter", profile.twitter_username))
    for cn in profile.commit_names:
        sources.append(("commit_name", cn))
    for el in profile.email_locals:
        sources.append(("email_local", el))
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for label, value in sources:
        for norm in _name_variants(value):
            if len(norm) >= 4 and (label, norm) not in seen:
                seen.add((label, norm))
                out.append((label, norm))
    return out


def score_author_against_members(
    profile: GitHubAuthorProfile,
    members: list[DiscordMemberInfo],
    *,
    max_candidates: int = MAX_CANDIDATES_PER_AUTHOR,
    member_index: _MemberIndex | None = None,
) -> tuple[tuple[DiscordMemberInfo, str, float] | None, list[CandidatePair]]:
    """Return either a unique deterministic strong match OR ranked fuzzy/ambiguous candidates."""
    index = member_index if member_index is not None else _MemberIndex(members)
    if not index.members:
        return None, []

    # 1. Direct Discord user URL/ID in bio or blog.
    for source_name, text in (("bio_discord_id", profile.bio), ("blog_discord_id", profile.blog)):
        hits = [index.by_id[uid] for uid in extract_discord_ids_from_text(text) if uid in index.by_id]
        if len(hits) == 1:
            return (hits[0], source_name, 0.99), []

    # 2. Explicit Discord handle in bio or blog.
    ambiguous_strong: dict[str, tuple[DiscordMemberInfo, list[str]]] = {}
    for source_name, text in (("bio_discord_handle", profile.bio), ("blog_discord_handle", profile.blog)):
        handles = extract_discord_handles_from_text(text)
        if not handles:
            continue
        hits_by_id: dict[str, DiscordMemberInfo] = {}
        for handle in handles:
            for member in index.exact_tier_hits(handle):
                hits_by_id[member.discord_id] = member
        hits = list(hits_by_id.values())
        if len(hits) == 1:
            return (hits[0], source_name, 0.98), []
        for member in hits:
            entry = ambiguous_strong.setdefault(member.discord_id, (member, []))
            entry[1].append(f"ambiguous_{source_name}")

    # 3. Exact normalized match of login, twitter_username, or GitHub profile name.
    strong_tiers = (
        ("exact_login", profile.login, 0.96),
        ("twitter_match", profile.twitter_username, 0.95),
        ("exact_name", profile.name, 0.93),
    )
    for method, value, confidence in strong_tiers:
        if not value:
            continue
        hits = index.exact_tier_hits(value)
        if len(hits) == 1:
            return (hits[0], method, confidence), []
        for member in hits:
            entry = ambiguous_strong.setdefault(member.discord_id, (member, []))
            entry[1].append(f"ambiguous_{method}")

    # 4. Collect ambiguous strong collisions + fuzzy/commit/email candidates for Jev/Haiku.
    # Every string pair of length >= 4 with SequenceMatcher ratio >= 0.72 (or len >= 5 substring)
    # must share at least one 2-gram, so narrowing by exact/initial + 2-gram index is exact and fast.
    source_norms = _profile_source_norms(profile)
    candidate_indices: set[int] = set()
    for uid in ambiguous_strong:
        idx = index.entry_idx_by_id.get(uid)
        if idx is not None:
            candidate_indices.add(idx)
    for commit_name in profile.commit_names:
        candidate_indices.update(index.exact_field_indices(commit_name))
        candidate_indices.update(index.initial_pattern_indices(commit_name))
    for email_local in profile.email_locals:
        candidate_indices.update(index.exact_field_indices(email_local))
    if profile.name:
        candidate_indices.update(index.initial_pattern_indices(profile.name))
    for _label, src_norm in source_norms:
        for pos in range(len(src_norm) - 1):
            hits_idx = index.by_bigram.get(src_norm[pos: pos + 2])
            if hits_idx:
                candidate_indices.update(hits_idx)

    candidates: list[CandidatePair] = []
    for idx in sorted(candidate_indices):
        member, raw_l, strict_s, base_s, norms = index.entries[idx]
        if member.discord_id in ambiguous_strong:
            _m, sigs = ambiguous_strong[member.discord_id]
            fuzzy_score, fuzzy_sigs = _fuzzy_score_entry(profile, source_norms, raw_l, strict_s, base_s, norms)
            combined_sigs = tuple(dict.fromkeys(sigs + fuzzy_sigs))
            candidates.append(CandidatePair(profile, member, max(0.90, fuzzy_score), combined_sigs))
            continue
        score, sigs = _fuzzy_score_entry(profile, source_norms, raw_l, strict_s, base_s, norms)
        if score >= FUZZY_CANDIDATE_THRESHOLD and sigs:
            candidates.append(CandidatePair(profile, member, score, tuple(sigs)))

    candidates.sort(key=lambda item: (-item.heuristic_score, item.member.discord_id))
    return None, candidates[:max_candidates]


def _serialize_pair_for_model(pair_key: str, pair: CandidatePair) -> dict:
    return {
        "pair": pair_key,
        "github": {
            "login": pair.profile.login,
            "name": pair.profile.name,
            "twitter_username": pair.profile.twitter_username,
            "bio": pair.profile.bio[:200],
            "blog": pair.profile.blog[:160],
            "commit_names": list(pair.profile.commit_names[:4]),
            "email_locals": list(pair.profile.email_locals[:4]),
        },
        "discord": {
            "id": pair.member.discord_id,
            "username": pair.member.username,
            "global_name": pair.member.global_name,
            "nick": pair.member.nick,
            "display_name": pair.member.display_name,
        },
        "signals": list(pair.signals),
    }


def _score_pairs_with_jev(
    config,
    pairs: list[CandidatePair],
    api_key: str,
    *,
    jev_transport=None,
    timeout: float = 15.0,
) -> dict[str, float]:
    """Send all candidate pairs to TypeSafe Jev in a single batched noul request."""
    model = getattr(config, "review_decision_model", None) or "jev-latest"
    state_items = []
    questions = {}
    for index, pair in enumerate(pairs):
        key = f"pair_{index}"
        state_items.append(_serialize_pair_for_model(key, pair))
        questions[key] = {
            "type": "noul",
            "instructions": (
                f"In {key}, are GitHub contributor '{pair.profile.login}' and "
                f"Discord member '{pair.member.username}' the same person?"
            ),
            "criteria": {
                "true": "Strong evidence from login, real name, commit name, email local part, bio/blog/twitter, and Discord username/global_name/nick that they are the same person",
                "false": "Different people, coincidence, or insufficient evidence to be sure",
            },
        }
    state = json.dumps({"candidate_pairs": state_items})
    payload = call_jev_questions(model, state, questions, api_key, transport=jev_transport, timeout=timeout)
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int):
        _note_cost(usage["input_tokens"] * JEV_INPUT_USD_PER_TOKEN)
    answers = payload.get("answers") if isinstance(payload, dict) else None
    scores: dict[str, float] = {}
    if isinstance(answers, dict):
        for key in questions:
            entry = answers.get(key)
            noul = entry.get("noul") if isinstance(entry, dict) else None
            if isinstance(noul, (int, float)):
                scores[key] = float(noul)
    return scores


def _has_haiku_credentials() -> bool:
    return any(os.environ.get(name, "").strip() for name in _LLM_API_KEY_ENVS)


def _score_pairs_with_haiku(
    config,
    pairs: list[CandidatePair],
    *,
    haiku_complete=None,
    timeout: float = 20.0,
) -> dict[str, float]:
    """Single batched Haiku fallback when TYPESAFE_API_KEY is not configured."""
    if haiku_complete is None and not _has_haiku_credentials():
        raise RuntimeError("No LLM API key configured for Haiku fallback")
    model = (
        getattr(config, "review_classifier_model", None)
        or getattr(config, "fast_model", None)
        or FAST_MODEL
    )
    state_items = [
        _serialize_pair_for_model(f"pair_{index}", pair)
        for index, pair in enumerate(pairs)
    ]
    prompt = json.dumps({"candidate_pairs": state_items})
    messages = [
        {
            "role": "system",
            "content": (
                "For each candidate pair in candidate_pairs, estimate the probability (0.0 to 1.0) "
                "that the GitHub contributor and Discord member are the same person. "
                "Only give >= 0.85 when there is clear, specific evidence; give low scores when uncertain. "
                "Reply with a single JSON object mapping each pair key (e.g. \"pair_0\") to a float."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    if haiku_complete is not None:
        raw = haiku_complete(model, messages)
    else:
        from .review_llm import complete_litellm
        raw = complete_litellm(model, messages, timeout=timeout)
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start: end + 1]) if isinstance(raw, str) and start >= 0 and end >= start else {}
    except (ValueError, AttributeError):
        parsed = {}
    scores: dict[str, float] = {}
    if isinstance(parsed, dict):
        for index in range(len(pairs)):
            key = f"pair_{index}"
            val = parsed.get(key)
            if isinstance(val, (int, float)):
                scores[key] = max(0.0, min(1.0, float(val)))
    return scores


def _deterministic_candidate_fallback(
    candidates: list[CandidatePair],
) -> dict[str, tuple[DiscordMemberInfo, str, float]]:
    """Conservative fallback when both Jev and Haiku are unavailable.

    Only accepts a unique exact normalized commit_name or email_local match
    (length >= 5) with no competing candidate for that GitHub author.
    """
    by_login: dict[str, list[CandidatePair]] = {}
    for pair in candidates:
        by_login.setdefault(pair.profile.login.casefold(), []).append(pair)

    winners: dict[str, tuple[DiscordMemberInfo, str, float]] = {}
    for login_key, pairs in by_login.items():
        if len(pairs) != 1:
            continue
        pair = pairs[0]
        if any(sig.startswith("ambiguous_") for sig in pair.signals):
            continue
        exact_secondary = [
            sig for sig in pair.signals
            if sig.startswith(("commit_name_match:", "email_local_match:"))
        ]
        if not exact_secondary:
            continue
        matched_value = exact_secondary[0].split(":", 1)[1]
        if len(normalize_strict(matched_value)) >= 5:
            winners[login_key] = (pair.member, "deterministic_fallback", 0.86)
    return winners


def _disambiguate_with_reasons(
    config,
    candidates: list[CandidatePair],
    *,
    threshold: float = JEV_MATCH_THRESHOLD,
    jev_transport=None,
    haiku_complete=None,
    deadline: float | None = None,
) -> tuple[dict[str, tuple[DiscordMemberInfo, str, float]], str, dict[str, str]]:
    """Score ambiguous/fuzzy pairs in one batched Jev call, Haiku fallback, or deterministic fallback."""
    if not candidates:
        return {}, "none", {}

    api_key = (
        os.environ.get("TYPESAFE_API_KEY", "").strip()
        or (getattr(config, "typesafe_api_key", None) or "").strip()
    )
    all_scores: dict[int, float] = {}
    source = "unavailable"
    any_llm_succeeded = False

    for offset in range(0, len(candidates), MAX_PAIRS_PER_BATCH):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 3.0:
                logging.warning(
                    "Weekly report deadline reached before LLM disambiguation batch %d; "
                    "falling back to deterministic matching",
                    offset // MAX_PAIRS_PER_BATCH + 1,
                )
                break
            jev_timeout = min(15.0, max(3.0, remaining))
            haiku_timeout = min(20.0, max(3.0, remaining))
        else:
            jev_timeout = 15.0
            haiku_timeout = 20.0
        batch = candidates[offset: offset + MAX_PAIRS_PER_BATCH]
        batch_scores: dict[str, float] = {}
        if api_key or jev_transport is not None:
            try:
                batch_scores = _score_pairs_with_jev(
                    config, batch, api_key, jev_transport=jev_transport, timeout=jev_timeout,
                )
                source = "jev"
                any_llm_succeeded = True
            except Exception as error:
                logging.warning("Jev contributor matching failed; trying Haiku fallback: %s", error)
                if deadline is not None and (deadline - time.monotonic()) <= 3.0:
                    batch_scores = {}
                else:
                    try:
                        batch_scores = _score_pairs_with_haiku(
                            config, batch, haiku_complete=haiku_complete, timeout=haiku_timeout,
                        )
                        source = "haiku"
                        any_llm_succeeded = True
                    except Exception as haiku_error:
                        logging.warning("Haiku contributor matching fallback unavailable: %s", haiku_error)
                        batch_scores = {}
        else:
            try:
                batch_scores = _score_pairs_with_haiku(
                    config, batch, haiku_complete=haiku_complete, timeout=haiku_timeout,
                )
                source = "haiku"
                any_llm_succeeded = True
            except Exception as haiku_error:
                logging.info("No TYPESAFE_API_KEY or Haiku key available for fuzzy candidates: %s", haiku_error)
                batch_scores = {}

        for idx, _pair in enumerate(batch):
            key = f"pair_{idx}"
            if key in batch_scores:
                all_scores[offset + idx] = batch_scores[key]

    candidate_logins = {pair.profile.login.casefold() for pair in candidates}
    if not any_llm_succeeded:
        fallback_winners = _deterministic_candidate_fallback(candidates)
        skip_reasons = {
            login_key: "llm_unavailable"
            for login_key in candidate_logins
            if login_key not in fallback_winners
        }
        return fallback_winners, ("deterministic_fallback" if fallback_winners else "unavailable"), skip_reasons

    by_login: dict[str, list[tuple[float, DiscordMemberInfo]]] = {}
    for idx, pair in enumerate(candidates):
        score = all_scores.get(idx)
        if score is None:
            continue
        by_login.setdefault(pair.profile.login.casefold(), []).append((score, pair.member))

    winners: dict[str, tuple[DiscordMemberInfo, str, float]] = {}
    skip_reasons: dict[str, str] = {}
    for login_key in candidate_logins:
        scored = by_login.get(login_key)
        if not scored:
            skip_reasons[login_key] = f"{source}_no_score"
            continue
        scored.sort(key=lambda item: -item[0])
        top_score, top_member = scored[0]
        if top_score < threshold:
            skip_reasons[login_key] = f"{source}_below_threshold"
            continue
        if len(scored) > 1:
            second_score = scored[1][0]
            if second_score >= threshold and (top_score - second_score) < JEV_RUNNER_UP_MARGIN:
                skip_reasons[login_key] = "ambiguous_tie"
                continue
        winners[login_key] = (top_member, source, top_score)
    return winners, source, skip_reasons


def disambiguate_candidates(
    config,
    candidates: list[CandidatePair],
    *,
    threshold: float = JEV_MATCH_THRESHOLD,
    jev_transport=None,
    haiku_complete=None,
) -> dict[str, tuple[DiscordMemberInfo, str, float]]:
    """Score ambiguous/fuzzy pairs in one batched Jev call (or Haiku / deterministic fallback)."""
    winners, _source, _reasons = _disambiguate_with_reasons(
        config,
        candidates,
        threshold=threshold,
        jev_transport=jev_transport,
        haiku_complete=haiku_complete,
    )
    return winners


def _match_is_fresh(matched_at: object, now: datetime, ttl_days: int = MATCH_CACHE_TTL_DAYS) -> bool:
    if not isinstance(matched_at, str) or not matched_at.strip():
        return False
    try:
        recorded = parse_time(matched_at)
    except Exception:
        return False
    return (now - recorded) <= timedelta(days=ttl_days)


def _search_rest_candidates_for_author(
    profile: GitHubAuthorProfile,
    guild_id: str,
    discord_token: str,
    *,
    rest_searcher=None,
) -> list[DiscordMemberInfo]:
    """Last-resort per-author REST member search when neither cache nor /members list is available."""
    if not guild_id or not is_discord_id(str(guild_id)) or not discord_token:
        return []
    searcher = rest_searcher or search_guild_members
    queries: list[str] = []
    for raw in (profile.login, profile.twitter_username, profile.name, *profile.commit_names[:1]):
        cleaned = (raw or "").strip()
        if cleaned and len(cleaned) >= 2 and cleaned.casefold() not in {q.casefold() for q in queries}:
            queries.append(cleaned)
    found_by_id: dict[str, DiscordMemberInfo] = {}
    for query in queries[:3]:
        try:
            items = searcher(str(guild_id), query, discord_token, limit=10)
        except Exception as error:
            logging.warning("Discord REST member search failed for %s: %s", query, error)
            break
        for item in items or ():
            info = member_from_raw(item)
            if info is not None:
                found_by_id[info.discord_id] = info
    return list(found_by_id.values())


def match_community_contributors_with_diagnostics(
    groups: list[dict],
    config,
    store,
    *,
    guild=None,
    loop=None,
    now: datetime | None = None,
    members: list[DiscordMemberInfo] | None = None,
    github_getter=None,
    jev_transport=None,
    haiku_complete=None,
    rest_fetcher=None,
    rest_searcher=None,
    skip_github_enrichment: bool = False,
    deadline: float | None = None,
) -> tuple[list[dict], int, list[MatchDetail], MatchDiagnostics]:
    """Resolve community PR authors to LiteLLM Discord guild members with diagnostics."""
    moment = now or datetime.now(UTC)
    moment_iso = iso(moment)
    guild_id = getattr(config, "discord_guild_id", None) or (str(getattr(guild, "id", "")) if guild is not None else None)
    discord_token = getattr(config, "discord_token", None)

    guild_members: list[DiscordMemberInfo] = []
    members_source = "none"
    if members is not None:
        guild_members = list(members)
        members_source = "injected"

    resolved: dict[str, tuple[dict, str, str, float]] = {}  # login_lower -> (group, discord_id, method, confidence)
    sqlite_resolved_keys: set[str] = set()
    pending: list[dict] = []
    skip_reason_by_login: dict[str, str] = {}

    manual_map = getattr(config, "github_discord_map", None) or {}
    for group in groups:
        login = group["login"]
        login_key = login.strip().casefold()
        user_id = group.get("user_id")

        # Optional pre-existing Discord ID on group, /link-github store, or static map.
        discord_id = group.get("discord_id")
        if discord_id and is_discord_id(str(discord_id)):
            resolved[login_key] = (group, str(discord_id), "prelinked", 1.0)
            continue

        if store is not None and isinstance(user_id, int) and hasattr(store, "link_for_github_user"):
            link = store.link_for_github_user(user_id)
            if link and is_discord_id(link.get("discord_id")):
                resolved[login_key] = (group, str(link["discord_id"]), "link_store", 1.0)
                continue

        mapped_id = manual_map.get(login_key)
        if mapped_id and is_discord_id(str(mapped_id)):
            resolved[login_key] = (group, str(mapped_id), "config_map", 1.0)
            continue

        # Persisted SQLite automatic match (computed once, expires after MATCH_CACHE_TTL_DAYS).
        if store is not None and hasattr(store, "get_confirmed_match"):
            cached = store.get_confirmed_match(login)
            if cached and is_discord_id(cached.get("discord_id")) and _match_is_fresh(cached.get("matched_at"), moment):
                cid = str(cached["discord_id"])
                resolved[login_key] = (
                    group,
                    cid,
                    f"sqlite:{cached['method']}",
                    float(cached.get("confidence") or 1.0),
                )
                sqlite_resolved_keys.add(login_key)
                continue

        pending.append(group)

    if members is None and (pending or sqlite_resolved_keys):
        t_members = time.monotonic()
        guild_members, members_source = load_guild_members_with_source(
            guild,
            loop=loop,
            guild_id=guild_id,
            discord_token=discord_token,
            rest_fetcher=rest_fetcher,
        )
        logging.info(
            "Weekly report: loaded %d Discord members (source=%s) in %.2fs",
            len(guild_members),
            members_source,
            time.monotonic() - t_members,
        )

    member_index = _MemberIndex(guild_members)
    member_by_id = member_index.by_id

    # If guild members were loaded, drop any stale SQLite match for a user who left the server
    # and re-queue them for fresh matching.
    if member_by_id and sqlite_resolved_keys:
        for login_key in list(sqlite_resolved_keys):
            group, cid, _method, _conf = resolved[login_key]
            if cid not in member_by_id:
                del resolved[login_key]
                pending.append(group)

    ambiguous_candidates: list[CandidatePair] = []
    llm_source = "none"

    cache_incomplete = guild is not None and not _cache_is_complete(guild, guild_members)
    if (
        members is None
        and pending
        and (not guild_members or (members_source == "guild_cache" and cache_incomplete))
        and guild_id
        and discord_token
    ):
        # Last-resort per-author REST member search when /members returned empty/403.
        searched_members: dict[str, DiscordMemberInfo] = {m.discord_id: m for m in guild_members}
        found_via_search = 0
        for group in pending:
            prof = GitHubAuthorProfile(login=group["login"], user_id=group.get("user_id"))
            for found in _search_rest_candidates_for_author(
                prof, str(guild_id), discord_token, rest_searcher=rest_searcher
            ):
                if found.discord_id not in searched_members:
                    found_via_search += 1
                searched_members[found.discord_id] = found
        if found_via_search:
            guild_members = list(searched_members.values())
            member_index = _MemberIndex(guild_members)
            member_by_id = member_index.by_id
            members_source = "guild_cache+rest_search" if members_source == "guild_cache" else "rest_search"

    enrichment_deferred = 0
    if pending and not guild_members:
        for group in pending:
            skip_reason_by_login[group["login"].strip().casefold()] = "no_guild_members"
    elif pending and guild_members:
        token = getattr(config, "github_token", None) or getattr(config, "github_actor_token", None)
        repository = getattr(config, "github_repository", "") or ""
        rate_limited = False
        pending_by_login = {g["login"].strip().casefold(): g for g in pending}
        enrich_deadline = (deadline - 20.0) if deadline is not None else None
        deadline_tripped = False

        for idx, group in enumerate(pending):
            login = group["login"]
            login_key = login.strip().casefold()

            # Fast path: if GitHub login alone uniquely matches a guild member, skip 2 GitHub API calls.
            login_hits = member_index.exact_tier_hits(login)
            if len(login_hits) == 1:
                resolved[login_key] = (group, login_hits[0].discord_id, "exact_login", 0.96)
                continue

            pull_numbers = [
                int(p["number"])
                for p in group.get("pulls", ())
                if isinstance(p, dict) and isinstance(p.get("number"), int)
            ]
            if enrich_deadline is not None and time.monotonic() >= enrich_deadline:
                if not deadline_tripped:
                    deadline_tripped = True
                    logging.warning(
                        "Weekly report GitHub enrichment deadline reached after %d/%d pending authors; "
                        "using cached profiles and deferring uncached authors to next run",
                        idx,
                        len(pending),
                    )
                cached = _cached_github_profile(store, login, moment, user_id=group.get("user_id"))
                if cached is not None:
                    profile = cached
                else:
                    enrichment_deferred += 1
                    profile = GitHubAuthorProfile(login=login, user_id=group.get("user_id"))
            elif skip_github_enrichment or rate_limited:
                cached = _cached_github_profile(store, login, moment, user_id=group.get("user_id"))
                if cached is not None:
                    profile = cached
                else:
                    if rate_limited:
                        enrichment_deferred += 1
                    profile = GitHubAuthorProfile(login=login, user_id=group.get("user_id"))
            else:
                profile, hit_limit = fetch_github_author_profile(
                    login,
                    repository,
                    pull_numbers,
                    token,
                    user_id=group.get("user_id"),
                    getter=github_getter,
                    store=store,
                    now=moment,
                    deadline=enrich_deadline,
                )
                if hit_limit:
                    rate_limited = True
                    enrichment_deferred += 1

            strong, candidates = score_author_against_members(
                profile,
                guild_members,
                member_index=member_index,
            )
            if strong is not None:
                member, method, confidence = strong
                resolved[login_key] = (group, member.discord_id, method, confidence)
            elif candidates:
                ambiguous_candidates.extend(candidates)
            else:
                skip_reason_by_login[login_key] = "no_signal_match"

        if ambiguous_candidates:
            llm_winners, llm_source, llm_skip_reasons = _disambiguate_with_reasons(
                config,
                ambiguous_candidates,
                jev_transport=jev_transport,
                haiku_complete=haiku_complete,
                deadline=(deadline - 10.0) if deadline is not None else None,
            )
            for login_key, (member, method, confidence) in llm_winners.items():
                if login_key not in resolved and login_key in pending_by_login:
                    resolved[login_key] = (pending_by_login[login_key], member.discord_id, method, confidence)
            for login_key, reason in llm_skip_reasons.items():
                if login_key not in resolved:
                    skip_reason_by_login[login_key] = reason

    # Enforce 1-to-1 Discord ID assignment (highest confidence wins; leave conflicts unmatched).
    claimed_discord: dict[str, tuple[str, float]] = {}
    for login_key, (_group, discord_id, _method, confidence) in sorted(
        resolved.items(), key=lambda item: (-item[1][3], item[0])
    ):
        existing = claimed_discord.get(discord_id)
        if existing is None or confidence > existing[1]:
            if existing is not None:
                skip_reason_by_login[existing[0]] = "discord_id_conflict"
            claimed_discord[discord_id] = (login_key, confidence)
        else:
            skip_reason_by_login[login_key] = "discord_id_conflict"

    winning_logins = {login_key for login_key, _conf in claimed_discord.values()}

    matched_groups: list[dict] = []
    details: list[MatchDetail] = []
    method_counter: Counter[str] = Counter()
    for login_key in winning_logins:
        group, discord_id, method, confidence = resolved[login_key]
        method_counter[method] += 1
        # Persist newly confirmed positive matches to SQLite so they are computed once.
        # Negative / unmatched results are never persisted.
        if (
            store is not None
            and hasattr(store, "save_confirmed_match")
            and method not in {"prelinked", "link_store", "config_map"}
            and not method.startswith("sqlite:")
        ):
            try:
                store.save_confirmed_match(
                    group["login"],
                    discord_id,
                    method=method,
                    confidence=confidence,
                    matched_at=moment_iso,
                    github_user_id=group.get("user_id"),
                )
            except Exception as error:
                logging.warning("Could not persist match for %s: %s", group["login"], error)

        matched_groups.append({
            **group,
            "discord_id": discord_id,
            "match_method": method,
            "match_confidence": confidence,
        })
        member = member_by_id.get(discord_id)
        details.append(MatchDetail(
            github_login=group["login"],
            discord_id=discord_id,
            method=method,
            confidence=confidence,
            discord_username=member.username if member else "",
            pr_count=len(group.get("pulls", ())),
            github_user_id=group.get("user_id") if isinstance(group.get("user_id"), int) else None,
        ))

    matched_groups.sort(key=lambda item: (-len(item["pulls"]), item["login"].lower()))
    details.sort(key=lambda item: (-item.pr_count, item.github_login.lower()))
    skipped_count = len(groups) - len(matched_groups)

    skip_counter: Counter[str] = Counter()
    for group in groups:
        login_key = group["login"].strip().casefold()
        if login_key not in winning_logins:
            skip_counter[skip_reason_by_login.get(login_key, "unmatched")] += 1

    diagnostics = MatchDiagnostics(
        contributor_count=len(groups),
        member_count=len(guild_members),
        members_source=members_source,
        candidate_pair_count=len(ambiguous_candidates),
        matched_count=len(matched_groups),
        skipped_count=skipped_count,
        llm_source=llm_source,
        skip_reasons=tuple(sorted(skip_counter.items())),
        method_counts=tuple(sorted(method_counter.items())),
        enrichment_deferred=enrichment_deferred,
    )
    logging.info(
        "Weekly match diagnostics: contributors=%d members=%d (source=%s) candidates=%d "
        "matched=%d skipped=%d llm=%s methods=%s skip_reasons=%s",
        diagnostics.contributor_count,
        diagnostics.member_count,
        diagnostics.members_source,
        diagnostics.candidate_pair_count,
        diagnostics.matched_count,
        diagnostics.skipped_count,
        diagnostics.llm_source,
        dict(diagnostics.method_counts),
        dict(diagnostics.skip_reasons),
    )
    return matched_groups, skipped_count, details, diagnostics


def match_community_contributors(
    groups: list[dict],
    config,
    store,
    *,
    guild=None,
    loop=None,
    now: datetime | None = None,
    members: list[DiscordMemberInfo] | None = None,
    github_getter=None,
    jev_transport=None,
    haiku_complete=None,
    rest_fetcher=None,
    rest_searcher=None,
    skip_github_enrichment: bool = False,
) -> tuple[list[dict], int, list[MatchDetail]]:
    """Resolve community PR authors to LiteLLM Discord guild members."""
    matched, skipped, details, _diagnostics = match_community_contributors_with_diagnostics(
        groups,
        config,
        store,
        guild=guild,
        loop=loop,
        now=now,
        members=members,
        github_getter=github_getter,
        jev_transport=jev_transport,
        haiku_complete=haiku_complete,
        rest_fetcher=rest_fetcher,
        rest_searcher=rest_searcher,
        skip_github_enrichment=skip_github_enrichment,
    )
    return matched, skipped, details
