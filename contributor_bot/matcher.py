"""Automatic GitHub-to-Discord contributor matching for the weekly summary and daily roles.

Matches community PR authors to LiteLLM Discord guild members without requiring
contributors to run /link-github:
1. Optional /link-github store or persisted SQLite auto-matches (with 30-day TTL).
2. Discord guild members loaded via a multi-layer chain so empty/partial caches
   never silently yield zero matches:
   a. guild.members cache (when chunked or >=50% of guild.member_count)
   b. guild.chunk(cache=True)
   c. guild.fetch_members(limit=None)
   d. Discord REST GET /guilds/{id}/members (15-min TTL cache)
   e. Per-author Discord REST GET /guilds/{id}/members/search fallback
3. Batched GitHub GraphQL enrichment (50 logins/query: name, bio, company, email,
   twitterUsername, websiteUrl, socialAccounts, 4KB profile README) + PR commit
   names/emails, cached in SQLite for 7 days.
4. Deterministic strong tiers (bio/blog/README/socials Discord ID/handle, exact
   login/twitter/name, pr_link_poster signal excluding staff) against a 7-day
   priority pool first, then full member list fallback.
5. List-wise LLM selection (_select_with_llm, up to 5 candidates per author,
   explicit 'none of these', via MATCH_MODEL in a thread pool) + 7-day negative
   cache (match_attempts) invalidated on signature or pr_link_posters changes.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import hashlib
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
from urllib.parse import urlsplit

from .api import (
    GITHUB_API,
    TransientAPIError,
    fetch_guild_members,
    github_get_json,
    github_graphql,
    github_rate_limit_status,
    search_guild_members,
)
from .config import is_discord_id
from .schedule import UTC, iso, parse_time

DEFAULT_MATCH_MODEL = "anthropic/claude-sonnet-4-5"
LLM_MULTI_SIGNAL_THRESHOLD = 0.80
LLM_SINGLE_SIGNAL_THRESHOLD = 0.90
PR_LINK_POSTER_CONFIDENCE = 0.92
PR_LINK_POSTER_NAME_SIMILARITY_THRESHOLD = 0.50
MAX_CANDIDATES_PER_AUTHOR = 5
FUZZY_CANDIDATE_THRESHOLD = 0.72
MATCH_CACHE_TTL_DAYS = 30
GITHUB_USER_CACHE_TTL_DAYS = 7
NEGATIVE_CACHE_TTL_DAYS = 7
PRIORITY_POOL_JOINED_DAYS = 7
RECENT_POSTERS_MAX_PER_GUILD = 5000
GRAPHQL_PROFILE_BATCH_SIZE = 50
PROFILE_README_MAX_BYTES = 4096
ENRICH_WORKERS = 8
LLM_SELECT_WORKERS = 8
REST_MEMBERS_CACHE_TTL_SECONDS = 900.0
GUILD_FETCH_TIMEOUT_SECONDS = 60.0

_rest_members_lock = threading.Lock()
_rest_members_cache: dict[str, tuple[list["DiscordMemberInfo"], float]] = {}
_recent_posters_lock = threading.Lock()
_recent_posters: dict[str, dict[str, datetime]] = {}

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
_PR_LINK_OWNERSHIP_RE = re.compile(
    r"\b(?:my\s+pr|i\s+(?:opened|made|raised|submitted))\b",
    re.IGNORECASE,
)
_GENERIC_WEBSITE_HOSTS = frozenset({
    "github.com", "gist.github.com", "gitlab.com", "bitbucket.org",
    "linkedin.com", "twitter.com", "x.com", "bsky.app", "mastodon.social",
    "medium.com", "dev.to", "hashnode.dev", "substack.com", "notion.so", "notion.site",
    "linktr.ee", "bio.link", "beacons.ai", "about.me", "carrd.co",
    "youtube.com", "youtu.be", "twitch.tv", "instagram.com", "facebook.com", "reddit.com",
    "discord.com", "discordapp.com", "discord.gg", "t.me", "telegram.me",
    "google.com", "gmail.com", "huggingface.co", "kaggle.com", "stackoverflow.com",
    "vercel.app", "netlify.app", "pages.dev", "github.io", "herokuapp.com",
    "wordpress.com", "blogspot.com", "wixsite.com", "squarespace.com",
})
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:inc\.?|llc\.?|ltd\.?|corp\.?|corporation|company|co\.?|gmbh|labs?|ai|tech|technologies|software|systems)\b",
    re.IGNORECASE,
)
_DEFINITIVE_NEGATIVE_REASONS = frozenset({
    "no_signal_match",
    "llm_none_of_these",
    "llm_below_threshold",
    "ambiguous_tie",
})

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
    with _recent_posters_lock:
        _recent_posters.clear()


def record_recent_poster(guild_id: str, discord_id: str, *, at: datetime | None = None) -> None:
    """Cheaply remember a guild member who recently posted a message (in-memory, bounded)."""
    if not is_discord_id(str(guild_id or "")) or not is_discord_id(str(discord_id or "")):
        return
    moment = at or datetime.now(UTC)
    key = str(guild_id)
    uid = str(discord_id)
    with _recent_posters_lock:
        bucket = _recent_posters.setdefault(key, {})
        bucket[uid] = moment
        if len(bucket) > RECENT_POSTERS_MAX_PER_GUILD:
            cutoff = moment - timedelta(days=PRIORITY_POOL_JOINED_DAYS)
            pruned = {k: v for k, v in bucket.items() if v >= cutoff}
            if len(pruned) > RECENT_POSTERS_MAX_PER_GUILD:
                newest = sorted(pruned.items(), key=lambda item: item[1], reverse=True)
                pruned = dict(newest[:RECENT_POSTERS_MAX_PER_GUILD])
            _recent_posters[key] = pruned


def recent_poster_ids_for_guild(
    guild_id: str | None,
    *,
    now: datetime | None = None,
    days: int = PRIORITY_POOL_JOINED_DAYS,
) -> frozenset[str]:
    if not guild_id or not is_discord_id(str(guild_id)):
        return frozenset()
    moment = now or datetime.now(UTC)
    cutoff = moment - timedelta(days=days)
    with _recent_posters_lock:
        bucket = _recent_posters.get(str(guild_id))
        if not bucket:
            return frozenset()
        return frozenset(uid for uid, seen_at in bucket.items() if seen_at >= cutoff)


@dataclass(frozen=True)
class GitHubAuthorProfile:
    login: str
    user_id: int | None = None
    name: str = ""
    blog: str = ""
    twitter_username: str = ""
    bio: str = ""
    company: str = ""
    email: str = ""
    social_accounts: tuple[dict[str, str], ...] = ()
    readme: str = ""
    commit_names: tuple[str, ...] = ()
    email_locals: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscordMemberInfo:
    discord_id: str
    username: str = ""
    global_name: str = ""
    nick: str = ""
    display_name: str = ""
    joined_at: datetime | None = None
    role_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PRLinkPosterEvidence:
    discord_id: str
    pr_numbers: tuple[int, ...]
    message_texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMSelectionResult:
    discord_id: str | None
    confidence: float
    reason: str
    evidence: tuple[str, ...] = ()
    model: str = ""


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
    evidence: tuple[str, ...] = ()


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


def _extract_website_domain_norm(url: str) -> str:
    """Extract a personal/vanity domain stem (e.g. 'janedoe.dev' -> 'janedoe'), ignoring generic hosts."""
    raw = (url or "").strip()
    if not raw:
        return ""
    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        host = (urlsplit(candidate).hostname or "").strip().lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    if not host or "." not in host:
        return ""
    if host in _GENERIC_WEBSITE_HOSTS:
        return ""
    parts = host.split(".")
    if len(parts) >= 2 and ".".join(parts[-2:]) in _GENERIC_WEBSITE_HOSTS:
        # Subdomain of a generic host (e.g. janedoe.github.io) -> use subdomain if distinctive!
        sub = parts[0]
        norm = normalize_strict(sub)
        return norm if len(norm) >= 4 else ""
    stem = parts[0]
    norm = normalize_strict(stem)
    return norm if len(norm) >= 4 else ""


def _extract_company_norm(company: str) -> str:
    cleaned = (company or "").strip().lstrip("@")
    if not cleaned:
        return ""
    cleaned = _COMPANY_SUFFIX_RE.sub("", cleaned).strip(" ,.-_/|@")
    norm = normalize_strict(cleaned)
    return norm if len(norm) >= 4 else ""


def _extract_social_discord_ids_and_handles(
    social_accounts: tuple[dict[str, str], ...],
) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    handles: list[str] = []
    for account in social_accounts:
        if not isinstance(account, dict):
            continue
        provider = str(account.get("provider") or "").strip().upper()
        display = str(account.get("displayName") or account.get("display_name") or "").strip()
        url = str(account.get("url") or "").strip()
        for uid in extract_discord_ids_from_text(f"{display} {url}"):
            if uid not in ids:
                ids.append(uid)
        if provider == "DISCORD" or "discord" in url.lower():
            for raw in (display,):
                cleaned = raw.lstrip("@").split("#", 1)[0].strip()
                if (
                    len(cleaned) >= 2
                    and cleaned.casefold() not in _IGNORED_DISCORD_HANDLE_WORDS
                    and cleaned.casefold() not in {h.casefold() for h in handles}
                ):
                    handles.append(cleaned)
            for h in extract_discord_handles_from_text(f"{display} {url}"):
                if h.casefold() not in {x.casefold() for x in handles}:
                    handles.append(h)
    return ids, handles


def _serialize_profile_cache(profile: GitHubAuthorProfile) -> str:
    return json.dumps({
        "company": profile.company,
        "email": profile.email,
        "social_accounts": list(profile.social_accounts[:10]),
        "readme": profile.readme[:PROFILE_README_MAX_BYTES],
        "commit_names": list(profile.commit_names[:8]),
        "email_locals": list(profile.email_locals[:8]),
    })


def _save_cached_github_profile(store, profile: GitHubAuthorProfile, moment: datetime) -> None:
    if store is None or not hasattr(store, "save_cached_github_user"):
        return
    try:
        store.save_cached_github_user(
            profile.login,
            name=profile.name,
            blog=profile.blog,
            twitter_username=profile.twitter_username,
            bio=profile.bio,
            email_local=profile.email_locals[0] if profile.email_locals else "",
            commit_name=profile.commit_names[0] if profile.commit_names else "",
            profile_json=_serialize_profile_cache(profile),
            fetched_at=iso(moment),
        )
    except Exception as cache_error:
        logging.warning("Could not cache GitHub profile for %s: %s", profile.login, cache_error)


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
    extra: dict = {}
    raw_json = cached_user.get("profile_json")
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                extra = parsed
        except ValueError:
            extra = {}
    commit_names: list[str] = []
    for cn in extra.get("commit_names") or []:
        if isinstance(cn, str) and cn.strip() and cn.strip().casefold() not in {c.casefold() for c in commit_names}:
            commit_names.append(cn.strip())
    cached_commit = (cached_user.get("commit_name") or "").strip()
    if cached_commit and cached_commit.casefold() not in {c.casefold() for c in commit_names}:
        commit_names.insert(0, cached_commit)

    email_locals: list[str] = []
    for el in extra.get("email_locals") or []:
        if isinstance(el, str) and el.strip() and el.strip().casefold() not in {e.casefold() for e in email_locals}:
            email_locals.append(el.strip())
    cached_email = (cached_user.get("email_local") or "").strip()
    if cached_email and cached_email.casefold() not in {e.casefold() for e in email_locals}:
        email_locals.insert(0, cached_email)

    raw_socials = extra.get("social_accounts") if isinstance(extra.get("social_accounts"), list) else []
    socials = tuple(
        {
            "provider": str(item.get("provider") or ""),
            "displayName": str(item.get("displayName") or ""),
            "url": str(item.get("url") or ""),
        }
        for item in raw_socials
        if isinstance(item, dict)
    )[:10]

    return GitHubAuthorProfile(
        login=login,
        user_id=user_id,
        name=(cached_user.get("name") or "").strip(),
        blog=(cached_user.get("blog") or "").strip(),
        twitter_username=(cached_user.get("twitter_username") or "").strip().lstrip("@"),
        bio=(cached_user.get("bio") or "").strip(),
        company=str(extra.get("company") or "").strip(),
        email=str(extra.get("email") or "").strip(),
        social_accounts=socials,
        readme=str(extra.get("readme") or "")[:PROFILE_README_MAX_BYTES],
        commit_names=tuple(commit_names),
        email_locals=tuple(email_locals),
    )


def fetch_github_profiles_graphql(
    logins: list[str],
    token: str | None,
    *,
    graphql_caller=None,
    batch_size: int = GRAPHQL_PROFILE_BATCH_SIZE,
    timeout: float = 20.0,
    deadline: float | None = None,
) -> tuple[dict[str, GitHubAuthorProfile], bool]:
    """Batch-fetch up to 50 GitHub user profiles + 4KB profile READMEs per GraphQL query."""
    caller = graphql_caller or github_graphql
    clean_logins: list[str] = []
    seen: set[str] = set()
    for raw in logins:
        cleaned = (raw or "").strip()
        if cleaned and re.fullmatch(r"[A-Za-z0-9-]+", cleaned) and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            clean_logins.append(cleaned)
    if not clean_logins:
        return {}, False
    if graphql_caller is None and not token:
        return {}, False

    step = max(1, min(int(batch_size), GRAPHQL_PROFILE_BATCH_SIZE))
    profiles: dict[str, GitHubAuthorProfile] = {}
    rate_limited = False

    for offset in range(0, len(clean_logins), step):
        if deadline is not None and (deadline - time.monotonic()) <= 1.5:
            rate_limited = True
            break
        batch = clean_logins[offset: offset + step]
        var_defs = ", ".join(f"$l{idx}: String!" for idx in range(len(batch)))
        selections = " ".join(
            f"u{idx}: user(login: $l{idx}) {{ ...U profileRepo: repository(name: $l{idx}) {{ ...R }} }}"
            for idx in range(len(batch))
        )
        query = (
            f"query BatchProfiles({var_defs}) {{ {selections} }} "
            "fragment U on User { "
            "login name bio company email twitterUsername websiteUrl "
            "socialAccounts(first: 10) { nodes { provider displayName url } } "
            "} "
            "fragment R on Repository { "
            'object(expression: "HEAD:README.md") { ... on Blob { text } } '
            "}"
        )
        variables = {f"l{idx}": login for idx, login in enumerate(batch)}
        req_timeout = min(timeout, max(3.0, deadline - time.monotonic())) if deadline is not None else timeout
        try:
            if graphql_caller is not None:
                payload = caller(token, query, variables)
            else:
                payload = caller(token, query, variables, timeout=req_timeout, attempts=1, max_retry_delay=0.0)
        except Exception as error:
            if _is_rate_limit_error(error):
                rate_limited = True
            logging.warning("GraphQL profile batch fetch failed (%d logins): %s", len(batch), error)
            break

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            continue
        for idx, requested_login in enumerate(batch):
            node = data.get(f"u{idx}")
            if not isinstance(node, dict):
                continue
            login_val = str(node.get("login") or requested_login).strip()
            name = str(node.get("name") or "").strip()
            bio = str(node.get("bio") or "").strip()
            company = str(node.get("company") or "").strip()
            email = str(node.get("email") or "").strip()
            twitter = str(node.get("twitterUsername") or "").strip().lstrip("@")
            website = str(node.get("websiteUrl") or "").strip()
            raw_socials = (
                (node.get("socialAccounts") or {}).get("nodes")
                if isinstance(node.get("socialAccounts"), dict)
                else []
            )
            socials = tuple(
                {
                    "provider": str(item.get("provider") or ""),
                    "displayName": str(item.get("displayName") or ""),
                    "url": str(item.get("url") or ""),
                }
                for item in (raw_socials or ())
                if isinstance(item, dict)
            )[:10]
            profile_repo = node.get("profileRepo") if isinstance(node.get("profileRepo"), dict) else {}
            readme_obj = profile_repo.get("object") if isinstance(profile_repo.get("object"), dict) else {}
            raw_readme = readme_obj.get("text") if isinstance(readme_obj.get("text"), str) else ""
            readme = raw_readme.encode("utf-8", errors="ignore")[:PROFILE_README_MAX_BYTES].decode("utf-8", errors="ignore")

            email_locals: list[str] = []
            local = _extract_email_local(email)
            if local:
                email_locals.append(local)

            profiles[requested_login.casefold()] = GitHubAuthorProfile(
                login=login_val,
                name=name,
                blog=website,
                twitter_username=twitter,
                bio=bio,
                company=company,
                email=email,
                social_accounts=socials,
                readme=readme,
                email_locals=tuple(email_locals),
            )
    return profiles, rate_limited


def _fetch_commit_signals_graphql(
    repository: str,
    pr_by_login: dict[str, list[int]],
    token: str | None,
    *,
    graphql_caller=None,
    batch_size: int = GRAPHQL_PROFILE_BATCH_SIZE,
    timeout: float = 20.0,
    deadline: float | None = None,
) -> tuple[dict[str, tuple[tuple[str, ...], tuple[str, ...]]], bool]:
    """Batch-fetch commit author names & email locals for up to 50 PRs per GraphQL query."""
    if not repository or "/" not in repository or not pr_by_login:
        return {}, False
    if graphql_caller is None and not token:
        return {}, False
    owner, repo_name = repository.split("/", 1)
    caller = graphql_caller or github_graphql

    # Pick up to 2 PRs per login to keep query volume tiny.
    pr_to_logins: dict[int, list[str]] = {}
    for login_key, nums in pr_by_login.items():
        for num in nums[:2]:
            if isinstance(num, int) and num > 0:
                pr_to_logins.setdefault(num, []).append(login_key)
    pr_numbers = sorted(pr_to_logins.keys())
    if not pr_numbers:
        return {}, False

    names_by_login: dict[str, list[str]] = {}
    emails_by_login: dict[str, list[str]] = {}
    rate_limited = False
    step = max(1, min(int(batch_size), GRAPHQL_PROFILE_BATCH_SIZE))

    for offset in range(0, len(pr_numbers), step):
        if deadline is not None and (deadline - time.monotonic()) <= 1.5:
            rate_limited = True
            break
        batch = pr_numbers[offset: offset + step]
        selections = " ".join(
            f"p{idx}: pullRequest(number: {num}) {{ "
            "commits(first: 10) { nodes { commit { author { name email user { login } } } } } "
            "}"
            for idx, num in enumerate(batch)
        )
        query = f"query BatchPRCommits($owner: String!, $repo: String!) {{ repository(owner: $owner, name: $repo) {{ {selections} }} }}"
        variables = {"owner": owner, "repo": repo_name}
        req_timeout = min(timeout, max(3.0, deadline - time.monotonic())) if deadline is not None else timeout
        try:
            if graphql_caller is not None:
                payload = caller(token, query, variables)
            else:
                payload = caller(token, query, variables, timeout=req_timeout, attempts=1, max_retry_delay=0.0)
        except Exception as error:
            if _is_rate_limit_error(error):
                rate_limited = True
            logging.warning("GraphQL commit batch fetch failed (%d PRs): %s", len(batch), error)
            break

        repo_data = ((payload.get("data") or {}).get("repository")) if isinstance(payload, dict) else None
        if not isinstance(repo_data, dict):
            continue
        for idx, num in enumerate(batch):
            pr_node = repo_data.get(f"p{idx}")
            commits_conn = pr_node.get("commits") if isinstance(pr_node, dict) else None
            nodes = commits_conn.get("nodes") if isinstance(commits_conn, dict) else None
            if not isinstance(nodes, list):
                continue
            for login_key in pr_to_logins.get(num, ()):
                c_names = names_by_login.setdefault(login_key, [])
                e_locals = emails_by_login.setdefault(login_key, [])
                for node in nodes:
                    commit = node.get("commit") if isinstance(node, dict) else None
                    author = commit.get("author") if isinstance(commit, dict) else None
                    if not isinstance(author, dict):
                        continue
                    user_obj = author.get("user") if isinstance(author.get("user"), dict) else {}
                    commit_login = user_obj.get("login")
                    if isinstance(commit_login, str) and commit_login.strip().casefold() != login_key:
                        continue
                    raw_name = author.get("name")
                    if isinstance(raw_name, str):
                        cleaned = raw_name.strip()
                        if (
                            cleaned
                            and cleaned.casefold() not in _IGNORED_COMMIT_NAMES
                            and cleaned.casefold() not in {n.casefold() for n in c_names}
                        ):
                            c_names.append(cleaned)
                    local = _extract_email_local(str(author.get("email") or ""))
                    if local and local.casefold() not in {e.casefold() for e in e_locals}:
                        e_locals.append(local)

    out: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for login_key in set(names_by_login) | set(emails_by_login):
        out[login_key] = (
            tuple(names_by_login.get(login_key, ())),
            tuple(emails_by_login.get(login_key, ())),
        )
    return out, rate_limited


def _profile_has_enrichment_data(profile: GitHubAuthorProfile | None) -> bool:
    """True when GitHub enrichment populated at least one profile or commit field."""
    if profile is None:
        return False
    return bool(
        (profile.name or "").strip()
        or (profile.blog or "").strip()
        or (profile.twitter_username or "").strip()
        or (profile.bio or "").strip()
        or (profile.company or "").strip()
        or (profile.email or "").strip()
        or profile.social_accounts
        or (profile.readme or "").strip()
        or profile.commit_names
        or profile.email_locals
    )


def _enrich_profiles_batch(
    groups: list[dict],
    repository: str,
    token: str | None,
    *,
    store=None,
    moment: datetime,
    github_getter=None,
    graphql_caller=None,
    deadline: float | None = None,
    skip_github_enrichment: bool = False,
) -> tuple[dict[str, GitHubAuthorProfile], set[str]]:
    """Batch-enrich profiles via GraphQL (50/query) or ThreadPoolExecutor(8) REST fallback.

    Returns ``(profiles_by_login, deferred_logins)``.
    """
    profiles: dict[str, GitHubAuthorProfile] = {}
    deferred_logins: set[str] = set()
    uncached_groups: list[dict] = []

    for group in groups:
        login = group["login"]
        login_key = login.strip().casefold()
        if login_key in profiles:
            continue
        cached = _cached_github_profile(store, login, moment, user_id=group.get("user_id"))
        if cached is not None:
            profiles[login_key] = cached
        elif skip_github_enrichment:
            profiles[login_key] = GitHubAuthorProfile(login=login, user_id=group.get("user_id"))
        else:
            uncached_groups.append(group)

    if not uncached_groups:
        return profiles, deferred_logins

    use_graphql = graphql_caller is not None or (
        (github_getter is None or github_getter is github_get_json) and bool(token)
    )
    if use_graphql:
        logins = [g["login"] for g in uncached_groups]
        gql_profiles, hit_limit = fetch_github_profiles_graphql(
            logins,
            token,
            graphql_caller=graphql_caller,
            deadline=deadline,
        )
        if gql_profiles or hit_limit or graphql_caller is not None:
            pr_by_login: dict[str, list[int]] = {}
            for g in uncached_groups:
                login_key = g["login"].strip().casefold()
                nums = [
                    int(p["number"])
                    for p in g.get("pulls", ())
                    if isinstance(p, dict) and isinstance(p.get("number"), int) and p["number"] > 0
                ]
                if nums:
                    pr_by_login[login_key] = nums
            commit_signals: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
            if pr_by_login and not hit_limit:
                commit_signals, _commit_limit = _fetch_commit_signals_graphql(
                    repository,
                    pr_by_login,
                    token,
                    graphql_caller=graphql_caller,
                    deadline=deadline,
                )
            for g in uncached_groups:
                login = g["login"]
                login_key = login.strip().casefold()
                base_prof = gql_profiles.get(login_key)
                if base_prof is None:
                    deferred_logins.add(login_key)
                    profiles[login_key] = GitHubAuthorProfile(login=login, user_id=g.get("user_id"))
                    continue
                c_names, c_emails = commit_signals.get(login_key, ((), ()))
                merged_emails = list(base_prof.email_locals)
                for el in c_emails:
                    if el.casefold() not in {e.casefold() for e in merged_emails}:
                        merged_emails.append(el)
                full_prof = GitHubAuthorProfile(
                    login=base_prof.login,
                    user_id=g.get("user_id"),
                    name=base_prof.name,
                    blog=base_prof.blog,
                    twitter_username=base_prof.twitter_username,
                    bio=base_prof.bio,
                    company=base_prof.company,
                    email=base_prof.email,
                    social_accounts=base_prof.social_accounts,
                    readme=base_prof.readme,
                    commit_names=c_names,
                    email_locals=tuple(merged_emails),
                )
                profiles[login_key] = full_prof
                _save_cached_github_profile(store, full_prof, moment)
            return profiles, deferred_logins

    # Concurrent REST fallback (ThreadPoolExecutor(8)) when GraphQL is unavailable or github_getter is injected.
    rate_limited = threading.Event()

    def _worker(group: dict) -> tuple[str, GitHubAuthorProfile, bool]:
        login = group["login"]
        login_key = login.strip().casefold()
        if rate_limited.is_set() or (deadline is not None and time.monotonic() >= deadline):
            return login_key, GitHubAuthorProfile(login=login, user_id=group.get("user_id")), True
        pull_numbers = [
            int(p["number"])
            for p in group.get("pulls", ())
            if isinstance(p, dict) and isinstance(p.get("number"), int)
        ]
        prof, hit_limit = fetch_github_author_profile(
            login,
            repository,
            pull_numbers,
            token,
            user_id=group.get("user_id"),
            getter=github_getter,
            store=store,
            now=moment,
            deadline=deadline,
        )
        if hit_limit:
            rate_limited.set()
        return login_key, prof, hit_limit

    workers = min(ENRICH_WORKERS, max(1, len(uncached_groups)))
    if workers == 1:
        for group in uncached_groups:
            login_key, prof, was_deferred = _worker(group)
            profiles[login_key] = prof
            if was_deferred:
                deferred_logins.add(login_key)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, group) for group in uncached_groups]
            for fut in as_completed(futures):
                login_key, prof, was_deferred = fut.result()
                profiles[login_key] = prof
                if was_deferred:
                    deferred_logins.add(login_key)
    return profiles, deferred_logins


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
    company = ""
    email = ""
    social_accounts: tuple[dict[str, str], ...] = ()
    readme = ""
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
            company = cached_profile.company
            email = cached_profile.email
            social_accounts = cached_profile.social_accounts
            readme = cached_profile.readme
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
                        company = (profile.get("company") or "").strip() if isinstance(profile.get("company"), str) else ""
                        email = (profile.get("email") or "").strip() if isinstance(profile.get("email"), str) else ""
                        _add_email(email)
                        _save_cached_github_profile(
                            store,
                            GitHubAuthorProfile(
                                login=login,
                                user_id=user_id,
                                name=name,
                                blog=blog,
                                twitter_username=twitter,
                                bio=bio,
                                company=company,
                                email=email,
                                social_accounts=social_accounts,
                                readme=readme,
                                commit_names=tuple(commit_names),
                                email_locals=tuple(email_locals),
                            ),
                            moment,
                        )
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
        if commits_added and (profile_fetched or used_cached_user):
            _save_cached_github_profile(
                store,
                GitHubAuthorProfile(
                    login=login,
                    user_id=user_id,
                    name=name,
                    blog=blog,
                    twitter_username=twitter,
                    bio=bio,
                    company=company,
                    email=email,
                    social_accounts=social_accounts,
                    readme=readme,
                    commit_names=tuple(commit_names),
                    email_locals=tuple(email_locals),
                ),
                moment,
            )

    return (
        GitHubAuthorProfile(
            login=login,
            user_id=user_id,
            name=name,
            blog=blog,
            twitter_username=twitter,
            bio=bio,
            company=company,
            email=email,
            social_accounts=social_accounts,
            readme=readme,
            commit_names=tuple(commit_names),
            email_locals=tuple(email_locals),
        ),
        rate_limited,
    )


def _parse_joined_at(raw: object) -> datetime | None:
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=UTC) if raw.tzinfo is None else raw.astimezone(UTC)
    if isinstance(raw, str) and raw.strip():
        try:
            return parse_time(raw)
        except Exception:
            return None
    return None


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
        raw_roles = member.get("roles") if isinstance(member.get("roles"), list) else []
        role_ids = tuple(str(r) for r in raw_roles if is_discord_id(str(r)))
        return DiscordMemberInfo(
            discord_id=str(member_id),
            username=username,
            global_name=global_name,
            nick=nick,
            display_name=display_name,
            joined_at=_parse_joined_at(member.get("joined_at")),
            role_ids=role_ids,
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
    obj_roles = getattr(member, "roles", None) or ()
    role_ids_list: list[str] = []
    for role in obj_roles:
        rid = str(getattr(role, "id", role) or "")
        if is_discord_id(rid):
            role_ids_list.append(rid)
    return DiscordMemberInfo(
        discord_id=str(member_id),
        username=username,
        global_name=global_name,
        nick=nick,
        display_name=display_name,
        joined_at=_parse_joined_at(getattr(member, "joined_at", None)),
        role_ids=tuple(role_ids_list),
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


def _member_login_fields(member: DiscordMemberInfo) -> tuple[str, ...]:
    """Fields eligible for exact_login matching: username and global_name ONLY (never nick)."""
    seen: set[str] = set()
    out: list[str] = []
    for val in (member.username, member.global_name):
        cleaned = (val or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return tuple(out)


def _member_fields(member: DiscordMemberInfo) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for val in (member.username, member.global_name, member.nick, member.display_name):
        cleaned = (val or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return tuple(out)


def _tokens_for_fields(fields: tuple[str, ...]) -> tuple[set[str], set[str], set[str]]:
    raw_lower: set[str] = set()
    strict_set: set[str] = set()
    base_set: set[str] = set()
    for field in fields:
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


def _member_tokens(member: DiscordMemberInfo) -> tuple[set[str], set[str], set[str]]:
    return _tokens_for_fields(_member_fields(member))


class _MemberIndex:
    """Precomputed O(1) lookup tables over a guild's members for fast matching at scale."""

    def __init__(self, members: list[DiscordMemberInfo]):
        self.members = list(members)
        self.by_id: dict[str, DiscordMemberInfo] = {m.discord_id: m for m in members}
        self.entry_idx_by_id: dict[str, int] = {}
        self.by_raw: dict[str, list[DiscordMemberInfo]] = {}
        self.by_strict: dict[str, list[DiscordMemberInfo]] = {}
        self.by_base: dict[str, list[DiscordMemberInfo]] = {}
        # Username/global_name-only tables for exact_login (never matches server nicknames).
        self.login_by_raw: dict[str, list[DiscordMemberInfo]] = {}
        self.login_by_strict: dict[str, list[DiscordMemberInfo]] = {}
        self.login_by_base: dict[str, list[DiscordMemberInfo]] = {}
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

            l_raw, l_strict, l_base = _tokens_for_fields(_member_login_fields(member))
            for tok in l_raw:
                self.login_by_raw.setdefault(tok, []).append(member)
            for tok in l_strict:
                self.login_by_strict.setdefault(tok, []).append(member)
            for tok in l_base:
                self.login_by_base.setdefault(tok, []).append(member)

            seen_bigrams: set[str] = set()
            for norm in norms:
                if len(norm) < 4:
                    continue
                for pos in range(len(norm) - 1):
                    bg = norm[pos: pos + 2]
                    if bg not in seen_bigrams:
                        seen_bigrams.add(bg)
                        self.by_bigram.setdefault(bg, []).append(idx)

    @staticmethod
    def _lookup_tables(
        value: str,
        raw_map: dict[str, list[DiscordMemberInfo]],
        strict_map: dict[str, list[DiscordMemberInfo]],
        base_map: dict[str, list[DiscordMemberInfo]],
    ) -> list[DiscordMemberInfo]:
        cleaned = (value or "").strip()
        if not cleaned:
            return []
        hits_by_id: dict[str, DiscordMemberInfo] = {}
        lowered = cleaned.casefold()
        if len(lowered) >= 3:
            for member in raw_map.get(lowered, ()):
                hits_by_id[member.discord_id] = member
        strict = normalize_strict(cleaned)
        if len(strict) >= 4:
            for member in strict_map.get(strict, ()):
                hits_by_id[member.discord_id] = member
        if hits_by_id:
            return list(hits_by_id.values())
        base = normalize_base(cleaned)
        if len(base) >= 4:
            for member in base_map.get(base, ()):
                hits_by_id[member.discord_id] = member
        return list(hits_by_id.values())

    def exact_login_hits(self, value: str) -> list[DiscordMemberInfo]:
        """Match exact_login ONLY against Discord username and global_name (never nick)."""
        return self._lookup_tables(value, self.login_by_raw, self.login_by_strict, self.login_by_base)

    def exact_tier_hits(self, value: str) -> list[DiscordMemberInfo]:
        """Match exact/strict normalized form first; strip trailing digit suffixes only if no strict hit."""
        return self._lookup_tables(value, self.by_raw, self.by_strict, self.by_base)

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
    domain_norm = _extract_website_domain_norm(profile.blog)
    if domain_norm:
        sources.append(("website_domain", domain_norm))
    for account in profile.social_accounts:
        if not isinstance(account, dict):
            continue
        display = str(account.get("displayName") or account.get("display_name") or "").strip().lstrip("@")
        if display:
            sources.append(("social_display", display))
    company_norm = _extract_company_norm(profile.company)
    if company_norm:
        sources.append(("company", company_norm))
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for label, value in sources:
        for norm in _name_variants(value):
            if len(norm) >= 4 and (label, norm) not in seen:
                seen.add((label, norm))
                out.append((label, norm))
    return out


def _author_member_name_similarity(profile: GitHubAuthorProfile, member: DiscordMemberInfo) -> float:
    """Maximum normalized SequenceMatcher similarity between full GitHub login/name and Discord username/global_name."""
    author_norms = {
        norm for val in (profile.login, profile.name)
        if val and (norm := normalize_strict(val))
    }
    member_norms = {
        norm for val in _member_login_fields(member)
        if (norm := normalize_strict(val))
    }
    best = 0.0
    for a in author_norms:
        if len(a) < 3:
            continue
        for b in member_norms:
            if len(b) < 3:
                continue
            if a == b:
                return 1.0
            if len(a) < 4 or len(b) < 4:
                continue
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            if ratio > best:
                best = ratio
            if a in b or b in a:
                sub_ratio = max(ratio, min(len(a), len(b)) / max(len(a), len(b)))
                if sub_ratio > best:
                    best = sub_ratio
    return best


def multi_author_poster_ids(
    pr_link_rows: list[dict],
    pr_number_to_author: dict[int, str],
) -> frozenset[str]:
    """Return Discord IDs who linked PRs belonging to >= 2 different GitHub authors (reviewers/triagers)."""
    authors_by_poster: dict[str, set[str]] = {}
    for row in pr_link_rows:
        uid = str(row.get("discord_id") or "")
        pr_num = row.get("pr_number")
        if not is_discord_id(uid) or not isinstance(pr_num, int):
            continue
        author = (pr_number_to_author.get(pr_num) or "").strip().lower()
        if author:
            authors_by_poster.setdefault(uid, set()).add(author)
    return frozenset(uid for uid, authors in authors_by_poster.items() if len(authors) >= 2)


def validate_pr_link_poster_match(
    profile: GitHubAuthorProfile,
    member: DiscordMemberInfo,
    ev: PRLinkPosterEvidence,
    *,
    eligible_posters: list[PRLinkPosterEvidence],
    excluded_discord_ids: frozenset[str] | set[str] = frozenset(),
    multi_author_ids: frozenset[str] | set[str] = frozenset(),
    rejected_discord_ids: frozenset[str] | set[str] = frozenset(),
) -> tuple[bool, str]:
    """Validate whether a PR-link poster qualifies for auto-accept under strict rules.

    Never auto-accepts on link count alone: requires sole eligible poster AND
    (ownership text like 'my pr' / 'I opened/made/raised/submitted' OR
    name similarity >= 0.50 to the GitHub login/name). Rejects staff and
    multi-author posters (reviewers/triagers).
    """
    uid = member.discord_id
    if uid in excluded_discord_ids:
        return False, "staff_or_org_member"
    if uid in multi_author_ids:
        return False, "multi_author_poster"
    if uid in rejected_discord_ids:
        return False, "rejected_pair"
    if not ev.pr_numbers:
        return False, "no_pr_links"
    if len(eligible_posters) != 1 or eligible_posters[0].discord_id != uid:
        return False, "multiple_posters"
    has_ownership = _has_pr_ownership_text(ev.message_texts)
    name_sim = _author_member_name_similarity(profile, member)
    if has_ownership or name_sim >= PR_LINK_POSTER_NAME_SIMILARITY_THRESHOLD:
        reasons = [f"pr_link_poster:{len(set(ev.pr_numbers))}_prs"]
        if has_ownership:
            reasons.append("ownership_text")
        if name_sim >= PR_LINK_POSTER_NAME_SIMILARITY_THRESHOLD:
            reasons.append(f"name_sim:{name_sim:.2f}")
        return True, ", ".join(reasons)
    return False, f"no_ownership_text_and_low_name_sim({name_sim:.2f})"


def _has_pr_ownership_text(message_texts: tuple[str, ...] | list[str]) -> bool:
    return any(isinstance(text, str) and _PR_LINK_OWNERSHIP_RE.search(text) for text in message_texts)


def _pr_link_evidence_from_rows(rows: list[dict]) -> list[PRLinkPosterEvidence]:
    grouped_prs: dict[str, set[int]] = {}
    grouped_texts: dict[str, list[str]] = {}
    for row in rows:
        uid = str(row.get("discord_id") or "")
        pr_num = row.get("pr_number")
        if not is_discord_id(uid) or not isinstance(pr_num, int) or pr_num <= 0:
            continue
        grouped_prs.setdefault(uid, set()).add(pr_num)
        text = str(row.get("message_text") or "").strip()
        if text:
            texts = grouped_texts.setdefault(uid, [])
            if text not in texts and len(texts) < 4:
                texts.append(text[:240])
    return [
        PRLinkPosterEvidence(
            discord_id=uid,
            pr_numbers=tuple(sorted(pr_nums)),
            message_texts=tuple(grouped_texts.get(uid, ())),
        )
        for uid, pr_nums in sorted(grouped_prs.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


def _excluded_staff_discord_ids(
    config,
    store,
    guild_members: list[DiscordMemberInfo],
    *,
    org_members: frozenset[str] = frozenset(),
    org: str = "",
) -> frozenset[str]:
    """Identify Discord IDs belonging to maintainers / BerriAI org staff so PR-link sharing excludes them."""
    excluded: set[str] = set()
    staff_logins = {
        login.strip().lower()
        for login in (getattr(config, "maintainer_logins", None) or ())
        if isinstance(login, str) and login.strip()
    }
    staff_logins.update(login.strip().lower() for login in org_members if isinstance(login, str) and login.strip())
    is_berri = (org or "").strip().lower() == "berriai"

    manual_map = getattr(config, "github_discord_map", None) or {}
    for login, uid in manual_map.items():
        low = login.strip().lower()
        if is_discord_id(str(uid)) and (
            low in staff_logins
            or (is_berri and (low.endswith("-berri") or low.endswith("-berriai") or low == "berriai"))
        ):
            excluded.add(str(uid))

    if store is not None and hasattr(store, "all_confirmed_matches"):
        try:
            for row in store.all_confirmed_matches():
                low = str(row.get("github_login") or "").strip().lower()
                uid = str(row.get("discord_id") or "")
                if is_discord_id(uid) and (
                    low in staff_logins
                    or (is_berri and (low.endswith("-berri") or low.endswith("-berriai") or low == "berriai"))
                ):
                    excluded.add(uid)
        except Exception:
            pass

    maintainer_role_id = getattr(config, "review_maintainer_role_id", None)
    for member in guild_members:
        if maintainer_role_id and maintainer_role_id in member.role_ids:
            excluded.add(member.discord_id)
            continue
        fields = _member_fields(member)
        for field in fields:
            low = field.strip().lower()
            strict = normalize_strict(field)
            if low in staff_logins or (strict and strict in {normalize_strict(s) for s in staff_logins}):
                excluded.add(member.discord_id)
                break
            if is_berri and (
                low.endswith("-berri")
                or low.endswith("-berriai")
                or low.endswith("_berri")
                or "berriai" in strict
            ):
                excluded.add(member.discord_id)
                break
    return frozenset(excluded)


def score_author_against_members(
    profile: GitHubAuthorProfile,
    members: list[DiscordMemberInfo],
    *,
    max_candidates: int = MAX_CANDIDATES_PER_AUTHOR,
    member_index: _MemberIndex | None = None,
    pr_link_posters: list[PRLinkPosterEvidence] | None = None,
    excluded_discord_ids: frozenset[str] | set[str] = frozenset(),
    multi_author_ids: frozenset[str] | set[str] = frozenset(),
    rejected_discord_ids: frozenset[str] | set[str] = frozenset(),
) -> tuple[tuple[DiscordMemberInfo, str, float] | None, list[CandidatePair]]:
    """Return either a unique deterministic strong match OR ranked fuzzy/ambiguous candidates."""
    index = member_index if member_index is not None else _MemberIndex(members)
    if not index.members:
        return None, []

    social_ids, social_handles = _extract_social_discord_ids_and_handles(profile.social_accounts)

    # 1. Direct Discord user URL/ID in bio, blog, README, or social accounts.
    id_sources = (
        ("bio_discord_id", extract_discord_ids_from_text(profile.bio)),
        ("blog_discord_id", extract_discord_ids_from_text(profile.blog)),
        ("readme_discord_id", extract_discord_ids_from_text(profile.readme)),
        ("social_discord_id", social_ids),
    )
    for source_name, uids in id_sources:
        hits = [
            index.by_id[uid]
            for uid in uids
            if uid in index.by_id and uid not in rejected_discord_ids
        ]
        if len(hits) == 1:
            return (hits[0], source_name, 0.99), []

    # 2. Explicit Discord handle in bio, blog, README, or social accounts.
    ambiguous_strong: dict[str, tuple[DiscordMemberInfo, list[str]]] = {}
    handle_sources = (
        ("bio_discord_handle", extract_discord_handles_from_text(profile.bio)),
        ("blog_discord_handle", extract_discord_handles_from_text(profile.blog)),
        ("readme_discord_handle", extract_discord_handles_from_text(profile.readme)),
        ("social_discord_handle", social_handles),
    )
    for source_name, handles in handle_sources:
        if not handles:
            continue
        hits_by_id: dict[str, DiscordMemberInfo] = {}
        for handle in handles:
            for member in index.exact_tier_hits(handle):
                if member.discord_id not in rejected_discord_ids:
                    hits_by_id[member.discord_id] = member
        hits = list(hits_by_id.values())
        if len(hits) == 1:
            return (hits[0], source_name, 0.98), []
        for member in hits:
            entry = ambiguous_strong.setdefault(member.discord_id, (member, []))
            entry[1].append(f"ambiguous_{source_name}")

    # 3. Exact normalized match of login (username/global_name ONLY, never nick),
    # twitter_username, or GitHub profile name.
    strong_tiers = (
        ("exact_login", profile.login, 0.96, index.exact_login_hits),
        ("twitter_match", profile.twitter_username, 0.95, index.exact_tier_hits),
        ("exact_name", profile.name, 0.93, index.exact_tier_hits),
    )
    for method, value, confidence, lookup_fn in strong_tiers:
        if not value:
            continue
        hits = [m for m in lookup_fn(value) if m.discord_id not in rejected_discord_ids]
        if len(hits) == 1:
            return (hits[0], method, confidence), []
        for member in hits:
            entry = ambiguous_strong.setdefault(member.discord_id, (member, []))
            entry[1].append(f"ambiguous_{method}")

    # 4. PR-link poster tier (after exact tiers, before fuzzy).
    # Excludes staff/BerriAI org members, multi-author reviewers/triagers (>=2 distinct authors),
    # and rejected pairs. NEVER auto-accepts on link count alone: requires ownership text
    # ('my pr' / 'I opened/made/raised/submitted') OR name similarity >= 0.50 to login/name.
    pr_poster_signals: dict[str, list[str]] = {}
    if pr_link_posters:
        eligible = [
            ev for ev in pr_link_posters
            if ev.discord_id in index.by_id
            and ev.discord_id not in excluded_discord_ids
            and ev.discord_id not in multi_author_ids
            and ev.discord_id not in rejected_discord_ids
            and ev.pr_numbers
        ]
        if len(eligible) == 1:
            ev = eligible[0]
            member = index.by_id[ev.discord_id]
            ok, _reason = validate_pr_link_poster_match(
                profile,
                member,
                ev,
                eligible_posters=eligible,
                excluded_discord_ids=excluded_discord_ids,
                multi_author_ids=multi_author_ids,
                rejected_discord_ids=rejected_discord_ids,
            )
            if ok:
                return (member, "pr_link_poster", PR_LINK_POSTER_CONFIDENCE), []
        for ev in eligible:
            distinct_prs = len(set(ev.pr_numbers))
            sigs = pr_poster_signals.setdefault(ev.discord_id, [])
            sigs.append(f"pr_link_poster:{distinct_prs}_prs")
            if _has_pr_ownership_text(ev.message_texts):
                sigs.append("pr_link_ownership_text")
            if ev.message_texts:
                snippet = re.sub(r"\s+", " ", ev.message_texts[0]).strip()[:120]
                if snippet:
                    sigs.append(f"pr_link_msg:{snippet}")

    # 5. Collect ambiguous strong collisions + PR-link poster signals + fuzzy/commit/email/social/domain/company candidates.
    source_norms = _profile_source_norms(profile)
    candidate_indices: set[int] = set()
    for uid in ambiguous_strong:
        idx = index.entry_idx_by_id.get(uid)
        if idx is not None:
            candidate_indices.add(idx)
    for uid in pr_poster_signals:
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
        if member.discord_id in rejected_discord_ids:
            continue
        extra_sigs: list[str] = []
        base_floor = 0.0
        if member.discord_id in ambiguous_strong:
            _m, sigs = ambiguous_strong[member.discord_id]
            extra_sigs.extend(sigs)
            base_floor = max(base_floor, 0.90)
        if member.discord_id in pr_poster_signals:
            extra_sigs.extend(pr_poster_signals[member.discord_id])
            base_floor = max(base_floor, 0.85)
        fuzzy_score, fuzzy_sigs = _fuzzy_score_entry(profile, source_norms, raw_l, strict_s, base_s, norms)
        if extra_sigs:
            combined_sigs = tuple(dict.fromkeys(extra_sigs + fuzzy_sigs))
            candidates.append(CandidatePair(profile, member, max(base_floor, fuzzy_score), combined_sigs))
        elif fuzzy_score >= FUZZY_CANDIDATE_THRESHOLD and fuzzy_sigs:
            candidates.append(CandidatePair(profile, member, fuzzy_score, tuple(fuzzy_sigs)))

    candidates.sort(key=lambda item: (-item.heuristic_score, item.member.discord_id))
    return None, candidates[:max_candidates]


def _has_llm_credentials() -> bool:
    return any(os.environ.get(name, "").strip() for name in _LLM_API_KEY_ENVS)


def _resolve_match_model(config) -> str:
    """Resolve list-wise match LLM: MATCH_MODEL -> existing config (review_model) -> Sonnet default."""
    configured = (getattr(config, "match_model", None) or "").strip()
    if configured:
        return configured
    env_model = os.environ.get("MATCH_MODEL", "").strip()
    if env_model:
        return env_model
    review_model = (getattr(config, "review_model", None) or "").strip()
    if review_model:
        return review_model
    return DEFAULT_MATCH_MODEL


def _select_with_llm(
    config,
    profile: GitHubAuthorProfile,
    candidates: list[CandidatePair],
    *,
    llm_complete=None,
    timeout: float = 20.0,
) -> LLMSelectionResult:
    """List-wise LLM selection across up to 5 Discord candidates with explicit 'none of these'."""
    if not candidates:
        return LLMSelectionResult(None, 0.0, "no_candidates", ())
    if llm_complete is None and not _has_llm_credentials():
        raise RuntimeError("No LLM API key configured for contributor matching")

    model = _resolve_match_model(config)
    bounded = candidates[:MAX_CANDIDATES_PER_AUTHOR]
    candidate_ids = {pair.member.discord_id for pair in bounded}

    payload = {
        "github_author": {
            "login": profile.login,
            "name": profile.name,
            "bio": profile.bio[:300],
            "company": profile.company[:120],
            "email": profile.email[:120],
            "twitter_username": profile.twitter_username,
            "website_url": profile.blog[:200],
            "social_accounts": list(profile.social_accounts[:10]),
            "readme_excerpt": profile.readme[:1000],
            "commit_names": list(profile.commit_names[:6]),
            "email_locals": list(profile.email_locals[:6]),
        },
        "candidates": [
            {
                "discord_id": pair.member.discord_id,
                "username": pair.member.username,
                "global_name": pair.member.global_name,
                "nick": pair.member.nick,
                "display_name": pair.member.display_name,
                "joined_at": iso(pair.member.joined_at) if pair.member.joined_at else None,
                "heuristic_score": round(pair.heuristic_score, 3),
                "signals": list(pair.signals),
            }
            for pair in bounded
        ],
        "options": [pair.member.discord_id for pair in bounded] + ["none of these"],
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You match a GitHub contributor to the correct Discord member from a shortlist of up to 5 candidates, "
                "or choose 'none of these' (discord_id: null) if none is clearly the same person. "
                "Never guess on common first names or weak coincidences. "
                "Return ONLY one JSON object with keys: "
                "discord_id (string matching one candidate's discord_id, or null for none of these), "
                "confidence (float 0.0 to 1.0), "
                "reason (short plain-English explanation), "
                "evidence (array of short concrete corroborating signals used, e.g. ['login matches username', 'commit name matches global_name', 'posted PR link']). "
                "No markdown fences."
            ),
        },
        {"role": "user", "content": json.dumps(payload)},
    ]

    if llm_complete is not None:
        raw = llm_complete(model, messages)
    else:
        from .review_llm import complete_litellm
        raw = complete_litellm(model, messages, timeout=timeout)

    parsed: dict = {}
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end >= start:
            try:
                loaded = json.loads(raw[start: end + 1])
                if isinstance(loaded, dict):
                    parsed = loaded
            except ValueError:
                parsed = {}

    raw_id = parsed.get("discord_id")
    if isinstance(raw_id, str):
        cleaned_id = raw_id.strip()
        if not cleaned_id or cleaned_id.lower() in {"null", "none", "none of these"}:
            selected_id: str | None = None
        elif cleaned_id in candidate_ids:
            selected_id = cleaned_id
        else:
            selected_id = None
    else:
        selected_id = None

    raw_conf = parsed.get("confidence")
    confidence = max(0.0, min(1.0, float(raw_conf))) if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool) else 0.0
    reason = str(parsed.get("reason") or "").strip()[:240]
    raw_evidence = parsed.get("evidence")
    evidence: list[str] = []
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            text = str(item or "").strip()
            if text and text.casefold() not in {e.casefold() for e in evidence}:
                evidence.append(text[:160])

    return LLMSelectionResult(
        discord_id=selected_id,
        confidence=confidence,
        reason=reason,
        evidence=tuple(evidence),
        model=model,
    )


def _llm_selection_accepted(result: LLMSelectionResult) -> bool:
    if not result.discord_id:
        return False
    if result.confidence >= LLM_SINGLE_SIGNAL_THRESHOLD:
        return True
    if result.confidence >= LLM_MULTI_SIGNAL_THRESHOLD and len(result.evidence) >= 2:
        return True
    return False


def _deterministic_candidate_fallback(
    candidates: list[CandidatePair],
) -> dict[str, tuple[DiscordMemberInfo, str, float]]:
    """Conservative fallback when no LLM is reachable.

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
    llm_complete=None,
    jev_transport=None,  # deprecated legacy kwarg kept for signature compatibility
    haiku_complete=None,  # deprecated alias for llm_complete
    deadline: float | None = None,
) -> tuple[
    dict[str, tuple[DiscordMemberInfo, str, float]],
    str,
    dict[str, str],
    dict[str, tuple[float, str]],
    dict[str, tuple[str, ...]],
]:
    """Run list-wise _select_with_llm per author in a thread pool (with deterministic fallback)."""
    del jev_transport
    if not candidates:
        return {}, "none", {}, {}, {}

    complete_fn = llm_complete if llm_complete is not None else haiku_complete
    by_login: dict[str, list[CandidatePair]] = {}
    for pair in candidates:
        by_login.setdefault(pair.profile.login.casefold(), []).append(pair)

    for login_key in by_login:
        by_login[login_key].sort(key=lambda item: (-item.heuristic_score, item.member.discord_id))
        by_login[login_key] = by_login[login_key][:MAX_CANDIDATES_PER_AUTHOR]

    if complete_fn is None and not _has_llm_credentials():
        fallback_winners = _deterministic_candidate_fallback(candidates)
        skip_reasons: dict[str, str] = {}
        skip_meta: dict[str, tuple[float, str]] = {}
        ev_map: dict[str, tuple[str, ...]] = {}
        for login_key, pairs in by_login.items():
            if login_key not in fallback_winners:
                skip_reasons[login_key] = "llm_unavailable"
                skip_meta[login_key] = (pairs[0].heuristic_score if pairs else 0.0, "no LLM credentials configured")
            else:
                ev_map[login_key] = pairs[0].signals
        source = "deterministic_fallback" if fallback_winners else "unavailable"
        return fallback_winners, source, skip_reasons, skip_meta, ev_map

    winners: dict[str, tuple[DiscordMemberInfo, str, float]] = {}
    skip_reasons = {}
    skip_meta = {}
    evidence_by_login: dict[str, tuple[str, ...]] = {}
    any_llm_succeeded = False
    model_used = _resolve_match_model(config)

    def _run_one(login_key: str, pairs: list[CandidatePair]):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 2.0:
                return login_key, None, RuntimeError("deadline reached")
            req_timeout = min(20.0, max(3.0, remaining))
        else:
            req_timeout = 20.0
        try:
            res = _select_with_llm(
                config,
                pairs[0].profile,
                pairs,
                llm_complete=complete_fn,
                timeout=req_timeout,
            )
            return login_key, res, None
        except Exception as error:
            return login_key, None, error

    items = list(by_login.items())
    workers = min(LLM_SELECT_WORKERS, max(1, len(items)))
    results_list = []
    if workers == 1:
        for login_key, pairs in items:
            results_list.append(_run_one(login_key, pairs))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one, login_key, pairs) for login_key, pairs in items]
            for fut in as_completed(futures):
                results_list.append(fut.result())

    failed_pairs: list[CandidatePair] = []
    for login_key, res, error in results_list:
        pairs = by_login[login_key]
        member_map = {pair.member.discord_id: pair.member for pair in pairs}
        top_heur = pairs[0].heuristic_score if pairs else 0.0
        if error is not None or res is None:
            logging.warning("List-wise LLM selection failed for %s: %s", login_key, error)
            failed_pairs.extend(pairs)
            skip_reasons[login_key] = "llm_unavailable"
            skip_meta[login_key] = (top_heur, str(error or "llm error")[:160])
            continue
        any_llm_succeeded = True
        if res.model:
            model_used = res.model
        if _llm_selection_accepted(res) and res.discord_id in member_map:
            winners[login_key] = (member_map[res.discord_id], "llm", res.confidence)
            ev_items = list(res.evidence)
            if res.reason and res.reason not in ev_items:
                ev_items.append(f"reason: {res.reason}")
            evidence_by_login[login_key] = tuple(ev_items)
        elif not res.discord_id:
            skip_reasons[login_key] = "llm_none_of_these"
            skip_meta[login_key] = (max(res.confidence, top_heur), res.reason or "none of these")
        else:
            skip_reasons[login_key] = "llm_below_threshold"
            detail = res.reason or f"confidence={res.confidence:.2f}, evidence={len(res.evidence)}"
            skip_meta[login_key] = (res.confidence, detail)

    if failed_pairs and not any_llm_succeeded:
        fallback_winners = _deterministic_candidate_fallback(failed_pairs)
        for login_key, win in fallback_winners.items():
            winners[login_key] = win
            skip_reasons.pop(login_key, None)
            skip_meta.pop(login_key, None)
            if login_key in by_login and by_login[login_key]:
                evidence_by_login[login_key] = by_login[login_key][0].signals
        return (
            winners,
            ("deterministic_fallback" if fallback_winners else "unavailable"),
            skip_reasons,
            skip_meta,
            evidence_by_login,
        )

    return winners, (f"llm:{model_used}" if any_llm_succeeded else "unavailable"), skip_reasons, skip_meta, evidence_by_login


def disambiguate_candidates(
    config,
    candidates: list[CandidatePair],
    *,
    llm_complete=None,
    jev_transport=None,
    haiku_complete=None,
) -> dict[str, tuple[DiscordMemberInfo, str, float]]:
    """Run list-wise LLM selection across ambiguous/fuzzy candidates (or deterministic fallback)."""
    winners, _source, _reasons, _meta, _ev = _disambiguate_with_reasons(
        config,
        candidates,
        llm_complete=llm_complete,
        jev_transport=jev_transport,
        haiku_complete=haiku_complete,
    )
    return winners


def _compute_attempt_signature(
    profile: GitHubAuthorProfile,
    candidates: list[CandidatePair],
    pr_link_rows: list[dict],
) -> str:
    """Stable SHA-256 signature over profile, candidate shortlist, and PR-link poster rows."""
    payload = {
        "profile": {
            "login": profile.login.casefold(),
            "name": profile.name,
            "bio": profile.bio,
            "company": profile.company,
            "email": profile.email,
            "twitter": profile.twitter_username,
            "blog": profile.blog,
            "socials": list(profile.social_accounts),
            "readme": profile.readme,
            "commit_names": list(profile.commit_names),
            "email_locals": list(profile.email_locals),
        },
        "candidates": [
            [c.member.discord_id, round(c.heuristic_score, 2), list(c.signals)]
            for c in candidates
        ],
        "pr_links": sorted(
            [int(r.get("pr_number") or 0), str(r.get("discord_id") or ""), str(r.get("message_id") or "")]
            for r in pr_link_rows
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _build_priority_pool(
    guild_members: list[DiscordMemberInfo],
    now: datetime,
    *,
    guild_id: str | None = None,
    extra_recent_ids: set[str] | frozenset[str] | None = None,
    days: int = PRIORITY_POOL_JOINED_DAYS,
) -> list[DiscordMemberInfo]:
    """Members who joined within ``days`` days OR posted recently in this guild."""
    cutoff = now - timedelta(days=days)
    recent_ids = set(recent_poster_ids_for_guild(guild_id, now=now, days=days))
    if extra_recent_ids:
        recent_ids.update(str(uid) for uid in extra_recent_ids if uid)
    pool: list[DiscordMemberInfo] = []
    for member in guild_members:
        if member.discord_id in recent_ids:
            pool.append(member)
            continue
        if member.joined_at is not None and member.joined_at >= cutoff:
            pool.append(member)
    return pool


def _author_pull_numbers(group: dict, store, repository: str) -> list[int]:
    nums: set[int] = set()
    for p in group.get("pulls", ()):
        if isinstance(p, dict) and isinstance(p.get("number"), int) and p["number"] > 0:
            nums.add(int(p["number"]))
    if store is not None and repository:
        uid = group.get("user_id")
        try:
            if isinstance(uid, int) and uid > 0 and hasattr(store, "pulls_for_author"):
                for row in store.pulls_for_author(repository, uid):
                    if isinstance(row.get("number"), int) and row["number"] > 0:
                        nums.add(int(row["number"]))
            elif hasattr(store, "pulls_for_author_login"):
                for row in store.pulls_for_author_login(repository, group["login"]):
                    if isinstance(row.get("number"), int) and row["number"] > 0:
                        nums.add(int(row["number"]))
        except Exception:
            pass
    return sorted(nums)


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
    graphql_caller=None,
    llm_complete=None,
    jev_transport=None,
    haiku_complete=None,
    rest_fetcher=None,
    rest_searcher=None,
    skip_github_enrichment: bool = False,
    deadline: float | None = None,
    recent_poster_ids: set[str] | frozenset[str] | None = None,
    org_members: frozenset[str] = frozenset(),
) -> tuple[list[dict], int, list[MatchDetail], MatchDiagnostics]:
    """Resolve community PR authors to LiteLLM Discord guild members with diagnostics."""
    moment = now or datetime.now(UTC)
    moment_iso = iso(moment)
    guild_id = getattr(config, "discord_guild_id", None) or (str(getattr(guild, "id", "")) if guild is not None else None)
    discord_token = getattr(config, "discord_token", None)
    complete_fn = llm_complete if llm_complete is not None else haiku_complete

    guild_members: list[DiscordMemberInfo] = []
    members_source = "none"
    if members is not None:
        guild_members = list(members)
        members_source = "injected"

    resolved: dict[str, tuple[dict, str, str, float]] = {}  # login_lower -> (group, discord_id, method, confidence)
    evidence_by_login: dict[str, tuple[str, ...]] = {}
    sqlite_resolved_keys: set[str] = set()
    pending: list[dict] = []
    skip_reason_by_login: dict[str, str] = {}
    skip_meta_by_login: dict[str, tuple[float, str]] = {}

    rejected_pairs: set[tuple[str, str]] = set()
    if store is not None and hasattr(store, "all_match_rejections"):
        try:
            rejected_pairs = store.all_match_rejections()
        except Exception as error:
            logging.warning("Could not load match_rejections: %s", error)
    rejected_by_login: dict[str, set[str]] = {}
    for r_login, r_uid in rejected_pairs:
        rejected_by_login.setdefault(r_login.casefold(), set()).add(r_uid)

    manual_map = getattr(config, "github_discord_map", None) or {}
    for group in groups:
        login = group["login"]
        login_key = login.strip().casefold()
        user_id = group.get("user_id")

        discord_id = group.get("discord_id")
        if discord_id and is_discord_id(str(discord_id)):
            resolved[login_key] = (group, str(discord_id), "prelinked", 1.0)
            evidence_by_login[login_key] = ("prelinked",)
            continue

        if store is not None and isinstance(user_id, int) and hasattr(store, "link_for_github_user"):
            link = store.link_for_github_user(user_id)
            if link and is_discord_id(link.get("discord_id")):
                resolved[login_key] = (group, str(link["discord_id"]), "link_store", 1.0)
                evidence_by_login[login_key] = ("github_links",)
                continue

        mapped_id = manual_map.get(login_key)
        if mapped_id and is_discord_id(str(mapped_id)):
            resolved[login_key] = (group, str(mapped_id), "config_map", 1.0)
            evidence_by_login[login_key] = ("GITHUB_DISCORD_MAP",)
            continue

        # Persisted SQLite automatic match (computed once, expires after MATCH_CACHE_TTL_DAYS).
        if store is not None and hasattr(store, "get_confirmed_match"):
            cached = store.get_confirmed_match(login)
            if cached and is_discord_id(cached.get("discord_id")) and _match_is_fresh(cached.get("matched_at"), moment):
                cid = str(cached["discord_id"])
                if (login_key, cid) not in rejected_pairs:
                    resolved[login_key] = (
                        group,
                        cid,
                        f"sqlite:{cached['method']}",
                        float(cached.get("confidence") or 1.0),
                    )
                    raw_ev = str(cached.get("evidence") or "").strip()
                    if raw_ev:
                        evidence_by_login[login_key] = (raw_ev,)
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

    if member_by_id and sqlite_resolved_keys:
        for login_key in list(sqlite_resolved_keys):
            group, cid, _method, _conf = resolved[login_key]
            if cid not in member_by_id:
                del resolved[login_key]
                pending.append(group)

    ambiguous_candidates: list[CandidatePair] = []
    llm_source = "none"
    attempt_signatures: dict[str, str] = {}

    cache_incomplete = guild is not None and not _cache_is_complete(guild, guild_members)
    if (
        members is None
        and pending
        and (not guild_members or (members_source == "guild_cache" and cache_incomplete))
        and guild_id
        and discord_token
    ):
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
    enrichment_blocked_for_llm: set[str] = set()
    if pending and not guild_members:
        for group in pending:
            login_key = group["login"].strip().casefold()
            skip_reason_by_login[login_key] = "no_guild_members"
            skip_meta_by_login[login_key] = (0.0, "guild member list empty")
    elif pending and guild_members:
        token = getattr(config, "github_token", None) or getattr(config, "github_actor_token", None)
        repository = getattr(config, "github_repository", "") or ""
        org = repository.split("/", 1)[0] if "/" in repository else ""
        pending_by_login = {g["login"].strip().casefold(): g for g in pending}
        enrich_deadline = (deadline - 20.0) if deadline is not None else None
        excluded_staff_ids = _excluded_staff_discord_ids(
            config, store, guild_members, org_members=org_members, org=org,
        )

        # Load PR-link poster evidence for all pending authors AND detect multi-author
        # reviewers/triagers across the whole repository's pr_link_posters table.
        pr_nums_by_login: dict[str, list[int]] = {}
        all_pr_nums: set[int] = set()
        pr_number_to_author: dict[int, str] = {}
        for group in groups:
            g_login = group["login"].strip().lower()
            for p in group.get("pulls", ()):
                if isinstance(p, dict) and isinstance(p.get("number"), int) and p["number"] > 0:
                    pr_number_to_author[int(p["number"])] = g_login
        for group in pending:
            login_key = group["login"].strip().casefold()
            nums = _author_pull_numbers(group, store, repository)
            pr_nums_by_login[login_key] = nums
            all_pr_nums.update(nums)
            for num in nums:
                pr_number_to_author[num] = login_key

        all_repo_pr_links: list[dict] = []
        pr_link_rows_by_pr: dict[int, list[dict]] = {}
        if store is not None and repository and hasattr(store, "all_pr_link_posters"):
            try:
                all_repo_pr_links = store.all_pr_link_posters(repository)
                for row in all_repo_pr_links:
                    pr_num = int(row["pr_number"])
                    if pr_num in all_pr_nums:
                        pr_link_rows_by_pr.setdefault(pr_num, []).append(row)
            except Exception as error:
                logging.warning("Could not load pr_link_posters for matching: %s", error)
        elif store is not None and repository and all_pr_nums and hasattr(store, "pr_link_posters_for_pulls"):
            try:
                all_repo_pr_links = store.pr_link_posters_for_pulls(repository, all_pr_nums)
                for row in all_repo_pr_links:
                    pr_link_rows_by_pr.setdefault(int(row["pr_number"]), []).append(row)
            except Exception as error:
                logging.warning("Could not load pr_link_posters for matching: %s", error)

        if all_repo_pr_links and store is not None and repository and hasattr(store, "pr_authors_for_numbers"):
            unknown_nums = {
                int(row["pr_number"])
                for row in all_repo_pr_links
                if isinstance(row.get("pr_number"), int) and int(row["pr_number"]) not in pr_number_to_author
            }
            if unknown_nums:
                try:
                    pr_number_to_author.update(store.pr_authors_for_numbers(repository, unknown_nums))
                except Exception as error:
                    logging.warning("Could not resolve PR authors for pr_link_posters: %s", error)

        multi_author_ids = multi_author_poster_ids(all_repo_pr_links, pr_number_to_author)

        pr_link_rows_by_login: dict[str, list[dict]] = {}
        pr_link_evidence_by_login: dict[str, list[PRLinkPosterEvidence]] = {}
        for login_key, nums in pr_nums_by_login.items():
            rows: list[dict] = []
            for num in nums:
                rows.extend(pr_link_rows_by_pr.get(num, ()))
            pr_link_rows_by_login[login_key] = rows
            pr_link_evidence_by_login[login_key] = _pr_link_evidence_from_rows(rows)

        recent_members = _build_priority_pool(
            guild_members,
            moment,
            guild_id=guild_id,
            extra_recent_ids=recent_poster_ids,
        )
        logging.info("match pool: recent=%d full=%d", len(recent_members), len(guild_members))
        use_priority_pool = bool(recent_members) and len(recent_members) < len(guild_members)
        recent_index = _MemberIndex(recent_members) if use_priority_pool else None

        # Fast path before GitHub API enrichment: unique exact_login match against
        # username/global_name ONLY (never nick), skipping rejected pairs.
        need_enrichment: list[dict] = []
        fast_index = recent_index if recent_index is not None else member_index
        for group in pending:
            login = group["login"]
            login_key = login.strip().casefold()
            rejected_ids = rejected_by_login.get(login_key, set())
            login_hits = [m for m in fast_index.exact_login_hits(login) if m.discord_id not in rejected_ids]
            if len(login_hits) == 1:
                hit = login_hits[0]
                resolved[login_key] = (group, hit.discord_id, "exact_login", 0.96)
                evidence_by_login[login_key] = (f"exact_login:{login}->{hit.username or hit.global_name}",)
                continue
            need_enrichment.append(group)

        # Batch-enrich remaining authors via GraphQL (50/query) or ThreadPoolExecutor(8) REST fallback.
        profiles_by_login, deferred_logins = _enrich_profiles_batch(
            need_enrichment,
            repository,
            token,
            store=store,
            moment=moment,
            github_getter=github_getter,
            graphql_caller=graphql_caller,
            deadline=enrich_deadline,
            skip_github_enrichment=skip_github_enrichment,
        )
        enrichment_deferred = len(deferred_logins)

        # Never allow the LLM to accept a match when GitHub profile enrichment was
        # deferred or returned zero profile/commit fields (treat as llm_unavailable
        # and retry next run rather than guessing from handle fragments alone).
        enrichment_blocked_for_llm: set[str] = set(deferred_logins)
        if not skip_github_enrichment:
            for group in need_enrichment:
                lk = group["login"].strip().casefold()
                if not _profile_has_enrichment_data(profiles_by_login.get(lk)):
                    enrichment_blocked_for_llm.add(lk)

        # Precompute full-pool scoring + negative-cache check for every still-pending author so
        # identical no_match attempts skip BOTH Pass 1 and Pass 2 LLM calls for 7 days.
        cached_negative_keys: set[str] = set()
        full_scores_cache: dict[str, tuple[tuple[DiscordMemberInfo, str, float] | None, list[CandidatePair]]] = {}
        for group in need_enrichment:
            login = group["login"]
            login_key = login.strip().casefold()
            profile = profiles_by_login.get(login_key) or GitHubAuthorProfile(login=login, user_id=group.get("user_id"))
            profiles_by_login[login_key] = profile
            full_strong, full_cands = score_author_against_members(
                profile,
                guild_members,
                member_index=member_index,
                pr_link_posters=pr_link_evidence_by_login.get(login_key),
                excluded_discord_ids=excluded_staff_ids,
                multi_author_ids=multi_author_ids,
                rejected_discord_ids=rejected_by_login.get(login_key, set()),
            )
            full_scores_cache[login_key] = (full_strong, full_cands)
            sig = _compute_attempt_signature(profile, full_cands, pr_link_rows_by_login.get(login_key, []))
            attempt_signatures[login_key] = sig
            if (
                full_strong is None
                and login_key not in enrichment_blocked_for_llm
                and store is not None
                and hasattr(store, "get_match_attempt")
            ):
                prev = store.get_match_attempt(login)
                if (
                    prev
                    and prev.get("outcome") == "no_match"
                    and prev.get("signature") == sig
                    and _match_is_fresh(prev.get("attempted_at"), moment, ttl_days=NEGATIVE_CACHE_TTL_DAYS)
                ):
                    cached_negative_keys.add(login_key)
                    skip_reason_by_login[login_key] = "cached_negative"
                    top_h = full_cands[0].heuristic_score if full_cands else 0.0
                    skip_meta_by_login[login_key] = (top_h, "cached negative match within 7d")

        def _strong_evidence(login_key: str, member: DiscordMemberInfo, method: str) -> tuple[str, ...]:
            if method == "pr_link_poster":
                evs = [e for e in pr_link_evidence_by_login.get(login_key, ()) if e.discord_id == member.discord_id]
                if evs:
                    ev = evs[0]
                    parts = [f"pr_link_poster:{len(set(ev.pr_numbers))}_prs({','.join(f'#{n}' for n in ev.pr_numbers[:5])})"]
                    if _has_pr_ownership_text(ev.message_texts):
                        parts.append("ownership_text")
                    sim = _author_member_name_similarity(profiles_by_login[login_key], member)
                    if sim >= PR_LINK_POSTER_NAME_SIMILARITY_THRESHOLD:
                        parts.append(f"name_sim:{sim:.2f}")
                    return tuple(parts)
            return (f"{method}:{member.username or member.global_name or member.discord_id}",)

        # Pass 1: priority pool (joined within 7 days / recent posters).
        if use_priority_pool and recent_index is not None:
            recent_candidates: list[CandidatePair] = []
            for group in need_enrichment:
                login_key = group["login"].strip().casefold()
                if login_key in resolved or login_key in cached_negative_keys:
                    continue
                profile = profiles_by_login[login_key]
                strong, candidates = score_author_against_members(
                    profile,
                    recent_members,
                    member_index=recent_index,
                    pr_link_posters=pr_link_evidence_by_login.get(login_key),
                    excluded_discord_ids=excluded_staff_ids,
                    multi_author_ids=multi_author_ids,
                    rejected_discord_ids=rejected_by_login.get(login_key, set()),
                )
                if strong is not None:
                    member, method, confidence = strong
                    resolved[login_key] = (group, member.discord_id, method, confidence)
                    evidence_by_login[login_key] = _strong_evidence(login_key, member, method)
                elif candidates and login_key not in enrichment_blocked_for_llm:
                    recent_candidates.extend(candidates)

            if recent_candidates:
                ambiguous_candidates.extend(recent_candidates)
                recent_winners, recent_llm_source, _recent_skips, _recent_meta, recent_ev = _disambiguate_with_reasons(
                    config,
                    recent_candidates,
                    llm_complete=complete_fn,
                    jev_transport=jev_transport,
                    deadline=(deadline - 10.0) if deadline is not None else None,
                )
                if recent_llm_source != "none":
                    llm_source = recent_llm_source
                for login_key, (member, method, confidence) in recent_winners.items():
                    if (login_key, member.discord_id) in rejected_pairs:
                        continue
                    if login_key not in resolved and login_key in pending_by_login:
                        resolved[login_key] = (pending_by_login[login_key], member.discord_id, method, confidence)
                        if login_key in recent_ev:
                            evidence_by_login[login_key] = recent_ev[login_key]

        # Pass 2: full member list fallback ONLY for authors still unmatched after Pass 1.
        full_candidates: list[CandidatePair] = []
        for group in need_enrichment:
            login_key = group["login"].strip().casefold()
            if login_key in resolved or login_key in cached_negative_keys:
                continue
            strong, candidates = full_scores_cache[login_key]
            if strong is not None:
                member, method, confidence = strong
                resolved[login_key] = (group, member.discord_id, method, confidence)
                evidence_by_login[login_key] = _strong_evidence(login_key, member, method)
            elif login_key in enrichment_blocked_for_llm:
                top_h = candidates[0].heuristic_score if candidates else 0.0
                skip_reason_by_login[login_key] = "llm_unavailable"
                skip_meta_by_login[login_key] = (top_h, "github profile enrichment deferred or empty; will retry next run")
                logging.info(
                    "Skipping LLM match for %s: GitHub profile enrichment was deferred or empty "
                    "(treating as llm_unavailable; will retry next run)",
                    group["login"],
                )
            elif candidates:
                full_candidates.extend(candidates)
            else:
                skip_reason_by_login[login_key] = "no_signal_match"
                skip_meta_by_login[login_key] = (0.0, "no candidate signals")

        if full_candidates:
            ambiguous_candidates.extend(full_candidates)
            llm_winners, full_llm_source, llm_skip_reasons, llm_skip_meta, full_ev = _disambiguate_with_reasons(
                config,
                full_candidates,
                llm_complete=complete_fn,
                jev_transport=jev_transport,
                deadline=(deadline - 10.0) if deadline is not None else None,
            )
            if full_llm_source != "none":
                llm_source = full_llm_source
            for login_key, (member, method, confidence) in llm_winners.items():
                if (login_key, member.discord_id) in rejected_pairs:
                    continue
                if login_key not in resolved and login_key in pending_by_login:
                    resolved[login_key] = (pending_by_login[login_key], member.discord_id, method, confidence)
                    if login_key in full_ev:
                        evidence_by_login[login_key] = full_ev[login_key]
            for login_key, reason in llm_skip_reasons.items():
                if login_key not in resolved:
                    skip_reason_by_login[login_key] = reason
                    if login_key in llm_skip_meta:
                        skip_meta_by_login[login_key] = llm_skip_meta[login_key]

    # Strict 1-to-1 bijection enforcement: one Discord member maps to at most one GitHub login
    # and vice versa. On ANY conflict (intra-batch OR against existing stored mappings), grant
    # nothing for the conflicting entries and log a WARNING.
    discord_to_logins: dict[str, set[str]] = {}
    login_to_discords: dict[str, set[str]] = {}
    for login_key, (_group, discord_id, _method, _conf) in resolved.items():
        discord_to_logins.setdefault(discord_id, set()).add(login_key)
        login_to_discords.setdefault(login_key, set()).add(discord_id)

    for m_login, m_uid in manual_map.items():
        low = str(m_login or "").strip().casefold()
        uid_s = str(m_uid or "").strip()
        if low and is_discord_id(uid_s):
            discord_to_logins.setdefault(uid_s, set()).add(low)
            login_to_discords.setdefault(low, set()).add(uid_s)

    if store is not None:
        if hasattr(store, "all_confirmed_matches"):
            try:
                for row in store.all_confirmed_matches():
                    r_login = str(row.get("github_login") or "").strip().casefold()
                    r_uid = str(row.get("discord_id") or "").strip()
                    if not (r_login and is_discord_id(r_uid) and (r_login, r_uid) not in rejected_pairs):
                        continue
                    r_method = str(row.get("method") or "").strip()
                    stale_departed = (
                        r_login in resolved
                        and resolved[r_login][1] != r_uid
                        and r_method != "mod_override"
                        and not _match_is_fresh(row.get("matched_at"), moment)
                        and bool(member_by_id)
                        and r_uid not in member_by_id
                        and not (hasattr(store, "grants_for_discord") and store.grants_for_discord(r_uid))
                    )
                    if stale_departed:
                        continue
                    discord_to_logins.setdefault(r_uid, set()).add(r_login)
                    login_to_discords.setdefault(r_login, set()).add(r_uid)
            except Exception:
                pass
        if hasattr(store, "all_links"):
            try:
                for row in store.all_links():
                    r_login = str(row.get("github_login") or "").strip().casefold()
                    r_uid = str(row.get("discord_id") or "").strip()
                    if r_login and is_discord_id(r_uid):
                        discord_to_logins.setdefault(r_uid, set()).add(r_login)
                        login_to_discords.setdefault(r_login, set()).add(r_uid)
            except Exception:
                pass

    winning_logins: set[str] = set()
    logged_conflicts: set[tuple[str, tuple[str, ...]]] = set()
    for login_key, (_group, discord_id, _method, confidence) in sorted(resolved.items()):
        mapped_logins = discord_to_logins.get(discord_id, set())
        mapped_discords = login_to_discords.get(login_key, set())
        if len(mapped_logins) > 1 or len(mapped_discords) > 1:
            conflict_key = (discord_id, tuple(sorted(mapped_logins)))
            if conflict_key not in logged_conflicts:
                logged_conflicts.add(conflict_key)
                logging.warning(
                    "1-to-1 match conflict for discord_id=%s (github_logins=%s, discord_ids_for_%s=%s); granting nothing",
                    discord_id,
                    sorted(mapped_logins),
                    login_key,
                    sorted(mapped_discords),
                )
            skip_reason_by_login[login_key] = "discord_id_conflict"
            skip_meta_by_login[login_key] = (
                confidence,
                f"1-to-1 conflict on discord_id={discord_id} (logins={sorted(mapped_logins)})",
            )
            continue
        winning_logins.add(login_key)

    matched_groups: list[dict] = []
    details: list[MatchDetail] = []
    method_counter: Counter[str] = Counter()
    for login_key in winning_logins:
        group, discord_id, method, confidence = resolved[login_key]
        ev_tuple = evidence_by_login.get(login_key, ())
        method_counter[method] += 1
        if store is not None and hasattr(store, "delete_match_attempt"):
            try:
                store.delete_match_attempt(group["login"])
            except Exception:
                pass
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
                    evidence=ev_tuple,
                )
            except Exception as error:
                logging.warning("Could not persist match for %s: %s", group["login"], error)

        matched_groups.append({
            **group,
            "discord_id": discord_id,
            "match_method": method,
            "match_confidence": confidence,
            "match_evidence": ev_tuple,
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
            evidence=ev_tuple,
        ))

    matched_groups.sort(key=lambda item: (-len(item["pulls"]), item["login"].lower()))
    details.sort(key=lambda item: (-item.pr_count, item.github_login.lower()))
    skipped_count = len(groups) - len(matched_groups)

    skip_counter: Counter[str] = Counter()
    for group in groups:
        login = group["login"]
        login_key = login.strip().casefold()
        if login_key not in winning_logins:
            reason = skip_reason_by_login.get(login_key, "unmatched")
            skip_counter[reason] += 1
            top_score, detail = skip_meta_by_login.get(login_key, (0.0, reason))
            logging.info(
                "Match skipped for %s: reason=%s top_score=%.2f detail=%s",
                login,
                reason,
                top_score,
                detail,
            )
            # Persist definitive no_match attempts for 7-day negative caching.
            if (
                reason in _DEFINITIVE_NEGATIVE_REASONS
                and login_key not in enrichment_blocked_for_llm
                and login_key in attempt_signatures
                and store is not None
                and hasattr(store, "save_match_attempt")
            ):
                try:
                    store.save_match_attempt(
                        login,
                        attempted_at=moment_iso,
                        signature=attempt_signatures[login_key],
                        outcome="no_match",
                    )
                except Exception as error:
                    logging.warning("Could not persist match_attempts row for %s: %s", login, error)

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
    logging.info("Match summary by method: %s", dict(diagnostics.method_counts))
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
    graphql_caller=None,
    llm_complete=None,
    jev_transport=None,
    haiku_complete=None,
    rest_fetcher=None,
    rest_searcher=None,
    skip_github_enrichment: bool = False,
    org_members: frozenset[str] = frozenset(),
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
        graphql_caller=graphql_caller,
        llm_complete=llm_complete,
        jev_transport=jev_transport,
        haiku_complete=haiku_complete,
        rest_fetcher=rest_fetcher,
        rest_searcher=rest_searcher,
        skip_github_enrichment=skip_github_enrichment,
        org_members=org_members,
    )
    return matched, skipped, details
