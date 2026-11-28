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
    search_guild_messages,
)
from .config import is_discord_id
from .schedule import UTC, iso, parse_time

DEFAULT_MATCH_MODEL = "anthropic/claude-sonnet-4-5"
LLM_CONFIDENCE_THRESHOLD = 0.85
LLM_MIN_SIGNALS = 2
LLM_MULTI_SIGNAL_THRESHOLD = LLM_CONFIDENCE_THRESHOLD
PR_LINK_POSTER_CONFIDENCE = 0.92
PR_LINK_POSTER_NAME_SIMILARITY_THRESHOLD = 0.50
MESSAGE_EVIDENCE_CLAIM_CONFIDENCE = 0.95
MESSAGE_EVIDENCE_OWNERSHIP_CONFIDENCE = 0.92
MESSAGE_EVIDENCE_STRONG_THRESHOLD = 0.90
MEMBER_MESSAGE_SCAN_MAX_PAGES = 2
MAX_MEMBER_MESSAGE_SCANS_PER_RUN = 25
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
ENRICH_RATE_LIMIT_MAX_RETRIES = 4
ENRICH_RATE_LIMIT_BASE_DELAY = 2.0
ENRICH_RATE_LIMIT_MAX_DELAY = 60.0
LLM_SELECT_WORKERS = 8
REST_MEMBERS_CACHE_TTL_SECONDS = 900.0
GUILD_FETCH_TIMEOUT_SECONDS = 60.0

_FREE_EMAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "ymail.com",
    "outlook.com", "hotmail.com", "live.com", "msn.com",
    "icloud.com", "me.com", "mac.com",
    "proton.me", "protonmail.com", "pm.me",
    "aol.com", "mail.com", "zoho.com", "gmx.com", "fastmail.com",
    "yandex.com", "yandex.ru", "qq.com", "163.com", "126.com",
    "users.noreply.github.com",
})

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
_MSG_OWNERSHIP_RE = re.compile(
    r"\b(?:my\s+(?:pr|pull\s+request|fix)|i\s+(?:opened|made|raised|submitted)|"
    r"i\s+(?:created|pushed|wrote|fixed)\s+(?:a|the|this)?\s*(?:pr|pull\s+request|#\d+)|"
    r"here(?:'s| is)\s+my\s+(?:pr|pull\s+request|fix))\b",
    re.IGNORECASE,
)
_GITHUB_PR_OR_ISSUE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/(pull|issues)/([1-9][0-9]*)\b",
    re.IGNORECASE,
)
_GITHUB_PROFILE_OR_REPO_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)(?:/([A-Za-z0-9_.-]+)(?:/[^\s)\]>]*)?)?(?=$|[\s)\]>,.;:!?'\"`#?])",
    re.IGNORECASE,
)
_GITHUB_SELF_HANDLE_RE = re.compile(
    r"(?:"
    r"\bmy\s+(?:github|gh)(?:\s+(?:username|handle|account|profile|id))?\s*(?:is|:|=)\s*|"
    r"\b(?:github|gh)(?:\s+(?:username|handle|account|profile))\s*(?:is|:|=)\s*|"
    r"\b(?:github|gh)\s*[:=]\s*"
    r")@?([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)\b",
    re.IGNORECASE,
)
_GITHUB_IM_ON_GH_RE = re.compile(
    r"\bi(?:'m| am)\s+@?([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)\s+on\s+(?:github|gh)\b",
    re.IGNORECASE,
)
_HASH_PR_NUM_RE = re.compile(r"(?<![A-Za-z0-9])#([1-9][0-9]{1,6})\b")
_CODE_PATH_RE = re.compile(
    r"\b([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.(?:py|ts|tsx|js|jsx|go|rs|yaml|yml|json|sql))\b"
)
_GITHUB_RESERVED_PATHS = frozenset({
    "about", "actions", "apps", "blob", "codespaces", "collections", "commit",
    "commits", "compare", "copilot", "customer-stories", "discussions", "enterprise",
    "events", "explore", "features", "gist", "issues", "join", "login", "marketplace",
    "new", "notifications", "orgs", "packages", "pricing", "projects", "pull", "pulls",
    "raw", "readme", "releases", "search", "security", "settings", "sponsors", "team",
    "topics", "tree", "trending", "users", "wiki",
})
_GENERIC_CODE_BASENAMES = frozenset({
    "__init__.py", "main.py", "utils.py", "types.py", "config.py", "index.ts", "index.js",
    "package.json", "pyproject.toml", "setup.py", "readme.md", "conftest.py",
})
_CODE_TITLE_STOPWORDS = frozenset({
    "a", "an", "and", "add", "adds", "adding", "allow", "also", "are", "as", "at", "be",
    "bug", "by", "can", "change", "changes", "chore", "code", "default", "do", "docs",
    "does", "feat", "feature", "file", "files", "fix", "fixes", "fixing", "for", "from",
    "get", "handle", "has", "have", "in", "into", "is", "issue", "it", "its", "litellm",
    "make", "model", "models", "more", "new", "not", "of", "on", "only", "or", "pr",
    "pull", "refactor", "remove", "request", "return", "set", "should", "support", "test",
    "tests", "that", "the", "this", "to", "update", "updates", "use", "using", "when", "with",
})
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
    email_domain: str = ""
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


def _rate_limit_backoff_delay(
    error: Exception,
    attempt: int,
    *,
    deadline: float | None = None,
) -> float | None:
    """Compute backoff delay in seconds before retrying a rate-limited GitHub enrichment call."""
    if attempt >= ENRICH_RATE_LIMIT_MAX_RETRIES:
        return None
    retry_after = getattr(error, "retry_after", None)
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        delay = float(retry_after)
    else:
        remaining, reset = github_rate_limit_status()
        if remaining is not None and remaining <= 1 and reset is not None and reset > time.time():
            delay = reset - time.time() + 0.5
        else:
            delay = ENRICH_RATE_LIMIT_BASE_DELAY * (2 ** attempt)
    delay = max(0.5, min(delay, ENRICH_RATE_LIMIT_MAX_DELAY))
    if deadline is not None:
        left = deadline - time.monotonic()
        if left <= 3.0:
            return None
        delay = min(delay, max(0.5, left - 2.5))
    return delay


def _call_github_enrichment(fetch, url: str, token: str | None, *, timeout: float = 10.0):
    if fetch is github_get_json:
        return github_get_json(url, token, pace=False, attempts=1, timeout=timeout, max_retry_delay=0.0)
    return fetch(url, token, pace=False)


def _extract_public_email_domain(email: str) -> str:
    """Extract non-free, non-noreply domain from a public GitHub email (e.g. 'alice@acme.io' -> 'acme.io')."""
    raw = (email or "").strip().lower()
    if "@" not in raw:
        return ""
    domain = raw.rsplit("@", 1)[1].strip(".")
    if not domain or "." not in domain:
        return ""
    if domain in _FREE_EMAIL_DOMAINS or domain.endswith(".noreply.github.com"):
        return ""
    return domain


def _extract_website_host(url: str) -> str:
    """Extract normalized non-generic website host (e.g. 'https://www.janedoe.dev/blog' -> 'janedoe.dev')."""
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
    if host in _GENERIC_WEBSITE_HOSTS or host in _FREE_EMAIL_DOMAINS:
        return ""
    return host


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
        "email_domain": profile.email_domain or _extract_public_email_domain(profile.email),
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

    cached_email_val = str(extra.get("email") or "").strip()
    cached_email_domain = str(extra.get("email_domain") or "").strip() or _extract_public_email_domain(cached_email_val)
    return GitHubAuthorProfile(
        login=login,
        user_id=user_id,
        name=(cached_user.get("name") or "").strip(),
        blog=(cached_user.get("blog") or "").strip(),
        twitter_username=(cached_user.get("twitter_username") or "").strip().lstrip("@"),
        bio=(cached_user.get("bio") or "").strip(),
        company=str(extra.get("company") or "").strip(),
        email=cached_email_val,
        email_domain=cached_email_domain,
        social_accounts=socials,
        readme=str(extra.get("readme") or "")[:PROFILE_README_MAX_BYTES],
        commit_names=tuple(commit_names),
        email_locals=tuple(email_locals),
    )


def _format_github_error(error: Exception) -> str:
    detail = getattr(error, "github_error_detail", "")
    if detail:
        return f"{error} ({detail})"
    return str(error)


def _inspect_graphql_errors(errors: object) -> tuple[bool, list[str], set[str]]:
    """Inspect GraphQL ``errors`` array.

    Returns ``(is_rate_limited, notable_error_summaries, not_found_user_aliases)``.
    Routine ``NOT_FOUND`` errors on ``[uN, "profileRepo"]`` (users without a
    ``<login>/<login>`` profile README repository) are ignored.
    """
    if not isinstance(errors, list):
        return False, [], set()
    rate_limited = False
    notable: list[str] = []
    not_found_aliases: set[str] = set()
    for err in errors:
        if not isinstance(err, dict):
            text = str(err).strip()
            if text:
                notable.append(text[:200])
            continue
        err_type = str(err.get("type") or "").strip()
        err_msg = str(err.get("message") or "").strip()
        err_path = err.get("path")
        if err_type == "RATE_LIMITED" or "rate limit" in err_msg.lower():
            rate_limited = True
        if (
            err_type == "NOT_FOUND"
            and isinstance(err_path, list)
            and len(err_path) == 2
            and err_path[1] == "profileRepo"
        ):
            continue
        if (
            err_type == "NOT_FOUND"
            and isinstance(err_path, list)
            and len(err_path) == 1
            and isinstance(err_path[0], str)
        ):
            not_found_aliases.add(err_path[0])
        piece = ": ".join(p for p in (err_type, err_msg) if p)
        if err_path:
            piece = f"{piece} (path={err_path})" if piece else f"path={err_path}"
        if piece:
            notable.append(piece[:200])
    return rate_limited, notable, not_found_aliases


def fetch_github_profiles_graphql(
    logins: list[str],
    token: str | None,
    *,
    graphql_caller=None,
    batch_size: int = GRAPHQL_PROFILE_BATCH_SIZE,
    timeout: float = 20.0,
    deadline: float | None = None,
    sleep_fn=time.sleep,
    on_batch=None,
) -> tuple[dict[str, GitHubAuthorProfile], bool]:
    """Batch-fetch up to 50 GitHub user profiles + 4KB profile READMEs per GraphQL query.

    Handles rate limits (HTTP and GraphQL-level) via exponential backoff + retry,
    logs actual GitHub error details, parses valid user nodes even alongside partial
    GraphQL errors, and bisects failed multi-login batches so one bad user never
    fails the rest of the batch.
    """
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

    pending_batches: list[list[str]] = [
        clean_logins[offset: offset + step]
        for offset in range(0, len(clean_logins), step)
    ]

    while pending_batches:
        batch = pending_batches.pop(0)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 1.5:
                logging.warning(
                    "GraphQL profile batch aborted before request (%d logins): deadline exhausted (%.1fs remaining)",
                    len(batch),
                    remaining,
                )
                rate_limited = True
                break
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
        payload = None
        for attempt in range(ENRICH_RATE_LIMIT_MAX_RETRIES + 1):
            req_timeout = min(timeout, max(3.0, deadline - time.monotonic())) if deadline is not None else timeout
            try:
                if graphql_caller is not None:
                    payload = caller(token, query, variables)
                else:
                    payload = caller(token, query, variables, timeout=req_timeout, attempts=1, max_retry_delay=0.0)
            except Exception as error:
                err_text = _format_github_error(error)
                if _is_rate_limit_error(error):
                    delay = _rate_limit_backoff_delay(error, attempt, deadline=deadline)
                    if delay is not None:
                        logging.warning(
                            "GraphQL profile batch rate-limited (%d logins, attempt %d/%d); waiting %.1fs before retry: %s",
                            len(batch),
                            attempt + 1,
                            ENRICH_RATE_LIMIT_MAX_RETRIES,
                            delay,
                            err_text,
                        )
                        sleep_fn(delay)
                        continue
                    rate_limited = True
                logging.warning("GraphQL profile batch fetch failed (%d logins): %s", len(batch), err_text)
                payload = None
                break

            gql_errors = payload.get("errors") if isinstance(payload, dict) else None
            gql_rl, notable_errors, _not_found = _inspect_graphql_errors(gql_errors)
            if gql_rl:
                rl_err = TransientAPIError("; ".join(notable_errors) or "GraphQL RATE_LIMITED")
                delay = _rate_limit_backoff_delay(rl_err, attempt, deadline=deadline)
                if delay is not None:
                    logging.warning(
                        "GraphQL profile batch hit GraphQL RATE_LIMITED (%d logins, attempt %d/%d); waiting %.1fs: %s",
                        len(batch),
                        attempt + 1,
                        ENRICH_RATE_LIMIT_MAX_RETRIES,
                        delay,
                        "; ".join(notable_errors[:3]),
                    )
                    sleep_fn(delay)
                    payload = None
                    continue
                rate_limited = True
                logging.warning(
                    "GraphQL profile batch hit GraphQL RATE_LIMITED (%d logins): %s",
                    len(batch),
                    "; ".join(notable_errors[:3]),
                )
                break
            if notable_errors:
                logging.warning(
                    "GraphQL profile batch returned errors (%d logins): %s",
                    len(batch),
                    "; ".join(notable_errors[:5]),
                )
            break

        if payload is None:
            if rate_limited:
                break
            if len(batch) > 1:
                mid = len(batch) // 2
                pending_batches = [batch[:mid], batch[mid:]] + pending_batches
            continue

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            if not rate_limited and len(batch) > 1:
                mid = len(batch) // 2
                pending_batches = [batch[:mid], batch[mid:]] + pending_batches
            continue
        batch_profiles: dict[str, GitHubAuthorProfile] = {}
        for idx, requested_login in enumerate(batch):
            node = data.get(f"u{idx}")
            if not isinstance(node, dict):
                continue
            login_val = str(node.get("login") or requested_login).strip()
            name = str(node.get("name") or "").strip()
            bio = str(node.get("bio") or "").strip()
            company = str(node.get("company") or "").strip()
            email = str(node.get("email") or "").strip()
            email_domain = _extract_public_email_domain(email)
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

            prof_obj = GitHubAuthorProfile(
                login=login_val,
                name=name,
                blog=website,
                twitter_username=twitter,
                bio=bio,
                company=company,
                email=email,
                email_domain=email_domain,
                social_accounts=socials,
                readme=readme,
                email_locals=tuple(email_locals),
            )
            req_key = requested_login.casefold()
            profiles[req_key] = prof_obj
            batch_profiles[req_key] = prof_obj
            if login_val and login_val.casefold() != req_key:
                profiles[login_val.casefold()] = prof_obj
        if on_batch is not None and batch_profiles:
            on_batch(batch_profiles)
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
    sleep_fn=time.sleep,
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
        payload = None
        for attempt in range(ENRICH_RATE_LIMIT_MAX_RETRIES + 1):
            req_timeout = min(timeout, max(3.0, deadline - time.monotonic())) if deadline is not None else timeout
            try:
                if graphql_caller is not None:
                    payload = caller(token, query, variables)
                else:
                    payload = caller(token, query, variables, timeout=req_timeout, attempts=1, max_retry_delay=0.0)
                break
            except Exception as error:
                err_text = _format_github_error(error)
                if _is_rate_limit_error(error):
                    delay = _rate_limit_backoff_delay(error, attempt, deadline=deadline)
                    if delay is not None:
                        logging.warning(
                            "GraphQL commit batch rate-limited (%d PRs, attempt %d/%d); waiting %.1fs before retry: %s",
                            len(batch),
                            attempt + 1,
                            ENRICH_RATE_LIMIT_MAX_RETRIES,
                            delay,
                            err_text,
                        )
                        sleep_fn(delay)
                        continue
                    rate_limited = True
                logging.warning("GraphQL commit batch fetch failed (%d PRs): %s", len(batch), err_text)
                payload = None
                break

        if payload is None:
            if rate_limited:
                break
            continue

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
        or (profile.email_domain or "").strip()
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
    sleep_fn=time.sleep,
) -> tuple[dict[str, GitHubAuthorProfile], set[str]]:
    """Batch-enrich profiles via GraphQL (50/query) or ThreadPoolExecutor(8) REST fallback.

    Persists each completed chunk to SQLite immediately and handles rate limits via
    backoff/retry rather than skipping by budget. Returns ``(profiles_by_login, failed_logins)``.
    """
    profiles: dict[str, GitHubAuthorProfile] = {}
    failed_logins: set[str] = set()
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
        return profiles, failed_logins

    use_graphql = graphql_caller is not None or (
        (github_getter is None or github_getter is github_get_json) and bool(token)
    )
    if use_graphql:
        logins = [g["login"] for g in uncached_groups]
        group_by_login = {g["login"].strip().casefold(): g for g in uncached_groups}

        def _persist_batch(batch_profs: dict[str, GitHubAuthorProfile]) -> None:
            for lk, prof in batch_profs.items():
                grp = group_by_login.get(lk)
                uid = grp.get("user_id") if grp else None
                with_uid = GitHubAuthorProfile(
                    login=prof.login,
                    user_id=uid,
                    name=prof.name,
                    blog=prof.blog,
                    twitter_username=prof.twitter_username,
                    bio=prof.bio,
                    company=prof.company,
                    email=prof.email,
                    email_domain=prof.email_domain,
                    social_accounts=prof.social_accounts,
                    readme=prof.readme,
                    commit_names=prof.commit_names,
                    email_locals=prof.email_locals,
                )
                _save_cached_github_profile(store, with_uid, moment)

        gql_profiles, hit_limit = fetch_github_profiles_graphql(
            logins,
            token,
            graphql_caller=graphql_caller,
            deadline=deadline,
            sleep_fn=sleep_fn,
            on_batch=_persist_batch,
        )
        if gql_profiles or hit_limit or graphql_caller is not None:
            pr_by_login: dict[str, list[int]] = {}
            for g in uncached_groups:
                login_key = g["login"].strip().casefold()
                if login_key not in gql_profiles:
                    continue
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
                    sleep_fn=sleep_fn,
                )
            for g in uncached_groups:
                login = g["login"]
                login_key = login.strip().casefold()
                base_prof = gql_profiles.get(login_key)
                if base_prof is None:
                    failed_logins.add(login_key)
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
                    email_domain=base_prof.email_domain,
                    social_accounts=base_prof.social_accounts,
                    readme=base_prof.readme,
                    commit_names=c_names,
                    email_locals=tuple(merged_emails),
                )
                profiles[login_key] = full_prof
                if c_names or c_emails:
                    _save_cached_github_profile(store, full_prof, moment)
            return profiles, failed_logins

    # Concurrent REST fallback (ThreadPoolExecutor(8)) when GraphQL is unavailable or github_getter is injected.
    def _worker(group: dict) -> tuple[str, GitHubAuthorProfile, bool]:
        login = group["login"]
        login_key = login.strip().casefold()
        if deadline is not None and time.monotonic() >= deadline:
            return login_key, GitHubAuthorProfile(login=login, user_id=group.get("user_id")), True
        pull_numbers = [
            int(p["number"])
            for p in group.get("pulls", ())
            if isinstance(p, dict) and isinstance(p.get("number"), int)
        ]
        prof, failed = fetch_github_author_profile(
            login,
            repository,
            pull_numbers,
            token,
            user_id=group.get("user_id"),
            getter=github_getter,
            store=store,
            now=moment,
            deadline=deadline,
            sleep_fn=sleep_fn,
        )
        return login_key, prof, failed

    workers = min(ENRICH_WORKERS, max(1, len(uncached_groups)))
    if workers == 1:
        for group in uncached_groups:
            login_key, prof, was_failed = _worker(group)
            profiles[login_key] = prof
            if was_failed:
                failed_logins.add(login_key)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, group) for group in uncached_groups]
            for fut in as_completed(futures):
                login_key, prof, was_failed = fut.result()
                profiles[login_key] = prof
                if was_failed:
                    failed_logins.add(login_key)
    return profiles, failed_logins


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
    sleep_fn=time.sleep,
) -> tuple[GitHubAuthorProfile, bool]:
    """Fetch GitHub user profile (with 7-day SQLite cache) + PR commit author names/email locals.

    Handles rate limits via backoff/retry rather than skipping by budget. Returns
    ``(profile, enrichment_failed)`` where ``enrichment_failed`` is True if the user
    profile could not be fetched (so callers do not match without enrichment).
    """
    fetch = getter or github_get_json
    moment = now or datetime.now(UTC)
    name = ""
    blog = ""
    twitter = ""
    bio = ""
    company = ""
    email = ""
    email_domain = ""
    social_accounts: tuple[dict[str, str], ...] = ()
    readme = ""
    email_locals: list[str] = []
    commit_names: list[str] = []
    enrichment_failed = False

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
            email_domain = cached_profile.email_domain or _extract_public_email_domain(email)
            social_accounts = cached_profile.social_accounts
            readme = cached_profile.readme
            commit_names.extend(cached_profile.commit_names)
            email_locals.extend(cached_profile.email_locals)

        if not used_cached_user:
            if deadline is not None and _time_left() <= 1.0:
                enrichment_failed = True
            else:
                for attempt in range(ENRICH_RATE_LIMIT_MAX_RETRIES + 1):
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
                            email_domain = _extract_public_email_domain(email)
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
                                    email_domain=email_domain,
                                    social_accounts=social_accounts,
                                    readme=readme,
                                    commit_names=tuple(commit_names),
                                    email_locals=tuple(email_locals),
                                ),
                                moment,
                            )
                        else:
                            enrichment_failed = True
                        break
                    except Exception as error:
                        err_text = _format_github_error(error)
                        if _is_rate_limit_error(error):
                            delay = _rate_limit_backoff_delay(error, attempt, deadline=deadline)
                            if delay is not None:
                                logging.warning(
                                    "GitHub profile fetch rate-limited for %s (attempt %d/%d); waiting %.1fs: %s",
                                    login,
                                    attempt + 1,
                                    ENRICH_RATE_LIMIT_MAX_RETRIES,
                                    delay,
                                    err_text,
                                )
                                sleep_fn(delay)
                                continue
                        enrichment_failed = True
                        logging.warning("Could not fetch GitHub profile for %s: %s", login, err_text)
                        break

    should_fetch_commits = (
        not enrichment_failed
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
                break
            commits = None
            for attempt in range(ENRICH_RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    req_timeout = min(10.0, max(2.0, _time_left()))
                    commits = _call_github_enrichment(
                        fetch,
                        f"{GITHUB_API}/repos/{repository}/pulls/{pr_number}/commits?per_page=30",
                        token,
                        timeout=req_timeout,
                    )
                    break
                except Exception as error:
                    if _is_rate_limit_error(error):
                        delay = _rate_limit_backoff_delay(error, attempt, deadline=deadline)
                        if delay is not None:
                            sleep_fn(delay)
                            continue
                    logging.warning("Could not fetch commits for %s#%s: %s", repository, pr_number, error)
                    commits = None
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
                    email_domain=email_domain,
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
            email_domain=email_domain,
            social_accounts=social_accounts,
            readme=readme,
            commit_names=tuple(commit_names),
            email_locals=tuple(email_locals),
        ),
        enrichment_failed,
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
    if profile.email_domain:
        email_dom_label = profile.email_domain.split(".", 1)[0]
        if len(normalize_strict(email_dom_label)) >= 4:
            sources.append(("email_domain", email_dom_label))
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


_X_URL_HANDLE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com)/([A-Za-z0-9_]{2,30})(?:[/?#\s]|$)",
    re.IGNORECASE,
)
_URL_HOST_IN_TEXT_RE = re.compile(
    r"https?://([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:[:/?#\s]|$)",
    re.IGNORECASE,
)
_GENERIC_X_PATHS = frozenset({"home", "explore", "search", "settings", "i", "intent", "share", "status"})


def _extract_profile_x_handles(profile: GitHubAuthorProfile) -> list[str]:
    handles: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        cleaned = (raw or "").strip().lstrip("@")
        if cleaned and len(cleaned) >= 2 and cleaned.casefold() not in _GENERIC_X_PATHS:
            if cleaned.casefold() not in seen:
                seen.add(cleaned.casefold())
                handles.append(cleaned)

    if profile.twitter_username:
        _add(profile.twitter_username)
    for account in profile.social_accounts:
        if not isinstance(account, dict):
            continue
        provider = str(account.get("provider") or "").strip().upper()
        url = str(account.get("url") or "").strip()
        display = str(account.get("displayName") or account.get("display_name") or "").strip()
        if provider in {"TWITTER", "X"}:
            _add(display)
        for match in _X_URL_HANDLE_RE.finditer(url):
            _add(match.group(1))
    return handles


def _extract_profile_website_hosts(profile: GitHubAuthorProfile) -> list[str]:
    hosts: list[str] = []
    seen: set[str] = set()

    def _add_host(raw_host: str) -> None:
        h = (raw_host or "").strip().lower().removeprefix("www.")
        if not h or "." not in h or h in _GENERIC_WEBSITE_HOSTS or h in _FREE_EMAIL_DOMAINS:
            return
        if h not in seen:
            seen.add(h)
            hosts.append(h)

    if profile.blog:
        _add_host(_extract_website_host(profile.blog))
    if profile.email_domain:
        _add_host(profile.email_domain)
    for account in profile.social_accounts:
        if not isinstance(account, dict):
            continue
        url = str(account.get("url") or "").strip()
        if url:
            _add_host(_extract_website_host(url))
    return hosts


def _corroborating_x_or_website(
    profile: GitHubAuthorProfile,
    member: DiscordMemberInfo,
    message_texts: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Return corroborating X/Twitter handle or website/email-domain signals for a PR-link poster."""
    signals: list[str] = []
    member_norms = {
        norm for val in _member_login_fields(member)
        if (norm := normalize_strict(val)) and len(norm) >= 3
    }
    msg_x_norms: set[str] = set()
    msg_hosts: set[str] = set()
    for text in message_texts:
        if not isinstance(text, str) or not text:
            continue
        for m in _X_URL_HANDLE_RE.finditer(text):
            h = m.group(1)
            if h.casefold() not in _GENERIC_X_PATHS:
                norm = normalize_strict(h)
                if len(norm) >= 3:
                    msg_x_norms.add(norm)
        for m in _URL_HOST_IN_TEXT_RE.finditer(text):
            host = m.group(1).strip().lower().removeprefix("www.")
            if host and host not in _GENERIC_WEBSITE_HOSTS and host not in _FREE_EMAIL_DOMAINS:
                msg_hosts.add(host)

    for handle in _extract_profile_x_handles(profile):
        h_norm = normalize_strict(handle)
        if len(h_norm) >= 3 and (h_norm in member_norms or h_norm in msg_x_norms):
            tag = f"same_x:{handle}"
            if tag not in signals:
                signals.append(tag)

    for host in _extract_profile_website_hosts(profile):
        host_label = normalize_strict(host.split(".", 1)[0])
        if host in msg_hosts or (len(host_label) >= 4 and host_label in member_norms):
            tag = f"same_website:{host}"
            if tag not in signals:
                signals.append(tag)

    return signals


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
    """Validate whether a PR-link poster qualifies for Tier (c) auto-accept.

    Never auto-accepts on link count alone: requires sole eligible poster AND at least
    one corroborating signal:
      - name similarity >= 0.50 to GitHub login/name (username/global_name, not nick)
      - ownership language ('my pr' / 'I opened/made/raised/submitted')
      - same X/Twitter handle or same website/email domain
    Rejects staff and multi-author posters (linked PRs by >= 2 different GitHub authors).
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
    x_or_web = _corroborating_x_or_website(profile, member, ev.message_texts)
    if has_ownership or name_sim >= PR_LINK_POSTER_NAME_SIMILARITY_THRESHOLD or x_or_web:
        reasons = [f"pr_link_poster:{len(set(ev.pr_numbers))}_prs"]
        if has_ownership:
            reasons.append("ownership_text")
        if name_sim >= PR_LINK_POSTER_NAME_SIMILARITY_THRESHOLD:
            reasons.append(f"name_sim:{name_sim:.2f}")
        reasons.extend(x_or_web)
        return True, ", ".join(reasons)
    return False, f"no_corroborating_signal(name_sim={name_sim:.2f})"


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

    # Tier (a): Direct link — Discord user ID or Discord username/handle listed in GitHub
    # profile socialAccounts / bio / blog / README -> match to that guild member (highest confidence).
    id_sources = (
        ("social_discord_id", social_ids),
        ("bio_discord_id", extract_discord_ids_from_text(profile.bio)),
        ("blog_discord_id", extract_discord_ids_from_text(profile.blog)),
        ("readme_discord_id", extract_discord_ids_from_text(profile.readme)),
    )
    for source_name, uids in id_sources:
        hits = [
            index.by_id[uid]
            for uid in uids
            if uid in index.by_id and uid not in rejected_discord_ids
        ]
        if len(hits) == 1:
            return (hits[0], source_name, 0.99), []

    ambiguous_strong: dict[str, tuple[DiscordMemberInfo, list[str]]] = {}
    handle_sources = (
        ("social_discord_handle", social_handles),
        ("bio_discord_handle", extract_discord_handles_from_text(profile.bio)),
        ("blog_discord_handle", extract_discord_handles_from_text(profile.blog)),
        ("readme_discord_handle", extract_discord_handles_from_text(profile.readme)),
    )
    for source_name, handles in handle_sources:
        if not handles:
            continue
        hits_by_id: dict[str, DiscordMemberInfo] = {}
        for handle in handles:
            for member in index.exact_login_hits(handle):
                if member.discord_id not in rejected_discord_ids:
                    hits_by_id[member.discord_id] = member
        hits = list(hits_by_id.values())
        if len(hits) == 1:
            return (hits[0], source_name, 0.98), []
        for member in hits:
            entry = ambiguous_strong.setdefault(member.discord_id, (member, []))
            entry[1].append(f"ambiguous_{source_name}")

    # Tier (b): Exact identity — GitHub login == Discord username or global_name
    # (case/sep-insensitive, NEVER nickname).
    if profile.login:
        hits = [m for m in index.exact_login_hits(profile.login) if m.discord_id not in rejected_discord_ids]
        if len(hits) == 1:
            return (hits[0], "exact_login", 0.96), []
        for member in hits:
            entry = ambiguous_strong.setdefault(member.discord_id, (member, []))
            entry[1].append("ambiguous_exact_login")

    # Tier (c): PR-link poster — member posted a link to that author's PR in Discord AND has
    # at least one corroborating signal (name sim >= 0.50 to GitHub name/login, ownership
    # language like 'my PR', or same X/website). Rejects staff and posters who linked PRs by
    # >= 2 different GitHub authors.
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
            for xw in _corroborating_x_or_website(profile, index.by_id[ev.discord_id], ev.message_texts):
                sigs.append(xw)
            if ev.message_texts:
                snippet = re.sub(r"\s+", " ", ev.message_texts[0]).strip()[:120]
                if snippet:
                    sigs.append(f"pr_link_msg:{snippet}")

    # Tier (d) candidate collection: top-5 guild candidates for list-wise LLM (>= 2 independent signals).
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
    if profile.name:
        candidate_indices.update(index.exact_field_indices(profile.name))
        candidate_indices.update(index.initial_pattern_indices(profile.name))
    if profile.twitter_username:
        candidate_indices.update(index.exact_field_indices(profile.twitter_username))
    for commit_name in profile.commit_names:
        candidate_indices.update(index.exact_field_indices(commit_name))
        candidate_indices.update(index.initial_pattern_indices(commit_name))
    for email_local in profile.email_locals:
        candidate_indices.update(index.exact_field_indices(email_local))
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
        if profile.twitter_username and _exact_field_match(profile.twitter_username, raw_l, strict_s, base_s):
            extra_sigs.append(f"twitter_match:{profile.twitter_username}")
            base_floor = max(base_floor, 0.88)
        if profile.name and _exact_field_match(profile.name, raw_l, strict_s, base_s):
            extra_sigs.append(f"exact_name:{profile.name}")
            base_floor = max(base_floor, 0.86)
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
            "email_domain": profile.email_domain[:120],
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
                "You MUST require at least 2 independent corroborating signals AND confidence >= 0.85 to select a candidate. "
                "Never accept on a single signal, common first name, or weak coincidence — return discord_id: null instead. "
                "Return ONLY one JSON object with keys: "
                "discord_id (string matching one candidate's discord_id, or null for none of these), "
                "confidence (float 0.0 to 1.0), "
                "reason (short plain-English explanation), "
                "evidence (array of >= 2 distinct concrete corroborating signals used, e.g. ['name matches global_name', 'twitter matches username', 'posted PR link']). "
                "No markdown fences."
            ),
        },
        {"role": "user", "content": json.dumps(payload)},
    ]

    if llm_complete is not None:
        raw = llm_complete(model, messages)
    else:
        from .review_llm import complete_litellm
        from .tracing import build_trace_metadata
        metadata = build_trace_metadata(
            "role_matcher",
            session_id=f"role_matcher:{profile.login}",
            extra={"github_login": profile.login},
        )
        raw = complete_litellm(model, messages, timeout=timeout, metadata=metadata)

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
    """Require >= 2 independent corroborating signals AND confidence >= 0.85; no single-signal accepts."""
    if not result.discord_id:
        return False
    return (
        result.confidence >= LLM_CONFIDENCE_THRESHOLD
        and len(result.evidence) >= LLM_MIN_SIGNALS
    )


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
    """Run list-wise _select_with_llm per author in a thread pool (never falls back to guessing)."""
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
        skip_reasons: dict[str, str] = {}
        skip_meta: dict[str, tuple[float, str]] = {}
        for login_key, pairs in by_login.items():
            skip_reasons[login_key] = "llm_unavailable"
            skip_meta[login_key] = (pairs[0].heuristic_score if pairs else 0.0, "no LLM credentials configured")
        return {}, "unavailable", skip_reasons, skip_meta, {}

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

    for login_key, res, error in results_list:
        pairs = by_login[login_key]
        member_map = {pair.member.discord_id: pair.member for pair in pairs}
        top_heur = pairs[0].heuristic_score if pairs else 0.0
        if error is not None or res is None:
            logging.warning("List-wise LLM selection failed for %s: %s", login_key, error)
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


def _snowflake_int(raw: object) -> int:
    text = str(raw or "").strip()
    return int(text) if text.isdigit() else 0


def _flatten_search_messages(payload: object, discord_id: str) -> list[dict]:
    """Extract messages authored by ``discord_id`` from a Discord search response (or flat list)."""
    raw_messages: list[dict] = []
    if isinstance(payload, dict):
        groups = payload.get("messages")
        if isinstance(groups, list):
            for entry in groups:
                if isinstance(entry, list):
                    for item in entry:
                        if isinstance(item, dict):
                            raw_messages.append(item)
                elif isinstance(entry, dict):
                    raw_messages.append(entry)
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, list):
                for item in entry:
                    if isinstance(item, dict):
                        raw_messages.append(item)
            elif isinstance(entry, dict):
                raw_messages.append(entry)

    seen_ids: set[str] = set()
    authored: list[dict] = []
    for msg in raw_messages:
        author = msg.get("author")
        if isinstance(author, dict):
            author_uid = str(author.get("id") or "").strip()
            if author_uid and author_uid != discord_id:
                continue
        msg_id = str(msg.get("id") or "").strip()
        if msg_id:
            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)
        authored.append(msg)
    return authored


def _discord_message_link(guild_id: str | None, channel_id: object, message_id: object) -> str:
    gid = str(guild_id or "").strip()
    cid = str(channel_id or "").strip()
    mid = str(message_id or "").strip()
    if is_discord_id(gid) and is_discord_id(cid) and is_discord_id(mid):
        return f"https://discord.com/channels/{gid}/{cid}/{mid}"
    return ""


def _distinctive_title_tokens(title: str) -> set[str]:
    tokens = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9_.-]{4,}", title or "")
        if tok.lower() not in _CODE_TITLE_STOPWORDS and not tok.isdigit()
    }
    return tokens


def _collect_author_pr_context(
    groups: list[dict],
    store,
    repository: str,
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """Collect changed code paths and PR titles per lowercase GitHub author login."""
    files_by_login: dict[str, set[str]] = {}
    titles_by_login: dict[str, list[str]] = {}

    def _add_pull_meta(login_key: str, title: object, files: object) -> None:
        if isinstance(title, str) and title.strip():
            titles = titles_by_login.setdefault(login_key, [])
            cleaned_title = title.strip()
            if cleaned_title not in titles:
                titles.append(cleaned_title)
        if isinstance(files, (list, tuple, set)):
            dest = files_by_login.setdefault(login_key, set())
            for raw_path in files:
                path = str(raw_path or "").strip().lstrip("./")
                if not path or "/" not in path:
                    continue
                basename = path.rsplit("/", 1)[-1].lower()
                if basename in _GENERIC_CODE_BASENAMES:
                    continue
                dest.add(path.lower())

    for group in groups:
        login_key = str(group.get("login") or "").strip().lower()
        if not login_key:
            continue
        for p in group.get("pulls", ()):
            if isinstance(p, dict):
                _add_pull_meta(login_key, p.get("title"), p.get("files"))
        if store is not None and repository:
            uid = group.get("user_id")
            try:
                rows: list[dict] = []
                if isinstance(uid, int) and uid > 0 and hasattr(store, "pulls_for_author"):
                    rows = store.pulls_for_author(repository, uid)
                elif hasattr(store, "pulls_for_author_login"):
                    rows = store.pulls_for_author_login(repository, group["login"])
                for row in rows:
                    _add_pull_meta(login_key, row.get("title"), row.get("files"))
            except Exception:
                pass
    return files_by_login, titles_by_login


def analyze_member_messages(
    discord_id: str,
    messages: list[dict],
    repository: str,
    *,
    guild_id: str | None = None,
    pr_number_to_author: dict[int, str] | None = None,
    author_pr_files: dict[str, set[str]] | None = None,
    author_pr_titles: dict[str, list[str]] | None = None,
    multi_author_ids: frozenset[str] | set[str] = frozenset(),
) -> dict:
    """Analyze a Discord member's guild messages for GitHub identity evidence.

    Extracts:
    - Explicit self-claimed GitHub handles / profile URLs / fork URLs of the tracked repo
    - Tracked-repo PR/issue links paired with ownership statements ('my PR', 'I opened', ...)
    - Code paths and PR topics discussed that align with a merged-PR author's changes
    """
    pr_map = pr_number_to_author or {}
    files_map = author_pr_files or {}
    titles_map = author_pr_titles or {}
    repo_org, repo_name = ("", "")
    if "/" in (repository or ""):
        raw_org, raw_name = repository.split("/", 1)
        repo_org = raw_org.strip().lower()
        repo_name = raw_name.strip().lower()

    # Invert non-generic changed file paths -> authors who touched them uniquely.
    file_to_authors: dict[str, set[str]] = {}
    for login_key, paths in files_map.items():
        for path in paths:
            file_to_authors.setdefault(path.lower(), set()).add(login_key)
    title_tokens_by_author: dict[str, set[str]] = {}
    for login_key, titles in titles_map.items():
        toks: set[str] = set()
        for title in titles:
            toks.update(_distinctive_title_tokens(title))
        if toks:
            title_tokens_by_author[login_key] = toks

    authored = _flatten_search_messages(messages, discord_id)
    max_msg_id = ""
    claimed_logins: dict[str, list[tuple[str, str]]] = {}
    owned_pr_authors: dict[str, list[tuple[str, str]]] = {}
    linked_pr_authors: dict[str, list[tuple[str, str]]] = {}
    code_hits_by_author: dict[str, list[tuple[str, str]]] = {}
    tracked_pr_links_found: list[tuple[int, str, str, str]] = []
    generic_snippets: list[tuple[str, str]] = []

    def _snippet(text: str) -> str:
        return " ".join(text.split())[:180]

    for msg in authored:
        mid = str(msg.get("id") or "").strip()
        cid = str(msg.get("channel_id") or "").strip()
        if _snowflake_int(mid) > _snowflake_int(max_msg_id):
            max_msg_id = mid
        link = _discord_message_link(guild_id, cid, mid)

        parts: list[str] = []
        content = str(msg.get("content") or "").strip()
        if content:
            parts.append(content)
        for embed in msg.get("embeds") or ():
            if isinstance(embed, dict):
                for key in ("url", "title", "description"):
                    val = str(embed.get(key) or "").strip()
                    if val:
                        parts.append(val)
        text = "\n".join(parts).strip()
        if not text:
            continue
        snippet = _snippet(content or text)
        has_ownership = bool(_MSG_OWNERSHIP_RE.search(text))

        # 1. Explicit self-claimed GitHub handle ("my github is @foo", "github: foo", "I'm foo on github").
        for pattern in (_GITHUB_SELF_HANDLE_RE, _GITHUB_IM_ON_GH_RE):
            for match in pattern.finditer(text):
                handle = match.group(1).strip().lower()
                if handle and handle not in _GITHUB_RESERVED_PATHS and handle != repo_org:
                    claimed_logins.setdefault(handle, []).append((f"github_handle_claim:{handle}: {snippet}", link))

        # 2. Standalone GitHub profile URLs (github.com/<login>) or tracked-repo fork URLs (github.com/<login>/<repo>).
        for match in _GITHUB_PROFILE_OR_REPO_URL_RE.finditer(text):
            owner = match.group(1).strip().lower()
            repo_part = (match.group(2) or "").strip().lower()
            if not owner or owner in _GITHUB_RESERVED_PATHS or owner == repo_org:
                continue
            if not repo_part:
                claimed_logins.setdefault(owner, []).append((f"github_profile_url:github.com/{owner}: {snippet}", link))
            elif repo_name and repo_part == repo_name:
                claimed_logins.setdefault(owner, []).append((f"github_fork_url:github.com/{owner}/{repo_name}: {snippet}", link))

        # 3. PR and issue links in the tracked repository (+ #NNN refs when accompanied by ownership wording).
        msg_pr_nums: set[int] = set()
        for match in _GITHUB_PR_OR_ISSUE_RE.finditer(text):
            m_org = match.group(1).strip().lower()
            m_repo = match.group(2).strip().lower()
            kind = match.group(3).strip().lower()
            num = int(match.group(4))
            if repo_org and repo_name and (m_org != repo_org or m_repo != repo_name):
                continue
            if kind == "pull":
                msg_pr_nums.add(num)
                if mid and cid:
                    tracked_pr_links_found.append((num, cid, mid, snippet))
            elif kind == "issues":
                generic_snippets.append((f"issue_link:#{num}: {snippet}", link))

        if has_ownership:
            for match in _HASH_PR_NUM_RE.finditer(text):
                num = int(match.group(1))
                if num in pr_map:
                    msg_pr_nums.add(num)

        for num in sorted(msg_pr_nums):
            author_login = pr_map.get(num)
            if not author_login:
                generic_snippets.append((f"pr_link:#{num}: {snippet}", link))
                continue
            linked_pr_authors.setdefault(author_login, []).append((f"pr_link:#{num}: {snippet}", link))
            if has_ownership:
                owned_pr_authors.setdefault(author_login, []).append(
                    (f"pr_ownership:#{num}: {snippet}", link)
                )

        # 4. Code paths and distinctive PR topic discussion.
        mentioned_paths = {
            m.group(1).strip().lstrip("./").lower()
            for m in _CODE_PATH_RE.finditer(text)
            if m.group(1).rsplit("/", 1)[-1].lower() not in _GENERIC_CODE_BASENAMES
        }
        for path in sorted(mentioned_paths):
            owners = file_to_authors.get(path, set())
            if len(owners) == 1:
                sole_author = next(iter(owners))
                code_hits_by_author.setdefault(sole_author, []).append(
                    (f"code_path:{path}: {snippet}", link)
                )

        if title_tokens_by_author and (mentioned_paths or has_ownership or "pr" in text.lower() or "fix" in text.lower()):
            msg_tokens = _distinctive_title_tokens(text)
            if len(msg_tokens) >= 2:
                overlap_hits: list[tuple[str, set[str]]] = []
                for author_login, author_toks in title_tokens_by_author.items():
                    shared = msg_tokens & author_toks
                    if len(shared) >= 2:
                        overlap_hits.append((author_login, shared))
                if len(overlap_hits) == 1:
                    sole_author, shared = overlap_hits[0]
                    code_hits_by_author.setdefault(sole_author, []).append(
                        (f"code_topic:{','.join(sorted(shared)[:4])}: {snippet}", link)
                    )

    is_multi_author = (discord_id in multi_author_ids) or (len(linked_pr_authors) >= 2)

    def _pack_items(items: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
        ev_out: list[str] = []
        links_out: list[str] = []
        for ev_text, ev_link in items:
            if ev_text and ev_text not in ev_out:
                ev_out.append(ev_text)
            if ev_link and ev_link not in links_out:
                links_out.append(ev_link)
        return ev_out[:12], links_out[:12]

    inferred_login: str | None = None
    confidence = 0.0
    chosen_items: list[tuple[str, str]] = []

    if len(claimed_logins) == 1:
        sole_login = next(iter(claimed_logins))
        inferred_login = sole_login
        confidence = MESSAGE_EVIDENCE_CLAIM_CONFIDENCE
        chosen_items.extend(claimed_logins[sole_login])
        chosen_items.extend(owned_pr_authors.get(sole_login, ()))
        chosen_items.extend(code_hits_by_author.get(sole_login, ())[:3])
    elif len(claimed_logins) >= 2:
        inferred_login = None
        confidence = 0.0
        chosen_items.append((f"conflicting_github_claims:{','.join(sorted(claimed_logins)[:4])}", ""))
    elif not is_multi_author and len(owned_pr_authors) == 1:
        sole_login = next(iter(owned_pr_authors))
        inferred_login = sole_login
        confidence = MESSAGE_EVIDENCE_OWNERSHIP_CONFIDENCE
        chosen_items.extend(owned_pr_authors[sole_login])
        chosen_items.extend(code_hits_by_author.get(sole_login, ())[:3])
    elif not is_multi_author:
        corroborating_authors = set(linked_pr_authors.keys()) | set(code_hits_by_author.keys())
        if len(corroborating_authors) == 1:
            sole_login = next(iter(corroborating_authors))
            inferred_login = sole_login
            has_pr_link = sole_login in linked_pr_authors
            has_code = sole_login in code_hits_by_author
            confidence = 0.70 if (has_pr_link and has_code) else 0.55
            chosen_items.extend(linked_pr_authors.get(sole_login, ()))
            chosen_items.extend(code_hits_by_author.get(sole_login, ()))
        elif generic_snippets:
            chosen_items.extend(generic_snippets[:5])
    else:
        chosen_items.append((f"multi_author_poster:{len(linked_pr_authors)}_authors", ""))

    evidence, message_links = _pack_items(chosen_items)
    return {
        "discord_id": discord_id,
        "inferred_github_login": inferred_login,
        "evidence": evidence,
        "message_links": message_links,
        "confidence": confidence,
        "last_message_id": max_msg_id,
        "messages_scanned": len(authored),
        "is_multi_author": is_multi_author,
        "tracked_pr_links": tracked_pr_links_found,
    }


def scan_member_message_evidence(
    discord_id: str,
    guild_id: str | None,
    discord_token: str | None,
    repository: str,
    *,
    store=None,
    now: datetime | None = None,
    pr_number_to_author: dict[int, str] | None = None,
    author_pr_files: dict[str, set[str]] | None = None,
    author_pr_titles: dict[str, list[str]] | None = None,
    multi_author_ids: frozenset[str] | set[str] = frozenset(),
    pr_link_messages_for_member: list[dict] | None = None,
    message_searcher=None,
    allow_live_search: bool = False,
    max_pages: int = MEMBER_MESSAGE_SCAN_MAX_PAGES,
    sleep_fn=time.sleep,
) -> dict | None:
    """Return a durable ``member_identity_notes`` record for ``discord_id``.

    Reuses the persisted SQLite note without calling Discord when no newer messages
    have arrived since ``last_message_id``. Scans via ``message_searcher`` (or
    ``search_guild_messages(..., author_id=discord_id)`` when ``allow_live_search``
    is True) on first inspection or when ``needs_refresh`` is set.
    """
    uid = str(discord_id or "").strip()
    if not is_discord_id(uid):
        return None
    moment = now or datetime.now(tz=UTC)
    moment_iso = iso(moment)

    existing = None
    if store is not None and hasattr(store, "get_member_identity_note"):
        try:
            existing = store.get_member_identity_note(uid)
        except Exception as error:
            logging.warning("Could not read member_identity_notes for %s: %s", uid, error)

    newest_pr_link_mid = ""
    for row in pr_link_messages_for_member or ():
        mid = str(row.get("message_id") or "").strip()
        if _snowflake_int(mid) > _snowflake_int(newest_pr_link_mid):
            newest_pr_link_mid = mid

    needs_refresh = existing is None or bool(existing.get("needs_refresh"))
    if existing is not None and _snowflake_int(newest_pr_link_mid) > _snowflake_int(existing.get("last_message_id")):
        needs_refresh = True

    if existing is not None and not needs_refresh:
        return existing

    can_search = message_searcher is not None or (
        allow_live_search and bool(guild_id) and is_discord_id(str(guild_id)) and bool(discord_token)
    )
    if not can_search:
        return existing

    search_fn = message_searcher
    fetched_messages: list[dict] = []
    scan_failed = False
    for page in range(max(1, int(max_pages))):
        offset = page * 25
        try:
            if search_fn is not None:
                payload = search_fn(str(guild_id or ""), uid, offset)
            else:
                payload = search_guild_messages(
                    str(guild_id),
                    "",
                    str(discord_token),
                    author_id=uid,
                    offset=offset,
                    sleep_fn=sleep_fn,
                )
        except Exception as error:
            logging.warning("Message-evidence scan failed for discord_id=%s (offset=%d): %s", uid, offset, error)
            scan_failed = True
            break
        page_msgs = _flatten_search_messages(payload, uid)
        if not page_msgs:
            break
        fetched_messages.extend(page_msgs)
        total_results = payload.get("total_results") if isinstance(payload, dict) else None
        if len(page_msgs) < 25 or (isinstance(total_results, int) and offset + 25 >= total_results):
            break

    if scan_failed and not fetched_messages:
        return existing

    # Fold in already-captured pr_link_posters messages for this member so we never lose them.
    seen_ids = {str(m.get("id") or "") for m in fetched_messages if m.get("id")}
    for row in pr_link_messages_for_member or ():
        mid = str(row.get("message_id") or "").strip()
        if mid and mid in seen_ids:
            continue
        text = str(row.get("message_text") or "").strip()
        pr_num = row.get("pr_number")
        if repository and isinstance(pr_num, int) and pr_num > 0 and f"/pull/{pr_num}" not in text:
            text = f"{text} https://github.com/{repository}/pull/{pr_num}".strip()
        fetched_messages.append({
            "id": mid,
            "channel_id": str(row.get("channel_id") or ""),
            "author": {"id": uid},
            "content": text,
        })

    analysis = analyze_member_messages(
        uid,
        fetched_messages,
        repository,
        guild_id=guild_id,
        pr_number_to_author=pr_number_to_author,
        author_pr_files=author_pr_files,
        author_pr_titles=author_pr_titles,
        multi_author_ids=multi_author_ids,
    )

    last_msg_id = analysis["last_message_id"]
    if existing is not None:
        if _snowflake_int(existing.get("last_message_id")) > _snowflake_int(last_msg_id):
            last_msg_id = str(existing.get("last_message_id") or "")
        # If new messages were seen via gateway and search returned messages, mark up to seen if scanned >= seen.
        if not last_msg_id and existing.get("last_seen_message_id"):
            last_msg_id = str(existing.get("last_seen_message_id") or "")

    # Record any newly discovered tracked-repo PR links in pr_link_posters as well.
    if store is not None and repository and hasattr(store, "record_pr_link_poster"):
        for pr_num, cid, mid, snippet in analysis.get("tracked_pr_links", ()):
            try:
                store.record_pr_link_poster(
                    repository,
                    pr_num,
                    uid,
                    cid,
                    mid,
                    snippet,
                    posted_at=moment_iso,
                )
            except Exception:
                pass

    if store is not None and hasattr(store, "save_member_identity_note"):
        try:
            store.save_member_identity_note(
                uid,
                inferred_github_login=analysis["inferred_github_login"],
                evidence=analysis["evidence"],
                message_links=analysis["message_links"],
                confidence=analysis["confidence"],
                last_message_id=last_msg_id,
                messages_scanned=analysis["messages_scanned"],
                updated_at=moment_iso,
            )
            saved = store.get_member_identity_note(uid)
            if saved is not None:
                return saved
        except Exception as error:
            logging.warning("Could not persist member_identity_notes for %s: %s", uid, error)

    return {
        "discord_id": uid,
        "inferred_github_login": analysis["inferred_github_login"],
        "evidence": analysis["evidence"],
        "message_links": analysis["message_links"],
        "confidence": analysis["confidence"],
        "last_message_id": last_msg_id,
        "last_seen_message_id": last_msg_id,
        "messages_scanned": analysis["messages_scanned"],
        "updated_at": moment_iso,
        "needs_refresh": False,
    }


def _member_matches_any_github_login(member: DiscordMemberInfo, author_logins: set[str]) -> bool:
    """Return True when the member's Discord username or global_name directly matches a GitHub login."""
    strict_logins = {normalize_strict(login) for login in author_logins if normalize_strict(login)}
    for raw in (member.username, member.global_name):
        cleaned = (raw or "").strip()
        if not cleaned:
            continue
        if cleaned.lower() in author_logins:
            return True
        strict = normalize_strict(cleaned)
        if strict and strict in strict_logins:
            return True
    return False


def _compute_attempt_signature(
    profile: GitHubAuthorProfile,
    candidates: list[CandidatePair],
    pr_link_rows: list[dict],
    identity_notes: list[dict] | None = None,
) -> str:
    """Stable SHA-256 signature over profile, candidate shortlist, PR-link rows, and member identity notes."""
    payload = {
        "profile": {
            "login": profile.login.casefold(),
            "name": profile.name,
            "bio": profile.bio,
            "company": profile.company,
            "email": profile.email,
            "email_domain": profile.email_domain,
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
        "identity_notes": sorted(
            [
                str(n.get("discord_id") or ""),
                str(n.get("inferred_github_login") or ""),
                round(float(n.get("confidence") or 0.0), 2),
                str(n.get("last_message_id") or ""),
                str(n.get("last_seen_message_id") or ""),
                str(n.get("updated_at") or ""),
            ]
            for n in (identity_notes or ())
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
    message_searcher=None,
    allow_live_message_search: bool | None = None,
    skip_github_enrichment: bool = False,
    deadline: float | None = None,
    recent_poster_ids: set[str] | frozenset[str] | None = None,
    org_members: frozenset[str] = frozenset(),
    sleep_fn=time.sleep,
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
    failed_enrichment_logins: set[str] = set()
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
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining > 5.0:
                llm_reserve = min(20.0, remaining * 0.35)
                enrich_deadline = time.monotonic() + max(5.0, remaining - llm_reserve)
            else:
                enrich_deadline = time.monotonic() + max(5.0, remaining)
        else:
            enrich_deadline = None
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

        # Never skip GitHub profile lookups: fully enrich EVERY pending author via GraphQL
        # batch (or REST fallback) with backoff/retry + per-batch SQLite persistence BEFORE
        # running any matching tier.
        profiles_by_login, failed_enrichment_logins = _enrich_profiles_batch(
            pending,
            repository,
            token,
            store=store,
            moment=moment,
            github_getter=github_getter,
            graphql_caller=graphql_caller,
            deadline=enrich_deadline,
            skip_github_enrichment=skip_github_enrichment,
            sleep_fn=sleep_fn,
        )
        enrichment_deferred = len(failed_enrichment_logins)

        # If enrichment ultimately failed for an author, do NOT match them in ANY tier this
        # run (retry next run; never negatively cache and never fall back to guessing).
        eligible_pending: list[dict] = []
        for group in pending:
            login_key = group["login"].strip().casefold()
            if login_key in failed_enrichment_logins:
                skip_reason_by_login[login_key] = "enrichment_failed"
                skip_meta_by_login[login_key] = (0.0, "github profile enrichment failed; will retry next run")
                logging.warning(
                    "Skipping matching for %s: GitHub profile enrichment failed after retries (will retry next run)",
                    group["login"],
                )
                continue
            eligible_pending.append(group)

        # Prepare message-evidence context & refresh any stale durable member_identity_notes
        # (only when new messages have arrived since the member was last scanned).
        author_pr_files, author_pr_titles = _collect_author_pr_context(groups, store, repository)
        pr_link_rows_by_member: dict[str, list[dict]] = {}
        for row in all_repo_pr_links:
            uid = str(row.get("discord_id") or "").strip()
            if is_discord_id(uid):
                pr_link_rows_by_member.setdefault(uid, []).append(row)

        live_message_search = (
            (members is None if allow_live_message_search is None else bool(allow_live_message_search))
            and bool(guild_id)
            and is_discord_id(str(guild_id))
            and bool(discord_token)
        )
        can_scan_messages = message_searcher is not None or live_message_search
        scans_used = 0
        identity_notes_by_member: dict[str, dict] = {}
        if store is not None and hasattr(store, "all_member_identity_notes"):
            try:
                for note in store.all_member_identity_notes():
                    uid = str(note.get("discord_id") or "")
                    if not is_discord_id(uid):
                        continue
                    pr_rows_m = pr_link_rows_by_member.get(uid, ())
                    newest_pr_mid = max(
                        (_snowflake_int(r.get("message_id")) for r in pr_rows_m),
                        default=0,
                    )
                    stale = bool(note.get("needs_refresh")) or (
                        newest_pr_mid > _snowflake_int(note.get("last_message_id"))
                    )
                    if (
                        stale
                        and can_scan_messages
                        and uid in member_by_id
                        and uid not in excluded_staff_ids
                        and scans_used < MAX_MEMBER_MESSAGE_SCANS_PER_RUN
                    ):
                        refreshed = scan_member_message_evidence(
                            uid,
                            str(guild_id) if guild_id else None,
                            discord_token,
                            repository,
                            store=store,
                            now=moment,
                            pr_number_to_author=pr_number_to_author,
                            author_pr_files=author_pr_files,
                            author_pr_titles=author_pr_titles,
                            multi_author_ids=multi_author_ids,
                            pr_link_messages_for_member=list(pr_rows_m),
                            message_searcher=message_searcher,
                            allow_live_search=live_message_search,
                            sleep_fn=sleep_fn,
                        )
                        scans_used += 1
                        if refreshed is not None:
                            note = refreshed
                    identity_notes_by_member[uid] = note
            except Exception as error:
                logging.warning("Could not load member_identity_notes: %s", error)

        identity_notes_by_login: dict[str, list[dict]] = {}
        for note in identity_notes_by_member.values():
            inferred = (note.get("inferred_github_login") or "").strip().casefold()
            if inferred:
                identity_notes_by_login.setdefault(inferred, []).append(note)

        # Precompute full-pool scoring + negative-cache check for every enriched author so
        # identical no_match attempts skip BOTH Pass 1 and Pass 2 LLM calls for 7 days.
        cached_negative_keys: set[str] = set()
        full_scores_cache: dict[str, tuple[tuple[DiscordMemberInfo, str, float] | None, list[CandidatePair]]] = {}
        for group in eligible_pending:
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
            sig = _compute_attempt_signature(
                profile,
                full_cands,
                pr_link_rows_by_login.get(login_key, []),
                identity_notes_by_login.get(login_key, []),
            )
            attempt_signatures[login_key] = sig
            if (
                full_strong is None
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
            profile = profiles_by_login[login_key]
            if method == "pr_link_poster":
                evs = [e for e in pr_link_evidence_by_login.get(login_key, ()) if e.discord_id == member.discord_id]
                if evs:
                    ev = evs[0]
                    parts = [f"pr_link_poster:{len(set(ev.pr_numbers))}_prs({','.join(f'#{n}' for n in ev.pr_numbers[:5])})"]
                    if _has_pr_ownership_text(ev.message_texts):
                        parts.append("ownership_text")
                    sim = _author_member_name_similarity(profile, member)
                    if sim >= PR_LINK_POSTER_NAME_SIMILARITY_THRESHOLD:
                        parts.append(f"name_sim:{sim:.2f}")
                    parts.extend(_corroborating_x_or_website(profile, member, ev.message_texts))
                    return tuple(parts)
            if method == "exact_login":
                return (f"exact_identity:{profile.login}->{member.username or member.global_name}",)
            if method.endswith(("_discord_id", "_discord_handle")):
                return (f"direct_link:{method}:{member.username or member.global_name or member.discord_id}",)
            return (f"{method}:{member.username or member.global_name or member.discord_id}",)

        def _llm_allowed(login_key: str) -> bool:
            if skip_github_enrichment:
                return True
            return _profile_has_enrichment_data(profiles_by_login.get(login_key))

        # Tiers (a)-(c): deterministic strong matches first (priority pool, then full pool).
        if use_priority_pool and recent_index is not None:
            for group in eligible_pending:
                login_key = group["login"].strip().casefold()
                if login_key in resolved or login_key in cached_negative_keys:
                    continue
                profile = profiles_by_login[login_key]
                strong, _candidates = score_author_against_members(
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

        for group in eligible_pending:
            login_key = group["login"].strip().casefold()
            if login_key in resolved or login_key in cached_negative_keys:
                continue
            strong, _candidates = full_scores_cache[login_key]
            if strong is not None:
                member, method, confidence = strong
                resolved[login_key] = (group, member.discord_id, method, confidence)
                evidence_by_login[login_key] = _strong_evidence(login_key, member, method)

        # Message-evidence step: for Discord members whose username/global_name does not
        # match a GitHub login and who are not yet linked to a GitHub profile, reuse durable
        # member_identity_notes or scan their guild messages (PR/issue links, GitHub URLs/handles,
        # ownership statements, code discussion) against real merged-PR authors.
        unmatched_groups = [
            g for g in eligible_pending
            if g["login"].strip().casefold() not in resolved
            and g["login"].strip().casefold() not in cached_negative_keys
        ]
        if unmatched_groups:
            unmatched_login_set = {g["login"].strip().casefold() for g in unmatched_groups}
            all_author_logins = {g["login"].strip().lower() for g in groups}

            already_mapped_discord_ids: set[str] = {uid for (_g, uid, _m, _c) in resolved.values()}
            for m_uid in manual_map.values():
                if is_discord_id(str(m_uid)):
                    already_mapped_discord_ids.add(str(m_uid))
            if store is not None:
                if hasattr(store, "all_links"):
                    try:
                        for row in store.all_links():
                            if is_discord_id(str(row.get("discord_id") or "")):
                                already_mapped_discord_ids.add(str(row["discord_id"]))
                    except Exception:
                        pass
                if hasattr(store, "all_confirmed_matches"):
                    try:
                        for row in store.all_confirmed_matches():
                            if is_discord_id(str(row.get("discord_id") or "")):
                                already_mapped_discord_ids.add(str(row["discord_id"]))
                    except Exception:
                        pass

            # Build prioritized list of unlinked, non-staff members to scan if uncached.
            ordered_member_ids: list[str] = []
            seen_member_ids: set[str] = set()

            def _enqueue_member(uid: str) -> None:
                if (
                    not is_discord_id(uid)
                    or uid in seen_member_ids
                    or uid not in member_by_id
                    or uid in excluded_staff_ids
                    or uid in already_mapped_discord_ids
                ):
                    return
                member = member_by_id[uid]
                if _member_matches_any_github_login(member, all_author_logins):
                    return
                seen_member_ids.add(uid)
                ordered_member_ids.append(uid)

            # 1. Members who already have a persisted note pointing to an unmatched author.
            for login_key in unmatched_login_set:
                for note in identity_notes_by_login.get(login_key, ()):
                    _enqueue_member(str(note.get("discord_id") or ""))
            # 2. Candidate shortlist + PR-link posters for unmatched authors.
            for login_key in unmatched_login_set:
                for cand in full_scores_cache[login_key][1]:
                    _enqueue_member(cand.member.discord_id)
                for ev in pr_link_evidence_by_login.get(login_key, ()):
                    _enqueue_member(ev.discord_id)
            # 3. Priority-pool recent members, then remaining unmatched guild members.
            for member in recent_members:
                _enqueue_member(member.discord_id)
            for member in guild_members:
                _enqueue_member(member.discord_id)

            for uid in ordered_member_ids:
                existing_note = identity_notes_by_member.get(uid)
                pr_rows_m = pr_link_rows_by_member.get(uid, ())
                newest_pr_mid = max((_snowflake_int(r.get("message_id")) for r in pr_rows_m), default=0)
                needs_scan = (
                    existing_note is None
                    or bool(existing_note.get("needs_refresh"))
                    or newest_pr_mid > _snowflake_int(existing_note.get("last_message_id"))
                )
                if not needs_scan:
                    continue
                if not can_scan_messages or scans_used >= MAX_MEMBER_MESSAGE_SCANS_PER_RUN:
                    continue
                scanned = scan_member_message_evidence(
                    uid,
                    str(guild_id) if guild_id else None,
                    discord_token,
                    repository,
                    store=store,
                    now=moment,
                    pr_number_to_author=pr_number_to_author,
                    author_pr_files=author_pr_files,
                    author_pr_titles=author_pr_titles,
                    multi_author_ids=multi_author_ids,
                    pr_link_messages_for_member=list(pr_rows_m),
                    message_searcher=message_searcher,
                    allow_live_search=live_message_search,
                    sleep_fn=sleep_fn,
                )
                scans_used += 1
                if scanned is not None:
                    identity_notes_by_member[uid] = scanned

            # Rebuild login -> notes mapping after any fresh scans.
            identity_notes_by_login = {}
            for uid, note in identity_notes_by_member.items():
                if uid not in member_by_id or uid in excluded_staff_ids:
                    continue
                inferred = (note.get("inferred_github_login") or "").strip().casefold()
                if inferred:
                    identity_notes_by_login.setdefault(inferred, []).append(note)

            # Evaluate strong deterministic message_evidence matches and corroborating LLM signals.
            msg_conflict_logins: set[str] = set()
            for group in unmatched_groups:
                login_key = group["login"].strip().casefold()
                notes_for_login = [
                    n for n in identity_notes_by_login.get(login_key, ())
                    if (login_key, str(n["discord_id"])) not in rejected_pairs
                    and str(n["discord_id"]) not in multi_author_ids
                ]
                if not notes_for_login:
                    continue

                # Refresh attempt signature so future runs reflect newly learned message notes.
                profile = profiles_by_login[login_key]
                attempt_signatures[login_key] = _compute_attempt_signature(
                    profile,
                    full_scores_cache[login_key][1],
                    pr_link_rows_by_login.get(login_key, []),
                    notes_for_login,
                )

                strong_notes = [
                    n for n in notes_for_login
                    if float(n.get("confidence") or 0.0) >= MESSAGE_EVIDENCE_STRONG_THRESHOLD
                ]
                if len(strong_notes) == 1:
                    chosen = strong_notes[0]
                    uid = str(chosen["discord_id"])
                    conf = float(chosen["confidence"])
                    resolved[login_key] = (group, uid, "message_evidence", conf)
                    ev_items: list[str] = []
                    for item in chosen.get("evidence") or ():
                        if item not in ev_items:
                            ev_items.append(str(item))
                    for link in chosen.get("message_links") or ():
                        tagged = f"msg_link:{link}"
                        if tagged not in ev_items:
                            ev_items.append(tagged)
                    evidence_by_login[login_key] = tuple(ev_items[:8]) or (f"message_evidence:{login_key}->{uid}",)
                    continue
                if len(strong_notes) >= 2:
                    msg_conflict_logins.add(login_key)
                    claimants = sorted(str(n["discord_id"]) for n in strong_notes)
                    logging.warning(
                        "1-to-1 message_evidence conflict for github_login=%s (discord_ids=%s); granting nothing",
                        group["login"],
                        claimants,
                    )
                    skip_reason_by_login[login_key] = "discord_id_conflict"
                    skip_meta_by_login[login_key] = (
                        max(float(n.get("confidence") or 0.0) for n in strong_notes),
                        f"1-to-1 conflict: multiple members claimed {group['login']} ({claimants})",
                    )
                    continue

                # Non-strong corroborating message evidence (e.g. code paths / PR topic / non-ownership link):
                # attach as an extra signal on CandidatePair so Tier (d) LLM can evaluate it alongside
                # GitHub profile signals under the >= 2 signals & >= 0.85 confidence rule.
                full_strong, existing_cands = full_scores_cache[login_key]
                cand_by_uid = {c.member.discord_id: c for c in existing_cands}
                updated = False
                for note in notes_for_login:
                    uid = str(note["discord_id"])
                    note_conf = float(note.get("confidence") or 0.0)
                    ev_list = list(note.get("evidence") or ())
                    if note_conf <= 0.0 or not ev_list:
                        continue
                    msg_signal = f"message_evidence:{ev_list[0][:100]}"
                    if uid in cand_by_uid:
                        old_c = cand_by_uid[uid]
                        if msg_signal not in old_c.signals:
                            cand_by_uid[uid] = CandidatePair(
                                profile=old_c.profile,
                                member=old_c.member,
                                heuristic_score=round(min(0.95, max(old_c.heuristic_score, note_conf) + 0.10), 4),
                                signals=old_c.signals + (msg_signal,),
                            )
                            updated = True
                    elif uid in member_by_id:
                        cand_by_uid[uid] = CandidatePair(
                            profile=profile,
                            member=member_by_id[uid],
                            heuristic_score=round(note_conf, 4),
                            signals=(msg_signal,),
                        )
                        updated = True
                if updated:
                    merged_cands = sorted(
                        cand_by_uid.values(),
                        key=lambda c: (-c.heuristic_score, c.member.discord_id),
                    )[:MAX_CANDIDATES_PER_AUTHOR]
                    full_scores_cache[login_key] = (full_strong, merged_cands)

        # Tier (d): List-wise LLM selection over remaining unmatched enriched authors.
        # Pass 1: priority pool candidates.
        if use_priority_pool and recent_index is not None:
            recent_candidates: list[CandidatePair] = []
            for group in eligible_pending:
                login_key = group["login"].strip().casefold()
                if login_key in resolved or login_key in cached_negative_keys or skip_reason_by_login.get(login_key) == "discord_id_conflict":
                    continue
                profile = profiles_by_login[login_key]
                _strong, candidates = score_author_against_members(
                    profile,
                    recent_members,
                    member_index=recent_index,
                    pr_link_posters=pr_link_evidence_by_login.get(login_key),
                    excluded_discord_ids=excluded_staff_ids,
                    multi_author_ids=multi_author_ids,
                    rejected_discord_ids=rejected_by_login.get(login_key, set()),
                )
                if candidates and _llm_allowed(login_key):
                    recent_candidates.extend(candidates)

            if recent_candidates:
                ambiguous_candidates.extend(recent_candidates)
                llm_deadline = (
                    (deadline - min(5.0, max(0.0, (deadline - time.monotonic()) * 0.2)))
                    if deadline is not None else None
                )
                recent_winners, recent_llm_source, _recent_skips, _recent_meta, recent_ev = _disambiguate_with_reasons(
                    config,
                    recent_candidates,
                    llm_complete=complete_fn,
                    jev_transport=jev_transport,
                    deadline=llm_deadline,
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

        # Pass 2: full member list LLM fallback ONLY for authors still unmatched after Pass 1.
        full_candidates: list[CandidatePair] = []
        for group in eligible_pending:
            login_key = group["login"].strip().casefold()
            if login_key in resolved or login_key in cached_negative_keys or skip_reason_by_login.get(login_key) == "discord_id_conflict":
                continue
            _strong, candidates = full_scores_cache[login_key]
            if candidates and _llm_allowed(login_key):
                full_candidates.extend(candidates)
            elif candidates:
                top_h = candidates[0].heuristic_score
                skip_reason_by_login[login_key] = "no_signal_match"
                skip_meta_by_login[login_key] = (top_h, "github profile lacks enrichment signals for LLM matching")
            else:
                skip_reason_by_login[login_key] = "no_signal_match"
                skip_meta_by_login[login_key] = (0.0, "no candidate signals")

        if full_candidates:
            ambiguous_candidates.extend(full_candidates)
            llm_deadline = (
                (deadline - min(5.0, max(0.0, (deadline - time.monotonic()) * 0.2)))
                if deadline is not None else None
            )
            llm_winners, full_llm_source, llm_skip_reasons, llm_skip_meta, full_ev = _disambiguate_with_reasons(
                config,
                full_candidates,
                llm_complete=complete_fn,
                jev_transport=jev_transport,
                deadline=llm_deadline,
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
                and login_key not in failed_enrichment_logins
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
    message_searcher=None,
    allow_live_message_search: bool | None = None,
    skip_github_enrichment: bool = False,
    org_members: frozenset[str] = frozenset(),
    sleep_fn=time.sleep,
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
        message_searcher=message_searcher,
        allow_live_message_search=allow_live_message_search,
        skip_github_enrichment=skip_github_enrichment,
        org_members=org_members,
        sleep_fn=sleep_fn,
    )
    return matched, skipped, details
