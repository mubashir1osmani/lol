"""Friday summary of community pull requests merged since the previous Friday post."""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from .api import GITHUB_API, TransientAPIError, fetch_guild_channels, github_get_json, post_message
from .config import is_discord_id
from .matcher import MatchDetail, MatchDiagnostics, match_community_contributors_with_diagnostics
from .routing import FAST_MODEL
from .schedule import UTC, iso, parse_time

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

WEEKLY_CHUNK_LIMIT = 1900
SUPPRESS_EMBEDS_FLAG = 1 << 2  # Discord message flag: SUPPRESS_EMBEDS (4)
WEEKLY_HEADER = "thank you to this week's contributors 🙏"
DEFAULT_WEEKLY_CHANNEL_NAME = "announcements"
_POSTABLE_CHANNEL_TYPES = frozenset({0, 5})  # GUILD_TEXT, GUILD_ANNOUNCEMENT
WEEKLY_LOOKBACK_MAX_DAYS = 14

COMMUNITY_AUTHOR_ASSOCIATIONS = frozenset({"CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR"})
EXCLUDED_AUTHOR_ASSOCIATIONS = frozenset({"MEMBER", "OWNER", "COLLABORATOR"})

ORG_MEMBERS_CACHE_TTL_SECONDS = 3600.0
_org_cache_lock = threading.Lock()
_org_members_cache: dict[str, tuple[frozenset[str], float]] = {}

_MENTION_ID = re.compile(r"<@([1-9][0-9]{0,19})>")
_LEADING_BRACKET_TAG = re.compile(r"^\s*\[[^\]]{1,40}\]\s*")
_TICKET_PREFIX = re.compile(r"^\s*(?:[A-Z]{2,12}-[0-9]+|GH-[0-9]+|#[0-9]+)\s*[:\-–—]?\s*", re.IGNORECASE)
_CONVENTIONAL_PREFIX = re.compile(
    r"^\s*(feat|feature|fix|bugfix|bug|chore|docs|doc|refactor|perf|test|tests|ci|build|style|revert)"
    r"(?:\(([^)]{1,40})\))?\s*!?\s*:\s*",
    re.IGNORECASE,
)
_TRAILING_REF = re.compile(
    r"(?:\s*\((?:(?:fix(?:es)?|close[sd]?|resolve[sd]?)\s+)?#[0-9]+\)|\s*\[#[0-9]+\]|\s+(?:fix(?:es)?|close[sd]?|resolve[sd]?)\s+#[0-9]+)+\s*$",
    re.IGNORECASE,
)
_INLINE_TICKET = re.compile(r"\b[A-Z]{2,12}-[0-9]+\b")
_DISALLOWED_MENTIONS = re.compile(r"<@[!&]?[0-9]+>|@(?:everyone|here)", re.IGNORECASE)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")

_LEADING_VERBS = frozenset({
    "add", "allow", "avoid", "bump", "cache", "check", "clean", "cleanup", "close",
    "convert", "correct", "create", "default", "defer", "delete", "disable", "drop",
    "emit", "enable", "enforce", "ensure", "expose", "extract", "fall", "fallback",
    "fetch", "filter", "fix", "format", "forward", "guard", "handle", "ignore",
    "implement", "improve", "include", "init", "initialize", "inject", "keep",
    "limit", "load", "log", "make", "map", "merge", "migrate", "move", "normalize",
    "parse", "pass", "persist", "populate", "preserve", "prevent", "prune", "raise",
    "read", "record", "reduce", "refactor", "register", "reject", "relay", "release",
    "remove", "rename", "render", "replace", "report", "require", "reset", "resolve",
    "respect", "restore", "retry", "return", "rewrite", "route", "run", "sanitize",
    "save", "send", "set", "show", "simplify", "skip", "sort", "split", "stop",
    "stream", "strip", "support", "switch", "sync", "tag", "track", "trim", "unify",
    "update", "upgrade", "use", "validate", "verify", "wire", "wrap", "write", "bill",
})
_TRAILING_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "if", "in",
    "instead", "into", "is", "not", "of", "on", "or", "that", "the", "to", "via",
    "when", "where", "which", "while", "with", "without",
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


@dataclass(frozen=True)
class WeeklyReport:
    chunks: list[str]
    matched_count: int
    skipped_count: int
    since: datetime
    rate_limited: bool = False
    match_details: tuple[MatchDetail, ...] = ()
    issues_url: str = ""
    diagnostics: MatchDiagnostics | None = None


def reset_org_members_cache() -> None:
    """Test helper."""
    with _org_cache_lock:
        _org_members_cache.clear()


class WeeklyFetchError(RuntimeError):
    """GitHub search failed while building the weekly merge summary."""

    def __init__(self, message: str, *, rate_limited: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.rate_limited = rate_limited
        self.retry_after = retry_after


def parse_weekday(value: str) -> int:
    key = (value or "friday").strip().lower()
    if key not in _WEEKDAYS:
        raise ValueError("MERGED_POST_WEEKDAY must be a weekday name")
    return _WEEKDAYS[key]


def _channel_type_int(channel) -> int | None:
    raw = channel.get("type") if isinstance(channel, dict) else getattr(channel, "type", None)
    val = getattr(raw, "value", raw)
    return val if isinstance(val, int) and not isinstance(val, bool) else None


def _channel_name_matches(channel, target_name: str) -> str | None:
    name = channel.get("name") if isinstance(channel, dict) else getattr(channel, "name", None)
    if not isinstance(name, str) or name.strip().lower() != target_name:
        return None
    ctype = _channel_type_int(channel)
    if ctype is not None and ctype not in _POSTABLE_CHANNEL_TYPES:
        return None
    cid = channel.get("id") if isinstance(channel, dict) else getattr(channel, "id", None)
    cid_str = str(cid) if cid is not None else ""
    return cid_str if is_discord_id(cid_str) else None


def resolve_weekly_channel_id(
    config,
    *,
    guild=None,
    channels_fetcher=None,
    channel_name: str = DEFAULT_WEEKLY_CHANNEL_NAME,
) -> str | None:
    """Resolve the weekly post channel.

    Uses ANNOUNCEMENTS_CHANNEL_ID standalone when set; otherwise falls back to
    the guild text/announcement channel named 'announcements' (cache, then REST).
    """
    announcements_id = str(getattr(config, "announcements_channel_id", "") or "").strip()
    if is_discord_id(announcements_id):
        return announcements_id

    target = (channel_name or DEFAULT_WEEKLY_CHANNEL_NAME).strip().lower()
    if guild is not None:
        for attr in ("text_channels", "channels"):
            for channel in getattr(guild, attr, ()) or ():
                found = _channel_name_matches(channel, target)
                if found:
                    return found

    guild_id = getattr(config, "discord_guild_id", None) or (str(getattr(guild, "id", "")) if guild is not None else "")
    token = getattr(config, "discord_token", None) or ""
    if is_discord_id(str(guild_id)) and token:
        fetcher = channels_fetcher or fetch_guild_channels
        try:
            for channel in fetcher(str(guild_id), token) or ():
                found = _channel_name_matches(channel, target)
                if found:
                    return found
        except Exception as error:
            logging.warning("Could not resolve #%s via guild channels REST lookup: %s", target, error)
    return None


def weekly_lookback_since(store, repository: str, moment: datetime) -> datetime:
    """Cover the week since the previous Friday post, defaulting to 7 days."""
    default_since = moment - timedelta(days=7)
    if store is None or not repository or not hasattr(store, "last_merged_week"):
        return default_since
    try:
        last = store.last_merged_week(repository)
    except Exception:
        return default_since
    if not last or not isinstance(last.get("posted_at"), str):
        return default_since
    try:
        previous = parse_time(last["posted_at"])
    except Exception:
        return default_since
    elapsed = moment - previous
    if timedelta(hours=1) <= elapsed <= timedelta(days=WEEKLY_LOOKBACK_MAX_DAYS):
        return previous
    return default_since


def week_key(moment: datetime) -> str:
    year, week, _day = moment.isocalendar()
    return f"{year}-W{week:02d}"


def weekly_is_due(now: datetime, weekday: int, hhmm: str, timezone: str, last_key: str | None) -> bool:
    local = now.astimezone(ZoneInfo(timezone))
    if local.weekday() != weekday:
        return False
    hour, minute = (int(part) for part in hhmm.split(":", 1))
    if (local.hour, local.minute) < (hour, minute):
        return False
    return week_key(local) != (last_key or "")


def _bot_login(login: str, user_type: str) -> bool:
    if (user_type or "").lower() == "bot":
        return True
    lowered = (login or "").lower()
    return lowered.endswith("[bot]") or lowered.endswith("-bot")


def _org_staff_login(login: str, org: str | None) -> bool:
    """BerriAI staff accounts often keep org membership private and use -berri handles."""
    if not org or org.lower() != "berriai":
        return False
    lowered = (login or "").lower()
    return lowered.endswith("-berri") or lowered.endswith("-berriai") or lowered == "berriai"


def fetch_org_public_members(
    org: str,
    token: str | None,
    *,
    ttl_seconds: float = ORG_MEMBERS_CACHE_TTL_SECONDS,
    clock=time.monotonic,
) -> frozenset[str]:
    """Lowercased logins of public GitHub org members, cached with a TTL."""
    key = (org or "").strip().lower()
    if not key:
        return frozenset()
    now = clock()
    with _org_cache_lock:
        cached = _org_members_cache.get(key)
        if cached is not None and now < cached[1]:
            return cached[0]
    members: set[str] = set()
    page = 1
    try:
        while page <= 5:
            payload = github_get_json(
                f"{GITHUB_API}/orgs/{org}/public_members?per_page=100&page={page}",
                token,
                pace=False,
            )
            if not isinstance(payload, list) or not payload:
                break
            for item in payload:
                login = item.get("login") if isinstance(item, dict) else None
                if isinstance(login, str) and login:
                    members.add(login.lower())
            if len(payload) < 100:
                break
            page += 1
    except Exception as error:
        logging.warning("Could not fetch public members for org %s: %s", org, error)
        return frozenset()
    result = frozenset(members)
    with _org_cache_lock:
        _org_members_cache[key] = (result, now + ttl_seconds)
    return result


def good_first_issue_url(repository: str, _token: str | None = None) -> str:
    repo = (repository or "BerriAI/litellm").strip()
    return f"https://github.com/{repo}/issues"


def weekly_footer(issues_url: str) -> str:
    return f"want to be on next week's list? grab a good first issue: <{issues_url}>"


def clean_pr_title(title: str, *, number: int | None = None, max_words: int = 9) -> str:
    """Deterministic lowercase ~8-word plain-English cleanup of a PR title."""
    text = (title or "").strip()
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _DISALLOWED_MENTIONS.sub("", text)
    text = text.replace("<", "").replace(">", "")
    text = _TRAILING_REF.sub("", text).strip()

    # Strip leading [TAG] / ticket IDs and conventional-commit prefixes.
    for _ in range(3):
        prev = text
        text = _LEADING_BRACKET_TAG.sub("", text).strip()
        text = _TICKET_PREFIX.sub("", text).strip()
        if text == prev:
            break

    prefix_kind = ""
    prefix_scope = ""
    match = _CONVENTIONAL_PREFIX.match(text)
    if match:
        prefix_kind = (match.group(1) or "").lower()
        prefix_scope = (match.group(2) or "").strip().lower()
        text = text[match.end():].strip()
        text = _TICKET_PREFIX.sub("", text).strip()

    text = _INLINE_TICKET.sub("", text)
    text = text.replace("`", "").replace('"', "").strip()
    text = re.sub(r"\s+", " ", text).strip(" -–—:;.,!?")
    lowered = text.lower()

    # If a fix(...) title has a noun phrase without a verb, keep "fix" (and scope if omitted).
    if lowered and prefix_kind in {"fix", "bug", "bugfix"}:
        first_word = re.sub(r"[^a-z]+", "", lowered.split()[0])
        if first_word and first_word not in _LEADING_VERBS:
            if prefix_scope and prefix_scope not in lowered and re.fullmatch(r"[a-z0-9._-]{2,20}", prefix_scope):
                lowered = f"fix {prefix_scope} {lowered}"
            else:
                lowered = f"fix {lowered}"

    words = lowered.split()
    if len(words) > max_words:
        words = words[:max_words]
        while len(words) > 3 and words[-1].rstrip(",;:") in _TRAILING_STOPWORDS:
            words.pop()
    result = " ".join(words).rstrip(" ,;:.!?-–—")
    if not result:
        return f"pr #{number}" if number else "update"
    return result


def _sanitize_summary(text: str, fallback: str) -> str:
    cleaned = (text or "").strip().strip("\"'")
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    cleaned = _DISALLOWED_MENTIONS.sub("", cleaned)
    cleaned = cleaned.replace("<", "").replace(">", "").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—:;.,!?").lower()
    words = cleaned.split()
    if not words or len(words) > 14:
        return fallback
    return " ".join(words).rstrip(" ,;:.!?-–—") or fallback


def _has_llm_credentials() -> bool:
    return any(os.environ.get(name, "").strip() for name in _LLM_API_KEY_ENVS)


def summarize_pr_titles(config, groups: list[dict], *, summarizer=None) -> dict[int, str]:
    """Rewrite matched PR titles into short lowercase plain-English summaries (~8 words).

    Uses one batched fast-tier (Haiku) call when configured, falling back to
    deterministic cleanup cleanly when offline or on any model error.
    """
    pulls_by_num: dict[int, str] = {}
    for group in groups:
        for pull in group.get("pulls", ()):
            num = pull.get("number")
            if isinstance(num, int) and num not in pulls_by_num:
                pulls_by_num[num] = pull.get("title") or f"#{num}"

    fallbacks = {num: clean_pr_title(title, number=num) for num, title in pulls_by_num.items()}
    if not pulls_by_num:
        return fallbacks

    if summarizer is None and not _has_llm_credentials():
        return fallbacks

    model = (
        getattr(config, "review_classifier_model", None)
        or getattr(config, "fast_model", None)
        or FAST_MODEL
    )
    items_payload = {str(num): title for num, title in pulls_by_num.items()}
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite each pull request title as a casual, lowercase plain-English summary "
                "of roughly 6 to 8 words. Drop conventional-commit prefixes (like fix(...): or feat:) "
                "and ticket/issue IDs. No trailing punctuation, no links, no mentions. "
                "Reply with a single JSON object mapping each PR number string to its summary."
            ),
        },
        {"role": "user", "content": json.dumps(items_payload)},
    ]
    try:
        if summarizer is not None:
            raw = summarizer(model, messages)
        else:
            from .review_llm import complete_litellm
            raw = complete_litellm(model, messages)
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start: end + 1]) if isinstance(raw, str) and start >= 0 and end >= start else {}
    except Exception as error:
        logging.warning("PR title summarization fallback to deterministic cleanup: %s", error)
        return fallbacks

    if not isinstance(parsed, dict):
        return fallbacks
    summaries = dict(fallbacks)
    for num, fallback in fallbacks.items():
        candidate = parsed.get(str(num))
        if isinstance(candidate, str) and candidate.strip():
            summaries[num] = _sanitize_summary(candidate, fallback)
    return summaries


def community_groups(
    items: list[dict],
    maintainers: tuple[str, ...],
    links: dict[int, str],
    *,
    org_members: frozenset[str] | set[str] = frozenset(),
    org: str | None = "BerriAI",
    require_discord: bool = False,
) -> list[dict]:
    """Group merged PRs by external community contributors only.

    Excludes bots, MAINTAINER_GITHUB_LOGINS, GitHub org public members,
    BerriAI staff handles, and PRs whose author_association is not
    CONTRIBUTOR or FIRST_TIME_CONTRIBUTOR (such as MEMBER/OWNER/COLLABORATOR).
    """
    skip = {login.lower() for login in maintainers} | {login.lower() for login in org_members}
    grouped: dict[str, dict] = {}
    for item in items:
        login = item.get("login") or ""
        if (
            not login
            or login.lower() in skip
            or _bot_login(login, item.get("type") or "")
            or _org_staff_login(login, org)
        ):
            continue
        association = (item.get("author_association") or "").strip().upper()
        if association not in COMMUNITY_AUTHOR_ASSOCIATIONS:
            continue
        user_id = item.get("user_id")
        discord_id = links.get(user_id)
        if require_discord and not (discord_id and is_discord_id(str(discord_id))):
            continue
        bucket = grouped.setdefault(login, {
            "login": login,
            "user_id": user_id if isinstance(user_id, int) else None,
            "discord_id": str(discord_id) if discord_id and is_discord_id(str(discord_id)) else None,
            "first_time": False,
            "pulls": [],
        })
        if association == "FIRST_TIME_CONTRIBUTOR" or item.get("first_time"):
            bucket["first_time"] = True
        bucket["pulls"].append({
            "number": item["number"],
            "title": item.get("title") or f"#{item['number']}",
            "url": item.get("url") or "",
            "author_association": association,
        })
    return sorted(grouped.values(), key=lambda group: (-len(group["pulls"]), group["login"].lower()))


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def chunk_mention_user_ids(chunk: str) -> list[str]:
    """Unique Discord user IDs tagged in a chunk, for explicit allowed_mentions.users."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _MENTION_ID.finditer(chunk or ""):
        uid = match.group(1)
        if uid not in seen and len(ordered) < 100:
            seen.add(uid)
            ordered.append(uid)
    return ordered


def _pull_entry(pull: dict, summaries: dict[int, str] | None = None) -> str:
    number = pull.get("number")
    summary = (
        summaries.get(number)
        if summaries and isinstance(number, int) and number in summaries
        else pull.get("summary") or clean_pr_title(pull.get("title") or "", number=number if isinstance(number, int) else None)
    )
    url = pull.get("url") or ""
    return f"{summary} (<{url}>)" if url else summary


def _contributor_lines(
    group: dict,
    *,
    summaries: dict[int, str] | None = None,
    limit: int = WEEKLY_CHUNK_LIMIT,
    first_line_limit: int | None = None,
) -> list[str]:
    """Render one line per contributor, splitting only if a single line exceeds ``limit``."""
    discord_id = group.get("discord_id")
    if not discord_id or not is_discord_id(str(discord_id)):
        raise ValueError("Weekly contributor lines require a resolved Discord user ID")
    badge = " 🎉" if group.get("first_time") else ""
    prefix = f"<@{discord_id}>{badge} - "
    entries = [_pull_entry(pull, summaries) for pull in group.get("pulls", ())]
    if not entries:
        return []
    head_limit = first_line_limit if first_line_limit is not None else limit
    full_line = prefix + ", ".join(entries)
    if _utf16_len(full_line) <= head_limit:
        return [full_line]

    # Only wrap across lines when a single contributor's PR list exceeds Discord's limit.
    lines: list[str] = []
    current = prefix + entries[0]
    active_limit = head_limit
    for entry in entries[1:]:
        candidate = f"{current}, {entry}"
        if _utf16_len(candidate) <= active_limit:
            current = candidate
        else:
            lines.append(current)
            current = f"<@{discord_id}> - {entry}"
            active_limit = limit
    lines.append(current)
    return lines


def _default_issues_url(repository: str = "BerriAI/litellm") -> str:
    return f"https://github.com/{repository}/issues"


def render_weekly(
    groups: list[dict],
    *,
    since: datetime,
    summaries: dict[int, str] | None = None,
    issues_url: str | None = None,
) -> str:
    return "\n\n".join(
        weekly_chunks(
            groups,
            since=since,
            summaries=summaries,
            issues_url=issues_url,
            limit=100_000,
        )
    )


def weekly_chunks(
    groups: list[dict],
    *,
    since: datetime,
    limit: int = WEEKLY_CHUNK_LIMIT,
    max_chunks: int | None = None,
    note: str = "",
    summaries: dict[int, str] | None = None,
    issues_url: str | None = None,
) -> list[str]:
    """Render the weekly thank-you post, splitting cleanly only when over ``limit``."""
    start = since.date().isoformat()
    matched = [group for group in groups if group.get("discord_id") and is_discord_id(str(group["discord_id"]))]
    if not matched:
        empty = f"no discord-matched community pull requests merged since {start}."
        if note:
            empty = f"{empty}\n\n{note}"
        return [empty[:limit]]

    first_line_budget = max(200, limit - _utf16_len(WEEKLY_HEADER) - 2)
    rendered_lines: list[str] = []
    for idx, group in enumerate(matched):
        rendered_lines.extend(
            _contributor_lines(
                group,
                summaries=summaries,
                limit=limit,
                first_line_limit=first_line_budget if idx == 0 else limit,
            )
        )

    chunks: list[str] = []
    current = WEEKLY_HEADER
    first_in_chunk = True

    for line in rendered_lines:
        sep = "\n\n" if (not chunks and first_in_chunk) else "\n"
        candidate = f"{current}{sep}{line}" if current else line
        if _utf16_len(candidate) <= limit:
            current = candidate
            first_in_chunk = False
            continue
        if current:
            chunks.append(current)
        if max_chunks is not None and len(chunks) >= max_chunks:
            current = ""
            break
        current = line
        first_in_chunk = False

    if current:
        chunks.append(current)

    footer_parts = [weekly_footer(issues_url or _default_issues_url())]
    if note:
        footer_parts.append(note)
    footer = "\n".join(footer_parts)

    if chunks and _utf16_len(chunks[-1] + "\n\n" + footer) <= limit:
        chunks[-1] = chunks[-1] + "\n\n" + footer
    elif max_chunks is None or len(chunks) < max_chunks:
        chunks.append(footer[:limit])
    return chunks


def _is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, TransientAPIError):
        return True
    text = str(error)
    return "HTTP 403" in text or "HTTP 429" in text or "rate limit" in text.lower()


def _search_merged_pages(
    repository: str,
    since: datetime,
    token: str | None,
    *,
    pace: bool = False,
    max_pages: int = 5,
) -> tuple[list[dict], bool]:
    cutoff = since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    found = []
    seen_numbers: set[int] = set()
    rate_limited = False
    page = 1
    while page <= max_pages:
        params = urlencode({
            "q": f"repo:{repository} is:pr is:merged merged:>={cutoff}",
            "sort": "updated",
            "order": "desc",
            "per_page": 100,
            "page": page,
        })
        try:
            payload = github_get_json(
                f"{GITHUB_API}/search/issues?{params}",
                token,
                pace=pace,
            )
        except Exception as error:
            if _is_rate_limit_error(error):
                retry_after = getattr(error, "retry_after", None)
                if found:
                    logging.warning("GitHub search rate limit hit on page %d; returning %d rows", page, len(found))
                    rate_limited = True
                    break
                raise WeeklyFetchError(str(error), rate_limited=True, retry_after=retry_after) from error
            raise WeeklyFetchError(str(error)) from error
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            break
        for item in items:
            if not isinstance(item, dict):
                continue
            user = item.get("user") if isinstance(item.get("user"), dict) else {}
            number = item.get("number")
            if not isinstance(number, int) or number in seen_numbers:
                continue
            seen_numbers.add(number)
            found.append({
                "number": number,
                "title": item.get("title") if isinstance(item.get("title"), str) else "",
                "url": item.get("html_url") if isinstance(item.get("html_url"), str)
                else f"https://github.com/{repository}/pull/{number}",
                "login": user.get("login") if isinstance(user.get("login"), str) else "",
                "type": user.get("type") if isinstance(user.get("type"), str) else "",
                "user_id": user.get("id") if isinstance(user.get("id"), int) else None,
                "author_association": item.get("author_association")
                if isinstance(item.get("author_association"), str) else "",
            })
        if len(items) < 100:
            break
        page += 1
    return found, rate_limited


def fetch_merged_community(
    repository: str,
    since: datetime,
    token: str | None,
    *,
    pace: bool = False,
    max_pages: int = 5,
) -> list[dict]:
    """Merged pull requests since ``since``. Search is discovery; each row is one issue."""
    rows, _rate_limited = _search_merged_pages(repository, since, token, pace=pace, max_pages=max_pages)
    return rows


def _mark_first_time_contributors(
    matched: list[dict],
    repository: str,
    token: str | None,
) -> list[dict]:
    """Detect first-time contributors whose author_association flipped to CONTRIBUTOR after merge.

    GitHub changes ``FIRST_TIME_CONTRIBUTOR`` to ``CONTRIBUTOR`` once a PR merges, so
    for each matched contributor not already flagged we check whether all of their
    merged PRs in ``repository`` landed in this week's batch (``total_count <= len(pulls)``).
    """
    if not matched or not repository:
        return matched
    updated: list[dict] = []
    rate_limited = False
    for group in matched:
        if group.get("first_time") or rate_limited:
            updated.append(group)
            continue
        login = group.get("login") or ""
        if not login or not re.fullmatch(r"[A-Za-z0-9-]+", login):
            updated.append(group)
            continue
        first_time = False
        try:
            params = urlencode({
                "q": f"repo:{repository} is:pr is:merged author:{login}",
                "per_page": 1,
            })
            payload = github_get_json(
                f"{GITHUB_API}/search/issues?{params}",
                token,
                pace=False,
            )
            total = payload.get("total_count") if isinstance(payload, dict) else None
            if isinstance(total, int) and total > 0 and total <= len(group.get("pulls", ())):
                first_time = True
        except Exception as error:
            if _is_rate_limit_error(error):
                rate_limited = True
            logging.info("Could not check prior merged PR count for %s: %s", login, error)
        updated.append({**group, "first_time": first_time} if first_time else group)
    return updated


def build_weekly_report(
    config,
    store,
    now: datetime | None = None,
    *,
    since: datetime | None = None,
    guild=None,
    loop=None,
    pace: bool = False,
    max_chunks: int | None = None,
    limit: int = WEEKLY_CHUNK_LIMIT,
    members=None,
    jev_transport=None,
    haiku_complete=None,
    title_summarizer=None,
    rest_fetcher=None,
    rest_searcher=None,
) -> WeeklyReport:
    moment = now or datetime.now(UTC)
    window_since = since if since is not None else weekly_lookback_since(store, config.github_repository, moment)
    token = config.github_token or getattr(config, "github_actor_token", None)
    org = config.github_repository.split("/", 1)[0] if "/" in config.github_repository else ""
    org_members = fetch_org_public_members(org, token) if org else frozenset()
    rows, rate_limited = _search_merged_pages(config.github_repository, window_since, token, pace=pace)
    raw_groups = community_groups(
        rows,
        config.maintainer_logins,
        {},
        org_members=org_members,
        org=org,
    )
    matched, skipped, details, diagnostics = match_community_contributors_with_diagnostics(
        raw_groups,
        config,
        store,
        guild=guild,
        loop=loop,
        now=moment,
        members=members,
        github_getter=github_get_json,
        jev_transport=jev_transport,
        haiku_complete=haiku_complete,
        rest_fetcher=rest_fetcher,
        rest_searcher=rest_searcher,
    )
    if matched and not rate_limited:
        matched = _mark_first_time_contributors(matched, config.github_repository, token)
    logging.info(
        "Weekly merge summary: %d contributors matched in Discord, %d unmatched skipped (members=%d source=%s)",
        len(matched),
        skipped,
        diagnostics.member_count,
        diagnostics.members_source,
    )
    summaries = summarize_pr_titles(config, matched, summarizer=title_summarizer) if matched else {}
    issues_url = good_first_issue_url(config.github_repository, token) if matched else ""
    note = ""
    if rate_limited:
        note = "(stopped early because of github search rate limits; set GITHUB_TOKEN for higher limits.)"
    chunks = weekly_chunks(
        matched,
        since=window_since,
        limit=limit,
        max_chunks=max_chunks,
        note=note,
        summaries=summaries,
        issues_url=issues_url,
    )
    return WeeklyReport(
        chunks=chunks,
        matched_count=len(matched),
        skipped_count=skipped,
        since=window_since,
        rate_limited=rate_limited,
        match_details=tuple(details),
        issues_url=issues_url,
        diagnostics=diagnostics,
    )


def build_weekly(config, store, now: datetime | None = None, *, guild=None, loop=None) -> str:
    report = build_weekly_report(config, store, now, guild=guild, loop=loop, pace=False)
    return "\n\n".join(report.chunks)


def build_weekly_chunks(
    config,
    store,
    now: datetime | None = None,
    *,
    guild=None,
    loop=None,
    pace: bool = False,
    max_chunks: int | None = None,
    limit: int = WEEKLY_CHUNK_LIMIT,
) -> list[str]:
    report = build_weekly_report(
        config,
        store,
        now,
        guild=guild,
        loop=loop,
        pace=pace,
        max_chunks=max_chunks,
        limit=limit,
    )
    return report.chunks


def post_weekly_if_due(
    config,
    store,
    now: datetime | None = None,
    *,
    post=None,
    guild=None,
    loop=None,
    channels_fetcher=None,
) -> str | None:
    """Post this week's summary once on the configured weekday/time.

    Safe across process restarts:
    - Before the scheduled time: returns None.
    - At or after the scheduled time on the target weekday: posts once and
      persists ``week_key`` in SQLite so later restarts never double-post.
    """
    if not config.merged_weekly_enabled:
        return None
    moment = now or datetime.now(UTC)
    local = moment.astimezone(ZoneInfo(config.merged_timezone))
    key = week_key(local)
    if not weekly_is_due(
        moment, config.merged_weekday, config.merged_time, config.merged_timezone,
        key if store.merged_week_posted(config.github_repository, key) else None,
    ):
        return None
    channel = resolve_weekly_channel_id(config, guild=guild, channels_fetcher=channels_fetcher)
    if not channel:
        logging.warning("Weekly merge summary skipped: no destination channel configured or #announcements found")
        return None
    try:
        report = build_weekly_report(
            config,
            store,
            moment,
            guild=guild,
            loop=loop,
            pace=True,
        )
    except WeeklyFetchError as error:
        if error.rate_limited:
            wait_note = f" (retry_after={error.retry_after:.0f}s)" if isinstance(error.retry_after, (int, float)) else ""
            logging.warning(
                "Weekly merge summary deferred due to GitHub rate limit%s; will retry automatically: %s",
                wait_note,
                error,
            )
            return None
        raise
    if report.matched_count == 0:
        diag = report.diagnostics.summary_line() if report.diagnostics is not None else ""
        logging.warning("Weekly merge summary skipped: 0 discord-matched contributors %s", diag)
        # If guild members were actually loaded and truly 0 matched, record the week so
        # the 60s scheduler does not re-query GitHub every minute. If 0 members were
        # loaded (transient outage), leave unmarked so the next tick retries.
        if report.diagnostics is not None and report.diagnostics.member_count > 0:
            store.mark_merged_week(config.github_repository, key, iso(moment))
        return None
    sender = post or post_message
    recorded = False
    for chunk in report.chunks:
        users = chunk_mention_user_ids(chunk)
        sender(channel, config.discord_token, {
            "content": chunk,
            "flags": SUPPRESS_EMBEDS_FLAG,
            "allowed_mentions": {"parse": [], "users": users},
        })
        if not recorded:
            recorded = True
            if not store.mark_merged_week(config.github_repository, key, iso(moment)):
                logging.info("Weekly merge summary for %s was already recorded", key)
    logging.info("Posted weekly merge summary for %s to channel %s (%d chunks)", key, channel, len(report.chunks))
    return "\n\n".join(report.chunks)
