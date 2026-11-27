"""Monday summary of community pull requests merged in the last seven days."""

import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from .api import GITHUB_API, TransientAPIError, github_get_json, post_message
from .config import is_discord_id
from .matcher import MatchDetail, match_community_contributors
from .schedule import UTC, iso

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

WEEKLY_CHUNK_LIMIT = 1900
WEEKLY_PREVIEW_MAX_CHUNKS = 5
WEEKLY_POST_MAX_CHUNKS = 10
WEEKLY_MAX_CONTRIBUTORS = 15
WEEKLY_MAX_PRS_PER_LINE = 3
SUPPRESS_EMBEDS_FLAG = 1 << 2  # Discord message flag: SUPPRESS_EMBEDS (4)

COMMUNITY_AUTHOR_ASSOCIATIONS = frozenset({"CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR"})
EXCLUDED_AUTHOR_ASSOCIATIONS = frozenset({"MEMBER", "OWNER", "COLLABORATOR"})

ORG_MEMBERS_CACHE_TTL_SECONDS = 3600.0
_org_cache_lock = threading.Lock()
_org_members_cache: dict[str, tuple[frozenset[str], float]] = {}

_MENTION_ID = re.compile(r"<@([1-9][0-9]{0,19})>")


@dataclass(frozen=True)
class WeeklyReport:
    chunks: list[str]
    matched_count: int
    skipped_count: int
    since: datetime
    rate_limited: bool = False
    match_details: tuple[MatchDetail, ...] = ()


def reset_org_members_cache() -> None:
    """Test helper."""
    with _org_cache_lock:
        _org_members_cache.clear()


class WeeklyFetchError(RuntimeError):
    """GitHub search failed while building the weekly merge summary."""

    def __init__(self, message: str, *, rate_limited: bool = False):
        super().__init__(message)
        self.rate_limited = rate_limited


def parse_weekday(value: str) -> int:
    key = (value or "monday").strip().lower()
    if key not in _WEEKDAYS:
        raise ValueError("MERGED_POST_WEEKDAY must be a weekday name")
    return _WEEKDAYS[key]


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
            "pulls": [],
        })
        bucket["pulls"].append({
            "number": item["number"],
            "title": item.get("title") or f"#{item['number']}",
            "url": item.get("url") or "",
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


def _compact_contributor_line(group: dict, *, max_prs: int = WEEKLY_MAX_PRS_PER_LINE) -> str:
    discord_id = group.get("discord_id")
    if not discord_id or not is_discord_id(str(discord_id)):
        raise ValueError("Weekly contributor lines require a resolved Discord user ID")
    pulls = group["pulls"]
    count = len(pulls)
    noun = "PR" if count == 1 else "PRs"
    shown = pulls[:max_prs]
    refs = []
    for pull in shown:
        number = pull["number"]
        url = pull.get("url") or ""
        refs.append(f"[#{number}](<{url}>)" if url else f"#{number}")
    if count > max_prs:
        refs.append(f"+{count - max_prs} more")
    return f"- <@{discord_id}>: {count} {noun} ({', '.join(refs)})"


def render_weekly(
    groups: list[dict],
    *,
    since: datetime,
    max_contributors: int = WEEKLY_MAX_CONTRIBUTORS,
) -> str:
    start = since.date().isoformat()
    matched = [group for group in groups if group.get("discord_id") and is_discord_id(str(group["discord_id"]))]
    if not matched:
        return f"no Discord-matched community pull requests merged since {start}."
    lines = [
        f"thanks to the community for what landed in litellm since {start}:",
        "",
    ]
    shown = matched[:max_contributors]
    for group in shown:
        lines.append(_compact_contributor_line(group))
    extra = matched[max_contributors:]
    if extra:
        extra_prs = sum(len(group["pulls"]) for group in extra)
        noun = "contributor" if len(extra) == 1 else "contributors"
        pr_noun = "PR" if extra_prs == 1 else "PRs"
        lines.append(f"and {len(extra)} more {noun} ({extra_prs} {pr_noun}).")
    return "\n".join(lines).strip()


def _trim_chunk_for_footer(chunk: str, footer: str, limit: int) -> str:
    lines = chunk.split("\n")
    while lines and _utf16_len("\n".join(lines) + "\n" + footer) > limit:
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return footer[:limit]
    return "\n".join(lines) + "\n" + footer


def weekly_chunks(
    groups: list[dict],
    *,
    since: datetime,
    limit: int = WEEKLY_CHUNK_LIMIT,
    max_chunks: int = WEEKLY_PREVIEW_MAX_CHUNKS,
    max_contributors: int = WEEKLY_MAX_CONTRIBUTORS,
    note: str = "",
) -> list[str]:
    """Render compact one-line-per-Discord-contributor summary under 2000 chars."""
    start = since.date().isoformat()
    matched = [group for group in groups if group.get("discord_id") and is_discord_id(str(group["discord_id"]))]
    if not matched:
        empty = f"no Discord-matched community pull requests merged since {start}."
        if note:
            empty = f"{empty}\n\n{note}"
        return [empty[:limit]]

    shown = matched[:max_contributors]
    overflow_groups = matched[max_contributors:]
    header = f"thanks to the community for what landed in litellm since {start}:\n"

    chunks: list[str] = []
    current = header
    dropped_from_shown: list[dict] = []

    for index, group in enumerate(shown):
        line = _compact_contributor_line(group)
        if _utf16_len(line) > limit - 4:
            line = _compact_contributor_line(group, max_prs=2)[: limit - 4]
        separator = "\n\n" if current == header else "\n"
        candidate = f"{current.rstrip()}{separator}{line}"
        if _utf16_len(candidate) <= limit:
            current = candidate
            continue
        if len(chunks) + 1 >= max_chunks:
            dropped_from_shown = shown[index:]
            break
        if current.strip():
            chunks.append(current.rstrip())
        current = line

    if current.strip():
        chunks.append(current.rstrip())

    all_extra = dropped_from_shown + overflow_groups
    footers = []
    if all_extra:
        extra_prs = sum(len(group["pulls"]) for group in all_extra)
        noun = "contributor" if len(all_extra) == 1 else "contributors"
        pr_noun = "PR" if extra_prs == 1 else "PRs"
        footers.append(f"and {len(all_extra)} more {noun} ({extra_prs} {pr_noun}).")
    if note:
        footers.append(note)
    if footers:
        footer = "\n".join(footers)
        if _utf16_len(chunks[-1] + "\n" + footer) <= limit:
            chunks[-1] = chunks[-1] + "\n" + footer
        elif len(chunks) < max_chunks:
            chunks.append(footer[:limit])
        else:
            chunks[-1] = _trim_chunk_for_footer(chunks[-1], footer, limit)
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
    day = since.astimezone(UTC).date().isoformat()
    found = []
    seen_numbers: set[int] = set()
    rate_limited = False
    page = 1
    while page <= max_pages:
        params = urlencode({
            "q": f"repo:{repository} is:pr is:merged merged:>={day}",
            "sort": "updated",
            "order": "desc",
            "per_page": 100,
            "page": page,
        })
        try:
            payload = github_get_json(
                f"{GITHUB_API}/search/issues?{params}",
                token,
                pace=pace and page < max_pages,
            )
        except Exception as error:
            if _is_rate_limit_error(error):
                if found:
                    logging.warning("GitHub search rate limit hit on page %d; returning %d rows", page, len(found))
                    rate_limited = True
                    break
                raise WeeklyFetchError(str(error), rate_limited=True) from error
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


def build_weekly_report(
    config,
    store,
    now: datetime | None = None,
    *,
    guild=None,
    loop=None,
    pace: bool = False,
    max_chunks: int = WEEKLY_PREVIEW_MAX_CHUNKS,
    max_contributors: int = WEEKLY_MAX_CONTRIBUTORS,
    limit: int = WEEKLY_CHUNK_LIMIT,
    members=None,
    jev_transport=None,
    haiku_complete=None,
) -> WeeklyReport:
    moment = now or datetime.now(UTC)
    since = moment - timedelta(days=7)
    token = config.github_token or getattr(config, "github_actor_token", None)
    org = config.github_repository.split("/", 1)[0] if "/" in config.github_repository else ""
    org_members = fetch_org_public_members(org, token) if org else frozenset()
    rows, rate_limited = _search_merged_pages(config.github_repository, since, token, pace=pace)
    raw_groups = community_groups(
        rows,
        config.maintainer_logins,
        {},
        org_members=org_members,
        org=org,
    )
    matched, skipped, details = match_community_contributors(
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
    )
    logging.info(
        "Weekly merge summary: %d contributors matched in Discord, %d unmatched skipped",
        len(matched),
        skipped,
    )
    note = ""
    if rate_limited:
        note = "(stopped early because of GitHub search rate limits; set GITHUB_TOKEN for higher limits.)"
    chunks = weekly_chunks(
        matched,
        since=since,
        limit=limit,
        max_chunks=max_chunks,
        max_contributors=max_contributors,
        note=note,
    )
    return WeeklyReport(
        chunks=chunks,
        matched_count=len(matched),
        skipped_count=skipped,
        since=since,
        rate_limited=rate_limited,
        match_details=tuple(details),
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
    max_chunks: int = WEEKLY_PREVIEW_MAX_CHUNKS,
    max_contributors: int = WEEKLY_MAX_CONTRIBUTORS,
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
        max_contributors=max_contributors,
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
) -> str | None:
    """Post this week's summary once. Return the text when a new post is made."""
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
    channel = config.oss_contributors_channel_id or config.contributions_channel_id
    if not channel:
        logging.warning("Weekly merge summary skipped: no destination channel configured")
        return None
    report = build_weekly_report(
        config,
        store,
        moment,
        guild=guild,
        loop=loop,
        pace=True,
        max_chunks=WEEKLY_POST_MAX_CHUNKS,
    )
    sender = post or post_message
    for chunk in report.chunks:
        users = chunk_mention_user_ids(chunk)
        sender(channel, config.discord_token, {
            "content": chunk,
            "flags": SUPPRESS_EMBEDS_FLAG,
            "allowed_mentions": {"parse": [], "users": users},
        })
    if not store.mark_merged_week(config.github_repository, key, iso(moment)):
        logging.info("Weekly merge summary for %s was already recorded", key)
    return "\n\n".join(report.chunks)
