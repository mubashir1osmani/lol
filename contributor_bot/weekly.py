"""Monday summary of community pull requests merged in the last seven days."""

import logging
from datetime import datetime, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from .api import GITHUB_API, TransientAPIError, github_get_json, post_message
from .reviews import discord_safe
from .schedule import UTC, iso

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

WEEKLY_CHUNK_LIMIT = 1900
WEEKLY_PREVIEW_MAX_CHUNKS = 5
WEEKLY_POST_MAX_CHUNKS = 10


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


def community_groups(items: list[dict], maintainers: tuple[str, ...], links: dict[int, str]) -> list[dict]:
    """Group merged pull requests by a human who is not on the maintainer list."""
    skip = {login.lower() for login in maintainers}
    grouped: dict[str, dict] = {}
    for item in items:
        login = item.get("login") or ""
        if not login or login.lower() in skip or _bot_login(login, item.get("type") or ""):
            continue
        bucket = grouped.setdefault(login, {
            "login": login,
            "discord_id": links.get(item.get("user_id")),
            "pulls": [],
        })
        bucket["pulls"].append({
            "number": item["number"],
            "title": item.get("title") or f"#{item['number']}",
            "url": item.get("url") or "",
        })
    return sorted(grouped.values(), key=lambda group: group["login"].lower())


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _group_lines(group: dict, *, max_line: int) -> list[str]:
    who = f"**{discord_safe(group['login'])}**"
    if group.get("discord_id"):
        who += f" <@{group['discord_id']}>"
    lines = [who]
    for pull in group["pulls"]:
        url = pull.get("url") or ""
        title = discord_safe(pull.get("title") or f"#{pull['number']}")
        line = f"- [{title}]({url})" if url else f"- {title}"
        if _utf16_len(line) > max_line:
            budget = max(20, max_line - _utf16_len(f"- []({url})") - 1)
            trimmed = title[:budget].rstrip()
            line = f"- [{trimmed}…]({url})" if url else f"- {trimmed}…"
        lines.append(line)
    return lines


def render_weekly(groups: list[dict], *, since: datetime) -> str:
    start = since.date().isoformat()
    if not groups:
        return f"no community pull requests merged since {start}."
    lines = [
        f"thanks for what landed in litellm since {start}.",
        "",
    ]
    for group in groups:
        lines.extend(_group_lines(group, max_line=WEEKLY_CHUNK_LIMIT))
        lines.append("")
    return "\n".join(lines).strip()


def _trim_chunk_for_footer(chunk: str, footer: str, limit: int) -> str:
    lines = chunk.split("\n")
    while lines and _utf16_len("\n".join(lines) + "\n\n" + footer) > limit:
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return footer[:limit]
    return "\n".join(lines) + "\n\n" + footer


def weekly_chunks(
    groups: list[dict],
    *,
    since: datetime,
    limit: int = WEEKLY_CHUNK_LIMIT,
    max_chunks: int = WEEKLY_PREVIEW_MAX_CHUNKS,
    note: str = "",
    show_maintainer_hint: bool = False,
) -> list[str]:
    """Split the weekly summary cleanly on line boundaries under Discord's 2000-char limit."""
    start = since.date().isoformat()
    if not groups:
        empty = f"no community pull requests merged since {start}."
        if note:
            empty = f"{empty}\n\n{note}"
        return [empty[:limit]]

    chunks: list[str] = []
    current = f"thanks for what landed in litellm since {start}.\n"
    remaining_groups = 0
    remaining_prs = 0

    for index, group in enumerate(groups):
        block = "\n".join(_group_lines(group, max_line=limit - 4))
        separator = "\n\n" if current.strip() else ""
        candidate = f"{current.rstrip()}{separator}{block}"
        if _utf16_len(candidate) <= limit:
            current = candidate
            continue
        # Block does not fit as a whole in current chunk. Try starting a new chunk.
        if current.strip():
            if len(chunks) + 1 >= max_chunks:
                remaining_groups = len(groups) - index
                remaining_prs = sum(len(item["pulls"]) for item in groups[index:])
                break
            chunks.append(current.rstrip())
            current = ""
        # Place the group into the fresh chunk, splitting by line if one contributor has many PRs.
        for line_index, line in enumerate(_group_lines(group, max_line=limit - 4)):
            piece = f"{current}\n{line}" if current else line
            if _utf16_len(piece) <= limit:
                current = piece
                continue
            if len(chunks) + 1 >= max_chunks:
                remaining_groups = len(groups) - index
                remaining_prs = len(group["pulls"]) - max(0, line_index - 1) + sum(
                    len(item["pulls"]) for item in groups[index + 1:]
                )
                break
            if current.strip():
                chunks.append(current.rstrip())
            current = line
        if remaining_groups:
            break

    if current.strip():
        chunks.append(current.rstrip())

    footers = []
    if remaining_groups:
        hint = " set MAINTAINER_GITHUB_LOGINS to exclude maintainers." if show_maintainer_hint else ""
        footers.append(
            f"…and {remaining_groups} more contributors ({remaining_prs} more PRs) not shown.{hint}"
        )
    if note:
        footers.append(note)
    if footers:
        footer = " ".join(footers)
        if _utf16_len(chunks[-1] + "\n\n" + footer) <= limit:
            chunks[-1] = chunks[-1] + "\n\n" + footer
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


def _weekly_groups(
    config,
    store,
    now: datetime | None = None,
    *,
    pace: bool = False,
) -> tuple[list[dict], datetime, bool]:
    moment = now or datetime.now(UTC)
    since = moment - timedelta(days=7)
    token = config.github_token or getattr(config, "github_actor_token", None)
    rows, rate_limited = _search_merged_pages(config.github_repository, since, token, pace=pace)
    links = {}
    if store is not None:
        for row in rows:
            user_id = row.get("user_id")
            if not isinstance(user_id, int):
                continue
            link = store.link_for_github_user(user_id)
            if link:
                links[user_id] = link["discord_id"]
    groups = community_groups(rows, config.maintainer_logins, links)
    return groups, since, rate_limited


def build_weekly(config, store, now: datetime | None = None) -> str:
    groups, since, _rate_limited = _weekly_groups(config, store, now, pace=False)
    return render_weekly(groups, since=since)


def build_weekly_chunks(
    config,
    store,
    now: datetime | None = None,
    *,
    pace: bool = False,
    max_chunks: int = WEEKLY_PREVIEW_MAX_CHUNKS,
    limit: int = WEEKLY_CHUNK_LIMIT,
) -> list[str]:
    groups, since, rate_limited = _weekly_groups(config, store, now, pace=pace)
    note = ""
    if rate_limited:
        note = "(stopped early because of GitHub search rate limits; set GITHUB_TOKEN for higher limits.)"
    return weekly_chunks(
        groups,
        since=since,
        limit=limit,
        max_chunks=max_chunks,
        note=note,
        show_maintainer_hint=not bool(config.maintainer_logins),
    )


def post_weekly_if_due(config, store, now: datetime | None = None, *, post=None) -> str | None:
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
    chunks = build_weekly_chunks(config, store, moment, pace=True, max_chunks=WEEKLY_POST_MAX_CHUNKS)
    sender = post or post_message
    for chunk in chunks:
        sender(channel, config.discord_token, {"content": chunk, "allowed_mentions": {"parse": []}})
    if not store.mark_merged_week(config.github_repository, key, iso(moment)):
        logging.info("Weekly merge summary for %s was already recorded", key)
    return "\n\n".join(chunks)
