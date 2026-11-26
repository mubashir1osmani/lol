"""Monday summary of community pull requests merged in the last seven days."""

import logging
from datetime import datetime, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from .api import GITHUB_API, github_get_json, post_message
from .reviews import discord_safe, review_chunks
from .schedule import UTC, iso

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


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


def render_weekly(groups: list[dict], *, since: datetime) -> str:
    start = since.date().isoformat()
    if not groups:
        return f"no community pull requests merged since {start}."
    lines = [
        f"thanks for what landed in litellm since {start}.",
        "",
    ]
    for group in groups:
        who = f"**{discord_safe(group['login'])}**"
        if group.get("discord_id"):
            who += f" <@{group['discord_id']}>"
        lines.append(who)
        for pull in group["pulls"]:
            title = discord_safe(pull["title"])
            lines.append(f"- [{title}]({pull['url']})")
        lines.append("")
    return "\n".join(lines).strip()


def fetch_merged_community(repository: str, since: datetime, token: str | None) -> list[dict]:
    """Merged pull requests since ``since``. Search is discovery; each row is one issue."""
    day = since.astimezone(UTC).date().isoformat()
    found = []
    page = 1
    while page <= 5:
        params = urlencode({
            "q": f"repo:{repository} is:pr is:merged merged:>={day}",
            "per_page": 50,
            "page": page,
        })
        payload = github_get_json(f"{GITHUB_API}/search/issues?{params}", token, pace=True)
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            break
        for item in items:
            if not isinstance(item, dict):
                continue
            user = item.get("user") if isinstance(item.get("user"), dict) else {}
            number = item.get("number")
            if not isinstance(number, int):
                continue
            found.append({
                "number": number,
                "title": item.get("title") if isinstance(item.get("title"), str) else "",
                "url": item.get("html_url") if isinstance(item.get("html_url"), str)
                else f"https://github.com/{repository}/pull/{number}",
                "login": user.get("login") if isinstance(user.get("login"), str) else "",
                "type": user.get("type") if isinstance(user.get("type"), str) else "",
                "user_id": user.get("id") if isinstance(user.get("id"), int) else None,
            })
        if len(items) < 50:
            break
        page += 1
    return found


def build_weekly(config, store, now: datetime | None = None) -> str:
    moment = now or datetime.now(UTC)
    since = moment - timedelta(days=7)
    rows = fetch_merged_community(config.github_repository, since, config.github_token)
    links = {}
    for row in rows:
        user_id = row.get("user_id")
        if not isinstance(user_id, int):
            continue
        link = store.link_for_github_user(user_id)
        if link:
            links[user_id] = link["discord_id"]
    groups = community_groups(rows, config.maintainer_logins, links)
    return render_weekly(groups, since=since)


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
    text = build_weekly(config, store, moment)
    channel = config.oss_contributors_channel_id or config.contributions_channel_id
    sender = post or post_message
    chunks = review_chunks(text, "")
    for chunk in chunks:
        sender(channel, config.discord_token, {"content": chunk, "allowed_mentions": {"parse": []}})
    if not store.mark_merged_week(config.github_repository, key, iso(moment)):
        logging.info("Weekly merge summary for %s was already recorded", key)
    return text
