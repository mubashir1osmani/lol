"""Calendar-based Friday boundaries, including daylight saving transitions."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .config import Config

UTC = timezone.utc


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return result.astimezone(UTC)


def iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return value.astimezone(UTC).isoformat(timespec="seconds")


def first_friday(config: Config, now: datetime) -> datetime:
    """Next Friday; on Friday allow a late startup to deliver today's roundup."""
    local = now.astimezone(ZoneInfo(config.roundup_timezone))
    hour, minute = map(int, config.roundup_time.split(":"))
    day = local.date() + timedelta(days=(4 - local.weekday()) % 7)
    return datetime(day.year, day.month, day.day, hour, minute,
                    tzinfo=local.tzinfo).astimezone(UTC)


def shift_week(config: Config, boundary: datetime, weeks: int = 1) -> datetime:
    # Local calendar arithmetic, not 168 UTC hours: DST weeks can be 167/169h.
    return (boundary.astimezone(ZoneInfo(config.roundup_timezone))
            + timedelta(weeks=weeks)).astimezone(UTC)


def new_window(config: Config, end: datetime) -> dict:
    return {"start": iso(shift_week(config, end, -1)), "end": iso(end),
            "timezone": config.roundup_timezone, "time": config.roundup_time,
            "merge_numbers": None, "prepared": False}
