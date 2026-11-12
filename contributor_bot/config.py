from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def load_dotenv(path: str = ".env") -> None:
    """Load a conventional KEY=VALUE .env file without dependencies."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _required(name: str, env: dict[str, str]) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def is_discord_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[1-9][0-9]{0,19}", value) is not None
        and int(value) < 2**64
    )


def _channel_id(name: str, env: dict[str, str]) -> str:
    value = _required(name, env)
    if not is_discord_id(value):
        raise ValueError(f"{name} must be a Discord channel ID")
    return value


@dataclass(frozen=True)
class Config:
    discord_token: str = field(repr=False)
    contributions_channel_id: str
    announcements_channel_id: str
    github_repository: str = "BerriAI/litellm"
    github_token: str | None = field(default=None, repr=False)
    poll_interval_seconds: float = 300
    state_file: str = ".data/contributors.json"
    github_discord_map: dict[str, str] = field(default_factory=dict)
    announcements_enabled: bool = True
    roundup_timezone: str = "America/Toronto"
    roundup_time: str = "09:00"

    @property
    def scope(self) -> dict[str, str]:
        return {
            "repository": self.github_repository.lower(),
            "contributions_channel_id": self.contributions_channel_id,
            "announcements_channel_id": self.announcements_channel_id,
        }


def get_config(env: dict[str, str] | None = None) -> Config:
    values = dict(os.environ if env is None else env)
    token = _required("DISCORD_BOT_TOKEN", values)
    source = _channel_id("OSS_CONTRIBUTIONS_CHANNEL_ID", values)
    destination = _channel_id("ANNOUNCEMENTS_CHANNEL_ID", values)
    if source == destination:
        raise ValueError("The contributions and announcements channels must differ")

    repository = values.get("GITHUB_REPOSITORY", "BerriAI/litellm").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("GITHUB_REPOSITORY must look like owner/repository")
    try:
        interval = float(values.get("POLL_INTERVAL_MINUTES", "5")) * 60
    except ValueError as error:
        raise ValueError("POLL_INTERVAL_MINUTES must be a positive number") from error
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("POLL_INTERVAL_MINUTES must be a positive number")

    try:
        mapping = json.loads(values.get("GITHUB_DISCORD_MAP", "{}").strip() or "{}")
    except json.JSONDecodeError as error:
        raise ValueError("GITHUB_DISCORD_MAP must be a JSON object") from error
    if not isinstance(mapping, dict) or any(
        not re.fullmatch(r"[A-Za-z0-9-]+(?:\[bot\])?", login)
        or not is_discord_id(discord_id)
        for login, discord_id in mapping.items()
    ):
        raise ValueError("GITHUB_DISCORD_MAP must map GitHub usernames to Discord ID strings")
    timezone = values.get("ROUNDUP_TIMEZONE", "America/Toronto").strip()
    try:
        ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise ValueError("ROUNDUP_TIMEZONE must be an installed IANA timezone") from error
    roundup_time = values.get("ROUNDUP_TIME", "09:00").strip()
    if not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", roundup_time):
        raise ValueError("ROUNDUP_TIME must be HH:MM (24-hour time)")

    enabled = values.get("ANNOUNCEMENTS_ENABLED", "true").strip().lower()
    if enabled not in {"true", "false"}:
        raise ValueError("ANNOUNCEMENTS_ENABLED must be true or false")

    return Config(
        discord_token=token,
        contributions_channel_id=source,
        announcements_channel_id=destination,
        github_repository=repository,
        github_token=values.get("GITHUB_TOKEN", "").strip() or None,
        poll_interval_seconds=interval,
        state_file=values.get("STATE_FILE", "").strip() or ".data/contributors.json",
        github_discord_map={login.lower(): user_id for login, user_id in mapping.items()},
        announcements_enabled=enabled == "true",
        roundup_timezone=timezone,
        roundup_time=roundup_time,
    )
