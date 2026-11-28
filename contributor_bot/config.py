from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .roles import DEFAULT_ROLE_NAMES, ROLE_ID_ENV, ROLE_KEYS, ROLE_NAME_ENV

# Same accounts as reviews.DEFAULT_REVIEW_BOTS. Kept here so config does not
# import the review modules (those import the poller, which imports config).
_DEFAULT_REVIEW_BOTS = ("cursor[bot]", "greptile-apps[bot]", "veria-ai[bot]")


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
    contributor_roles_enabled: bool = False
    discord_guild_id: str | None = None
    discord_application_id: str | None = None
    discord_public_key: str | None = field(default=None, repr=False)
    github_oauth_client_id: str | None = None
    github_oauth_client_secret: str | None = field(default=None, repr=False)
    github_webhook_secret: str | None = field(default=None, repr=False)
    github_oauth_redirect_uri: str | None = None
    role_announcements_channel_id: str | None = None
    discord_invite_url: str | None = None
    pr_comment_invites_enabled: bool = False
    merge_shoutouts_enabled: bool = True
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    roles_db_path: str = ".data/contributor_roles.db"
    role_ids: dict[str, str] = field(default_factory=dict)
    role_names: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROLE_NAMES))
    backfill_since: str = "2023-01-01T00:00:00+00:00"
    pr_review_enabled: bool = False
    review_channel_ids: tuple[str, ...] = ()
    review_model: str = ""
    review_bot_logins: tuple[str, ...] = ()
    review_maintainer_role_id: str | None = None
    review_cooldown_seconds: int = 600
    review_webhook_recheck: bool = False
    review_require_ask: bool = False
    review_classify: bool = False
    review_watch_all_channels: bool = False
    review_max_steps: int = 0
    review_usage_cap: int = 0
    review_timeout_seconds: int = 600
    review_daily_run_budget: int = 0
    review_followup_cooldown_seconds: int = 45
    review_classifier_model: str = ""
    review_agent_tracing: bool | None = None
    ack_reactions: tuple[str, ...] | None = None
    github_actor_token: str | None = field(default=None, repr=False)
    review_trigger_comments: tuple[tuple[str, str], ...] = ()
    review_post_summary: bool = False
    review_dry_run: bool = False
    review_trigger_poll_seconds: int = 120
    review_trigger_wait_seconds: int = 1200
    review_trigger_hourly_limit: int = 4
    merged_weekly_enabled: bool = True
    maintainer_logins: tuple[str, ...] = ()
    merged_weekday: int = 4
    merged_time: str = "10:00"
    merged_timezone: str = "America/Toronto"
    answers_enabled: bool = True
    typesafe_api_key: str | None = field(default=None, repr=False)
    review_decision_model: str = "jev-latest"
    decision_reply_threshold: float = 0.7
    fast_model: str = "anthropic/claude-haiku-4-5"
    discord_members_intent: bool = False
    bot_status_text: str = "skedaddling 🏃‍♂️"
    bot_status_rotate: bool = False
    pr_board_project_id: str | None = None
    pr_board_owner: str | None = None
    pr_board_number: int | None = None
    pr_board_sync_seconds: int = 600
    pr_board_status_field: str = "Status"
    roles_daily_time: str = "10:00"
    roles_backfill_on_boot: bool = False

    @property
    def github_oauth_enabled(self) -> bool:
        return bool(
            self.github_oauth_client_id
            and self.github_oauth_client_secret
            and self.github_oauth_redirect_uri
        )

    @property
    def pr_board_enabled(self) -> bool:
        return bool(self.pr_board_project_id) or (
            bool(self.pr_board_owner) and self.pr_board_number is not None
        )

    @property
    def effective_github_token(self) -> str | None:
        return self.github_token or self.github_actor_token

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

    roles_enabled = _flag(values, "CONTRIBUTOR_ROLES_ENABLED", False)
    review_enabled = _flag(values, "PR_REVIEW_ENABLED", False)
    members_intent = _flag(values, "DISCORD_MEMBERS_INTENT", roles_enabled)
    bind_host, bind_port = _http_bind(values)
    role_settings = _role_settings(values) if roles_enabled else None
    review_settings = _review_settings(values) if review_enabled else None
    discord_guild_id = (
        (role_settings or {}).get("discord_guild_id")
        or (review_settings or {}).get("discord_guild_id")
        or _optional_discord_id(values, "DISCORD_GUILD_ID")
    )
    discord_application_id = (
        (review_settings or {}).get("discord_application_id")
        or (role_settings or {}).get("discord_application_id")
        or _optional_discord_id(values, "DISCORD_APPLICATION_ID")
    )
    discord_public_key = (
        (review_settings or {}).get("discord_public_key")
        or (role_settings or {}).get("discord_public_key")
        or (values.get("DISCORD_PUBLIC_KEY", "").strip() or None)
    )
    webhook_secret = (
        (review_settings or {}).get("github_webhook_secret")
        or (role_settings or {}).get("github_webhook_secret")
    )

    actor_token = values.get("GITHUB_ACTOR_TOKEN", "").strip() or None
    github_token = values.get("GITHUB_TOKEN", "").strip() or actor_token
    board_project_id, board_owner, board_number = _pr_board_target(values)
    board_status_field = values.get("PR_BOARD_STATUS_FIELD", "").strip() or "Status"
    if len(board_status_field) > 80 or any(ch in board_status_field for ch in "\r\n"):
        raise ValueError("PR_BOARD_STATUS_FIELD must be a single-line field name")

    return Config(
        discord_token=token,
        contributions_channel_id=source,
        announcements_channel_id=destination,
        github_repository=repository,
        github_token=github_token,
        poll_interval_seconds=interval,
        state_file=values.get("STATE_FILE", "").strip() or ".data/contributors.json",
        github_discord_map={login.lower(): user_id for login, user_id in mapping.items()},
        announcements_enabled=enabled == "true",
        roundup_timezone=timezone,
        roundup_time=roundup_time,
        contributor_roles_enabled=roles_enabled,
        discord_guild_id=discord_guild_id,
        discord_application_id=discord_application_id,
        discord_public_key=discord_public_key,
        github_oauth_client_id=None if role_settings is None else role_settings["github_oauth_client_id"],
        github_oauth_client_secret=None if role_settings is None else role_settings["github_oauth_client_secret"],
        github_webhook_secret=webhook_secret,
        github_oauth_redirect_uri=None if role_settings is None else role_settings["github_oauth_redirect_uri"],
        role_announcements_channel_id=None if role_settings is None else role_settings["role_announcements_channel_id"],
        discord_invite_url=None if role_settings is None else role_settings["discord_invite_url"],
        pr_comment_invites_enabled=False if role_settings is None else role_settings["pr_comment_invites_enabled"],
        merge_shoutouts_enabled=True if role_settings is None else role_settings["merge_shoutouts_enabled"],
        http_host=bind_host,
        http_port=bind_port,
        roles_db_path=values.get("ROLES_DB_PATH", "").strip() or ".data/contributor_roles.db",
        role_ids={} if role_settings is None else role_settings["role_ids"],
        role_names=dict(DEFAULT_ROLE_NAMES) if role_settings is None else role_settings["role_names"],
        backfill_since="2023-01-01T00:00:00+00:00" if role_settings is None else role_settings["backfill_since"],
        pr_review_enabled=review_enabled,
        review_channel_ids=() if review_settings is None else review_settings["review_channel_ids"],
        review_model="" if review_settings is None else review_settings["review_model"],
        review_bot_logins=() if review_settings is None else review_settings["review_bot_logins"],
        review_maintainer_role_id=None if review_settings is None else review_settings["review_maintainer_role_id"],
        review_cooldown_seconds=600 if review_settings is None else review_settings["review_cooldown_seconds"],
        review_webhook_recheck=False if review_settings is None else review_settings["review_webhook_recheck"],
        review_require_ask=False if review_settings is None else review_settings["review_require_ask"],
        review_classify=False if review_settings is None else review_settings["review_classify"],
        review_watch_all_channels=False if review_settings is None else review_settings["review_watch_all_channels"],
        review_max_steps=0 if review_settings is None else review_settings["review_max_steps"],
        review_usage_cap=0 if review_settings is None else review_settings["review_usage_cap"],
        review_timeout_seconds=600 if review_settings is None else review_settings["review_timeout_seconds"],
        review_daily_run_budget=0 if review_settings is None else review_settings["review_daily_run_budget"],
        review_followup_cooldown_seconds=45 if review_settings is None else review_settings["review_followup_cooldown_seconds"],
        review_classifier_model="" if review_settings is None else review_settings["review_classifier_model"],
        review_agent_tracing=None if review_settings is None else review_settings["review_agent_tracing"],
        ack_reactions=None if review_settings is None else review_settings["ack_reactions"],
        github_actor_token=values.get("GITHUB_ACTOR_TOKEN", "").strip() or None,
        review_trigger_comments=_trigger_comments(values),
        review_post_summary=_flag(values, "REVIEW_POST_SUMMARY", False),
        review_dry_run=_flag(values, "REVIEW_DRY_RUN", False),
        review_trigger_poll_seconds=_bounded_int(values, "REVIEW_TRIGGER_POLL_SECONDS", 120, 30, 600),
        review_trigger_wait_seconds=_bounded_int(values, "REVIEW_TRIGGER_WAIT_SECONDS", 1200, 0, 3600),
        review_trigger_hourly_limit=_bounded_int(values, "REVIEW_TRIGGER_HOURLY_LIMIT", 4, 1, 20),
        merged_weekly_enabled=_flag(values, "MERGED_WEEKLY_ENABLED", True),
        maintainer_logins=_logins(values, "MAINTAINER_GITHUB_LOGINS"),
        merged_weekday=_weekday(values),
        merged_time=_clock(values, "MERGED_POST_TIME", "10:00"),
        merged_timezone=_timezone(values),
        answers_enabled=_flag(values, "ANSWERS_ENABLED", True),
        typesafe_api_key=values.get("TYPESAFE_API_KEY", "").strip() or None,
        review_decision_model=values.get("REVIEW_DECISION_MODEL", "").strip() or "jev-latest",
        decision_reply_threshold=_threshold(values),
        fast_model=values.get("REVIEW_FAST_MODEL", "").strip() or "anthropic/claude-haiku-4-5",
        discord_members_intent=members_intent,
        bot_status_text=(values.get("BOT_STATUS_TEXT", "").strip() or "skedaddling 🏃‍♂️")[:128],
        bot_status_rotate=_flag(values, "BOT_STATUS_ROTATE", False),
        pr_board_project_id=board_project_id,
        pr_board_owner=board_owner,
        pr_board_number=board_number,
        pr_board_sync_seconds=_bounded_int(values, "PR_BOARD_SYNC_SECONDS", 600, 60, 86400),
        pr_board_status_field=board_status_field,
        roles_daily_time="10:00" if role_settings is None else role_settings["roles_daily_time"],
        roles_backfill_on_boot=False if role_settings is None else role_settings["roles_backfill_on_boot"],
    )


def _pr_board_target(values: dict[str, str]) -> tuple[str | None, str | None, int | None]:
    project_id = values.get("PR_BOARD_PROJECT_ID", "").strip() or None
    owner = values.get("PR_BOARD_OWNER", "").strip() or None
    number_raw = values.get("PR_BOARD_NUMBER", "").strip()
    number: int | None = None
    if project_id is not None and not re.fullmatch(r"[A-Za-z0-9_=-]{4,120}", project_id):
        raise ValueError("PR_BOARD_PROJECT_ID must be a GitHub ProjectV2 node ID (e.g. PVT_...)")
    if owner is not None and not re.fullmatch(r"[A-Za-z0-9-]+", owner):
        raise ValueError("PR_BOARD_OWNER must be a GitHub organization or user login")
    if number_raw:
        if not re.fullmatch(r"[1-9][0-9]*", number_raw):
            raise ValueError("PR_BOARD_NUMBER must be a positive integer")
        number = int(number_raw)
        if number > 100000:
            raise ValueError("PR_BOARD_NUMBER must be between 1 and 100000")
    if project_id is None and bool(owner) != (number is not None):
        raise ValueError("PR_BOARD_OWNER and PR_BOARD_NUMBER must both be set (or set PR_BOARD_PROJECT_ID)")
    return project_id, owner, number


def _optional_cap(values: dict[str, str], name: str) -> int:
    """Unset, empty, or 0 means no cap. A positive integer is a limit."""
    raw = values.get(name)
    if raw is None or not str(raw).strip() or str(raw).strip() == "0":
        return 0
    text = str(raw).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(text)


def _optional_timeout(values: dict[str, str], name: str, default: int) -> int:
    """Unset or empty uses ``default``. 0 means no wall-time cap."""
    raw = values.get(name)
    if raw is None or not str(raw).strip():
        return default
    text = str(raw).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"{name} must be a non-negative integer")
    seconds = int(text)
    if seconds > 7200:
        raise ValueError(f"{name} must be between 0 and 7200")
    return seconds


def _bounded_int(values: dict[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = values.get(name, "").strip() or str(default)
    if not re.fullmatch(r"[0-9]+", raw):
        raise ValueError(f"{name} must be an integer")
    number = int(raw)
    if not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return number


def _optional_discord_id(values: dict[str, str], name: str) -> str | None:
    raw = values.get(name, "").strip()
    if not raw:
        return None
    if not is_discord_id(raw):
        raise ValueError(f"{name} must be a Discord ID")
    return raw


def _logins(values: dict[str, str], name: str) -> tuple[str, ...]:
    raw = values.get(name, "").strip()
    if not raw:
        return ()
    logins = tuple(part.strip() for part in raw.split(",") if part.strip())
    if any(not re.fullmatch(r"[A-Za-z0-9-]+(?:\[bot\])?", login) for login in logins):
        raise ValueError(f"{name} must be comma-separated GitHub logins")
    return logins


def _weekday(values: dict[str, str]) -> int:
    from .weekly import parse_weekday
    return parse_weekday(values.get("MERGED_POST_WEEKDAY", "friday"))


def _clock(values: dict[str, str], name: str, default: str) -> str:
    text = values.get(name, "").strip() or default
    if not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", text):
        raise ValueError(f"{name} must be HH:MM")
    return text


def _timezone(values: dict[str, str]) -> str:
    name = values.get("MERGED_POST_TIMEZONE", "").strip() or values.get("ROUNDUP_TIMEZONE", "").strip() or "America/Toronto"
    try:
        ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise ValueError("MERGED_POST_TIMEZONE must be an installed IANA timezone") from error
    return name


def _threshold(values: dict[str, str]) -> float:
    raw = values.get("DECISION_REPLY_THRESHOLD", "").strip() or "0.7"
    try:
        number = float(raw)
    except ValueError as error:
        raise ValueError("DECISION_REPLY_THRESHOLD must be between 0 and 1") from error
    if not 0 <= number <= 1:
        raise ValueError("DECISION_REPLY_THRESHOLD must be between 0 and 1")
    return number


def _trigger_comments(values: dict[str, str]):
    from .triggers import parse_trigger_comments
    return parse_trigger_comments(values.get("REVIEW_TRIGGER_COMMENTS"))


def _ack_reactions(values: dict[str, str]) -> tuple[str, ...] | None:
    from .ack import parse_ack_reactions
    return parse_ack_reactions(values.get("ACK_REACTIONS"))


def _optional_flag(values: dict[str, str], name: str) -> bool | None:
    """Unset or empty means the caller picks. true and false are explicit."""
    raw = values.get(name)
    if raw is None or not str(raw).strip():
        return None
    value = str(raw).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def _flag(values: dict[str, str], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def _https_url(name: str, value: str, *, localhost_http: bool = False) -> str:
    parts = urlsplit(value)
    local = localhost_http and parts.scheme == "http" and parts.hostname in {"localhost", "127.0.0.1"}
    remote = parts.scheme == "https" and bool(parts.hostname)
    if (not local and not remote) or parts.username or parts.password or not parts.netloc:
        detail = "an https URL" if not localhost_http else "an https URL (http://localhost is allowed)"
        raise ValueError(f"{name} must be {detail}")
    return value


def _secret(name: str, values: dict[str, str], *, minimum: int = 8) -> str:
    value = values.get(name, "").strip()
    if len(value) < minimum or any(char.isspace() for char in value):
        raise ValueError(f"{name} must be at least {minimum} characters with no whitespace")
    return value


def _iso_timestamp(name: str, value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def _discord_application(values: dict[str, str]) -> dict[str, str]:
    guild_id = values.get("DISCORD_GUILD_ID", "").strip()
    application_id = values.get("DISCORD_APPLICATION_ID", "").strip()
    if not is_discord_id(guild_id):
        raise ValueError("DISCORD_GUILD_ID must be a Discord guild ID")
    if not is_discord_id(application_id):
        raise ValueError("DISCORD_APPLICATION_ID must be a Discord application ID")
    public_key = values.get("DISCORD_PUBLIC_KEY", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", public_key):
        raise ValueError("DISCORD_PUBLIC_KEY must be a 64-character hex Ed25519 public key")
    return {
        "discord_guild_id": guild_id,
        "discord_application_id": application_id,
        "discord_public_key": public_key,
    }


def _http_bind(values: dict[str, str]) -> tuple[str, int]:
    port_text = values.get("HTTP_PORT", "").strip() or values.get("PORT", "").strip() or "8080"
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError("HTTP_PORT must be a TCP port") from error
    if not 1 <= port <= 65535:
        raise ValueError("HTTP_PORT must be a TCP port")
    host = values.get("HTTP_HOST", "").strip() or "0.0.0.0"
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,253}", host):
        raise ValueError("HTTP_HOST is not a valid bind address")
    return host, port


def _review_bots(values: dict[str, str]) -> tuple[str, ...]:
    raw = values.get("REVIEW_BOT_LOGINS", "").strip() or ",".join(_DEFAULT_REVIEW_BOTS)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts or any(not re.fullmatch(r"[A-Za-z0-9-]+(?:\[bot\])?", part) for part in parts):
        raise ValueError("REVIEW_BOT_LOGINS must be comma-separated GitHub logins")
    ordered: list[str] = []
    for part in parts:
        login = part.lower()
        if login not in ordered:
            ordered.append(login)
    return tuple(ordered)


def _review_settings(values: dict[str, str]) -> dict:
    app = _discord_application(values)
    host, port = _http_bind(values)
    raw_channels = values.get("REVIEW_CHANNEL_IDS", "").strip()
    if not raw_channels:
        raise ValueError("REVIEW_CHANNEL_IDS must list at least one Discord channel ID")
    channels = [part.strip() for part in raw_channels.split(",") if part.strip()]
    if not channels or any(not is_discord_id(channel) for channel in channels):
        raise ValueError("REVIEW_CHANNEL_IDS must be comma-separated Discord channel IDs")
    if len(set(channels)) != len(channels):
        raise ValueError("REVIEW_CHANNEL_IDS must be distinct")
    model = values.get("REVIEW_MODEL", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_./:+-]{1,200}", model):
        raise ValueError("REVIEW_MODEL must be a LiteLLM model name, like openai/gpt-4o-mini")
    role_id = values.get("REVIEW_MAINTAINER_ROLE_ID", "").strip()
    if role_id and not is_discord_id(role_id):
        raise ValueError("REVIEW_MAINTAINER_ROLE_ID must be a Discord role ID")
    cooldown_text = values.get("REVIEW_COOLDOWN_SECONDS", "").strip() or "600"
    if not re.fullmatch(r"[0-9]+", cooldown_text):
        raise ValueError("REVIEW_COOLDOWN_SECONDS must be a positive number of seconds")
    cooldown = int(cooldown_text)
    if not 30 <= cooldown <= 86400:
        raise ValueError("REVIEW_COOLDOWN_SECONDS must be between 30 and 86400")
    webhook_recheck = _flag(values, "REVIEW_WEBHOOK_RECHECK", False)
    classifier = values.get("REVIEW_CLASSIFIER_MODEL", "").strip() or "anthropic/claude-haiku-4-5"
    if not re.fullmatch(r"[A-Za-z0-9_./:+-]{1,200}", classifier):
        raise ValueError("REVIEW_CLASSIFIER_MODEL must be a LiteLLM model name")
    settings = {
        **app,
        "http_host": host,
        "http_port": port,
        "review_channel_ids": tuple(channels),
        "review_model": model,
        "review_bot_logins": _review_bots(values),
        "review_maintainer_role_id": role_id or None,
        "review_cooldown_seconds": cooldown,
        "review_webhook_recheck": webhook_recheck,
        "review_require_ask": _flag(values, "REVIEW_REQUIRE_ASK", False),
        "review_classify": _flag(values, "REVIEW_CLASSIFY_MESSAGES", True),
        "review_watch_all_channels": _flag(values, "REVIEW_WATCH_ALL_CHANNELS", False),
        "review_max_steps": _optional_cap(values, "REVIEW_AGENT_MAX_STEPS"),
        "review_usage_cap": _optional_cap(values, "REVIEW_AGENT_MAX_TOKENS"),
        "review_timeout_seconds": _optional_timeout(values, "REVIEW_AGENT_TIMEOUT_SECONDS", 600),
        "review_daily_run_budget": _optional_cap(values, "REVIEW_DAILY_RUN_BUDGET"),
        "review_followup_cooldown_seconds": _bounded_int(values, "REVIEW_FOLLOWUP_COOLDOWN_SECONDS", 45, 10, 3600),
        "review_classifier_model": classifier,
        "review_agent_tracing": _optional_flag(values, "REVIEW_AGENT_TRACING"),
        "ack_reactions": _ack_reactions(values),
    }
    if webhook_recheck:
        settings["github_webhook_secret"] = _secret("GITHUB_WEBHOOK_SECRET", values, minimum=8)
    return settings


def _role_settings(values: dict[str, str]) -> dict:
    guild_id = values.get("DISCORD_GUILD_ID", "").strip()
    if not is_discord_id(guild_id):
        raise ValueError("DISCORD_GUILD_ID must be a Discord guild ID")
    application_id = _optional_discord_id(values, "DISCORD_APPLICATION_ID")
    raw_public_key = values.get("DISCORD_PUBLIC_KEY", "").strip().lower()
    if raw_public_key and not re.fullmatch(r"[0-9a-f]{64}", raw_public_key):
        raise ValueError("DISCORD_PUBLIC_KEY must be a 64-character hex Ed25519 public key")
    public_key = raw_public_key or None

    raw_oauth = (
        values.get("GITHUB_OAUTH_CLIENT_ID", "").strip(),
        values.get("GITHUB_OAUTH_CLIENT_SECRET", "").strip(),
        values.get("GITHUB_OAUTH_REDIRECT_URI", "").strip(),
    )
    if any(raw_oauth):
        oauth_client_id = _secret("GITHUB_OAUTH_CLIENT_ID", values, minimum=8)
        oauth_client_secret = _secret("GITHUB_OAUTH_CLIENT_SECRET", values, minimum=8)
        oauth_redirect_uri = _https_url(
            "GITHUB_OAUTH_REDIRECT_URI",
            raw_oauth[2],
            localhost_http=True,
        )
    else:
        oauth_client_id = None
        oauth_client_secret = None
        oauth_redirect_uri = None

    raw_webhook = values.get("GITHUB_WEBHOOK_SECRET", "").strip()
    webhook_secret = _secret("GITHUB_WEBHOOK_SECRET", values, minimum=8) if raw_webhook else None

    host, port = _http_bind(values)
    channel = values.get("ROLE_ANNOUNCEMENTS_CHANNEL_ID", "").strip()
    if channel and not is_discord_id(channel):
        raise ValueError("ROLE_ANNOUNCEMENTS_CHANNEL_ID must be a Discord channel ID")
    invites = _flag(values, "PR_COMMENT_INVITES_ENABLED", False)
    invite_url = values.get("DISCORD_INVITE_URL", "").strip()
    if invites:
        if not invite_url:
            raise ValueError("DISCORD_INVITE_URL is required when PR_COMMENT_INVITES_ENABLED=true")
        invite_url = _https_url("DISCORD_INVITE_URL", invite_url)
    elif invite_url:
        invite_url = _https_url("DISCORD_INVITE_URL", invite_url)
    role_ids = {}
    role_names = dict(DEFAULT_ROLE_NAMES)
    for key in ROLE_KEYS:
        role_id = values.get(ROLE_ID_ENV[key], "").strip()
        if not is_discord_id(role_id):
            raise ValueError(f"{ROLE_ID_ENV[key]} must be a Discord role ID")
        role_ids[key] = role_id
        custom = values.get(ROLE_NAME_ENV[key], "").strip()
        if custom:
            if len(custom) > 100 or any(char in custom for char in "\r\n@"):
                raise ValueError(f"{ROLE_NAME_ENV[key]} must be a single-line role name")
            role_names[key] = custom
    if len(set(role_ids.values())) != len(role_ids):
        raise ValueError("Contributor role IDs must be distinct")
    db_path = values.get("ROLES_DB_PATH", "").strip() or ".data/contributor_roles.db"
    if db_path == ":memory:" or "\x00" in db_path:
        raise ValueError("ROLES_DB_PATH must be a file path")
    return {
        "discord_guild_id": guild_id,
        "discord_application_id": application_id,
        "discord_public_key": public_key,
        "github_oauth_client_id": oauth_client_id,
        "github_oauth_client_secret": oauth_client_secret,
        "github_webhook_secret": webhook_secret,
        "github_oauth_redirect_uri": oauth_redirect_uri,
        "role_announcements_channel_id": channel or None,
        "discord_invite_url": invite_url or None,
        "pr_comment_invites_enabled": invites,
        "merge_shoutouts_enabled": _flag(values, "MERGE_SHOUTOUTS_ENABLED", True),
        "http_host": host,
        "http_port": port,
        "role_ids": role_ids,
        "role_names": role_names,
        "backfill_since": _iso_timestamp(
            "BACKFILL_SINCE",
            values.get("BACKFILL_SINCE", "").strip() or "2023-01-01T00:00:00+00:00",
        ),
        "roles_daily_time": _clock(values, "ROLES_DAILY_TIME", "10:00"),
        "roles_backfill_on_boot": _flag(values, "ROLES_BACKFILL_ON_BOOT", False),
    }
