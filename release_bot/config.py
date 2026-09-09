from dataclasses import dataclass
import os
from pathlib import Path
import re


DEFAULT_BLOG_FEED_URL = "https://docs.litellm.ai/blog/rss.xml"


def load_dotenv(path: str = ".env") -> None:
    """Load a small, conventional KEY=VALUE .env file without dependencies."""
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


def _boolean(name: str, fallback: bool, env: dict[str, str]) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return fallback
    value = raw.strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _positive_float(name: str, fallback: float, env: dict[str, str]) -> float:
    try:
        value = float(env.get(name, fallback))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive number") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _id_list(name: str, env: dict[str, str]) -> tuple[str, ...]:
    raw = env.get(name, "")
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    for value in values:
        if not value.isdigit():
            raise ValueError(f"{name} must be comma-separated Discord IDs")
    return values


def _url_list(name: str, env: dict[str, str]) -> tuple[str, ...]:
    raw = env.get(name, "")
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    for value in values:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"{name} must be comma-separated http(s) URLs")
    return values


@dataclass(frozen=True)
class Config:
    discord_token: str
    discord_channel_id: str
    github_repository: str = "BerriAI/litellm"
    github_token: str | None = None
    poll_interval_seconds: float = 300
    include_prereleases: bool = True
    post_latest_on_startup: bool = True
    state_file: str = ".data/releases.json"
    discord_guild_id: str | None = None
    support_user_id: str | None = None
    support_channel_ids: tuple[str, ...] = ()
    thread_poll_interval_seconds: float = 60
    updates_channel_id: str = ""
    blog_feed_url: str = DEFAULT_BLOG_FEED_URL
    social_feed_urls: tuple[str, ...] = ()
    feed_poll_interval_seconds: float = 1800
    anthropic_api_key: str | None = None
    llm_base_url: str | None = None
    claude_model: str = "claude-sonnet-4-6"
    agent_replies_in_threads: bool = True
    mention_role_id: str | None = None
    mention_everyone: bool = False

    @property
    def feeds(self) -> tuple[tuple[str, str], ...]:
        """(label, url) pairs for every configured feed."""
        pairs = []
        if self.blog_feed_url:
            pairs.append(("blog", self.blog_feed_url))
        pairs.extend(("social", url) for url in self.social_feed_urls)
        return tuple(pairs)


def get_config(env: dict[str, str] | None = None) -> Config:
    values = dict(os.environ if env is None else env)
    repository = values.get("GITHUB_REPOSITORY", "BerriAI/litellm").strip()
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        raise ValueError("GITHUB_REPOSITORY must look like owner/repository")

    support_user_id = values.get("SUPPORT_USER_ID", "").strip() or None
    if support_user_id and not support_user_id.isdigit():
        raise ValueError("SUPPORT_USER_ID must be a Discord user ID")

    mention_role_id = values.get("MENTION_ROLE_ID", "").strip() or None
    if mention_role_id and not mention_role_id.isdigit():
        raise ValueError("MENTION_ROLE_ID must be a Discord role ID")

    guild_id = values.get("DISCORD_GUILD_ID", "").strip() or None
    if guild_id and not guild_id.isdigit():
        raise ValueError("DISCORD_GUILD_ID must be a Discord server ID")
    if support_user_id and not guild_id:
        raise ValueError("SUPPORT_USER_ID requires DISCORD_GUILD_ID to be set")

    discord_token = _required("DISCORD_BOT_TOKEN", values)
    discord_channel_id = _required("DISCORD_CHANNEL_ID", values)

    return Config(
        discord_token=discord_token,
        discord_channel_id=discord_channel_id,
        github_repository=repository,
        github_token=values.get("GITHUB_TOKEN", "").strip() or None,
        poll_interval_seconds=_positive_float("POLL_INTERVAL_MINUTES", 5, values) * 60,
        include_prereleases=_boolean("INCLUDE_PRERELEASES", True, values),
        post_latest_on_startup=_boolean("POST_LATEST_ON_STARTUP", True, values),
        state_file=values.get("STATE_FILE", ".data/releases.json").strip()
        or ".data/releases.json",
        discord_guild_id=guild_id,
        support_user_id=support_user_id,
        support_channel_ids=_id_list("SUPPORT_CHANNEL_IDS", values),
        thread_poll_interval_seconds=_positive_float(
            "THREAD_POLL_INTERVAL_MINUTES", 1, values
        )
        * 60,
        updates_channel_id=values.get("UPDATES_CHANNEL_ID", "").strip()
        or discord_channel_id,
        blog_feed_url=values.get("BLOG_FEED_URL", DEFAULT_BLOG_FEED_URL).strip(),
        social_feed_urls=_url_list("SOCIAL_FEED_URLS", values),
        feed_poll_interval_seconds=_positive_float(
            "FEED_POLL_INTERVAL_MINUTES", 30, values
        )
        * 60,
        anthropic_api_key=values.get("LITELLM_PROXY_API_KEY", "").strip()
        or values.get("ANTHROPIC_API_KEY", "").strip()
        or None,
        llm_base_url=values.get("LITELLM_PROXY_BASE_URL", "").strip() or None,
        claude_model=values.get("CLAUDE_MODEL", "claude-sonnet-4-6").strip()
        or "claude-sonnet-4-6",
        agent_replies_in_threads=_boolean("AGENT_REPLIES_IN_THREADS", True, values),
        mention_role_id=mention_role_id,
        mention_everyone=_boolean("MENTION_EVERYONE", False, values),
    )
