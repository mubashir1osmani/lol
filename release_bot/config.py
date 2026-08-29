from dataclasses import dataclass
import os
from pathlib import Path
import re


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


def get_config(env: dict[str, str] | None = None) -> Config:
    values = dict(os.environ if env is None else env)
    repository = values.get("GITHUB_REPOSITORY", "BerriAI/litellm").strip()
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        raise ValueError("GITHUB_REPOSITORY must look like owner/repository")

    return Config(
        discord_token=_required("DISCORD_BOT_TOKEN", values),
        discord_channel_id=_required("DISCORD_CHANNEL_ID", values),
        github_repository=repository,
        github_token=values.get("GITHUB_TOKEN", "").strip() or None,
        poll_interval_seconds=_positive_float("POLL_INTERVAL_MINUTES", 5, values) * 60,
        include_prereleases=_boolean("INCLUDE_PRERELEASES", True, values),
        post_latest_on_startup=_boolean("POST_LATEST_ON_STARTUP", True, values),
        state_file=values.get("STATE_FILE", ".data/releases.json").strip()
        or ".data/releases.json",
    )
