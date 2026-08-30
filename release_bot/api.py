import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


GITHUB_API = "https://api.github.com"
DISCORD_API = "https://discord.com/api/v10"


def _request_json(request: Request) -> Any:
    try:
        with urlopen(request, timeout=15) as response:
            if response.status == 204:
                return None
            return json.load(response)
    except HTTPError as error:
        details = error.read(500).decode("utf-8", errors="replace")
        service = "Discord" if "discord.com" in request.full_url else "GitHub"
        raise RuntimeError(
            f"{service} API returned {error.code}: {details}"
        ) from error


def _discord_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bot {token}",
        "User-Agent": "DiscordBot (litellm-release-discord-bot, 1.0.0)",
    }


def fetch_releases(repository: str, token: str | None) -> list[dict[str, Any]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "litellm-release-discord-bot",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        f"{GITHUB_API}/repos/{repository}/releases?per_page=100", headers=headers
    )
    releases = _request_json(request)
    if not isinstance(releases, list):
        raise RuntimeError("GitHub returned an unexpected response")
    return releases


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return "No release notes provided."
    if len(text) <= limit:
        return text
    return f"{text[: limit - 20].rstrip()}\n\n…[read more]"


def release_message(
    release: dict[str, Any], repository: str, digest: str | None = None
) -> dict[str, Any]:
    tag = release["tag_name"]
    author = release.get("author") or {}
    author_data = {
        "name": author.get("login") or repository,
        "url": author.get("html_url") or f"https://github.com/{repository}",
    }
    if author.get("avatar_url"):
        author_data["icon_url"] = author["avatar_url"]

    return {
        "content": f"🚀 **{repository} {tag} has been released!**",
        "embeds": [
            {
                "title": (release.get("name") or tag)[:256],
                "url": release["html_url"],
                "description": _truncate(
                    (digest or release.get("body") or "").strip(), 3_800
                ),
                "color": 0xF0A500 if release.get("prerelease") else 0x2DA44E,
                "author": author_data,
                "fields": [
                    {
                        "name": "Release type",
                        "value": "Pre-release"
                        if release.get("prerelease")
                        else "Stable",
                        "inline": True,
                    }
                ],
                "timestamp": release.get("published_at") or release.get("created_at"),
                "footer": {"text": "GitHub Releases"},
            }
        ],
        "allowed_mentions": {"parse": []},
    }


def apply_mention(
    message: dict[str, Any],
    role_id: str | None,
    everyone: bool,
) -> dict[str, Any]:
    """Prefix a role/@everyone ping and whitelist it in allowed_mentions —
    without the whitelist Discord renders the mention but pings nobody."""
    if everyone:
        message["content"] = f"@everyone {message.get('content', '')}".strip()
        message["allowed_mentions"] = {"parse": ["everyone"]}
    elif role_id:
        message["content"] = f"<@&{role_id}> {message.get('content', '')}".strip()
        message["allowed_mentions"] = {"parse": [], "roles": [role_id]}
    return message


def post_message(channel_id: str, token: str, message: dict[str, Any]) -> None:
    request = Request(
        f"{DISCORD_API}/channels/{channel_id}/messages",
        data=json.dumps(message).encode("utf-8"),
        method="POST",
        headers={**_discord_headers(token), "Content-Type": "application/json"},
    )
    _request_json(request)


def post_release(
    channel_id: str,
    token: str,
    release: dict[str, Any],
    repository: str,
    digest: str | None = None,
) -> None:
    post_message(channel_id, token, release_message(release, repository, digest))


def fetch_active_threads(guild_id: str, token: str) -> list[dict[str, Any]]:
    request = Request(
        f"{DISCORD_API}/guilds/{guild_id}/threads/active",
        headers=_discord_headers(token),
    )
    response = _request_json(request)
    threads = response.get("threads") if isinstance(response, dict) else None
    if not isinstance(threads, list):
        raise RuntimeError("Discord returned an unexpected active-threads response")
    return threads


def fetch_first_message(thread_id: str, token: str) -> str:
    """Best-effort read of a thread's opening message (needs the Read Message
    History permission); returns an empty string when unavailable."""
    request = Request(
        f"{DISCORD_API}/channels/{thread_id}/messages?limit=1&after=0",
        headers=_discord_headers(token),
    )
    try:
        messages = _request_json(request)
    except Exception:
        return ""
    if isinstance(messages, list) and messages:
        return str(messages[0].get("content") or "")
    return ""


def add_thread_member(thread_id: str, user_id: str, token: str) -> None:
    request = Request(
        f"{DISCORD_API}/channels/{thread_id}/thread-members/{user_id}",
        data=b"",
        method="PUT",
        headers=_discord_headers(token),
    )
    _request_json(request)
