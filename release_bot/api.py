import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


GITHUB_API = "https://api.github.com"
DISCORD_API = "https://discord.com/api/v10"


def _request_json(request: Request) -> Any:
    try:
        with urlopen(request, timeout=15) as response:
            return json.load(response)
    except HTTPError as error:
        details = error.read(500).decode("utf-8", errors="replace")
        service = "Discord" if "discord.com" in request.full_url else "GitHub"
        raise RuntimeError(
            f"{service} API returned {error.code}: {details}"
        ) from error


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


def release_message(release: dict[str, Any], repository: str) -> dict[str, Any]:
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
                "description": _truncate((release.get("body") or "").strip(), 3_800),
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


def post_release(
    channel_id: str,
    token: str,
    release: dict[str, Any],
    repository: str,
) -> None:
    body = json.dumps(release_message(release, repository)).encode("utf-8")
    request = Request(
        f"{DISCORD_API}/channels/{channel_id}/messages",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "DiscordBot (litellm-release-discord-bot, 1.0.0)",
        },
    )
    _request_json(request)
